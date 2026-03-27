#!/usr/bin/env python3
"""
LLM-based Contrastive Skill Selection v2 — Fixed Skill Menu

Instead of letting the LLM invent skill names (which produces 137 unique names
for ~5 actual skills), this version provides a fixed menu of 8 candidate skills
derived from the convergent themes in v1's 25 runs. The LLM's job is to:
  1. Examine failed trajectories
  2. Assign each failed task to 1-3 skills from the menu
  3. Optionally flag tasks that don't fit any skill

This produces consistent, comparable results across runs.

Usage:
  python contrastive_skill_selection_llm_v2.py \
    --retail data/old_simulations/qwen-3-30b-retail-qwen3-30b.json \
    --airline data/old_simulations/qwen-3-30b-airline-qwen3-30b.json \
    --runs 15 \
    --output-dir contrastive_llm_runs_v2
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fixed skill menu (derived from v1's 25-run convergence analysis)
# ---------------------------------------------------------------------------

SKILL_MENU = [
    # --- Our 4 training skills (S1-S4) ---
    {
        "id": "S1",
        "name": "Structured data reasoning",
        "short": "Agent fails to correctly parse, cross-reference, or reason over structured data returned by tools — selects wrong items, misreads fields, or ignores attribute constraints.",
    },
    {
        "id": "S2",
        "name": "Tool calling precision",
        "short": "Agent calls the right tool but with wrong arguments — wrong IDs, wrong amounts, wrong payment methods, wrong parameters extracted from prior tool outputs.",
    },
    {
        "id": "S3",
        "name": "Multi-step task completion",
        "short": "Agent completes some sub-tasks but silently drops others in compound requests, or executes steps in the wrong order causing later steps to fail.",
    },
    {
        "id": "S4",
        "name": "Precondition verification",
        "short": "Agent executes actions without checking whether required conditions are met — performs forbidden operations, ignores eligibility rules, or caves to user pressure instead of refusing.",
    },
    # --- Emergent skills (S5-S8) ---
    {
        "id": "S5",
        "name": "Numerical reasoning",
        "short": "Agent makes arithmetic errors in totals, refunds, payment splits, or fails to use the calculator tool when needed.",
    },
    {
        "id": "S6",
        "name": "Record discovery and retrieval",
        "short": "Agent fails to find all relevant records, stops at the first match, acts on the wrong record, or doesn't try alternative lookup strategies.",
    },
    {
        "id": "S7",
        "name": "Information communication",
        "short": "Agent completes the action correctly but fails to state specific values the user asked for (totals, tracking numbers, counts).",
    },
    {
        "id": "S8",
        "name": "Conditional and state-aware reasoning",
        "short": "Agent ignores if-then-else branches in user instructions, or calls the wrong API for an entity's current lifecycle state.",
    },
    # --- Distractor / noise skills (S9-S16) ---
    {
        "id": "S9",
        "name": "Language fluency",
        "short": "Agent produces grammatically incorrect, incoherent, or hard-to-understand responses.",
    },
    {
        "id": "S10",
        "name": "Response latency awareness",
        "short": "Agent takes too many turns to respond or unnecessarily delays providing information already available.",
    },
    {
        "id": "S11",
        "name": "Tone and empathy",
        "short": "Agent fails to show appropriate empathy, uses overly formal or robotic tone, or is rude to the user.",
    },
    {
        "id": "S12",
        "name": "Hallucination of nonexistent tools",
        "short": "Agent tries to call tools that do not exist in the available tool set.",
    },
    {
        "id": "S13",
        "name": "Format compliance",
        "short": "Agent produces malformed JSON, invalid tool call syntax, or responses that cannot be parsed by the system.",
    },
    {
        "id": "S14",
        "name": "Context window management",
        "short": "Agent loses track of earlier conversation content due to long context, forgetting what was already discussed.",
    },
    {
        "id": "S15",
        "name": "Multilingual handling",
        "short": "Agent fails when the user switches languages or uses non-English terms.",
    },
    {
        "id": "S16",
        "name": "Proactive upselling",
        "short": "Agent fails to suggest relevant additional products or services that the user might benefit from.",
    },
]


# ---------------------------------------------------------------------------
# Trajectory summarization
# ---------------------------------------------------------------------------

def summarize_trajectories(data: dict, domain: str) -> list[dict]:
    """Extract compact pass/fail summaries from a tau2-bench result file."""
    summaries = []
    for sim in data["simulations"]:
        tid = sim["task_id"]
        reward = sim["reward_info"]["reward"]
        breakdown = sim["reward_info"].get("reward_breakdown") or {}
        term = sim["termination_reason"]
        msgs = sim["messages"]

        tool_calls = []
        for m in msgs:
            tc = m.get("tool_calls")
            if tc:
                for t in tc:
                    fn = t.get("function", {})
                    name = fn.get("name", "?") if isinstance(fn, dict) else str(fn)
                    tool_calls.append(name)

        first_user = ""
        for m in msgs:
            if m.get("role") == "user":
                first_user = str(m.get("content", ""))[:300]
                break

        last_asst = ""
        for m in reversed(msgs):
            if m.get("role") == "assistant" and m.get("content"):
                last_asst = str(m["content"])[:300]
                break

        tool_errors = sum(
            1 for m in msgs
            if m.get("role") == "tool" and "error" in str(m.get("content", "")).lower()
        )

        n_user_turns = sum(1 for m in msgs if m.get("role") == "user")

        summaries.append({
            "task_id": f"{domain[0].upper()}{tid}",
            "domain": domain,
            "reward": reward,
            "db_reward": breakdown.get("DB", -1),
            "comm_reward": breakdown.get("COMMUNICATE", -1),
            "termination": term,
            "user_request": first_user,
            "last_response": last_asst,
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
            "n_messages": len(msgs),
            "n_user_turns": n_user_turns,
        })
    return summaries


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(summaries: list[dict], attempt_num: int) -> str:
    passed = [s for s in summaries if s["reward"] == 1.0]
    failed = [s for s in summaries if s["reward"] != 1.0]

    # Format skill menu
    menu_text = ""
    for s in SKILL_MENU:
        menu_text += f"  {s['id']}: {s['name']} — {s['short']}\n"

    # Format failed trajectories
    failed_text = ""
    for s in failed:
        tools_str = ", ".join(s["tool_calls"][:15])
        if len(s["tool_calls"]) > 15:
            tools_str += f" ... ({len(s['tool_calls'])} total)"
        failed_text += f"""
--- {s['task_id']} (DB={s['db_reward']}, COMM={s['comm_reward']}, term={s['termination']}) ---
Request: {s['user_request'][:250]}
Tools used: [{tools_str}]
Tool errors: {s['tool_errors']}
Messages: {s['n_messages']}, User turns: {s['n_user_turns']}
Last response: {s['last_response'][:200]}
"""

    prompt = f"""You are analyzing a customer service agent's failures across {len(summaries)} tasks
({len(passed)} passed, {len(failed)} failed) in airline and retail domains.

Below is a FIXED MENU of 8 skill categories. Your job is to examine each failed task
and assign it to 1-3 skills from this menu that best explain WHY it failed.

## SKILL MENU
{menu_text}

## INSTRUCTIONS
1. Read each failed trajectory below.
2. For each failed task, determine which skill(s) from the menu above the agent lacked.
3. Assign each failed task to 1-3 skill IDs (e.g., S1, S3).
4. If a task doesn't fit ANY skill, assign it to "NONE".
5. Be precise — only assign a skill if you see clear evidence in the trajectory.

## FAILED TRAJECTORIES ({len(failed)} tasks)
{failed_text}

## OUTPUT FORMAT
Return a JSON object mapping each failed task ID to its assigned skill IDs:
{{
  "attempt": {attempt_num},
  "assignments": {{
    "A5": ["S1", "S3"],
    "R12": ["S2"],
    ...
  }},
  "skill_summary": {{
    "S1": {{"count": 30, "example_tasks": ["A5", "R12"]}},
    "S2": {{"count": 15, "example_tasks": ["R3", "A18"]}},
    ...
  }}
}}

Output ONLY the JSON, no other text.
"""
    return prompt


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str = "claude-sonnet-4-20250514", max_retries: int = 3) -> str:
    import anthropic
    client = anthropic.Anthropic()
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}: {e}")
                time.sleep(5)
            else:
                raise


def parse_response(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_runs(all_runs: list[dict], all_failed_task_ids: set) -> dict:
    n_runs = len(all_runs)

    # Per-skill: how many tasks assigned in each run
    skill_ids = [s["id"] for s in SKILL_MENU]

    # Per-skill per-run counts
    skill_run_counts = {sid: [] for sid in skill_ids}
    # Per-skill: which tasks assigned across all runs
    skill_task_sets = {sid: [] for sid in skill_ids}

    for run in all_runs:
        assignments = run.get("assignments", {})
        run_skill_tasks = {sid: set() for sid in skill_ids}

        for task_id, skills in assignments.items():
            if isinstance(skills, str):
                skills = [skills]
            for sid in skills:
                sid = sid.strip().upper()
                if sid in run_skill_tasks:
                    run_skill_tasks[sid].add(task_id)

        for sid in skill_ids:
            skill_run_counts[sid].append(len(run_skill_tasks[sid]))
            skill_task_sets[sid].append(run_skill_tasks[sid])

    # Compute stats
    skill_stats = {}
    for sid in skill_ids:
        counts = skill_run_counts[sid]
        task_sets = skill_task_sets[sid]
        union = set().union(*task_sets) if task_sets else set()
        intersection = set.intersection(*task_sets) if task_sets else set()

        skill_stats[sid] = {
            "name": next(s["name"] for s in SKILL_MENU if s["id"] == sid),
            "count_mean": round(float(np.mean(counts)), 1),
            "count_std": round(float(np.std(counts)), 1),
            "count_min": int(np.min(counts)),
            "count_max": int(np.max(counts)),
            "union_size": len(union),
            "intersection_size": len(intersection),
            "union_tasks": sorted(union),
            "intersection_tasks": sorted(intersection),
            "counts_per_run": counts,
        }

    # Per-run total coverage
    run_coverages = []
    for run in all_runs:
        covered = set()
        for task_id, skills in run.get("assignments", {}).items():
            if isinstance(skills, str):
                skills = [skills]
            if any(s.strip().upper() != "NONE" for s in skills):
                covered.add(task_id)
        run_coverages.append(len(covered))

    # Pairwise Jaccard on per-task assignments
    jaccard_per_task = []
    for task_id in all_failed_task_ids:
        assignments_across_runs = []
        for run in all_runs:
            a = run.get("assignments", {}).get(task_id, [])
            if isinstance(a, str):
                a = [a]
            assignments_across_runs.append(set(s.strip().upper() for s in a))

        # Pairwise Jaccard for this task
        for i in range(len(assignments_across_runs)):
            for j in range(i+1, len(assignments_across_runs)):
                si, sj = assignments_across_runs[i], assignments_across_runs[j]
                if si | sj:
                    jaccard_per_task.append(len(si & sj) / len(si | sj))

    return {
        "n_runs": n_runs,
        "total_failed_tasks": len(all_failed_task_ids),
        "skill_stats": skill_stats,
        "run_coverages": run_coverages,
        "coverage_mean": round(float(np.mean(run_coverages)), 1),
        "coverage_std": round(float(np.std(run_coverages)), 1),
        "per_task_assignment_jaccard_mean": round(float(np.mean(jaccard_per_task)), 3) if jaccard_per_task else 0,
        "per_task_assignment_jaccard_std": round(float(np.std(jaccard_per_task)), 3) if jaccard_per_task else 0,
    }


def plot_analysis(analysis: dict, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    stats = analysis["skill_stats"]
    skill_ids = sorted(stats.keys())

    # Plot 1: Mean task count per skill with error bars
    ax1 = axes[0]
    names = [f"{sid}\n{stats[sid]['name'][:20]}" for sid in skill_ids]
    means = [stats[sid]["count_mean"] for sid in skill_ids]
    stds = [stats[sid]["count_std"] for sid in skill_ids]
    mins = [stats[sid]["count_min"] for sid in skill_ids]
    maxs = [stats[sid]["count_max"] for sid in skill_ids]

    x = np.arange(len(names))
    bars = ax1.bar(x, means, color="steelblue", alpha=0.7)
    ax1.errorbar(x, means,
                 yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
                 fmt="none", color="black", capsize=5, capthick=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
    ax1.set_ylabel("Tasks Assigned")
    ax1.set_title("Mean Tasks Per Skill (± Min/Max)")

    # Plot 2: Coverage per run
    ax2 = axes[1]
    coverages = analysis["run_coverages"]
    ax2.bar(range(1, len(coverages)+1), coverages, color="coral", alpha=0.7)
    ax2.axhline(y=analysis["coverage_mean"], color="red", linestyle="--",
                label=f"Mean: {analysis['coverage_mean']:.1f}")
    ax2.fill_between([0.5, len(coverages)+0.5],
                     analysis["coverage_mean"] - analysis["coverage_std"],
                     analysis["coverage_mean"] + analysis["coverage_std"],
                     alpha=0.15, color="red")
    ax2.set_xlabel("Run #")
    ax2.set_ylabel("Tasks Covered")
    ax2.set_title(f"Coverage Per Run (total: {analysis['total_failed_tasks']})")
    ax2.legend()

    # Plot 3: Union vs Intersection per skill
    ax3 = axes[2]
    union_sizes = [stats[sid]["union_size"] for sid in skill_ids]
    inter_sizes = [stats[sid]["intersection_size"] for sid in skill_ids]
    w = 0.35
    ax3.bar(x - w/2, union_sizes, w, label="Union (ever assigned)", color="steelblue", alpha=0.7)
    ax3.bar(x + w/2, inter_sizes, w, label="Intersection (always assigned)", color="orange", alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{sid}" for sid in skill_ids], fontsize=9)
    ax3.set_ylabel("# Tasks")
    ax3.set_title("Task Assignment Consistency Per Skill")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-based contrastive skill selection v2")
    parser.add_argument("--retail", type=str, default=None)
    parser.add_argument("--airline", type=str, default=None)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--output-dir", type=str, default="contrastive_llm_runs_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for path, domain in [(args.retail, "retail"), (args.airline, "airline")]:
        if path is None:
            continue
        with open(path) as f:
            data = json.load(f)
        summaries = summarize_trajectories(data, domain)
        all_summaries.extend(summaries)
        n_pass = sum(1 for s in summaries if s["reward"] == 1.0)
        n_fail = len(summaries) - n_pass
        print(f"  {domain}: {len(summaries)} tasks, {n_pass} pass, {n_fail} fail")

    all_failed_ids = {s["task_id"] for s in all_summaries if s["reward"] != 1.0}
    print(f"  Combined: {len(all_summaries)} tasks, {len(all_failed_ids)} failed\n")

    all_runs = []
    for i in range(1, args.runs + 1):
        print(f"Run {i}/{args.runs}...")
        prompt = build_prompt(all_summaries, attempt_num=i)

        try:
            response = call_llm(prompt, model=args.model)
            result = parse_response(response)
            result["attempt"] = i

            run_path = output_dir / f"run_{i:02d}.json"
            with open(run_path, "w") as f:
                json.dump(result, f, indent=2)

            n_assigned = len(result.get("assignments", {}))
            print(f"  -> {n_assigned} tasks assigned")
            all_runs.append(result)

        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

    if not all_runs:
        print("No successful runs.")
        return

    # Save combined
    with open(output_dir / "all_runs.json", "w") as f:
        json.dump(all_runs, f, indent=2)

    # Analyze
    print(f"\nAnalyzing {len(all_runs)} runs...")
    analysis = analyze_runs(all_runs, all_failed_ids)

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SKILL ASSIGNMENT ANALYSIS ({len(all_runs)} runs)")
    print(f"{'='*70}")
    print(f"  {'Skill':<40s} {'Mean':>6s} {'Std':>5s} {'Min':>5s} {'Max':>5s} {'Union':>6s} {'Inter':>6s}")
    print(f"  {'-'*74}")
    for sid, st in sorted(analysis["skill_stats"].items()):
        print(f"  {sid} {st['name']:<36s} {st['count_mean']:>6.1f} {st['count_std']:>5.1f} "
              f"{st['count_min']:>5d} {st['count_max']:>5d} {st['union_size']:>6d} {st['intersection_size']:>6d}")

    print(f"\n  Coverage: {analysis['coverage_mean']:.1f} ± {analysis['coverage_std']:.1f} "
          f"(range {min(analysis['run_coverages'])}-{max(analysis['run_coverages'])})")
    print(f"  Per-task assignment Jaccard: {analysis['per_task_assignment_jaccard_mean']:.3f} "
          f"± {analysis['per_task_assignment_jaccard_std']:.3f}")

    try:
        plot_analysis(analysis, str(output_dir / "analysis_plot.png"))
    except ImportError:
        print("  (matplotlib not available)")

    print(f"\n  Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
