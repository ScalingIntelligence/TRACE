#!/usr/bin/env python3
"""
LLM-based Contrastive Skill Selection (N independent runs)

Sends pass/fail trajectory summaries to an LLM judge N times, each time
asking it to independently identify 4-6 skills the model lacks. Each run
is fully independent — the LLM has no memory of previous runs.

After all runs, produces:
  1. Per-run JSON files with identified skills + task coverage
  2. A combined analysis JSON with overlap stats, coverage bars, etc.
  3. A bar chart PNG with error bars showing coverage min/max across runs

Usage:
  python contrastive_skill_selection_llm.py \
    --retail data/old_simulations/qwen-3-30b-retail-qwen3-30b.json \
    --airline data/old_simulations/qwen-3-30b-airline-qwen3-30b.json \
    --runs 15 \
    --output-dir contrastive_llm_runs
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
# Trajectory summarization (compact representation for the LLM)
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

        # Extract tool calls made
        tool_calls = []
        for m in msgs:
            tc = m.get("tool_calls")
            if tc:
                for t in tc:
                    fn = t.get("function", {})
                    name = fn.get("name", "?") if isinstance(fn, dict) else str(fn)
                    tool_calls.append(name)

        # Extract first user message (the request)
        first_user = ""
        for m in msgs:
            if m.get("role") == "user":
                first_user = str(m.get("content", ""))[:300]
                break

        # Extract last assistant message
        last_asst = ""
        for m in reversed(msgs):
            if m.get("role") == "assistant" and m.get("content"):
                last_asst = str(m["content"])[:300]
                break

        # Count tool errors
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


def build_llm_prompt(summaries: list[dict], attempt_num: int) -> str:
    """Build the prompt for the LLM judge."""

    passed = [s for s in summaries if s["reward"] == 1.0]
    failed = [s for s in summaries if s["reward"] != 1.0]

    # Format failed trajectories with detail
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

    # Format passed trajectories (more compact)
    passed_text = ""
    for s in passed:
        tools_str = ", ".join(set(s["tool_calls"]))
        passed_text += f"  {s['task_id']}: tools=[{tools_str}], msgs={s['n_messages']}\n"

    prompt = f"""You are analyzing a customer service agent's performance across {len(summaries)} tasks
({len(passed)} passed, {len(failed)} failed) spanning airline and retail domains.

Your goal: identify 4-6 broad SKILLS that the agent consistently lacks, based on the
failure patterns. These skills should describe general capabilities (not domain-specific
rules or task-specific hints).

IMPORTANT CONSTRAINTS:
- Skills must be GENERAL agent capabilities, not domain knowledge
- Do NOT reference specific policies, rules, dollar amounts, or domain procedures
- Do NOT give hints about how to solve specific tasks
- Think about WHAT CAPABILITY is missing, not WHAT ANSWER is correct
- Each skill should cover multiple failed tasks across domains if possible
- Skills should be things that could improve ANY tool-using agent, not just this one

GOOD skill examples: "multi-step planning", "error recovery", "numerical reasoning"
BAD skill examples: "check cancellation eligibility", "verify basic economy rules",
"compute baggage fees correctly"

== FAILED TRAJECTORIES ({len(failed)} tasks) ==
{failed_text}

== PASSED TRAJECTORIES ({len(passed)} tasks, summary only) ==
{passed_text}

Now analyze the failures. For each skill you identify:
1. Give it a short, general name (2-5 words)
2. Write a 1-2 sentence description of what capability is missing
3. List which failed task IDs this skill would help (e.g., ["A5", "R12", "A31"])
4. Briefly explain WHY those tasks need this skill (1 sentence)

Output as a JSON object:
{{
  "attempt": {attempt_num},
  "skills": [
    {{
      "name": "short general name",
      "description": "1-2 sentence description of the missing capability",
      "covered_tasks": ["A5", "R12", ...],
      "reasoning": "why these tasks need this skill"
    }}
  ]
}}

Output ONLY the JSON, no other text.
"""
    return prompt


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str = "claude-sonnet-4-20250514", max_retries: int = 3) -> str:
    """Call Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=1.0,  # diversity across runs
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}: {e}")
                time.sleep(5)
            else:
                raise


def parse_llm_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Try to find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_runs(all_runs: list[dict], all_failed_tasks: set) -> dict:
    """Analyze overlap and coverage across all runs."""

    n_runs = len(all_runs)
    total_failed = len(all_failed_tasks)

    # Collect all unique skill names
    all_skill_names = Counter()
    skill_tasks = defaultdict(list)  # skill_name -> list of task sets (one per run)
    skill_descriptions = defaultdict(list)

    for run in all_runs:
        for skill in run.get("skills", []):
            name = skill["name"].lower().strip()
            all_skill_names[name] += 1
            skill_tasks[name].append(set(skill.get("covered_tasks", [])))
            skill_descriptions[name].append(skill.get("description", ""))

    # Per-skill stats
    skill_stats = {}
    for name, count in all_skill_names.most_common():
        task_sets = skill_tasks[name]
        all_tasks_union = set().union(*task_sets) if task_sets else set()
        all_tasks_intersection = set.intersection(*task_sets) if task_sets else set()

        coverages = [len(ts) for ts in task_sets]

        skill_stats[name] = {
            "appearances": count,
            "appearance_rate": round(count / n_runs, 3),
            "descriptions": skill_descriptions[name],
            "coverage_per_run": coverages,
            "coverage_mean": round(np.mean(coverages), 1) if coverages else 0,
            "coverage_std": round(np.std(coverages), 1) if coverages else 0,
            "coverage_min": min(coverages) if coverages else 0,
            "coverage_max": max(coverages) if coverages else 0,
            "union_tasks": sorted(all_tasks_union),
            "intersection_tasks": sorted(all_tasks_intersection),
            "union_size": len(all_tasks_union),
            "intersection_size": len(all_tasks_intersection),
        }

    # Overall coverage per run
    run_coverages = []
    for run in all_runs:
        covered = set()
        for skill in run.get("skills", []):
            covered.update(skill.get("covered_tasks", []))
        run_coverages.append(len(covered & all_failed_tasks))

    # Pairwise skill name overlap (Jaccard similarity)
    pairwise_jaccard = []
    for i in range(n_runs):
        names_i = {s["name"].lower().strip() for s in all_runs[i].get("skills", [])}
        for j in range(i + 1, n_runs):
            names_j = {s["name"].lower().strip() for s in all_runs[j].get("skills", [])}
            if names_i | names_j:
                jaccard = len(names_i & names_j) / len(names_i | names_j)
                pairwise_jaccard.append(jaccard)

    return {
        "n_runs": n_runs,
        "total_failed_tasks": total_failed,
        "skill_stats": skill_stats,
        "run_coverages": run_coverages,
        "coverage_mean": round(np.mean(run_coverages), 1),
        "coverage_std": round(np.std(run_coverages), 1),
        "coverage_min": int(np.min(run_coverages)),
        "coverage_max": int(np.max(run_coverages)),
        "pairwise_name_jaccard_mean": round(np.mean(pairwise_jaccard), 3) if pairwise_jaccard else 0,
        "pairwise_name_jaccard_std": round(np.std(pairwise_jaccard), 3) if pairwise_jaccard else 0,
    }


def plot_analysis(analysis: dict, output_path: str):
    """Generate bar chart with error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Skill appearance frequency
    ax1 = axes[0]
    stats = analysis["skill_stats"]
    # Sort by appearance rate
    sorted_skills = sorted(stats.items(), key=lambda x: x[1]["appearances"], reverse=True)
    names = [s[0][:30] for s in sorted_skills[:15]]
    appearances = [s[1]["appearances"] for s in sorted_skills[:15]]
    n_runs = analysis["n_runs"]

    ax1.barh(range(len(names)), appearances, color="steelblue")
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Appearances across runs")
    ax1.set_title(f"Skill Consistency ({n_runs} runs)")
    ax1.axvline(x=n_runs * 0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax1.legend()
    ax1.invert_yaxis()

    # Plot 2: Coverage per run with error bars
    ax2 = axes[1]
    run_coverages = analysis["run_coverages"]
    ax2.bar(range(1, len(run_coverages) + 1), run_coverages, color="coral", alpha=0.7)
    ax2.axhline(y=analysis["coverage_mean"], color="red", linestyle="--",
                label=f"Mean: {analysis['coverage_mean']:.1f}")
    ax2.fill_between(
        [0.5, len(run_coverages) + 0.5],
        analysis["coverage_mean"] - analysis["coverage_std"],
        analysis["coverage_mean"] + analysis["coverage_std"],
        alpha=0.2, color="red",
    )
    ax2.set_xlabel("Run #")
    ax2.set_ylabel("Failed tasks covered")
    ax2.set_title(f"Coverage per Run (total failed: {analysis['total_failed_tasks']})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-based contrastive skill selection")
    parser.add_argument("--retail", type=str, default=None)
    parser.add_argument("--airline", type=str, default=None)
    parser.add_argument("--runs", type=int, default=15,
                        help="Number of independent LLM runs")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Anthropic model to use as judge")
    parser.add_argument("--output-dir", type=str, default="contrastive_llm_runs",
                        help="Directory for per-run JSONs and analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectories
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

    all_failed_tasks = {s["task_id"] for s in all_summaries if s["reward"] != 1.0}
    print(f"  Combined: {len(all_summaries)} tasks, {len(all_failed_tasks)} failed\n")

    # Run N independent LLM calls
    all_runs = []
    for i in range(1, args.runs + 1):
        print(f"Run {i}/{args.runs}...")
        prompt = build_llm_prompt(all_summaries, attempt_num=i)

        try:
            response = call_llm(prompt, model=args.model)
            result = parse_llm_response(response)
            result["attempt"] = i

            # Save per-run result
            run_path = output_dir / f"run_{i:02d}.json"
            with open(run_path, "w") as f:
                json.dump(result, f, indent=2)

            n_skills = len(result.get("skills", []))
            covered = set()
            for s in result.get("skills", []):
                covered.update(s.get("covered_tasks", []))
            print(f"  -> {n_skills} skills, covers {len(covered)} failed tasks")

            all_runs.append(result)

        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

    if not all_runs:
        print("No successful runs. Exiting.")
        return

    # Save all runs
    all_runs_path = output_dir / "all_runs.json"
    with open(all_runs_path, "w") as f:
        json.dump(all_runs, f, indent=2)

    # Analyze
    print(f"\nAnalyzing {len(all_runs)} runs...")
    analysis = analyze_runs(all_runs, all_failed_tasks)

    # Save analysis
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  CONTRASTIVE SKILL SELECTION ANALYSIS")
    print(f"  {len(all_runs)} independent runs")
    print(f"{'='*60}")

    print(f"\n  Coverage per run: mean={analysis['coverage_mean']:.1f} "
          f"± {analysis['coverage_std']:.1f} "
          f"(min={analysis['coverage_min']}, max={analysis['coverage_max']}) "
          f"out of {analysis['total_failed_tasks']} failed tasks")

    print(f"  Pairwise skill name overlap (Jaccard): "
          f"mean={analysis['pairwise_name_jaccard_mean']:.3f} "
          f"± {analysis['pairwise_name_jaccard_std']:.3f}")

    print(f"\n  Top skills by consistency:")
    print(f"  {'Skill':<35s} {'Appears':>8s} {'Rate':>6s} {'Cov Mean':>9s} {'Cov Range':>12s}")
    print(f"  {'-'*72}")
    for name, stats in sorted(analysis["skill_stats"].items(),
                               key=lambda x: x[1]["appearances"], reverse=True)[:15]:
        print(f"  {name:<35s} {stats['appearances']:>8d} "
              f"{stats['appearance_rate']:>5.0%} "
              f"{stats['coverage_mean']:>9.1f} "
              f"{stats['coverage_min']:>5d}-{stats['coverage_max']:<5d}")

    # Plot
    try:
        plot_path = output_dir / "analysis_plot.png"
        plot_analysis(analysis, str(plot_path))
    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    print(f"\n  Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
