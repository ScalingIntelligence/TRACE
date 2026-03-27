#!/usr/bin/env python3
"""Test classifier routing accuracy at the critical failure turn.

Instead of routing from just the first user message, this script feeds the
conversation up to the annotated critical turn (where the base model first
goes wrong) and asks the classifier to pick a skill. This measures whether
the router can identify the correct behavioral skill given enough context.

Inputs:
  - scaling_skills.json: skill definitions with task labels
  - critical_turns_{domain}.json: per-task annotations with turn index
  - Base model trajectory data: the actual conversations to extract prefixes
  - A vLLM server for the classifier model

Usage:
    python test_critical_turn_classifier.py \
        --skills-json scaling_skills.json \
        --critical-turns critical_turns_airline.json \
        --trajectories /path/to/qwen-3-30b-airline-qwen3-30b.json \
        --domain airline --port 9002
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

tau2_src = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "src"
sys.path.insert(0, str(tau2_src))
tau2_data = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(tau2_data))

from tau2.agent.orchestrator_agent import OrchestratorAgent, OrchestratorConfig, SkillConfig
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage


def slugify(title: str) -> str:
    return title.lower().replace(" ", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-json", required=True)
    parser.add_argument("--critical-turns", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--domain", default="airline", choices=["airline", "retail"])
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--task-ids", nargs="*", type=str, default=None)
    parser.add_argument("--compare-first-msg", action="store_true",
                        help="Also run routing from first message only for comparison")
    args = parser.parse_args()

    # Load data
    with open(args.skills_json) as f:
        skills_data = json.load(f)
    with open(args.critical_turns) as f:
        critical_turns = {ct["task_id"]: ct for ct in json.load(f)}
    with open(args.trajectories) as f:
        trajectories = {s["task_id"]: s for s in json.load(f)["simulations"]}

    domain_prefix = "A" if args.domain == "airline" else "R"
    port_url = f"http://localhost:{args.port}/v1"

    # Build skill configs
    skill_configs = [SkillConfig(
        name="general_service",
        description="No specialized training. General inquiries only.",
        model=f"openai/{args.model}",
        llm_args={"api_base": port_url, "api_key": "EMPTY",
                   "temperature": 0.0, "max_context_length": 32000},
    )]

    expected_routing = {}
    for skill in skills_data["skills"]:
        skill_name = slugify(skill["title"])
        skill_configs.append(SkillConfig(
            name=skill_name,
            description=skill["principle"],
            model=f"openai/{args.model}",
            llm_args={"api_base": port_url, "api_key": "EMPTY",
                       "temperature": 0.0, "max_context_length": 32000},
        ))
        for task_ref in skill["failed_tasks"]:
            if task_ref.startswith(domain_prefix):
                tid = task_ref[1:]
                expected_routing.setdefault(tid, []).append(skill_name)

    orch_config = OrchestratorConfig(
        orchestrator_model=f"openai/{args.model}",
        orchestrator_llm_args={"temperature": 0.0, "api_base": port_url, "api_key": "EMPTY"},
        skills=skill_configs,
        routing_strategy="per_conversation",
        routing_mode="classifier",
        routing_topk=1,
        routing_context_window=None,  # use full context
    )

    # Load environment for system prompt
    from tau2.registry import registry
    env = registry.get_env_constructor(args.domain)()
    tools = env.get_tools()
    policy = env.get_policy()

    agent = OrchestratorAgent(
        tools=tools, domain_policy=policy,
        orchestrator_config=orch_config, llm_args={},
    )

    # Determine task IDs
    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = sorted(critical_turns.keys(), key=int)

    print(f"Testing {len(task_ids)} tasks | domain={args.domain} | skills={len(skill_configs)}")
    print(f"{'TID':>5} {'turn':>5} {'skill_needed':>35} {'first_msg_got':>35} {'crit_turn_got':>35} {'F':>2} {'C':>2}")
    print("-" * 145)

    first_msg_correct = 0
    crit_turn_correct = 0
    total = 0
    results = []

    for tid in task_ids:
        if tid not in critical_turns or tid not in trajectories:
            continue
        ct = critical_turns[tid]
        traj = trajectories[tid]
        exp = expected_routing.get(tid, [])
        if not exp:
            continue

        total += 1
        crit_turn_idx = ct["critical_turn"]
        skill_needed = ct["skill"]
        messages = traj["messages"]

        # Build message list up to critical turn
        def build_messages(msg_list):
            result = agent.get_init_state().system_messages[:]
            for m in msg_list:
                role = m["role"]
                content = m.get("content") or ""
                if role == "assistant":
                    result.append(AssistantMessage(role="assistant", content=content))
                elif role == "user":
                    result.append(UserMessage(role="user", content=content))
                elif role == "tool":
                    # Include tool results as user messages for context
                    tool_name = m.get("name", "tool")
                    result.append(UserMessage(role="user", content=f"[Tool result from {tool_name}]: {content[:500]}"))
            return result

        # Route at critical turn
        crit_messages = build_messages(messages[:crit_turn_idx + 1])
        agent.set_seed(42)
        crit_decision = agent._route(crit_messages)
        crit_got = crit_decision.selected_skill
        crit_ok = crit_got in exp

        if crit_ok:
            crit_turn_correct += 1

        # Route at first message (for comparison)
        first_msg_got = ""
        first_ok = False
        if args.compare_first_msg:
            # Just greeting + first user message
            first_msgs = build_messages(messages[:2])  # [0]=greeting, [1]=user msg
            agent.set_seed(42)
            first_decision = agent._route(first_msgs)
            first_msg_got = first_decision.selected_skill
            first_ok = first_msg_got in exp
            if first_ok:
                first_msg_correct += 1

        exp_str = exp[0] if len(exp) == 1 else "|".join(exp)
        f_status = "✓" if first_ok else "✗" if args.compare_first_msg else "-"
        c_status = "✓" if crit_ok else "✗"

        print(f"T{tid:>4} {crit_turn_idx:>5} {exp_str:>35} {first_msg_got:>35} {crit_got:>35} {f_status:>2} {c_status:>2}")

        results.append({
            "task_id": tid,
            "expected": exp,
            "critical_turn": crit_turn_idx,
            "crit_turn_prediction": crit_got,
            "crit_turn_correct": crit_ok,
            "first_msg_prediction": first_msg_got,
            "first_msg_correct": first_ok,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"Critical-turn accuracy: {crit_turn_correct}/{total} ({100*crit_turn_correct/total:.1f}%)")
    if args.compare_first_msg:
        print(f"First-message accuracy: {first_msg_correct}/{total} ({100*first_msg_correct/total:.1f}%)")
        improvement = crit_turn_correct - first_msg_correct
        print(f"Improvement from context: +{improvement} tasks ({100*improvement/total:.1f}%)")
    print(f"{'='*60}")

    # Per-skill breakdown
    print(f"\nPer-skill accuracy (critical turn):")
    by_skill = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        for skill in r["expected"]:
            by_skill[skill]["total"] += 1
            if r["crit_turn_correct"]:
                by_skill[skill]["correct"] += 1
    for skill, counts in sorted(by_skill.items(), key=lambda x: -x[1]["total"]):
        c, t = counts["correct"], counts["total"]
        print(f"  {skill:>40}: {c}/{t} ({100*c/t:.0f}%)")

    # Misclassified
    print(f"\nMisclassified at critical turn:")
    for r in results:
        if not r["crit_turn_correct"]:
            print(f"  T{r['task_id']}: needed={r['expected']}, got={r['crit_turn_prediction']}")


if __name__ == "__main__":
    main()
