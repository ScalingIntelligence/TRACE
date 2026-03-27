#!/usr/bin/env python3
"""Test classifier routing accuracy using scaling_skills.json as the skill config.

Reads skills from scaling_skills.json where:
  - skill_id (S1, S2, ...) → label (A, B, ...)
  - title → skill name
  - principle → skill description
  - failed_tasks → list of tasks this skill handles (A=airline, R=retail)

Builds an OrchestratorConfig dynamically, runs the classifier router on each
task, and reports exact accuracy (topk=1 primary matches an acceptable skill).
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import yaml

tau2_src = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "src"
sys.path.insert(0, str(tau2_src))
tau2_data = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(tau2_data))

from tau2.agent.orchestrator_agent import OrchestratorAgent, OrchestratorConfig, SkillConfig
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator


def get_task_seed(base_seed: int, trial: int, task_id: str) -> int:
    combined = f"tau2_seed_{base_seed}_trial_{trial}_task_{task_id}"
    hash_bytes = hashlib.md5(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "big") % 1000000


def slugify(title: str) -> str:
    """Convert skill title to a valid skill name."""
    return title.lower().replace(" ", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser(description="Test classifier routing with scaling_skills.json")
    parser.add_argument("--skills-json", required=True,
                        help="Path to scaling_skills.json")
    parser.add_argument("--domain", type=str, default="airline",
                        choices=["airline", "retail"],
                        help="Domain to test (default: airline)")
    parser.add_argument("--port", type=int, default=9002,
                        help="vLLM server port")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="Base model name")
    parser.add_argument("--task-ids", nargs="*", type=str, default=None,
                        help="Task IDs to test (default: all tasks covered by skills)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load scaling skills
    with open(args.skills_json) as f:
        skills_data = json.load(f)

    domain_prefix = "A" if args.domain == "airline" else "R"
    port_url = f"http://localhost:{args.port}/v1"

    # Build skill configs and expected routing from the JSON
    skill_configs = []
    expected_routing = {}  # task_id -> list of acceptable skill names
    all_task_ids = set()

    # Always include a general_service / base model skill
    skill_configs.append(SkillConfig(
        name="general_service",
        description="No specialized training. General inquiries only.",
        model=f"openai/{args.model}",
        llm_args={
            "api_base": port_url,
            "api_key": "EMPTY",
            "temperature": 0.0,
            "max_context_length": 32000,
        },
    ))

    for skill in skills_data["skills"]:
        skill_name = slugify(skill["title"])
        description = skill["principle"]

        skill_configs.append(SkillConfig(
            name=skill_name,
            description=description,
            # No adapter — all skills use the base model for classification only
            model=f"openai/{args.model}",
            llm_args={
                "api_base": port_url,
                "api_key": "EMPTY",
                "temperature": 0.0,
                "max_context_length": 32000,
            },
        ))

        # Parse failed_tasks for this domain
        for task_ref in skill["failed_tasks"]:
            if not task_ref.startswith(domain_prefix):
                continue
            task_id = task_ref[1:]  # Strip A/R prefix
            all_task_ids.add(task_id)
            if task_id not in expected_routing:
                expected_routing[task_id] = []
            expected_routing[task_id].append(skill_name)

    print(f"Loaded {len(skills_data['skills'])} skills, {len(skill_configs)} total (incl. general_service)")
    print(f"Domain: {args.domain}, tasks with coverage: {len(expected_routing)}")
    print(f"\nSkills:")
    for sc in skill_configs:
        print(f"  {sc.name}: {sc.description[:80]}...")
    print()

    # Build orchestrator config
    orch_config = OrchestratorConfig(
        orchestrator_model=f"openai/{args.model}",
        orchestrator_llm_args={
            "temperature": 0.0,
            "api_base": port_url,
            "api_key": "EMPTY",
        },
        skills=skill_configs,
        routing_strategy="per_conversation",
        routing_mode="classifier",
        routing_topk=1,
        routing_context_window=10,
    )

    # Load domain environment
    from tau2.registry import registry
    env_constructor = registry.get_env_constructor(args.domain)
    environment = env_constructor()
    tools = environment.get_tools()
    policy = environment.get_policy()

    # Build orchestrator agent
    agent = OrchestratorAgent(
        tools=tools,
        domain_policy=policy,
        orchestrator_config=orch_config,
        llm_args={},
    )

    # Determine task IDs
    if args.task_ids:
        task_ids = [str(t) for t in args.task_ids]
    else:
        task_ids = sorted(all_task_ids, key=lambda x: int(x))

    tasks = get_tasks(task_set_name=args.domain, task_ids=task_ids)
    print(f"Testing classifier routing on {len(tasks)} tasks\n")

    # User sim config
    user_llm = f"openai/{args.model}"
    user_llm_args = {
        "temperature": 0.0,
        "max_context_length": 32000,
        "api_base": port_url,
        "api_key": "EMPTY",
    }

    # Run classifier for each task
    correct = 0
    total = 0
    results = []

    for task in sorted(tasks, key=lambda t: int(t.id)):
        tid = task.id
        task_seed = get_task_seed(args.seed, 0, tid)

        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None
        user_sim = UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=user_llm,
            llm_args=deepcopy(user_llm_args),
        )
        user_sim.set_seed(task_seed)
        user_state = user_sim.get_init_state()
        greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        user_msg, user_state = user_sim.generate_next_message(greeting, user_state)

        # Build messages for routing
        agent_state = agent.get_init_state(message_history=[greeting])
        agent_state.messages.append(user_msg)
        full_messages = agent_state.system_messages + agent_state.messages
        agent.set_seed(task_seed)

        # Run routing
        decision = agent._route(full_messages)

        exp = expected_routing.get(tid)
        if exp is not None:
            is_correct = decision.selected_skill in exp
            correct += is_correct
            total += 1
            status = "✓" if is_correct else "✗"
            results.append((tid, exp, decision.selected_skill, is_correct))
            acc_str = "|".join(exp)
            print(f"  T{tid}: {status}  acceptable=[{acc_str}]  got={decision.selected_skill}")
            if not is_correct:
                print(f"       reason: {decision.reasoning}")
        else:
            results.append((tid, None, decision.selected_skill, None))
            print(f"  T{tid}: routed -> {decision.selected_skill} (no coverage)")

    # Summary
    print(f"\n{'='*60}")
    if total:
        print(f"Exact rate: {correct}/{total} ({100*correct/total:.1f}%)")

        print(f"\nMisclassified:")
        for tid, exp, got, ok in results:
            if ok is False:
                print(f"  T{tid}: acceptable={exp}, got={got}")

        # Routing distribution
        print(f"\nRouting distribution:")
        by_skill = defaultdict(list)
        for tid, _, got, _ in results:
            by_skill[got].append(tid)
        for skill, tids in sorted(by_skill.items(), key=lambda x: -len(x[1])):
            print(f"  {skill}: {len(tids)} tasks")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
