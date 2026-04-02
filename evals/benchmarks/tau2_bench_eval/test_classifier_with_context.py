#!/usr/bin/env python3
"""Test classifier routing with full context: domain policy + tool definitions + user message.

Same as test_classifier.py but injects the domain policy and tool schemas
into the classifier's system prompt, giving it the same information the
agent has during actual evaluation.
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

from tau2.agent.orchestrator_agent import OrchestratorAgent, OrchestratorConfig
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator


def get_task_seed(base_seed: int, trial: int, task_id: str) -> int:
    combined = f"tau2_seed_{base_seed}_trial_{trial}_task_{task_id}"
    hash_bytes = hashlib.md5(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "big") % 1000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-ids", nargs="*", type=str, default=None)
    parser.add_argument("--solve-files", nargs="*", default=None)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    port_url = f"http://localhost:{args.port}/v1"
    if "vllm" in config:
        config["vllm"]["base_url"] = port_url
    orch = config.get("orchestrator", {})
    if "orchestrator_llm_args" in orch:
        orch["orchestrator_llm_args"]["api_base"] = port_url

    orch_dict = config["orchestrator"]
    orch_config = OrchestratorConfig(**orch_dict)

    domain = config.get("domain", "airline")
    from tau2.registry import registry
    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor()
    tools = environment.get_tools()
    policy = environment.get_policy()

    agent_llm_args = config.get("agent_llm_args", {})
    agent = OrchestratorAgent(
        tools=tools, domain_policy=policy,
        orchestrator_config=orch_config, llm_args=agent_llm_args,
    )

    # Get the domain policy text and tool schemas
    domain_policy_text = policy
    tool_names = [tool.name for tool in tools]
    tool_descriptions = []
    for tool in tools:
        desc = tool.short_desc if hasattr(tool, 'short_desc') else ""
        tool_descriptions.append(f"{tool.name}: {desc}")

    print(f"Domain: {domain}")
    print(f"Policy length: {len(domain_policy_text)} chars")
    print(f"Tools: {', '.join(tool_names)}")

    # Monkey-patch the classifier system prompt to include policy + tools
    original_prompt = agent._classifier_system_prompt
    enhanced_prompt = (
        "You are a routing classifier for a customer service agent system.\n"
        "Given the domain policy, available tools, and conversation, select "
        "the skill that should handle the agent's next response.\n\n"
        "=== DOMAIN POLICY ===\n"
        f"{domain_policy_text}\n\n"
        "=== AVAILABLE TOOLS ===\n"
        f"{', '.join(tool_names)}\n\n"
        "=== SKILLS ===\n"
        + "\n".join(
            f"{label}: {agent._label_to_skill[label]} - "
            + next(s.description for s in orch_config.skills
                   if s.name == agent._label_to_skill[label])
            for label in agent._classifier_labels
        )
        + "\n\nSelect the skill whose capability best matches what this task requires. "
        "Only output the label (e.g. A, B, C). Do not output anything else."
    )
    agent._classifier_system_prompt = enhanced_prompt

    print(f"Enhanced classifier prompt: {len(enhanced_prompt)} chars")
    print()

    # Parse solve files
    expected_routing = {}
    if args.solve_files:
        solve_sets = {}
        for spec in args.solve_files:
            skill_name, path = spec.split("=", 1)
            with open(path) as f:
                data = json.load(f)
            solved = set()
            for sim in data["simulations"]:
                if sim.get("reward_info", {}).get("reward", 0) == 1:
                    solved.add(sim["task_id"])
            solve_sets[skill_name] = solved
            print(f"  {skill_name}: {len(solved)} tasks solved")
        all_solved = set()
        for solved in solve_sets.values():
            all_solved |= solved
        for tid in all_solved:
            expected_routing[tid] = [s for s, solved in solve_sets.items() if tid in solved]
        print(f"  Solvable tasks: {len(expected_routing)}\n")

    if args.task_ids:
        task_ids = [str(t) for t in args.task_ids]
    elif expected_routing:
        task_ids = list(expected_routing.keys())
    else:
        task_ids = None

    tasks = get_tasks(task_set_name=domain, task_ids=task_ids)

    # User sim config
    user_llm = config.get("user_llm", config.get("agent_llm"))
    if user_llm and user_llm.startswith("vllm://"):
        user_llm = "openai/" + user_llm[7:]
    user_llm_args = {"temperature": 0.0, **config.get("user_llm_args", {})}
    vllm_config = config.get("vllm", {})
    base_url = vllm_config.get("base_url")
    if base_url:
        user_llm_args.setdefault("api_base", base_url)
        user_llm_args.setdefault("api_key", vllm_config.get("api_key", "EMPTY"))
    base_seed = config.get("seed", 42)

    routing_topp = orch_config.routing_topp

    def apply_topp(skill_weights):
        if not skill_weights:
            return set()
        sorted_skills = sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)
        if routing_topp and len(sorted_skills) > 1:
            cumsum = 0.0
            kept = set()
            for name, w in sorted_skills:
                kept.add(name)
                cumsum += w
                if cumsum >= routing_topp:
                    break
            return kept
        return set(skill_weights.keys())

    print(f"Testing routing (classifier with full context) on {len(tasks)} tasks\n")

    correct = 0
    total = 0
    results = []

    for task in sorted(tasks, key=lambda t: int(t.id)):
        tid = task.id
        task_seed = get_task_seed(base_seed, trial=0, task_id=tid)

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

        agent_state = agent.get_init_state(message_history=[greeting])
        agent_state.messages.append(user_msg)
        full_messages = agent_state.system_messages + agent_state.messages
        agent.set_seed(task_seed)

        decision = agent._route(full_messages)
        predicted_set = apply_topp(decision.skill_weights or {decision.selected_skill: 1.0})

        exp = expected_routing.get(tid)
        acceptable = (exp if isinstance(exp, list) else [exp]) if exp is not None else None
        results.append((tid, acceptable, decision.selected_skill, predicted_set))

        pred_str = ", ".join(sorted(predicted_set))
        if acceptable:
            any_hit = bool(predicted_set & set(acceptable))
            status = "✓" if any_hit else "✗"
            acc_str = "|".join(acceptable)
            if any_hit:
                correct += 1
            total += 1
            print(f"  T{tid}: {status}  acceptable=[{acc_str}]  predicted=[{pred_str}]  top1={decision.selected_skill}")
        else:
            print(f"  T{tid}: predicted=[{pred_str}]  top1={decision.selected_skill} (no solver)")

    # Summary
    if total:
        print(f"\n{'='*60}")
        potential = sum(1 for _, acc, _, ps in results if acc and ps & set(acc))
        exact_total = sum(1 for _, acc, _, ps in results if acc and len(ps) == 1)
        exact_correct = sum(1 for _, acc, _, ps in results if acc and len(ps) == 1 and ps & set(acc))
        print(f"Potential rate: {potential}/{total} ({100*potential/total:.1f}%)")
        if exact_total:
            print(f"Exact rate:     {exact_correct}/{exact_total} ({100*exact_correct/exact_total:.1f}%)")

        print(f"\nMisclassified:")
        for tid, acc, top1, ps in results:
            if acc and not (ps & set(acc)):
                print(f"  T{tid}: acceptable={acc}, got={top1}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
