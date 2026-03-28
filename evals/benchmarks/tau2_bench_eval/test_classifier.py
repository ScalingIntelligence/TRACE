#!/usr/bin/env python3
"""
Routing-only test for orchestrator skill selection.

Runs the exact same code path as a full tau2-bench evaluation, but stops
after the first routing decision. This lets you quickly test whether
skill descriptions produce correct classifications without running the
full multi-turn simulation.

Supports both routing modes:
  - classifier: single-token structured output with logprobs (fast, deterministic)
  - llm: free-form chain-of-thought reasoning with SELECTED_SKILL: (slower, richer)

Usage:
    python test_classifier.py --config single_lora/config-airline-5skill-orchestrator.yml
    python test_classifier.py --config config.yml --routing-mode llm
    python test_classifier.py --config config.yml --routing-mode classifier

Requires a vLLM server running with the base model (for user sim + routing).
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

# Ensure tau2 is importable
tau2_src = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "src"
sys.path.insert(0, str(tau2_src))

# Set TAU2_DATA_DIR before importing tau2
tau2_data = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(tau2_data))

from tau2.agent.orchestrator_agent import OrchestratorAgent, OrchestratorConfig
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator


def get_task_seed(base_seed: int, trial: int, task_id: str) -> int:
    """Same deterministic seed as tau2's run_tasks."""
    combined = f"tau2_seed_{base_seed}_trial_{trial}_task_{task_id}"
    hash_bytes = hashlib.md5(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "big") % 1000000


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Test orchestrator classifier routing")
    parser.add_argument("--config", required=True, help="Path to orchestrator config YAML")
    parser.add_argument("--task-ids", nargs="*", type=str, default=None,
                        help="Task IDs to test (default: use expected_routing from config or all)")
    parser.add_argument("--expected", type=str, default=None,
                        help="JSON mapping task_id -> expected_skill or list (optional)")
    parser.add_argument("--solve-files", nargs="*", default=None,
                        help="skill_name=path pairs to build expected routing from result files, "
                             "e.g. general_service=base.json structured_data_reasoning=std.json")
    parser.add_argument("--routing-mode", type=str, default=None,
                        choices=["classifier", "llm", "llm_multi"],
                        help="Override routing mode from config (classifier, llm, or llm_multi)")
    parser.add_argument("--port", type=int, default=8080,
                        help="vLLM server port (default: 8080)")
    args = parser.parse_args()

    config = load_config(args.config)

    # ---- Override port if specified ----
    port_url = f"http://localhost:{args.port}/v1"
    if "vllm" in config:
        config["vllm"]["base_url"] = port_url
    orch = config.get("orchestrator", {})
    if "orchestrator_llm_args" in orch:
        orch["orchestrator_llm_args"]["api_base"] = port_url

    # ---- Override routing mode if specified ----
    if args.routing_mode:
        config.setdefault("orchestrator", {})["routing_mode"] = args.routing_mode

    # ---- Build orchestrator config ----
    orch_dict = config["orchestrator"]
    orch_config = OrchestratorConfig(**orch_dict)

    # ---- Load domain environment (for tools + policy) ----
    domain = config.get("domain", "airline")
    from tau2.registry import registry
    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor()
    tools = environment.get_tools()
    policy = environment.get_policy()

    # ---- Build orchestrator agent (for classifier logic) ----
    agent_llm_args = config.get("agent_llm_args", {})
    agent = OrchestratorAgent(
        tools=tools,
        domain_policy=policy,
        orchestrator_config=orch_config,
        llm_args=agent_llm_args,
    )

    # ---- Determine task IDs ----
    # Parse expected routing from --expected JSON or --solve-files
    expected_routing = {}
    if args.solve_files:
        # Build expected from result files: skill=path pairs
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
        # For each task solved by at least one skill, list acceptable skills
        all_solved_tasks = set()
        for solved in solve_sets.values():
            all_solved_tasks |= solved
        for tid in all_solved_tasks:
            acceptable = [s for s, solved in solve_sets.items() if tid in solved]
            expected_routing[tid] = acceptable
        print(f"  Solvable tasks: {len(expected_routing)}\n")
    elif args.expected:
        expected_routing = json.loads(args.expected)

    # Get task IDs from args, expected routing, or config
    if args.task_ids:
        task_ids = [str(t) for t in args.task_ids]
    elif expected_routing:
        task_ids = list(expected_routing.keys())
    elif config.get("task_ids"):
        task_ids = [str(t) for t in config["task_ids"]]
    else:
        task_ids = None  # all tasks

    # ---- Load tasks ----
    tasks = get_tasks(task_set_name=domain, task_ids=task_ids)
    routing_mode = orch_config.routing_mode
    print(f"Testing routing ({routing_mode} mode) on {len(tasks)} tasks\n")

    # ---- User sim config ----
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

    # ---- Top-k / top-p filtering helper (mirrors _set_weighted_lora logic) ----
    routing_topp = orch_config.routing_topp
    routing_topk = getattr(orch_config, 'routing_topk', None)

    def apply_filtering(skill_weights):
        """Apply top-k then top-p filtering to skill_weights dict, return set of kept skill names."""
        if not skill_weights:
            return set()
        # Sort by weight descending
        sorted_skills = sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)
        # Step 1: top-k filtering
        if routing_topk and len(sorted_skills) > routing_topk:
            sorted_skills = sorted_skills[:routing_topk]
        # Step 2: top-p filtering
        if routing_topp and len(sorted_skills) > 1:
            cumsum = 0.0
            keep = []
            for name, w in sorted_skills:
                keep.append(name)
                cumsum += w
                if cumsum >= routing_topp:
                    break
            return set(keep)
        return set(s[0] for s in sorted_skills)

    # ---- Run classifier for each task ----
    results = []  # (tid, acceptable, selected_skill, predicted_set)

    for task in sorted(tasks, key=lambda t: int(t.id)):
        tid = task.id
        task_seed = get_task_seed(base_seed, trial=0, task_id=tid)

        # Build user simulator (same as run_task)
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

        # Initialize user state
        user_state = user_sim.get_init_state()

        # Agent greeting (same as orchestrator.py DEFAULT_FIRST_AGENT_MESSAGE)
        greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)

        # Generate user sim's first response
        user_msg, user_state = user_sim.generate_next_message(greeting, user_state)

        print(f"T{tid} user_msg: {(user_msg.content or '')[:150]}")

        # Build the message list the orchestrator would pass to _route()
        agent_state = agent.get_init_state(message_history=[greeting])
        agent_state.messages.append(user_msg)
        full_messages = agent_state.system_messages + agent_state.messages

        # Set seed on agent (same as orchestrator.py)
        agent.set_seed(task_seed)

        # Run the routing decision
        decision = agent._route(full_messages)

        # Apply top-p to get predicted skill set
        predicted_set = apply_filtering(decision.skill_weights or {decision.selected_skill: 1.0})

        exp = expected_routing.get(tid)
        acceptable = (exp if isinstance(exp, list) else [exp]) if exp is not None else None
        results.append((tid, acceptable, decision.selected_skill, predicted_set))

        pred_str = ", ".join(sorted(predicted_set))
        if acceptable:
            any_hit = bool(predicted_set & set(acceptable))
            status = "\u2713" if any_hit else "\u2717"
            acc_str = "|".join(acceptable)
            print(f"  T{tid}: {status}  acceptable=[{acc_str}]  predicted=[{pred_str}]  top1={decision.selected_skill}")
            if not any_hit:
                print(f"       reason: {decision.reasoning}")
        else:
            print(f"  T{tid}: predicted=[{pred_str}]  top1={decision.selected_skill} (no solver)")

        print()

    # ---- Summary ----
    if expected_routing:
        # Exact rate: router predicts exactly 1 skill → is it in acceptable?
        exact_correct = 0
        exact_total = 0
        # Potential rate: any skill in predicted set is in acceptable?
        potential_correct = 0
        potential_total = 0

        for tid, acceptable, top1, predicted_set in results:
            if acceptable is None:
                continue
            acc_set = set(acceptable)
            potential_total += 1
            if predicted_set & acc_set:
                potential_correct += 1
            if len(predicted_set) == 1:
                exact_total += 1
                if predicted_set & acc_set:
                    exact_correct += 1

        print(f"\n{'='*60}")
        if potential_total:
            print(f"Potential rate: {potential_correct}/{potential_total} ({100*potential_correct/potential_total:.1f}%)  — any predicted skill ∈ acceptable")
        if exact_total:
            exact_hit_ids = [tid for tid, acc, _, ps in results if acc is not None and len(ps) == 1 and ps & set(acc)]
            exact_miss_ids = [tid for tid, acc, _, ps in results if acc is not None and len(ps) == 1 and not (ps & set(acc))]
            print(f"Exact rate:     {exact_correct}/{exact_total} ({100*exact_correct/exact_total:.1f}%)  — router predicted exactly 1 skill")
            print(f"  hits:   {', '.join(f'T{t}' for t in exact_hit_ids)}")
            print(f"  misses: {', '.join(f'T{t}' for t in exact_miss_ids)}")
        multi_total = potential_total - exact_total
        multi_correct = potential_correct - exact_correct
        if multi_total:
            multi_hit_ids = [tid for tid, acc, _, ps in results if acc is not None and len(ps) > 1 and ps & set(acc)]
            multi_miss_ids = [tid for tid, acc, _, ps in results if acc is not None and len(ps) > 1 and not (ps & set(acc))]
            print(f"Multi rate:     {multi_correct}/{multi_total} ({100*multi_correct/multi_total:.1f}%)  — router predicted >1 skill, any hit")
            print(f"  hits:   {', '.join(f'T{t}' for t in multi_hit_ids)}")
            if multi_miss_ids:
                print(f"  misses: {', '.join(f'T{t}' for t in multi_miss_ids)}")
        print(f"{'='*60}")

        print(f"\nPredicted-set size distribution:")
        size_counts = defaultdict(int)
        for _, acceptable, _, predicted_set in results:
            if acceptable is not None:
                size_counts[len(predicted_set)] += 1
        for sz in sorted(size_counts):
            print(f"  {sz} skill(s): {size_counts[sz]} tasks")

        print("\nMisclassified (solvable tasks only):")
        for tid, acceptable, top1, predicted_set in results:
            if acceptable and not (predicted_set & set(acceptable)):
                pred_str = ", ".join(sorted(predicted_set))
                print(f"  T{tid}: acceptable={acceptable}, predicted=[{pred_str}]")

        print("\nUnsolvable task routing:")
        for tid, acceptable, top1, predicted_set in results:
            if acceptable is None:
                pred_str = ", ".join(sorted(predicted_set))
                print(f"  T{tid}: -> [{pred_str}]")
    else:
        print(f"\n{'='*60}")
        print(f"Routing decisions for {len(results)} tasks (no expected routing provided)")
        print(f"{'='*60}")
        by_skill = defaultdict(list)
        for tid, _, got, _ in results:
            by_skill[got].append(tid)
        for skill, tids in sorted(by_skill.items()):
            print(f"  {skill}: {', '.join(f'T{t}' for t in tids)}")



if __name__ == "__main__":
    main()
