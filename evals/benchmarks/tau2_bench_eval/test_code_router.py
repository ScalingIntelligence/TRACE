#!/usr/bin/env python3
"""
Code-based routing test: uses full game source code as skill descriptions.

Instead of short text descriptions, each skill's description is the entire
training game source code. The LLM router reads the code to understand what
each skill was trained on and routes accordingly.

Usage:
    python test_code_router.py --config single_lora/config-airline-5skill-orchestrator.yml \
        --solve-files general_service=base.json tool_calling=tc.json ...
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

tau2_data = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(tau2_data))

from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator

# ---------------------------------------------------------------------------
# Game source files (relative to games root)
# ---------------------------------------------------------------------------

GAMES_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SKILL_CODE_FILES = {
    "structured_data_reasoning": [
        GAMES_ROOT / "structured_data_game_deprecated.py",
    ],
    "tool_calling": [
        GAMES_ROOT / "tau_tool_calling_env" / "game.py",
        GAMES_ROOT / "tau_tool_calling_env" / "scenarios.py",
        GAMES_ROOT / "tau_tool_calling_env" / "verification.py",
    ],
    "precondition": [
        GAMES_ROOT / "precondition_game" / "game.py",
        GAMES_ROOT / "precondition_game" / "scenarios.py",
        GAMES_ROOT / "precondition_game" / "verification.py",
    ],
    "multistep_task": [
        GAMES_ROOT / "multistep_task_game_deprecated.py",
    ],
}


def get_task_seed(base_seed: int, trial: int, task_id: str) -> int:
    combined = f"tau2_seed_{base_seed}_trial_{trial}_task_{task_id}"
    hash_bytes = hashlib.md5(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "big") % 1000000


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_code_routing_prompt(skill_names: list[str], mode: str = "llm") -> str:
    """Build a routing system prompt with full game code as descriptions."""
    parts = [
        "You are a routing orchestrator for a customer service agent system.\n",
        "Each skill below was trained by reinforcement learning on a specific game environment.",
        "The game's source code is provided so you can understand exactly what each skill learned.\n",
        "Given the customer's message, decide which skill should handle the response.\n",
    ]

    for skill_name in skill_names:
        if skill_name == "general_service":
            parts.append(f"## {skill_name}")
            parts.append("The base model with no specialized training. Use as default when no other skill clearly applies.\n")
            continue

        code_files = SKILL_CODE_FILES.get(skill_name, [])
        if code_files:
            parts.append(f"## {skill_name}")
            for code_file in code_files:
                if not code_file.exists():
                    continue
                code = code_file.read_text()
                # Truncate very long files
                if len(code) > 15000:
                    lines = code.split("\n")
                    code = "\n".join(lines[:400])
                    code += "\n# ... (truncated) ..."
                parts.append(f"### {code_file.name}\n```python\n{code}\n```\n")
        else:
            parts.append(f"## {skill_name}\nNo code available.\n")

    if mode == "classifier":
        # Build label mapping for classifier
        label_lines = []
        for i, name in enumerate(skill_names):
            label = chr(ord("A") + i)
            label_lines.append(f"{label}: {name}")
        parts.append("\nLabel mapping:\n" + "\n".join(label_lines))
        parts.append("\nOnly output the label (e.g. A, B, C). Do not output anything else.")
    else:
        parts.append(
            "Based on the conversation, select the single best skill.\n"
            "Think step-by-step about which game's training is most relevant, "
            "then state your choice.\n\n"
            "Format: end with SELECTED_SKILL: skill_name"
        )

    return "\n".join(parts)


def route_with_code(client, model, system_prompt, conversation_messages,
                    skill_names, mode="llm", seed=None):
    """Route using LLM or classifier with code-based system prompt."""
    messages = [{"role": "system", "content": system_prompt}]

    for m in conversation_messages:
        content = getattr(m, "content", None) or ""
        if not content:
            continue
        role = "assistant" if m.role == "assistant" else "user"
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": "Which skill should handle the next response?"})

    import re

    if mode == "classifier":
        labels = [chr(ord("A") + i) for i in range(len(skill_names))]
        label_to_skill = {chr(ord("A") + i): name for i, name in enumerate(skill_names)}
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"structured_outputs": {"choice": labels}},
            temperature=0.0,
            max_tokens=1,
            seed=seed,
        )
        label = resp.choices[0].message.content.strip()
        skill = label_to_skill.get(label)
        return skill, f"classifier label {label}"
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=2000,
            seed=seed,
        )
        content = resp.choices[0].message.content or ""
        match = re.search(r'SELECTED_SKILL:\s*(\S+)', content, re.IGNORECASE)
        if match:
            skill = match.group(1).strip().rstrip(".")
            reasoning = content[:match.start()].strip()
            return skill, reasoning[-200:]
        return None, content[:200]


def main():
    parser = argparse.ArgumentParser(description="Test code-based routing")
    parser.add_argument("--config", required=True)
    parser.add_argument("--routing-mode", type=str, default="classifier",
                        choices=["classifier", "llm"])
    parser.add_argument("--task-ids", nargs="*", type=str, default=None)
    parser.add_argument("--solve-files", nargs="*", default=None)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    config = load_config(args.config)
    port_url = f"http://localhost:{args.port}/v1"

    # Build expected routing from solve files
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
        all_solved_tasks = set()
        for solved in solve_sets.values():
            all_solved_tasks |= solved
        for tid in all_solved_tasks:
            acceptable = [s for s, solved in solve_sets.items() if tid in solved]
            expected_routing[tid] = acceptable
        print(f"  Solvable tasks: {len(expected_routing)}\n")

    # Get skill names from config
    skill_names = [s["name"] for s in config["orchestrator"]["skills"]]

    # Build code-based routing prompt
    system_prompt = build_code_routing_prompt(skill_names, mode=args.routing_mode)
    print(f"Routing mode: {args.routing_mode}")
    print(f"Routing prompt: {len(system_prompt)} chars, ~{len(system_prompt.split())} words\n")

    # Task IDs
    if args.task_ids:
        task_ids = [str(t) for t in args.task_ids]
    elif expected_routing:
        task_ids = list(expected_routing.keys())
    else:
        task_ids = None

    domain = config.get("domain", "airline")
    from tau2.registry import registry
    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor()

    tasks = get_tasks(task_set_name=domain, task_ids=task_ids)
    print(f"Testing code-based routing on {len(tasks)} tasks\n")

    # User sim config
    user_llm = config.get("user_llm", config.get("agent_llm"))
    if user_llm and user_llm.startswith("vllm://"):
        user_llm = "openai/" + user_llm[7:]
    user_llm_args = {"temperature": 0.0, **config.get("user_llm_args", {})}
    vllm_config = config.get("vllm", {})
    base_url = vllm_config.get("base_url", port_url)
    user_llm_args.setdefault("api_base", base_url)
    user_llm_args.setdefault("api_key", "EMPTY")

    base_seed = config.get("seed", 42)

    from openai import OpenAI
    client = OpenAI(base_url=port_url, api_key="EMPTY")
    model_name = config.get("agent_llm", "openai/Qwen/Qwen3-30B-A3B-Instruct-2507")
    if model_name.startswith("openai/"):
        model_name = model_name[len("openai/"):]

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

        print(f"T{tid} user_msg: {(user_msg.content or '')[:150]}")

        # Build message list
        conversation = [greeting, user_msg]

        skill, reasoning = route_with_code(
            client, model_name, system_prompt, conversation,
            skill_names=skill_names, mode=args.routing_mode, seed=task_seed,
        )

        exp = expected_routing.get(tid)
        if exp is not None:
            acceptable = exp if isinstance(exp, list) else [exp]
            is_correct = skill in acceptable
            correct += is_correct
            total += 1
            status = "✓" if is_correct else "✗"
            results.append((tid, acceptable, skill, is_correct))
            acc_str = "|".join(acceptable)
            print(f"  T{tid}: {status}  acceptable=[{acc_str}]  got={skill}")
            if not is_correct:
                print(f"       reason: {reasoning}")
        else:
            results.append((tid, None, skill, None))
            print(f"  T{tid}: routed -> {skill}")
        print()

    if expected_routing:
        print(f"\n{'='*60}")
        print(f"Classification Rate: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"{'='*60}")

        unique_correct = 0
        unique_total = 0
        for tid, acceptable, got, ok in results:
            if acceptable and len(acceptable) == 1:
                unique_total += 1
                if ok:
                    unique_correct += 1
        if unique_total:
            print(f"Uniquely-solvable tasks: {unique_correct}/{unique_total} ({100*unique_correct/unique_total:.1f}%)")

        print("\nMisclassified (solvable tasks only):")
        for tid, acceptable, got, ok in results:
            if ok is False:
                print(f"  T{tid}: acceptable={acceptable}, got={got}")


if __name__ == "__main__":
    main()
