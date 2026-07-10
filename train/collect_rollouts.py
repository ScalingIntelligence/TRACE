
import sys
import json
import time
import argparse
import threading
import importlib
import os
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from loguru import logger as _loguru_logger

    _loguru_logger.remove()
except ImportError:
    pass

import requests

from game_registry import get_game_spec, list_game_names


def _auto_import_game_modules() -> None:
    """Import generated capability game modules before argparse builds choices.

    TRACE-generated environments are normally written as
    `capability_<name>_game.py` at the repository root and register themselves at
    module import time. Without this import pass, `--env capability_<name>` is
    rejected before the module has a chance to call `register_game(...)`.
    """
    roots = [
        Path.cwd(),
        REPO_ROOT,
    ]
    module_names = set()
    for root in roots:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        for path in root.glob("capability_*_game.py"):
            module_names.add(path.stem)

    extra = [m.strip() for m in os.environ.get("TRACE_GAME_MODULES", "").split(",") if m.strip()]
    module_names.update(extra)

    for name in sorted(module_names):
        try:
            importlib.import_module(name)
        except Exception as e:
            print(f"[collect_rollouts] warning: failed to import {name}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# vLLM client
# ---------------------------------------------------------------------------
class VLLMClient:

    def __init__(self, base_url: str, model: str, max_tokens: int = 1024,
                 temperature: float = 0.0, top_p: float = None,
                 top_k: int = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=128, pool_connections=128)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _extra_params(self) -> dict:
        extra = {}
        if self.top_p is not None:
            extra["top_p"] = self.top_p
        if self.top_k is not None:
            extra["extra_body"] = {"top_k": self.top_k}
        return extra

    def generate_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": self.model,
            "messages": all_messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self._extra_params(),
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=120,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        return {
            "content": choice.get("content"),
            "tool_calls": choice.get("tool_calls"),
        }

    def generate_text(self, system_prompt: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self._extra_params(),
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Conversation format converter
# ---------------------------------------------------------------------------
def conversation_to_openai_messages(
    conversation: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    if conversation and "content" in conversation[0] and "text" not in conversation[0]:
        return list(conversation)

    messages = []
    tool_call_counter = 0

    i = 0
    while i < len(conversation):
        msg = conversation[i]
        role, text = msg["role"], msg["text"]

        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": text, "tool_calls": None})
        elif role == "tool_call":
            tc = json.loads(text)
            tc_id = f"tool-{tool_call_counter:04d}"
            tool_call_counter += 1
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc_id,
                    "name": tc["name"],
                    "arguments": tc.get("arguments", {}),
                }],
            })
            if (i + 1 < len(conversation)
                    and conversation[i + 1]["role"] == "tool_result"):
                messages.append({
                    "role": "tool",
                    "content": conversation[i + 1]["text"],
                    "id": tc_id,
                })
                i += 1
        i += 1

    return messages


# ---------------------------------------------------------------------------
# Generic episode runner
# ---------------------------------------------------------------------------
def run_episode(game, client: VLLMClient, seed: int) -> Dict[str, Any]:
    try:
        game.reset(seed)
    except Exception as e:
        return {"seed": seed, "error": str(e)}

    system_prompt = game.get_system_prompt()
    tools = game.get_tool_schemas()
    max_steps = getattr(game, "max_steps", getattr(game, "_max_steps", 30))

    step = 0
    while not game.done and step < max_steps:
        messages = game.get_messages()

        try:
            result = client.generate_with_tools(system_prompt, messages, tools)
        except Exception as e:
            if not game.done and hasattr(game, "_finalize_with_verification"):
                game._finalize_with_verification()
            break

        content = result.get("content")
        tool_calls = result.get("tool_calls")

        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "")
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            action = json.dumps({"name": name, "arguments": arguments})
        elif content:
            action = json.dumps({"name": "respond_to_user", "arguments": {"message": content}})
        else:
            action = json.dumps({"name": "respond_to_user",
                                 "arguments": {"message": "Let me help you with that."}})

        game.step(action)
        step += 1

    if not game.done and hasattr(game, "_finalize_with_verification"):
        game._finalize_with_verification()

    summary = game.get_summary() if hasattr(game, "get_summary") else {}
    reward = summary.get("reward", game.rewards.get(0, 0.0))

    return {
        "seed": seed,
        "reward": reward,
        "reason": summary.get("reason", ""),
        "steps": summary.get("steps", step),
        "conversation": list(getattr(game, "_conversation", [])),
        "system_prompt": system_prompt,
        "tool_schemas": tools,
        "domain": summary.get("domain", ""),
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def write_output(selected: List[Dict], output_path: Path):
    by_domain: Dict[str, list] = defaultdict(list)
    for rollout in selected:
        domain = rollout.get("domain") or "unknown"
        by_domain[domain].append(rollout)

    output_paths = []
    total_simulations = 0

    for domain, rollouts in sorted(by_domain.items()):
        simulations = []
        for rollout in rollouts:
            openai_msgs = conversation_to_openai_messages(rollout["conversation"])
            simulations.append({
                "task_id": str(rollout["seed"]),
                "sample_idx": rollout.get("sample_idx"),
                "reward_info": {"reward": rollout["reward"]},
                "messages": openai_msgs,
            })

        output_data = {
            "info": {"environment_info": {"domain_name": domain}},
            "simulations": simulations,
        }

        if len(by_domain) == 1:
            path = output_path
        else:
            path = output_path.parent / f"{output_path.stem}_{domain}{output_path.suffix}"

        with open(path, "w") as f:
            json.dump(output_data, f, default=str)

        output_paths.append(path)
        total_simulations += len(simulations)
        print(f"  {domain}: {len(simulations)} simulations -> {path}")

    return output_paths, total_simulations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _auto_import_game_modules()
    all_games = list_game_names()

    parser = argparse.ArgumentParser(
        description="Collect agent trajectories for game environments",
    )
    parser.add_argument("--env", required=True, choices=all_games, help="Game environment")
    parser.add_argument("--base-url", default="http://localhost:9090/v1", help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Agent model name")
    parser.add_argument("--num-seeds", type=int, required=True, help="Number of seeds to run")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--workers", "-w", type=int, default=100, help="Parallel workers")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per generation (default: from game registry)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--num-samples", type=int, default=1, help="Rollout attempts per seed")
    parser.add_argument("--select-topk", type=int, default=1, help="Keep top-k per seed by reward")
    parser.add_argument("--reward-threshold", type=float, default=1.0, help="Minimum reward to keep")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--output", "-o", default=None, help="Override output file path")

    args = parser.parse_args()

    game_spec = get_game_spec(args.env)
    max_tokens = args.max_tokens if args.max_tokens is not None else game_spec.max_gen_tokens

    num_samples = max(1, args.num_samples)
    select_topk = max(1, min(args.select_topk, num_samples))
    reward_threshold = args.reward_threshold

    client = VLLMClient(args.base_url, args.model, max_tokens,
                        temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k)

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path("game_rollouts")
        agent_short = args.model.split("/")[-1]
        parts = [args.env, agent_short, f"n{args.num_seeds}", f"s{args.start_seed}"]
        if num_samples > 1:
            parts.append(f"samp{num_samples}")
            if select_topk > 1:
                parts.append(f"top{select_topk}")
        if args.temperature != 0.0:
            parts.append(f"t{args.temperature}")
        if reward_threshold != 1.0:
            parts.append(f"r{reward_threshold}")
        output_path = output_dir / f"{'-'.join(parts)}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print config
    sampling_parts = [f"temp={args.temperature}"]
    if args.top_p is not None:
        sampling_parts.append(f"top_p={args.top_p}")
    if args.top_k is not None:
        sampling_parts.append(f"top_k={args.top_k}")
    print(f"Environment:  {args.env}")
    print(f"Agent model:  {args.model} @ {args.base_url} ({', '.join(sampling_parts)})")
    print(f"Max tokens:   {max_tokens}")
    print(f"Seeds:        {args.num_seeds} (start={args.start_seed})")
    print(f"Workers:      {args.workers}")
    if num_samples > 1:
        print(f"Samples/seed: {num_samples}  select top-{select_topk}")
    print(f"Threshold:    {reward_threshold}")
    print(f"Output:       {output_path}")

    # Connectivity test
    try:
        test_msgs = [{"role": "user", "content": "Say OK."}]
        client.generate_with_tools("You are helpful.", test_msgs, [])
        print("Connection:   OK\n")
    except Exception as e:
        print(f"Connection FAILED: {e}")
        sys.exit(1)

    # Game factory — uses the game registry (same as train_grpo.py)
    def make_game():
        return game_spec.make_env()

    def _run_one(seed, sample_idx):
        game = make_game()
        result = run_episode(game, client, seed)
        result["sample_idx"] = sample_idx
        return result

    # -------------------------------------------------------------------
    # Run all seeds in parallel
    # -------------------------------------------------------------------
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    work_items = [(seed, si) for seed in seeds for si in range(num_samples)]

    results_by_seed: Dict[int, list] = {seed: [] for seed in seeds}
    pending_per_seed = {seed: num_samples for seed in seeds}
    completed_seeds = 0
    passed_seeds = 0
    total_episodes = 0
    t0 = time.time()
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_item = {
            executor.submit(_run_one, seed, sidx): (seed, sidx)
            for seed, sidx in work_items
        }
        for future in as_completed(future_to_item):
            seed, sample_idx = future_to_item[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"seed": seed, "error": str(e)}

            with print_lock:
                results_by_seed[seed].append((sample_idx, result))
                pending_per_seed[seed] -= 1
                total_episodes += 1
                elapsed = time.time() - t0

                tag = ""
                if pending_per_seed[seed] == 0:
                    completed_seeds += 1
                    best_r = max(
                        (res.get("reward", -999)
                         for _, res in results_by_seed[seed]
                         if "error" not in res),
                        default=-999,
                    )
                    if best_r >= reward_threshold:
                        tag = " PASS"
                        passed_seeds += 1
                    else:
                        tag = " FAIL"

                r = result.get("reward", 0)
                err = " ERROR" if "error" in result else ""
                sample_tag = f"s={sample_idx+1}/{num_samples} " if num_samples > 1 else ""
                print(f"  [{passed_seeds:3d}/{completed_seeds}/{len(seeds)}] "
                      f"seed={seed:4d} {sample_tag}"
                      f"reward={r:+5.2f} "
                      f"steps={result.get('steps', 0):2d} "
                      f"({elapsed:.0f}s){err}{tag}")

    elapsed = time.time() - t0

    # -------------------------------------------------------------------
    # Select top-k per seed by reward
    # -------------------------------------------------------------------
    selected = []
    all_raw_rewards = []
    rejected_count = 0

    for seed in seeds:
        attempts = results_by_seed[seed]
        sorted_by_idx = sorted(attempts, key=lambda x: x[0])
        sample_rewards = [res.get("reward", None) for _, res in sorted_by_idx]
        all_raw_rewards.extend([r for r in sample_rewards if r is not None])

        passing = [
            (sidx, res) for sidx, res in attempts
            if "error" not in res
            and res.get("reward", -999) >= reward_threshold
        ]
        if not passing:
            rejected_count += 1
            continue

        if select_topk >= len(passing):
            # Calibration needs the full reward distribution for each seed.
            # Preserve sample order when keeping all attempts.
            passing.sort(key=lambda x: x[0])
        else:
            passing.sort(key=lambda x: (x[1]["reward"], -x[1]["steps"]), reverse=True)
        for sidx, result in passing[:select_topk]:
            selected.append(result)

    # -------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------
    output_paths, total_simulations = write_output(selected, output_path)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    selected_rewards = [r["reward"] for r in selected]

    print(f"\n{'=' * 65}")
    print(f"  ROLLOUT COLLECTION COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 65}")
    print(f"  Seeds run:        {len(seeds)}")
    print(f"  Seeds passed:     {passed_seeds} "
          f"(pass rate {passed_seeds/max(len(seeds),1)*100:.1f}%)")
    print(f"  Seeds rejected:   {rejected_count} "
          f"(no sample with reward >= {reward_threshold})")
    print(f"  Total episodes:   {total_episodes}")

    if selected_rewards:
        avg = sum(selected_rewards) / len(selected_rewards)
        perfect = sum(1 for r in selected_rewards if r >= 1.0)
        print(f"\n  Selected trajectories: {len(selected)}")
        print(f"  Avg reward (selected): {avg:+.3f}")
        print(f"  Perfect (selected):    {perfect}/{len(selected_rewards)} "
              f"({perfect/len(selected_rewards)*100:.1f}%)")

    if all_raw_rewards:
        raw_avg = sum(all_raw_rewards) / len(all_raw_rewards)
        raw_pass = sum(1 for r in all_raw_rewards if r >= reward_threshold)
        print(f"\n  All attempts ({len(all_raw_rewards)}):")
        print(f"    Avg reward:   {raw_avg:+.3f}")
        print(f"    Pass rate:    {raw_pass}/{len(all_raw_rewards)} "
              f"({raw_pass/len(all_raw_rewards)*100:.1f}%)")

    print(f"\n  Simulations saved: {total_simulations}")
    for p in output_paths:
        print(f"    {p}")


if __name__ == "__main__":
    main()
