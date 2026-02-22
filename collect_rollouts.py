"""Collect agent-user trajectories for tau_tool_calling or adversarial_policy environments.

Runs a vLLM-served agent model and user model through seeds in parallel,
saving trajectories as JSON in tau2-bench eval format compatible with
train_sft.py (via SFTBuffer).

Seeds [start_seed, start_seed + num_seeds) are run once. Trajectories
meeting --reward-threshold are kept; the rest are discarded.

Usage:
    # Collect trajectories for 100 seeds:
    python collect_rollouts.py \
        --env tau_tool_calling \
        --base-url http://localhost:9090/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --num-seeds 100

    # Sample 5 attempts per seed, keep top 2 by reward:
    python collect_rollouts.py \
        --env tau_tool_calling \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --num-seeds 100 \
        --num-samples 5 \
        --select-topk 2 \
        --temperature 0.7

    # Lower threshold to keep partial successes:
    python collect_rollouts.py \
        --env adversarial_policy \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --num-seeds 100 \
        --reward-threshold 0.5

Output:
    JSON file(s) in tau2-bench eval format (one per domain), directly
    loadable by SFTBuffer: python train_sft.py --sft-data <output.json>
"""

import sys
import json
import time
import random
import argparse
import threading
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

import requests

from adversarial_policy_game.llm_user import UserLLMClient
from adversarial_policy_game.constants import AIRLINE_POLICY, RETAIL_POLICY


# ---------------------------------------------------------------------------
# vLLM client with function calling
# ---------------------------------------------------------------------------
class VLLMClient:
    def __init__(self, base_url: str, model: str, max_tokens: int = 512,
                 temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=128, pool_connections=128)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

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
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        return {
            "content": choice.get("content"),
            "tool_calls": choice.get("tool_calls"),
        }


# ---------------------------------------------------------------------------
# Conversation format converter: internal -> tau2-bench message format
# ---------------------------------------------------------------------------

def conversation_to_tau2_messages(
    conversation: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Convert internal conversation to tau2-bench message format.

    Internal format uses: {"role": "user"|"assistant"|"tool_call"|"tool_result", "text": "..."}
    Tau2-bench format uses: {"role": "user"|"assistant"|"tool", "content": "...", ...}

    The output is directly compatible with SFTBuffer.load_sft_samples().
    """
    messages = []
    tool_call_counter = 0

    i = 0
    while i < len(conversation):
        msg = conversation[i]
        role, text = msg["role"], msg["text"]

        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "assistant":
            messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": None,
            })
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
            # Pair with tool_result if next message is one
            if (i + 1 < len(conversation)
                    and conversation[i + 1]["role"] == "tool_result"):
                messages.append({
                    "role": "tool",
                    "content": conversation[i + 1]["text"],
                    "id": tc_id,
                })
                i += 1
        # skip standalone tool_result (already handled above)
        i += 1

    return messages


# ---------------------------------------------------------------------------
# Episode runner (unified for both envs)
# ---------------------------------------------------------------------------
def run_episode(game, client: VLLMClient, seed: int, env_type: str,
                user_difficulty: str = None) -> Dict[str, Any]:
    """Run one episode using OpenAI function calling API.

    Works with both TauToolCallingEnv and AdversarialPolicyGame since they
    share the same get_system_prompt/get_messages/get_tool_schemas/step API.
    """
    try:
        if env_type == "adversarial_policy":
            game.reset(seed, user_difficulty=user_difficulty)
        else:
            game.reset(seed)
    except Exception as e:
        return {"seed": seed, "error": str(e)}

    system_prompt = game.get_system_prompt()
    tools = game.get_tool_schemas()

    step = 0
    while not game.done and step < game.max_steps:
        messages = game.get_messages()

        try:
            result = client.generate_with_tools(system_prompt, messages, tools)
        except Exception as e:
            if not game.done:
                if hasattr(game, "_finalize_with_verification"):
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
            game.step(action)
        elif content:
            action = json.dumps({
                "name": "respond_to_user",
                "arguments": {"message": content},
            })
            game.step(action)
        else:
            action = json.dumps({
                "name": "respond_to_user",
                "arguments": {"message": "Let me help you with that."},
            })
            game.step(action)

        step += 1

    if not game.done:
        if hasattr(game, "_finalize_with_verification"):
            game._finalize_with_verification()

    summary = game.get_summary()

    # Build result with all trajectory data
    result = {
        "seed": seed,
        "reward": summary["reward"],
        "reason": summary.get("reason", ""),
        "steps": summary["steps"],
        "transferred": summary.get("transferred", False),
        "conversation": list(game._conversation),
        "system_prompt": system_prompt,
        "user_system_prompt": game._scenario.user_system_prompt,
        "tool_schemas": tools,
        "tool_calls": summary.get("tool_calls", []),
        "domain": summary.get("domain", ""),
    }

    if env_type == "tau_tool_calling":
        result.update({
            "scenario_type": summary.get("scenario_type", ""),
            "description": summary.get("description", ""),
            "is_refusal": summary.get("is_refusal", False),
            "expected_actions": summary.get("expected_actions", []),
            "communicate_info": summary.get("communicate_info", []),
            "key_facts": summary.get("key_facts", {}),
            "conversation_length": summary.get("conversation_length", 0),
        })
    elif env_type == "adversarial_policy":
        sc = game._scenario
        result.update({
            "template_id": summary.get("template_id", -1),
            "template_name": summary.get("template_name", ""),
            "pressure_type": summary.get("pressure_type", ""),
            "correct_behavior": summary.get("correct_behavior", ""),
            "is_adversarial": sc.template_id <= 12 if sc else False,
            "communicate_info": summary.get("communicate_info", []),
            "key_facts": summary.get("key_facts", {}),
        })
        if user_difficulty:
            result["user_difficulty"] = user_difficulty

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect agent-user trajectories for game environments"
    )
    parser.add_argument("--env", required=True,
                        choices=["tau_tool_calling", "adversarial_policy"],
                        help="Environment to use")
    parser.add_argument("--base-url", default="http://localhost:9090/v1",
                        help="vLLM server URL for agent (default: http://localhost:9090/v1)")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="Agent model name")
    parser.add_argument("--user-base-url", default=None,
                        help="User LLM server URL (default: same as --base-url)")
    parser.add_argument("--user-model", default=None,
                        help="User LLM model name (default: same as --model)")
    parser.add_argument("--num-seeds", type=int, required=True,
                        help="Number of seeds to run")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed (default: 0)")
    parser.add_argument("--workers", "-w", type=int, default=100,
                        help="Number of parallel workers (default: 100)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per agent generation (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for agent LLM (default: 0.0)")
    parser.add_argument("--user-temperature", type=float, default=0.7,
                        help="Temperature for user LLM (default: 0.7)")
    # Sampling, selection, and filtering
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of rollout attempts per seed (default: 1)")
    parser.add_argument("--select-topk", type=int, default=1,
                        help="Keep top-k trajectories per seed by reward (default: 1)")
    parser.add_argument("--reward-threshold", type=float, default=1.0,
                        help="Minimum reward to keep a trajectory (default: 1.0)")
    # tau_tool_calling specific
    parser.add_argument("--domain", type=str, default=None,
                        choices=["airline", "retail"],
                        help="Domain filter for tau_tool_calling (default: both)")
    # adversarial_policy specific
    parser.add_argument("--adversarial-ratio", type=float, default=0.2,
                        help="Adversarial ratio for adversarial_policy (default: 0.2)")
    parser.add_argument("--user-difficulty", default="random",
                        choices=["easy", "medium", "hard", "random"],
                        help="User difficulty for adversarial_policy (default: random)")
    # output
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: /root/games/game_rollouts)")
    parser.add_argument("--output", "-o", default=None,
                        help="Override output file path (ignores --output-dir naming)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-episode details")

    args = parser.parse_args()

    num_samples = max(1, args.num_samples)
    select_topk = max(1, min(args.select_topk, num_samples))
    reward_threshold = args.reward_threshold

    # Resolve models
    user_base_url = args.user_base_url or args.base_url
    user_model = args.user_model or args.model

    # Build clients
    client = VLLMClient(args.base_url, args.model, args.max_tokens,
                        temperature=args.temperature)
    user_client = UserLLMClient(
        user_base_url, user_model,
        max_tokens=256, temperature=args.user_temperature,
    )

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path("/root/games/game_rollouts")
        agent_short = args.model.split("/")[-1]
        user_short = user_model.split("/")[-1]
        output_path = output_dir / f"{args.env}-{agent_short}-{user_short}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print config
    print(f"Environment:  {args.env}")
    print(f"Agent model:  {args.model} @ {args.base_url} (temp={args.temperature})")
    print(f"User  model:  {user_model} @ {user_base_url} (temp={args.user_temperature})")
    print(f"Seeds:        {args.num_seeds} (start={args.start_seed})")
    print(f"Workers:      {args.workers}")
    if num_samples > 1:
        print(f"Samples/seed: {num_samples}  select top-{select_topk}")
    print(f"Threshold:    {reward_threshold}")
    print(f"Output:       {output_path}")
    if args.env == "tau_tool_calling":
        print(f"Domain:       {args.domain or 'both'}")
    elif args.env == "adversarial_policy":
        print(f"Adv. ratio:   {args.adversarial_ratio}")
        print(f"User diff.:   {args.user_difficulty}")

    # Connectivity test
    try:
        test_msgs = [{"role": "user", "content": "Say OK."}]
        client.generate_with_tools("You are helpful.", test_msgs, [])
        print("Connection:   OK\n")
    except Exception as e:
        print(f"Connection FAILED: {e}")
        sys.exit(1)

    # Game factory (creates per-thread instances)
    def make_game():
        if args.env == "tau_tool_calling":
            from tau_tool_calling_env.game import TauToolCallingEnv
            return TauToolCallingEnv(
                max_steps=30,
                user_client=user_client,
                domain=args.domain,
            )
        else:
            from adversarial_policy_game.game import AdversarialPolicyGame
            return AdversarialPolicyGame(
                max_steps=20,
                user_client=user_client,
                adversarial_ratio=args.adversarial_ratio,
            )

    # Deterministic per-seed difficulty (independent of seed processing order)
    def get_seed_difficulty(seed):
        if args.env != "adversarial_policy":
            return None
        if args.user_difficulty == "random":
            return random.Random(seed + 7919).choice(["easy", "medium", "hard"])
        return args.user_difficulty

    def _run_one(seed, sample_idx, difficulty):
        game = make_game()
        return run_episode(game, client, seed, args.env,
                           user_difficulty=difficulty)

    # -------------------------------------------------------------------
    # Run all seeds in parallel (single pass, no retry waves)
    # -------------------------------------------------------------------
    num_workers = max(1, args.workers)
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

    # Build work items
    seed_difficulties = {}
    work_items = []
    for seed in seeds:
        diff = get_seed_difficulty(seed)
        seed_difficulties[seed] = diff
        for sample_idx in range(num_samples):
            work_items.append((seed, sample_idx, diff))

    results_by_seed = {seed: [] for seed in seeds}
    pending_per_seed = {seed: num_samples for seed in seeds}
    completed_seeds = 0
    passed_seeds = 0
    total_episodes = 0
    t0 = time.time()
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_item = {
            executor.submit(_run_one, seed, sidx, diff): (seed, sidx)
            for seed, sidx, diff in work_items
        }
        for future in as_completed(future_to_item):
            seed, sample_idx = future_to_item[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"seed": seed, "error": str(e)}

            results_by_seed[seed].append((sample_idx, result))
            pending_per_seed[seed] -= 1
            total_episodes += 1

            with print_lock:
                elapsed = time.time() - t0
                r = result.get("reward", 0)
                err = " ERROR" if "error" in result else ""
                sample_tag = (f"s={sample_idx+1}/{num_samples} "
                              if num_samples > 1 else "")

                # Check if this seed just completed all samples
                tag_pass = ""
                if pending_per_seed[seed] == 0:
                    completed_seeds += 1
                    best_r = max(
                        (res.get("reward", -999)
                         for _, res in results_by_seed[seed]
                         if "error" not in res),
                        default=-999,
                    )
                    if best_r >= reward_threshold:
                        tag_pass = " PASS"
                        passed_seeds += 1
                    else:
                        tag_pass = " FAIL"

                if args.env == "tau_tool_calling":
                    domain = result.get("domain", "?")
                    stype = result.get("scenario_type",
                                       result.get("error", "err"))[:14]
                    print(f"  [{passed_seeds:3d}/{completed_seeds}/{len(seeds)}] "
                          f"seed={seed:4d} {sample_tag}"
                          f"{domain:8s} {stype:14s} "
                          f"reward={r:+5.2f} "
                          f"steps={result.get('steps', 0):2d} "
                          f"({elapsed:.0f}s){err}{tag_pass}")
                else:
                    tid = result.get("template_id", "?")
                    tname = result.get("template_name",
                                       result.get("error", "err"))[:22]
                    is_adv = "ADV" if result.get("is_adversarial") else "COP"
                    print(f"  [{passed_seeds:3d}/{completed_seeds}/{len(seeds)}] "
                          f"seed={seed:4d} {sample_tag}"
                          f"T{tid:02d}:{tname:22s} {is_adv} "
                          f"reward={r:+5.2f} "
                          f"steps={result.get('steps', 0):2d} "
                          f"({elapsed:.0f}s){err}{tag_pass}")

    elapsed = time.time() - t0

    # -------------------------------------------------------------------
    # Select top-k per seed by reward and collect stats
    # -------------------------------------------------------------------
    selected = []  # list of rollout dicts
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

        # Sort by reward (desc), tiebreak by fewer steps
        passing.sort(
            key=lambda x: (x[1]["reward"], -x[1]["steps"]),
            reverse=True,
        )
        for sidx, result in passing[:select_topk]:
            selected.append(result)

    # -------------------------------------------------------------------
    # Group by domain and write tau2-bench JSON (SFTBuffer-compatible)
    # -------------------------------------------------------------------
    POLICIES = {"airline": AIRLINE_POLICY, "retail": RETAIL_POLICY}

    by_domain = defaultdict(list)
    for rollout in selected:
        domain = rollout.get("domain", "unknown")
        by_domain[domain].append(rollout)

    output_paths = []
    total_simulations = 0

    for domain, rollouts in sorted(by_domain.items()):
        policy = POLICIES.get(domain, "")

        simulations = []
        for rollout in rollouts:
            tau2_msgs = conversation_to_tau2_messages(rollout["conversation"])
            simulations.append({
                "task_id": str(rollout["seed"]),
                "reward_info": {"reward": rollout["reward"]},
                "messages": tau2_msgs,
            })

        output_data = {
            "info": {
                "environment_info": {
                    "domain_name": domain,
                    "policy": policy,
                }
            },
            "simulations": simulations,
        }

        # Determine per-domain output path
        if len(by_domain) == 1:
            path = output_path
        else:
            path = output_path.parent / f"{output_path.stem}_{domain}{output_path.suffix}"

        with open(path, "w") as f:
            json.dump(output_data, f, default=str)

        output_paths.append(path)
        total_simulations += len(simulations)
        print(f"  {domain}: {len(simulations)} simulations -> {path}")

    # -------------------------------------------------------------------
    # Summary stats
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
