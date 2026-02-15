"""Benchmark a model on the adversarial policy environment.

Runs the model through N seeds, reports per-template accuracy, reward
distribution, adversarial vs cooperative breakdown, and failure analysis.

Uses the OpenAI function-calling API (matching tau2-bench eval format).

Usage:
    python -m adversarial_policy_game.benchmark \
        --base-url http://localhost:9000/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --num-seeds 100 \
        --adversarial-ratio 0.2 \
        --rollouts rollouts.jsonl \
        --output summary.json

    # Adversarial-only (stress test policy adherence):
    python -m adversarial_policy_game.benchmark --adversarial-ratio 1.0

    # Cooperative-only (test helpfulness):
    python -m adversarial_policy_game.benchmark --adversarial-ratio 0.0

Output files:
    --rollouts FILE   JSONL with full conversations (one JSON object per line).
                      Each line has: seed, template, reward, reason, conversation
                      (list of {role, text} messages), system_prompt, tool_calls, etc.
                      Written incrementally — tail -f to watch progress.
    --output FILE     JSON summary with config, stats, and per-episode results
                      (without full conversations).
"""

import sys
import json
import time
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

import requests

from .game import AdversarialPolicyGame
from .llm_user import UserLLMClient
from .scenarios import generate_scenario, ADVERSARIAL_TEMPLATE_WEIGHTS, COOPERATIVE_TEMPLATE_WEIGHTS


# ---------------------------------------------------------------------------
# vLLM client with function calling
# ---------------------------------------------------------------------------
class VLLMClient:
    def __init__(self, base_url: str, model: str, max_tokens: int = 512):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.session = requests.Session()

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
            "temperature": 0.0,
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
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    game: AdversarialPolicyGame,
    client: VLLMClient,
    seed: int,
    verbose: bool = False,
    user_difficulty: str = None,
) -> Dict[str, Any]:
    """Run one episode using OpenAI function calling API."""
    try:
        game.reset(seed, user_difficulty=user_difficulty)
    except Exception as e:
        return {"seed": seed, "error": str(e)}

    sc = game._scenario

    if verbose:
        print(f"\n--- Seed {seed} | T{sc.template_id}:{sc.template_name} | {sc.pressure_type.value} ---")
        print(f"    Correct: {sc.ground_truth.correct_behavior}")

    system_prompt = game.get_system_prompt()
    tools = game.get_tool_schemas()

    step = 0
    while not game.done and step < game.max_steps:
        messages = game.get_messages()

        try:
            result = client.generate_with_tools(system_prompt, messages, tools)
        except Exception as e:
            if verbose:
                print(f"    ERROR generating: {e}")
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

            if verbose:
                print(f"    Step {step}: {name}({json.dumps(arguments, default=str)[:80]})")

            action = json.dumps({"name": name, "arguments": arguments})
            game.step(action)

        elif content:
            if verbose:
                print(f"    Step {step}: respond_to_user -> {content[:80]}...")

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

    summary = game.get_summary()

    is_adversarial = sc.template_id <= 12
    result = {
        "seed": seed,
        "template_id": summary["template_id"],
        "template_name": summary["template_name"],
        "domain": summary["domain"],
        "pressure_type": summary["pressure_type"],
        "reward": summary["reward"],
        "reason": summary["reason"],
        "steps": summary["steps"],
        "transferred": summary["transferred"],
        "correct_behavior": summary["correct_behavior"],
        "tool_calls": summary["tool_calls"],
        "key_facts": summary["key_facts"],
        "is_adversarial": is_adversarial,
        "conversation": list(game._conversation),
        "system_prompt": game.get_system_prompt(),
        "communicate_info": summary["communicate_info"],
    }
    if user_difficulty:
        result["user_difficulty"] = user_difficulty
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(results: List[Dict[str, Any]], elapsed: float):
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid:
        print("No valid results.")
        return

    rewards = [r["reward"] for r in valid]
    avg_reward = sum(rewards) / len(rewards)
    perfect = sum(1 for r in rewards if r == 1.0)
    partial = sum(1 for r in rewards if 0 < r < 1.0)
    neutral = sum(1 for r in rewards if r == 0.0)
    wrong_transfer = sum(1 for r in rewards if r == -0.5)
    violation = sum(1 for r in rewards if r == -1.0)

    n = len(valid)
    print(f"\n{'=' * 75}")
    print(f"  ADVERSARIAL POLICY BENCHMARK — {n} episodes in {elapsed:.1f}s")
    print(f"{'=' * 75}")

    print(f"\n  Overall: avg_reward = {avg_reward:+.3f}")
    print(f"  ┌─────────────────┬───────┬─────────┐")
    print(f"  │ Category        │ Count │   Rate  │")
    print(f"  ├─────────────────┼───────┼─────────┤")
    print(f"  │ Perfect (+1.0)  │ {perfect:5d} │ {perfect/n*100:5.1f}%  │")
    print(f"  │ Partial (0,+1)  │ {partial:5d} │ {partial/n*100:5.1f}%  │")
    print(f"  │ Neutral (0.0)   │ {neutral:5d} │ {neutral/n*100:5.1f}%  │")
    print(f"  │ Bad xfer (-0.5) │ {wrong_transfer:5d} │ {wrong_transfer/n*100:5.1f}%  │")
    print(f"  │ Violation(-1.0) │ {violation:5d} │ {violation/n*100:5.1f}%  │")
    print(f"  └─────────────────┴───────┴─────────┘")

    if errors:
        print(f"\n  Errors: {len(errors)} seeds failed to run")

    # Adversarial vs Cooperative
    adv = [r for r in valid if r["is_adversarial"]]
    coop = [r for r in valid if not r["is_adversarial"]]
    if adv and coop:
        adv_avg = sum(r["reward"] for r in adv) / len(adv)
        coop_avg = sum(r["reward"] for r in coop) / len(coop)
        adv_perfect = sum(1 for r in adv if r["reward"] == 1.0)
        coop_perfect = sum(1 for r in coop if r["reward"] == 1.0)
        adv_viol = sum(1 for r in adv if r["reward"] == -1.0)
        coop_viol = sum(1 for r in coop if r["reward"] == -1.0)

        print(f"\n  Adversarial vs Cooperative:")
        print(f"  ┌──────────────┬───────┬──────────┬─────────┬───────────┐")
        print(f"  │ Type         │   N   │ Avg Rwd  │ Perfect │ Violation │")
        print(f"  ├──────────────┼───────┼──────────┼─────────┼───────────┤")
        print(f"  │ Adversarial  │ {len(adv):5d} │ {adv_avg:+7.3f} │ {adv_perfect:4d}/{len(adv):<3d} │ {adv_viol:4d}/{len(adv):<3d}  │")
        print(f"  │ Cooperative  │ {len(coop):5d} │ {coop_avg:+7.3f} │ {coop_perfect:4d}/{len(coop):<3d} │ {coop_viol:4d}/{len(coop):<3d}  │")
        print(f"  └──────────────┴───────┴──────────┴─────────┴───────────┘")

    # By template
    print(f"\n  By Template:")
    template_results = defaultdict(list)
    for r in valid:
        key = f"T{r['template_id']:02d}:{r['template_name']}"
        template_results[key].append(r["reward"])

    print(f"  ┌────────────────────────────────────────────┬─────┬──────────┬─────────┬───────────┐")
    print(f"  │ Template                                   │  N  │ Avg Rwd  │ Perfect │ Violation │")
    print(f"  ├────────────────────────────────────────────┼─────┼──────────┼─────────┼───────────┤")
    for tname in sorted(template_results.keys()):
        rs = template_results[tname]
        avg = sum(rs) / len(rs)
        perf = sum(1 for r in rs if r == 1.0)
        viol = sum(1 for r in rs if r == -1.0)
        tname_display = tname[:42]
        print(f"  │ {tname_display:<42s} │ {len(rs):3d} │ {avg:+7.3f} │ {perf:3d}/{len(rs):<3d} │ {viol:3d}/{len(rs):<3d}  │")
    print(f"  └────────────────────────────────────────────┴─────┴──────────┴─────────┴───────────┘")

    # By domain
    print(f"\n  By Domain:")
    domain_results = defaultdict(list)
    for r in valid:
        domain_results[r["domain"]].append(r["reward"])
    for domain in sorted(domain_results.keys()):
        rs = domain_results[domain]
        avg = sum(rs) / len(rs)
        perf = sum(1 for r in rs if r == 1.0)
        print(f"    {domain:10s}  n={len(rs):3d}  avg={avg:+.3f}  perfect={perf}/{len(rs)}")

    # By pressure type
    print(f"\n  By Pressure Type:")
    pressure_results = defaultdict(list)
    for r in valid:
        pressure_results[r["pressure_type"]].append(r["reward"])
    for ptype in sorted(pressure_results.keys()):
        rs = pressure_results[ptype]
        avg = sum(rs) / len(rs)
        perf = sum(1 for r in rs if r == 1.0)
        viol = sum(1 for r in rs if r == -1.0)
        print(f"    {ptype:20s}  n={len(rs):3d}  avg={avg:+.3f}  perfect={perf}/{len(rs)}  violation={viol}/{len(rs)}")

    # Failure analysis
    failures = [r for r in valid if r["reward"] <= -0.5]
    if failures:
        print(f"\n  Failure Analysis ({len(failures)} failures):")
        reason_counts = Counter(r["reason"] for r in failures)
        for reason, count in reason_counts.most_common(10):
            print(f"    [{count:3d}] {reason[:75]}")

        print(f"\n  Sample failures (up to 10):")
        for r in failures[:10]:
            tool_names = [tc["name"] for tc in r.get("tool_calls", [])]
            print(f"    Seed {r['seed']:4d} T{r['template_id']:02d}:{r['template_name'][:25]:25s} "
                  f"reward={r['reward']:+.1f}")
            print(f"      Correct: {r['correct_behavior'][:70]}")
            print(f"      Tools: {tool_names}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a model on the adversarial policy environment"
    )
    parser.add_argument("--base-url", default="http://localhost:9000/v1",
                        help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="Model name")
    parser.add_argument("--num-seeds", type=int, default=100,
                        help="Number of seeds to evaluate")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per generation")
    parser.add_argument("--adversarial-ratio", type=float, default=0.4,
                        help="Fraction adversarial vs cooperative (0.0=all coop, 1.0=all adv)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-episode details")
    parser.add_argument("--output", "-o", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--user-base-url", default=None,
                        help="User LLM server URL (default: same as --base-url)")
    parser.add_argument("--user-model", default=None,
                        help="User LLM model name (default: same as --model)")
    parser.add_argument("--user-temperature", type=float, default=0.7,
                        help="Temperature for user LLM")
    parser.add_argument("--rollouts", default=None,
                        help="Save full conversation rollouts to JSONL file")
    parser.add_argument("--user-difficulty", default="random",
                        choices=["easy", "medium", "hard", "random"],
                        help="User difficulty level (easy/medium/hard/random). "
                             "Only affects adversarial scenarios (T1-T12).")
    args = parser.parse_args()

    client = VLLMClient(args.base_url, args.model, args.max_tokens)

    user_base_url = args.user_base_url or args.base_url
    user_model = args.user_model or args.model
    user_client = UserLLMClient(
        user_base_url, user_model,
        max_tokens=256, temperature=args.user_temperature,
    )

    game = AdversarialPolicyGame(
        max_steps=30,
        user_client=user_client,
        adversarial_ratio=args.adversarial_ratio,
    )

    # Connectivity test
    print(f"Agent model: {args.model} @ {args.base_url}")
    print(f"User  model: {user_model} @ {user_base_url} (temp={args.user_temperature})")
    print(f"Adversarial ratio: {args.adversarial_ratio}")
    print(f"User difficulty: {args.user_difficulty}")
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.num_seeds - 1}")

    try:
        test_msgs = [{"role": "user", "content": "Say OK."}]
        client.generate_with_tools("You are helpful.", test_msgs, [])
        print("Connection: OK\n")
    except Exception as e:
        print(f"Connection FAILED: {e}")
        sys.exit(1)

    results = []
    t0 = time.time()
    rollout_file = open(args.rollouts, "w") if args.rollouts else None

    difficulty_rng = random.Random(42)
    for i, seed in enumerate(range(args.start_seed, args.start_seed + args.num_seeds)):
        if args.user_difficulty == "random":
            episode_difficulty = difficulty_rng.choice(["easy", "medium", "hard"])
        else:
            episode_difficulty = args.user_difficulty
        result = run_episode(game, client, seed, verbose=args.verbose,
                             user_difficulty=episode_difficulty)
        results.append(result)

        # Stream rollout to JSONL as each episode completes
        if rollout_file:
            rollout_file.write(json.dumps(result, default=str) + "\n")
            rollout_file.flush()

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (args.num_seeds - i - 1) / rate if rate > 0 else 0
        r = result.get("reward", 0)
        tid = result.get("template_id", "?")
        tname = result.get("template_name", result.get("error", "error"))[:22]
        is_adv = "ADV" if result.get("is_adversarial") else "COP"
        err = " ERROR" if "error" in result else ""
        print(f"  [{i+1:3d}/{args.num_seeds}] seed={seed:4d} T{tid:02d}:{tname:22s} "
              f"{is_adv} reward={r:+5.2f} ({elapsed:.0f}s, ETA {eta:.0f}s){err}")

    if rollout_file:
        rollout_file.close()
        print(f"\nRollouts saved to {args.rollouts}")

    elapsed = time.time() - t0
    print_report(results, elapsed)

    if args.output:
        valid = [r for r in results if "error" not in r]
        rewards = [r["reward"] for r in valid]
        # Strip conversations from summary JSON (they're in --rollouts JSONL)
        summary_results = [
            {k: v for k, v in r.items() if k not in ("conversation", "system_prompt")}
            for r in results
        ]
        with open(args.output, "w") as f:
            json.dump({
                "config": {
                    "model": args.model,
                    "user_model": user_model,
                    "num_seeds": args.num_seeds,
                    "start_seed": args.start_seed,
                    "adversarial_ratio": args.adversarial_ratio,
                    "user_temperature": args.user_temperature,
                    "elapsed_seconds": elapsed,
                },
                "summary": {
                    "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
                    "perfect_rate": sum(1 for r in rewards if r == 1.0) / len(rewards) * 100 if rewards else 0,
                    "violation_rate": sum(1 for r in rewards if r == -1.0) / len(rewards) * 100 if rewards else 0,
                    "total_episodes": len(valid),
                    "errors": len(results) - len(valid),
                },
                "results": summary_results,
            }, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
