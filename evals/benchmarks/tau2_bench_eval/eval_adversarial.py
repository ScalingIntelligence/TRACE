"""Evaluate adversarial policy game against vLLM model.

Uses OpenAI function calling API (matching tau2-bench format) to generate
agent actions. Measures whether failures correlate with policy adherence deficits.
"""
import sys
sys.path.insert(0, "/home/ubuntu/lambda-stanford/tarun/games")

import json
import time
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

import requests

from adversarial_policy_game import AdversarialPolicyGame, extract_action, SYSTEM_PROMPT, UserLLMClient
from adversarial_policy_game.scenarios import generate_scenario


# ---------------------------------------------------------------------------
# vLLM client with function calling support
# ---------------------------------------------------------------------------
class VLLMClient:
    def __init__(self, base_url: str, model: str, max_tokens: int = 512):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.session = requests.Session()

    def generate(self, system_prompt: str, user_content: str) -> str:
        """Simple text generation (for backwards compat / quick tests)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def generate_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate with OpenAI function calling API.

        Returns the assistant message dict with 'content' and/or 'tool_calls'.
        """
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
        data = resp.json()
        choice = data["choices"][0]["message"]
        return {
            "content": choice.get("content"),
            "tool_calls": choice.get("tool_calls"),
        }


# ---------------------------------------------------------------------------
# Evaluation loop (function calling mode)
# ---------------------------------------------------------------------------
def run_episode_fc(
    game: AdversarialPolicyGame,
    client: VLLMClient,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one episode using OpenAI function calling API."""
    game.reset(seed)
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

        conv_before = len(game._conversation)

        if tool_calls:
            # Model made a tool call
            tc = tool_calls[0]  # Handle first tool call
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
            # Model sent text — this is a message to the user
            if verbose:
                print(f"    Step {step}: respond_to_user → {content[:80]}...")

            action = json.dumps({
                "name": "respond_to_user",
                "arguments": {"message": content},
            })
            game.step(action)

        else:
            # Neither content nor tool_calls
            if verbose:
                print(f"    Step {step}: [empty response]")
            action = json.dumps({
                "name": "respond_to_user",
                "arguments": {"message": "Let me help you with that."},
            })
            game.step(action)

        # Log user messages added during this step
        if verbose:
            for msg in game._conversation[conv_before:]:
                if msg["role"] == "user":
                    print(f"    [USER]: {msg['text'][:100]}")

        step += 1

    summary = game.get_summary()

    if verbose:
        print(f"    Result: reward={summary['reward']}, reason={summary['reason']}")

    return {
        "seed": seed,
        "template_id": summary["template_id"],
        "template_name": summary["template_name"],
        "domain": summary["domain"],
        "pressure_type": summary["pressure_type"],
        "reward": summary["reward"],
        "reason": summary["reason"],
        "steps": summary["steps"],
        "correct_behavior": summary["correct_behavior"],
        "tool_calls": summary["tool_calls"],
        "key_facts": summary["key_facts"],
    }


# ---------------------------------------------------------------------------
# Evaluation loop (text mode, for models without function calling)
# ---------------------------------------------------------------------------
def run_episode_text(
    game: AdversarialPolicyGame,
    client: VLLMClient,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one episode using text-based generation (no function calling)."""
    game.reset(seed)
    sc = game._scenario

    if verbose:
        print(f"\n--- Seed {seed} | T{sc.template_id}:{sc.template_name} | {sc.pressure_type.value} ---")
        print(f"    Correct: {sc.ground_truth.correct_behavior}")

    step = 0
    while not game.done and step < game.max_steps:
        obs = game.observe(0)

        try:
            raw_output = client.generate(SYSTEM_PROMPT, obs)
        except Exception as e:
            if verbose:
                print(f"    ERROR generating: {e}")
            break

        action = extract_action(raw_output, game.legal_actions())

        if verbose:
            if action:
                parsed = json.loads(action)
                name = parsed.get("name", "?")
                if name == "respond_to_user":
                    msg = parsed["arguments"].get("message", "")
                    print(f"    Step {step}: respond_to_user → {msg[:80]}...")
                else:
                    print(f"    Step {step}: {name}({json.dumps(parsed.get('arguments', {}))[:60]})")
            else:
                print(f"    Step {step}: [no action parsed] raw={raw_output[:100]}...")

        if action is None:
            action = json.dumps({
                "name": "respond_to_user",
                "arguments": {"message": raw_output[:500]},
            })

        game.step(action)
        step += 1

    summary = game.get_summary()

    if verbose:
        print(f"    Result: reward={summary['reward']}, reason={summary['reason']}")

    return {
        "seed": seed,
        "template_id": summary["template_id"],
        "template_name": summary["template_name"],
        "domain": summary["domain"],
        "pressure_type": summary["pressure_type"],
        "reward": summary["reward"],
        "reason": summary["reason"],
        "steps": summary["steps"],
        "correct_behavior": summary["correct_behavior"],
        "tool_calls": summary["tool_calls"],
        "key_facts": summary["key_facts"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial policy game")
    parser.add_argument("--base-url", default="http://localhost:9000/v1", help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Model name")
    parser.add_argument("--num-seeds", type=int, default=50, help="Number of seeds to evaluate")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-episode details")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--mode", choices=["fc", "text"], default="fc",
                        help="Mode: 'fc' for function calling (default), 'text' for text-based")
    parser.add_argument("--user-base-url", default=None,
                        help="User LLM server URL (default: same as --base-url)")
    parser.add_argument("--user-model", default=None,
                        help="User LLM model name (default: same as --model)")
    parser.add_argument("--user-temperature", type=float, default=0.7,
                        help="Temperature for user LLM (default: 0.7)")
    parser.add_argument("--scripted-user", action="store_true",
                        help="Use scripted user instead of LLM user")
    args = parser.parse_args()

    client = VLLMClient(args.base_url, args.model, args.max_tokens)

    # Create user LLM client (unless scripted mode requested)
    user_client = None
    if not args.scripted_user:
        user_base_url = args.user_base_url or args.base_url
        user_model = args.user_model or args.model
        user_client = UserLLMClient(
            user_base_url, user_model,
            max_tokens=256, temperature=args.user_temperature,
        )
        print(f"User LLM: {user_model} @ {user_base_url} (temp={args.user_temperature})")
    else:
        print("User mode: scripted (no LLM)")

    game = AdversarialPolicyGame(max_steps=20, user_client=user_client)

    # Quick connectivity test
    print(f"Testing connection to {args.base_url}...")
    try:
        test_resp = client.generate("You are helpful.", "Say 'OK'.")
        print(f"  Model responding: {test_resp[:50]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    run_fn = run_episode_fc if args.mode == "fc" else run_episode_text
    print(f"Mode: {args.mode} ({'function calling' if args.mode == 'fc' else 'text-based'})")

    results = []
    t0 = time.time()

    for i, seed in enumerate(range(args.start_seed, args.start_seed + args.num_seeds)):
        result = run_fn(game, client, seed, verbose=args.verbose)
        results.append(result)

        # Progress
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (args.num_seeds - i - 1) / rate if rate > 0 else 0
        r = result["reward"]
        print(f"[{i+1}/{args.num_seeds}] Seed {seed} T{result['template_id']}:{result['template_name'][:20]:20s} "
              f"reward={r:+.1f} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    # Aggregate results
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE: {len(results)} episodes in {elapsed:.1f}s")
    print(f"{'='*70}")

    rewards = [r["reward"] for r in results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    pass_rate = sum(1 for r in rewards if r > 0) / len(rewards) * 100 if rewards else 0
    fail_rate = sum(1 for r in rewards if r < 0) / len(rewards) * 100 if rewards else 0
    neutral_rate = sum(1 for r in rewards if r == 0) / len(rewards) * 100 if rewards else 0

    print(f"\nOverall: avg_reward={avg_reward:.3f}, pass={pass_rate:.1f}%, fail={fail_rate:.1f}%, neutral={neutral_rate:.1f}%")

    # By template
    print(f"\nBy Template:")
    template_results = defaultdict(list)
    for r in results:
        template_results[f"T{r['template_id']}:{r['template_name']}"].append(r["reward"])
    for tname in sorted(template_results.keys()):
        rs = template_results[tname]
        avg = sum(rs) / len(rs)
        passes = sum(1 for r in rs if r > 0)
        fails = sum(1 for r in rs if r < 0)
        print(f"  {tname:45s} n={len(rs):3d} avg={avg:+.2f} pass={passes}/{len(rs)} fail={fails}/{len(rs)}")

    # By pressure type
    print(f"\nBy Pressure Type:")
    pressure_results = defaultdict(list)
    for r in results:
        pressure_results[r["pressure_type"]].append(r["reward"])
    for ptype in sorted(pressure_results.keys()):
        rs = pressure_results[ptype]
        avg = sum(rs) / len(rs)
        passes = sum(1 for r in rs if r > 0)
        print(f"  {ptype:25s} n={len(rs):3d} avg={avg:+.2f} pass={passes}/{len(rs)}")

    # By domain
    print(f"\nBy Domain:")
    domain_results = defaultdict(list)
    for r in results:
        domain_results[r["domain"]].append(r["reward"])
    for domain in sorted(domain_results.keys()):
        rs = domain_results[domain]
        avg = sum(rs) / len(rs)
        passes = sum(1 for r in rs if r > 0)
        print(f"  {domain:15s} n={len(rs):3d} avg={avg:+.2f} pass={passes}/{len(rs)}")

    # Failure analysis
    failures = [r for r in results if r["reward"] < 0]
    if failures:
        print(f"\nFailure Analysis ({len(failures)} failures):")
        reason_counts = Counter(r["reason"].split(":")[0].strip() for r in failures)
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")

        print(f"\nDetailed Failures:")
        for r in failures[:20]:
            print(f"  Seed {r['seed']} T{r['template_id']}:{r['template_name']} "
                  f"reward={r['reward']:+.1f} | {r['reason'][:80]}")
            print(f"    Correct: {r['correct_behavior'][:80]}")
            tool_names = [tc["name"] for tc in r["tool_calls"]]
            print(f"    Tools called: {tool_names}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "config": {
                    "model": args.model,
                    "num_seeds": args.num_seeds,
                    "start_seed": args.start_seed,
                    "mode": args.mode,
                    "user_model": args.user_model or args.model,
                    "user_mode": "scripted" if args.scripted_user else "llm",
                    "user_temperature": args.user_temperature,
                    "elapsed_seconds": elapsed,
                },
                "summary": {
                    "avg_reward": avg_reward,
                    "pass_rate": pass_rate,
                    "fail_rate": fail_rate,
                    "neutral_rate": neutral_rate,
                },
                "results": results,
            }, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
