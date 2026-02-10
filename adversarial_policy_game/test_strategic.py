"""Strategic test: run full episodes with LLM agent + LLM adversarial user.

Tests the training environment against patterns from tau2-bench regressions:
- Template 1 (ineligible_cancellation): maps to airline tasks 0, 1, 36, 45
- Template 2 (basic_economy_modification): maps to airline task 36
- Template 5 (false_policy_claims): maps to airline task 47
- Template 6 (wrong_payment_method): maps to retail task 16
- Template 7 (individual_item_cancel): maps to retail task 31
- Template 12 (multi_reservation_selective): multi-entity tracking

Usage:
    python -m adversarial_policy_game.test_strategic [--seeds N] [--template T]
"""

import json
import sys
import time
import argparse
import requests
from typing import Dict, List, Any, Optional

from .game import AdversarialPolicyGame
from .llm_user import UserLLMClient
from .scenarios import generate_scenario


# =====================================================================
# Agent LLM Client (generates agent actions from observations)
# =====================================================================

class AgentLLMClient:
    """Calls vLLM to generate agent actions."""

    def __init__(self, base_url: str, model: str, max_tokens: int = 512, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = requests.Session()

    def generate_action(self, system_prompt: str, observation: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""


def run_episode(
    seed: int,
    agent_client: AgentLLMClient,
    user_client: UserLLMClient,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single episode and return results."""
    game = AdversarialPolicyGame(max_steps=30, user_client=user_client)

    t0 = time.time()
    try:
        game.reset(seed)
    except Exception as e:
        return {"seed": seed, "error": f"reset failed: {e}"}

    scenario = game._scenario
    if verbose:
        print(f"\n{'='*70}")
        print(f"Seed {seed} | Template {scenario.template_id}: {scenario.template_name}")
        print(f"Domain: {scenario.domain} | Pressure: {scenario.pressure_type.value}")
        print(f"Correct behavior: {scenario.ground_truth.correct_behavior}")
        print(f"communicate_info: {scenario.ground_truth.communicate_info}")
        print(f"{'='*70}")
        print(f"\n[USER]: {game._conversation[0]['text']}\n")

    step = 0
    while not game.done:
        obs = game.observe(0)
        raw = agent_client.generate_action("", obs)

        if verbose:
            # Show what agent generated
            try:
                tc = json.loads(raw) if raw.strip().startswith("{") else None
            except json.JSONDecodeError:
                tc = None

            if tc and tc.get("name") == "respond_to_user":
                print(f"[AGENT]: {tc['arguments'].get('message', '')}")
            elif tc:
                print(f"[TOOL]:  {tc['name']}({json.dumps(tc.get('arguments', {}))})")
            else:
                print(f"[RAW]:   {raw[:200]}")

        game.step(raw)
        step += 1

        # Print new user messages
        if verbose:
            for msg in game._conversation:
                if msg["role"] == "tool_result" and msg == game._conversation[-1]:
                    result_preview = msg["text"][:150]
                    print(f"[RESULT]: {result_preview}...")
                elif msg["role"] == "user" and msg == game._conversation[-1]:
                    print(f"\n[USER]: {msg['text']}\n")

    elapsed = time.time() - t0
    summary = game.get_summary()
    summary["seed"] = seed
    summary["elapsed_s"] = round(elapsed, 1)

    if verbose:
        print(f"\n--- Result ---")
        print(f"Reward: {summary['reward']}")
        print(f"Reason: {summary['reason']}")
        print(f"Steps: {summary['steps']}, Transferred: {summary['transferred']}")
        print(f"Time: {elapsed:.1f}s")

    return summary


# =====================================================================
# Seed selection: strategic coverage of regression-relevant templates
# =====================================================================

def find_seeds_for_templates(target_templates: List[int], count_per_template: int = 2) -> Dict[int, List[int]]:
    """Find seeds that produce each target template."""
    result = {t: [] for t in target_templates}
    seed = 0
    while any(len(v) < count_per_template for v in result.values()):
        if seed > 500:
            break
        try:
            sc = generate_scenario(seed)
            if sc.template_id in result and len(result[sc.template_id]) < count_per_template:
                result[sc.template_id].append(seed)
        except Exception:
            pass
        seed += 1
    return result


def main():
    parser = argparse.ArgumentParser(description="Strategic adversarial policy game testing")
    parser.add_argument("--base-url", default="http://localhost:9000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--seeds", type=int, nargs="*", help="Specific seeds to test")
    parser.add_argument("--template", type=int, help="Test specific template")
    parser.add_argument("--count", type=int, default=1, help="Seeds per template")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    agent = AgentLLMClient(args.base_url, args.model)
    user = UserLLMClient(args.base_url, args.model, max_tokens=256, temperature=0.7)

    if args.seeds:
        seeds = args.seeds
    elif args.template:
        mapping = find_seeds_for_templates([args.template], args.count)
        seeds = mapping.get(args.template, [])
        if not seeds:
            print(f"No seeds found for template {args.template}")
            return
    else:
        # Strategic selection: cover regression-relevant templates
        target_templates = [1, 2, 5, 6, 7, 12]  # Key regression-mapped templates
        mapping = find_seeds_for_templates(target_templates, args.count)
        seeds = []
        for tid in sorted(mapping):
            seeds.extend(mapping[tid])
            if not args.quiet:
                print(f"Template {tid:2d}: seeds {mapping[tid]}")

    print(f"\nRunning {len(seeds)} episodes...\n")

    results = []
    for seed in seeds:
        r = run_episode(seed, agent, user, verbose=not args.quiet)
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Seed':>5} {'Template':>10} {'Domain':>8} {'Reward':>7} {'Steps':>6} {'Reason':<40}")
    print(f"{'-'*5:>5} {'-'*10:>10} {'-'*8:>8} {'-'*7:>7} {'-'*6:>6} {'-'*40:<40}")

    total_reward = 0
    for r in results:
        if "error" in r:
            print(f"{r['seed']:>5} {'ERROR':>10} {'':>8} {'':>7} {'':>6} {r['error']:<40}")
        else:
            reward = r["reward"]
            total_reward += reward
            reason_short = r["reason"][:40]
            print(f"{r['seed']:>5} {r.get('template_id','?'):>10} {r.get('domain','?'):>8} {reward:>7.2f} {r.get('steps',0):>6} {reason_short:<40}")

    n = len([r for r in results if "error" not in r])
    if n > 0:
        print(f"\nAverage reward: {total_reward/n:.3f} ({n} episodes)")


if __name__ == "__main__":
    main()
