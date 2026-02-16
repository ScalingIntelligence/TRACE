"""Benchmark a model on the tau tool-calling microenvironment.

Runs the model through N seeds using OpenAI function-calling API,
reports per-scenario-type accuracy, reward distribution, domain
breakdown, and failure analysis matching tau2-bench patterns.

Usage:
    python -m tau_tool_calling_env.benchmark \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --num-seeds 100 \
        --rollouts rollouts.jsonl \
        --output summary.json

    # Retail-only:
    python -m tau_tool_calling_env.benchmark --domain retail

    # Airline-only:
    python -m tau_tool_calling_env.benchmark --domain airline

Output files:
    --rollouts FILE   JSONL with full conversations (one JSON per line).
                      Each line has: seed, scenario_type, domain, reward,
                      reason, conversation, tool_calls, expected_actions, etc.
    --output FILE     JSON summary with config, stats, and per-episode results.
"""

import sys
import json
import time
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

import requests

from .game import TauToolCallingEnv
from .scenarios import generate_scenario
from .verification import (
    compute_gold_db_hash,
    compute_predicted_db_hash,
    check_communicate_info,
)
from adversarial_policy_game.llm_user import UserLLMClient


# ---------------------------------------------------------------------------
# vLLM client with function calling
# ---------------------------------------------------------------------------
class VLLMClient:
    def __init__(self, base_url: str, model: str, max_tokens: int = 512):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
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
    game: TauToolCallingEnv,
    client: VLLMClient,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one episode using OpenAI function calling API."""
    try:
        game.reset(seed)
    except Exception as e:
        return {"seed": seed, "error": str(e)}

    sc = game._scenario

    if verbose:
        print(f"\n--- Seed {seed} | {sc.scenario_type} | {sc.domain} | refusal={sc.is_refusal} ---")
        print(f"    Description: {sc.description}")
        if sc.expected_actions:
            for ea in sc.expected_actions:
                print(f"    Expected: {ea.name}({json.dumps(ea.arguments, default=str)[:80]})")

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
            # Finalize with verification so reward is properly computed
            # instead of staying at the default 0.0
            if not game.done:
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

    # If the loop exited due to step limit without game finalizing, finalize now
    if not game.done:
        game._finalize_with_verification()

    summary = game.get_summary()

    # Compute detailed DB/COMM breakdown for analysis
    db_pass = None
    comm_pass = None
    if sc.expected_actions and not game._transferred:
        try:
            gold_hash = compute_gold_db_hash(sc.domain, sc.expected_actions)
            predicted_hash = compute_predicted_db_hash(game._tools)
            db_pass = gold_hash == predicted_hash
        except Exception:
            db_pass = False
        comm_pass = check_communicate_info(sc.communicate_info, game._conversation)

    # Classify failure mode
    failure_mode = _classify_failure(
        summary, sc, game._tools, db_pass, comm_pass,
    )

    result = {
        "seed": seed,
        "scenario_type": summary["scenario_type"],
        "domain": summary["domain"],
        "description": summary["description"],
        "is_refusal": summary["is_refusal"],
        "reward": summary["reward"],
        "reason": summary["reason"],
        "steps": summary["steps"],
        "transferred": summary["transferred"],
        "tool_calls": summary["tool_calls"],
        "expected_actions": summary["expected_actions"],
        "communicate_info": summary["communicate_info"],
        "key_facts": summary["key_facts"],
        "conversation": list(game._conversation),
        "system_prompt": game.get_system_prompt(),
        "db_pass": db_pass,
        "comm_pass": comm_pass,
        "failure_mode": failure_mode,
        "conversation_length": summary["conversation_length"],
    }

    if verbose and summary["reward"] < 1.0:
        print(f"    >>> FAILED: {summary['reason']}")
        if failure_mode:
            print(f"    >>> Failure mode: {failure_mode}")

    return result


# ---------------------------------------------------------------------------
# Failure classification (mirrors tau2-bench failure taxonomy)
# ---------------------------------------------------------------------------

# Tools that modify DB state
_WRITE_TOOLS = {
    "cancel_reservation", "update_reservation_flights",
    "update_reservation_passengers", "update_reservation_baggages",
    "book_reservation", "send_certificate",
    "cancel_pending_order", "modify_pending_order_items",
    "modify_pending_order_address", "modify_pending_order_payment",
    "return_delivered_order_items", "exchange_delivered_order_items",
    "modify_user_address", "modify_user_email",
}

# Tools that only read
_READ_TOOLS = {
    "get_user_details", "get_reservation_details", "get_order_details",
    "get_product_details", "search_direct_flight", "search_onestop_flight",
    "find_user_id_by_email", "find_user_id_by_name_zip",
    "list_all_airports", "calculate", "think", "transfer_to_human_agents",
}


def _classify_failure(
    summary: Dict,
    scenario,
    tool_executor,
    db_pass: Optional[bool],
    comm_pass: Optional[bool],
) -> str:
    """Classify failure into tau-bench-style categories."""
    reward = summary["reward"]
    if reward >= 1.0:
        return ""  # Success

    reason = summary["reason"]
    tool_calls = summary.get("tool_calls", [])
    expected = summary.get("expected_actions", [])
    tc_names = [tc.get("name", "") for tc in tool_calls]

    # Category: Loop/timeout
    if "Loop detected" in reason:
        return "loop_detected"
    if summary["steps"] >= 28:  # Near max_steps
        return "timeout_max_steps"

    # Category: Invalid action
    if "Invalid action" in reason or "No action" in reason:
        return "invalid_action"

    # Category: Unnecessary transfer
    if summary["transferred"] and expected:
        return "unnecessary_transfer"

    # Category: Refusal task violations
    if scenario.is_refusal:
        writes = [n for n in tc_names if n in _WRITE_TOOLS]
        if writes:
            return f"refusal_violation:{writes[0]}"
        if reward == 0.0:
            return "refusal_no_action"
        return "refusal_partial"

    # For action tasks
    if not expected:
        if comm_pass is False:
            return "info_missing_communication"
        return "info_other"

    # DB pass/fail analysis
    if db_pass is False and comm_pass is True:
        # Most common: right communication, wrong DB action
        # Sub-classify the DB failure
        return _classify_db_failure(tool_calls, expected, tc_names)
    elif db_pass is False and comm_pass is False:
        return "both_fail:" + _classify_db_failure(tool_calls, expected, tc_names)
    elif db_pass is True and comm_pass is False:
        return "db_pass_comm_fail"
    elif db_pass is None:
        return "transferred_on_action_task"

    return "unknown"


def _classify_db_failure(
    tool_calls: List[Dict],
    expected: List[Dict],
    tc_names: List[str],
) -> str:
    """Sub-classify DB failure into mechanical vs reasoning categories."""
    if not expected:
        return "no_expected_actions"

    exp_name = expected[0].get("name", "")
    exp_args = expected[0].get("arguments", {})

    # Check if the correct tool was even called
    writes_made = [tc for tc in tool_calls if tc.get("name") in _WRITE_TOOLS]

    if not writes_made:
        # No write tool called at all
        reads = [tc for tc in tool_calls if tc.get("name") in _READ_TOOLS]
        if not reads:
            return "no_tool_calls"
        return "missing_write_action"

    # Check if correct tool name was used
    write_names = [tc.get("name") for tc in writes_made]
    if exp_name not in write_names:
        return f"wrong_tool:{write_names[0]}_instead_of_{exp_name}"

    # Correct tool called — arguments must be wrong
    matching_calls = [tc for tc in writes_made if tc.get("name") == exp_name]
    if matching_calls:
        actual_args = matching_calls[0].get("arguments", {})
        # Find which arguments differ
        diffs = []
        for k, v in exp_args.items():
            actual_v = actual_args.get(k)
            if actual_v != v:
                diffs.append(k)
        if diffs:
            return f"wrong_arguments:{','.join(diffs[:3])}"

    return "wrong_arguments:unknown"


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
    perfect = sum(1 for r in rewards if r >= 1.0)
    partial = sum(1 for r in rewards if 0 < r < 1.0)
    zero = sum(1 for r in rewards if r == 0.0)
    negative = sum(1 for r in rewards if r < 0)

    n = len(valid)
    print(f"\n{'=' * 78}")
    print(f"  TAU TOOL-CALLING BENCHMARK — {n} episodes in {elapsed:.1f}s")
    print(f"{'=' * 78}")

    print(f"\n  Overall: avg_reward = {avg_reward:+.3f}")
    print(f"  ┌─────────────────────┬───────┬─────────┐")
    print(f"  │ Category            │ Count │   Rate  │")
    print(f"  ├─────────────────────┼───────┼─────────┤")
    print(f"  │ Perfect  (1.0)      │ {perfect:5d} │ {perfect/n*100:5.1f}%  │")
    print(f"  │ Partial  (0,1)      │ {partial:5d} │ {partial/n*100:5.1f}%  │")
    print(f"  │ Zero     (0.0)      │ {zero:5d} │ {zero/n*100:5.1f}%  │")
    print(f"  │ Negative (<0)       │ {negative:5d} │ {negative/n*100:5.1f}%  │")
    print(f"  └─────────────────────┴───────┴─────────┘")

    if errors:
        print(f"\n  Errors: {len(errors)} seeds failed to run")
        for e in errors[:5]:
            print(f"    Seed {e['seed']}: {e['error'][:80]}")

    # ----- By domain -----
    print(f"\n  By Domain:")
    domain_results = defaultdict(list)
    for r in valid:
        domain_results[r["domain"]].append(r)

    print(f"  ┌───────────┬─────┬──────────┬─────────┐")
    print(f"  │ Domain    │  N  │ Avg Rwd  │ Perfect │")
    print(f"  ├───────────┼─────┼──────────┼─────────┤")
    for domain in sorted(domain_results.keys()):
        rs = domain_results[domain]
        rws = [r["reward"] for r in rs]
        avg = sum(rws) / len(rws)
        perf = sum(1 for r in rws if r >= 1.0)
        print(f"  │ {domain:<9s} │ {len(rs):3d} │ {avg:+7.3f} │ {perf:3d}/{len(rs):<3d} │")
    print(f"  └───────────┴─────┴──────────┴─────────┘")

    # ----- By scenario type -----
    print(f"\n  By Scenario Type:")
    type_results = defaultdict(list)
    for r in valid:
        type_results[r["scenario_type"]].append(r)

    print(f"  ┌────────────────┬─────┬──────────┬─────────┬──────────┐")
    print(f"  │ Type           │  N  │ Avg Rwd  │ Perfect │ Negative │")
    print(f"  ├────────────────┼─────┼──────────┼─────────┼──────────┤")
    for stype in ["explicit", "selection", "policy_gated"]:
        if stype not in type_results:
            continue
        rs = type_results[stype]
        rws = [r["reward"] for r in rs]
        avg = sum(rws) / len(rws)
        perf = sum(1 for r in rws if r >= 1.0)
        neg = sum(1 for r in rws if r < 0)
        print(f"  │ {stype:<14s} │ {len(rs):3d} │ {avg:+7.3f} │ {perf:3d}/{len(rs):<3d} │ {neg:3d}/{len(rs):<3d}  │")
    print(f"  └────────────────┴─────┴──────────┴─────────┴──────────┘")

    # ----- By domain × scenario type -----
    print(f"\n  By Domain × Scenario Type:")
    for domain in sorted(domain_results.keys()):
        domain_by_type = defaultdict(list)
        for r in domain_results[domain]:
            domain_by_type[r["scenario_type"]].append(r)
        for stype in ["explicit", "selection", "policy_gated"]:
            if stype not in domain_by_type:
                continue
            rs = domain_by_type[stype]
            rws = [r["reward"] for r in rs]
            avg = sum(rws) / len(rws)
            perf = sum(1 for r in rws if r >= 1.0)
            print(f"    {domain:8s} / {stype:14s}  n={len(rs):3d}  avg={avg:+.3f}  perfect={perf}/{len(rs)}")

    # ----- DB/COMM breakdown (tau bench style) -----
    action_results = [r for r in valid if r.get("expected_actions") and not r["is_refusal"]]
    if action_results:
        db_pass_comm_pass = sum(1 for r in action_results if r.get("db_pass") and r.get("comm_pass"))
        db_pass_comm_fail = sum(1 for r in action_results if r.get("db_pass") and not r.get("comm_pass"))
        db_fail_comm_pass = sum(1 for r in action_results if not r.get("db_pass") and r.get("comm_pass"))
        db_fail_comm_fail = sum(1 for r in action_results if not r.get("db_pass") and not r.get("comm_pass"))
        transferred = sum(1 for r in action_results if r["transferred"])

        na = len(action_results)
        print(f"\n  Reward Decomposition (action tasks, n={na}):")
        print(f"  ┌─────────────────────────┬───────┬─────────┐")
        print(f"  │ Category                │ Count │ % Tasks │")
        print(f"  ├─────────────────────────┼───────┼─────────┤")
        print(f"  │ DB_PASS + COMM_PASS     │ {db_pass_comm_pass:5d} │ {db_pass_comm_pass/na*100:5.1f}%  │")
        print(f"  │ DB_PASS + COMM_FAIL     │ {db_pass_comm_fail:5d} │ {db_pass_comm_fail/na*100:5.1f}%  │")
        print(f"  │ DB_FAIL + COMM_PASS     │ {db_fail_comm_pass:5d} │ {db_fail_comm_pass/na*100:5.1f}%  │")
        print(f"  │ DB_FAIL + COMM_FAIL     │ {db_fail_comm_fail:5d} │ {db_fail_comm_fail/na*100:5.1f}%  │")
        print(f"  │ Transferred (no DB chk) │ {transferred:5d} │ {transferred/na*100:5.1f}%  │")
        print(f"  └─────────────────────────┴───────┴─────────┘")

    # ----- Failure mode analysis -----
    failures = [r for r in valid if r["reward"] < 1.0]
    if failures:
        print(f"\n  Failure Mode Analysis ({len(failures)} failures):")
        mode_counts = Counter(r.get("failure_mode", "unknown") for r in failures)
        print(f"  ┌─────────────────────────────────────────────────────┬───────┬─────────┐")
        print(f"  │ Failure Mode                                        │ Count │ % Fails │")
        print(f"  ├─────────────────────────────────────────────────────┼───────┼─────────┤")
        nf = len(failures)
        for mode, count in mode_counts.most_common(20):
            mode_display = mode[:51] if mode else "(unknown)"
            print(f"  │ {mode_display:<51s} │ {count:5d} │ {count/nf*100:5.1f}%  │")
        print(f"  └─────────────────────────────────────────────────────┴───────┴─────────┘")

        # Group into tau-bench categories
        mechanical = sum(c for m, c in mode_counts.items()
                         if "wrong_arguments" in m)
        wrong_tool = sum(c for m, c in mode_counts.items()
                         if "wrong_tool" in m or "missing_write" in m)
        no_action = sum(c for m, c in mode_counts.items()
                         if "no_tool_calls" in m or "no_action" in m)
        system = sum(c for m, c in mode_counts.items()
                      if "loop" in m or "timeout" in m or "invalid" in m)
        refusal_viol = sum(c for m, c in mode_counts.items()
                            if "refusal_violation" in m)
        comm_only = sum(c for m, c in mode_counts.items()
                         if "comm_fail" in m or "communication" in m)
        other = nf - mechanical - wrong_tool - no_action - system - refusal_viol - comm_only

        print(f"\n  Failure Category Summary (tau-bench style):")
        print(f"    Mechanical (wrong args)     : {mechanical:3d} ({mechanical/nf*100:.1f}%)")
        print(f"    Wrong tool / missing write  : {wrong_tool:3d} ({wrong_tool/nf*100:.1f}%)")
        print(f"    No action taken             : {no_action:3d} ({no_action/nf*100:.1f}%)")
        print(f"    System (loop/timeout/invalid): {system:3d} ({system/nf*100:.1f}%)")
        print(f"    Refusal violations          : {refusal_viol:3d} ({refusal_viol/nf*100:.1f}%)")
        print(f"    Communication-only          : {comm_only:3d} ({comm_only/nf*100:.1f}%)")
        print(f"    Other                       : {other:3d} ({other/nf*100:.1f}%)")

    # ----- Sample failures -----
    action_failures = [r for r in valid
                       if r["reward"] < 1.0
                       and r.get("expected_actions")
                       and not r["is_refusal"]]
    if action_failures:
        print(f"\n  Sample Action Failures (up to 15):")
        for r in action_failures[:15]:
            tc_names = [tc["name"] for tc in r.get("tool_calls", [])]
            exp_names = [ea["name"] for ea in r.get("expected_actions", [])]
            print(f"    Seed {r['seed']:4d} | {r['domain']:8s} | {r['scenario_type']:14s} | "
                  f"reward={r['reward']:+5.2f} | db={r.get('db_pass','?')} comm={r.get('comm_pass','?')}")
            print(f"      Description: {r['description'][:70]}")
            print(f"      Expected: {exp_names}")
            print(f"      Actual:   {tc_names}")
            print(f"      Mode: {r.get('failure_mode', '?')}")
            print()

    # ----- Refusal results -----
    refusal_results = [r for r in valid if r["is_refusal"]]
    if refusal_results:
        rr = refusal_results
        perf = sum(1 for r in rr if r["reward"] >= 1.0)
        viol = sum(1 for r in rr if r["reward"] < 0)
        print(f"\n  Refusal Scenarios (n={len(rr)}):")
        print(f"    Correct refusal: {perf}/{len(rr)} ({perf/len(rr)*100:.1f}%)")
        print(f"    Violations:      {viol}/{len(rr)} ({viol/len(rr)*100:.1f}%)")

    # ----- Average steps -----
    success_steps = [r["steps"] for r in valid if r["reward"] >= 1.0]
    fail_steps = [r["steps"] for r in valid if r["reward"] < 1.0]
    if success_steps:
        print(f"\n  Average Steps:")
        print(f"    Success: {sum(success_steps)/len(success_steps):.1f} (range {min(success_steps)}-{max(success_steps)})")
    if fail_steps:
        print(f"    Failure: {sum(fail_steps)/len(fail_steps):.1f} (range {min(fail_steps)}-{max(fail_steps)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a model on the tau tool-calling microenvironment"
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="Model name")
    parser.add_argument("--num-seeds", type=int, default=100,
                        help="Number of seeds to evaluate")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per agent generation")
    parser.add_argument("--domain", type=str, default=None,
                        choices=["airline", "retail"],
                        help="Domain filter (default: both)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-episode details")
    parser.add_argument("--output", "-o", default=None,
                        help="Save summary to JSON file")
    parser.add_argument("--user-base-url", default=None,
                        help="User LLM server URL (default: same as --base-url)")
    parser.add_argument("--user-model", default=None,
                        help="User LLM model name (default: same as --model)")
    parser.add_argument("--user-temperature", type=float, default=0.7,
                        help="Temperature for user LLM")
    parser.add_argument("--rollouts", default=None,
                        help="Save full conversations to JSONL file")
    parser.add_argument("--workers", "-w", type=int, default=100,
                        help="Number of parallel workers (default: 100)")
    args = parser.parse_args()

    client = VLLMClient(args.base_url, args.model, args.max_tokens)

    user_base_url = args.user_base_url or args.base_url
    user_model = args.user_model or args.model
    user_client = UserLLMClient(
        user_base_url, user_model,
        max_tokens=256, temperature=args.user_temperature,
    )

    # Connectivity test
    print(f"Agent model: {args.model} @ {args.base_url}")
    print(f"User  model: {user_model} @ {user_base_url} (temp={args.user_temperature})")
    print(f"Domain:      {args.domain or 'both'}")
    print(f"Seeds:       {args.start_seed} to {args.start_seed + args.num_seeds - 1}")

    try:
        test_msgs = [{"role": "user", "content": "Say OK."}]
        client.generate_with_tools("You are helpful.", test_msgs, [])
        print("Connection:  OK\n")
    except Exception as e:
        print(f"Connection FAILED: {e}")
        sys.exit(1)

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    results = [None] * len(seeds)
    t0 = time.time()
    completed_count = 0

    def _run_seed(seed):
        """Worker function: create per-thread game instance and run episode."""
        local_game = TauToolCallingEnv(
            max_steps=30,
            user_client=user_client,
            domain=args.domain,
        )
        return run_episode(local_game, client, seed, verbose=args.verbose)

    num_workers = max(1, args.workers)
    if num_workers == 1:
        # Sequential path (simpler output)
        rollout_file = open(args.rollouts, "w") if args.rollouts else None
        for i, seed in enumerate(seeds):
            result = _run_seed(seed)
            results[i] = result
            completed_count += 1

            if rollout_file:
                rollout_file.write(json.dumps(result, default=str) + "\n")
                rollout_file.flush()

            elapsed = time.time() - t0
            rate = completed_count / elapsed if elapsed > 0 else 0
            eta = (len(seeds) - completed_count) / rate if rate > 0 else 0
            r = result.get("reward", 0)
            domain_name = result.get("domain", "?")
            stype = result.get("scenario_type", result.get("error", "err"))[:14]
            err = " ERROR" if "error" in result else ""
            db = result.get("db_pass")
            cm = result.get("comm_pass")
            db_str = "T" if db else ("F" if db is False else "-")
            cm_str = "T" if cm else ("F" if cm is False else "-")

            print(f"  [{completed_count:3d}/{len(seeds)}] seed={seed:4d} {domain_name:8s} {stype:14s} "
                  f"reward={r:+5.2f} db={db_str} comm={cm_str} "
                  f"steps={result.get('steps', 0):2d} ({elapsed:.0f}s, ETA {eta:.0f}s){err}")
        if rollout_file:
            rollout_file.close()
    else:
        # Parallel path
        print(f"Workers:     {num_workers}\n")
        import threading
        print_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_run_seed, seed): i
                for i, seed in enumerate(seeds)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                seed = seeds[idx]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"seed": seed, "error": str(e)}
                results[idx] = result
                completed_count += 1

                with print_lock:
                    elapsed = time.time() - t0
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (len(seeds) - completed_count) / rate if rate > 0 else 0
                    r = result.get("reward", 0)
                    domain_name = result.get("domain", "?")
                    stype = result.get("scenario_type", result.get("error", "err"))[:14]
                    err = " ERROR" if "error" in result else ""
                    db = result.get("db_pass")
                    cm = result.get("comm_pass")
                    db_str = "T" if db else ("F" if db is False else "-")
                    cm_str = "T" if cm else ("F" if cm is False else "-")

                    print(f"  [{completed_count:3d}/{len(seeds)}] seed={seed:4d} {domain_name:8s} {stype:14s} "
                          f"reward={r:+5.2f} db={db_str} comm={cm_str} "
                          f"steps={result.get('steps', 0):2d} ({elapsed:.0f}s, ETA {eta:.0f}s){err}")

        # Write rollouts in seed order
        if args.rollouts:
            with open(args.rollouts, "w") as f:
                for result in results:
                    if result is not None:
                        f.write(json.dumps(result, default=str) + "\n")

    if args.rollouts:
        print(f"\nRollouts saved to {args.rollouts}")

    elapsed = time.time() - t0
    print_report(results, elapsed)

    if args.output:
        valid = [r for r in results if "error" not in r]
        rewards = [r["reward"] for r in valid]
        # Strip conversations and system_prompt from summary
        summary_results = [
            {k: v for k, v in r.items()
             if k not in ("conversation", "system_prompt")}
            for r in results
        ]
        with open(args.output, "w") as f:
            json.dump({
                "config": {
                    "model": args.model,
                    "user_model": user_model,
                    "num_seeds": args.num_seeds,
                    "start_seed": args.start_seed,
                    "domain": args.domain,
                    "elapsed_seconds": elapsed,
                },
                "summary": {
                    "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
                    "perfect_rate": sum(1 for r in rewards if r >= 1.0) / len(rewards) * 100 if rewards else 0,
                    "zero_rate": sum(1 for r in rewards if r == 0.0) / len(rewards) * 100 if rewards else 0,
                    "negative_rate": sum(1 for r in rewards if r < 0) / len(rewards) * 100 if rewards else 0,
                    "total_episodes": len(valid),
                    "errors": len(results) - len(valid),
                },
                "results": summary_results,
            }, f, indent=2, default=str)
        print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    main()
