#!/usr/bin/env python3
"""
Contrastive Skill Selection (unified across domains)

Analyzes successful vs failed trajectories from retail AND airline to identify
domain-agnostic skills the model consistently fails at. Runs the selection
N times with bootstrap resampling for robustness.

Usage:
  python contrastive_skill_selection.py \
    --retail data/old_simulations/qwen-3-30b-retail-qwen3-30b.json \
    --airline data/old_simulations/qwen-3-30b-airline-qwen3-30b.json \
    --attempts 5 --threshold 8 --top_k 5 \
    --output contrastive_skill_results.json
"""

import argparse
import json
import random
from collections import defaultdict


# ──────────────────────── universal skill categorizer ──────────────────────

def categorize_universal_skill(action_names: list[str]) -> str:
    """Domain-agnostic skill categories that generalize across retail & airline."""
    actions = set(action_names)

    read_actions = {
        "find_user_id_by_name_zip", "find_user_id_by_email",
        "get_order_details", "get_user_details", "get_product_details",
        "get_reservation_details", "search_direct_flight", "calculate",
    }
    transfer_actions = {"transfer_to_human_agents"}

    mutating = actions - read_actions - transfer_actions
    has_calculate = "calculate" in actions
    has_search = bool(actions & {"get_product_details", "search_direct_flight"})

    if actions & transfer_actions:
        return "escalation_to_human"

    n_mutating = len(mutating)

    if n_mutating == 0:
        if has_search:
            return "information_lookup_with_search"
        return "simple_information_lookup"

    if n_mutating >= 3:
        return "complex_multi_step_operation"

    if n_mutating == 2:
        return "two_step_modification"

    # n_mutating == 1
    mut_action = list(mutating)[0]

    cancel_ops = {"cancel_pending_order", "cancel_reservation"}
    create_ops = {"book_reservation"}
    notify_ops = {"send_certificate"}

    if mut_action in cancel_ops:
        if has_search:
            return "cancellation_with_search"
        return "simple_cancellation"

    if mut_action in create_ops:
        if has_search:
            return "new_booking_with_search"
        return "new_booking"

    if mut_action in notify_ops:
        return "notification_sending"

    # All remaining single-mutation ops are "updates"
    if has_search and has_calculate:
        return "update_with_search_and_calculation"
    if has_search:
        return "update_with_search"
    if has_calculate:
        return "update_with_calculation"
    return "simple_update"


# ──────────────────────────── core logic ──────────────────────────────────

def build_records(data: dict, domain: str) -> list[dict]:
    task_map = {str(t["id"]): t for t in data["tasks"]}
    records = []
    for sim in data["simulations"]:
        task = task_map[str(sim["task_id"])]
        actions = [a["name"] for a in task["evaluation_criteria"].get("actions", [])]
        skill = categorize_universal_skill(actions)
        records.append({
            "task_id": f"{domain[0].upper()}{sim['task_id']}",
            "domain": domain,
            "skill": skill,
            "reward": sim["reward_info"]["reward"],
        })
    return records


def select_failing_skills(
    records: list[dict],
    threshold_pct: float,
    top_k: int,
    bootstrap: bool = False,
    seed: int | None = None,
) -> list[dict]:
    rng = random.Random(seed)

    if bootstrap:
        records = [rng.choice(records) for _ in range(len(records))]

    total = len(records)
    skill_stats = defaultdict(lambda: {
        "passed": [], "failed": [],
        "retail_failed": [], "retail_passed": [],
        "airline_failed": [], "airline_passed": [],
    })

    for rec in records:
        key = "passed" if rec["reward"] == 1.0 else "failed"
        skill_stats[rec["skill"]][key].append(rec["task_id"])
        skill_stats[rec["skill"]][f"{rec['domain']}_{key}"].append(rec["task_id"])

    results = []
    for skill, stats in skill_stats.items():
        n_fail = len(stats["failed"])
        n_pass = len(stats["passed"])
        n_total = n_fail + n_pass
        fail_rate = n_fail / n_total if n_total > 0 else 0
        pct_of_total = n_fail / total * 100

        if pct_of_total < threshold_pct:
            continue

        # Which domains this skill appears in
        domains = []
        if stats["retail_failed"] or stats["retail_passed"]:
            domains.append("retail")
        if stats["airline_failed"] or stats["airline_passed"]:
            domains.append("airline")

        results.append({
            "skill": skill,
            "success_rate": round(1.0 - fail_rate, 4),
            "fail_rate": round(fail_rate, 4),
            "failed_cases": sorted(set(stats["failed"])),
            "passed_cases": sorted(set(stats["passed"])),
            "fail_count": n_fail,
            "pass_count": n_pass,
            "pct_of_total": round(pct_of_total, 2),
            "domains": domains,
            "retail_fail": len(stats["retail_failed"]),
            "retail_pass": len(stats["retail_passed"]),
            "airline_fail": len(stats["airline_failed"]),
            "airline_pass": len(stats["airline_passed"]),
        })

    results.sort(key=lambda x: x["fail_count"], reverse=True)
    return results[:top_k]


def run_attempts(
    records: list[dict],
    n_attempts: int,
    threshold_pct: float,
    top_k: int,
    base_seed: int = 42,
) -> list[dict]:
    attempts = []
    for i in range(n_attempts):
        bootstrap = (i > 0)
        seed = base_seed + i

        selected = select_failing_skills(
            records,
            threshold_pct=threshold_pct,
            top_k=top_k,
            bootstrap=bootstrap,
            seed=seed,
        )

        result = {}
        for s in selected:
            result[s["skill"]] = {
                "assigned_model": "base",
                "success_rate": s["success_rate"],
                "failed_cases": s["failed_cases"],
                "passed_cases": s["passed_cases"],
                "fail_count": s["fail_count"],
                "pass_count": s["pass_count"],
                "pct_of_total": s["pct_of_total"],
                "total_cases": s["fail_count"] + s["pass_count"],
                "domains": s["domains"],
                "retail_fail": s["retail_fail"],
                "retail_pass": s["retail_pass"],
                "airline_fail": s["airline_fail"],
                "airline_pass": s["airline_pass"],
            }

        attempts.append({
            "attempt": i + 1,
            "result": result,
        })
    return attempts


def main():
    parser = argparse.ArgumentParser(description="Contrastive skill selection (unified)")
    parser.add_argument("--retail", type=str, default=None)
    parser.add_argument("--airline", type=str, default=None)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=8.0,
                        help="Min %% of total tasks a skill must fail on")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", type=str, default="contrastive_skill_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    for path, domain in [(args.retail, "retail"), (args.airline, "airline")]:
        if path is None:
            continue
        with open(path) as f:
            data = json.load(f)
        domain_records = build_records(data, domain)
        records.extend(domain_records)
        n_pass = sum(1 for r in domain_records if r["reward"] == 1.0)
        n_fail = len(domain_records) - n_pass
        print(f"  {domain}: {len(domain_records)} tasks, {n_pass} pass, {n_fail} fail")

    print(f"  Combined: {len(records)} tasks")
    print()

    attempts = run_attempts(
        records,
        n_attempts=args.attempts,
        threshold_pct=args.threshold,
        top_k=args.top_k,
        base_seed=args.seed,
    )

    for att in attempts:
        print(f"Attempt {att['attempt']}:")
        if not att["result"]:
            print("  (no skills met threshold)")
        for skill, info in att["result"].items():
            domains_str = "|".join(info["domains"])
            print(f"  {skill:<40} fail={info['fail_count']:>3} pass={info['pass_count']:>3} "
                  f"rate={info['success_rate']:.2f} impact={info['pct_of_total']:.1f}% [{domains_str}]")
        print()

    with open(args.output, "w") as f:
        json.dump(attempts, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
