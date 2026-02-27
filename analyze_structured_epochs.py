#!/usr/bin/env python3
"""
Detailed task-level comparison across structured training epochs on tau2-bench.
Compares base model vs structured_Xepoch_run1 for both airline and retail domains.
"""

import json
import os
import sys
from collections import defaultdict

SIM_DIR = "/root/games/evals/benchmarks/tau2_bench_eval/data/simulations"
TASK_DIR = "/root/games/tau2-bench/data/tau2/domains"

# ── File definitions ──────────────────────────────────────────────────────────

AIRLINE_FILES = {
    "base":  "qwen-3-30b-airline-qwen3-30b.json",
    "e10":   "qwen-3-30b-structured_10epoch_run1-airline-qwen3-30b.json",
    "e20":   "qwen-3-30b-structured_20epoch_run1-airline-qwen3-30b.json",
    "e25":   "qwen-3-30b-structured_25epoch_run1-airline-qwen3-30b.json",
    "e30":   "qwen-3-30b-structured_30epoch_run1-airline-qwen3-30b.json",
    "e35":   "qwen-3-30b-structured_35epoch_run1-airline-qwen3-30b.json",
}

RETAIL_FILES = {
    "base":  "qwen-3-30b-retail-qwen3-30b.json",
    "e05":   "qwen-3-30b-structured_5epoch_run1-retail-qwen3-30b.json",
    "e10":   "qwen-3-30b-structured_10epoch_run1-retail-qwen3-30b.json",
    "e20":   "qwen-3-30b-structured_20epoch_run1-retail-qwen3-30b.json",
    "e25":   "qwen-3-30b-structured_25epoch_run1-retail-qwen3-30b.json",
    "e30":   "qwen-3-30b-structured_30epoch_run1-retail-qwen3-30b.json",
    "e35":   "qwen-3-30b-structured_35epoch_run1-retail-qwen3-30b.json",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_rewards(filepath):
    """Load simulation file and return {task_id: reward}."""
    with open(filepath) as f:
        data = json.load(f)
    rewards = {}
    for sim in data["simulations"]:
        tid = sim["task_id"]
        reward = sim["reward_info"]["reward"]
        rewards[tid] = reward
    return rewards


def load_tasks(domain):
    """Load task definitions and return {task_id: task_dict}."""
    path = os.path.join(TASK_DIR, domain, "tasks.json")
    with open(path) as f:
        tasks = json.load(f)
    # Key by both int and string for flexible lookup
    result = {}
    for t in tasks:
        result[int(t["id"])] = t
        result[str(t["id"])] = t
    return result


def binarize(reward):
    """Convert reward to 1 (pass) or 0 (fail)."""
    return 1 if reward >= 0.5 else 0


def extract_task_summary(task):
    """Extract a short summary of what the task tests."""
    desc = task.get("description", {})
    purpose = desc.get("purpose") or ""
    notes = desc.get("notes") or ""
    user_sc = task.get("user_scenario", {}).get("instructions", {})
    reason = user_sc.get("reason_for_call", "") or ""
    task_instr = user_sc.get("task_instructions", "") or ""

    # Combine for summary
    summary_parts = []
    if purpose:
        summary_parts.append(f"Purpose: {purpose[:120]}")
    if reason:
        summary_parts.append(f"Reason: {reason[:120]}")
    if notes:
        summary_parts.append(f"Notes: {notes[:80]}")
    return " | ".join(summary_parts) if summary_parts else "(no description)"


def extract_action_types(task):
    """Extract the tool/action names required by the task."""
    eval_criteria = task.get("evaluation_criteria", {})
    actions = eval_criteria.get("actions", [])
    action_names = [a.get("name", "unknown") for a in actions]
    return action_names


def classify_task_skill(task):
    """Classify what skill category a task tests based on actions and description."""
    actions = extract_action_types(task)
    action_set = set(actions)
    desc = task.get("description", {})
    purpose = (desc.get("purpose") or "").lower()
    notes = (desc.get("notes") or "").lower()
    reason = (task.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", "") or "").lower()

    categories = []

    # Data reasoning indicators
    data_reasoning_actions = {
        "search_direct_flight", "search_onestop_flight",
        "get_flight_details", "list_all_airports",
        "get_product_details", "list_all_product_types",
    }
    if action_set & data_reasoning_actions:
        categories.append("data_reasoning")

    # Price / computation indicators
    if any(kw in purpose + notes + reason for kw in ["price", "cost", "cheaper", "cheapest", "total", "compute", "calculate", "refund amount"]):
        categories.append("price_computation")

    # Modification actions
    modify_actions = {
        "update_reservation_flights", "update_reservation_passengers",
        "update_reservation_baggages", "modify_pending_order_items",
        "modify_pending_order_address", "modify_pending_order_payment",
    }
    if action_set & modify_actions:
        categories.append("modification")

    # Cancellation
    cancel_actions = {"cancel_reservation", "cancel_pending_order"}
    if action_set & cancel_actions:
        categories.append("cancellation")

    # Exchange
    exchange_actions = {"exchange_delivered_order_items"}
    if action_set & exchange_actions:
        categories.append("exchange")

    # Return
    return_actions = {"return_delivered_order_items"}
    if action_set & return_actions:
        categories.append("return")

    # Policy / refusal
    nl_assertions = task.get("evaluation_criteria", {}).get("nl_assertions", []) or []
    for a in nl_assertions:
        if a and ("refuse" in a.lower() or "not" in a.lower() or "deny" in a.lower()):
            categories.append("policy_compliance")
            break

    # Transfer
    transfer_actions = {"transfer_to_human_agents"}
    if action_set & transfer_actions:
        categories.append("transfer")

    # Booking
    book_actions = {"book_reservation"}
    if action_set & book_actions:
        categories.append("booking")

    if not categories:
        categories.append("other")

    return categories


# ── Analysis per domain ───────────────────────────────────────────────────────

def analyze_domain(domain, file_map):
    print(f"\n{'='*120}")
    print(f"  DOMAIN: {domain.upper()}")
    print(f"{'='*120}")

    tasks_meta = load_tasks(domain)

    # Load all rewards
    epoch_rewards = {}
    for epoch_label, filename in sorted(file_map.items()):
        path = os.path.join(SIM_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {filename} not found, skipping")
            continue
        epoch_rewards[epoch_label] = load_rewards(path)

    epoch_labels = sorted(epoch_rewards.keys(), key=lambda x: (x != "base", x))

    # Get all task IDs (union across epochs) - convert to int for sorting
    all_task_ids = sorted(set().union(*[set(r.keys()) for r in epoch_rewards.values()]), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
    n_tasks = len(all_task_ids)

    # ── Full Matrix ───────────────────────────────────────────────────────
    print(f"\n  Total tasks: {n_tasks}")
    print(f"  Epochs: {epoch_labels}")

    # Print summary scores first
    print(f"\n  ── AGGREGATE SCORES ──")
    for el in epoch_labels:
        rewards = epoch_rewards[el]
        n = len(rewards)
        passed = sum(1 for r in rewards.values() if r >= 0.5)
        avg = sum(rewards.values()) / n if n else 0
        print(f"    {el:>6s}: {passed}/{n} passed ({100*passed/n:.1f}%), avg reward = {avg:.3f}")

    # Build binary matrix
    matrix = {}  # task_id -> {epoch_label: 0/1}
    for tid in all_task_ids:
        matrix[tid] = {}
        for el in epoch_labels:
            r = epoch_rewards[el].get(tid)
            matrix[tid][el] = binarize(r) if r is not None else None

    # ── Print Full Matrix ─────────────────────────────────────────────────
    print(f"\n  ── FULL TASK × EPOCH MATRIX (1=pass, 0=fail, .=missing) ──")
    header = f"  {'task':>5s}"
    for el in epoch_labels:
        header += f"  {el:>5s}"
    header += f"  {'sum':>4s}  actions / purpose"
    print(header)
    print(f"  {'-'*len(header)}")

    for tid in all_task_ids:
        row = f"  {str(tid):>5s}"
        pass_count = 0
        total_count = 0
        for el in epoch_labels:
            v = matrix[tid][el]
            if v is None:
                row += f"  {'.' :>5s}"
            else:
                row += f"  {v:>5d}"
                pass_count += v
                total_count += 1
        row += f"  {pass_count:>2d}/{total_count:<2d}"

        # Task metadata
        tmeta = tasks_meta.get(tid, {})
        actions = extract_action_types(tmeta)
        action_str = ", ".join(actions) if actions else "(none)"
        purpose = (tmeta.get("description", {}).get("purpose") or "")[:80]
        if purpose:
            row += f"  [{action_str}] {purpose}"
        else:
            reason = (tmeta.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", "") or "")[:80]
            row += f"  [{action_str}] {reason}"

        print(row)

    # ── Classification ────────────────────────────────────────────────────
    consistently_easy = []
    consistently_hard = []
    improved = []
    regressed = []

    base_label = "base"
    trained_labels = [el for el in epoch_labels if el != "base"]

    for tid in all_task_ids:
        base_val = matrix[tid].get(base_label)
        trained_vals = [matrix[tid].get(el) for el in trained_labels if matrix[tid].get(el) is not None]
        all_vals = [matrix[tid].get(el) for el in epoch_labels if matrix[tid].get(el) is not None]

        if all(v == 1 for v in all_vals):
            consistently_easy.append(tid)
        elif all(v == 0 for v in all_vals):
            consistently_hard.append(tid)

        if base_val == 0 and any(v == 1 for v in trained_vals):
            improved.append(tid)
        if base_val == 1 and any(v == 0 for v in trained_vals):
            regressed.append(tid)

    # ── Consistently Easy ─────────────────────────────────────────────────
    print(f"\n  ── CONSISTENTLY EASY (pass ALL epochs including base): {len(consistently_easy)} tasks ──")
    for tid in consistently_easy:
        tmeta = tasks_meta.get(tid, {})
        cats = classify_task_skill(tmeta)
        print(f"    Task {str(tid):>3s}: {', '.join(cats):30s}  {extract_task_summary(tmeta)[:90]}")

    # ── Consistently Hard ─────────────────────────────────────────────────
    print(f"\n  ── CONSISTENTLY HARD (fail ALL epochs including base): {len(consistently_hard)} tasks ──")
    for tid in consistently_hard:
        tmeta = tasks_meta.get(tid, {})
        cats = classify_task_skill(tmeta)
        actions = extract_action_types(tmeta)
        print(f"    Task {str(tid):>3s}: {', '.join(cats):30s}  {extract_task_summary(tmeta)[:90]}")
        if actions:
            print(f"            actions: {', '.join(actions)}")

    # ── Improved ──────────────────────────────────────────────────────────
    print(f"\n  ── IMPROVED FROM BASE (base=fail, any epoch=pass): {len(improved)} tasks ──")
    for tid in improved:
        tmeta = tasks_meta.get(tid, {})
        cats = classify_task_skill(tmeta)
        actions = extract_action_types(tmeta)
        # Show which epochs passed
        passes = [el for el in trained_labels if matrix[tid].get(el) == 1]
        fails = [el for el in trained_labels if matrix[tid].get(el) == 0]
        print(f"    Task {str(tid):>3s}: {', '.join(cats):30s}  passed@{passes}  failed@{fails}")
        print(f"            {extract_task_summary(tmeta)[:110]}")
        if actions:
            print(f"            actions: {', '.join(actions)}")

    # ── Regressed ─────────────────────────────────────────────────────────
    print(f"\n  ── REGRESSED FROM BASE (base=pass, any epoch=fail): {len(regressed)} tasks ──")
    for tid in regressed:
        tmeta = tasks_meta.get(tid, {})
        cats = classify_task_skill(tmeta)
        actions = extract_action_types(tmeta)
        passes = [el for el in trained_labels if matrix[tid].get(el) == 1]
        fails = [el for el in trained_labels if matrix[tid].get(el) == 0]
        print(f"    Task {str(tid):>3s}: {', '.join(cats):30s}  passed@{passes}  failed@{fails}")
        print(f"            {extract_task_summary(tmeta)[:110]}")
        if actions:
            print(f"            actions: {', '.join(actions)}")

    # ── Skill category breakdown ──────────────────────────────────────────
    print(f"\n  ── SKILL CATEGORY BREAKDOWN ──")
    cat_stats = defaultdict(lambda: {"total": 0, "base_pass": 0, "best_pass": 0, "improved": 0, "regressed": 0})
    for tid in all_task_ids:
        tmeta = tasks_meta.get(tid, {})
        cats = classify_task_skill(tmeta)
        base_val = matrix[tid].get(base_label, 0) or 0
        trained_vals = [matrix[tid].get(el, 0) or 0 for el in trained_labels]
        best_val = max(trained_vals) if trained_vals else 0

        for cat in cats:
            cat_stats[cat]["total"] += 1
            cat_stats[cat]["base_pass"] += base_val
            cat_stats[cat]["best_pass"] += best_val
            if base_val == 0 and best_val == 1:
                cat_stats[cat]["improved"] += 1
            if base_val == 1 and min(trained_vals) == 0:
                cat_stats[cat]["regressed"] += 1

    print(f"    {'category':30s} {'total':>5s} {'base':>6s} {'best':>6s} {'impr':>5s} {'regr':>5s}")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        print(f"    {cat:30s} {s['total']:>5d} {s['base_pass']:>5d}  {s['best_pass']:>5d}  {s['improved']:>5d} {s['regressed']:>5d}")

    # Return data for cross-domain analysis
    return matrix, epoch_labels, tasks_meta, improved, regressed, consistently_easy, consistently_hard


# ── Per-epoch improvement tracking ───────────────────────────────────────────

def epoch_trajectory_analysis(domain, matrix, epoch_labels, tasks_meta, improved_tasks):
    """For improved tasks, show the exact epoch trajectory."""
    print(f"\n  ── EPOCH TRAJECTORY FOR IMPROVED TASKS ({domain.upper()}) ──")
    base_label = "base"
    trained_labels = [el for el in epoch_labels if el != "base"]

    for tid in improved_tasks:
        tmeta = tasks_meta.get(tid, {})
        cats = classify_task_skill(tmeta)
        trajectory = " -> ".join(f"{el}={matrix[tid].get(el, '?')}" for el in epoch_labels)
        print(f"    Task {str(tid):>3s} [{', '.join(cats):25s}]: {trajectory}")


# ── Airline-specific: data reasoning analysis ─────────────────────────────────

def airline_data_reasoning_analysis(matrix, epoch_labels, tasks_meta, improved_tasks):
    print(f"\n  ── AIRLINE: DATA REASONING ANALYSIS FOR IMPROVED TASKS ──")

    data_reasoning_keywords = [
        "search", "flight", "price", "cost", "cheap", "compare",
        "compute", "total", "baggage", "fare", "route", "airport"
    ]

    dr_improved = []
    non_dr_improved = []

    for tid in improved_tasks:
        tmeta = tasks_meta.get(tid, {})
        cats = classify_task_skill(tmeta)
        actions = extract_action_types(tmeta)

        purpose = (tmeta.get("description", {}).get("purpose") or "").lower()
        reason = (tmeta.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", "") or "").lower()
        all_text = purpose + " " + reason + " " + " ".join(actions)

        is_dr = any(kw in all_text for kw in data_reasoning_keywords)

        if is_dr or "data_reasoning" in cats or "price_computation" in cats:
            dr_improved.append(tid)
        else:
            non_dr_improved.append(tid)

    print(f"    Improved tasks involving data reasoning: {len(dr_improved)}")
    for tid in dr_improved:
        tmeta = tasks_meta.get(tid, {})
        actions = extract_action_types(tmeta)
        purpose = (tmeta.get("description", {}).get("purpose") or "")[:100]
        reason = (tmeta.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", "") or "")[:100]
        print(f"      Task {str(tid):>3s}: {purpose or reason}")
        print(f"               actions: {', '.join(actions)}")

    print(f"\n    Improved tasks NOT involving data reasoning: {len(non_dr_improved)}")
    for tid in non_dr_improved:
        tmeta = tasks_meta.get(tid, {})
        actions = extract_action_types(tmeta)
        purpose = (tmeta.get("description", {}).get("purpose") or "")[:100]
        reason = (tmeta.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", "") or "")[:100]
        print(f"      Task {str(tid):>3s}: {purpose or reason}")
        print(f"               actions: {', '.join(actions)}")


# ── Stability analysis ────────────────────────────────────────────────────────

def stability_analysis(domain, matrix, epoch_labels):
    """Categorize tasks by stability across trained epochs."""
    print(f"\n  ── STABILITY ANALYSIS ({domain.upper()}) ──")
    trained_labels = [el for el in epoch_labels if el != "base"]

    stable_pass = 0
    stable_fail = 0
    flaky = 0
    flaky_tasks = []

    for tid in sorted(matrix.keys()):
        vals = [matrix[tid].get(el) for el in trained_labels if matrix[tid].get(el) is not None]
        if not vals:
            continue
        if all(v == 1 for v in vals):
            stable_pass += 1
        elif all(v == 0 for v in vals):
            stable_fail += 1
        else:
            flaky += 1
            flaky_tasks.append(tid)

    total = stable_pass + stable_fail + flaky
    print(f"    Stable pass (all trained epochs pass): {stable_pass}/{total} ({100*stable_pass/total:.1f}%)")
    print(f"    Stable fail (all trained epochs fail): {stable_fail}/{total} ({100*stable_fail/total:.1f}%)")
    print(f"    Flaky (mixed across epochs):           {flaky}/{total} ({100*flaky/total:.1f}%)")

    if flaky_tasks:
        print(f"\n    Flaky task details:")
        for tid in flaky_tasks:
            vals_str = " ".join(f"{el}={matrix[tid].get(el, '?')}" for el in epoch_labels)
            print(f"      Task {str(tid):>3s}: {vals_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Airline analysis
    a_matrix, a_epochs, a_tasks, a_improved, a_regressed, a_easy, a_hard = analyze_domain("airline", AIRLINE_FILES)
    epoch_trajectory_analysis("airline", a_matrix, a_epochs, a_tasks, a_improved)
    airline_data_reasoning_analysis(a_matrix, a_epochs, a_tasks, a_improved)
    stability_analysis("airline", a_matrix, a_epochs)

    # Retail analysis
    r_matrix, r_epochs, r_tasks, r_improved, r_regressed, r_easy, r_hard = analyze_domain("retail", RETAIL_FILES)
    epoch_trajectory_analysis("retail", r_matrix, r_epochs, r_tasks, r_improved)
    stability_analysis("retail", r_matrix, r_epochs)

    # ── Cross-domain summary ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  CROSS-DOMAIN SUMMARY")
    print(f"{'='*120}")
    print(f"\n  Airline:")
    print(f"    Consistently easy:  {len(a_easy)}/50")
    print(f"    Consistently hard:  {len(a_hard)}/50")
    print(f"    Improved from base: {len(a_improved)}/50")
    print(f"    Regressed:          {len(a_regressed)}/50")
    print(f"\n  Retail:")
    print(f"    Consistently easy:  {len(r_easy)}/114")
    print(f"    Consistently hard:  {len(r_hard)}/114")
    print(f"    Improved from base: {len(r_improved)}/114")
    print(f"    Regressed:          {len(r_regressed)}/114")
