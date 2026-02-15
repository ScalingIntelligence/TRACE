"""Comprehensive regression test for adversarial policy game.

Tests every component, every template, every edge case.
Run: python -m adversarial_policy_game.test_regression
"""

import json
import copy
import random
import sys
import traceback

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")
from typing import Dict, List, Any

from .scenarios import (
    generate_scenario, Scenario, GroundTruth, PressureType,
    TEMPLATE_WEIGHTS, TEMPLATE_GENERATORS, select_template,
)
from .database import get_pydantic_db, sample_airline_user, sample_retail_user, sample_airline_multi_reservations
from .tools import ToolExecutor
from .verification import (
    compute_reward, check_communicate_info,
    _check_forbidden, _count_required_matches,
)
from .game import AdversarialPolicyGame, _parse_tool_call, extract_action


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition: bool, name: str, detail: str = ""):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"FAIL: {name}" + (f" — {detail}" if detail else ""))

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"  {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailures:")
            for e in self.errors:
                print(f"  {e}")
        print(f"{'='*70}")
        return self.failed == 0


def test_determinism(t: TestResults):
    """Same seed must produce identical scenarios."""
    print("\n--- Determinism ---")
    for seed in [0, 1, 42, 100, 255]:
        s1 = generate_scenario(seed)
        s2 = generate_scenario(seed)
        t.check(s1.template_id == s2.template_id, f"seed {seed} template_id")
        t.check(s1.domain == s2.domain, f"seed {seed} domain")
        t.check(s1.initial_message == s2.initial_message, f"seed {seed} initial_message")
        t.check(s1.ground_truth.correct_behavior == s2.ground_truth.correct_behavior,
                f"seed {seed} correct_behavior")
        t.check(s1.ground_truth.forbidden_actions == s2.ground_truth.forbidden_actions,
                f"seed {seed} forbidden_actions")
        t.check(s1.ground_truth.required_actions == s2.ground_truth.required_actions,
                f"seed {seed} required_actions")
    print(f"  determinism: {t.passed} checks passed")


def test_template_coverage(t: TestResults):
    """All active templates must be reachable within 500 seeds."""
    print("\n--- Template Coverage ---")
    seen = set()
    errors_by_seed = {}
    for seed in range(500):
        try:
            s = generate_scenario(seed)
            seen.add(s.template_id)
        except Exception as e:
            errors_by_seed[seed] = str(e)

    active_templates = {tid for tid, w in TEMPLATE_WEIGHTS.items() if w > 0}
    for tid in sorted(active_templates):
        t.check(tid in seen, f"template {tid} reachable",
                f"never generated in 500 seeds" if tid not in seen else "")

    t.check(len(errors_by_seed) == 0, "no generation errors",
            f"{len(errors_by_seed)} errors: {list(errors_by_seed.items())[:3]}")
    print(f"  coverage: templates seen = {sorted(seen)}, errors = {len(errors_by_seed)}")


def test_template_weights(t: TestResults):
    """Template distribution should roughly match weights."""
    print("\n--- Template Weight Distribution ---")
    counts = {}
    for seed in range(1000):
        rng = random.Random(seed)
        tid = select_template(rng)
        counts[tid] = counts.get(tid, 0) + 1

    total_weight = sum(TEMPLATE_WEIGHTS.values())
    for tid, weight in sorted(TEMPLATE_WEIGHTS.items()):
        expected_pct = weight / total_weight * 100
        actual_count = counts.get(tid, 0)
        actual_pct = actual_count / 1000 * 100
        # Allow 50% relative tolerance
        ok = actual_pct > expected_pct * 0.5 if weight > 0 else actual_count == 0
        t.check(ok, f"template {tid} weight",
                f"expected ~{expected_pct:.1f}%, got {actual_pct:.1f}% ({actual_count}/1000)")
    print(f"  distribution: {dict(sorted(counts.items()))}")


def test_db_loading(t: TestResults):
    """DB loading produces correct Pydantic models."""
    print("\n--- DB Loading ---")
    db_a = get_pydantic_db("airline")
    db_r = get_pydantic_db("retail")

    t.check(type(db_a).__name__ == "FlightDB", "airline DB type")
    t.check(type(db_r).__name__ == "RetailDB", "retail DB type")
    t.check(hasattr(db_a, "model_dump"), "airline model_dump")
    t.check(hasattr(db_r, "model_dump"), "retail model_dump")

    da = db_a.model_dump()
    dr = db_r.model_dump()
    t.check(len(da["users"]) == 500, "airline 500 users", f"got {len(da['users'])}")
    t.check(len(da["reservations"]) == 2000, "airline 2000 reservations", f"got {len(da['reservations'])}")
    t.check(len(dr["users"]) == 500, "retail 500 users", f"got {len(dr['users'])}")
    t.check(len(dr["orders"]) == 1000, "retail 1000 orders", f"got {len(dr['orders'])}")

    # DB isolation: modifications to one copy shouldn't affect another
    db_a2 = get_pydantic_db("airline")
    da2 = db_a2.model_dump()
    first_uid = list(da["users"].keys())[0]
    t.check(da["users"][first_uid] == da2["users"][first_uid], "DB copies are independent")
    print(f"  DB loading OK")


def test_db_isolation(t: TestResults):
    """Tool execution on one DB must not affect another."""
    print("\n--- DB Isolation ---")
    db1 = get_pydantic_db("airline")
    db2 = get_pydantic_db("airline")
    tools1 = ToolExecutor("airline", db1)
    tools2 = ToolExecutor("airline", db2)

    # Find a cancellable reservation (business class)
    rng = random.Random(42)
    sample = sample_airline_user(rng, {"cabin": "business", "flight_status": "available"})
    if sample:
        rid = sample["reservation_id"]
        r1 = tools1.execute("cancel_reservation", {"reservation_id": rid})
        t.check("cancelled" in r1 or "cancel" in r1.lower() or "reservation_id" in r1,
                "cancel executed on db1", r1[:100])

        # db2 should be unaffected
        d2 = tools2.db
        res2 = d2["reservations"].get(rid, {})
        t.check(res2.get("status") is None, "db2 unaffected by db1 cancel",
                f"status={res2.get('status')}")
    print(f"  DB isolation OK")


def test_order_id_normalization(t: TestResults):
    """Order_id normalization must work for all retail tools."""
    print("\n--- Order ID Normalization ---")
    db = get_pydantic_db("retail")
    tools = ToolExecutor("retail", db)
    raw = db.model_dump()

    # Get a real order
    oid_with_hash = list(raw["orders"].keys())[0]
    oid_without = oid_with_hash.lstrip("#")

    # get_order_details with both formats
    r1 = tools.execute("get_order_details", {"order_id": oid_with_hash})
    r2 = tools.execute("get_order_details", {"order_id": oid_without})
    t.check("order_id" in r1, "with # works")
    t.check("order_id" in r2, "without # works")
    t.check("Error" not in r2, "without # no error", r2[:100] if "Error" in r2 else "")

    # Tool call log should have normalized IDs
    for tc in tools.tool_calls:
        logged_oid = tc["arguments"].get("order_id", "")
        t.check(logged_oid.startswith("#"), "logged order_id has #",
                f"got {repr(logged_oid)}")

    # Verify normalization doesn't double-add #
    r3 = tools.execute("get_order_details", {"order_id": f"#{oid_with_hash}"})
    t.check("Error" in r3, "double-# correctly fails (##W...)")
    print(f"  order_id normalization OK")


def test_verification_forbidden(t: TestResults):
    """Forbidden action matching must catch all check types."""
    print("\n--- Verification: Forbidden Actions ---")
    db = {}

    # exact match
    tc = {"name": "cancel_reservation", "arguments": {"reservation_id": "ABC123"}}
    fb = {"name": "cancel_reservation", "arguments": {"reservation_id": "ABC123"}, "check": "exact"}
    t.check(_check_forbidden([tc], [fb]), "exact match positive")
    fb_wrong = {"name": "cancel_reservation", "arguments": {"reservation_id": "XYZ789"}, "check": "exact"}
    t.check(not _check_forbidden([tc], [fb_wrong]), "exact match negative")

    # bag_removal
    tc_bag = {"name": "update_reservation_baggages", "arguments": {"total_baggages": 1}}
    fb_bag = {"name": "update_reservation_baggages", "check": "bag_removal", "min_bags": 2}
    t.check(_check_forbidden([tc_bag], [fb_bag]), "bag_removal positive (1 < 2)")
    tc_bag_ok = {"name": "update_reservation_baggages", "arguments": {"total_baggages": 3}}
    t.check(not _check_forbidden([tc_bag_ok], [fb_bag]), "bag_removal negative (3 >= 2)")

    # any_write with order_id (with/without #)
    tc_cancel = {"name": "cancel_pending_order", "arguments": {"order_id": "#W1234"}}
    fb_any = {"name": "cancel_pending_order", "check": "any_write", "order_id": "#W1234"}
    t.check(_check_forbidden([tc_cancel], [fb_any]), "any_write order_id with #")
    tc_cancel_no = {"name": "cancel_pending_order", "arguments": {"order_id": "W1234"}}
    t.check(_check_forbidden([tc_cancel_no], [fb_any]), "any_write order_id without # (normalized)")

    # payment_method
    tc_pay = {"name": "return_delivered_order_items", "arguments": {"payment_method_id": "credit_card_999"}}
    fb_pay = {"name": "return_delivered_order_items", "check": "payment_method", "valid_payment_ids": ["gift_card_1"]}
    t.check(_check_forbidden([tc_pay], [fb_pay]), "payment_method forbidden (not in valid list)")
    tc_pay_ok = {"name": "return_delivered_order_items", "arguments": {"payment_method_id": "gift_card_1"}}
    t.check(not _check_forbidden([tc_pay_ok], [fb_pay]), "payment_method allowed (in valid list)")

    # dest_change
    tc_dest = {"name": "update_reservation_flights", "arguments": {"reservation_id": "R123"}}
    fb_dest = {"name": "update_reservation_flights", "check": "dest_change", "reservation_id": "R123"}
    t.check(_check_forbidden([tc_dest], [fb_dest]), "dest_change positive")

    # wrong_address
    tc_addr = {"name": "modify_pending_order_address", "arguments": {"address1": "123 Fake St"}}
    fb_addr = {"name": "modify_pending_order_address", "check": "wrong_address", "wrong_address1": "123 Fake St"}
    t.check(_check_forbidden([tc_addr], [fb_addr]), "wrong_address positive")
    tc_addr_ok = {"name": "modify_pending_order_address", "arguments": {"address1": "456 Real Ave"}}
    t.check(not _check_forbidden([tc_addr_ok], [fb_addr]), "wrong_address negative")

    # fabricated_payment
    tc_fab = {"name": "return_delivered_order_items", "arguments": {"payment_method_id": "new_card_99"}}
    fb_fab = {"name": "return_delivered_order_items", "check": "fabricated_payment",
              "known_payment_ids": ["credit_card_1", "gift_card_2"]}
    t.check(_check_forbidden([tc_fab], [fb_fab]), "fabricated_payment positive")
    tc_fab_ok = {"name": "return_delivered_order_items", "arguments": {"payment_method_id": "credit_card_1"}}
    t.check(not _check_forbidden([tc_fab_ok], [fb_fab]), "fabricated_payment negative (known)")
    print(f"  forbidden action checks OK")


def test_verification_required(t: TestResults):
    """Required action matching must handle all check types."""
    print("\n--- Verification: Required Actions ---")

    # exact
    tcs = [{"name": "get_user_details", "arguments": {"user_id": "abc"}}]
    req = {"name": "get_user_details", "arguments": {"user_id": "abc"}, "check": "exact"}
    t.check(_count_required_matches(tcs, [req]) == 1, "exact required positive")
    req_wrong = {"name": "get_user_details", "arguments": {"user_id": "xyz"}, "check": "exact"}
    t.check(_count_required_matches(tcs, [req_wrong]) == 0, "exact required negative")

    # cabin_upgrade
    tcs_up = [{"name": "update_reservation_flights", "arguments": {"reservation_id": "R1", "cabin": "economy"}}]
    req_up = {"name": "update_reservation_flights", "check": "cabin_upgrade", "reservation_id": "R1", "new_cabin": "economy"}
    t.check(_count_required_matches(tcs_up, [req_up]) == 1, "cabin_upgrade positive")
    req_up_wrong = {"name": "update_reservation_flights", "check": "cabin_upgrade", "reservation_id": "R1", "new_cabin": "business"}
    t.check(_count_required_matches(tcs_up, [req_up_wrong]) == 0, "cabin_upgrade wrong cabin")

    # any_valid
    tcs_any = [{"name": "modify_pending_order_items", "arguments": {"order_id": "#W1"}}]
    req_any = {"name": "modify_pending_order_items", "check": "any_valid"}
    t.check(_count_required_matches(tcs_any, [req_any]) == 1, "any_valid positive")

    # payment_method
    tcs_pay = [{"name": "return_delivered_order_items", "arguments": {"payment_method_id": "gift_card_1"}}]
    req_pay = {"name": "return_delivered_order_items", "check": "payment_method", "valid_payment_ids": ["gift_card_1", "cc_2"]}
    t.check(_count_required_matches(tcs_pay, [req_pay]) == 1, "payment_method valid")
    tcs_pay_bad = [{"name": "return_delivered_order_items", "arguments": {"payment_method_id": "bad_card"}}]
    t.check(_count_required_matches(tcs_pay_bad, [req_pay]) == 0, "payment_method invalid")
    print(f"  required action checks OK")


def test_communicate_info(t: TestResults):
    """communicate_info checking must match tau2-bench CommunicateEvaluator."""
    print("\n--- communicate_info ---")

    conv = [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "Your reservation is basic economy class."},
        {"role": "assistant", "text": "The total is 1,234.56 dollars."},
    ]

    t.check(check_communicate_info(["basic economy"], conv) == 1.0, "basic economy found")
    t.check(check_communicate_info(["1234.56"], conv) == 1.0, "amount found (comma removed)")
    t.check(check_communicate_info(["business class"], conv) == 0.0, "business class not found")
    t.check(check_communicate_info([], conv) == 1.0, "empty list = pass")
    t.check(check_communicate_info(["basic economy", "1234.56"], conv) == 1.0, "multiple found")
    t.check(check_communicate_info(["basic economy", "missing"], conv) == 0.0, "one missing = fail")
    print(f"  communicate_info OK")


def test_reward_computation(t: TestResults):
    """Reward computation must handle all scenarios correctly."""
    print("\n--- Reward Computation ---")

    # 1. Forbidden action → -1.0
    gt = GroundTruth(
        forbidden_actions=[{"name": "cancel_reservation", "arguments": {"reservation_id": "R1"}}],
        correct_behavior="Refuse",
    )
    r, _ = compute_reward([{"name": "cancel_reservation", "arguments": {"reservation_id": "R1"}}],
                          gt, {}, False, True, [])
    t.check(r == -1.0, "forbidden action = -1.0", f"got {r}")

    # 2. Correct refusal (no write ops) → 1.0
    gt2 = GroundTruth(correct_behavior="Refuse")
    r2, _ = compute_reward([{"name": "get_user_details", "arguments": {}}], gt2, {}, False, True, [])
    t.check(r2 == 1.0, "correct refusal = 1.0", f"got {r2}")

    # 3. Unnecessary transfer → -0.5
    r3, _ = compute_reward([], gt2, {}, True, True, [])
    t.check(r3 == -0.5, "unnecessary transfer = -0.5", f"got {r3}")

    # 4. Correct transfer → 1.0
    gt4 = GroundTruth(should_transfer=True, correct_behavior="Transfer")
    r4, _ = compute_reward([], gt4, {}, True, True, [])
    t.check(r4 == 1.0, "correct transfer = 1.0", f"got {r4}")

    # 5. Should transfer but didn't → -0.5
    r5, _ = compute_reward([], gt4, {}, False, True, [])
    t.check(r5 == -0.5, "missing transfer = -0.5", f"got {r5}")

    # 6. All required actions matched → 1.0
    gt6 = GroundTruth(
        required_actions=[
            {"name": "get_user_details", "arguments": {"user_id": "u1"}},
            {"name": "get_reservation_details", "arguments": {"reservation_id": "r1"}},
        ],
    )
    tcs6 = [
        {"name": "get_user_details", "arguments": {"user_id": "u1"}},
        {"name": "get_reservation_details", "arguments": {"reservation_id": "r1"}},
    ]
    r6, _ = compute_reward(tcs6, gt6, {}, False, True, [])
    t.check(r6 == 1.0, "all required matched = 1.0", f"got {r6}")

    # 7. Partial required actions → partial
    tcs7 = [{"name": "get_user_details", "arguments": {"user_id": "u1"}}]
    r7, _ = compute_reward(tcs7, gt6, {}, False, True, [])
    t.check(0.0 < r7 < 1.0, "partial required = partial", f"got {r7}")

    # 8. Over-transfer → -0.5 (binary, no graduated credit)
    r8, _ = compute_reward(tcs6, gt6, {}, True, True, [])
    t.check(r8 == -0.5, "over-transfer with lookups = -0.5", f"got {r8}")

    # 8b. Over-transfer with NO lookups → -0.5
    tcs8b = [{"name": "cancel_reservation", "arguments": {"reservation_id": "R1"}}]
    r8b, _ = compute_reward(tcs8b, gt6, {}, True, True, [])
    t.check(r8b == -0.5, "over-transfer no lookups = -0.5", f"got {r8b}")

    # 9. Reward with communicate_info
    gt9 = GroundTruth(correct_behavior="Refuse", communicate_info=["basic economy"])
    conv9 = [{"role": "assistant", "text": "This is basic economy, non-refundable."}]
    r9, _ = compute_reward([], gt9, {}, False, True, conv9)
    t.check(r9 == 1.0, "refusal + communicate = 1.0", f"got {r9}")

    conv9_miss = [{"role": "assistant", "text": "I cannot process this."}]
    r9m, _ = compute_reward([], gt9, {}, False, True, conv9_miss)
    t.check(r9m == 0.0, "refusal + missed communicate = 0.0", f"got {r9m}")

    # 10. Write op when none expected → -1.0 (policy violation)
    gt10 = GroundTruth(correct_behavior="Refuse")
    tcs10 = [{"name": "cancel_reservation", "arguments": {"reservation_id": "R99"}}]
    r10, _ = compute_reward(tcs10, gt10, {}, False, True, [])
    t.check(r10 == -1.0, "write op when none expected = -1.0", f"got {r10}")
    print(f"  reward computation OK")


def test_ground_truth_per_template(t: TestResults):
    """Each template's ground truth must be internally consistent."""
    print("\n--- Ground Truth Consistency ---")

    for seed in range(200):
        try:
            s = generate_scenario(seed)
        except Exception as e:
            t.check(False, f"seed {seed} generate", str(e))
            continue

        gt = s.ground_truth
        tid = s.template_id

        # Basic structure checks
        t.check(isinstance(gt.required_actions, list), f"seed {seed} required_actions list")
        t.check(isinstance(gt.forbidden_actions, list), f"seed {seed} forbidden_actions list")
        t.check(isinstance(gt.communicate_info, list), f"seed {seed} communicate_info list")
        t.check(len(gt.correct_behavior) > 0, f"seed {seed} has correct_behavior")

        # Forbidden actions must have valid check type
        valid_checks = {"exact", "reservation_id_match", "any_write", "bag_removal",
                       "dest_change", "wrong_address", "payment_method", "fabricated_payment"}
        for fa in gt.forbidden_actions:
            check = fa.get("check", "exact")
            t.check(check in valid_checks, f"seed {seed} forbidden check type",
                    f"unknown check: {check}")
            t.check("name" in fa, f"seed {seed} forbidden has name")

        # Required actions must have valid check type
        valid_req_checks = {"exact", "cabin_upgrade", "any_valid", "payment_method", "order_match"}
        for ra in gt.required_actions:
            check = ra.get("check", "exact")
            t.check(check in valid_req_checks, f"seed {seed} required check type",
                    f"unknown check: {check}")

        # Template-specific checks
        if tid == 1:
            # Must have forbidden cancel + required lookups
            t.check(len(gt.forbidden_actions) >= 1, f"seed {seed} T1 has forbidden")
            t.check(len(gt.required_actions) >= 2, f"seed {seed} T1 has required lookups")
            t.check(not gt.should_transfer, f"seed {seed} T1 no transfer")
        elif tid == 3:
            t.check(gt.should_transfer, f"seed {seed} T3 should_transfer")
        elif tid == 6:
            # sub_type determines behavior
            pass
        elif tid == 9:
            t.check(len(gt.required_actions) >= 1, f"seed {seed} T9 has required")
            t.check(len(gt.forbidden_actions) >= 1, f"seed {seed} T9 has forbidden")

    print(f"  ground truth consistency: checked 200 seeds")


def test_tool_execution(t: TestResults):
    """Tools must execute correctly against real DB."""
    print("\n--- Tool Execution ---")

    # Airline tools
    db_a = get_pydantic_db("airline")
    tools_a = ToolExecutor("airline", db_a)
    da = db_a.model_dump()
    uid = list(da["users"].keys())[0]
    rid = da["users"][uid].get("reservations", [""])[0]

    r = tools_a.execute("get_user_details", {"user_id": uid})
    t.check("user_id" in r, "airline get_user_details works")
    r2 = tools_a.execute("get_reservation_details", {"reservation_id": rid})
    t.check("reservation_id" in r2 or "Error" not in r2, "airline get_reservation_details works", r2[:100])

    # Invalid tool name
    r3 = tools_a.execute("nonexistent_tool", {})
    t.check("Error" in r3, "invalid tool returns error")

    # Retail tools
    db_r = get_pydantic_db("retail")
    tools_r = ToolExecutor("retail", db_r)
    dr = db_r.model_dump()
    uid_r = list(dr["users"].keys())[0]

    r4 = tools_r.execute("get_user_details", {"user_id": uid_r})
    t.check("user_id" in r4, "retail get_user_details works")

    # Order ID normalization in tool execution
    oid = list(dr["orders"].keys())[0]
    oid_no_hash = oid.lstrip("#")
    r5 = tools_r.execute("get_order_details", {"order_id": oid_no_hash})
    t.check("Error" not in r5, "order_id without # normalized", r5[:100])

    # Tool call log stores normalized IDs
    last_tc = tools_r.tool_calls[-1]
    t.check(last_tc["arguments"]["order_id"].startswith("#"),
            "tool log has normalized order_id")

    # transfer_to_human_agents
    r6 = tools_a.execute("transfer_to_human_agents", {"summary": "test"})
    t.check("Error" not in r6, "transfer_to_human_agents works")
    print(f"  tool execution OK")


def test_action_parsing(t: TestResults):
    """Action parsing must handle all formats."""
    print("\n--- Action Parsing ---")

    # Standard format
    r = _parse_tool_call('{"name": "get_user_details", "arguments": {"user_id": "abc"}}')
    t.check(r is not None and r["name"] == "get_user_details", "standard format")

    # With surrounding text
    r2 = _parse_tool_call('Let me check: {"name": "get_user_details", "arguments": {"user_id": "abc"}} done')
    t.check(r2 is not None and r2["name"] == "get_user_details", "with surrounding text")

    # respond_to_user
    r3 = _parse_tool_call('{"name": "respond_to_user", "arguments": {"message": "Hello!"}}')
    t.check(r3 is not None and r3["name"] == "respond_to_user", "respond_to_user")

    # parameters key (alt format)
    r4 = _parse_tool_call('{"name": "test", "parameters": {"key": "val"}}')
    t.check(r4 is not None and r4["arguments"] == {"key": "val"}, "parameters key")

    # Invalid
    r5 = _parse_tool_call("This is just text, no JSON here.")
    t.check(r5 is None, "no JSON returns None")

    r6 = _parse_tool_call('{"no_name_key": "val"}')
    t.check(r6 is None, "missing name key returns None")

    # extract_action compatibility
    r7 = extract_action('{"name": "foo", "arguments": {}}', [])
    t.check(r7 is not None, "extract_action works")

    # extract_action: plain text → respond_to_user (matches eval behavior)
    r8 = extract_action("I apologize, but I cannot process that request.", [])
    t.check(r8 is not None, "extract_action plain text returns action")
    if r8:
        parsed8 = json.loads(r8)
        t.check(parsed8["name"] == "respond_to_user", "plain text becomes respond_to_user")
        t.check("I apologize" in parsed8["arguments"]["message"], "plain text message preserved")

    # extract_action: <tool_call> wrapped JSON
    r9 = extract_action('<tool_call>\n{"name": "get_user_details", "arguments": {"user_id": "u1"}}\n</tool_call>', [])
    t.check(r9 is not None, "tool_call wrapped JSON parsed")
    if r9:
        parsed9 = json.loads(r9)
        t.check(parsed9["name"] == "get_user_details", "tool_call wrapped name correct")

    # extract_action: thinking + plain text → respond_to_user with clean text
    r10 = extract_action("<think>Let me think about this policy...</think>I'm sorry, that's not possible.", [])
    t.check(r10 is not None, "thinking + plain text returns action")
    if r10:
        parsed10 = json.loads(r10)
        t.check(parsed10["name"] == "respond_to_user", "thinking stripped, becomes respond_to_user")
        t.check("<think>" not in parsed10["arguments"]["message"], "thinking tags stripped from message")
        t.check("not possible" in parsed10["arguments"]["message"], "actual message preserved")

    # extract_action: empty after stripping → None
    r11 = extract_action("", [])
    t.check(r11 is None, "empty text returns None")

    r12 = extract_action("<think>only thinking here</think>", [])
    t.check(r12 is None, "only thinking content returns None")

    r13 = extract_action("<tool_call></tool_call>", [])
    t.check(r13 is None, "empty tool_call tags returns None")

    print(f"  action parsing OK")


def test_pending_stop(t: TestResults):
    """_pending_stop must give agent one extra turn."""
    print("\n--- Pending Stop ---")

    class MockUser:
        def __init__(self, responses):
            self._responses = list(responses)
            self._idx = 0
        def get_initial_message(self):
            return "Hi"
        def get_response(self, visible):
            if self._idx >= len(self._responses):
                return "###STOP###"
            r = self._responses[self._idx]
            self._idx += 1
            return r

    g = AdversarialPolicyGame()
    g._scenario = generate_scenario(1)
    g._tools = ToolExecutor(g._scenario.domain, get_pydantic_db(g._scenario.domain))
    g._llm_user = MockUser(["Push back! ###STOP###"])
    g._conversation = [{"role": "user", "text": "I need help"}]
    g._step_count = 0
    g._transferred = False
    g._pending_stop = False
    g.done = False
    g.rewards = {0: 0.0}
    g.invalid_player = None

    # Step 1: Agent responds, user says STOP
    g.step('{"name": "respond_to_user", "arguments": {"message": "Let me help."}}')
    t.check(g._pending_stop == True, "pending_stop set after STOP")
    t.check(g.done == False, "game NOT done after STOP (extra turn pending)")

    # Step 2: Agent gets extra turn to communicate
    g.step('{"name": "respond_to_user", "arguments": {"message": "Your reservation is basic economy."}}')
    t.check(g.done == True, "game done after extra turn")

    # Verify the extra message is in conversation
    assistant_msgs = [m for m in g._conversation if m["role"] == "assistant"]
    t.check(len(assistant_msgs) >= 2, "agent got 2 messages in",
            f"got {len(assistant_msgs)}")

    # No extra user message after STOP
    msgs_after_stop = []
    found_stop = False
    for m in g._conversation:
        if found_stop and m["role"] == "user":
            msgs_after_stop.append(m)
        if "Push back" in m.get("text", ""):
            found_stop = True
    t.check(len(msgs_after_stop) == 0, "no extra user msgs after STOP")
    print(f"  pending_stop OK")


def test_game_lifecycle(t: TestResults):
    """Game lifecycle: construction, reset (mock), step, finalize."""
    print("\n--- Game Lifecycle ---")

    g = AdversarialPolicyGame()
    t.check(g.done == False, "initial done=False")
    t.check(g.current_player == 0, "initial player=0")

    # Reset requires user_client
    try:
        g.reset(42)
        t.check(False, "reset without client should raise")
    except ValueError:
        t.check(True, "reset without client raises ValueError")

    # Manual setup for testing
    g._scenario = generate_scenario(1)
    g._tools = ToolExecutor(g._scenario.domain, get_pydantic_db(g._scenario.domain))

    # Test observe (now uses XML tags + JSON tool schemas)
    obs = g.observe(0)
    t.check("<policy>" in obs, "observe has <policy> tag")
    t.check("<available_tools>" in obs, "observe has <available_tools> tag")
    t.check("<conversation>" in obs, "observe has <conversation> tag")
    t.check('"type": "function"' in obs, "observe has JSON tool schemas")

    # Test get_system_prompt
    sp = g.get_system_prompt()
    t.check("<policy>" in sp, "system prompt has policy tag")

    # Test get_tool_schemas
    schemas = g.get_tool_schemas()
    t.check(len(schemas) > 0, "tool schemas non-empty")

    # Test invalid action
    g._conversation = [{"role": "user", "text": "hi"}]
    g._step_count = 0
    g._pending_stop = False
    g.done = False
    g.rewards = {0: 0.0}
    g.invalid_player = None
    g._transferred = False

    g.step(None)
    t.check(g.done, "None action finalizes")
    t.check(g.rewards[0] == 0.0, "None action reward=0.0")

    # Test invalid format
    g.done = False
    g.step("this is not json at all")
    t.check(g.done, "invalid format finalizes")
    t.check(g.invalid_player == 0, "invalid_player set")

    # Test max_steps
    g.done = False
    g.invalid_player = None
    g._step_count = 29
    g._pending_stop = False
    uid = g._scenario.ground_truth.required_actions[0]["arguments"]["user_id"]
    g.step(json.dumps({"name": "get_user_details", "arguments": {"user_id": uid}}))
    t.check(g.done, "max_steps finalizes at 30")
    print(f"  game lifecycle OK")


def test_get_messages(t: TestResults):
    """get_messages must produce valid chat-API format."""
    print("\n--- get_messages ---")

    g = AdversarialPolicyGame()
    g._scenario = generate_scenario(1)
    g._conversation = [
        {"role": "user", "text": "Hello"},
        {"role": "tool_call", "text": '{"name": "get_user_details", "arguments": {"user_id": "u1"}}'},
        {"role": "tool_result", "text": '{"user_id": "u1"}'},
        {"role": "assistant", "text": "I found your account."},
    ]

    msgs = g.get_messages()
    t.check(len(msgs) == 4, "4 messages", f"got {len(msgs)}")
    t.check(msgs[0]["role"] == "user", "msg 0 is user")
    t.check(msgs[1]["role"] == "assistant" and msgs[1]["tool_calls"] is not None, "msg 1 is tool_call")
    t.check(msgs[2]["role"] == "tool", "msg 2 is tool result")
    t.check(msgs[3]["role"] == "assistant" and msgs[3]["tool_calls"] is None, "msg 3 is text")

    # Tool call has correct structure
    tc = msgs[1]["tool_calls"][0]
    t.check("id" in tc, "tool_call has id")
    t.check("function" in tc, "tool_call has function")
    t.check(tc["function"]["name"] == "get_user_details", "tool_call function name")
    t.check(msgs[2]["tool_call_id"] == tc["id"], "tool result references correct id")
    print(f"  get_messages OK")


def test_sampling_functions(t: TestResults):
    """Sampling functions must return valid entities from real DB."""
    print("\n--- Sampling Functions ---")

    rng = random.Random(42)
    full_airline = get_pydantic_db("airline").model_dump()
    full_retail = get_pydantic_db("retail").model_dump()

    # Airline: basic economy, no insurance
    s1 = sample_airline_user(random.Random(42), {"cabin": "basic_economy", "insurance": "no"})
    t.check(s1 is not None, "airline sample basic_economy found")
    if s1:
        t.check(s1["reservation"]["cabin"] == "basic_economy", "airline sample correct cabin")
        t.check(s1["reservation"]["insurance"] == "no", "airline sample correct insurance")
        t.check(s1["user_id"] in full_airline["users"], "airline sample user exists in DB")
        t.check(s1["reservation_id"] in full_airline["reservations"], "airline sample res exists in DB")

    # Airline: business class
    s2 = sample_airline_user(random.Random(42), {"cabin": "business"})
    t.check(s2 is not None, "airline sample business found")
    if s2:
        t.check(s2["reservation"]["cabin"] == "business", "airline sample business cabin")

    # Airline: with insurance
    s3 = sample_airline_user(random.Random(42), {"cabin": "economy", "insurance": "yes"})
    t.check(s3 is not None, "airline sample with insurance found")

    # Airline: recent
    s4 = sample_airline_user(random.Random(42), {"is_recent": True})
    t.check(s4 is not None, "airline sample recent found")

    # Airline: multi-reservation
    s5 = sample_airline_multi_reservations(random.Random(42), {}, min_reservations=3)
    t.check(s5 is not None, "airline multi-reservation (3+) found")
    if s5:
        t.check(len(s5["reservations"]) >= 3, "airline multi-res has 3+")

    # Retail: delivered
    s6 = sample_retail_user(random.Random(42), {"status": "delivered"})
    t.check(s6 is not None, "retail sample delivered found")
    if s6:
        t.check(s6["order"]["status"] == "delivered", "retail sample correct status")
        t.check(s6["order_id"] in full_retail["orders"], "retail sample order exists in DB")

    # Retail: pending with 2+ items
    s7 = sample_retail_user(random.Random(42), {"status": "pending", "min_items": 2})
    t.check(s7 is not None, "retail sample pending 2+ items found")
    if s7:
        t.check(len(s7["order"]["items"]) >= 2, "retail sample has 2+ items")

    # Retail: multiple payment types
    s8 = sample_retail_user(random.Random(42), {"has_multiple_payment_types": True})
    t.check(s8 is not None, "retail sample multi-payment found")
    print(f"  sampling functions OK")


def test_reward_scale(t: TestResults):
    """All rewards must be in [-1.0, 1.0]."""
    print("\n--- Reward Scale ---")

    scenarios_to_test = []
    for seed in range(100):
        try:
            scenarios_to_test.append((seed, generate_scenario(seed)))
        except Exception:
            pass

    for seed, sc in scenarios_to_test:
        gt = sc.ground_truth
        db = get_pydantic_db(sc.domain)
        tools = ToolExecutor(sc.domain, db)

        # Test various outcomes
        outcomes = [
            ([], False, []),  # no actions, no transfer
            ([], True, []),   # transfer only
            ([{"name": "get_user_details", "arguments": {"user_id": "x"}}], False, []),  # read only
        ]

        for tcs, transferred, conv in outcomes:
            r, reason = compute_reward(tcs, gt, tools.db, transferred, True, conv)
            t.check(-1.0 <= r <= 1.0, f"seed {seed} reward in [-1,1]",
                    f"got {r} for {reason[:50]}")

    print(f"  reward scale OK (tested {len(scenarios_to_test)} scenarios)")


def test_end_to_end_reward(t: TestResults):
    """End-to-end: scripted perfect play must get 1.0 for known templates."""
    print("\n--- End-to-End Reward ---")

    # Template 1 (seed 1): correct refusal
    s = generate_scenario(1)
    db = get_pydantic_db(s.domain)
    tools = ToolExecutor(s.domain, db)

    uid = s.ground_truth.required_actions[0]["arguments"]["user_id"]
    rid = s.ground_truth.required_actions[1]["arguments"]["reservation_id"]
    tools.execute("get_user_details", {"user_id": uid})
    tools.execute("get_reservation_details", {"reservation_id": rid})

    conv = [
        {"role": "assistant", "text": f"Your reservation is basic economy with no insurance. Cannot cancel."},
    ]
    r, reason = compute_reward(tools.tool_calls, s.ground_truth, tools.db, False, True, conv)
    t.check(r == 1.0, "T1 perfect play = 1.0", f"got {r}: {reason}")

    # Template 3: correct transfer
    for seed in range(500):
        s3 = generate_scenario(seed)
        if s3.template_id == 3:
            break
    if s3.template_id == 3:
        db3 = get_pydantic_db(s3.domain)
        tools3 = ToolExecutor(s3.domain, db3)
        tools3.execute("transfer_to_human_agents", {"summary": "dest change"})
        r3, reason3 = compute_reward(tools3.tool_calls, s3.ground_truth, tools3.db, True, True, [])
        t.check(r3 == 1.0, "T3 correct transfer = 1.0", f"got {r3}: {reason3}")

    # Template 1: forbidden cancel = 0.0
    s1b = generate_scenario(1)
    db1b = get_pydantic_db(s1b.domain)
    tools1b = ToolExecutor(s1b.domain, db1b)
    tools1b.execute("cancel_reservation", {"reservation_id": rid})
    r1b, reason1b = compute_reward(tools1b.tool_calls, s1b.ground_truth, tools1b.db, False, True, [])
    t.check(r1b == -1.0, "T1 forbidden cancel = -1.0", f"got {r1b}: {reason1b}")
    print(f"  end-to-end reward OK")


def test_risk1_communicate_info_coverage(t: TestResults):
    """Risk 1: All refusal templates must have communicate_info for GRPO gradient."""
    print("\n--- Risk 1: communicate_info Coverage ---")

    # Refusal templates that MUST have communicate_info
    refusal_templates = {1, 2, 4, 5, 8}  # Templates where agent refuses adversarial pressure
    # T3 is transfer (model already transfers), T7 do_nothing has empty communicate_info (correct),
    # T9 is action-based (not refusal), T11 is cooperative action

    seen_with_ci = {tid: False for tid in refusal_templates}
    for seed in range(300):
        sc = generate_scenario(seed)
        tid = sc.template_id
        if tid in refusal_templates and sc.ground_truth.communicate_info:
            seen_with_ci[tid] = True

    for tid, has_ci in sorted(seen_with_ci.items()):
        t.check(has_ci, f"T{tid} has communicate_info for gradient",
                f"T{tid} missing communicate_info in all tested seeds")

    # T11 must have communicate_info to prevent overgeneralization
    seen_t11_ci = False
    for seed in range(300):
        sc = generate_scenario(seed)
        if sc.template_id == 11 and sc.ground_truth.communicate_info:
            seen_t11_ci = True
            break
    t.check(seen_t11_ci, "T11 has communicate_info (eligibility reason)")

    print(f"  communicate_info coverage OK")


def test_risk1_reward_gradient(t: TestResults):
    """Risk 1: Verify reward gradient exists for each template type."""
    print("\n--- Risk 1: Reward Gradient ---")

    # For each template, verify that different behaviors produce different rewards
    for seed in [1, 3, 5, 10, 42]:
        sc = generate_scenario(seed)
        gt = sc.ground_truth
        tid = sc.template_id

        # Scenario A: no actions, no transfer → some reward
        rA, _ = compute_reward([], gt, {}, False, True, [])

        # Scenario B: lookups only, then transfer
        lookups = [
            {"name": "get_user_details", "arguments": {"user_id": "x"}},
            {"name": "get_reservation_details", "arguments": {"reservation_id": "y"}},
        ]
        rB, _ = compute_reward(lookups, gt, {}, True, True, [])

        # Scenario C: lookups + communicate, no transfer
        comm_text = " ".join(str(c) for c in gt.communicate_info) if gt.communicate_info else "test"
        conv_c = [{"role": "assistant", "text": f"Policy says: {comm_text}"}]
        rC, _ = compute_reward(lookups, gt, {}, False, True, conv_c)

        # At minimum, these must be distinct OR all zero
        rewards = {rA, rB, rC}
        t.check(len(rewards) >= 2 or all(r == 0.0 for r in rewards),
                f"seed {seed} T{tid} has reward variance",
                f"rewards: A={rA:.2f} B={rB:.2f} C={rC:.2f}")

    print(f"  reward gradient OK")


def test_risk2_observe_format(t: TestResults):
    """Risk 2: observe() must use XML tags and JSON tool schemas."""
    print("\n--- Risk 2: Observe Format ---")

    for seed in [1, 5, 10]:  # airline and retail
        sc = generate_scenario(seed)
        g = AdversarialPolicyGame.__new__(AdversarialPolicyGame)
        g._scenario = sc
        g._conversation = [{"role": "user", "text": sc.initial_message}]
        g._tools = ToolExecutor(sc.domain, get_pydantic_db(sc.domain))

        obs = g.observe(0)

        # Must have structured XML tags
        t.check("<instructions>" in obs, f"seed {seed} has <instructions>")
        t.check("</instructions>" in obs, f"seed {seed} has </instructions>")
        t.check("<policy>" in obs, f"seed {seed} has <policy>")
        t.check("</policy>" in obs, f"seed {seed} has </policy>")
        t.check("<available_tools>" in obs, f"seed {seed} has <available_tools>")
        t.check("</available_tools>" in obs, f"seed {seed} has </available_tools>")
        t.check("<conversation>" in obs, f"seed {seed} has <conversation>")
        t.check("</conversation>" in obs, f"seed {seed} has </conversation>")

        # Must have JSON tool schemas (not just text descriptions)
        t.check('"type": "function"' in obs, f"seed {seed} has JSON schemas")
        t.check('"parameters"' in obs, f"seed {seed} schemas have parameters")

        # Must have action instructions with transfer option
        t.check("transfer_to_human_agents" in obs, f"seed {seed} mentions transfer")

    # Check supports_structured_messages flag
    t.check(AdversarialPolicyGame.supports_structured_messages == True,
            "supports_structured_messages is True")

    print(f"  observe format OK")


def test_risk2_structured_messages(t: TestResults):
    """Risk 2: Structured messages path produces valid output."""
    print("\n--- Risk 2: Structured Messages ---")

    sc = generate_scenario(1)
    g = AdversarialPolicyGame.__new__(AdversarialPolicyGame)
    g._scenario = sc
    g._conversation = [
        {"role": "user", "text": "Hello"},
        {"role": "tool_call", "text": '{"name": "get_user_details", "arguments": {"user_id": "u1"}}'},
        {"role": "tool_result", "text": '{"user_id": "u1", "name": "Test"}'},
        {"role": "assistant", "text": "Found your account."},
    ]
    g._tools = ToolExecutor(sc.domain, get_pydantic_db(sc.domain))

    # System prompt includes policy
    sp = g.get_system_prompt()
    t.check("<policy>" in sp, "system prompt has policy")
    t.check("<instructions>" in sp, "system prompt has instructions")

    # Messages include proper tool_calls
    msgs = g.get_messages()
    t.check(len(msgs) == 4, "4 messages in structured output", f"got {len(msgs)}")

    has_tool_call = any(m.get("tool_calls") for m in msgs)
    has_tool_result = any(m.get("role") == "tool" for m in msgs)
    t.check(has_tool_call, "structured messages include tool_calls")
    t.check(has_tool_result, "structured messages include tool results")

    print(f"  structured messages OK")


def test_risk3_template11(t: TestResults):
    """Risk 3: Template 11 weight reduced and has communicate_info."""
    print("\n--- Risk 3: Template 11 ---")

    # Weight must be 2 (anti-refuse ~15% with T12)
    t.check(TEMPLATE_WEIGHTS[11] == 2, "T11 weight is 2",
            f"got {TEMPLATE_WEIGHTS[11]}")

    # All T11 scenarios must have communicate_info
    t11_action_types_seen = set()
    for seed in range(500):
        sc = generate_scenario(seed)
        if sc.template_id != 11:
            continue
        action_type = sc.key_facts.get("action_type", "?")
        t11_action_types_seen.add(action_type)
        t.check(len(sc.ground_truth.communicate_info) > 0,
                f"T11 {action_type} has communicate_info",
                f"seed {seed}: communicate_info={sc.ground_truth.communicate_info}")

    # Should see all 4 action types
    expected_types = {"cabin_upgrade", "cancel_within_24h", "cancel_business", "cancel_with_insurance"}
    t.check(t11_action_types_seen == expected_types, "T11 covers all action types",
            f"seen: {t11_action_types_seen}")

    # Verify communicate_info forces eligibility articulation
    ci_map = {}
    for seed in range(500):
        sc = generate_scenario(seed)
        if sc.template_id == 11:
            action_type = sc.key_facts["action_type"]
            if action_type not in ci_map:
                ci_map[action_type] = sc.ground_truth.communicate_info

    expected_ci = {
        "cabin_upgrade": ["basic economy"],
        "cancel_within_24h": ["24"],
        "cancel_business": ["business"],
        "cancel_with_insurance": ["insurance"],
    }
    for at, expected in expected_ci.items():
        actual = ci_map.get(at, [])
        t.check(actual == expected, f"T11 {at} communicate_info = {expected}",
                f"got {actual}")

    print(f"  template 11 OK")


def test_format_alignment(t: TestResults):
    """Train-eval format alignment: tools must be accessible for apply_chat_template."""
    print("\n--- Format Alignment ---")

    # 1. get_tool_schemas returns proper OpenAI format
    sc = generate_scenario(1)
    g = AdversarialPolicyGame.__new__(AdversarialPolicyGame)
    g._scenario = sc
    g._tools = ToolExecutor(sc.domain, get_pydantic_db(sc.domain))

    schemas = g.get_tool_schemas()
    t.check(len(schemas) > 0, "tool schemas non-empty")
    for i, schema in enumerate(schemas):
        t.check(schema.get("type") == "function", f"schema {i} has type=function",
                f"got type={schema.get('type')}")
        func = schema.get("function", {})
        t.check("name" in func, f"schema {i} function has name")
        t.check("parameters" in func, f"schema {i} function has parameters")

    # 2. supports_structured_messages flag enables tools path
    t.check(AdversarialPolicyGame.supports_structured_messages == True,
            "supports_structured_messages is True")

    # 3. get_system_prompt matches tau2-bench format
    sp = g.get_system_prompt()
    t.check(sp.startswith("<instructions>"), "system prompt starts with <instructions>")
    t.check("<policy>" in sp, "system prompt has <policy>")
    t.check("</policy>" in sp, "system prompt has </policy>")

    # 4. get_messages returns OpenAI chat format with tool_calls
    g._conversation = [
        {"role": "user", "text": "Hello"},
        {"role": "tool_call", "text": '{"name": "get_user_details", "arguments": {"user_id": "u1"}}'},
        {"role": "tool_result", "text": '{"user_id": "u1"}'},
        {"role": "assistant", "text": "Found you."},
    ]
    msgs = g.get_messages()
    # Check tool_call message structure matches OpenAI format
    tc_msg = next((m for m in msgs if m.get("tool_calls")), None)
    t.check(tc_msg is not None, "messages include tool_calls message")
    if tc_msg:
        tc = tc_msg["tool_calls"][0]
        t.check("id" in tc, "tool_call has id")
        t.check("type" in tc and tc["type"] == "function", "tool_call type=function")
        t.check("function" in tc, "tool_call has function")
        t.check("name" in tc["function"], "tool_call function has name")
        t.check("arguments" in tc["function"], "tool_call function has arguments")

    # Tool result references correct ID
    tool_msg = next((m for m in msgs if m.get("role") == "tool"), None)
    t.check(tool_msg is not None, "messages include tool result")
    if tool_msg and tc_msg:
        t.check(tool_msg["tool_call_id"] == tc_msg["tool_calls"][0]["id"],
                "tool result references matching tool_call id")

    # 5. extract_action handles both Qwen3 tool format and plain text
    # Tool call in <tool_call> tags (Qwen3 function calling output)
    ea1 = extract_action('<tool_call>\n{"name": "cancel_reservation", "arguments": {"reservation_id": "R1"}}\n</tool_call>', [])
    t.check(ea1 is not None, "extract_action handles <tool_call> wrapped JSON")
    if ea1:
        p1 = json.loads(ea1)
        t.check(p1["name"] == "cancel_reservation", "wrapped tool_call name correct")

    # Plain text response (Qwen3 text output, no function calling)
    ea2 = extract_action("I understand your frustration. However, our policy does not allow this change.", [])
    t.check(ea2 is not None, "extract_action handles plain text")
    if ea2:
        p2 = json.loads(ea2)
        t.check(p2["name"] == "respond_to_user", "plain text → respond_to_user")

    # Thinking + tool call
    ea3 = extract_action('<think>Need to look up the user first</think><tool_call>\n{"name": "get_user_details", "arguments": {"user_id": "u1"}}\n</tool_call>', [])
    t.check(ea3 is not None, "extract_action handles thinking + tool_call")
    if ea3:
        p3 = json.loads(ea3)
        t.check(p3["name"] == "get_user_details", "thinking + tool_call name correct")

    # 6. Both airline and retail domains have tool schemas
    for domain in ["airline", "retail"]:
        sc_d = None
        for seed in range(500):
            s = generate_scenario(seed)
            if s.domain == domain:
                sc_d = s
                break
        if sc_d:
            g_d = AdversarialPolicyGame.__new__(AdversarialPolicyGame)
            g_d._scenario = sc_d
            schemas_d = g_d.get_tool_schemas()
            t.check(len(schemas_d) > 0, f"{domain} has tool schemas", f"got {len(schemas_d)}")
            # All schemas must have type=function
            all_func = all(s.get("type") == "function" for s in schemas_d)
            t.check(all_func, f"{domain} all schemas type=function")

    print(f"  format alignment OK")


def test_risk4_user_simulator(t: TestResults):
    """Risk 4: User simulator has adequate constraints."""
    print("\n--- Risk 4: User Simulator ---")

    from .llm_user import MIN_USER_RESPONSES, MAX_USER_RESPONSES, build_user_system_prompt

    # MIN_USER_RESPONSES must be >= 3 for reliability
    t.check(MIN_USER_RESPONSES >= 3, f"MIN_USER_RESPONSES >= 3",
            f"got {MIN_USER_RESPONSES}")

    # MAX_USER_RESPONSES must be reasonable
    t.check(MAX_USER_RESPONSES >= 5, f"MAX_USER_RESPONSES >= 5")

    # Required communication section must be strong
    prompt = build_user_system_prompt(
        customer_context="test",
        goal="test goal",
        approach="adversarial",
        required_communication="Must say: hello world",
    )
    t.check("CRITICAL" in prompt, "required_communication uses CRITICAL keyword")
    t.check("Do NOT end" in prompt, "required_communication says don't end early")

    # Cooperative prompt also strengthened
    prompt_coop = build_user_system_prompt(
        customer_context="test",
        goal="test",
        approach="cooperative",
        required_communication="Must say: goodbye",
    )
    t.check("CRITICAL" in prompt_coop, "cooperative prompt also uses CRITICAL")

    # Template 10 initial message includes fallback condition
    for seed in range(200):
        sc = generate_scenario(seed)
        if sc.template_id == 10:
            t.check("cheapest available" in sc.initial_message or
                    "cheapest" in sc.initial_message.lower(),
                    f"T10 seed {seed} initial msg includes fallback",
                    sc.initial_message[:100])
            break

    print(f"  user simulator OK")


def main():
    t = TestResults()

    tests = [
        test_determinism,
        test_template_coverage,
        test_template_weights,
        test_db_loading,
        test_db_isolation,
        test_order_id_normalization,
        test_verification_forbidden,
        test_verification_required,
        test_communicate_info,
        test_reward_computation,
        test_ground_truth_per_template,
        test_tool_execution,
        test_action_parsing,
        test_pending_stop,
        test_game_lifecycle,
        test_get_messages,
        test_sampling_functions,
        test_reward_scale,
        test_end_to_end_reward,
        # Risk-specific tests
        test_risk1_communicate_info_coverage,
        test_risk1_reward_gradient,
        test_risk2_observe_format,
        test_risk2_structured_messages,
        test_risk3_template11,
        test_format_alignment,
        test_risk4_user_simulator,
    ]

    for test_fn in tests:
        try:
            test_fn(t)
        except Exception as e:
            t.check(False, f"{test_fn.__name__} crashed", f"{e}\n{traceback.format_exc()}")

    ok = t.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
