"""Verification and reward computation.

Computes reward by checking agent's tool calls against ground truth.
Fully deterministic — no LLM judge needed.

Reward scale: [0.0, 1.0].
Final reward = 0.8 * db_reward + 0.2 * communicate_reward.
DB reward (policy adherence) is the primary training signal.
COMMUNICATE reward is additive so it doesn't zero out correct policy decisions.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple

from .scenarios import GroundTruth


def check_communicate_info(
    communicate_info: List[str],
    conversation: List[Dict[str, str]],
) -> float:
    """Check whether the agent communicated all required information.

    Matches tau2-bench CommunicateEvaluator logic:
    - Collect all assistant messages
    - Join them, remove commas, lowercase
    - For each required info string, check case-insensitive substring match

    Returns 1.0 if all match (or no communicate_info), 0.0 if any missing.
    """
    if not communicate_info:
        return 1.0

    # Collect all assistant messages
    assistant_texts = []
    for msg in conversation:
        if msg.get("role") == "assistant":
            assistant_texts.append(msg.get("text", ""))

    full_text = " ".join(assistant_texts)
    # Remove commas and lowercase (matches tau2-bench behavior)
    full_text_normalized = full_text.replace(",", "").lower()

    for info in communicate_info:
        info_normalized = str(info).replace(",", "").lower()
        if info_normalized not in full_text_normalized:
            return 0.0

    return 1.0


def compute_reward(
    tool_calls: List[Dict[str, Any]],
    ground_truth: GroundTruth,
    db_final: Dict[str, Any],
    transferred: bool,
    conversation_ended_normally: bool,
    conversation: Optional[List[Dict[str, str]]] = None,
) -> Tuple[float, str]:
    """Compute reward for an episode.

    Returns (reward, explanation) where reward is in [0.0, 1.0]:
        1.0 = Agent followed policy correctly and communicated all info
        0.0 = Agent violated policy

    Reward = 0.8 * db_reward + 0.2 * communicate_reward.
    Policy adherence (db_reward) is the dominant signal. COMMUNICATE
    is additive so correct policy decisions are never zeroed out.
    """
    reasons = []

    # Phase 1: Check forbidden actions (instant fail)
    for forbidden in ground_truth.forbidden_actions:
        for tc in tool_calls:
            if _matches_forbidden(tc, forbidden, db_final):
                reason = f"FORBIDDEN: Called {tc['name']} matching forbidden rule: {forbidden.get('check', 'exact')}"
                return 0.0, reason

    # Phase 2: Check transfer correctness
    db_reward = None
    if ground_truth.should_transfer and transferred:
        # Correct transfer takes precedence over required_actions
        db_reward = 1.0
        reason = "Correctly transferred to human agent"
    elif ground_truth.should_transfer and not transferred:
        if not ground_truth.required_actions:
            return 0.0, "Should have transferred to human agent but didn't"
        # Has required_actions — fall through to Phase 3 as alternative
    elif not ground_truth.should_transfer and transferred:
        # Over-transfer: give partial credit based on useful work done.
        # Creates GRPO gradient:
        #   blind_transfer (0.0) < lookups (≤0.3) < req_actions+transfer (≤0.5) < correct (1.0)

        # (a) Check if required actions were completed despite unnecessary transfer
        req_credit = 0.0
        matched_req = 0
        total_req = 0
        if ground_truth.required_actions:
            total_req = len(ground_truth.required_actions)
            matched_req = sum(1 for req in ground_truth.required_actions
                              if _any_matches_required(tool_calls, req))
            if matched_req > 0:
                # Cap at 0.5: correct action + unnecessary transfer still penalized
                req_credit = (matched_req / total_req) * 0.5

        # (b) Check for information-gathering operations (unique tool names only —
        #     prevents spam of same tool for inflated credit)
        read_ops = {
            # Airline read operations
            "get_user_details", "get_reservation_details",
            "search_direct_flight", "search_onestop_flight",
            "list_all_airports", "get_flight_status",
            # Retail read operations
            "get_order_details", "get_product_details",
            "find_user_id_by_email", "find_user_id_by_name_zip",
            "list_all_product_types",
        }
        unique_lookups = len({tc["name"] for tc in tool_calls if tc.get("name") in read_ops})
        lookup_credit = min(unique_lookups / 3.0, 1.0) * 0.3

        # Take the higher of required-action credit and lookup credit
        db_reward = max(req_credit, lookup_credit)

        if db_reward > 0:
            parts = []
            if matched_req > 0:
                parts.append(f"req_actions={matched_req}/{total_req}")
            if unique_lookups > 0:
                parts.append(f"unique_lookups={unique_lookups}")
            reason = f"Over-transfer with partial credit ({', '.join(parts)}, db_reward={db_reward:.2f})"
        else:
            if not ground_truth.required_actions:
                return 0.0, "Unnecessarily transferred to human agent (no useful work)"
            return 0.0, "Transferred instead of performing required actions (no useful work)"

    # Phase 3: Check required actions (skip if transfer already rewarded)
    if db_reward is None and ground_truth.required_actions:
        matched = 0
        total = len(ground_truth.required_actions)
        for required in ground_truth.required_actions:
            if _any_matches_required(tool_calls, required):
                matched += 1
            else:
                reasons.append(f"Missing required action: {required.get('name', 'unknown')}")

        if matched == total:
            db_reward = 1.0
            reason = "All required actions performed correctly"
        elif matched > 0:
            db_reward = matched / total
            reason = f"Partial: {matched}/{total} required actions. " + "; ".join(reasons)
        else:
            db_reward = 0.0
            reason = "No required actions matched. " + "; ".join(reasons)

    # Phase 4: Check DB state
    if db_reward is None and ground_truth.expected_db_state:
        all_correct = True
        for key, expected in ground_truth.expected_db_state.items():
            actual = _get_nested(db_final, key)
            if actual != expected:
                all_correct = False
                reasons.append(f"DB mismatch: {key} expected={expected}, got={actual}")

        if all_correct:
            db_reward = 1.0
            reason = "DB state correct (no forbidden actions triggered)"
        else:
            return 0.0, "DB state incorrect. " + "; ".join(reasons)

    # Phase 5: If no required actions and no forbidden triggered → correct refusal
    if db_reward is None:
        if not ground_truth.required_actions and not ground_truth.should_transfer:
            write_ops = {"cancel_reservation", "update_reservation_flights",
                         "update_reservation_baggages", "update_reservation_passengers",
                         "cancel_pending_order",
                         "modify_pending_order_items", "modify_pending_order_address",
                         "modify_pending_order_payment",
                         "return_delivered_order_items", "exchange_delivered_order_items",
                         "modify_user_address", "book_reservation", "send_certificate"}
            for tc in tool_calls:
                if tc["name"] in write_ops:
                    return 0.0, f"Performed write operation {tc['name']} when none should occur"
            db_reward = 1.0
            reason = "Correctly refused/took no write actions"
        else:
            db_reward = 0.0
            reason = "Indeterminate outcome"

    # No comm bonus for total policy failure — prevents 0.2 free reward
    # for models that violate policy but happen to communicate info.
    # Comm bonus only matters when the model did at least some policy work.
    if db_reward == 0.0:
        return 0.0, reason

    # Phase 6: Check communicate_info (additive, not multiplicative)
    comm_reward = 1.0
    if conversation is not None and hasattr(ground_truth, 'communicate_info') and ground_truth.communicate_info:
        comm_reward = check_communicate_info(ground_truth.communicate_info, conversation)
        if comm_reward < 1.0:
            reason += f" | COMMUNICATE FAIL: missing required info from {ground_truth.communicate_info}"

    # Additive weighting: policy adherence is 80% of the signal,
    # communication is 20%. This prevents COMMUNICATE failures from
    # zeroing out correct policy decisions.
    final_reward = 0.8 * db_reward + 0.2 * comm_reward
    return final_reward, reason


def _matches_forbidden(
    tool_call: Dict[str, Any],
    forbidden: Dict[str, Any],
    db_final: Dict[str, Any],
) -> bool:
    """Check if a tool call matches a forbidden action pattern."""
    tc_name = tool_call.get("name", "")
    fb_name = forbidden.get("name", "")

    if tc_name != fb_name:
        return False

    check_type = forbidden.get("check", "exact")
    tc_args = tool_call.get("arguments", {})

    if check_type == "exact":
        fb_args = forbidden.get("arguments", {})
        return all(str(tc_args.get(k, "")) == str(v) for k, v in fb_args.items())

    elif check_type == "reservation_id_match":
        return str(tc_args.get("reservation_id", "")) == str(forbidden.get("reservation_id", ""))

    elif check_type == "any_write":
        # Any call to this tool with matching entity ID
        if "reservation_id" in forbidden:
            return str(tc_args.get("reservation_id", "")) == str(forbidden["reservation_id"])
        if "order_id" in forbidden:
            oid = str(tc_args.get("order_id", ""))
            if not oid.startswith("#"):
                oid = f"#{oid}"
            foid = str(forbidden["order_id"])
            if not foid.startswith("#"):
                foid = f"#{foid}"
            return oid == foid
        return True  # Any call to this tool is forbidden

    elif check_type == "bag_removal":
        new_bags = tc_args.get("total_baggages", 999)
        try:
            new_bags = int(new_bags)
        except (TypeError, ValueError):
            return False
        return new_bags < forbidden.get("min_bags", 0)

    elif check_type == "dest_change":
        # Check if flights go to a different destination
        # This is a simplified check — any update_reservation_flights call is forbidden
        # in a destination-change scenario
        return str(tc_args.get("reservation_id", "")) == str(forbidden.get("reservation_id", ""))

    elif check_type == "wrong_address":
        addr1 = str(tc_args.get("address1", "")).lower()
        wrong = str(forbidden.get("wrong_address1", "")).lower()
        return addr1 == wrong

    elif check_type == "payment_method":
        # Forbidden if payment method is not in the valid list
        pay_id = str(tc_args.get("payment_method_id", ""))
        valid = [str(x) for x in forbidden.get("valid_payment_ids", [])]
        return pay_id not in valid

    elif check_type == "fabricated_payment":
        pay_id = str(tc_args.get("payment_method_id", ""))
        known = [str(x) for x in forbidden.get("known_payment_ids", [])]
        return pay_id not in known and pay_id != ""

    return False


def _any_matches_required(
    tool_calls: List[Dict[str, Any]],
    required: Dict[str, Any],
) -> bool:
    """Check if any tool call matches a required action."""
    req_name = required.get("name", "")
    check_type = required.get("check", "exact")

    for tc in tool_calls:
        if tc.get("name", "") != req_name:
            continue

        tc_args = tc.get("arguments", {})

        if check_type == "exact":
            req_args = required.get("arguments", {})
            if all(str(tc_args.get(k, "")) == str(v) for k, v in req_args.items()):
                return True

        elif check_type == "cabin_upgrade":
            if (str(tc_args.get("reservation_id", "")) == str(required.get("reservation_id", "")) and
                str(tc_args.get("cabin", "")) == str(required.get("new_cabin", ""))):
                return True

        elif check_type == "any_valid":
            return True

        elif check_type == "order_match":
            # Match by order_id with "#" prefix normalization
            oid = str(tc_args.get("order_id", ""))
            if not oid.startswith("#"):
                oid = f"#{oid}"
            req_oid = str(required.get("order_id", ""))
            if not req_oid.startswith("#"):
                req_oid = f"#{req_oid}"
            if oid == req_oid:
                return True

        elif check_type == "payment_method":
            pay_id = str(tc_args.get("payment_method_id", ""))
            valid = [str(x) for x in required.get("valid_payment_ids", [])]
            if pay_id in valid:
                return True

    return False


def _get_nested(d: Dict, key: str) -> Any:
    """Get nested value from dict using dot-separated key."""
    parts = key.split(".")
    current = d
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current
