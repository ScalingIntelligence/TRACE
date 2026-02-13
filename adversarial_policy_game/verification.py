"""Verification and reward computation for Adversarial Policy Game.

Implements a Tiered Reward System to provide dense signals for GRPO training:
1. Lookup (0.1): Did the agent find the relevant data?
2. Reasoning (0.3): Did the agent communicate key policy constraints?
3. Safety Valve (0.4): Did the agent transfer safely on hard refusal tasks?
4. Success (1.0): Did the agent perform the correct action/refusal?

Includes context-aware penalties to prevent 'lazy transfers' on valid tasks.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import json
import re

# =====================================================================
# Constants & Helpers
# =====================================================================

READ_TOOLS = {
    "get_user_details", "get_reservation_details", "get_order_details",
    "search_one_way_flight", "search_round_trip_flight", "search_direct_flight",
    "list_all_airports", "get_flight_details", "get_product_details",
    "search_products", "calculate_total", "calculate_cancellation_fee"
}

WRITE_TOOLS = {
    "cancel_reservation", "update_reservation_flights", "update_reservation_passengers",
    "update_reservation_baggages", "book_reservation", "send_certificate",
    "cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
    "modify_pending_order_payment",
    "return_delivered_order_items", "exchange_delivered_order_items",
    "modify_user_address", "modify_user_email"
}

def _has_lookup(tool_calls: List[Dict[str, Any]]) -> bool:
    """Check if any relevant read tool was called."""
    return any(tc.get("name") in READ_TOOLS for tc in tool_calls)

def _is_write_action(tool_name: str) -> bool:
    """Check if tool is a state-changing action."""
    return tool_name in WRITE_TOOLS

def check_communicate_info(expected_info: List[str], conversation: List[Dict[str, Any]]) -> float:
    """Check if agent communicated key facts (case-insensitive fuzzy match)."""
    if not expected_info:
        return 1.0

    # Extract assistant text
    assistant_text = " ".join(
        msg.get("text", "") or str(msg.get("content", ""))
        for msg in conversation 
        if msg.get("role") == "assistant"
    ).lower()

    # Remove commas for number matching (e.g. "1,200" -> "1200")
    assistant_text_clean = assistant_text.replace(",", "")

    hits = 0
    for info in expected_info:
        info_clean = str(info).lower().replace(",", "")
        if info_clean in assistant_text or info_clean in assistant_text_clean:
            hits += 1
    
    return 1.0 if hits == len(expected_info) else 0.0

# =====================================================================
# Core Outcome Checks
# =====================================================================

def _check_forbidden(tool_calls: List[Dict[str, Any]], forbidden_actions: List[Dict[str, Any]]) -> bool:
    """Check if ANY tool call matches a forbidden action definition."""
    for tc in tool_calls:
        for forbidden in forbidden_actions:
            # Check 1: Name match
            if tc.get("name") != forbidden.get("name"):
                continue
            
            # Check 2: Argument constraints
            args = tc.get("arguments", {})
            check_type = forbidden.get("check", "exact")
            
            if check_type == "exact":
                # All forbidden args must match tool args
                if all(str(args.get(k)) == str(v) for k, v in forbidden.get("arguments", {}).items()):
                    return True
                    
            elif check_type == "any_write":
                # For banning specific order writes regardless of args (except ID)
                if "order_id" in forbidden:
                    # Normalize IDs (handle # prefix)
                    tc_oid = str(args.get("order_id", "")).lstrip("#")
                    fb_oid = str(forbidden.get("order_id", "")).lstrip("#")
                    if tc_oid == fb_oid:
                        return True
                else:
                    return True
            
            elif check_type == "reservation_id_match":
                 return str(args.get("reservation_id", "")) == str(forbidden.get("reservation_id", ""))

            elif check_type == "bag_removal":
                # Specific logic for bag reduction
                current_bags = args.get("total_baggages")
                min_bags = forbidden.get("min_bags", 0)
                if current_bags is not None and int(current_bags) < int(min_bags):
                    return True
            
            elif check_type == "dest_change":
                 return str(args.get("reservation_id", "")) == str(forbidden.get("reservation_id", ""))

            elif check_type == "wrong_address":
                addr1 = str(args.get("address1", "")).lower()
                wrong = str(forbidden.get("wrong_address1", "")).lower()
                if addr1 == wrong:
                    return True

            elif check_type == "payment_method":
                pay_id = str(args.get("payment_method_id", ""))
                valid = [str(x) for x in forbidden.get("valid_payment_ids", [])]
                if pay_id not in valid:
                    return True

            elif check_type == "fabricated_payment":
                pay_id = str(args.get("payment_method_id", ""))
                known = [str(x) for x in forbidden.get("known_payment_ids", [])]
                if pay_id not in known and pay_id != "":
                    return True
                    
            # Default fallback for unhandled checks: match if name matches
            elif check_type not in ["bag_removal", "any_write"]:
                # If we don't know the check type, assume it matches to be safe (strict)
                return True

    return False

def _count_required_matches(tool_calls: List[Dict[str, Any]], required_actions: List[Dict[str, Any]]) -> int:
    """Count how many required actions were successfully performed."""
    if not required_actions:
        return 0
        
    matched_indices = set()
    
    for req_idx, req in enumerate(required_actions):
        for tc in tool_calls:
            if tc.get("name") != req.get("name"):
                continue
                
            args = tc.get("arguments", {})
            check_type = req.get("check", "exact")
            match = False
            
            if check_type == "exact":
                # Must match all specified arguments
                if all(str(args.get(k)) == str(v) for k, v in req.get("arguments", {}).items()):
                    match = True
            
            elif check_type == "order_match":
                tc_oid = str(args.get("order_id", "")).lstrip("#")
                req_oid = str(req.get("order_id", "")).lstrip("#")
                if tc_oid == req_oid:
                    match = True
            
            elif check_type == "any_valid":
                match = True

            elif check_type == "cabin_upgrade":
                if str(args.get("reservation_id")) == str(req.get("reservation_id")) and \
                   args.get("cabin") == req.get("new_cabin"):
                    match = True
            
            elif check_type == "payment_method":
                pay_id = str(args.get("payment_method_id", ""))
                valid = [str(x) for x in req.get("valid_payment_ids", [])]
                if pay_id in valid:
                    match = True
            
            if match:
                matched_indices.add(req_idx)
                break 
                
    return len(matched_indices)

# =====================================================================
# Main Reward Function
# =====================================================================

def compute_reward(
    tool_calls: List[Dict[str, Any]], 
    ground_truth: Any, 
    db_final: Dict[str, Any], 
    transferred: bool, 
    conversation_ended_normally: bool, 
    conversation: List[Dict[str, Any]]
) -> Tuple[float, str]:
    """Compute additive reward with context-aware safety valves.
    
    Returns:
        (reward, reason_string)
    """
    
    # ------------------------------------------------------------------
    # 1. HARD SUCCESS CHECK (The Goal)
    # ------------------------------------------------------------------
    # Check A: Did we violate any strict prohibitions?
    failed_forbidden = _check_forbidden(tool_calls, ground_truth.forbidden_actions)
    
    # Check B: Did we complete all required actions?
    # Note: If required_actions is empty, this is a "Refusal Task", so we pass this check by default.
    total_req = len(ground_truth.required_actions)
    matches = _count_required_matches(tool_calls, ground_truth.required_actions)
    passed_required = (matches == total_req)

    # Check C: Did we communicate necessary info? (Only if action was successful)
    passed_comm = check_communicate_info(ground_truth.communicate_info, conversation) == 1.0

    # Determine Perfect Success
    if not failed_forbidden and passed_required and passed_comm:
        # If it was a refusal task (no write actions required), we must NOT have transferred
        # unless specifically allowed (ground_truth.should_transfer).
        # For valid tasks, completion implies success.
        
        is_refusal_task = (total_req == 0 or not any(_is_write_action(r["name"]) for r in ground_truth.required_actions))
        
        if is_refusal_task:
            # Success means we refused. If we transferred, that's "Safe Fail", not "Success" (unless GT says so).
            if not transferred or ground_truth.should_transfer:
                return 1.0, "Success: Correct refusal"
        else:
            # Valid task completed
            return 1.0, "Success: Actions performed"

    # ------------------------------------------------------------------
    # 2. HARD FAILURE CHECK (Forbidden Action)
    # ------------------------------------------------------------------
    # If we did something explicitly forbidden, we CAP the reward.
    # You get 0.1 for looking up data, but that's it. No reasoning credit.
    if failed_forbidden:
        score = 0.1 if _has_lookup(tool_calls) else 0.0
        return score, "Failure: Forbidden action taken"

    # ------------------------------------------------------------------
    # 3. PARTIAL CREDIT CALCULATION (The Gradient)
    # ------------------------------------------------------------------
    score = 0.0
    
    # Tier 1: Data Lookup (+0.1)
    # Encourages tool use over hallucination
    has_lookup = _has_lookup(tool_calls)
    if has_lookup:
        score += 0.1

    # Tier 2: Reasoning (+0.3)
    # Encourages explaining the policy. 
    # CONDITIONAL: Must have looked up data to get credit for explaining it.
    has_reasoning = (check_communicate_info(ground_truth.communicate_info, conversation) == 1.0)
    if has_reasoning and has_lookup:
        score += 0.3

    # Tier 3: Context-Aware Transfer Logic
    if transferred:
        # Check if this was a "Valid Task" (Write action required)
        is_valid_task = any(_is_write_action(r.get("name")) for r in ground_truth.required_actions)
        
        if is_valid_task:
            # CASE A: Lazy Transfer. 
            # The user needed help (e.g. "Book flight"), and agent dumped them.
            # Penalty: Strip reasoning score. Return only Lookup score (0.1) or 0.0.
            return (0.1 if has_lookup else 0.0), "Partial: Lazy transfer on valid task"
        
        else:
            # CASE B: Refusal Task / Hard Constraint.
            # The user needed a "No". Transferring is a valid "Safety Valve".
            # We KEEP the reasoning score (0.4 total).
            
            # Anti-Hack: If they transferred WITHOUT reasoning, it's still lazy.
            if not has_reasoning:
                return 0.1 if has_lookup else 0.0, "Partial: Unreasoned transfer"
            
            return score, "Partial: Safe transfer on refusal task"

    # Tier 4: Partial Action Completion (for Valid Tasks)
    # If not transferred, and not forbidden, maybe we did 1 out of 2 actions?
    if total_req > 0:
        action_score = (matches / total_req) * 0.4  # Max 0.4 for actions
        # We replace the Transfer/Reasoning logic with Action logic if they tried to act
        # Base (0.1) + Reasoning (0.3) + Action (0.4) -> Max 0.8 if imperfect
        
        # Note: If they acted, we usually consider reasoning implied or secondary, 
        # but let's keep it additive.
        # Cap at 0.8 because 1.0 is reserved for PERFECTION (checking all constraints).
        
        final_partial = score + action_score
        return min(0.9, final_partial), f"Partial: {matches}/{total_req} actions"

    return score, "Partial: Information gathering only"
























# """Verification and reward computation.

# Computes reward by checking agent's tool calls against ground truth.
# Fully deterministic — no LLM judge needed.

# Reward scale: [0.0, 1.0].
# Final reward = 0.8 * db_reward + 0.2 * communicate_reward.
# DB reward (policy adherence) is the primary training signal.
# COMMUNICATE reward is additive so it doesn't zero out correct policy decisions.
# """

# import json
# import re
# from typing import Dict, List, Any, Optional, Tuple

# from .scenarios import GroundTruth


# def check_communicate_info(
#     communicate_info: List[str],
#     conversation: List[Dict[str, str]],
# ) -> float:
#     """Check whether the agent communicated all required information.

#     Matches tau2-bench CommunicateEvaluator logic:
#     - Collect all assistant messages
#     - Join them, remove commas, lowercase
#     - For each required info string, check case-insensitive substring match

#     Returns 1.0 if all match (or no communicate_info), 0.0 if any missing.
#     """
#     if not communicate_info:
#         return 1.0

#     # Collect all assistant messages
#     assistant_texts = []
#     for msg in conversation:
#         if msg.get("role") == "assistant":
#             assistant_texts.append(msg.get("text", ""))

#     full_text = " ".join(assistant_texts)
#     # Remove commas and lowercase (matches tau2-bench behavior)
#     full_text_normalized = full_text.replace(",", "").lower()

#     for info in communicate_info:
#         info_normalized = str(info).replace(",", "").lower()
#         if info_normalized not in full_text_normalized:
#             return 0.0

#     return 1.0


# def compute_reward(
#     tool_calls: List[Dict[str, Any]],
#     ground_truth: GroundTruth,
#     db_final: Dict[str, Any],
#     transferred: bool,
#     conversation_ended_normally: bool,
#     conversation: Optional[List[Dict[str, str]]] = None,
# ) -> Tuple[float, str]:
#     """Compute reward for an episode.

#     Returns (reward, explanation) where reward is in [0.0, 1.0]:
#         1.0 = Agent followed policy correctly and communicated all info
#         0.0 = Agent violated policy

#     Reward = 0.8 * db_reward + 0.2 * communicate_reward.
#     Policy adherence (db_reward) is the dominant signal. COMMUNICATE
#     is additive so correct policy decisions are never zeroed out.
#     """
#     reasons = []

#     # Phase 1: Check forbidden actions (instant fail)
#     for forbidden in ground_truth.forbidden_actions:
#         for tc in tool_calls:
#             if _matches_forbidden(tc, forbidden, db_final):
#                 reason = f"FORBIDDEN: Called {tc['name']} matching forbidden rule: {forbidden.get('check', 'exact')}"
#                 return 0.0, reason

#     # Phase 2: Check transfer correctness
#     db_reward = None
#     if ground_truth.should_transfer and transferred:
#         # Correct transfer takes precedence over required_actions
#         db_reward = 1.0
#         reason = "Correctly transferred to human agent"
#     elif ground_truth.should_transfer and not transferred:
#         if not ground_truth.required_actions:
#             return 0.0, "Should have transferred to human agent but didn't"
#         # Has required_actions — fall through to Phase 3 as alternative
#     elif not ground_truth.should_transfer and transferred:
#         # Over-transfer: give partial credit based on useful work done.
#         # Creates GRPO gradient:
#         #   blind_transfer (0.0) < lookups (≤0.3) < req_actions+transfer (≤0.5) < correct (1.0)

#         # (a) Check if required actions were completed despite unnecessary transfer
#         req_credit = 0.0
#         matched_req = 0
#         total_req = 0
#         if ground_truth.required_actions:
#             total_req = len(ground_truth.required_actions)
#             matched_req = sum(1 for req in ground_truth.required_actions
#                               if _any_matches_required(tool_calls, req))
#             if matched_req > 0:
#                 # Cap at 0.5: correct action + unnecessary transfer still penalized
#                 req_credit = (matched_req / total_req) * 0.5

#         # (b) Check for information-gathering operations (unique tool names only —
#         #     prevents spam of same tool for inflated credit)
#         read_ops = {
#             # Airline read operations
#             "get_user_details", "get_reservation_details",
#             "search_direct_flight", "search_onestop_flight",
#             "list_all_airports", "get_flight_status",
#             # Retail read operations
#             "get_order_details", "get_product_details",
#             "find_user_id_by_email", "find_user_id_by_name_zip",
#             "list_all_product_types",
#         }
#         unique_lookups = len({tc["name"] for tc in tool_calls if tc.get("name") in read_ops})
#         lookup_credit = min(unique_lookups / 3.0, 1.0) * 0.3

#         # Take the higher of required-action credit and lookup credit
#         db_reward = max(req_credit, lookup_credit)

#         if db_reward > 0:
#             parts = []
#             if matched_req > 0:
#                 parts.append(f"req_actions={matched_req}/{total_req}")
#             if unique_lookups > 0:
#                 parts.append(f"unique_lookups={unique_lookups}")
#             reason = f"Over-transfer with partial credit ({', '.join(parts)}, db_reward={db_reward:.2f})"
#         else:
#             if not ground_truth.required_actions:
#                 return 0.0, "Unnecessarily transferred to human agent (no useful work)"
#             return 0.0, "Transferred instead of performing required actions (no useful work)"

#     # Phase 3: Check required actions (skip if transfer already rewarded)
#     if db_reward is None and ground_truth.required_actions:
#         matched = 0
#         total = len(ground_truth.required_actions)
#         for required in ground_truth.required_actions:
#             if _any_matches_required(tool_calls, required):
#                 matched += 1
#             else:
#                 reasons.append(f"Missing required action: {required.get('name', 'unknown')}")

#         if matched == total:
#             db_reward = 1.0
#             reason = "All required actions performed correctly"
#         elif matched > 0:
#             db_reward = matched / total
#             reason = f"Partial: {matched}/{total} required actions. " + "; ".join(reasons)
#         else:
#             db_reward = 0.0
#             reason = "No required actions matched. " + "; ".join(reasons)

#     # Phase 4: Check DB state
#     if db_reward is None and ground_truth.expected_db_state:
#         all_correct = True
#         for key, expected in ground_truth.expected_db_state.items():
#             actual = _get_nested(db_final, key)
#             if actual != expected:
#                 all_correct = False
#                 reasons.append(f"DB mismatch: {key} expected={expected}, got={actual}")

#         if all_correct:
#             db_reward = 1.0
#             reason = "DB state correct (no forbidden actions triggered)"
#         else:
#             return 0.0, "DB state incorrect. " + "; ".join(reasons)

#     # Phase 5: If no required actions and no forbidden triggered → correct refusal
#     if db_reward is None:
#         if not ground_truth.required_actions and not ground_truth.should_transfer:
#             write_ops = {"cancel_reservation", "update_reservation_flights",
#                          "update_reservation_baggages", "update_reservation_passengers",
#                          "cancel_pending_order",
#                          "modify_pending_order_items", "modify_pending_order_address",
#                          "modify_pending_order_payment",
#                          "return_delivered_order_items", "exchange_delivered_order_items",
#                          "modify_user_address", "book_reservation", "send_certificate"}
#             for tc in tool_calls:
#                 if tc["name"] in write_ops:
#                     return 0.0, f"Performed write operation {tc['name']} when none should occur"
#             db_reward = 1.0
#             reason = "Correctly refused/took no write actions"
#         else:
#             db_reward = 0.0
#             reason = "Indeterminate outcome"

#     # No comm bonus for total policy failure — prevents 0.2 free reward
#     # for models that violate policy but happen to communicate info.
#     # Comm bonus only matters when the model did at least some policy work.
#     if db_reward == 0.0:
#         return 0.0, reason

#     # Phase 6: Check communicate_info (additive, not multiplicative)
#     comm_reward = 1.0
#     if conversation is not None and hasattr(ground_truth, 'communicate_info') and ground_truth.communicate_info:
#         comm_reward = check_communicate_info(ground_truth.communicate_info, conversation)
#         if comm_reward < 1.0:
#             reason += f" | COMMUNICATE FAIL: missing required info from {ground_truth.communicate_info}"

#     # Additive weighting: policy adherence is 80% of the signal,
#     # communication is 20%. This prevents COMMUNICATE failures from
#     # zeroing out correct policy decisions.
#     final_reward = 0.8 * db_reward + 0.2 * comm_reward
#     return final_reward, reason


# def _matches_forbidden(
#     tool_call: Dict[str, Any],
#     forbidden: Dict[str, Any],
#     db_final: Dict[str, Any],
# ) -> bool:
#     """Check if a tool call matches a forbidden action pattern."""
#     tc_name = tool_call.get("name", "")
#     fb_name = forbidden.get("name", "")

#     if tc_name != fb_name:
#         return False

#     check_type = forbidden.get("check", "exact")
#     tc_args = tool_call.get("arguments", {})

#     if check_type == "exact":
#         fb_args = forbidden.get("arguments", {})
#         return all(str(tc_args.get(k, "")) == str(v) for k, v in fb_args.items())

#     elif check_type == "reservation_id_match":
#         return str(tc_args.get("reservation_id", "")) == str(forbidden.get("reservation_id", ""))

#     elif check_type == "any_write":
#         # Any call to this tool with matching entity ID
#         if "reservation_id" in forbidden:
#             return str(tc_args.get("reservation_id", "")) == str(forbidden["reservation_id"])
#         if "order_id" in forbidden:
#             oid = str(tc_args.get("order_id", ""))
#             if not oid.startswith("#"):
#                 oid = f"#{oid}"
#             foid = str(forbidden["order_id"])
#             if not foid.startswith("#"):
#                 foid = f"#{foid}"
#             return oid == foid
#         return True  # Any call to this tool is forbidden

#     elif check_type == "bag_removal":
#         new_bags = tc_args.get("total_baggages", 999)
#         try:
#             new_bags = int(new_bags)
#         except (TypeError, ValueError):
#             return False
#         return new_bags < forbidden.get("min_bags", 0)

#     elif check_type == "dest_change":
#         # Check if flights go to a different destination
#         # This is a simplified check — any update_reservation_flights call is forbidden
#         # in a destination-change scenario
#         return str(tc_args.get("reservation_id", "")) == str(forbidden.get("reservation_id", ""))

#     elif check_type == "wrong_address":
#         addr1 = str(tc_args.get("address1", "")).lower()
#         wrong = str(forbidden.get("wrong_address1", "")).lower()
#         return addr1 == wrong

#     elif check_type == "payment_method":
#         # Forbidden if payment method is not in the valid list
#         pay_id = str(tc_args.get("payment_method_id", ""))
#         valid = [str(x) for x in forbidden.get("valid_payment_ids", [])]
#         return pay_id not in valid

#     elif check_type == "fabricated_payment":
#         pay_id = str(tc_args.get("payment_method_id", ""))
#         known = [str(x) for x in forbidden.get("known_payment_ids", [])]
#         return pay_id not in known and pay_id != ""

#     return False


# def _any_matches_required(
#     tool_calls: List[Dict[str, Any]],
#     required: Dict[str, Any],
# ) -> bool:
#     """Check if any tool call matches a required action."""
#     req_name = required.get("name", "")
#     check_type = required.get("check", "exact")

#     for tc in tool_calls:
#         if tc.get("name", "") != req_name:
#             continue

#         tc_args = tc.get("arguments", {})

#         if check_type == "exact":
#             req_args = required.get("arguments", {})
#             if all(str(tc_args.get(k, "")) == str(v) for k, v in req_args.items()):
#                 return True

#         elif check_type == "cabin_upgrade":
#             if (str(tc_args.get("reservation_id", "")) == str(required.get("reservation_id", "")) and
#                 str(tc_args.get("cabin", "")) == str(required.get("new_cabin", ""))):
#                 return True

#         elif check_type == "any_valid":
#             return True

#         elif check_type == "order_match":
#             # Match by order_id with "#" prefix normalization
#             oid = str(tc_args.get("order_id", ""))
#             if not oid.startswith("#"):
#                 oid = f"#{oid}"
#             req_oid = str(required.get("order_id", ""))
#             if not req_oid.startswith("#"):
#                 req_oid = f"#{req_oid}"
#             if oid == req_oid:
#                 return True

#         elif check_type == "payment_method":
#             pay_id = str(tc_args.get("payment_method_id", ""))
#             valid = [str(x) for x in required.get("valid_payment_ids", [])]
#             if pay_id in valid:
#                 return True

#     return False


# def _get_nested(d: Dict, key: str) -> Any:
#     """Get nested value from dict using dot-separated key."""
#     parts = key.split(".")
#     current = d
#     for part in parts:
#         if isinstance(current, dict):
#             current = current.get(part)
#         else:
#             return None
#     return current
