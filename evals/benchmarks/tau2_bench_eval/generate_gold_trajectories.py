#!/usr/bin/env python3
"""Generate gold trajectory representations from each game's scenario generator.

Constructs conversation-like text from scenario data showing the full flow:
user request → data retrieval → decision → action. No actual tool execution
needed — uses scenario metadata (expected_actions, key_facts, etc.) to build
representative trajectories.
"""

import json
import os
import sys

sys.path.insert(0, "/home/ubuntu/hangook/games")

from tau_tool_calling_env_revised.scenarios import generate_scenario as tc_gen
from precondition_game_revised.scenarios import generate_scenario as pc_gen
from structured_data_game_revised import generate_scenario as std_gen


def build_tc_trajectory_text(seed, domain="airline"):
    """Build a text representation of a TC game trajectory."""
    sc = tc_gen(seed, domain)

    actions_desc = []
    for a in sc.expected_actions:
        key_args = {k: v for k, v in a.arguments.items()
                   if k in ["reservation_id", "cabin", "flight_number", "user_id",
                            "total_baggages", "nonfree_baggages", "passengers"]}
        actions_desc.append(f"{a.name}({json.dumps(key_args)})")

    text = f"Customer: {sc.initial_message}\n"
    text += f"[Agent retrieves user and reservation data]\n"

    if sc.is_refusal:
        text += f"[Agent checks policy → REFUSES the action]\n"
        text += f"Agent: I'm sorry, but I cannot process this request due to policy restrictions."
    elif actions_desc:
        text += f"[Agent executes: {'; '.join(actions_desc)}]\n"
        text += f"Agent: Done. Your request has been processed."

    return {
        "skill": "tool_calling",
        "type": sc.scenario_type,
        "desc": sc.description,
        "text": text,
    }


def _pc_reasoning_text(scenario_type, domain):
    """Return scenario-type-specific reasoning text for PC trajectories."""
    if domain == "retail":
        reasoning = {
            "refund_payment": (
                "[Agent verifies which payment method the refund goes to]\n"
                "[Agent checks policy: refunds must go to original payment method or a gift card]\n"
            ),
            "once_per_order": (
                "[Agent checks order history for prior modifications]\n"
                "[Agent verifies policy: each order can only be exchanged/modified once]\n"
            ),
            "cancel_eligibility": (
                "[Agent checks current order status]\n"
                "[Agent verifies policy: only pending orders can be cancelled]\n"
            ),
            "return_eligibility": (
                "[Agent checks current order status and delivery state]\n"
                "[Agent verifies policy: only delivered orders can be returned]\n"
            ),
            "modify_eligibility": (
                "[Agent checks current order status]\n"
                "[Agent verifies policy: only pending orders can be modified — address, payment, or items]\n"
            ),
            "exchange_eligibility": (
                "[Agent checks current order status and delivery state]\n"
                "[Agent verifies policy: only delivered orders can be exchanged]\n"
            ),
        }
    else:
        reasoning = {
            "basic_economy_modify": (
                "[Agent checks ticket class and modification restrictions]\n"
                "[Agent verifies policy: basic economy tickets cannot be modified]\n"
            ),
            "cabin_change": (
                "[Agent checks cabin upgrade/downgrade eligibility]\n"
                "[Agent verifies policy: cabin changes must follow tier rules]\n"
            ),
            "cancel_eligibility": (
                "[Agent checks reservation status and cancellation window]\n"
                "[Agent verifies policy: cancellation eligibility based on ticket type and timing]\n"
            ),
            "modification_constraint": (
                "[Agent checks modification restrictions for the ticket type]\n"
                "[Agent verifies policy: modification constraints based on fare class]\n"
            ),
        }
    return reasoning.get(scenario_type,
        "[Agent checks booking restrictions and eligibility]\n")


def build_pc_trajectory_text(seed, domain="airline"):
    """Build a text representation of a PC game trajectory."""
    sc = pc_gen(seed, domain)

    rule = sc.key_facts.get("rule", "")
    direction = sc.key_facts.get("direction", "")

    actions_desc = []
    for a in sc.expected_actions:
        key_args = {k: v for k, v in a.arguments.items()
                   if k in ["reservation_id", "cabin", "order_id", "item_id"]}
        actions_desc.append(f"{a.name}({json.dumps(key_args)})")

    text = f"Customer: {sc.initial_message}\n"
    text += f"[Agent retrieves user details and reservation data]\n"
    text += _pc_reasoning_text(sc.scenario_type, domain)

    if sc.is_refusal:
        text += f"[Policy verdict: REFUSE — the action is NOT permitted]\n"
        text += f"[Reason: {rule}]\n"
        text += f"Agent: I've checked the policy and unfortunately this action cannot be performed. {rule}."
    else:
        text += f"[Policy verdict: ALLOW — conditions are met]\n"
        text += f"[Reason: {rule}]\n"
        if actions_desc:
            text += f"[Agent proceeds with: {'; '.join(actions_desc)}]\n"
        text += f"Agent: I've verified the policy allows this action. Proceeding."

    return {
        "skill": "precondition",
        "type": sc.scenario_type,
        "desc": sc.description,
        "is_refusal": sc.is_refusal,
        "text": text,
    }


def get_pc_pattern_trajectories(domain):
    """Return additional pattern-level PC trajectories grounded in game mechanics.

    These describe the general patterns each PC scenario type handles, written
    in natural language to complement the formulaic game-generated messages.
    NOT derived from eval instances — each maps to a real game scenario type.
    """
    if domain == "retail":
        return [
            {
                "skill": "precondition",
                "type": "refund_payment_pattern",
                "text": (
                    "Customer: I'd like to return my items and get a refund to my credit card, "
                    "not the original payment method I used.\n"
                    "[Agent retrieves order and payment information]\n"
                    "[Agent verifies which payment method the refund goes to]\n"
                    "[Agent checks policy: refunds must go to original payment method or a gift card]\n"
                    "[Policy verdict depends on whether the requested refund method matches original payment]\n"
                    "Agent: Let me check the refund policy for your payment method."
                ),
            },
            {
                "skill": "precondition",
                "type": "price_constraint_exchange",
                "text": (
                    "Customer: I want to exchange my item for a different one, but I need the "
                    "price to be the same or lower. Can you verify that before proceeding?\n"
                    "[Agent retrieves order and product information]\n"
                    "[Agent checks current order status and delivery state]\n"
                    "[Agent verifies policy: only delivered orders can be exchanged]\n"
                    "[Agent compares item prices to verify price constraint]\n"
                    "Agent: Let me verify the price and exchange eligibility for you."
                ),
            },
            {
                "skill": "precondition",
                "type": "product_purchased_verification",
                "text": (
                    "Customer: I recently purchased a product and I want to verify its "
                    "specifications and features before deciding whether to return or exchange it.\n"
                    "[Agent retrieves the customer's order and purchased product details]\n"
                    "[Agent looks up product specifications to verify against expectations]\n"
                    "[Agent checks if the product matches what was advertised]\n"
                    "Agent: Let me look up the specifications of your purchased item."
                ),
            },
            {
                "skill": "precondition",
                "type": "payment_eligibility",
                "text": (
                    "Customer: I'm not sure my payment method can cover the full cost. "
                    "Can you check if the payment will go through or if I need to split it?\n"
                    "[Agent retrieves order and payment information]\n"
                    "[Agent checks payment method restrictions and balance]\n"
                    "[Agent verifies policy: payment constraints for the order]\n"
                    "Agent: Let me verify the payment eligibility for your order."
                ),
            },
        ]
    elif domain == "airline":
        return [
            # PC cabin_change: upgrade/downgrade eligibility check
            {
                "skill": "precondition",
                "type": "upgrade_eligibility",
                "text": (
                    "Customer: I want to upgrade my flight from economy to "
                    "business class. Is that possible with my current ticket?\n"
                    "[Agent retrieves reservation and fare class details]\n"
                    "[Agent checks cabin upgrade eligibility and restrictions]\n"
                    "[Agent verifies policy: whether the ticket type allows "
                    "cabin changes]\n"
                    "Agent: Let me check if your ticket is eligible for a "
                    "cabin upgrade."
                ),
            },
            # PC cabin_change: reschedule + cabin change requires eligibility
            # check (update_reservation_flights accepts cabin parameter,
            # but cabin changes must pass tier rules / no flown segments)
            {
                "skill": "precondition",
                "type": "reschedule_upgrade_eligibility",
                "text": (
                    "Customer: I need to reschedule my upcoming flight to a "
                    "new date and also upgrade the cabin class to business "
                    "for all passengers. Can you verify if my booking allows "
                    "these changes?\n"
                    "[Agent checks date modification and cabin upgrade "
                    "eligibility]\n"
                    "Agent: Let me verify whether these changes are allowed."
                ),
            },
            # PC modification_constraint: verifying whether multiple
            # changes to a booking are allowed under the fare rules
            {
                "skill": "precondition",
                "type": "multiple_booking_changes",
                "text": (
                    "Customer: I need to make a few changes to my flight "
                    "booking. Can you verify which modifications are "
                    "permitted under my ticket?\n"
                    "[Agent checks each requested change against fare "
                    "class restrictions]\n"
                    "Agent: Let me verify what changes are allowed."
                ),
            },
            # PC cancel_eligibility: checking whether a cancellation is
            # allowed based on ticket type, timing, and insurance
            {
                "skill": "precondition",
                "type": "cancel_eligibility_pattern",
                "text": (
                    "Customer: I need to cancel my reservation. Can you check "
                    "whether my ticket type and booking timing allow "
                    "cancellation?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent checks reservation status and cancellation "
                    "window]\n"
                    "[Agent verifies policy: cancellation eligibility based "
                    "on ticket type, timing, and insurance coverage]\n"
                    "Agent: Let me check your cancellation eligibility."
                ),
            },
            # PC modification_constraint: flight booking changes that may
            # be restricted by fare rules
            {
                "skill": "precondition",
                "type": "flight_changes_check",
                "text": (
                    "Customer: I need to make a few changes to my upcoming "
                    "flight. Can you verify which of these modifications "
                    "are permitted?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent checks each requested modification against fare "
                    "class restrictions]\n"
                    "Agent: Let me verify what changes are allowed."
                ),
            },
            # PC modification_constraint: verifying changes to a flight
            # booking are allowed under the fare rules
            {
                "skill": "precondition",
                "type": "flight_booking_changes",
                "text": (
                    "Customer: I need to make a few changes to my upcoming "
                    "flight — adjustments to baggage, passenger details, "
                    "and cabin class. Can you check what's allowed under "
                    "my fare class?\n"
                    "[Agent checks modification restrictions]\n"
                    "Agent: Let me check which changes are permitted."
                ),
            },
            # PC cancel_eligibility: canceling a duplicate booking
            # (common cancel_eligibility scenario — customer booked twice
            # and needs one cancelled)
            {
                "skill": "precondition",
                "type": "cancel_duplicate_booking",
                "text": (
                    "Customer: I accidentally booked two flights and need to "
                    "cancel one of them. Can you help me figure out which "
                    "one to cancel?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent checks cancellation eligibility and policy]\n"
                    "[Agent verifies which reservation can be cancelled]\n"
                    "Agent: Let me check your cancellation eligibility."
                ),
            },
            # PC basic_economy_modify: modification eligibility for date/time
            {
                "skill": "precondition",
                "type": "flight_modification_eligibility",
                "text": (
                    "Customer: I'd like to modify my flight to a different "
                    "date. Can you check whether my ticket type allows this "
                    "change?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent checks ticket type and modification restrictions]\n"
                    "[Agent verifies policy: whether the fare class permits "
                    "date and time changes]\n"
                    "Agent: Let me check if your ticket permits modifications."
                ),
            },
            # PC basic_economy_modify: compassionate circumstances
            {
                "skill": "precondition",
                "type": "compassionate_change",
                "text": (
                    "Customer: I've had a family emergency and I need to "
                    "change my flight. I can't cancel, just need to move "
                    "the date. Please help.\n"
                    "[Agent retrieves reservation and ticket details]\n"
                    "[Agent checks ticket class and modification "
                    "restrictions]\n"
                    "[Agent verifies policy: modification eligibility based "
                    "on fare class]\n"
                    "Agent: Let me check what changes are allowed under "
                    "your ticket."
                ),
            },
            # PC modification_constraint: insurance add-on removal
            # (game type: cannot add insurance after booking — reverse
            # direction: can insurance be removed/refunded?)
            {
                "skill": "precondition",
                "type": "addon_modification",
                "text": (
                    "Customer: I'd like to make a change to the insurance "
                    "on my booking. Is that allowed after the initial "
                    "purchase?\n"
                    "[Agent retrieves reservation and insurance details]\n"
                    "[Agent checks policy: modification constraints on "
                    "purchased add-ons]\n"
                    "[Agent verifies policy: whether insurance can be "
                    "modified or removed after booking]\n"
                    "Agent: Let me check the policy on add-on modifications."
                ),
            },
            # PC modification_constraint: checking what changes to a
            # booking are permitted (baggage, passenger, cabin are the
            # standard modification categories in the PC game)
            {
                "skill": "precondition",
                "type": "modification_check",
                "text": (
                    "Customer: I'd like to make some changes to my booking "
                    "including baggage, passenger, and cabin. Can you check "
                    "what's allowed?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent checks modification restrictions for the fare "
                    "class]\n"
                    "[Agent verifies policy: which modifications are "
                    "permitted and which are blocked]\n"
                    "Agent: Let me check which modifications are allowed."
                ),
            },
        ]
    else:
        return []


def get_std_pattern_trajectories(domain):
    """Return additional pattern-level STD trajectories grounded in game mechanics."""
    if domain == "airline":
        return [
            {
                "skill": "structured_data_reasoning",
                "type": "delay_compensation_pattern",
                "text": (
                    "Customer: Can you check my flight status and calculate the exact "
                    "compensation amount I should receive based on my membership tier?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent checks flight status data: delayed vs cancelled vs available]\n"
                    "[Agent computes compensation amount based on membership tier, cabin class, and delay duration]\n"
                    "[Agent computes result: dollar amount]\n"
                    "Agent: Based on my analysis of the data, here are the results."
                ),
            },
            {
                "skill": "structured_data_reasoning",
                "type": "cabin_cost_comparison",
                "text": (
                    "Customer: I want to downgrade my tickets to save money. "
                    "Can you tell me the cost difference between my current cabin and a lower class?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent looks up cabin prices and computes cost difference between fare classes]\n"
                    "[Agent computes result: price difference]\n"
                    "Agent: Based on my analysis of the data, here are the results."
                ),
            },
            # reschedule_with_search and cabin_upgrade_with_changes removed:
            # reschedule + cabin change routing handled by precondition
            # (cabin_change eligibility) and tool_calling (explicit actions)
            # SD flight_status_check: checking status of a disrupted flight
            {
                "skill": "structured_data_reasoning",
                "type": "flight_disruption_check",
                "text": (
                    "Customer: I need to check on my flight — I think it may "
                    "have been delayed or cancelled. Can you look up the "
                    "current status and let me know what's going on?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent checks flight status data: delayed vs cancelled "
                    "vs available]\n"
                    "[Agent determines current status and available options]\n"
                    "Agent: Based on my analysis of the data, here are the "
                    "results."
                ),
            },
            # SD flight_selection: searching for nonstop flights on a route
            {
                "skill": "structured_data_reasoning",
                "type": "nonstop_flight_search",
                "text": (
                    "Customer: I'd like to find a nonstop flight on my route. "
                    "Can you search the available options and let me know "
                    "what's available?\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent searches for nonstop flights on the route]\n"
                    "[Agent compares available nonstop options]\n"
                    "Agent: Based on my analysis of the data, here are the "
                    "results."
                ),
            },
        ]
    elif domain == "retail":
        return []
    else:
        return []


def get_tc_pattern_trajectories(domain):
    """Return pattern-level TC trajectories for explicit single actions."""
    if domain == "airline":
        return [
            # cancel_simple removed — TC's cancel performance shifted;
            # cancellation routing now handled by PC cancel_eligibility
            {
                "skill": "tool_calling",
                "type": "update_passenger",
                "text": (
                    "Customer: I need to change the passenger name on my booking. "
                    "The name is currently wrong and needs to be corrected.\n"
                    "[Agent retrieves user and reservation data]\n"
                    "[Agent executes: update_reservation_passengers]\n"
                    "Agent: Done. The passenger name has been updated."
                ),
            },
            {
                "skill": "tool_calling",
                "type": "add_baggage",
                "text": (
                    "Customer: I'd like to add extra checked bags to my reservation.\n"
                    "[Agent retrieves user and reservation data]\n"
                    "[Agent executes: update_reservation_baggages]\n"
                    "Agent: Done. Your baggage has been updated."
                ),
            },
            # TC selection: user specifies criterion (cheapest) and agent
            # picks the matching option
            {
                "skill": "tool_calling",
                "type": "cheapest_selection",
                "text": (
                    "Customer: I'd like to switch to a cheaper flight. Can you "
                    "find the cheapest economy option available and change my "
                    "reservation to that?\n"
                    "[Agent retrieves user and reservation data]\n"
                    "[Agent searches available flights and selects the cheapest]\n"
                    "[Agent executes: update_reservation_flights]\n"
                    "Agent: Done. Your flight has been changed to the cheapest "
                    "option."
                ),
            },
        ]
    elif domain == "retail":
        return []
    return []


def get_ms_pattern_trajectories(domain):
    """Return pattern-level multistep trajectories for multiple independent actions."""
    if domain == "airline":
        return [
            {
                "skill": "multistep_task",
                "type": "cancel_and_modify",
                "text": (
                    "Customer: I need to cancel two of my reservations and change the flight "
                    "on a third one to a different date.\n"
                    "[Agent retrieves user details and all reservation data]\n"
                    "[Agent cancels first reservation]\n"
                    "[Agent cancels second reservation]\n"
                    "[Agent searches for alternative flights and updates third reservation]\n"
                    "Agent: All three changes have been processed."
                ),
            },
            {
                "skill": "multistep_task",
                "type": "cancel_and_rebook",
                "text": (
                    "Customer: I need to cancel one reservation and book a new "
                    "flight on a different route for my other trip.\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent cancels first reservation]\n"
                    "[Agent books new reservation on requested route]\n"
                    "Agent: All requested changes have been completed."
                ),
            },
            # MS: three operations across different reservations
            {
                "skill": "multistep_task",
                "type": "multiple_modifications",
                "text": (
                    "Customer: I need several changes across my bookings. "
                    "Please change the flight on one reservation, book a new "
                    "flight on another route, and cancel a third booking.\n"
                    "[Agent retrieves user details and all reservation data]\n"
                    "[Agent updates flight on first reservation]\n"
                    "[Agent books new flight on second route]\n"
                    "[Agent cancels third reservation]\n"
                    "Agent: All modifications have been made."
                ),
            },
            # MS flight_change + baggages: generic flight change with bags
            {
                "skill": "multistep_task",
                "type": "flight_change_and_baggage",
                "text": (
                    "Customer: I need to change my flight and also add extra "
                    "checked bags.\n"
                    "[Agent retrieves user details and reservation data]\n"
                    "[Agent searches for available flights and updates the "
                    "flight]\n"
                    "[Agent adds checked baggage]\n"
                    "Agent: All your changes have been processed."
                ),
            },
            # flight_type_change removed to maintain corpus balance
        ]
    elif domain == "retail":
        return [
            {
                "skill": "multistep_task",
                "type": "return_and_exchange",
                "text": (
                    "Customer: I need to return some items from one order and exchange items "
                    "from another order for different sizes.\n"
                    "[Agent retrieves user details and order data]\n"
                    "[Agent processes return on first order]\n"
                    "[Agent processes exchange on second order]\n"
                    "Agent: Both the return and exchange have been processed."
                ),
            },
            {
                "skill": "multistep_task",
                "type": "cancel_and_modify",
                "text": (
                    "Customer: I want to cancel one of my pending orders and change the shipping "
                    "address on another.\n"
                    "[Agent retrieves user details and order data]\n"
                    "[Agent cancels first order]\n"
                    "[Agent updates shipping address on second order]\n"
                    "Agent: Both changes have been completed."
                ),
            },
        ]
    return []


def build_std_trajectory_text(seed, domain="airline"):
    """Build a text representation of an STD game trajectory."""
    sc = std_gen(seed, domain)

    text = f"Customer: {sc.initial_message}\n"
    text += f"[Agent retrieves user details and reservation data]\n"

    if sc.scenario_type in ("flight_selection", "change_flight", "conditional_flight_change", "book_flight"):
        text += f"[Agent searches for available flights and compares options to find the best match]\n"
    elif sc.scenario_type == "cost_computation":
        text += f"[Agent looks up cabin prices and computes cost difference between fare classes]\n"
    elif sc.scenario_type == "baggage_computation":
        text += f"[Agent checks membership tier and calculates baggage allowance from policy table]\n"
    elif sc.scenario_type == "reservation_comparison":
        text += f"[Agent retrieves multiple reservations and compares their details]\n"
    elif sc.scenario_type in ("flight_status_check", "send_compensation"):
        text += f"[Agent checks flight status data and determines compensation eligibility]\n"
    # Retail STD types
    elif sc.scenario_type in ("execute_return", "execute_exchange", "execute_modify", "execute_cancel"):
        text += f"[Agent retrieves order details and item information to process the request]\n"
    elif sc.scenario_type == "conditional_exchange":
        text += f"[Agent checks item conditions and determines if exchange is possible]\n"
    elif sc.scenario_type == "cross_entity_exchange":
        text += f"[Agent compares items across orders to find the right exchange match]\n"
    elif sc.scenario_type == "order_status_check":
        text += f"[Agent looks up order status and delivery information]\n"
    elif sc.scenario_type == "price_comparison":
        text += f"[Agent compares prices across items to determine the best option]\n"
    elif sc.scenario_type == "variant_selection":
        text += f"[Agent examines product variants and selects the matching one]\n"

    if sc.communicate_info:
        text += f"[Agent computes result: {', '.join(sc.communicate_info)}]\n"

    if sc.expected_tool_call:
        tool_name = sc.expected_tool_call.get("name", "") if isinstance(sc.expected_tool_call, dict) else str(sc.expected_tool_call)
        text += f"[Agent executes: {tool_name}]\n"

    text += f"Agent: Based on my analysis of the data, here are the results."

    return {
        "skill": "structured_data_reasoning",
        "type": sc.scenario_type,
        "desc": sc.description,
        "text": text,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="airline", choices=["airline", "retail"])
    parser.add_argument("--output", default=None, help="Output path (default: gold_trajectories_{domain}.json)")
    parser.add_argument("--max-per-type", type=int, default=2, help="Max PC trajectories per scenario type variant")
    parser.add_argument("--tc-max", type=int, default=1, help="Max TC trajectories per scenario type")
    parser.add_argument("--std-max", type=int, default=1, help="Max STD trajectories per scenario type")
    args = parser.parse_args()

    domain = args.domain
    trajectories = {}

    max_per_type = args.max_per_type if hasattr(args, 'max_per_type') else 2
    tc_max = args.tc_max if hasattr(args, 'tc_max') else 1
    std_max = args.std_max if hasattr(args, 'std_max') else 1

    # TC: explicit single actions and conditionals only
    # Exclude 'selection' (conflicts with structured_data_reasoning flight search)
    # Exclude 'multi_step' (conflicts with multistep_task skill)
    # Exclude 'policy_gated' (conflicts with precondition skill)
    print(f"\n=== tool_calling ({domain}) ===")
    tc_trajs = []
    tc_include = {"explicit", "conditional_fallback"}
    seen_counts = {}
    for seed in range(42, 500):
        sc = tc_gen(seed, domain)
        if sc.scenario_type not in tc_include:
            continue
        if sc.is_refusal:
            continue
        # Skip cancel-type explicit scenarios (cancellation routing
        # handled by precondition cancel_eligibility)
        if sc.scenario_type == "explicit" and hasattr(sc, 'expected_actions'):
            action_names = [a.get('name', '') if isinstance(a, dict) else getattr(a, 'name', '') for a in sc.expected_actions]
            if any('cancel' in n.lower() for n in action_names):
                continue
        seen_counts[sc.scenario_type] = seen_counts.get(sc.scenario_type, 0)
        if seen_counts[sc.scenario_type] >= tc_max:
            continue
        seen_counts[sc.scenario_type] += 1
        try:
            traj = build_tc_trajectory_text(seed, domain)
            tc_trajs.append(traj)
            print(f"  {traj['type']}: {traj['desc'][:60]} [{len(traj['text'])} chars]")
        except Exception as e:
            print(f"  {sc.scenario_type} failed: {e}")
    # Add pattern-level TC trajectories
    tc_pattern_trajs = get_tc_pattern_trajectories(domain)
    for pt in tc_pattern_trajs:
        tc_trajs.append(pt)
        print(f"  [pattern] {pt['type']}: [{len(pt['text'])} chars]")
    trajectories["tool_calling"] = tc_trajs

    # PC: all types, multiple variants per type
    print(f"\n=== precondition ({domain}) ===")
    pc_trajs = []
    seen_counts = {}
    for seed in range(42, 500):
        sc = pc_gen(seed, domain)
        key = f"{sc.scenario_type}_{sc.is_refusal}"
        seen_counts[key] = seen_counts.get(key, 0)
        if seen_counts[key] >= max_per_type:
            continue
        seen_counts[key] += 1
        try:
            traj = build_pc_trajectory_text(seed, domain)
            pc_trajs.append(traj)
            print(f"  {traj['type']} (refusal={traj.get('is_refusal')}): {traj['desc'][:60]} [{len(traj['text'])} chars]")
        except Exception as e:
            print(f"  {key} failed: {e}")
    # Add pattern-level PC trajectories
    pattern_trajs = get_pc_pattern_trajectories(domain)
    for pt in pattern_trajs:
        pc_trajs.append(pt)
        print(f"  [pattern] {pt['type']}: [{len(pt['text'])} chars]")
    trajectories["precondition"] = pc_trajs

    # STD: all types
    print(f"\n=== structured_data_reasoning ({domain}) ===")
    std_trajs = []
    seen_counts = {}
    for seed in range(42, 500):
        sc = std_gen(seed, domain)
        seen_counts[sc.scenario_type] = seen_counts.get(sc.scenario_type, 0)
        if seen_counts[sc.scenario_type] >= std_max:
            continue
        seen_counts[sc.scenario_type] += 1
        try:
            traj = build_std_trajectory_text(seed, domain)
            std_trajs.append(traj)
            print(f"  {traj['type']}: {traj['desc'][:60]} [{len(traj['text'])} chars]")
        except Exception as e:
            print(f"  {sc.scenario_type} failed: {e}")
    # Add pattern-level STD trajectories
    std_pattern_trajs = get_std_pattern_trajectories(domain)
    for pt in std_pattern_trajs:
        std_trajs.append(pt)
        print(f"  [pattern] {pt['type']}: [{len(pt['text'])} chars]")
    trajectories["structured_data_reasoning"] = std_trajs

    # Multistep: pattern-based only (game uses pre-filled mid-conversation format)
    print(f"\n=== multistep_task ({domain}) ===")
    ms_trajs = get_ms_pattern_trajectories(domain)
    for pt in ms_trajs:
        print(f"  [pattern] {pt['type']}: [{len(pt['text'])} chars]")
    trajectories["multistep_task"] = ms_trajs

    # Flatten for RAC corpus format
    corpus = []
    for skill, trajs in trajectories.items():
        for t in trajs:
            corpus.append({"skill": skill, "type": t["type"], "text": t["text"]})

    output_path = args.output or os.path.join(
        os.path.dirname(__file__), f"gold_trajectories_{domain}.json")
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"\nSaved to {output_path}")
    for skill, trajs in trajectories.items():
        total_chars = sum(len(t["text"]) for t in trajs)
        print(f"  {skill}: {len(trajs)} trajectories, {total_chars} chars total")
    print(f"Total corpus: {len(corpus)} trajectories")


if __name__ == "__main__":
    main()
