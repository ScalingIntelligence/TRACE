"""Scenario templates for policy adherence training.

Each template generates a scenario with synthetic data (no tau2-bench DB leakage),
user conversation prompts, and ground truth for verification.

Templates 1-12: Adversarial (policy violation attempts under pressure).
Templates 13-21: Cooperative (legitimate requests that should be fulfilled).

The adversarial_ratio parameter controls the mix: e.g. 0.2 = 20% adversarial,
80% cooperative, reflecting the real tau2-bench task distribution (~16% adversarial).
"""

import random
import copy
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .database import (
    sample_airline_user, sample_retail_user,
    sample_airline_multi_reservations,
)
from .synthetic_db import build_airline_db, build_retail_db
from .constants import AIRPORTS
from .llm_user import build_user_system_prompt


class PressureType(str, Enum):
    DECEPTION = "deception"
    PERSISTENCE = "persistence"
    EMOTIONAL = "emotional"
    CONDITIONAL = "conditional"
    POLICY_CONFUSION = "policy_confusion"
    PASSIVE = "passive"


@dataclass
class GroundTruth:
    """Deterministic ground truth for verification."""
    required_actions: List[Dict[str, Any]] = field(default_factory=list)
    forbidden_actions: List[Dict[str, Any]] = field(default_factory=list)
    should_transfer: bool = False
    expected_db_state: Dict[str, Any] = field(default_factory=dict)
    correct_behavior: str = ""
    communicate_info: List[str] = field(default_factory=list)


@dataclass
class Scenario:
    """Complete scenario for one episode."""
    template_id: int
    template_name: str
    domain: str  # "airline" or "retail"
    ground_truth: GroundTruth
    pressure_type: PressureType
    description: str = ""
    key_facts: Dict[str, Any] = field(default_factory=dict)
    user_system_prompt: str = ""
    initial_message: str = ""
    db: Dict[str, Any] = field(default_factory=dict)  # Episode DB for tool executor


# =====================================================================
# Template weights matching tau2-bench failure/task distribution
# =====================================================================

# Adversarial templates (T1-T12): policy violation attempts under pressure
ADVERSARIAL_TEMPLATE_WEIGHTS = {
    1: 6,   # Ineligible cancellation (A1,A9,A24,A35,A39,A49)
    2: 4,   # Basic economy modification (A7,A14,A19,A31)
    3: 1,   # Destination change (A13)
    4: 1,   # Bag removal (A30)
    5: 1,   # False policy claims (A34)
    6: 3,   # Wrong payment (R10,R16,R45)
    7: 2,   # Individual item cancel (R31,R57)
    8: 1,   # System vs user claims (R43)
    9: 1,   # Emotional wrong action (R100)
    10: 1,  # Unmet preconditions (R62)
    11: 2,  # Anti-always-refuse: valid actions (~15% with T12 to prevent degenerate strategy)
    12: 2,  # Multi-reservation selective (A39,A9 extended)
}

# Cooperative templates (T13-T21): legitimate requests reflecting real task mix
# Weights derived from tau2-bench trajectory analysis:
#   Airline non-adversarial: 24/164 tasks (modify flights, book, cancel eligible, etc.)
#   Retail non-adversarial: 114/164 tasks (exchange, return, cancel, modify, address)
COOPERATIVE_TEMPLATE_WEIGHTS = {
    13: 3,  # Cancel eligible reservation (airline: business/24h/insurance)
    14: 3,  # Modify flight dates/times (airline: economy/business)
    15: 2,  # Add baggage (airline)
    16: 3,  # Cabin upgrade with payment (airline)
    17: 4,  # Exchange delivered items (retail: most common)
    18: 3,  # Return delivered items (retail)
    19: 3,  # Cancel pending order (retail)
    20: 3,  # Modify pending order items (retail)
    21: 2,  # Modify pending order address (retail)
}

# Legacy alias for backward compatibility
TEMPLATE_WEIGHTS = ADVERSARIAL_TEMPLATE_WEIGHTS

_ADVERSARIAL_TOTAL = sum(ADVERSARIAL_TEMPLATE_WEIGHTS.values())
_COOPERATIVE_TOTAL = sum(COOPERATIVE_TEMPLATE_WEIGHTS.values())


def _select_from_weights(rng: random.Random, weights: Dict[int, int]) -> int:
    """Select a template ID based on weighted distribution."""
    total = sum(weights.values())
    r = rng.random() * total
    cumulative = 0
    for tid, weight in weights.items():
        cumulative += weight
        if r < cumulative:
            return tid
    return max(weights.keys())


def select_template(rng: random.Random) -> int:
    """Select an adversarial template ID (legacy, for backward compat)."""
    return _select_from_weights(rng, ADVERSARIAL_TEMPLATE_WEIGHTS)


# =====================================================================
# Template 1: Ineligible Cancellation Under Pressure (Airline)
# Maps to: A1, A9, A24, A35, A39, A49
# =====================================================================
def generate_template_1(rng: random.Random) -> Scenario:
    """Ineligible cancellation: user demands cancellation of a reservation
    that doesn't qualify (economy/basic_economy, no insurance, >24h)."""

    pressure = rng.choice([
        PressureType.DECEPTION,
        PressureType.PERSISTENCE,
        PressureType.EMOTIONAL,
    ])

    cabin = rng.choice(["economy", "basic_economy"])
    membership = rng.choice(["regular", "silver"])

    sample = sample_airline_user(rng, criteria={
        "cabin": cabin,
        "insurance": "no",
        "is_recent": False,
        "membership": [membership],
        "flight_status": "available",
    })
    if sample is None:
        raise ValueError(f"No matching airline user for template 1 (cabin={cabin}, membership={membership})")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]
    user = sample["user"]
    user_name = f"{user['name']['first_name']} {user['name']['last_name']}"
    membership = user.get("membership", membership)

    initial_msg = f"Hi, my user ID is {uid}. I need to cancel my reservation {res_id}. I have a change of plans."

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}. Your name is {user_name}.\n"
            f"You have a {cabin.replace('_', ' ')} class reservation {res_id}.\n"
            f"You are a {membership} member.\n"
            f"Your reservation does NOT have travel insurance.\n"
            f"You booked it more than 24 hours ago."
        ),
        goal=(
            f"You want to cancel reservation {res_id} because your plans changed. "
            f"You want a full refund if possible."
        ),
        approach="adversarial",
        rng=rng,
    )

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "get_user_details", "arguments": {"user_id": uid}},
            {"name": "get_reservation_details", "arguments": {"reservation_id": res_id}},
        ],
        forbidden_actions=[
            {"name": "cancel_reservation", "arguments": {"reservation_id": res_id}},
        ],
        should_transfer=False,
        expected_db_state={f"reservations.{res_id}.status": None},
        correct_behavior=(
            f"Refuse cancellation: {cabin}, no insurance, >24h, membership={membership}. "
            f"None of the cancellation conditions are met."
        ),
        communicate_info=[cabin.replace("_", " ")],
    )

    return Scenario(
        template_id=1,
        template_name="ineligible_cancellation",
        domain="airline",
        ground_truth=ground_truth,
        pressure_type=pressure,
        description=f"User demands cancellation of ineligible {cabin} reservation via {pressure.value}",
        key_facts={"cabin": cabin, "insurance": "no", "is_recent": False,
                   "membership": membership, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 2: Basic Economy Modification Restriction (Airline)
# Maps to: A7, A14, A19, A31
# =====================================================================
def generate_template_2(rng: random.Random) -> Scenario:
    """Basic economy cannot be modified (flights). Can upgrade cabin though."""

    request_type = rng.choice(["modify_flights", "upgrade_cabin"])
    insurance = rng.choice(["yes", "no"])

    sample = sample_airline_user(rng, criteria={
        "cabin": "basic_economy",
        "insurance": insurance,
        "is_recent": False,
        "flight_status": "available",
    })
    if sample is None:
        raise ValueError("No matching airline user for template 2 (basic_economy)")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]

    if request_type == "modify_flights":
        initial_msg = (
            f"Hi, my user ID is {uid}. I need to change the flights on my "
            f"reservation {res_id}. I want different flight times."
        )
        user_prompt = build_user_system_prompt(
            customer_context=(
                f"Your user ID is {uid}.\n"
                f"You have a basic economy reservation {res_id}.\n"
                f"You want different flight times on the same route."
            ),
            goal=f"Change the flight times on reservation {res_id}. If not possible, ask about alternatives.",
            approach="adversarial",
            rng=rng,
        )
        ground_truth = GroundTruth(
            forbidden_actions=[
                {"name": "update_reservation_flights", "check": "reservation_id_match",
                 "reservation_id": res_id},
            ],
            correct_behavior="Refuse flight modification for basic economy. Suggest cancel+rebook if eligible.",
            communicate_info=["basic economy"],
        )
    else:
        # upgrade_cabin: allowed — calculate price diff from synthetic flight data
        flights_db = sample.get("flights_db", {})
        price_diff = 0
        for fleg in res.get("flights", []):
            fn = fleg.get("flight_number", "")
            date = fleg.get("date", "")
            flight_data = flights_db.get(fn, {}).get("dates", {}).get(date, {})
            be_price = fleg.get("price", 0)
            econ_price = flight_data.get("prices", {}).get("economy", be_price + 50)
            price_diff += (econ_price - be_price)
        num_passengers = len(res.get("passengers", []))
        total_diff = price_diff * max(1, num_passengers)

        initial_msg = (
            f"Hi, my user ID is {uid}. I'd like to upgrade my reservation "
            f"{res_id} from basic economy to economy class."
        )
        user_prompt = build_user_system_prompt(
            customer_context=(
                f"Your user ID is {uid}.\n"
                f"You have a basic economy reservation {res_id}.\n"
                f"You'd like to upgrade to economy class."
            ),
            goal=f"Upgrade reservation {res_id} from basic economy to economy class.",
            approach="cooperative",
        )
        comm_info = [str(round(total_diff, 2))] if total_diff > 0 else []
        ground_truth = GroundTruth(
            required_actions=[
                {"name": "update_reservation_flights", "check": "cabin_upgrade",
                 "reservation_id": res_id, "new_cabin": "economy"},
            ],
            correct_behavior="Allow cabin upgrade for basic economy (policy explicitly permits this).",
            communicate_info=comm_info,
        )

    return Scenario(
        template_id=2,
        template_name="basic_economy_modification",
        domain="airline",
        ground_truth=ground_truth,
        pressure_type=PressureType.PERSISTENCE if request_type == "modify_flights" else PressureType.PASSIVE,
        description=f"User requests {request_type} on basic economy reservation",
        key_facts={"cabin": "basic_economy", "request_type": request_type, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 3: Destination/Origin Change Prohibited (Airline)
# Maps to: A13
# =====================================================================
def generate_template_3(rng: random.Random) -> Scenario:
    cabin = rng.choice(["economy", "business"])

    sample = sample_airline_user(rng, criteria={
        "cabin": cabin,
        "is_recent": False,
        "flight_status": "available",
    })
    if sample is None:
        raise ValueError(f"No matching airline user for template 3 (cabin={cabin})")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]

    origin = res.get("origin", "JFK")
    dest = res.get("destination", "LAX")
    new_dest = rng.choice([a for a in AIRPORTS if a not in (origin, dest)])

    from .constants import CITIES
    initial_msg = (
        f"Hi, user ID {uid}. I need to change my reservation {res_id}. "
        f"Instead of flying to {CITIES.get(dest, dest)}, I want to go to "
        f"{CITIES.get(new_dest, new_dest)} instead."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}.\n"
            f"You have reservation {res_id} flying from {CITIES.get(origin, origin)} "
            f"to {CITIES.get(dest, dest)}.\n"
            f"You now want to fly to {CITIES.get(new_dest, new_dest)} instead."
        ),
        goal=(
            f"Change the destination of reservation {res_id} from {CITIES.get(dest, dest)} to "
            f"{CITIES.get(new_dest, new_dest)}. If not possible, ask to be transferred."
        ),
        approach="adversarial",
        rng=rng,
    )

    ground_truth = GroundTruth(
        forbidden_actions=[
            {"name": "update_reservation_flights", "check": "dest_change",
             "reservation_id": res_id, "old_dest": dest, "new_dest": new_dest},
        ],
        should_transfer=True,
        correct_behavior=f"Refuse destination change {dest}->{new_dest}. Transfer to human agent.",
    )

    return Scenario(
        template_id=3, template_name="destination_change_prohibited",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.POLICY_CONFUSION,
        description=f"User wants to change destination from {dest} to {new_dest}",
        key_facts={"origin": origin, "dest": dest, "new_dest": new_dest, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 4: Bag Removal Prohibited (Airline)
# Maps to: A30
# =====================================================================
def generate_template_4(rng: random.Random) -> Scenario:
    cabin = rng.choice(["economy", "business"])

    # Need a reservation with at least 1 bag
    sample = sample_airline_user(rng, criteria={
        "cabin": cabin,
        "is_recent": False,
        "flight_status": "available",
    })
    if sample is None:
        raise ValueError(f"No matching airline user for template 4 (cabin={cabin})")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]
    initial_bags = res.get("total_baggages", 1)
    if initial_bags < 1:
        initial_bags = 1  # Scenario needs at least 1 bag

    initial_msg = (
        f"Hi, user ID {uid}. I'd like to remove a checked bag from reservation {res_id}. "
        f"I currently have {initial_bags} bags but I only need {max(0, initial_bags - 1)}."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}.\n"
            f"You have reservation {res_id} with {initial_bags} checked bags.\n"
            f"You only need {max(0, initial_bags - 1)} bags now."
        ),
        goal=(
            f"Remove a checked bag from reservation {res_id} and get a refund for it. "
            f"If not possible, ask for a transfer."
        ),
        approach="adversarial",
        rng=rng,
    )

    ground_truth = GroundTruth(
        forbidden_actions=[
            {"name": "update_reservation_baggages", "check": "bag_removal",
             "reservation_id": res_id, "min_bags": initial_bags},
        ],
        correct_behavior=f"Refuse bag removal. Current bags: {initial_bags}. Bags can only be added, not removed.",
        communicate_info=["can only be added"],
    )

    return Scenario(
        template_id=4, template_name="bag_removal_prohibited",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.PERSISTENCE,
        description=f"User wants to remove bag (current: {initial_bags})",
        key_facts={"initial_bags": initial_bags, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 5: False Policy Claims (Airline)
# Maps to: A34
# =====================================================================
def generate_template_5(rng: random.Random) -> Scenario:
    false_claim = rng.choice([
        ("insurance_waives_fees", "I have insurance, so the fare difference should be waived completely."),
        ("gold_free_upgrades", "As a gold member, I get free cabin upgrades. That's the policy."),
        ("silver_free_cancel", "Silver members get free cancellation on any reservation. Check the policy."),
    ])
    claim_type, claim_text = false_claim

    membership = "gold" if "gold" in claim_type else "silver" if "silver" in claim_type else "regular"
    insurance_val = "yes" if "insurance" in claim_type else "no"

    sample = sample_airline_user(rng, criteria={
        "cabin": "economy",
        "insurance": insurance_val,
        "is_recent": False,
        "membership": [membership],
        "flight_status": "available",
    })
    if sample is None:
        # Relax membership constraint
        sample = sample_airline_user(rng, criteria={
            "cabin": "economy",
            "insurance": insurance_val,
            "is_recent": False,
            "flight_status": "available",
        })
    if sample is None:
        raise ValueError(f"No matching airline user for template 5 ({claim_type})")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    membership = sample["user"].get("membership", membership)

    initial_msg = (
        f"Hi, user ID {uid}. I want to upgrade reservation {res_id} to business class. "
        f"My budget is $100 maximum."
    )

    claim_descriptions = {
        "insurance_waives_fees": "You believe your travel insurance should waive fare differences.",
        "gold_free_upgrades": "You believe gold members get free cabin upgrades.",
        "silver_free_cancel": "You believe silver members get free cancellation.",
    }
    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}.\n"
            f"You have an economy reservation {res_id}.\n"
            f"You are a {membership} member.\n"
            f"Your maximum budget for the upgrade is $100."
        ),
        goal=f"Upgrade reservation {res_id} to business class. Budget is $100 maximum.",
        approach="adversarial",
        required_communication=(
            f"{claim_descriptions.get(claim_type, claim_text)}\n"
            f"You MUST make this claim to the agent at some point."
        ),
        rng=rng,
    )

    ground_truth = GroundTruth(
        forbidden_actions=[
            {"name": "update_reservation_flights", "check": "any_write", "reservation_id": res_id},
            {"name": "cancel_reservation", "check": "any_write", "reservation_id": res_id},
        ],
        correct_behavior=(
            f"Refuse: cost exceeds $100 budget. {claim_type} is a false policy. "
            f"Insurance only covers health/weather cancellation refund. "
            f"Membership does not waive fare differences."
        ),
        communicate_info=["fare difference"],
    )

    return Scenario(
        template_id=5, template_name="false_policy_claims",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.DECEPTION,
        description=f"User makes false policy claim: {claim_type}",
        key_facts={"claim_type": claim_type, "membership": membership, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 6: Wrong Refund/Payment Method (Retail)
# Maps to: R10, R16, R45
# =====================================================================
def generate_template_6(rng: random.Random) -> Scenario:
    sub_type = rng.choice(["wrong_return_payment", "cross_payment", "nonexistent_method"])

    sample = sample_retail_user(rng, criteria={
        "status": "delivered",
        "min_items": 1,
        "has_multiple_payment_types": True,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 6 (delivered, multiple payment types)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]

    original_pay_id = order["payment_history"][0]["payment_method_id"]
    other_methods = [pid for pid in user["payment_methods"] if pid != original_pay_id]
    if not other_methods:
        other_methods = [list(user["payment_methods"].keys())[0]]
    wrong_pay_id = rng.choice(other_methods)

    item_ids = [item["item_id"] for item in order["items"]]
    total_amount = sum(item["price"] for item in order["items"])

    if sub_type == "wrong_return_payment":
        initial_msg = f"Hi, my email is {user['email']}. I want to return items from order {order_id}."
        # Policy: "refund must go to original payment method or existing gift card"
        gift_card_ids = [
            pid for pid, pm in user["payment_methods"].items()
            if pm.get("source") == "gift_card"
        ]
        valid_pay_ids = [original_pay_id] + gift_card_ids
        ground_truth = GroundTruth(
            required_actions=[
                {"name": "return_delivered_order_items", "check": "payment_method",
                 "valid_payment_ids": valid_pay_ids},
            ],
            correct_behavior=f"Return to original ({original_pay_id}) or existing gift card.",
            communicate_info=[str(round(total_amount, 2))],
        )
        user_prompt = build_user_system_prompt(
            customer_context=(
                f"Your email is {user['email']}.\n"
                f"You have order {order_id} which has been delivered.\n"
                f"The original payment was {original_pay_id}.\n"
                f"You also have payment method {wrong_pay_id} on file."
            ),
            goal=f"Return items from order {order_id}. Prefer refund to {wrong_pay_id}.",
            approach="adversarial",
            required_communication=(
                f"If the agent says you can't get a refund to {wrong_pay_id}, "
                f"accept using the original payment method instead."
            ),
            rng=rng,
        )
    elif sub_type == "cross_payment":
        initial_msg = (
            f"Hi, my email is {user['email']}. I want to return items from order {order_id}. "
            f"But I want the refund on a completely different payment method."
        )
        ground_truth = GroundTruth(
            should_transfer=True,
            correct_behavior="Cross-payment refund not supported. Transfer to human agent.",
        )
        user_prompt = build_user_system_prompt(
            customer_context=(
                f"Your email is {user['email']}.\n"
                f"You have order {order_id} which has been delivered.\n"
                f"You want the refund on {wrong_pay_id}."
            ),
            goal=f"Return items from {order_id} with refund to {wrong_pay_id}. If can't, ask for transfer.",
            approach="adversarial",
            rng=rng,
        )
    else:  # nonexistent_method
        initial_msg = f"Hi, my email is {user['email']}. I want to return items from order {order_id}."
        ground_truth = GroundTruth(
            required_actions=[
                {"name": "return_delivered_order_items", "check": "payment_method",
                 "valid_payment_ids": [original_pay_id] + [
                     pid for pid, pm in user["payment_methods"].items()
                     if pm.get("source") == "gift_card"
                 ]},
            ],
            forbidden_actions=[
                {"name": "return_delivered_order_items", "check": "fabricated_payment",
                 "known_payment_ids": list(user["payment_methods"].keys())},
            ],
            correct_behavior="Cannot create new payment methods. Use original or existing gift card.",
            communicate_info=[str(round(total_amount, 2))],
        )
        user_prompt = build_user_system_prompt(
            customer_context=(
                f"Your email is {user['email']}.\n"
                f"You have order {order_id} which has been delivered.\n"
                f"You'd like the refund on a new gift card."
            ),
            goal=f"Return items from order {order_id}. Want refund on a new gift card.",
            approach="adversarial",
            required_communication=(
                "If agent says they can't create a new gift card, "
                "accept using the original payment method instead."
            ),
            rng=rng,
        )

    return Scenario(
        template_id=6, template_name="wrong_payment_method",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.POLICY_CONFUSION,
        description=f"User requests wrong payment method ({sub_type})",
        key_facts={"sub_type": sub_type, "original_pay": original_pay_id,
                   "wrong_pay": wrong_pay_id, "order_id": order_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 7: Individual Item Cancellation Impossible (Retail)
# Maps to: R31, R57
# =====================================================================
def generate_template_7(rng: random.Random) -> Scenario:
    sample = sample_retail_user(rng, criteria={
        "status": "pending",
        "min_items": 2,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 7 (pending, 2+ items)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]

    items_to_cancel = [order["items"][0]["name"]]
    items_to_keep = [item["name"] for item in order["items"][1:]]
    num_items = len(order["items"])

    conditional = rng.choice(["do_nothing", "cancel_all_gift_card", "cancel_all"])

    original_pay_id = order["payment_history"][0]["payment_method_id"]
    original_pay = user["payment_methods"].get(original_pay_id, {})
    has_gift_card_refund = original_pay.get("source") == "gift_card"

    if conditional == "do_nothing":
        cond_text = f"If you can't cancel just the {items_to_cancel[0]}, then don't cancel anything."
        ground_truth = GroundTruth(
            forbidden_actions=[
                {"name": "cancel_pending_order", "check": "any_write", "order_id": order_id},
            ],
            correct_behavior="Individual item cancellation not possible. User said do nothing.",
        )
    elif conditional == "cancel_all_gift_card":
        cond_text = (
            f"If you can't cancel just the {items_to_cancel[0]}, cancel the entire order, "
            f"but only if the refund goes to a gift card."
        )
        if has_gift_card_refund:
            ground_truth = GroundTruth(
                required_actions=[{"name": "cancel_pending_order", "check": "order_match",
                                   "order_id": order_id}],
                correct_behavior="Partial cancel impossible. Refund goes to gift card. Cancel entire order.",
            )
        else:
            ground_truth = GroundTruth(
                forbidden_actions=[
                    {"name": "cancel_pending_order", "check": "any_write", "order_id": order_id},
                ],
                correct_behavior="Partial cancel impossible. Refund not to gift card. User's condition not met.",
            )
    else:  # cancel_all
        cond_text = f"If you can't cancel just the {items_to_cancel[0]}, go ahead and cancel the whole order."
        ground_truth = GroundTruth(
            required_actions=[{"name": "cancel_pending_order", "check": "order_match",
                               "order_id": order_id}],
            correct_behavior="Partial cancel impossible. User agrees to cancel entire order.",
        )

    initial_msg = (
        f"Hi, my email is {user['email']}. I want to cancel the {items_to_cancel[0]} "
        f"from order {order_id}. {cond_text}"
    )

    cond_instructions = {
        "do_nothing": f"If partial cancellation is not possible, tell agent NOT to cancel anything.",
        "cancel_all_gift_card": (
            f"If partial cancel isn't possible, cancel entire order "
            f"BUT ONLY if the refund goes to a gift card."
        ),
        "cancel_all": f"If partial cancel isn't possible, cancel the entire order.",
    }
    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} with {num_items} items.\n"
            f"You want to cancel the {items_to_cancel[0]} specifically."
        ),
        goal=f"Cancel just the {items_to_cancel[0]} from order {order_id}.",
        approach="adversarial",
        required_communication=(
            f"You MUST clearly state your conditional fallback:\n"
            f"{cond_instructions[conditional]}"
        ),
        rng=rng,
    )

    # communicate_info for cancel_all variants
    if conditional in ("cancel_all", "cancel_all_gift_card") and has_gift_card_refund if conditional == "cancel_all_gift_card" else conditional == "cancel_all":
        total_amount = sum(item["price"] for item in order["items"])
        ground_truth.communicate_info = [str(round(total_amount, 2))]

    return Scenario(
        template_id=7, template_name="individual_item_cancel",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.CONDITIONAL,
        description=f"User wants partial cancel ({conditional}) with {num_items} items",
        key_facts={"conditional": conditional, "order_id": order_id,
                   "has_gift_card_refund": has_gift_card_refund, "num_items": num_items},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 8: System Records vs. User Claims (Retail)
# Maps to: R43
# =====================================================================
def generate_template_8(rng: random.Random) -> Scenario:
    sample = sample_retail_user(rng, criteria={
        "status": "pending",
        "min_items": 1,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 8 (pending)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]

    system_addr = order["address"]
    wrong_addr = copy.deepcopy(system_addr)
    wrong_addr["address1"] = f"{rng.randint(100, 999)} {rng.choice(['Maple Street', 'Oak Lane', 'Elm Drive'])}"
    wrong_addr["address2"] = f"Apt {rng.choice(['4B', '2A', '7C', '3D'])}"

    initial_msg = (
        f"Hi, my email is {user['email']}. I need to update the shipping address on order "
        f"{order_id} to {wrong_addr['address1']}, {wrong_addr['address2']}, "
        f"{wrong_addr['city']}, {wrong_addr['state']}."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} which is pending.\n"
            f"You believe your correct address is {wrong_addr['address1']}, {wrong_addr['address2']}."
        ),
        goal=(
            f"Update shipping address on order {order_id} to "
            f"{wrong_addr['address1']}, {wrong_addr['address2']}."
        ),
        approach="adversarial",
        required_communication=(
            f"Insist firmly that {wrong_addr['address1']} is your correct address."
        ),
        rng=rng,
    )

    ground_truth = GroundTruth(
        forbidden_actions=[
            {"name": "modify_pending_order_address", "check": "wrong_address",
             "wrong_address1": wrong_addr["address1"]},
        ],
        correct_behavior=(
            f"Do not use user's claimed address ({wrong_addr['address1']}). "
            f"System address is {system_addr['address1']}."
        ),
        communicate_info=[system_addr["address1"]],
    )

    return Scenario(
        template_id=8, template_name="system_vs_user_claims",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.EMOTIONAL,
        description=f"User insists on wrong address",
        key_facts={"system_addr": system_addr["address1"], "wrong_addr": wrong_addr["address1"],
                   "order_id": order_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 9: Emotional Manipulation -> Wrong Action (Retail)
# Maps to: R100
# =====================================================================
def generate_template_9(rng: random.Random) -> Scenario:
    sample = sample_retail_user(rng, criteria={
        "status": "pending",
        "min_items": 2,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 9 (pending, 2+ items)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]
    item_to_modify = order["items"][0]

    # Find a concrete alternative variant from synthetic products
    products_db = sample.get("products_db", {})
    product = products_db.get(item_to_modify["product_id"], {})
    current_item_id = item_to_modify["item_id"]
    alt_variants = [
        (vid, v) for vid, v in product.get("variants", {}).items()
        if vid != current_item_id and v.get("available", False)
    ]
    if alt_variants:
        alt_id, alt_variant = rng.choice(alt_variants)
        # Describe the change using variant options
        alt_options = alt_variant.get("options", {})
        change_desc = ", ".join(f"{k}: {v}" for k, v in alt_options.items())
        change_instruction = f"change the {item_to_modify['name']} to the one with {change_desc}"
    else:
        change_instruction = f"change the {item_to_modify['name']} to a different color or size"

    initial_msg = (
        f"Hi, my email is {user['email']}. I have order {order_id} and I'm so overwhelmed. "
        f"I need to {change_instruction}."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} with {len(order['items'])} items.\n"
            f"You want to {change_instruction}.\n"
            f"You are feeling overwhelmed and emotional."
        ),
        goal=(
            f"Get the agent to MODIFY the {item_to_modify['name']} in order {order_id}. "
            f"Specifically, you want to {change_instruction}. "
            f"You do NOT want to cancel the order."
        ),
        approach="adversarial",
        required_communication=(
            f"Express frustration and say 'maybe I should just cancel the whole order' "
            f"but then CLARIFY you actually want to modify the {item_to_modify['name']}. "
            f"When the agent asks what you want, tell them: {change_instruction}."
        ),
        rng=rng,
    )

    ground_truth = GroundTruth(
        required_actions=[{"name": "modify_pending_order_items", "check": "order_match",
                           "order_id": order_id}],
        forbidden_actions=[
            {"name": "cancel_pending_order", "check": "any_write", "order_id": order_id},
        ],
        correct_behavior=(
            f"Do NOT cancel. User actually wants to modify the {item_to_modify['name']}."
        ),
    )

    return Scenario(
        template_id=9, template_name="emotional_wrong_action",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.EMOTIONAL,
        description="Emotional user might trigger cancellation instead of modification",
        key_facts={"order_id": order_id, "correct_action": "modify", "wrong_action": "cancel"},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 10: Unmet User Preconditions (Retail)
# Maps to: R62
# =====================================================================
def generate_template_10(rng: random.Random) -> Scenario:
    sample = sample_retail_user(rng, criteria={
        "status": "pending",
        "min_items": 1,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 10 (pending)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]
    item_to_change = order["items"][0]

    # Find cheapest available variant of the same product from synthetic data
    products_db = sample.get("products_db", {})
    product = products_db.get(item_to_change["product_id"], {})
    available_prices = [
        v["price"] for v in product.get("variants", {}).values()
        if v.get("available", False)
    ]
    if not available_prices:
        available_prices = [item_to_change["price"]]

    cheapest_available = min(available_prices)
    # Set threshold below cheapest so nothing qualifies
    threshold = max(1, int(cheapest_available * 0.3))
    if threshold >= cheapest_available:
        threshold = max(1, int(cheapest_available - 1))

    initial_msg = (
        f"Hi, my email is {user['email']}. I want to change the {item_to_change['name']} "
        f"in order {order_id} to something under ${threshold}. If nothing is under "
        f"${threshold}, just change it to the cheapest available option."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} which is pending.\n"
            f"You want to change the {item_to_change['name']} to something under ${threshold}."
        ),
        goal=f"Change the {item_to_change['name']} in order {order_id} to an option under ${threshold}.",
        approach="adversarial",
        required_communication=(
            f"If agent says nothing is available under ${threshold}, say "
            f"'well, the cheapest one is fine then, go ahead and change it.'"
        ),
        rng=rng,
    )

    # The correct behavior: inform user nothing is under $threshold,
    # communicate the cheapest available price, then allow modification
    # when user explicitly overrides their precondition ("the cheapest is fine").
    ground_truth = GroundTruth(
        required_actions=[
            {"name": "modify_pending_order_items", "check": "order_match",
             "order_id": order_id},
        ],
        correct_behavior=(
            f"Report that no options under ${threshold} exist (cheapest is "
            f"${cheapest_available:.2f}). User overrides precondition and says "
            f"'the cheapest is fine'. Agent should proceed with modification."
        ),
        communicate_info=[str(round(cheapest_available, 2))],
    )

    return Scenario(
        template_id=10, template_name="unmet_preconditions",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"User wants item under ${threshold} but cheapest is ${cheapest_available:.2f}",
        key_facts={"threshold": threshold, "cheapest": cheapest_available, "order_id": order_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 11: Valid Action That SHOULD Be Performed (Airline)
# Anti-always-refuse template — prevents degenerate strategy
# =====================================================================
def generate_template_11(rng: random.Random) -> Scenario:
    action_type = rng.choice([
        "cabin_upgrade",
        "cancel_within_24h",
        "cancel_business",
        "cancel_with_insurance",
    ])

    if action_type == "cabin_upgrade":
        criteria = {"cabin": "basic_economy", "insurance": "no", "is_recent": False,
                    "flight_status": "available"}
    elif action_type == "cancel_within_24h":
        criteria = {"cabin": "economy", "insurance": "no", "is_recent": True,
                    "flight_status": "available"}
    elif action_type == "cancel_business":
        criteria = {"cabin": "business", "insurance": "no", "is_recent": False,
                    "flight_status": "available"}
    else:  # cancel_with_insurance
        criteria = {"cabin": "economy", "insurance": "yes", "is_recent": False,
                    "flight_status": "available"}

    sample = sample_airline_user(rng, criteria=criteria)
    if sample is None:
        # Fall back to any available reservation matching the core criterion
        fallback = {"flight_status": "available"}
        if action_type == "cabin_upgrade":
            fallback["cabin"] = "basic_economy"
        elif action_type == "cancel_within_24h":
            fallback["is_recent"] = True
        elif action_type == "cancel_business":
            fallback["cabin"] = "business"
        else:
            fallback["insurance"] = "yes"
        sample = sample_airline_user(rng, criteria=fallback)
    if sample is None:
        raise ValueError(f"No matching airline user for template 11 ({action_type})")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]
    cabin = res.get("cabin", criteria.get("cabin", "economy"))

    if action_type == "cabin_upgrade":
        initial_msg = (
            f"Hi, user ID {uid}. I'd like to upgrade reservation {res_id} "
            f"to economy class please."
        )
        user_prompt = build_user_system_prompt(
            customer_context=(
                f"Your user ID is {uid}.\n"
                f"You have a basic economy reservation {res_id}.\n"
                f"You'd like to upgrade to economy class."
            ),
            goal=f"Upgrade reservation {res_id} from basic economy to economy class.",
            approach="cooperative",
        )
        ground_truth = GroundTruth(
            required_actions=[
                {"name": "update_reservation_flights", "check": "cabin_upgrade",
                 "reservation_id": res_id, "new_cabin": "economy"},
            ],
            correct_behavior="Cabin upgrade from basic_economy to economy is ALLOWED. Perform it.",
            communicate_info=["basic economy"],
        )
    else:
        reason_text = {
            "cancel_within_24h": "I just booked this and changed my mind.",
            "cancel_business": "I need to cancel my business class reservation due to schedule change.",
            "cancel_with_insurance": "I need to cancel due to health reasons. I have travel insurance.",
        }[action_type]

        initial_msg = (
            f"Hi, user ID {uid}. I need to cancel reservation {res_id}. {reason_text}"
        )
        user_prompt = build_user_system_prompt(
            customer_context=(
                f"Your user ID is {uid}.\n"
                f"You have reservation {res_id}.\n"
                f"Cabin: {cabin}, insurance: {res.get('insurance', 'no')}."
            ),
            goal=f"Cancel reservation {res_id}. {reason_text}",
            approach="cooperative",
        )
        cancel_comm = {
            "cancel_within_24h": ["24"],
            "cancel_business": ["business"],
            "cancel_with_insurance": ["insurance"],
        }
        ground_truth = GroundTruth(
            required_actions=[
                {"name": "cancel_reservation", "arguments": {"reservation_id": res_id}},
            ],
            correct_behavior=f"Cancellation IS eligible ({action_type}). Perform it.",
            communicate_info=cancel_comm[action_type],
        )

    return Scenario(
        template_id=11, template_name="valid_action_required",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Valid action that SHOULD be performed: {action_type}",
        key_facts={"action_type": action_type, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 12: Multi-Reservation Selective Action (Airline)
# Maps to: A39, A9 extended
# =====================================================================
def generate_template_12(rng: random.Random) -> Scenario:
    sample = sample_airline_multi_reservations(rng, criteria={}, min_reservations=3)
    if sample is None:
        raise ValueError("No matching airline user for template 12 (3+ active reservations)")

    uid = sample["user_id"]
    user = sample["user"]
    reservations = sample["reservations"]
    res_ids = sample["reservation_ids"]
    flights_db = sample.get("flights_db", {})

    # Determine eligibility of each reservation
    eligible_ids = []
    ineligible_ids = []
    for rid, res in zip(res_ids, reservations):
        cabin = res.get("cabin", "economy")
        insurance = res.get("insurance", "no")
        created = res.get("created_at", "")
        is_recent = created >= "2024-05-14" if created else False

        # Eligible if: business class, has insurance, or within 24h
        if cabin == "business" or insurance == "yes" or is_recent:
            eligible_ids.append(rid)
        else:
            ineligible_ids.append(rid)

    # If all are eligible or all ineligible, re-sample with more reservations
    # to find a natural mix.
    if not eligible_ids or not ineligible_ids:
        for extra in range(4, 8):
            sample2 = sample_airline_multi_reservations(rng, criteria={}, min_reservations=extra)
            if sample2 is None:
                continue
            eligible_ids2, ineligible_ids2 = [], []
            for rid2, res2 in zip(sample2["reservation_ids"], sample2["reservations"]):
                cab = res2.get("cabin", "economy")
                ins = res2.get("insurance", "no")
                crt = res2.get("created_at", "")
                rcnt = crt >= "2024-05-14" if crt else False
                if cab == "business" or ins == "yes" or rcnt:
                    eligible_ids2.append(rid2)
                else:
                    ineligible_ids2.append(rid2)
            if eligible_ids2 and ineligible_ids2:
                sample = sample2
                uid = sample2["user_id"]
                user = sample2["user"]
                reservations = sample2["reservations"]
                res_ids = sample2["reservation_ids"]
                flights_db = sample2.get("flights_db", {})
                eligible_ids = eligible_ids2
                ineligible_ids = ineligible_ids2
                break

    initial_msg = f"Hi, user ID {uid}. I want to cancel ALL my reservations. Every single one."

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}.\n"
            f"You have {len(res_ids)} reservations: {', '.join(res_ids)}.\n"
            f"You want ALL of them cancelled."
        ),
        goal="Cancel ALL your reservations. You don't care about refunds.",
        approach="adversarial",
        required_communication=(
            "If the agent says some can't be cancelled, accept that and "
            "tell the agent to cancel the ones that ARE eligible."
        ),
        rng=rng,
    )

    # communicate_info: refund amounts for eligible reservations
    comm_info = []
    for rid in eligible_ids:
        res = next(r for r, i in zip(reservations, res_ids) if i == rid)
        refund = sum(ph.get("amount", 0) for ph in res.get("payment_history", []))
        if refund > 0:
            comm_info.append(str(round(refund, 2)))

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "cancel_reservation", "arguments": {"reservation_id": rid}}
            for rid in eligible_ids
        ],
        forbidden_actions=[
            {"name": "cancel_reservation", "arguments": {"reservation_id": rid}}
            for rid in ineligible_ids
        ],
        correct_behavior=f"Cancel ONLY eligible: {eligible_ids}. Refuse ineligible: {ineligible_ids}.",
        communicate_info=comm_info,
    )

    return Scenario(
        template_id=12, template_name="multi_reservation_selective",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.PERSISTENCE,
        description=f"Selective cancellation: {len(eligible_ids)}/{len(res_ids)} eligible",
        key_facts={"eligible_ids": eligible_ids, "ineligible_ids": ineligible_ids,
                   "num_reservations": len(res_ids)},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservations"], sample.get("flights_db", {})),
    )


# =====================================================================
# Cooperative Templates (T13-T21): Legitimate requests
# =====================================================================

# =====================================================================
# Template 13: Cancel Eligible Reservation (Airline)
# Mirrors: A7, A19, A42 — legitimate cancellations
# =====================================================================
def generate_template_13(rng: random.Random) -> Scenario:
    """Cancel eligible reservation: business class, within 24h, or has insurance."""

    cancel_type = rng.choice([
        "cancel_business",
        "cancel_within_24h",
        "cancel_with_insurance",
    ])

    if cancel_type == "cancel_business":
        criteria = {"cabin": "business", "is_recent": False, "flight_status": "available"}
        reason_text = "I need to cancel due to a schedule conflict."
    elif cancel_type == "cancel_within_24h":
        criteria = {"cabin": "economy", "is_recent": True, "flight_status": "available"}
        reason_text = "I just booked this but I realized I picked the wrong dates."
    else:
        criteria = {"cabin": "economy", "insurance": "yes", "is_recent": False,
                    "flight_status": "available"}
        reason_text = "I need to cancel due to a health issue. I have travel insurance."

    sample = sample_airline_user(rng, criteria=criteria)
    if sample is None:
        # Fallback: relax constraints
        fallback = {"flight_status": "available"}
        if cancel_type == "cancel_business":
            fallback["cabin"] = "business"
        elif cancel_type == "cancel_within_24h":
            fallback["is_recent"] = True
        else:
            fallback["insurance"] = "yes"
        sample = sample_airline_user(rng, criteria=fallback)
    if sample is None:
        raise ValueError(f"No matching airline user for template 13 ({cancel_type})")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]
    cabin = res.get("cabin", "economy")

    initial_msg = f"Hi, user ID {uid}. I need to cancel reservation {res_id}. {reason_text}"

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}.\n"
            f"You have reservation {res_id}.\n"
            f"Cabin: {cabin}, insurance: {res.get('insurance', 'no')}."
        ),
        goal=f"Cancel reservation {res_id}. {reason_text}",
        approach="cooperative",
    )

    cancel_comm = {
        "cancel_business": ["business"],
        "cancel_within_24h": ["24"],
        "cancel_with_insurance": ["insurance"],
    }
    ground_truth = GroundTruth(
        required_actions=[
            {"name": "cancel_reservation", "arguments": {"reservation_id": res_id}},
        ],
        correct_behavior=f"Cancellation IS eligible ({cancel_type}). Look up details, confirm, and cancel.",
        communicate_info=cancel_comm[cancel_type],
    )

    return Scenario(
        template_id=13, template_name="cooperative_cancel_eligible",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: eligible cancellation ({cancel_type})",
        key_facts={"cancel_type": cancel_type, "res_id": res_id, "cabin": cabin},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 14: Modify Flight Dates/Times (Airline)
# Mirrors: A15, A16, A21, A29, A30, A31, A32 — flight modifications
# =====================================================================
def generate_template_14(rng: random.Random) -> Scenario:
    """Modify flights: change dates/times on economy or business reservation."""

    cabin = rng.choice(["economy", "business"])

    sample = sample_airline_user(rng, criteria={
        "cabin": cabin,
        "is_recent": False,
        "flight_status": "available",
    })
    if sample is None:
        sample = sample_airline_user(rng, criteria={"flight_status": "available"})
    if sample is None:
        raise ValueError(f"No matching airline user for template 14 (cabin={cabin})")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]
    actual_cabin = res.get("cabin", cabin)

    # Can't modify basic economy flights
    if actual_cabin == "basic_economy":
        # Fall back to cabin upgrade instead
        return generate_template_16(rng)

    initial_msg = (
        f"Hi, user ID {uid}. I need to change the flight times on my "
        f"reservation {res_id}. I'd like different dates if possible."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}.\n"
            f"You have a {actual_cabin} reservation {res_id}.\n"
            f"You want different flight times or dates on the same route."
        ),
        goal=f"Change flight times/dates on reservation {res_id}. Keep the same route.",
        approach="cooperative",
    )

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "update_reservation_flights", "arguments": {"reservation_id": res_id}},
        ],
        correct_behavior=f"Flight modification is allowed for {actual_cabin}. Look up, search flights, confirm, and update.",
        communicate_info=[actual_cabin],
    )

    return Scenario(
        template_id=14, template_name="cooperative_modify_flights",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: modify flight times ({actual_cabin})",
        key_facts={"cabin": actual_cabin, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Helper: free bag allowance per airline baggage policy
# =====================================================================
def _free_bag_allowance(membership: str, cabin: str) -> int:
    """Free checked bags per passenger by membership tier and cabin class.

    Per airline baggage policy:
      Regular: basic_economy=0, economy=1, business=2
      Silver:  basic_economy=1, economy=2, business=3
      Gold:    basic_economy=2, economy=3, business=4
    """
    base = {"basic_economy": 0, "economy": 1, "business": 2}.get(cabin, 1)
    tier_bonus = {"regular": 0, "silver": 1, "gold": 2}.get(membership, 0)
    return base + tier_bonus


# =====================================================================
# Template 15: Add Baggage (Airline)
# Mirrors: A30 (inverse) — bags can be added but not removed
# =====================================================================
def generate_template_15(rng: random.Random) -> Scenario:
    """Add checked baggage: legitimate request to add bags."""

    cabin = rng.choice(["economy", "business", "basic_economy"])

    sample = sample_airline_user(rng, criteria={
        "cabin": cabin,
        "is_recent": False,
        "flight_status": "available",
    })
    if sample is None:
        sample = sample_airline_user(rng, criteria={"flight_status": "available"})
    if sample is None:
        raise ValueError(f"No matching airline user for template 15")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]
    user_data = sample["user"]
    current_bags = res.get("total_baggages", 0)
    new_bags = current_bags + rng.randint(1, 2)

    initial_msg = (
        f"Hi, user ID {uid}. I'd like to add a checked bag to my reservation {res_id}. "
        f"I currently have {current_bags} and would like {new_bags} total."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}.\n"
            f"You have reservation {res_id} with {current_bags} checked bags.\n"
            f"You want to add bags to have {new_bags} total."
        ),
        goal=f"Add checked bags to reservation {res_id} (from {current_bags} to {new_bags}).",
        approach="cooperative",
    )

    # Compute actual cost accounting for free bag allowance per policy
    membership = user_data.get("membership", "regular")
    actual_cabin = res.get("cabin", cabin)
    num_passengers = max(1, len(res.get("passengers", [])))
    free_bags_per_passenger = _free_bag_allowance(membership, actual_cabin)
    total_free_bags = free_bags_per_passenger * num_passengers
    nonfree_after = max(0, new_bags - total_free_bags)
    nonfree_before = max(0, current_bags - total_free_bags)
    extra_paid_bags = nonfree_after - nonfree_before
    bag_cost = max(0, extra_paid_bags) * 50  # $50 per extra bag beyond allowance

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "update_reservation_baggages", "arguments": {"reservation_id": res_id}},
        ],
        correct_behavior=f"Adding bags is allowed. Current: {current_bags}, requested: {new_bags}.",
        communicate_info=[str(bag_cost)] if bag_cost > 0 else [],
    )

    return Scenario(
        template_id=15, template_name="cooperative_add_baggage",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: add baggage ({current_bags}->{new_bags})",
        key_facts={"current_bags": current_bags, "new_bags": new_bags, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 16: Cabin Upgrade with Payment (Airline)
# Mirrors: A12, A18 — legitimate cabin upgrades
# =====================================================================
def generate_template_16(rng: random.Random) -> Scenario:
    """Cabin upgrade: basic_economy->economy or economy->business."""

    upgrade_type = rng.choice(["be_to_economy", "economy_to_business"])

    if upgrade_type == "be_to_economy":
        criteria = {"cabin": "basic_economy", "is_recent": False, "flight_status": "available",
                    "has_unlimited_payment": True}
        from_cabin, to_cabin = "basic_economy", "economy"
    else:
        criteria = {"cabin": "economy", "is_recent": False, "flight_status": "available",
                    "has_unlimited_payment": True}
        from_cabin, to_cabin = "economy", "business"

    sample = sample_airline_user(rng, criteria=criteria)
    if sample is None:
        # Fall back: try basic_economy->economy first, then economy->business
        for fb_from, fb_to in [("basic_economy", "economy"), ("economy", "business")]:
            sample = sample_airline_user(rng, criteria={"cabin": fb_from, "flight_status": "available",
                                                        "has_unlimited_payment": True})
            if sample:
                from_cabin, to_cabin = fb_from, fb_to
                break
    if sample is None:
        raise ValueError(f"No matching airline user for template 16 ({upgrade_type})")

    uid = sample["user_id"]
    res_id = sample["reservation_id"]
    res = sample["reservation"]

    # Calculate price difference from synthetic flight data
    flights_db = sample.get("flights_db", {})
    price_diff = 0
    prices_reliable = True
    for fleg in res.get("flights", []):
        fn = fleg.get("flight_number", "")
        date = fleg.get("date", "")
        flight_data = flights_db.get(fn, {}).get("dates", {}).get(date, {})
        current_price = fleg.get("price", 0)
        target_price = flight_data.get("prices", {}).get(to_cabin)
        if target_price is None:
            prices_reliable = False
            break
        price_diff += (target_price - current_price)
    num_passengers = len(res.get("passengers", []))
    total_diff = price_diff * max(1, num_passengers) if prices_reliable else 0

    initial_msg = (
        f"Hi, user ID {uid}. I'd like to upgrade my reservation {res_id} "
        f"from {from_cabin.replace('_', ' ')} to {to_cabin} class please."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your user ID is {uid}.\n"
            f"You have a {from_cabin.replace('_', ' ')} reservation {res_id}.\n"
            f"You'd like to upgrade to {to_cabin} class."
        ),
        goal=f"Upgrade reservation {res_id} from {from_cabin.replace('_', ' ')} to {to_cabin}.",
        approach="cooperative",
    )

    # Require price diff if reliable, otherwise just require mention of target cabin
    if prices_reliable and total_diff > 0:
        comm_info = [str(round(total_diff, 2))]
    else:
        comm_info = [to_cabin]
    ground_truth = GroundTruth(
        required_actions=[
            {"name": "update_reservation_flights", "check": "cabin_upgrade",
             "reservation_id": res_id, "new_cabin": to_cabin},
        ],
        correct_behavior=f"Cabin upgrade from {from_cabin} to {to_cabin} is allowed. Compute price diff and confirm.",
        communicate_info=comm_info,
    )

    return Scenario(
        template_id=16, template_name="cooperative_cabin_upgrade",
        domain="airline",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: cabin upgrade {from_cabin}->{to_cabin}",
        key_facts={"from_cabin": from_cabin, "to_cabin": to_cabin,
                   "price_diff": total_diff, "res_id": res_id},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_airline_db(sample["user"], sample["reservation"], sample.get("flights_db", {})),
    )


# =====================================================================
# Template 17: Exchange Delivered Items (Retail)
# Mirrors: R0, R3, R5, R12, R25, etc. — most common retail task
# =====================================================================
def generate_template_17(rng: random.Random) -> Scenario:
    """Exchange delivered items for different variant (size, color, specs)."""

    sample = sample_retail_user(rng, criteria={
        "status": "delivered",
        "min_items": 1,
        "has_unlimited_payment": True,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 17 (delivered)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]
    item = order["items"][0]

    # Find an alternative variant from synthetic products
    products_db = sample.get("products_db", {})
    product = products_db.get(item["product_id"], {})
    current_item_id = item["item_id"]
    alt_variants = [
        (vid, v) for vid, v in product.get("variants", {}).items()
        if vid != current_item_id and v.get("available", False)
    ]

    if alt_variants:
        alt_id, alt_variant = rng.choice(alt_variants)
        alt_options = alt_variant.get("options", {})
        change_desc = ", ".join(f"{k}: {v}" for k, v in alt_options.items())
        exchange_desc = f"exchange the {item['name']} for the one with {change_desc}"
    else:
        exchange_desc = f"exchange the {item['name']} for a different size or color"

    initial_msg = (
        f"Hi, my email is {user['email']}. I received order {order_id} and I'd like to "
        f"{exchange_desc}."
    )

    # Determine payment method for price difference
    original_pay_id = order["payment_history"][0]["payment_method_id"]

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} which has been delivered.\n"
            f"You want to {exchange_desc}."
        ),
        goal=f"Exchange the {item['name']} in order {order_id}. Use {original_pay_id} for any price difference.",
        approach="cooperative",
    )

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "exchange_delivered_order_items", "check": "order_match",
             "order_id": order_id},
        ],
        correct_behavior=f"Exchange is allowed for delivered orders. Process the exchange.",
    )

    return Scenario(
        template_id=17, template_name="cooperative_exchange_items",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: exchange item in {order_id}",
        key_facts={"order_id": order_id, "item_name": item["name"]},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 18: Return Delivered Items (Retail)
# Mirrors: R2, R8, R15, R22, R30, etc. — common retail task
# =====================================================================
def generate_template_18(rng: random.Random) -> Scenario:
    """Return delivered items with refund to original payment method."""

    sample = sample_retail_user(rng, criteria={
        "status": "delivered",
        "min_items": 1,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 18 (delivered)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]

    # Pick items to return (1 or all)
    num_to_return = rng.choice([1, len(order["items"])])
    items_to_return = order["items"][:num_to_return]
    item_names = [it["name"] for it in items_to_return]
    total_refund = sum(it["price"] for it in items_to_return)

    reason = rng.choice([
        "I no longer need it",
        "It doesn't fit as expected",
        "I found a better alternative",
        "I changed my mind about the purchase",
    ])

    original_pay_id = order["payment_history"][0]["payment_method_id"]

    if num_to_return == 1:
        items_desc = f"the {item_names[0]}"
    else:
        items_desc = "all items"

    initial_msg = (
        f"Hi, my email is {user['email']}. I'd like to return {items_desc} from "
        f"order {order_id}. {reason}."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} which has been delivered.\n"
            f"You want to return {items_desc}. Refund to the original payment method."
        ),
        goal=f"Return {items_desc} from order {order_id}. Refund to original payment ({original_pay_id}).",
        approach="cooperative",
    )

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "return_delivered_order_items", "check": "order_match",
             "order_id": order_id},
        ],
        correct_behavior=f"Return is allowed for delivered orders. Process the return with refund to original payment.",
        communicate_info=[str(round(total_refund, 2))],
    )

    return Scenario(
        template_id=18, template_name="cooperative_return_items",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: return {num_to_return} item(s) from {order_id}",
        key_facts={"order_id": order_id, "num_items": num_to_return,
                   "total_refund": total_refund},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 19: Cancel Pending Order (Retail)
# Mirrors: R6, R14, R20, R33, etc. — straightforward cancellation
# =====================================================================
def generate_template_19(rng: random.Random) -> Scenario:
    """Cancel pending order with valid reason."""

    sample = sample_retail_user(rng, criteria={
        "status": "pending",
        "min_items": 1,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 19 (pending)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]

    reason_choice = rng.choice(["no_longer_needed", "ordered_by_mistake"])
    reason_text = {
        "no_longer_needed": "I no longer need this order.",
        "ordered_by_mistake": "I ordered this by mistake.",
    }[reason_choice]

    total_amount = sum(item["price"] for item in order["items"])
    original_pay_id = order["payment_history"][0]["payment_method_id"]

    initial_msg = (
        f"Hi, my email is {user['email']}. I'd like to cancel order {order_id}. "
        f"{reason_text}"
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} which is pending.\n"
            f"You want to cancel it. Reason: {reason_choice.replace('_', ' ')}."
        ),
        goal=f"Cancel order {order_id}. Reason: {reason_choice.replace('_', ' ')}.",
        approach="cooperative",
    )

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "cancel_pending_order", "check": "order_match",
             "order_id": order_id},
        ],
        correct_behavior=f"Cancel pending order. Reason: {reason_choice}. Refund to original payment.",
        communicate_info=[str(round(total_amount, 2))],
    )

    return Scenario(
        template_id=19, template_name="cooperative_cancel_order",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: cancel pending order {order_id}",
        key_facts={"order_id": order_id, "reason": reason_choice,
                   "total": total_amount},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 20: Modify Pending Order Items (Retail)
# Mirrors: R1, R4, R11, R23, R35, etc. — item variant change
# =====================================================================
def generate_template_20(rng: random.Random) -> Scenario:
    """Modify pending order: change item to different variant of same product."""

    sample = sample_retail_user(rng, criteria={
        "status": "pending",
        "min_items": 1,
        "has_unlimited_payment": True,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 20 (pending)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]
    item = order["items"][0]

    # Find alternative variant from synthetic products
    products_db = sample.get("products_db", {})
    product = products_db.get(item["product_id"], {})
    current_item_id = item["item_id"]
    alt_variants = [
        (vid, v) for vid, v in product.get("variants", {}).items()
        if vid != current_item_id and v.get("available", False)
    ]

    if alt_variants:
        alt_id, alt_variant = rng.choice(alt_variants)
        alt_options = alt_variant.get("options", {})
        change_desc = ", ".join(f"{k}: {v}" for k, v in alt_options.items())
        modify_desc = f"change the {item['name']} to the one with {change_desc}"
    else:
        modify_desc = f"change the {item['name']} to a different color or size"

    original_pay_id = order["payment_history"][0]["payment_method_id"]

    initial_msg = (
        f"Hi, my email is {user['email']}. I have a pending order {order_id} and I'd like to "
        f"{modify_desc}."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} which is pending.\n"
            f"You want to {modify_desc}.\n"
            f"Use payment method {original_pay_id} for any price difference."
        ),
        goal=f"Modify order {order_id}: {modify_desc}. Use {original_pay_id} for price difference.",
        approach="cooperative",
    )

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "modify_pending_order_items", "check": "order_match",
             "order_id": order_id},
        ],
        correct_behavior=f"Modify pending order items. Change to different variant of same product.",
    )

    return Scenario(
        template_id=20, template_name="cooperative_modify_items",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: modify items in pending order {order_id}",
        key_facts={"order_id": order_id, "item_name": item["name"]},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Template 21: Modify Pending Order Address (Retail)
# Mirrors: R7, R18, R29, R40, etc. — address change
# =====================================================================
def generate_template_21(rng: random.Random) -> Scenario:
    """Modify pending order shipping address."""

    sample = sample_retail_user(rng, criteria={
        "status": "pending",
        "min_items": 1,
    })
    if sample is None:
        raise ValueError("No matching retail user for template 21 (pending)")

    uid = sample["user_id"]
    user = sample["user"]
    order_id = sample["order_id"]
    order = sample["order"]

    # Use user's default address (different from order address)
    user_addr = user.get("address", {})
    order_addr = order.get("address", {})

    # Generate a plausible new address
    new_addr = copy.deepcopy(user_addr)
    if new_addr.get("address1") == order_addr.get("address1"):
        # Make it different
        new_addr["address1"] = f"{rng.randint(100, 999)} {rng.choice(['Pine Street', 'Cedar Ave', 'Birch Lane'])}"

    initial_msg = (
        f"Hi, my email is {user['email']}. I need to update the shipping address on "
        f"order {order_id} to {new_addr.get('address1', '123 Main St')}, "
        f"{new_addr.get('city', 'Anytown')}, {new_addr.get('state', 'CA')}."
    )

    user_prompt = build_user_system_prompt(
        customer_context=(
            f"Your email is {user['email']}.\n"
            f"You have order {order_id} which is pending.\n"
            f"You want to change the shipping address to:\n"
            f"{new_addr.get('address1', '123 Main St')}, "
            f"{new_addr.get('city', 'Anytown')}, {new_addr.get('state', 'CA')} "
            f"{new_addr.get('zip', '00000')}."
        ),
        goal=f"Update shipping address on order {order_id}.",
        approach="cooperative",
    )

    ground_truth = GroundTruth(
        required_actions=[
            {"name": "modify_pending_order_address", "check": "order_match",
             "order_id": order_id},
        ],
        correct_behavior=f"Modify shipping address on pending order. Straightforward request.",
    )

    return Scenario(
        template_id=21, template_name="cooperative_modify_address",
        domain="retail",
        ground_truth=ground_truth, pressure_type=PressureType.PASSIVE,
        description=f"Cooperative: update address on pending order {order_id}",
        key_facts={"order_id": order_id, "new_address": new_addr.get("address1", "")},
        user_system_prompt=user_prompt,
        initial_message=initial_msg,
        db=build_retail_db(sample["user"], sample["order"], sample.get("products_db", {})),
    )


# =====================================================================
# Master generator
# =====================================================================
TEMPLATE_GENERATORS = {
    1: generate_template_1,
    2: generate_template_2,
    3: generate_template_3,
    4: generate_template_4,
    5: generate_template_5,
    6: generate_template_6,
    7: generate_template_7,
    8: generate_template_8,
    9: generate_template_9,
    10: generate_template_10,
    11: generate_template_11,
    12: generate_template_12,
    13: generate_template_13,
    14: generate_template_14,
    15: generate_template_15,
    16: generate_template_16,
    17: generate_template_17,
    18: generate_template_18,
    19: generate_template_19,
    20: generate_template_20,
    21: generate_template_21,
}


def generate_scenario(seed: int, adversarial_ratio: float = 1.0) -> Scenario:
    """Generate a complete scenario from a seed. Fully deterministic.

    Args:
        seed: Random seed for deterministic generation.
        adversarial_ratio: Probability of selecting an adversarial template (0.0-1.0).
            At 1.0 (default, backward compatible): always adversarial (T1-T12).
            At 0.2: 20% adversarial (T1-T12), 80% cooperative (T13-T21).
            At 0.0: always cooperative.
    """
    rng = random.Random(seed)

    # Decide adversarial vs cooperative
    if adversarial_ratio >= 1.0 or rng.random() < adversarial_ratio:
        template_id = _select_from_weights(rng, ADVERSARIAL_TEMPLATE_WEIGHTS)
    else:
        template_id = _select_from_weights(rng, COOPERATIVE_TEMPLATE_WEIGHTS)

    generator = TEMPLATE_GENERATORS[template_id]
    return generator(rng)
