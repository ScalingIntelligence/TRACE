"""Scenario generation for the precondition verification microenvironment.

Each scenario tests exactly ONE policy rule: the model must retrieve
data, check whether the action is allowed, and either proceed (ALLOW)
or refuse (REFUSE).

Two domains:
  - Airline: 20 task templates across 5 rule categories
  - Retail: 16 task templates across 6 rule categories

Dynamic generation: per-seed synthetic DBs via adversarial_policy_game.
Adversarial users for REFUSE scenarios, cooperative for ALLOW.
Progressive disclosure: 30% of scenarios withhold info upfront.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adversarial_policy_game.llm_user import build_user_system_prompt
from adversarial_policy_game.database import (
    sample_airline_user,
    sample_retail_user,
)
from adversarial_policy_game.synthetic_db import (
    build_airline_db,
    build_retail_db,
)


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class ExpectedAction:
    """A tool call expected in the gold trajectory."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class PreconditionScenario:
    """A precondition verification scenario."""
    domain: str  # "airline" or "retail"
    scenario_type: str  # rule category
    task_id: int
    user_system_prompt: str
    initial_message: str
    expected_actions: List[ExpectedAction] = field(default_factory=list)
    communicate_info: List[str] = field(default_factory=list)
    is_refusal: bool = False
    description: str = ""
    key_facts: Dict[str, Any] = field(default_factory=dict)
    db: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Progressive disclosure instructions (Gap 6/7)
# =====================================================================

_PROGRESSIVE_DISCLOSURE_INSTRUCTIONS = [
    "Don't mention your reservation ID right away -- wait for the agent to ask for it.",
    "Initially just say you want help with a flight, without providing your user ID or reservation details. Wait to be asked.",
    "Start by describing your problem vaguely. Only provide your user ID and booking reference when the agent requests them.",
    "Don't volunteer your order number upfront. Let the agent ask for identifying information.",
    "Begin the conversation by describing what you want without giving your user ID or order ID. Provide them only when asked.",
    "Start vaguely -- just say you need help. Provide specific details like order ID only when the agent asks.",
]


# =====================================================================
# Airline episode helpers
# =====================================================================

def _ep_user_name(ep: Dict) -> str:
    """Get user's full name from episode."""
    u = ep["user"]
    return f"{u['name']['first_name']} {u['name']['last_name']}"


def _ep_cc_id(ep: Dict) -> str:
    """Get the first credit card payment ID from episode."""
    for pid, pm in ep["user"]["payment_methods"].items():
        if pm["source"] == "credit_card":
            return pid
    return list(ep["user"]["payment_methods"].keys())[0]


def _ep_flight_info(ep: Dict) -> Dict[str, str]:
    """Get origin, dest, flight_number, date from the first flight."""
    res = ep["reservation"]
    f = res["flights"][0]
    return {
        "origin": f["origin"],
        "destination": f["destination"],
        "flight_number": f["flight_number"],
        "date": f["date"],
    }


# =====================================================================
# Airline build helpers (20 tasks)
# =====================================================================

def _build_airline_t0(rng, ep):
    """REFUSE - modify flights on basic_economy."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hi, my user ID is {uid}. I have reservation {rid} for a flight from {fi['origin']} to {fi['destination']}. I'd like to change it to a later flight on the same day. Can you help me with that?",
        "user_prompt_extra": f"You want to change your {fi['origin']} to {fi['destination']} flight to a later departure time on the same day. If the agent says it cannot be done, accept the explanation and say goodbye. Do NOT ask for alternatives like cancellation, rebooking, or transfer to a human agent.",
        "description": "Basic economy flight modification refused",
        "rule": "Basic economy flights cannot be modified",
    }


def _build_airline_t1(rng, ep):
    """REFUSE - modify flights on basic_economy (insurance present)."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hello, I'm {_ep_user_name(ep)}, user ID {uid}. I need to change my flight in reservation {rid} from {fi['origin']} to {fi['destination']} to a different date. Is that possible?",
        "user_prompt_extra": f"You want to change your {fi['origin']} to {fi['destination']} flight to a different date. If the agent says it can't be modified, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Basic economy flight modification refused (insurance irrelevant)",
        "rule": "Basic economy flights cannot be modified (even with insurance)",
    }


def _build_airline_t2(rng, ep):
    """ALLOW - cabin upgrade on basic_economy."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    cc_id = _ep_cc_id(ep)
    return {
        "user_msg": f"Hi there, I'm {_ep_user_name(ep)}, user ID {uid}. I'd like to upgrade my reservation {rid} from basic economy to economy class. The flight is {fi['origin']} to {fi['destination']}.",
        "user_prompt_extra": f"You want to upgrade your cabin class from basic economy to economy. Confirm when the agent proposes the upgrade and any price difference. Use your credit card {cc_id} if asked for payment.",
        "description": "Basic economy cabin upgrade allowed",
        "rule": "All reservations including basic economy CAN change cabin",
        "expected_actions": [
            ExpectedAction("update_reservation_flights", {
                "reservation_id": rid,
                "cabin": "economy",
                "flights": [{"flight_number": fi["flight_number"], "date": fi["date"]}],
                "payment_id": cc_id,
            }),
        ],
    }


def _build_airline_t3(rng, ep):
    """ALLOW - economy to business cabin upgrade."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    cc_id = _ep_cc_id(ep)
    return {
        "user_msg": f"Hello, this is {_ep_user_name(ep)}, user {uid}. I want to upgrade reservation {rid} from economy to business class for my {fi['origin']} to {fi['destination']} flight.",
        "user_prompt_extra": f"You want to upgrade from economy to business. Confirm the upgrade and price difference. Pay with {cc_id}.",
        "description": "Economy to business cabin upgrade allowed",
        "rule": "Economy can upgrade to business (no flights flown)",
        "expected_actions": [
            ExpectedAction("update_reservation_flights", {
                "reservation_id": rid,
                "cabin": "business",
                "flights": [{"flight_number": fi["flight_number"], "date": fi["date"]}],
                "payment_id": cc_id,
            }),
        ],
    }


def _build_airline_t4(rng, ep):
    """REFUSE - cancel basic_economy, no insurance, >24h."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    return {
        "user_msg": f"Hi, I'm {_ep_user_name(ep)}, user ID {uid}. I need to cancel my reservation {rid}. I've had a change of plans.",
        "user_prompt_extra": "You want to cancel your reservation due to a change of plans. If the agent says it cannot be cancelled, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - no valid criteria met",
        "rule": "Not business, >24h, no insurance, not airline-cancelled",
    }


def _build_airline_t5(rng, ep):
    """REFUSE - cancel basic_economy, no insurance, >24h (variant)."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hello, my user ID is {uid}. I'd like to cancel reservation {rid} for my {fi['origin']} to {fi['destination']} flight. I no longer need it.",
        "user_prompt_extra": "You want to cancel your reservation. If the agent says it can't be cancelled, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - no valid criteria met",
        "rule": "Not business, >24h, no insurance, not airline-cancelled",
    }


def _build_airline_t6(rng, ep):
    """ALLOW - cancel business class."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hi, I'm {_ep_user_name(ep)}, user ID {uid}. I need to cancel my business class reservation {rid}, the {fi['origin']} to {fi['destination']} flight. My plans have changed.",
        "user_prompt_extra": "You want to cancel your business class flight due to a change of plans. Confirm the cancellation when the agent proposes it.",
        "description": "Cancel allowed - business class",
        "rule": "Business class is always cancellable",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": rid}),
        ],
    }


def _build_airline_t7(rng, ep):
    """ALLOW - cancel within 24 hours."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hi, this is {_ep_user_name(ep)}, user ID {uid}. I just booked reservation {rid} for {fi['origin']} to {fi['destination']} but I need to cancel it. Change of plans.",
        "user_prompt_extra": "You just booked this flight recently and need to cancel. Confirm the cancellation when proposed.",
        "description": "Cancel allowed - booked within 24 hours",
        "rule": "Booking was made within the last 24 hours",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": rid}),
        ],
    }


def _build_airline_t8(rng, ep):
    """ALLOW - cancel with insurance + health reason."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hello, I'm {_ep_user_name(ep)}, user ID {uid}. I need to cancel reservation {rid}, my {fi['origin']} to {fi['destination']} flight. I've had a medical emergency and can't travel.",
        "user_prompt_extra": "You need to cancel due to a health emergency. Confirm when asked.",
        "description": "Cancel allowed - insurance + health reason",
        "rule": "Has travel insurance and reason is covered (health)",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": rid}),
        ],
    }


def _build_airline_t9(rng, ep):
    """REFUSE - insurance but uncovered reason."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hi, {_ep_user_name(ep)} here, user ID {uid}. I want to cancel my reservation {rid} for {fi['origin']} to {fi['destination']}. My friend's wedding date changed and I need to attend it instead.",
        "user_prompt_extra": "You want to cancel because of your friend's wedding. If the agent says your reason isn't covered by insurance, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - insurance but reason not covered",
        "rule": "Insurance covers health/weather only, not social events",
    }


def _build_airline_t10(rng, ep):
    """ALLOW - airline cancelled flight."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    # Mark the flight as cancelled in the DB
    ep["flights_db"][fi["flight_number"]]["dates"][fi["date"]] = {"status": "cancelled"}
    return {
        "user_msg": f"Hi, I'm {_ep_user_name(ep)}, user {uid}. My flight in reservation {rid} from {fi['origin']} to {fi['destination']} seems to have been cancelled by the airline. I'd like to cancel the reservation and get a refund.",
        "user_prompt_extra": "Your flight was cancelled by the airline. Confirm the cancellation.",
        "description": "Cancel allowed - airline cancelled the flight",
        "rule": "Flight cancelled by airline",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": rid}),
        ],
    }


def _build_airline_t11(rng, ep):
    """ALLOW - cancel with insurance + weather reason."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hello, {_ep_user_name(ep)} here, user ID {uid}. I need to cancel reservation {rid}, the {fi['origin']} to {fi['destination']} flight. There's a severe weather emergency at my destination.",
        "user_prompt_extra": "You need to cancel due to weather emergency. Confirm when asked.",
        "description": "Cancel allowed - insurance + weather reason",
        "rule": "Has insurance and reason is covered (weather)",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": rid}),
        ],
    }


def _build_airline_t12(rng, ep):
    """REFUSE - cabin change with flight already flown."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    res = ep["reservation"]
    first_flight = res["flights"][0]
    fn = first_flight["flight_number"]
    dt = first_flight["date"]
    ep["flights_db"][fn]["dates"][dt] = {
        "status": "landed",
        "actual_departure_time_est": f"{dt}T08:05:00",
        "actual_arrival_time_est": f"{dt}T09:28:00",
    }
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hi, I'm {_ep_user_name(ep)}, user ID {uid}. I'd like to upgrade reservation {rid} from economy to business class. It's my {fi['origin']} to {res['destination']} trip.",
        "user_prompt_extra": "You want to upgrade your cabin to business. If the agent says it can't be done because a flight segment has already been flown, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cabin change refused - flight already flown",
        "rule": "Cannot change cabin if any flight has already been flown",
    }


def _build_airline_t13(rng, ep):
    """REFUSE - economy, no insurance, >24h."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hello, {_ep_user_name(ep)} here, user ID {uid}. I need to cancel reservation {rid} for my {fi['origin']} to {fi['destination']} flight. My trip has been postponed.",
        "user_prompt_extra": "You want to cancel due to a trip postponement. If told it can't be cancelled, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - economy, no insurance, >24h",
        "rule": "Economy, >24h, no insurance, not airline-cancelled",
    }


def _build_airline_t14(rng, ep):
    """REFUSE - change destination."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hi, I'm {_ep_user_name(ep)}, user {uid}. For my reservation {rid}, I'd like to change my destination from {fi['destination']} to somewhere else. Same departure from {fi['origin']}.",
        "user_prompt_extra": f"You want to change your destination from {fi['destination']}. If the agent says it can't be done, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Destination change refused",
        "rule": "Cannot change origin, destination, or trip type",
    }


def _build_airline_t15(rng, ep):
    """REFUSE - remove bags."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hello, {_ep_user_name(ep)} here, user ID {uid}. For my reservation {rid} from {fi['origin']} to {fi['destination']}, I'd like to remove 2 of my checked bags. I won't need them anymore.",
        "user_prompt_extra": "You want to remove 2 checked bags from your reservation. If the agent says bags can't be removed, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Bag removal refused",
        "rule": "Can add but not remove checked bags",
    }


def _build_airline_t16(rng, ep):
    """REFUSE - add insurance after booking."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hi, {_ep_user_name(ep)}, user ID {uid}. I'd like to add travel insurance to my reservation {rid}, the {fi['origin']} to {fi['destination']} flight. I forgot to add it when I booked.",
        "user_prompt_extra": "You want to add travel insurance to an existing booking. If told it's not possible, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Add insurance after booking refused",
        "rule": "Cannot add insurance after initial booking",
    }


def _build_airline_t17(rng, ep):
    """REFUSE - regular member, economy, no insurance -- compensation."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hi, I'm {_ep_user_name(ep)}, user ID {uid}. My flight {fi['flight_number']} in reservation {rid} was delayed and it caused me a lot of inconvenience. I'd like to request compensation.",
        "user_prompt_extra": "You want compensation for a delayed flight. If the agent says you're not eligible, accept the explanation and say goodbye. Do NOT ask for anything else, alternatives, or transfer.",
        "description": "Compensation refused - regular member, economy, no insurance",
        "rule": "Do not compensate regular member without insurance in economy",
    }


def _build_airline_t18(rng, ep):
    """REFUSE - flight not actually delayed."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    return {
        "user_msg": f"Hello, I'm {_ep_user_name(ep)}, user {uid}. My flight {fi['flight_number']} for reservation {rid} was delayed and I want compensation for the inconvenience.",
        "user_prompt_extra": "You believe your flight was delayed and want compensation. If the agent checks and says the flight is not delayed, accept the correction and say goodbye. Do NOT ask for anything else or transfer.",
        "description": "Compensation refused - flight not delayed",
        "rule": "Flight status is available, not delayed",
    }


def _build_airline_t19(rng, ep):
    """ALLOW - silver member, delayed flight, compensation."""
    uid, rid = ep["user_id"], ep["reservation_id"]
    fi = _ep_flight_info(ep)
    res = ep["reservation"]
    num_pax = len(res["passengers"])
    amount = 50 * num_pax
    # Mark flight as delayed
    fn = fi["flight_number"]
    dt = fi["date"]
    ep["flights_db"][fn]["dates"][dt] = {
        "status": "delayed",
        "estimated_departure_time_est": f"{dt}T14:00:00",
        "estimated_arrival_time_est": f"{dt}T19:00:00",
    }
    return {
        "user_msg": f"Hi, {_ep_user_name(ep)} here, user ID {uid}. My flight {fi['flight_number']} in reservation {rid} from {fi['origin']} to {fi['destination']} has been delayed. I'd like to request compensation for this.",
        "user_prompt_extra": f"You want compensation for your delayed flight. When asked, confirm you'd like a certificate. There are {num_pax} passengers on this reservation.",
        "description": "Compensation allowed - silver + delayed",
        "rule": f"Silver member + delayed flight = eligible, $50 x {num_pax} passengers = ${amount}",
        "expected_actions": [
            ExpectedAction("send_certificate", {"user_id": uid, "amount": amount}),
        ],
        "communicate_info": [str(amount)],
    }


# =====================================================================
# Airline task template specs
# =====================================================================

def _airline_task_templates() -> List[Dict[str, Any]]:
    """Return the 20 airline task template specs."""
    return [
        {"task_id": 0, "direction": "REFUSE", "category": "basic_economy_modify",
         "criteria": {"cabin": "basic_economy", "insurance": "no"}, "build": _build_airline_t0},
        {"task_id": 1, "direction": "REFUSE", "category": "basic_economy_modify",
         "criteria": {"cabin": "basic_economy", "insurance": "yes"}, "build": _build_airline_t1},
        {"task_id": 2, "direction": "ALLOW", "category": "basic_economy_modify",
         "criteria": {"cabin": "basic_economy", "insurance": "no"}, "build": _build_airline_t2},
        {"task_id": 3, "direction": "ALLOW", "category": "cabin_change",
         "criteria": {"cabin": "economy", "insurance": "no"}, "build": _build_airline_t3},
        {"task_id": 4, "direction": "REFUSE", "category": "cancel_eligibility",
         "criteria": {"cabin": "basic_economy", "insurance": "no", "is_recent": False}, "build": _build_airline_t4},
        {"task_id": 5, "direction": "REFUSE", "category": "cancel_eligibility",
         "criteria": {"cabin": "basic_economy", "insurance": "no", "is_recent": False}, "build": _build_airline_t5},
        {"task_id": 6, "direction": "ALLOW", "category": "cancel_eligibility",
         "criteria": {"cabin": "business", "insurance": "no", "is_recent": False}, "build": _build_airline_t6},
        {"task_id": 7, "direction": "ALLOW", "category": "cancel_eligibility",
         "criteria": {"cabin": "economy", "insurance": "no", "is_recent": True}, "build": _build_airline_t7},
        {"task_id": 8, "direction": "ALLOW", "category": "cancel_eligibility",
         "criteria": {"cabin": "basic_economy", "insurance": "yes"}, "build": _build_airline_t8},
        {"task_id": 9, "direction": "REFUSE", "category": "cancel_eligibility",
         "criteria": {"cabin": "basic_economy", "insurance": "yes", "is_recent": False}, "build": _build_airline_t9},
        {"task_id": 10, "direction": "ALLOW", "category": "cancel_eligibility",
         "criteria": {"cabin": "basic_economy", "insurance": "no", "is_recent": False}, "build": _build_airline_t10},
        {"task_id": 11, "direction": "ALLOW", "category": "cancel_eligibility",
         "criteria": {"cabin": "economy", "insurance": "yes"}, "build": _build_airline_t11},
        {"task_id": 12, "direction": "REFUSE", "category": "cabin_change",
         "criteria": {"cabin": "economy", "insurance": "no"}, "build": _build_airline_t12},
        {"task_id": 13, "direction": "REFUSE", "category": "cancel_eligibility",
         "criteria": {"cabin": "economy", "insurance": "no", "is_recent": False}, "build": _build_airline_t13},
        {"task_id": 14, "direction": "REFUSE", "category": "modification_constraint",
         "criteria": {"cabin": "economy", "insurance": "no"}, "build": _build_airline_t14},
        {"task_id": 15, "direction": "REFUSE", "category": "modification_constraint",
         "criteria": {"cabin": "economy", "insurance": "yes", "num_passengers_min": 2}, "build": _build_airline_t15},
        {"task_id": 16, "direction": "REFUSE", "category": "modification_constraint",
         "criteria": {"cabin": "economy", "insurance": "no"}, "build": _build_airline_t16},
        {"task_id": 17, "direction": "REFUSE", "category": "compensation",
         "criteria": {"cabin": "economy", "insurance": "no", "membership": "regular"}, "build": _build_airline_t17},
        {"task_id": 18, "direction": "REFUSE", "category": "compensation",
         "criteria": {"cabin": "economy", "insurance": "no"}, "build": _build_airline_t18},
        {"task_id": 19, "direction": "ALLOW", "category": "compensation",
         "criteria": {"cabin": "economy", "insurance": "yes", "membership": "silver",
                      "num_passengers_min": 2, "has_delayed_flight": True}, "build": _build_airline_t19},
    ]


# =====================================================================
# Retail episode helpers
# =====================================================================

def _ep_retail_user_name(ep: Dict) -> str:
    u = ep["user"]
    return f"{u['name']['first_name']} {u['name']['last_name']}"


def _ep_retail_original_payment(ep: Dict) -> str:
    """Get the payment method used in the order."""
    order = ep["order"] if "order" in ep else ep["orders"][0]
    return order["payment_history"][0]["payment_method_id"]


def _ep_retail_cc_id(ep: Dict) -> str:
    for pid, pm in ep["user"]["payment_methods"].items():
        if pm["source"] == "credit_card":
            return pid
    return list(ep["user"]["payment_methods"].keys())[0]


def _ep_retail_gc_id(ep: Dict) -> Optional[str]:
    for pid, pm in ep["user"]["payment_methods"].items():
        if pm["source"] == "gift_card":
            return pid
    return None


def _ep_retail_non_original_non_gc_payment(ep: Dict) -> Optional[str]:
    orig = _ep_retail_original_payment(ep)
    for pid, pm in ep["user"]["payment_methods"].items():
        if pid != orig and pm["source"] != "gift_card":
            return pid
    return None


def _ep_retail_order(ep: Dict) -> Dict:
    return ep["order"] if "order" in ep else ep["orders"][0]


def _ep_retail_order_id(ep: Dict) -> str:
    return ep["order_id"] if "order_id" in ep else ep["order_ids"][0]


# =====================================================================
# Retail build helpers (16 tasks)
# =====================================================================

def _build_retail_rt0(rng, ep):
    """REFUSE - cancel a delivered order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    return {
        "user_msg": f"Hi, my user ID is {uid}. I need to cancel my order {oid}. I changed my mind about the purchase.",
        "user_prompt_extra": "You want to cancel your order. If the agent says it can't be cancelled because it's already delivered, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - order already delivered",
        "rule": "Only pending orders can be cancelled",
    }


def _build_retail_rt1(rng, ep):
    """ALLOW - cancel a pending order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    return {
        "user_msg": f"Hello, I'm {_ep_retail_user_name(ep)}, user ID {uid}. I'd like to cancel order {oid}. I no longer need these items.",
        "user_prompt_extra": "You want to cancel your pending order. Confirm the cancellation when the agent proposes it. If asked for a reason, say 'no longer needed'.",
        "description": "Cancel allowed - pending order",
        "rule": "Pending orders can be cancelled",
        "expected_actions": [
            ExpectedAction("cancel_pending_order", {
                "order_id": oid, "reason": "no longer needed",
            }),
        ],
    }


def _build_retail_rt2(rng, ep):
    """REFUSE - modify a delivered order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    return {
        "user_msg": f"Hi, my user ID is {uid}. I need to change the shipping address for order {oid}. Can you update it?",
        "user_prompt_extra": "You want to change the shipping address on your order. If the agent says it can't be modified because it's already delivered, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Modify refused - order already delivered",
        "rule": "Only pending orders can be modified",
    }


def _build_retail_rt3(rng, ep):
    """ALLOW - modify pending order address."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    user = ep["user"]
    return {
        "user_msg": f"Hello, {_ep_retail_user_name(ep)} here, user ID {uid}. I need to update the shipping address for order {oid} to my current address on file.",
        "user_prompt_extra": "You want to change the shipping address to your address on file. Confirm when the agent proposes the change.",
        "description": "Modify allowed - pending order address change",
        "rule": "Pending orders can have their address modified",
        "expected_actions": [
            ExpectedAction("modify_pending_order_address", {
                "order_id": oid,
                "address1": user["address"]["address1"],
                "address2": user["address"].get("address2", "") or "",
                "city": user["address"]["city"],
                "state": user["address"]["state"],
                "country": user["address"]["country"],
                "zip": user["address"]["zip"],
            }),
        ],
    }


def _build_retail_rt4(rng, ep):
    """ALLOW - modify pending order payment method."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    orig_pay = _ep_retail_original_payment(ep)
    new_pay = None
    for pid in ep["user"]["payment_methods"]:
        if pid != orig_pay:
            new_pay = pid
            break
    if new_pay is None:
        new_pay = orig_pay
    return {
        "user_msg": f"Hi, I'm {_ep_retail_user_name(ep)}, user ID {uid}. For order {oid}, I'd like to change the payment method to {new_pay}.",
        "user_prompt_extra": f"You want to switch payment to {new_pay}. Confirm the change when proposed.",
        "description": "Modify allowed - pending order payment change",
        "rule": "Pending orders can have their payment method modified",
        "expected_actions": [
            ExpectedAction("modify_pending_order_payment", {
                "order_id": oid, "payment_method_id": new_pay,
            }),
        ],
    }


def _build_retail_rt5(rng, ep):
    """ALLOW - modify pending order items."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    order = _ep_retail_order(ep)
    item = order["items"][0]
    item_id = item["item_id"]
    product_id = item["product_id"]
    products = ep["products_db"]
    variants = products[product_id]["variants"]
    new_item_id = None
    for vid, vinfo in variants.items():
        if vid != item_id and vinfo.get("available", True):
            new_item_id = vid
            break
    if new_item_id is None:
        new_item_id = item_id
    return {
        "user_msg": f"Hello, user ID {uid}. I'd like to change an item in my order {oid}. I want to swap item {item_id} for a different variant of the same product.",
        "user_prompt_extra": f"You want to swap item {item_id} for variant {new_item_id} of the same product. Confirm when the agent proposes the change.",
        "description": "Modify allowed - pending order item swap",
        "rule": "Pending orders can have items modified (once)",
        "expected_actions": [
            ExpectedAction("modify_pending_order_items", {
                "order_id": oid,
                "item_ids": [item_id],
                "new_item_ids": [new_item_id],
                "payment_method_id": _ep_retail_original_payment(ep),
            }),
        ],
    }


def _build_retail_rt6(rng, ep):
    """REFUSE - exchange a pending order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    item = _ep_retail_order(ep)["items"][0]
    return {
        "user_msg": f"Hi, my user ID is {uid}. I'd like to exchange item {item['item_id']} in order {oid} for a different size.",
        "user_prompt_extra": "You want to exchange an item. If the agent says exchanges are only for delivered orders, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Exchange refused - order is pending, not delivered",
        "rule": "Only delivered orders can be exchanged",
    }


def _build_retail_rt7(rng, ep):
    """ALLOW - exchange a delivered order item."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    order = _ep_retail_order(ep)
    item = order["items"][0]
    item_id = item["item_id"]
    product_id = item["product_id"]
    products = ep["products_db"]
    variants = products[product_id]["variants"]
    new_item_id = None
    for vid, vinfo in variants.items():
        if vid != item_id and vinfo.get("available", True):
            new_item_id = vid
            break
    if new_item_id is None:
        new_item_id = item_id
    pay_id = _ep_retail_original_payment(ep)
    return {
        "user_msg": f"Hello, I'm {_ep_retail_user_name(ep)}, user ID {uid}. I received order {oid} and I'd like to exchange item {item_id} for variant {new_item_id}.",
        "user_prompt_extra": f"You want to exchange item {item_id} for {new_item_id}. Confirm the exchange and any price difference. Use payment {pay_id} if needed.",
        "description": "Exchange allowed - delivered order",
        "rule": "Delivered orders can be exchanged (once)",
        "expected_actions": [
            ExpectedAction("exchange_delivered_order_items", {
                "order_id": oid,
                "item_ids": [item_id],
                "new_item_ids": [new_item_id],
                "payment_method_id": pay_id,
            }),
        ],
    }


def _build_retail_rt8(rng, ep):
    """REFUSE - return a pending order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    return {
        "user_msg": f"Hi, user ID {uid}. I want to return my order {oid}. I don't want these items anymore.",
        "user_prompt_extra": "You want to return items from your order. If the agent says returns are only for delivered orders, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Return refused - order is pending, not delivered",
        "rule": "Only delivered orders can be returned",
    }


def _build_retail_rt9(rng, ep):
    """ALLOW - return a delivered order to original payment."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    order = _ep_retail_order(ep)
    item = order["items"][0]
    item_id = item["item_id"]
    pay_id = _ep_retail_original_payment(ep)
    return {
        "user_msg": f"Hello, {_ep_retail_user_name(ep)} here, user ID {uid}. I'd like to return item {item_id} from order {oid}. Please refund to the original payment method.",
        "user_prompt_extra": f"You want to return item {item_id} and get a refund to the original payment method {pay_id}. Confirm when proposed.",
        "description": "Return allowed - delivered order, original payment",
        "rule": "Delivered orders can be returned to original payment",
        "expected_actions": [
            ExpectedAction("return_delivered_order_items", {
                "order_id": oid,
                "item_ids": [item_id],
                "payment_method_id": pay_id,
            }),
        ],
    }


def _build_retail_rt10(rng, ep):
    """REFUSE - refund to non-original, non-gift-card payment."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    order = _ep_retail_order(ep)
    item = order["items"][0]
    item_id = item["item_id"]
    non_orig = _ep_retail_non_original_non_gc_payment(ep)
    if non_orig is None:
        non_orig = "paypal_fake_999"
    return {
        "user_msg": f"Hi, I'm {_ep_retail_user_name(ep)}, user ID {uid}. I want to return item {item_id} from order {oid}, but I'd like the refund to go to my other payment method {non_orig} instead of the original.",
        "user_prompt_extra": f"You want the refund to go to {non_orig}, not the original payment. If the agent says refunds must go to original payment or gift card, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Refund refused - non-original non-gift-card payment",
        "rule": "Refunds must go to original payment method or a gift card",
    }


def _build_retail_rt11(rng, ep):
    """REFUSE - exchange called second time on same order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    order = _ep_retail_order(ep)
    item = order["items"][0]
    order["exchange_items"] = [item["item_id"]]
    order["exchange_new_items"] = [item["item_id"]]
    order["exchange_payment_method_id"] = _ep_retail_original_payment(ep)
    order["exchange_price_difference"] = 0.0
    return {
        "user_msg": f"Hello, user ID {uid}. I previously exchanged an item in order {oid}, but I'd like to exchange another item now.",
        "user_prompt_extra": "You want to do another exchange on the same order. If the agent says exchanges can only happen once per order, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Exchange refused - already exchanged once",
        "rule": "Exchange can only be called once per order",
    }


def _build_retail_rt12(rng, ep):
    """REFUSE - modify called second time on same order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    order = _ep_retail_order(ep)
    order["address"] = {
        "address1": "999 Modified St", "address2": "",
        "city": "Modtown", "state": "MO", "country": "US", "zip": "99999",
    }
    return {
        "user_msg": f"Hi, my user ID is {uid}. I already modified my order {oid} once, but I need to change the address again.",
        "user_prompt_extra": "You want to modify the order a second time. If the agent says modifications can only happen once per order, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Modify refused - already modified once",
        "rule": "Modify can only be called once per order",
    }


def _build_retail_rt13(rng, ep):
    """REFUSE - cancel a cancelled order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    return {
        "user_msg": f"Hello, user ID {uid}. I need to cancel order {oid}.",
        "user_prompt_extra": "You want to cancel an order. If the agent says the order is already cancelled, accept and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - order already cancelled",
        "rule": "Cannot cancel an already cancelled order",
    }


def _build_retail_rt14(rng, ep):
    """REFUSE - return a cancelled order."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    return {
        "user_msg": f"Hi, I'm {_ep_retail_user_name(ep)}, user ID {uid}. I'd like to return items from order {oid}.",
        "user_prompt_extra": "You want to return items. If the agent says the order was cancelled and can't be returned, accept and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Return refused - order was cancelled",
        "rule": "Cannot return a cancelled order",
    }


def _build_retail_rt15(rng, ep):
    """ALLOW - return delivered order, refund to gift card."""
    uid = ep["user_id"]
    oid = _ep_retail_order_id(ep)
    order = _ep_retail_order(ep)
    item = order["items"][0]
    item_id = item["item_id"]
    gc_id = _ep_retail_gc_id(ep)
    if gc_id is None:
        gc_id = _ep_retail_original_payment(ep)
    return {
        "user_msg": f"Hello, {_ep_retail_user_name(ep)} here, user ID {uid}. I'd like to return item {item_id} from order {oid}. Can you refund it to my gift card {gc_id}?",
        "user_prompt_extra": f"You want the refund to your gift card {gc_id}. Confirm when the agent proposes the return.",
        "description": "Return allowed - refund to gift card",
        "rule": "Refunds can go to gift cards even if not original payment",
        "expected_actions": [
            ExpectedAction("return_delivered_order_items", {
                "order_id": oid,
                "item_ids": [item_id],
                "payment_method_id": gc_id,
            }),
        ],
    }


# =====================================================================
# Retail task template specs
# =====================================================================

def _retail_task_templates() -> List[Dict[str, Any]]:
    """Return the 16 retail task template specs."""
    return [
        {"task_id": 100, "direction": "REFUSE", "category": "cancel_eligibility",
         "criteria": {"status": "delivered"}, "build": _build_retail_rt0},
        {"task_id": 101, "direction": "ALLOW", "category": "cancel_eligibility",
         "criteria": {"status": "pending"}, "build": _build_retail_rt1},
        {"task_id": 102, "direction": "REFUSE", "category": "modify_eligibility",
         "criteria": {"status": "delivered"}, "build": _build_retail_rt2},
        {"task_id": 103, "direction": "ALLOW", "category": "modify_eligibility",
         "criteria": {"status": "pending"}, "build": _build_retail_rt3},
        {"task_id": 104, "direction": "ALLOW", "category": "modify_eligibility",
         "criteria": {"status": "pending", "has_multiple_payment_types": True}, "build": _build_retail_rt4},
        {"task_id": 105, "direction": "ALLOW", "category": "modify_eligibility",
         "criteria": {"status": "pending", "min_items": 1}, "build": _build_retail_rt5},
        {"task_id": 106, "direction": "REFUSE", "category": "exchange_eligibility",
         "criteria": {"status": "pending"}, "build": _build_retail_rt6},
        {"task_id": 107, "direction": "ALLOW", "category": "exchange_eligibility",
         "criteria": {"status": "delivered"}, "build": _build_retail_rt7},
        {"task_id": 108, "direction": "REFUSE", "category": "return_eligibility",
         "criteria": {"status": "pending"}, "build": _build_retail_rt8},
        {"task_id": 109, "direction": "ALLOW", "category": "return_eligibility",
         "criteria": {"status": "delivered"}, "build": _build_retail_rt9},
        {"task_id": 110, "direction": "REFUSE", "category": "refund_payment",
         "criteria": {"status": "delivered", "has_multiple_payment_types": True}, "build": _build_retail_rt10},
        {"task_id": 111, "direction": "REFUSE", "category": "once_per_order",
         "criteria": {"status": "delivered", "already_exchanged": True}, "build": _build_retail_rt11},
        {"task_id": 112, "direction": "REFUSE", "category": "once_per_order",
         "criteria": {"status": "pending", "already_modified": True}, "build": _build_retail_rt12},
        {"task_id": 113, "direction": "REFUSE", "category": "cancel_eligibility",
         "criteria": {"status": "cancelled"}, "build": _build_retail_rt13},
        {"task_id": 114, "direction": "REFUSE", "category": "return_eligibility",
         "criteria": {"status": "cancelled"}, "build": _build_retail_rt14},
        {"task_id": 115, "direction": "ALLOW", "category": "refund_payment",
         "criteria": {"status": "delivered", "has_gift_card": True}, "build": _build_retail_rt15},
    ]


# =====================================================================
# Dynamic DB generation
# =====================================================================

def _generate_airline_db_for_template(rng: random.Random, template: Dict) -> Dict:
    """Generate a per-seed airline DB and scenario fields."""
    criteria = dict(template["criteria"])
    # Pop flags not understood by sample_airline_user
    criteria.pop("flight_cancelled", None)
    criteria.pop("has_flown_segment", None)
    criteria.pop("has_delayed_flight", None)

    ep = sample_airline_user(rng, criteria)
    if ep is None:
        raise RuntimeError("Failed to generate airline episode")

    fields = template["build"](rng, ep)
    db = build_airline_db(ep["user"], ep["reservation"], ep["flights_db"])

    return {"db": db, "user_id": ep["user_id"],
            "reservation_id": ep["reservation_id"], **fields}


def _generate_retail_db_for_template(rng: random.Random, template: Dict) -> Dict:
    """Generate a per-seed retail DB and scenario fields."""
    criteria = dict(template["criteria"])
    criteria.pop("already_exchanged", None)
    criteria.pop("already_modified", None)
    status = criteria.get("status", "pending")

    # For cancelled status, generate as pending then flip
    gen_status = status if status != "cancelled" else "pending"
    criteria["status"] = gen_status

    ep = sample_retail_user(rng, criteria)
    if ep is None:
        raise RuntimeError("Failed to generate retail episode")

    if status == "cancelled":
        ep["order"]["status"] = "cancelled"
        ep["order"]["cancel_reason"] = "no longer needed"

    fields = template["build"](rng, ep)
    db = build_retail_db(ep["user"], ep["order"], ep["products_db"])

    return {"db": db, "user_id": ep["user_id"],
            "order_id": ep["order_id"], **fields}


# =====================================================================
# Template pools (module-level singletons)
# =====================================================================

_AIRLINE_TEMPLATES = _airline_task_templates()
_RETAIL_TEMPLATES = _retail_task_templates()


# =====================================================================
# Main scenario generation
# =====================================================================

def generate_scenario(seed: int, domain: Optional[str] = None) -> PreconditionScenario:
    """Generate a precondition verification scenario from a seed.

    Dynamic generation: each seed gets a fresh synthetic DB.
    Domain selection: 50/50 airline/retail unless domain is specified.
    Adversarial users for REFUSE scenarios (Gap 1).
    Progressive disclosure for ~30% of scenarios (Gap 6/7).
    """
    rng = random.Random(seed)

    # Domain selection
    if domain is None:
        chosen_domain = rng.choice(["airline", "retail"])
    else:
        chosen_domain = domain

    # Pick template from the appropriate pool
    if chosen_domain == "airline":
        pool = _AIRLINE_TEMPLATES
    else:
        pool = _RETAIL_TEMPLATES

    task_idx = seed % len(pool)
    template = pool[task_idx]

    # Generate dynamic DB and scenario fields
    if chosen_domain == "airline":
        result = _generate_airline_db_for_template(rng, template)
        entity_id = result.get("reservation_id", "")
        entity_label = "reservation"
    else:
        result = _generate_retail_db_for_template(rng, template)
        entity_id = result.get("order_id", "")
        entity_label = "order"

    user_id = result["user_id"]
    is_refusal = template["direction"] == "REFUSE"

    # Gap 1: Mix adversarial/cooperative across BOTH refuse and allow to prevent
    # the model learning "adversarial tone = refuse" shortcut.
    # REFUSE: 70% adversarial, 30% cooperative (policy-violating but polite users)
    # ALLOW: 30% adversarial, 70% cooperative (legitimate but pushy users)
    if is_refusal:
        approach = "adversarial" if rng.random() < 0.7 else "cooperative"
    else:
        approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    domain_label = "airline reservation" if chosen_domain == "airline" else "online retail order"
    customer_context = (
        f"Your user ID is {user_id}. "
        f"Your {entity_label} ID is {entity_id}. "
        f"You are calling about a {domain_label}."
    )

    # Gap 6/7: progressive disclosure for 30% of scenarios
    progressive_extra = ""
    if rng.random() < 0.3:
        progressive_extra = " " + rng.choice(_PROGRESSIVE_DISCLOSURE_INSTRUCTIONS)

    user_system_prompt = build_user_system_prompt(
        customer_context=customer_context,
        goal=result["user_prompt_extra"] + progressive_extra,
        approach=approach,
        rng=rng,
    )

    return PreconditionScenario(
        domain=chosen_domain,
        scenario_type=template["category"],
        task_id=template["task_id"],
        user_system_prompt=user_system_prompt,
        initial_message=result["user_msg"],
        expected_actions=result.get("expected_actions", []),
        communicate_info=result.get("communicate_info", []),
        is_refusal=is_refusal,
        description=result["description"],
        key_facts={
            "rule": result["rule"],
            "direction": template["direction"],
            "user_id": user_id,
            entity_label + "_id": entity_id,
        },
        db=result["db"],
    )
