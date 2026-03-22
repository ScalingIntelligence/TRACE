"""Multi-Step Task Game - tau2-bench Aligned (Airline + Retail)

Mid-conversation game in exact tau2-bench format. Model is placed after
auth + lookups, and must produce the correct sequence of write-action
tool calls to complete 2-5 operations.

Pre-fills the conversation with greeting, authentication, data lookups,
agent summary, and user confirmation. The model then produces tool calls
freely (writes and/or additional lookups). Each tool call is executed via
ToolExecutor with the result appended. Game ends when the model sends text
(respond_to_user) or hits max_steps.

Matches the dominant tau2-bench eval pattern (verified):
  Phase 1: Auth + lookups (pre-filled)
  Phase 2: Agent summary + user confirmation (pre-filled)
  Phase 3: Model fires write tool calls back-to-back (model's turn)

No LLM user simulator needed. Uses ToolExecutor and the same GameEnv
protocol as precondition_game and tau_tool_calling_env.

Airline operations:
  - cancel_reservation
  - update_reservation_baggages
  - update_reservation_flights (change to different flight)
  - update_reservation_passengers (correct passenger info)

Retail operations:
  - cancel_pending_order (cancel a pending order)
  - exchange_delivered_order_items (exchange items in delivered order)
  - modify_pending_order_items (modify items in pending order)
  - modify_pending_order_address (change address on pending order)
  - return_delivered_order_items (return items from delivered order)
  - modify_user_address (change user profile address)

Reward: completion bonus (0.4) + linear partial credit (max 0.6).
"""

import random
import copy
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from adversarial_policy_game.tools import ToolExecutor
from adversarial_policy_game.constants import (
    AIRLINE_POLICY,
    AIRLINE_TOOL_SCHEMAS,
    RETAIL_POLICY,
    RETAIL_TOOL_SCHEMAS,
)
from adversarial_policy_game.synthetic_db import (
    FIRST_NAMES, LAST_NAMES, CITIES_STATES_ZIPS, STREETS,
    _gen_dob, _gen_email, _gen_payment_id,
    build_airline_db,
    generate_retail_multi_orders, build_retail_db,
)


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class TaskOperation:
    """One operation the agent must complete."""
    description: str  # human-readable for user message
    tool_name: str
    tool_args: Dict[str, Any]
    key_args: List[str] = field(default_factory=list)  # arg keys to check for match


@dataclass
class MultiStepScenario:
    """A multi-step scenario with multiple operations."""
    domain: str  # "airline" or "retail"
    messages: List[Dict[str, Any]]  # Pre-filled conversation (OpenAI format)
    operations: List[TaskOperation]
    description: str = ""
    db: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Constants
# =====================================================================

# Exact tau2-bench system prompt instruction (from llm_agent.py AGENT_INSTRUCTION)
AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the "
    "<policy> provided below.\n"
    "In each turn you can either:\n"
    "- Send a message to the user.\n"
    "- Make a tool call.\n"
    "You cannot do both at the same time.\n\n"
    "Try to be helpful and always follow the policy. "
    "Always make sure you generate valid JSON only."
)

IATA_CODES = [
    "JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "SEA",
    "MIA", "EWR", "PHX", "IAH", "LAS", "PHL", "DTW", "BOS",
    "MSP", "CLT", "MCO", "TPA",
]

CABIN_CLASSES = ["basic_economy", "economy", "business"]

FREE_BAGS = {
    ("regular", "basic_economy"): 0, ("regular", "economy"): 1, ("regular", "business"): 2,
    ("silver", "basic_economy"): 1, ("silver", "economy"): 2, ("silver", "business"): 3,
    ("gold", "basic_economy"): 2, ("gold", "economy"): 3, ("gold", "business"): 4,
}


# =====================================================================
# Synthetic DB generation
# =====================================================================

def _gen_flight(rng: random.Random, origin: str, dest: str, date: str,
                base_price: float) -> Tuple[str, Dict]:
    """Generate a single flight entry."""
    fnum = f"MS{rng.randint(100, 999)}"
    hour = rng.randint(6, 21)
    minute = rng.choice([0, 15, 30, 45])
    dep_time = f"{hour:02d}:{minute:02d}:00"
    duration_h = rng.randint(2, 7)
    arr_hour = (hour + duration_h) % 24
    arr_time = f"{arr_hour:02d}:{minute:02d}:00"

    econ_price = round(base_price + rng.uniform(-40, 40), 0)
    be_price = round(econ_price * 0.65, 0)
    biz_price = round(econ_price * 2.5, 0)

    return fnum, {
        "flight_number": fnum,
        "origin": origin,
        "destination": dest,
        "scheduled_departure_time_est": dep_time,
        "scheduled_arrival_time_est": arr_time,
        "dates": {
            date: {
                "status": "available",
                "available_seats": {
                    "basic_economy": rng.randint(5, 30),
                    "economy": rng.randint(10, 50),
                    "business": rng.randint(3, 15),
                },
                "prices": {
                    "basic_economy": int(be_price),
                    "economy": int(econ_price),
                    "business": int(biz_price),
                },
            }
        },
    }


def _gen_scenario_db(rng: random.Random, n_reservations: int = 4
                      ) -> Tuple[Dict, List[Dict], Dict]:
    """Generate user with multiple reservations and flights for multi-step ops."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    uid = f"{first.lower()}_{last.lower()}_{rng.randint(1000, 9999)}"
    membership = rng.choice(["regular", "silver", "gold"])

    # Payment methods (2-3)
    payment_methods = {}
    cc_id = _gen_payment_id(rng, "credit_card")
    payment_methods[cc_id] = {
        "source": "credit_card", "id": cc_id,
        "brand": rng.choice(["visa", "mastercard", "amex"]),
        "last_four": str(rng.randint(1000, 9999)),
    }
    gc_id = _gen_payment_id(rng, "gift_card")
    payment_methods[gc_id] = {
        "source": "gift_card", "id": gc_id,
        "amount": rng.choice([200, 300, 500]),
    }
    if rng.random() < 0.5:
        cert_id = _gen_payment_id(rng, "certificate")
        payment_methods[cert_id] = {
            "source": "certificate", "id": cert_id,
            "amount": rng.choice([50, 100, 150]),
        }

    city, state, zipcode = rng.choice(CITIES_STATES_ZIPS)
    dob = _gen_dob(rng)

    user = {
        "user_id": uid,
        "name": {"first_name": first, "last_name": last},
        "address": {
            "address1": f"{rng.randint(100, 999)} {rng.choice(STREETS)}",
            "address2": "", "city": city, "state": state,
            "zip": zipcode, "country": "USA",
        },
        "email": _gen_email(first, last, rng),
        "dob": dob,
        "payment_methods": payment_methods,
        "saved_passengers": [{"first_name": first, "last_name": last, "dob": dob}],
        "membership": membership,
        "reservations": [],
    }

    flights_db = {}
    reservations = []
    used_routes = set()

    for i in range(n_reservations):
        origin = rng.choice(IATA_CODES)
        dest = rng.choice([c for c in IATA_CODES if c != origin])
        while (origin, dest) in used_routes:
            origin = rng.choice(IATA_CODES)
            dest = rng.choice([c for c in IATA_CODES if c != origin])
        used_routes.add((origin, dest))

        day = rng.randint(18, 28)
        date = f"2024-05-{day:02d}"
        cabin = rng.choice(["economy", "business"])
        base_price = rng.uniform(200, 500)

        fnum, fdata = _gen_flight(rng, origin, dest, date, base_price)
        flights_db[fnum] = fdata
        price = fdata["dates"][date]["prices"][cabin]

        # Also add 2-3 alternative flights on same route/date for flight changes
        for _ in range(rng.randint(2, 3)):
            alt_fnum, alt_fdata = _gen_flight(rng, origin, dest, date,
                                               base_price + rng.uniform(-100, 100))
            flights_db[alt_fnum] = alt_fdata

        n_pax = rng.randint(1, 2)
        passengers = [{"first_name": first, "last_name": last, "dob": dob}]
        if n_pax > 1:
            pax_first = rng.choice(FIRST_NAMES)
            passengers.append({
                "first_name": pax_first, "last_name": last,
                "dob": _gen_dob(rng),
            })

        total_bags = rng.randint(0, n_pax * 2)
        free_per_pax = FREE_BAGS.get((membership, cabin), 0)
        nonfree = max(0, total_bags - free_per_pax * n_pax)

        res_id = f"MS{rng.randint(10000, 99999)}"
        reservation = {
            "reservation_id": res_id,
            "user_id": uid,
            "origin": origin,
            "destination": dest,
            "flight_type": "one_way",
            "cabin": cabin,
            "flights": [{"flight_number": fnum, "date": date,
                         "origin": origin, "destination": dest, "price": int(price)}],
            "passengers": passengers,
            "payment_history": [{"payment_id": cc_id, "amount": int(price) * n_pax}],
            "created_at": f"2024-05-{rng.randint(1, 14):02d}T{rng.randint(8, 18):02d}:00:00",
            "total_baggages": total_bags,
            "nonfree_baggages": nonfree,
            "insurance": rng.choice(["yes", "no"]),
            "status": None,
        }
        reservations.append(reservation)
        user["reservations"].append(res_id)

    return user, reservations, flights_db


# =====================================================================
# Operation generators
# =====================================================================

def _op_cancel_reservation(rng: random.Random, res: Dict) -> Optional[TaskOperation]:
    """Cancel a reservation. Only if policy-eligible."""
    cabin = res["cabin"]
    insurance = res.get("insurance", "no")
    # Policy: can cancel if business, or has insurance, or booked within 24h
    # We set created_at within 24h for non-business/non-insured to make it valid
    if cabin != "business" and insurance != "yes":
        # Make it recently booked (within 24h of current time 2024-05-15 15:00)
        res["created_at"] = "2024-05-15T10:00:00"
    return TaskOperation(
        description=f"cancel reservation {res['reservation_id']}",
        tool_name="cancel_reservation",
        tool_args={"reservation_id": res["reservation_id"]},
        key_args=["reservation_id"],
    )


def _op_update_baggages(rng: random.Random, res: Dict, user: Dict
                         ) -> Optional[TaskOperation]:
    """Add bags to a reservation."""
    n_pax = len(res["passengers"])
    membership = user["membership"]
    cabin = res["cabin"]
    current_bags = res["total_baggages"]
    free_per_pax = FREE_BAGS.get((membership, cabin), 0)

    # Add 1-2 bags
    bags_to_add = rng.randint(1, 2)
    new_total = current_bags + bags_to_add
    new_nonfree = max(0, new_total - free_per_pax * n_pax)

    # Pick a payment method
    pm_id = list(user["payment_methods"].keys())[0]

    return TaskOperation(
        description=(
            f"add {bags_to_add} checked bag{'s' if bags_to_add > 1 else ''} "
            f"to reservation {res['reservation_id']}"
        ),
        tool_name="update_reservation_baggages",
        tool_args={
            "reservation_id": res["reservation_id"],
            "total_baggages": new_total,
            "nonfree_baggages": new_nonfree,
            "payment_id": pm_id,
        },
        key_args=["reservation_id", "total_baggages", "nonfree_baggages"],
    )


def _op_change_flight(rng: random.Random, res: Dict, flights_db: Dict,
                       user: Dict) -> Optional[TaskOperation]:
    """Change to a different flight on the same route."""
    current_flight = res["flights"][0]
    fnum = current_flight["flight_number"]
    origin = res["origin"]
    dest = res["destination"]
    date = current_flight["date"]
    cabin = res["cabin"]

    # Find alternative flights on same route+date
    alternatives = []
    for fn, fdata in flights_db.items():
        if (fn != fnum and fdata["origin"] == origin and
            fdata["destination"] == dest and date in fdata["dates"] and
            fdata["dates"][date].get("status") == "available"):
            alternatives.append((fn, fdata))

    if not alternatives:
        return None

    new_flight = rng.choice(alternatives)
    new_fnum = new_flight[0]
    new_price = new_flight[1]["dates"][date]["prices"][cabin]

    pm_id = list(user["payment_methods"].keys())[0]

    # Vague description: don't give exact flight number, model must pick from search results
    dep_time = new_flight[1]["scheduled_departure_time_est"]
    return TaskOperation(
        description=(
            f"change reservation {res['reservation_id']} to a different "
            f"{cabin} flight departing around {dep_time[:5]}"
        ),
        tool_name="update_reservation_flights",
        tool_args={
            "reservation_id": res["reservation_id"],
            "cabin": cabin,
            "flights": [{"flight_number": new_fnum, "date": date,
                         "origin": origin, "destination": dest,
                         "price": int(new_price)}],
            "payment_id": pm_id,
        },
        key_args=["reservation_id", "cabin", "flights"],
    )


def _op_update_passenger(rng: random.Random, res: Dict) -> Optional[TaskOperation]:
    """Correct passenger DOB or name."""
    if not res["passengers"]:
        return None

    pax = rng.choice(res["passengers"])
    # Fix DOB
    old_dob = pax["dob"]
    # Generate a slightly different DOB (correction scenario)
    new_dob = _gen_dob(rng)
    while new_dob == old_dob:
        new_dob = _gen_dob(rng)

    new_passengers = copy.deepcopy(res["passengers"])
    for p in new_passengers:
        if p["first_name"] == pax["first_name"] and p["last_name"] == pax["last_name"]:
            p["dob"] = new_dob
            break

    return TaskOperation(
        description=(
            f"correct the date of birth for {pax['first_name']} {pax['last_name']} "
            f"on reservation {res['reservation_id']} to {new_dob}"
        ),
        tool_name="update_reservation_passengers",
        tool_args={
            "reservation_id": res["reservation_id"],
            "passengers": new_passengers,
        },
        key_args=["reservation_id"],
    )


def _op_send_certificate(rng: random.Random, user: Dict,
                          res: Dict, flights_db: Dict) -> Optional[TaskOperation]:
    """Send a travel certificate. Only valid when flight is cancelled/delayed
    AND user is silver/gold or has insurance or flies business."""
    membership = user["membership"]
    cabin = res["cabin"]
    insurance = res.get("insurance", "no")

    # Check eligibility
    eligible = (membership in ("silver", "gold") or
                insurance == "yes" or cabin == "business")
    if not eligible:
        return None

    # Set the flight to cancelled so the request is valid
    flight_info = res["flights"][0]
    fnum = flight_info["flight_number"]
    date = flight_info["date"]
    flights_db[fnum]["dates"][date] = {"status": "cancelled"}

    n_pax = len(res["passengers"])
    # Policy: $100 per passenger for cancelled flights
    amount = 100 * n_pax

    return TaskOperation(
        description=(
            f"my flight {fnum} on reservation {res['reservation_id']} was cancelled, "
            f"I'd like a travel certificate as compensation"
        ),
        tool_name="send_certificate",
        tool_args={
            "user_id": user["user_id"],
            "amount": amount,
        },
        key_args=["user_id", "amount"],
    )


def _op_book_reservation(rng: random.Random, user: Dict,
                          flights_db: Dict) -> Optional[TaskOperation]:
    """Book a new reservation on a specific route.

    Mirrors eval failure: book_reservation with wrong flight/payment (10% of ops).
    """
    origin = rng.choice(IATA_CODES)
    dest = rng.choice([c for c in IATA_CODES if c != origin])
    day = rng.randint(18, 25)
    date = f"2024-05-{day:02d}"
    cabin = rng.choice(["economy", "business"])

    # Generate 3-4 flights for search results
    search_flights = []
    used_prices = set()
    for _ in range(rng.randint(3, 4)):
        fnum, fdata = _gen_flight(rng, origin, dest, date,
                                   base_price=rng.uniform(200, 500))
        p = fdata["dates"][date]["prices"][cabin]
        while p in used_prices:
            p += 1
        used_prices.add(p)
        fdata["dates"][date]["prices"][cabin] = p
        flights_db[fnum] = fdata
        search_flights.append((fnum, fdata))

    pm_id = list(user["payment_methods"].keys())[0]

    return TaskOperation(
        description=(
            f"book the cheapest {cabin} flight from {origin} to {dest} on May {day}"
        ),
        tool_name="book_reservation",
        tool_args={
            "user_id": user["user_id"],
            "origin": origin,
            "destination": dest,
            "flight_type": "one_way",
            "cabin": cabin,
            "payment_id": pm_id,
        },
        key_args=["user_id", "origin", "destination", "cabin"],
    )


# =====================================================================
# Scenario generation
# =====================================================================

# Weighted by eval failure frequency (exact match):
# update_reservation_flights: 44%, cancel_reservation: 28%, baggages: 13%, book: 10%, passenger: 5%
_OP_GENERATORS = [
    ("flight_change", _op_change_flight, 44),
    ("cancel", _op_cancel_reservation, 28),
    ("baggages", _op_update_baggages, 13),
    ("book", _op_book_reservation, 10),
    ("passenger", _op_update_passenger, 5),
]


# =====================================================================
# Retail operation generators
# =====================================================================

CANCEL_REASONS = ["no longer needed", "ordered by mistake"]


def _describe_options(options: Dict[str, str]) -> str:
    """Build a human-readable string from item options dict, e.g. 'blue, large, cotton'."""
    return ", ".join(str(v) for v in options.values())


def _pick_different_variant(rng: random.Random, product_entry: Dict,
                            current_item_id: str) -> Optional[Dict]:
    """Pick an available variant different from current_item_id."""
    variants = product_entry.get("variants", {})
    candidates = [
        v for vid, v in variants.items()
        if vid != current_item_id and v.get("available", False)
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _retail_op_cancel(rng: random.Random, order: Dict, user: Dict,
                      products_db: Dict) -> Optional[TaskOperation]:
    """Cancel a pending order."""
    if order["status"] != "pending":
        return None
    reason = rng.choice(CANCEL_REASONS)
    return TaskOperation(
        description=f"cancel order {order['order_id']} (reason: {reason})",
        tool_name="cancel_pending_order",
        tool_args={"order_id": order["order_id"], "reason": reason},
        key_args=["order_id", "reason"],
    )


def _retail_op_exchange(rng: random.Random, order: Dict, user: Dict,
                        products_db: Dict) -> Optional[TaskOperation]:
    """Exchange an item in a delivered order for a different variant."""
    if order["status"] != "delivered":
        return None
    if not order["items"]:
        return None

    item = rng.choice(order["items"])
    product_id = item["product_id"]
    product_entry = products_db.get(product_id)
    if not product_entry:
        return None

    new_variant = _pick_different_variant(rng, product_entry, item["item_id"])
    if not new_variant:
        return None

    pm_id = list(user["payment_methods"].keys())[0]
    # Vague description: mention one distinguishing attribute of the target variant
    new_opts = new_variant["options"]
    # Pick one attribute to mention (the model must figure out the rest)
    attr_keys = list(new_opts.keys())
    hint_key = rng.choice(attr_keys) if attr_keys else ""
    hint_val = new_opts.get(hint_key, "")

    return TaskOperation(
        description=(
            f"exchange the {item['name']} in order {order['order_id']} "
            f"for one that is {hint_val}"
        ),
        tool_name="exchange_delivered_order_items",
        tool_args={
            "order_id": order["order_id"],
            "item_ids": [item["item_id"]],
            "new_item_ids": [new_variant["item_id"]],
            "payment_method_id": pm_id,
        },
        key_args=["order_id", "item_ids", "new_item_ids"],
    )


def _retail_op_modify_items(rng: random.Random, order: Dict, user: Dict,
                            products_db: Dict) -> Optional[TaskOperation]:
    """Modify an item in a pending order to a different variant."""
    if order["status"] != "pending":
        return None
    if not order["items"]:
        return None

    item = rng.choice(order["items"])
    product_id = item["product_id"]
    product_entry = products_db.get(product_id)
    if not product_entry:
        return None

    new_variant = _pick_different_variant(rng, product_entry, item["item_id"])
    if not new_variant:
        return None

    pm_id = list(user["payment_methods"].keys())[0]
    # Vague description: mention one distinguishing attribute
    new_opts = new_variant["options"]
    attr_keys = list(new_opts.keys())
    hint_key = rng.choice(attr_keys) if attr_keys else ""
    hint_val = new_opts.get(hint_key, "")

    return TaskOperation(
        description=(
            f"modify the {item['name']} in order {order['order_id']} "
            f"to one that is {hint_val}"
        ),
        tool_name="modify_pending_order_items",
        tool_args={
            "order_id": order["order_id"],
            "item_ids": [item["item_id"]],
            "new_item_ids": [new_variant["item_id"]],
            "payment_method_id": pm_id,
        },
        key_args=["order_id", "item_ids", "new_item_ids"],
    )


def _retail_op_modify_address(rng: random.Random, order: Dict, user: Dict,
                              products_db: Dict) -> Optional[TaskOperation]:
    """Change shipping address on a pending order."""
    if order["status"] != "pending":
        return None

    city, state, zipcode = rng.choice(CITIES_STATES_ZIPS)
    new_addr = {
        "address1": f"{rng.randint(100, 999)} {rng.choice(STREETS)}",
        "address2": "",
        "city": city,
        "state": state,
        "country": "USA",
        "zip": zipcode,
    }

    return TaskOperation(
        description=(
            f"change the shipping address on order {order['order_id']} "
            f"to {new_addr['address1']}, {city}, {state} {zipcode}"
        ),
        tool_name="modify_pending_order_address",
        tool_args={
            "order_id": order["order_id"],
            **new_addr,
        },
        key_args=["order_id", "city", "zip"],
    )


def _retail_op_return(rng: random.Random, order: Dict, user: Dict,
                      products_db: Dict) -> Optional[TaskOperation]:
    """Return an item from a delivered order."""
    if order["status"] != "delivered":
        return None
    if not order["items"]:
        return None

    item = rng.choice(order["items"])
    pm_id = list(user["payment_methods"].keys())[0]

    return TaskOperation(
        description=f"return the {item['name']} from order {order['order_id']}",
        tool_name="return_delivered_order_items",
        tool_args={
            "order_id": order["order_id"],
            "item_ids": [item["item_id"]],
            "payment_method_id": pm_id,
        },
        key_args=["order_id", "item_ids"],
    )


def _retail_op_modify_user_address(rng: random.Random, order: Dict, user: Dict,
                                    products_db: Dict) -> Optional[TaskOperation]:
    """Change the user's profile address."""
    city, state, zipcode = rng.choice(CITIES_STATES_ZIPS)
    new_addr = {
        "address1": f"{rng.randint(100, 999)} {rng.choice(STREETS)}",
        "address2": "",
        "city": city,
        "state": state,
        "country": "USA",
        "zip": zipcode,
    }
    return TaskOperation(
        description=(
            f"update my profile address to "
            f"{new_addr['address1']}, {city}, {state} {zipcode}"
        ),
        tool_name="modify_user_address",
        tool_args={"user_id": user["user_id"], **new_addr},
        key_args=["user_id"],
    )


# Weighted by eval failure frequency (from verified multi-step failed tasks):
# modify_items: 27%, return: 20%, modify_address: 20%, exchange: 13%, cancel: 12%, modify_user_address: 7%
_RETAIL_OP_GENERATORS = [
    ("cancel", _retail_op_cancel, "pending", 12),
    ("exchange", _retail_op_exchange, "delivered", 13),
    ("modify_items", _retail_op_modify_items, "pending", 27),
    ("modify_address", _retail_op_modify_address, "pending", 20),
    ("return", _retail_op_return, "delivered", 20),
    ("modify_user_address", _retail_op_modify_user_address, "any", 7),
]


# =====================================================================
# Pre-filled conversation builder
# =====================================================================

def _build_prefilled_conversation(
    scenario_domain: str,
    user: Dict,
    operations: List[TaskOperation],
    db: Dict,
) -> List[Dict[str, Any]]:
    """Build pre-filled conversation in OpenAI message format.

    Executes auth + data lookups via ToolExecutor, building the same
    message sequence the model would see mid-conversation in tau2-bench eval.
    """
    te = ToolExecutor(scenario_domain, copy.deepcopy(db))
    msgs: List[Dict[str, Any]] = []
    call_id = 0

    def _add_tool_call(name: str, args: Dict) -> str:
        nonlocal call_id
        cid = f"call_{call_id:04d}"
        call_id += 1
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }],
        })
        result = te.execute(name, args)
        msgs.append({
            "role": "tool",
            "content": result,
            "tool_call_id": cid,
        })
        return result

    # Phase 1: Greeting
    if scenario_domain == "retail":
        msgs.append({"role": "assistant", "content": "Hi! How can I help you today?"})
    else:
        msgs.append({"role": "assistant", "content":
                      "Welcome to our airline customer service. How can I assist you today?"})

    # Phase 2: User request with all tasks
    name_d = user["name"]
    task_lines = [op.description for op in operations]
    task_text = "; ".join(task_lines)

    if scenario_domain == "retail":
        zipcode = user["address"]["zip"]
        user_msg = (
            f"Hi, I'm {name_d['first_name']} {name_d['last_name']}, "
            f"zip code {zipcode}. I need help with the following: {task_text}."
        )
    else:
        uid = user["user_id"]
        user_msg = (
            f"Hi, I'm {name_d['first_name']} {name_d['last_name']}. "
            f"My user ID is {uid}. I need help with: {task_text}."
        )
    msgs.append({"role": "user", "content": user_msg})

    # Phase 3: Auth lookups
    if scenario_domain == "retail":
        _add_tool_call("find_user_id_by_name_zip", {
            "first_name": name_d["first_name"],
            "last_name": name_d["last_name"],
            "zip": user["address"]["zip"],
        })
    _add_tool_call("get_user_details", {"user_id": user["user_id"]})

    # Phase 4: Data lookups
    if scenario_domain == "retail":
        # Look up all orders
        for oid in user.get("orders", []):
            _add_tool_call("get_order_details", {"order_id": oid})
        # Look up products referenced by exchange/modify operations
        seen_products: set = set()
        for op in operations:
            if op.tool_name in ("exchange_delivered_order_items",
                                "modify_pending_order_items"):
                for oid in user.get("orders", []):
                    order_data = db.get("orders", {}).get(oid, {})
                    for item in order_data.get("items", []):
                        pid = item.get("product_id")
                        if pid and pid not in seen_products:
                            _add_tool_call("get_product_details",
                                           {"product_id": pid})
                            seen_products.add(pid)
    else:
        # Airline: look up all reservations
        for res_id in user.get("reservations", []):
            _add_tool_call("get_reservation_details",
                           {"reservation_id": res_id})
        # Search for flights if any flight change or book operations
        searched_routes: set = set()
        for op in operations:
            if op.tool_name in ("update_reservation_flights",
                                "book_reservation"):
                flight_args = op.tool_args.get("flights", [])
                origin = (flight_args[0].get("origin", "")
                          if flight_args
                          else op.tool_args.get("origin", ""))
                dest = (flight_args[0].get("destination", "")
                        if flight_args
                        else op.tool_args.get("destination", ""))
                route = (origin, dest)
                if origin and dest and route not in searched_routes:
                    _add_tool_call("search_direct_flight",
                                   {"origin": origin, "destination": dest})
                    searched_routes.add(route)

    # Phase 5: Agent summary + user confirmation
    op_summaries = []
    for i, op in enumerate(operations, 1):
        op_summaries.append(f"{i}. {op.description}")
    summary = "\n".join(op_summaries)

    msgs.append({
        "role": "assistant",
        "content": (
            f"I've reviewed your account and found the relevant information. "
            f"I'll process the following for you:\n{summary}\n\n"
            f"Shall I proceed with all of these?"
        ),
    })
    msgs.append({"role": "user", "content": "Yes, please proceed with all of them."})

    return msgs


def _generate_retail_scenario(rng: random.Random) -> MultiStepScenario:
    """Generate a multi-step retail scenario with 2-5 operations."""
    # Eval failure distribution (verified from qwen-3-30b base results):
    # 48% 2-op, 39% 3-op, 10% 4-op, 3% 5-op
    n_ops_target = rng.choices([2, 3, 4, 5], weights=[48, 39, 10, 3])[0]

    # We need a mix of pending and delivered orders for diverse ops.
    # Generate enough orders: at least n_ops_target, with a mix of statuses.
    n_pending = max(2, n_ops_target // 2 + 1)
    n_delivered = max(2, n_ops_target - n_pending + 1)
    order_specs = (
        [{"status": "pending", "min_items": 2} for _ in range(n_pending)]
        + [{"status": "delivered", "min_items": 2} for _ in range(n_delivered)]
    )
    rng.shuffle(order_specs)

    result = generate_retail_multi_orders(rng, {}, order_specs=order_specs)
    user = result["user"]
    orders = result["orders"]
    products_db = result["products_db"]
    name = user["name"]

    # Build operations using weighted selection matching eval failure distribution
    operations: List[TaskOperation] = []
    used_order_ids: set = set()

    attempts = 0
    while len(operations) < n_ops_target and attempts < 30:
        attempts += 1
        # Pure weighted selection (no "prefer unused" bias)
        idx = rng.choices(range(len(_RETAIL_OP_GENERATORS)),
                          weights=[t[3] for t in _RETAIL_OP_GENERATORS])[0]
        op_type, gen_fn, required_status, _ = _RETAIL_OP_GENERATORS[idx]

        if required_status == "any":
            candidates = [
                o for o in orders
                if o["order_id"] not in used_order_ids
            ]
        else:
            candidates = [
                o for o in orders
                if o["order_id"] not in used_order_ids and o["status"] == required_status
            ]
        if not candidates:
            # modify_user_address doesn't need an order
            if required_status == "any":
                op = gen_fn(rng, orders[0] if orders else {}, user, products_db)
                if op is not None:
                    operations.append(op)
                continue
            continue

        order = rng.choice(candidates)
        op = gen_fn(rng, order, user, products_db)
        if op is not None:
            operations.append(op)
            used_order_ids.add(order["order_id"])

    # Pad if needed
    while len(operations) < n_ops_target:
        unused = [o for o in orders if o["order_id"] not in used_order_ids]
        if not unused:
            break
        order = rng.choice(unused)
        matching = [
            (t, fn, w) for t, fn, st, w in _RETAIL_OP_GENERATORS
            if st == order["status"] or st == "any"
        ]
        if not matching:
            used_order_ids.add(order["order_id"])
            continue
        m_weights = [t[2] for t in matching]
        m_fns = [t[1] for t in matching]
        idx = rng.choices(range(len(matching)), weights=m_weights)[0]
        gen_fn = m_fns[idx]
        op = gen_fn(rng, order, user, products_db)
        if op is not None:
            operations.append(op)
        used_order_ids.add(order["order_id"])

    if not operations:
        # Fallback: at minimum generate a cancel for a pending order
        for order in orders:
            if order["status"] == "pending":
                op = _retail_op_cancel(rng, order, user, products_db)
                if op:
                    operations.append(op)
                    break

    db = build_retail_db(user, orders, products_db)

    messages = _build_prefilled_conversation("retail", user, operations, db)

    op_types = [op.tool_name for op in operations]
    return MultiStepScenario(
        domain="retail",
        messages=messages,
        operations=operations,
        description=f"Multi-step retail: {len(operations)} ops ({', '.join(op_types)})",
        db=db,
    )


def generate_scenario(seed: int, domain: Optional[str] = None) -> MultiStepScenario:
    """Generate a multi-step scenario with 3-5 operations.

    Args:
        seed: Random seed for reproducibility.
        domain: "airline", "retail", or None (50/50 random).
    """
    rng = random.Random(seed)

    # Choose domain
    if domain is None:
        chosen_domain = rng.choice(["airline", "retail"])
    else:
        chosen_domain = domain

    if chosen_domain == "retail":
        return _generate_retail_scenario(rng)

    # --- Airline path ---
    # Eval failure distribution: 50% 2-op, 29% 3-op, 14% 4-op, 7% 5-op
    n_ops_target = rng.choices([2, 3, 4, 5], weights=[50, 29, 14, 7])[0]
    n_reservations = max(n_ops_target, 3)

    user, reservations, flights_db = _gen_scenario_db(rng, n_reservations)
    uid = user["user_id"]
    name = user["name"]

    # Build operations using weighted selection matching eval failure distribution.
    # Use raw weights every pick (no "prefer unused" bias) to preserve distribution.
    operations: List[TaskOperation] = []
    used_res_ids = set()

    attempts = 0
    while len(operations) < n_ops_target and attempts < 30:
        attempts += 1
        # Pure weighted selection from all op types
        idx = rng.choices(range(len(_OP_GENERATORS)), weights=[t[2] for t in _OP_GENERATORS])[0]
        op_type, gen_fn, _ = _OP_GENERATORS[idx]

        unused = [r for r in reservations if r["reservation_id"] not in used_res_ids]

        op = None
        if op_type == "book":
            op = gen_fn(rng, user, flights_db)
        elif op_type in ("cancel", "passenger"):
            res = rng.choice(unused) if unused else None
            if res:
                op = gen_fn(rng, res)
                if op:
                    used_res_ids.add(res["reservation_id"])
        elif op_type == "baggages":
            res = rng.choice(unused) if unused else None
            if res:
                op = gen_fn(rng, res, user)
                if op:
                    used_res_ids.add(res["reservation_id"])
        elif op_type == "flight_change":
            res = rng.choice(unused) if unused else None
            if res:
                op = gen_fn(rng, res, flights_db, user)
                if op:
                    used_res_ids.add(res["reservation_id"])

        if op is not None:
            operations.append(op)

    # Pad if needed — same weights
    while len(operations) < n_ops_target:
        unused = [r for r in reservations if r["reservation_id"] not in used_res_ids]
        if not unused:
            break
        res = rng.choice(unused)
        idx = rng.choices(range(len(_OP_GENERATORS)), weights=[t[2] for t in _OP_GENERATORS])[0]
        op_type, gen_fn, _ = _OP_GENERATORS[idx]
        if op_type == "book":
            op = gen_fn(rng, user, flights_db)
        elif op_type == "flight_change":
            op = gen_fn(rng, res, flights_db, user)
        elif op_type == "baggages":
            op = gen_fn(rng, res, user)
        else:
            op = gen_fn(rng, res)
        if op is None:
            op = _op_cancel_reservation(rng, res)
        if op is not None:
            operations.append(op)
            used_res_ids.add(res["reservation_id"])

    db = build_airline_db(user, reservations, flights_db)

    messages = _build_prefilled_conversation("airline", user, operations, db)

    op_types = [op.tool_name for op in operations]
    return MultiStepScenario(
        domain="airline",
        messages=messages,
        operations=operations,
        description=f"Multi-step: {len(operations)} ops ({', '.join(op_types)})",
        db=db,
    )


# =====================================================================
# Reward computation
# =====================================================================

def _match_operation(call_name: str, call_args: Dict, op: TaskOperation) -> bool:
    """Check if a tool call matches an expected operation."""
    if call_name != op.tool_name:
        return False

    # Check key args
    for key in op.key_args:
        expected = op.tool_args.get(key)
        actual = call_args.get(key)
        if expected is None:
            continue

        # Handle different types
        if isinstance(expected, (int, float)):
            try:
                if abs(float(actual) - float(expected)) > 0.01:
                    return False
            except (ValueError, TypeError):
                return False
        elif isinstance(expected, str):
            if str(actual).strip() != str(expected).strip():
                return False
        # For list args: compare contents
        elif isinstance(expected, list):
            if not actual:
                return False
            # For simple ID lists (item_ids, new_item_ids): compare as sets
            if all(isinstance(x, str) for x in expected):
                if set(str(x) for x in expected) != set(str(x) for x in (actual if isinstance(actual, list) else [])):
                    return False
            # For dicts (flights, passengers): compare key fields
            elif all(isinstance(x, dict) for x in expected):
                if not isinstance(actual, list) or len(actual) != len(expected):
                    return False
                for exp_item, act_item in zip(expected, actual):
                    if isinstance(act_item, dict):
                        # For flights: check flight_number
                        if "flight_number" in exp_item:
                            if exp_item.get("flight_number") != act_item.get("flight_number"):
                                return False
                    elif str(exp_item) != str(act_item):
                        return False

    return True


def compute_reward(tool_calls: List[Dict], operations: List[TaskOperation]
                    ) -> Tuple[float, str]:
    """Compute reward based on operation completion.

    Near-binary reward to avoid the partial-credit trap:
      - All ops correct:  1.0  (full success)
      - All-but-one correct:  0.1  (near miss — small positive signal per AWM paper)
      - Otherwise:  0.0  (no partial credit for distant attempts)

    The 0.1 near-miss signal (following AWM: Agent World Model) is intentionally
    small: just enough to give slight GRPO advantage over total failure, but too
    small to become a stable attractor. A higher value (e.g. 0.5) risks creating
    a new constant-group trap where the model settles at "almost done."

    Wrong-call penalty: -0.25 per wrong call (applied to non-zero rewards).
    """
    n_ops = len(operations)
    if n_ops == 0:
        return 0.0, "No operations expected."

    completed = [False] * n_ops
    wrong_calls = 0

    # Filter to write-only calls (skip lookups)
    LOOKUP_TOOLS = {
        # Airline lookups
        "get_user_details", "get_reservation_details", "search_direct_flight",
        "search_onestop_flight", "get_flight_status", "list_all_airports",
        # Retail lookups
        "find_user_id_by_name_zip", "find_user_id_by_email",
        "get_order_details", "get_product_details", "list_all_product_types",
        # Shared
        "calculate", "transfer_to_human_agents",
    }

    for call in tool_calls:
        call_name = call.get("name", "")
        call_args = call.get("arguments", {})

        if call_name in LOOKUP_TOOLS or call_name == "respond_to_user":
            continue

        matched = False
        for i, op in enumerate(operations):
            if not completed[i] and _match_operation(call_name, call_args, op):
                completed[i] = True
                matched = True
                break

        if not matched:
            wrong_calls += 1

    correct_count = sum(completed)

    if correct_count == n_ops:
        reward = 1.0
    elif correct_count == n_ops - 1 and correct_count > 0:
        reward = 0.1
    else:
        reward = 0.0

    # Penalty for wrong calls (only applies to non-zero rewards)
    if reward > 0:
        reward -= 0.25 * wrong_calls
        reward = max(0.0, round(reward, 3))

    details = []
    for i, (op, done) in enumerate(zip(operations, completed)):
        details.append(f"Op{i+1} ({op.tool_name}): {'DONE' if done else 'MISSED'}")
    if wrong_calls:
        details.append(f"Wrong calls: {wrong_calls}")

    reason = f"{correct_count}/{n_ops} ops completed. " + "; ".join(details)
    return reward, reason


# =====================================================================
# Game class
# =====================================================================

class RealisticMultiStepGame:
    """Mid-conversation multi-step task game in tau2-bench format.

    Pre-fills conversation with auth + lookups (Phase 1-2).
    Model produces tool calls freely in Phase 3 (writes + optional lookups).
    No LLM user simulator needed.

    Implements the tool-calling game interface (supports_structured_messages):
    - get_system_prompt() / get_messages() / get_tool_schemas() / step()
    """

    supports_structured_messages = True

    def __init__(self, max_steps: int = 15,
                 user_client=None,  # Accepted but ignored (backward compat)
                 domain: Optional[str] = None):
        self._max_steps = max_steps
        self._domain = domain  # None = 50/50, "airline", or "retail"

        # GameEnv protocol
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        # Internal state
        self._scenario: Optional[MultiStepScenario] = None
        self._tools: Optional[ToolExecutor] = None
        self._conversation: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._call_id_counter: int = 0
        self._last_call_key: Optional[str] = None
        self._repeat_count: int = 0

    def reset(self, seed: int) -> None:
        """Reset with new seed."""
        self._scenario = generate_scenario(seed, domain=self._domain)
        self._tools = ToolExecutor(
            self._scenario.domain,
            copy.deepcopy(self._scenario.db),
        )
        self._conversation = copy.deepcopy(self._scenario.messages)
        self._tool_calls = []
        self._step_count = 0
        # Start call_id counter after pre-filled messages
        self._call_id_counter = sum(
            1 for m in self._conversation if m.get("tool_calls")
        )
        self._last_call_key = None
        self._repeat_count = 0

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    # -----------------------------------------------------------------
    # Structured message interface
    # -----------------------------------------------------------------

    def get_system_prompt(self) -> str:
        policy = RETAIL_POLICY if self._scenario and self._scenario.domain == "retail" else AIRLINE_POLICY
        return (
            "<instructions>\n"
            f"{AGENT_INSTRUCTION}\n"
            "</instructions>\n"
            "<policy>\n"
            f"{policy}\n"
            "</policy>"
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._scenario and self._scenario.domain == "retail":
            return RETAIL_TOOL_SCHEMAS
        return AIRLINE_TOOL_SCHEMAS

    def get_messages(self) -> List[Dict[str, Any]]:
        return list(self._conversation)

    # -----------------------------------------------------------------
    # GameEnv protocol
    # -----------------------------------------------------------------

    def observe(self, player_id: int) -> str:
        return "This game uses tool-calling interface, not observe()."

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]) -> None:
        if self.done:
            return

        self._step_count += 1

        if action is None:
            self._finalize(0.0, "No action provided")
            return

        tool_call = _parse_tool_call(action)
        if tool_call is None:
            self._finalize(0.0, "Invalid JSON format")
            self.invalid_player = 0
            return

        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        # --- Text response to user -> end game ---
        if tool_name in ("respond_to_user", "send_message"):
            reward, reason = compute_reward(self._tool_calls, self._scenario.operations)
            self._finalize(reward, reason)
            return

        # --- Transfer -> end game ---
        if tool_name == "transfer_to_human_agents":
            reward, reason = compute_reward(self._tool_calls, self._scenario.operations)
            self._finalize(reward, f"Transferred. {reason}")
            return

        # --- Loop detection ---
        call_key = json.dumps(tool_call, sort_keys=True)
        if call_key == self._last_call_key:
            self._repeat_count += 1
            if self._repeat_count >= 3:
                reward, reason = compute_reward(self._tool_calls, self._scenario.operations)
                self._finalize(reward, f"Loop detected. {reason}")
                return
        else:
            self._last_call_key = call_key
            self._repeat_count = 0

        # --- Track tool call for reward ---
        self._tool_calls.append({"name": tool_name, "arguments": tool_args})

        # --- Execute tool via ToolExecutor ---
        cid = f"call_{self._call_id_counter:04d}"
        self._call_id_counter += 1

        self._conversation.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args),
                },
            }],
        })

        try:
            result = self._tools.execute(tool_name, tool_args)
        except Exception as e:
            result = json.dumps({"error": str(e)})

        self._conversation.append({
            "role": "tool",
            "content": result,
            "tool_call_id": cid,
        })

        # --- Max steps check ---
        if self._step_count >= self._max_steps:
            reward, reason = compute_reward(self._tool_calls, self._scenario.operations)
            self._finalize(reward, f"Max steps. {reason}")

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    def get_summary(self) -> Dict[str, Any]:
        return {
            "n_ops": len(self._scenario.operations) if self._scenario else 0,
            "description": self._scenario.description if self._scenario else "",
            "domain": self._scenario.domain if self._scenario else "",
            "steps": self._step_count,
            "tool_calls": self._tool_calls,
            "reward": self.rewards.get(0, 0.0),
            "reason": getattr(self, "_reason", ""),
        }


# =====================================================================
# Action parsing
# =====================================================================

def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    try:
        depth = 0
        start = -1
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if "name" in obj:
                            return {
                                "name": obj["name"],
                                "arguments": obj.get("arguments", obj.get("parameters", {})),
                            }
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass
    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    tool_call = _parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"</?tool_call>", "", clean).strip()
    tool_call = _parse_tool_call(clean)
    if tool_call:
        return json.dumps(tool_call)
    if clean:
        return json.dumps({"name": "respond_to_user", "arguments": {"message": clean}})
    return None


# =====================================================================
# System prompt (for game registry — not used for tool-calling games)
# =====================================================================

SYSTEM_PROMPT = ""


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    print("Testing Multi-Step Task Game (tau2-bench aligned)")
    print("=" * 70)

    for test_domain in ["airline", "retail"]:
        print(f"\n{'=' * 70}")
        print(f"Domain: {test_domain}")
        print(f"{'=' * 70}")

        op_counts = {}
        for seed in range(100):
            scenario = generate_scenario(seed, domain=test_domain)
            n = len(scenario.operations)
            op_counts[n] = op_counts.get(n, 0) + 1
            if seed < 5:
                print(f"\nSeed {seed}: {scenario.description}")
                print(f"  Pre-filled messages: {len(scenario.messages)}")
                print(f"  Operations:")
                for op in scenario.operations:
                    print(f"    - {op.tool_name}: {op.description}")
                db_keys = list(scenario.db.keys())
                print(f"  DB keys: {db_keys}, users={list(scenario.db['users'].keys())}")
                # Show message roles
                roles = [m['role'] + ('(tc)' if m.get('tool_calls') else '')
                         for m in scenario.messages]
                print(f"  Message roles: {roles}")

        print(f"\nOp count distribution (100 seeds): {op_counts}")

        # Verify ToolExecutor compatibility
        from adversarial_policy_game.tools import ToolExecutor
        for seed in range(20):
            scenario = generate_scenario(seed, domain=test_domain)
            try:
                tools = ToolExecutor(test_domain, copy.deepcopy(scenario.db))
            except Exception as e:
                print(f"FAIL seed {seed} ({test_domain}): {e}")
                break
        else:
            print(f"All 20 {test_domain} DBs validated by ToolExecutor.")

    # Test 50/50 mixed mode
    print(f"\n{'=' * 70}")
    print("Mixed mode (domain=None)")
    domain_counts = {"airline": 0, "retail": 0}
    for seed in range(100):
        scenario = generate_scenario(seed, domain=None)
        domain_counts[scenario.domain] += 1
    print(f"Domain distribution (100 seeds): {domain_counts}")
