"""Scenario generation for the tau bench tool-calling microenvironment.

Generates simplified tau bench tasks from seeds. Each scenario produces:
  - user_system_prompt: fed to the LLM user simulator
  - initial_message: first customer message
  - expected_actions: list of tool calls to replay for gold DB hash
  - communicate_info: strings the agent must communicate
  - domain: "airline" or "retail"
  - is_refusal: True if the correct action is to refuse

Three scenario types:
  Type 1 (50%): Single-action explicit — user states exactly what they want
  Type 2 (30%): Single-action selection — user needs model to find right option
  Type 3 (20%): Policy-gated — action may or may not be allowed (50/50)
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adversarial_policy_game.database import (
    sample_airline_user,
    sample_retail_user,
)
from adversarial_policy_game.synthetic_db import build_airline_db, build_retail_db
from adversarial_policy_game.llm_user import build_user_system_prompt


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class ExpectedAction:
    """A tool call expected in the gold trajectory."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class GeneratedScenario:
    """A generated training scenario."""
    domain: str  # "airline" or "retail"
    scenario_type: str  # "explicit", "selection", "policy_gated"
    user_system_prompt: str
    initial_message: str
    expected_actions: List[ExpectedAction] = field(default_factory=list)
    communicate_info: List[str] = field(default_factory=list)
    is_refusal: bool = False
    description: str = ""
    key_facts: Dict[str, Any] = field(default_factory=dict)
    db: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# User prompt styles
# =====================================================================

_COOPERATIVE_STYLES = [
    "You are polite and cooperative.",
    "You are concise and business-like.",
    "You are friendly and patient.",
    "You are direct and efficient.",
    "You are calm and straightforward.",
]

_INSISTENT_STYLES = [
    "You are polite but insistent.",
    "You are determined but reasonable.",
    "You are firm but courteous.",
]


def _pick_style(rng: random.Random, cooperative: bool = True) -> str:
    pool = _COOPERATIVE_STYLES if cooperative else _INSISTENT_STYLES
    return rng.choice(pool)


# =====================================================================
# Cancel reasons
# =====================================================================

_CANCEL_REASONS = [
    "no longer needed",
    "ordered by mistake",
    "found a better price elsewhere",
    "changed my mind",
    "duplicate order",
]

_AIRLINE_CANCEL_REASONS = [
    "I have a medical emergency.",
    "My plans changed.",
    "I need to cancel due to a family emergency.",
    "I can no longer make the trip.",
]

# =====================================================================
# Helper: get user identity info for instructions
# =====================================================================

def _user_identity_airline(user_data: Dict) -> str:
    """Build known_info string for airline user."""
    name = user_data["user"]["name"]
    uid = user_data["user_id"]
    return f"Your name is {name['first_name']} {name['last_name']}. Your user id is {uid}."


def _user_identity_retail(user_data: Dict) -> str:
    """Build known_info string for retail user."""
    name = user_data["user"]["name"]
    zipcode = user_data["user"]["address"]["zip"]
    return f"Your name is {name['first_name']} {name['last_name']}. Your zip code is {zipcode}."


# =====================================================================
# TYPE 1: Explicit action scenarios
# =====================================================================

def _gen_retail_cancel_pending(rng: random.Random) -> Optional[GeneratedScenario]:
    """Cancel a pending order (retail)."""
    data = sample_retail_user(rng, {"status": "pending"})
    if data is None:
        return None

    order_id = data["order_id"]
    reason = rng.choice(_CANCEL_REASONS)
    identity = _user_identity_retail(data)
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    initial_msg = (
        f"Hi, I'd like to cancel my order {order_id}. "
        f"The reason is: {reason}."
    )

    prompt = build_user_system_prompt(
        customer_context=f"{identity} You have a pending order {order_id} that you want to cancel.",
        goal=f"Cancel order {order_id} because: {reason}.",
        approach="cooperative",
        required_communication=f"Confirm when order {order_id} is cancelled.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("cancel_pending_order", {
                "order_id": order_id,
                "reason": reason,
            }),
        ],
        communicate_info=["cancel"],
        description=f"Cancel pending order {order_id}",
        key_facts={"order_id": order_id, "user_id": data["user_id"]},
        db=db,
    )


def _gen_retail_return_delivered(rng: random.Random) -> Optional[GeneratedScenario]:
    """Return items from a delivered order (retail)."""
    data = sample_retail_user(rng, {"status": "delivered", "min_items": 1})
    if data is None:
        return None

    order = data["order"]
    order_id = data["order_id"]
    user = data["user"]
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    # Pick one item to return
    items = order.get("items", [])
    if not items:
        return None
    item = rng.choice(items)
    item_id = item["item_id"]
    item_name = item["name"]

    # Find original payment method (first payment in order)
    payment_history = order.get("payment_history", [])
    if payment_history:
        payment_method_id = payment_history[0].get("payment_method_id", "")
    else:
        # Fallback: first user payment method
        pm = user.get("payment_methods", {})
        payment_method_id = list(pm.keys())[0] if pm else ""

    identity = _user_identity_retail(data)

    initial_msg = (
        f"Hi, I'd like to return the {item_name} from my order {order_id}. "
        f"Please refund to my original payment method."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have a delivered order {order_id} and want to return "
            f"the {item_name} (item ID: {item_id})."
        ),
        goal=f"Return the {item_name} from order {order_id} and get a refund to original payment.",
        approach="cooperative",
        required_communication=f"Confirm when the return for {item_name} is processed.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("return_delivered_order_items", {
                "order_id": order_id,
                "item_ids": [item_id],
                "payment_method_id": payment_method_id,
            }),
        ],
        communicate_info=["return"],
        description=f"Return {item_name} from order {order_id}",
        key_facts={
            "order_id": order_id,
            "item_id": item_id,
            "item_name": item_name,
            "payment_method_id": payment_method_id,
            "user_id": data["user_id"],
        },
        db=db,
    )


def _gen_retail_modify_address(rng: random.Random) -> Optional[GeneratedScenario]:
    """Modify shipping address of a pending order (retail)."""
    data = sample_retail_user(rng, {"status": "pending"})
    if data is None:
        return None

    order_id = data["order_id"]
    identity = _user_identity_retail(data)
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    # Generate a new address
    streets = [
        "123 Oak Street", "456 Maple Avenue", "789 Pine Road",
        "321 Elm Drive", "654 Cedar Lane", "987 Birch Court",
    ]
    cities_states = [
        ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
        ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
        ("Phoenix", "AZ", "85001"), ("Seattle", "WA", "98101"),
    ]
    street = rng.choice(streets)
    city, state, zipcode = rng.choice(cities_states)

    initial_msg = (
        f"Hi, I need to update the shipping address for my order {order_id}. "
        f"The new address is {street}, {city}, {state} {zipcode}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have a pending order {order_id} and need to "
            f"change the shipping address."
        ),
        goal=(
            f"Change the shipping address for order {order_id} to: "
            f"{street}, {city}, {state} {zipcode}, USA."
        ),
        approach="cooperative",
        required_communication=f"Confirm when the address is updated for order {order_id}.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("modify_pending_order_address", {
                "order_id": order_id,
                "address1": street,
                "address2": "",
                "city": city,
                "state": state,
                "country": "USA",
                "zip": zipcode,
            }),
        ],
        communicate_info=["address", "updated"],
        description=f"Modify address for order {order_id}",
        key_facts={
            "order_id": order_id,
            "new_address": f"{street}, {city}, {state} {zipcode}",
            "user_id": data["user_id"],
        },
        db=db,
    )


def _gen_retail_modify_payment(rng: random.Random) -> Optional[GeneratedScenario]:
    """Modify payment method of a pending order (retail)."""
    data = sample_retail_user(rng, {
        "status": "pending",
        "has_multiple_payment_types": True,
    })
    if data is None:
        return None

    order_id = data["order_id"]
    user = data["user"]
    identity = _user_identity_retail(data)
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    # Pick a different payment method than current
    order = data["order"]
    current_payment_ids = {
        p.get("payment_method_id") for p in order.get("payment_history", [])
    }
    pm = user.get("payment_methods", {})
    other_methods = [pid for pid in pm if pid not in current_payment_ids]
    if not other_methods:
        other_methods = list(pm.keys())
    if not other_methods:
        return None

    new_pm_id = rng.choice(other_methods)
    new_pm_source = pm[new_pm_id].get("source", "unknown")

    initial_msg = (
        f"Hi, I'd like to change the payment method for my order {order_id} "
        f"to my {new_pm_source.replace('_', ' ')}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have a pending order {order_id} and want to "
            f"change the payment method to your {new_pm_source.replace('_', ' ')} "
            f"(ID: {new_pm_id})."
        ),
        goal=f"Change payment for order {order_id} to payment method {new_pm_id}.",
        approach="cooperative",
        required_communication=f"Confirm when payment is updated for order {order_id}.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("modify_pending_order_payment", {
                "order_id": order_id,
                "payment_method_id": new_pm_id,
            }),
        ],
        communicate_info=["payment", "updated"],
        description=f"Modify payment for order {order_id}",
        key_facts={
            "order_id": order_id,
            "new_payment_id": new_pm_id,
            "user_id": data["user_id"],
        },
        db=db,
    )


def _gen_airline_cancel_eligible(rng: random.Random) -> Optional[GeneratedScenario]:
    """Cancel an eligible airline reservation.

    Eligible: has insurance, or within 24h, or business class.
    """
    # Try insurance first
    for criteria in [
        {"insurance": "yes"},
        {"is_recent": True},
        {"cabin": "business"},
    ]:
        data = sample_airline_user(rng, criteria)
        if data is not None:
            break
    else:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    identity = _user_identity_airline(data)
    db = build_airline_db(data["user"], data["reservation"], data.get("flights_db", {}))
    reason = rng.choice(_AIRLINE_CANCEL_REASONS)

    eligibility = []
    if res.get("insurance") == "yes":
        eligibility.append("You have travel insurance.")
    if res.get("cabin") == "business":
        eligibility.append("Your reservation is business class.")

    eligibility_text = " ".join(eligibility) if eligibility else ""

    initial_msg = (
        f"Hi, I need to cancel my reservation {res_id}. {reason}"
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You want to cancel reservation {res_id}. "
            f"{eligibility_text}"
        ),
        goal=f"Cancel reservation {res_id}.",
        approach="cooperative",
        required_communication=f"Confirm when reservation {res_id} is cancelled.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("cancel_reservation", {
                "reservation_id": res_id,
            }),
        ],
        communicate_info=["cancel"],
        description=f"Cancel eligible reservation {res_id}",
        key_facts={
            "reservation_id": res_id,
            "user_id": data["user_id"],
            "insurance": res.get("insurance"),
            "cabin": res.get("cabin"),
        },
        db=db,
    )


def _gen_airline_update_baggages(rng: random.Random) -> Optional[GeneratedScenario]:
    """Add baggage to an airline reservation."""
    data = sample_airline_user(rng, {})
    if data is None:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    user = data["user"]
    identity = _user_identity_airline(data)
    db = build_airline_db(data["user"], data["reservation"], data.get("flights_db", {}))

    # Current baggages
    current_baggages = res.get("baggages", {})
    current_total = current_baggages.get("total_baggages", 0) if current_baggages else 0
    current_nonfree = current_baggages.get("nonfree_baggages", 0) if current_baggages else 0

    # Add 1-2 bags
    bags_to_add = rng.randint(1, 2)
    new_total = current_total + bags_to_add
    new_nonfree = current_nonfree + bags_to_add

    # Find a valid payment method (credit_card or paypal)
    pm = user.get("payment_methods", {})
    valid_pms = [pid for pid, v in pm.items() if v.get("source") in ("credit_card", "paypal")]
    if not valid_pms:
        return None
    payment_id = rng.choice(valid_pms)

    initial_msg = (
        f"Hi, I'd like to add {bags_to_add} checked bag{'s' if bags_to_add > 1 else ''} "
        f"to my reservation {res_id}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You want to add {bags_to_add} checked bag(s) to reservation {res_id}."
        ),
        goal=f"Add {bags_to_add} checked bag(s) to reservation {res_id}.",
        approach="cooperative",
        required_communication=f"Confirm when bags are added to reservation {res_id}.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("update_reservation_baggages", {
                "reservation_id": res_id,
                "total_baggages": new_total,
                "nonfree_baggages": new_nonfree,
                "payment_id": payment_id,
            }),
        ],
        communicate_info=["bag"],
        description=f"Add {bags_to_add} bags to reservation {res_id}",
        key_facts={
            "reservation_id": res_id,
            "user_id": data["user_id"],
            "bags_to_add": bags_to_add,
            "payment_id": payment_id,
        },
        db=db,
    )


def _gen_airline_change_flight(rng: random.Random) -> Optional[GeneratedScenario]:
    """Change a flight date on a non-basic-economy reservation.

    The model must:
    1. Look up the reservation and confirm cabin is NOT basic_economy
    2. Search for available flights on the new date (same route)
    3. Call update_reservation_flights with ALL flight segments
    """
    # Must be economy or business (basic_economy cannot change flights)
    cabin = rng.choice(["economy", "business"])
    data = sample_airline_user(rng, {"cabin": cabin, "has_unlimited_payment": True})
    if data is None:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    user = data["user"]
    identity = _user_identity_airline(data)
    flights_db = data.get("flights_db", {})
    db = build_airline_db(data["user"], data["reservation"], flights_db)

    flights = res.get("flights", [])
    if not flights:
        return None

    # Pick a flight segment to change
    seg_idx = rng.randrange(len(flights))
    old_seg = flights[seg_idx]
    old_date = old_seg["date"]
    origin = old_seg["origin"]
    dest = old_seg["destination"]

    # Find an available alternative flight on a nearby date (from synthetic flights_db)
    all_flights = flights_db

    # Collect candidates: same origin->dest, different date, status=available
    candidates = []
    for fn, flight_data in all_flights.items():
        if flight_data["origin"] != origin or flight_data["destination"] != dest:
            continue
        for date, date_info in flight_data.get("dates", {}).items():
            if date == old_date:
                continue
            if not isinstance(date_info, dict):
                continue
            status = date_info.get("status", "")
            if status != "available":
                continue
            seats = date_info.get("available_seats", {})
            n_pax = len(res.get("passengers", [1]))
            if seats.get(cabin, 0) < n_pax:
                continue
            prices = date_info.get("prices", {})
            price = prices.get(cabin, 0)
            candidates.append({
                "flight_number": fn,
                "date": date,
                "price": price,
            })

    if not candidates:
        return None

    new_seg = rng.choice(candidates)

    # Build the full flights array for the expected action (all segments)
    new_flights = []
    for i, seg in enumerate(flights):
        if i == seg_idx:
            new_flights.append({
                "flight_number": new_seg["flight_number"],
                "date": new_seg["date"],
            })
        else:
            new_flights.append({
                "flight_number": seg["flight_number"],
                "date": seg["date"],
            })

    # Find a valid payment method (credit_card or paypal, not certificate)
    pm = user.get("payment_methods", {})
    valid_pms = [pid for pid, v in pm.items()
                 if v.get("source") in ("credit_card", "paypal")]
    if not valid_pms:
        return None
    payment_id = rng.choice(valid_pms)

    # Get payment method description for user message
    pm_info = pm[payment_id]
    pm_source = pm_info.get("source", "credit_card")
    pm_last4 = payment_id.split("_")[-1][-4:]  # last 4 chars of ID
    pm_desc = f"my {pm_source.replace('_', ' ')} ending in {pm_last4}"

    initial_msg = (
        f"Hi, I need to change my flight from {origin} to {dest} "
        f"on {old_date} in reservation {res_id}. "
        f"I'd like to switch to flight {new_seg['flight_number']} on {new_seg['date']}. "
        f"Please charge any difference to {pm_desc}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have reservation {res_id} with a {cabin} class flight "
            f"from {origin} to {dest} on {old_date}. "
            f"You want to change to flight {new_seg['flight_number']} on {new_seg['date']}. "
            f"Use payment method {payment_id} for any price difference."
        ),
        goal=(
            f"Change the {origin} to {dest} flight in reservation {res_id} "
            f"to flight {new_seg['flight_number']} on {new_seg['date']}. "
            f"Payment method: {payment_id}."
        ),
        approach="cooperative",
        required_communication=f"Confirm when the flight change is processed for reservation {res_id}.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("update_reservation_flights", {
                "reservation_id": res_id,
                "cabin": cabin,
                "flights": new_flights,
                "payment_id": payment_id,
            }),
        ],
        communicate_info=["changed", "flight"],
        description=f"Change flight {origin}->{dest} from {old_date} to {new_seg['date']} on reservation {res_id}",
        key_facts={
            "reservation_id": res_id,
            "user_id": data["user_id"],
            "cabin": cabin,
            "old_flight": old_seg["flight_number"],
            "old_date": old_date,
            "new_flight": new_seg["flight_number"],
            "new_date": new_seg["date"],
            "origin": origin,
            "destination": dest,
            "payment_id": payment_id,
        },
        db=db,
    )


def _gen_airline_change_flight_basic_economy(rng: random.Random) -> Optional[GeneratedScenario]:
    """Try to change flights on a basic_economy reservation (forbidden by policy).

    The policy says basic economy flights cannot be modified (only cabin upgrades
    are allowed). The API does NOT enforce this — the agent must refuse.
    """
    data = sample_airline_user(rng, {"cabin": "basic_economy"})
    if data is None:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    identity = _user_identity_airline(data)
    db = build_airline_db(data["user"], data["reservation"], data.get("flights_db", {}))

    flights = res.get("flights", [])
    if not flights:
        return None

    seg = flights[0]
    origin = seg["origin"]
    dest = seg["destination"]
    old_date = seg["date"]

    # Propose a new date (1-3 days later)
    try:
        from datetime import datetime, timedelta
        old_dt = datetime.strptime(old_date, "%Y-%m-%d")
        shift = rng.randint(1, 3)
        new_date = (old_dt + timedelta(days=shift)).strftime("%Y-%m-%d")
    except Exception:
        new_date = "2024-05-20"

    reason = rng.choice([
        "My schedule changed and I need a different date.",
        "Something came up and I need to fly a day later.",
        "I have a conflict on that date, can you move the flight?",
    ])

    initial_msg = (
        f"Hi, I need to change my flight from {origin} to {dest} "
        f"on {old_date} in reservation {res_id} to {new_date}. {reason}"
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have reservation {res_id} with a basic economy flight "
            f"from {origin} to {dest} on {old_date}. "
            f"You want to change this flight to {new_date}."
        ),
        goal=(
            f"Change the flight in reservation {res_id} from {old_date} to {new_date}."
        ),
        approach="cooperative",
        required_communication=(
            f"If the agent says the flight can't be changed because it's basic economy, "
            f"accept it gracefully."
        ),
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="policy_gated",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[],  # Refusal — no write action expected
        communicate_info=["cannot", "change"],
        is_refusal=True,
        description=f"Change basic economy flight {origin}->{dest} on reservation {res_id} (should be refused)",
        key_facts={
            "reservation_id": res_id,
            "user_id": data["user_id"],
            "cabin": "basic_economy",
            "origin": origin,
            "destination": dest,
            "old_date": old_date,
            "requested_date": new_date,
        },
        db=db,
    )


def _gen_airline_update_passengers(rng: random.Random) -> Optional[GeneratedScenario]:
    """Update passenger info on an airline reservation."""
    data = sample_airline_user(rng, {"num_passengers_min": 1})
    if data is None:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    identity = _user_identity_airline(data)
    db = build_airline_db(data["user"], data["reservation"], data.get("flights_db", {}))

    passengers = res.get("passengers", [])
    if not passengers:
        return None

    # Update DOB of first passenger (common correction scenario)
    passenger = copy.deepcopy(passengers[0])
    # Change DOB slightly
    orig_dob = passenger.get("dob", "1990-01-01")
    try:
        year, month, day = orig_dob.split("-")
        new_day = str(min(28, int(day) + rng.randint(1, 5))).zfill(2)
        new_dob = f"{year}-{month}-{new_day}"
    except (ValueError, IndexError):
        new_dob = "1990-06-15"

    updated_passengers = copy.deepcopy(passengers)
    updated_passengers[0]["dob"] = new_dob

    first_name = passenger.get("first_name", "Unknown")
    last_name = passenger.get("last_name", "Unknown")

    initial_msg = (
        f"Hi, I need to correct the date of birth for {first_name} {last_name} "
        f"on reservation {res_id}. The correct DOB is {new_dob}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You need to correct passenger information on reservation {res_id}. "
            f"The DOB for {first_name} {last_name} should be {new_dob}."
        ),
        goal=f"Update the date of birth for {first_name} {last_name} to {new_dob} on reservation {res_id}.",
        approach="cooperative",
        required_communication=f"Confirm when passenger info is updated on reservation {res_id}.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("update_reservation_passengers", {
                "reservation_id": res_id,
                "passengers": [p for p in updated_passengers],
            }),
        ],
        communicate_info=["updated", "passenger"],
        description=f"Update passenger DOB on reservation {res_id}",
        key_facts={
            "reservation_id": res_id,
            "user_id": data["user_id"],
            "passenger_name": f"{first_name} {last_name}",
            "new_dob": new_dob,
        },
        db=db,
    )


# =====================================================================
# TYPE 2: Selection scenarios (single-action with choice)
# =====================================================================

def _gen_retail_exchange_cheapest(rng: random.Random) -> Optional[GeneratedScenario]:
    """Exchange a delivered item for the cheapest variant of the same product."""
    data = sample_retail_user(rng, {"status": "delivered", "min_items": 1})
    if data is None:
        return None

    order = data["order"]
    order_id = data["order_id"]
    user = data["user"]

    items = order.get("items", [])
    if not items:
        return None
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    # Pick an item with a multi-variant product (from synthetic products_db)
    products_db = data.get("products_db", {})
    rng_copy = random.Random(rng.random())
    shuffled_items = list(items)
    rng_copy.shuffle(shuffled_items)

    for item in shuffled_items:
        product_id = item.get("product_id", "")
        product = products_db.get(product_id, {})
        variants = product.get("variants", {})
        if len(variants) < 3:
            continue

        # Find cheapest *available* variant that's different from current
        current_item_id = item["item_id"]
        variant_list = [
            (vid, v) for vid, v in variants.items()
            if vid != current_item_id and v.get("available", True)
        ]
        if not variant_list:
            continue

        variant_list.sort(key=lambda x: x[1].get("price", float("inf")))
        cheapest_vid, cheapest_variant = variant_list[0]
        cheapest_price = cheapest_variant.get("price", 0)

        item_name = item["name"]

        # Find payment method
        payment_history = order.get("payment_history", [])
        if payment_history:
            payment_method_id = payment_history[0].get("payment_method_id", "")
        else:
            pm = user.get("payment_methods", {})
            # Prefer credit card or paypal for exchanges (unlimited source)
            valid_pms = [pid for pid, v in pm.items() if v.get("source") in ("credit_card", "paypal")]
            if valid_pms:
                payment_method_id = valid_pms[0]
            elif pm:
                payment_method_id = list(pm.keys())[0]
            else:
                continue

        identity = _user_identity_retail(data)

        initial_msg = (
            f"Hi, I'd like to exchange the {item_name} in my order {order_id} "
            f"for the cheapest available option of the same product."
        )

        prompt = build_user_system_prompt(
            customer_context=(
                f"{identity} You have a delivered order {order_id} with a {item_name}. "
                f"You want to exchange it for the cheapest available variant."
            ),
            goal=(
                f"Exchange the {item_name} (item {current_item_id}) in order {order_id} "
                f"for the cheapest available variant of the same product."
            ),
            approach="cooperative",
            required_communication=f"Confirm when the exchange is processed.",
        )

        return GeneratedScenario(
            domain="retail",
            scenario_type="selection",
            user_system_prompt=prompt,
            initial_message=initial_msg,
            expected_actions=[
                ExpectedAction("exchange_delivered_order_items", {
                    "order_id": order_id,
                    "item_ids": [current_item_id],
                    "new_item_ids": [cheapest_vid],
                    "payment_method_id": payment_method_id,
                }),
            ],
            communicate_info=["exchange"],
            description=f"Exchange {item_name} for cheapest variant in order {order_id}",
            key_facts={
                "order_id": order_id,
                "item_id": current_item_id,
                "item_name": item_name,
                "product_id": product_id,
                "cheapest_variant_id": cheapest_vid,
                "cheapest_price": cheapest_price,
                "payment_method_id": payment_method_id,
                "num_variants": len(variants),
                "user_id": data["user_id"],
            },
            db=db,
        )

    return None


def _gen_retail_exchange_specific_attr(rng: random.Random) -> Optional[GeneratedScenario]:
    """Exchange item for a specific variant by attribute (color, size, etc.)."""
    data = sample_retail_user(rng, {"status": "delivered", "min_items": 1})
    if data is None:
        return None

    order = data["order"]
    order_id = data["order_id"]
    user = data["user"]

    items = order.get("items", [])
    if not items:
        return None
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    products_db = data.get("products_db", {})
    rng_copy = random.Random(rng.random())
    shuffled_items = list(items)
    rng_copy.shuffle(shuffled_items)

    for item in shuffled_items:
        product_id = item.get("product_id", "")
        product = products_db.get(product_id, {})
        variants = product.get("variants", {})
        if len(variants) < 3:
            continue

        current_item_id = item["item_id"]
        current_options = item.get("options", {})

        # Find an available variant with different options
        for vid, v in variants.items():
            if vid == current_item_id:
                continue
            if not v.get("available", True):
                continue
            v_options = v.get("options", {})
            # Find a single attribute difference
            diffs = {}
            for k in v_options:
                if k in current_options and v_options[k] != current_options[k]:
                    diffs[k] = v_options[k]
            if len(diffs) >= 1:
                # Pick one differing attribute
                attr_name = list(diffs.keys())[0]
                attr_value = diffs[attr_name]

                item_name = item["name"]

                # Payment method
                payment_history = order.get("payment_history", [])
                if payment_history:
                    payment_method_id = payment_history[0].get("payment_method_id", "")
                else:
                    pm = user.get("payment_methods", {})
                    valid_pms = [pid for pid, pv in pm.items() if pv.get("source") in ("credit_card", "paypal")]
                    payment_method_id = valid_pms[0] if valid_pms else (list(pm.keys())[0] if pm else "")

                if not payment_method_id:
                    continue

                identity = _user_identity_retail(data)

                initial_msg = (
                    f"Hi, I'd like to exchange the {item_name} in my order {order_id} "
                    f"for one with {attr_name}: {attr_value}."
                )

                prompt = build_user_system_prompt(
                    customer_context=(
                        f"{identity} You have a delivered order {order_id} with a {item_name}. "
                        f"You want to exchange it for a variant with {attr_name} = {attr_value}."
                    ),
                    goal=(
                        f"Exchange the {item_name} in order {order_id} for a variant "
                        f"with {attr_name}: {attr_value}."
                    ),
                    approach="cooperative",
                    required_communication=f"Confirm when the exchange is processed.",
                )

                return GeneratedScenario(
                    domain="retail",
                    scenario_type="selection",
                    user_system_prompt=prompt,
                    initial_message=initial_msg,
                    expected_actions=[
                        ExpectedAction("exchange_delivered_order_items", {
                            "order_id": order_id,
                            "item_ids": [current_item_id],
                            "new_item_ids": [vid],
                            "payment_method_id": payment_method_id,
                        }),
                    ],
                    communicate_info=["exchange"],
                    description=f"Exchange {item_name} for {attr_name}={attr_value}",
                    key_facts={
                        "order_id": order_id,
                        "item_id": current_item_id,
                        "item_name": item_name,
                        "target_variant_id": vid,
                        "target_attribute": attr_name,
                        "target_value": attr_value,
                        "payment_method_id": payment_method_id,
                        "user_id": data["user_id"],
                    },
                    db=db,
                )

    return None


def _gen_retail_modify_items(rng: random.Random) -> Optional[GeneratedScenario]:
    """Modify items in a pending order (swap for a different variant)."""
    data = sample_retail_user(rng, {"status": "pending", "min_items": 1})
    if data is None:
        return None

    order = data["order"]
    order_id = data["order_id"]
    user = data["user"]

    items = order.get("items", [])
    if not items:
        return None
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    products_db = data.get("products_db", {})
    rng_copy = random.Random(rng.random())
    shuffled_items = list(items)
    rng_copy.shuffle(shuffled_items)

    for item in shuffled_items:
        product_id = item.get("product_id", "")
        product = products_db.get(product_id, {})
        variants = product.get("variants", {})
        if len(variants) < 2:
            continue

        current_item_id = item["item_id"]
        # Pick a different available variant
        other_variants = [
            vid for vid in variants
            if vid != current_item_id and variants[vid].get("available", True)
        ]
        if not other_variants:
            continue
        new_variant_id = rng_copy.choice(other_variants)
        new_variant = variants[new_variant_id]
        new_options = new_variant.get("options", {})

        item_name = item["name"]

        # Payment method
        pm = user.get("payment_methods", {})
        valid_pms = [pid for pid, pv in pm.items() if pv.get("source") in ("credit_card", "paypal")]
        payment_method_id = valid_pms[0] if valid_pms else (list(pm.keys())[0] if pm else "")
        if not payment_method_id:
            continue

        # Build option description
        option_desc = ", ".join(f"{k}: {v}" for k, v in new_options.items())

        identity = _user_identity_retail(data)

        initial_msg = (
            f"Hi, I'd like to change the {item_name} in my pending order {order_id} "
            f"to the variant with {option_desc}."
        )

        prompt = build_user_system_prompt(
            customer_context=(
                f"{identity} You have a pending order {order_id} and want to change "
                f"the {item_name} to a different variant ({option_desc})."
            ),
            goal=(
                f"Modify the {item_name} in pending order {order_id} to the variant "
                f"with {option_desc}."
            ),
            approach="cooperative",
            required_communication=f"Confirm when the item is changed in order {order_id}.",
        )

        return GeneratedScenario(
            domain="retail",
            scenario_type="selection",
            user_system_prompt=prompt,
            initial_message=initial_msg,
            expected_actions=[
                ExpectedAction("modify_pending_order_items", {
                    "order_id": order_id,
                    "item_ids": [current_item_id],
                    "new_item_ids": [new_variant_id],
                    "payment_method_id": payment_method_id,
                }),
            ],
            communicate_info=["modified", item_name.lower()],
            description=f"Modify {item_name} in pending order {order_id}",
            key_facts={
                "order_id": order_id,
                "item_id": current_item_id,
                "new_variant_id": new_variant_id,
                "item_name": item_name,
                "payment_method_id": payment_method_id,
                "user_id": data["user_id"],
            },
            db=db,
        )

    return None


# =====================================================================
# TYPE 3: Policy-gated scenarios
# =====================================================================

def _gen_retail_cancel_nonpending(rng: random.Random) -> Optional[GeneratedScenario]:
    """Try to cancel a non-pending order (forbidden by policy)."""
    data = sample_retail_user(rng, {"status": "delivered"})
    if data is None:
        return None

    order_id = data["order_id"]
    identity = _user_identity_retail(data)
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    initial_msg = (
        f"Hi, I want to cancel my order {order_id}. I changed my mind about it."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have order {order_id} which has already been delivered. "
            f"You want to cancel it."
        ),
        goal=f"Cancel order {order_id}.",
        approach="cooperative",
        required_communication=(
            f"If the agent says the order can't be cancelled, accept it gracefully."
        ),
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="policy_gated",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[],  # No actions expected (refusal)
        communicate_info=["cannot", "cancel"],
        is_refusal=True,
        description=f"Cancel delivered order {order_id} (should be refused)",
        key_facts={"order_id": order_id, "user_id": data["user_id"]},
        db=db,
    )


def _gen_airline_cancel_ineligible(rng: random.Random) -> Optional[GeneratedScenario]:
    """Try to cancel an ineligible reservation (forbidden by policy).

    Ineligible: economy or basic_economy, no insurance, not recent.
    """
    data = sample_airline_user(rng, {
        "cabin": "economy",
        "insurance": "no",
        "is_recent": False,
    })
    if data is None:
        return None

    res_id = data["reservation_id"]
    identity = _user_identity_airline(data)
    db = build_airline_db(data["user"], data["reservation"], data.get("flights_db", {}))

    initial_msg = (
        f"Hi, I need to cancel my reservation {res_id}. My plans have changed."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You want to cancel reservation {res_id}. "
            f"You do not have travel insurance."
        ),
        goal=f"Cancel reservation {res_id}.",
        approach="cooperative",
        required_communication=(
            f"If the agent says the reservation can't be cancelled, accept it gracefully."
        ),
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="policy_gated",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[],  # Refusal
        communicate_info=["cannot", "cancel"],
        is_refusal=True,
        description=f"Cancel ineligible reservation {res_id} (should be refused)",
        key_facts={
            "reservation_id": res_id,
            "user_id": data["user_id"],
        },
        db=db,
    )


def _gen_retail_info_query(rng: random.Random) -> Optional[GeneratedScenario]:
    """Ask about order status (no write action needed)."""
    # Pick any status
    status = rng.choice(["pending", "delivered", "processed"])
    data = sample_retail_user(rng, {"status": status})
    if data is None:
        return None

    order_id = data["order_id"]
    identity = _user_identity_retail(data)
    db = build_retail_db(data["user"], data["order"], data.get("products_db", {}))

    initial_msg = (
        f"Hi, I'd like to check on the status of my order {order_id}."
    )

    prompt = build_user_system_prompt(
        customer_context=f"{identity} You want to know the status of order {order_id}.",
        goal=f"Get the status of order {order_id}.",
        approach="cooperative",
        required_communication=f"Acknowledge the order status.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="policy_gated",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[],  # No write actions
        communicate_info=[status],
        is_refusal=False,  # Not refusal, just info query
        description=f"Query status of order {order_id}",
        key_facts={"order_id": order_id, "user_id": data["user_id"], "status": status},
        db=db,
    )


def _gen_airline_info_query(rng: random.Random) -> Optional[GeneratedScenario]:
    """Ask about reservation details (no write action needed)."""
    data = sample_airline_user(rng, {})
    if data is None:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    identity = _user_identity_airline(data)
    db = build_airline_db(data["user"], data["reservation"], data.get("flights_db", {}))

    initial_msg = (
        f"Hi, can you tell me the details of my reservation {res_id}?"
    )

    prompt = build_user_system_prompt(
        customer_context=f"{identity} You want to check your reservation {res_id} details.",
        goal=f"Get details of reservation {res_id}.",
        approach="cooperative",
        required_communication=f"Acknowledge the reservation details.",
    )

    origin = res.get("origin", "")
    destination = res.get("destination", "")

    return GeneratedScenario(
        domain="airline",
        scenario_type="policy_gated",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[],
        communicate_info=[origin, destination] if origin and destination else [],
        is_refusal=False,
        description=f"Query details of reservation {res_id}",
        key_facts={"reservation_id": res_id, "user_id": data["user_id"]},
        db=db,
    )


# =====================================================================
# Scenario dispatch
# =====================================================================

# Type 1: Explicit action generators (domain-mixed)
_TYPE1_GENERATORS = [
    _gen_retail_cancel_pending,
    _gen_retail_return_delivered,
    _gen_retail_modify_address,
    _gen_retail_modify_payment,
    _gen_airline_cancel_eligible,
    _gen_airline_update_baggages,
    _gen_airline_update_passengers,
    _gen_airline_change_flight,
]

# Type 2: Selection generators
_TYPE2_GENERATORS = [
    _gen_retail_exchange_cheapest,
    _gen_retail_exchange_specific_attr,
    _gen_retail_modify_items,
]

# Type 3: Policy-gated generators (50% forbidden, 50% allowed/info)
_TYPE3_FORBIDDEN = [
    _gen_retail_cancel_nonpending,
    _gen_airline_cancel_ineligible,
    _gen_airline_change_flight_basic_economy,
]

_TYPE3_ALLOWED = [
    _gen_retail_info_query,
    _gen_airline_info_query,
    _gen_airline_cancel_eligible,  # Reuse — eligible cancellation is also policy-gated
    _gen_retail_cancel_pending,  # Reuse — pending cancellation is allowed
]


def generate_scenario(seed: int, domain: Optional[str] = None) -> GeneratedScenario:
    """Generate a simplified tau bench task from a seed.

    Args:
        seed: Random seed for deterministic generation.
        domain: Optional domain filter ("airline" or "retail").
               If None, domain is chosen based on scenario type.

    Returns:
        A GeneratedScenario ready for use in the environment.
    """
    rng = random.Random(seed)

    # Select scenario type with plan-specified weights
    scenario_type = rng.choices(
        ["explicit", "selection", "policy_gated"],
        weights=[50, 30, 20],
    )[0]

    def _get_generators(stype, sub_rng):
        """Get generator list for a scenario type."""
        if stype == "explicit":
            return list(_TYPE1_GENERATORS)
        elif stype == "selection":
            return list(_TYPE2_GENERATORS)
        elif stype == "policy_gated":
            if sub_rng.random() < 0.5:
                return list(_TYPE3_FORBIDDEN)
            else:
                return list(_TYPE3_ALLOWED)
        return []

    # Try generators with retries (some may fail to find matching entities)
    max_retries = 10

    # Try preferred scenario type first, then fall back to others
    type_order = [scenario_type] + [
        t for t in ["explicit", "selection", "policy_gated"]
        if t != scenario_type
    ]

    for attempt in range(max_retries):
        retry_rng = random.Random(rng.randint(0, 2**31 - 1))

        for stype in type_order:
            generators = _get_generators(stype, retry_rng)
            retry_rng.shuffle(generators)
            for gen in generators:
                result = gen(retry_rng)
                if result is not None:
                    if domain and result.domain != domain:
                        continue
                    return result

    # Ultimate fallback
    fallback_rng = random.Random(seed + 99999)
    if domain == "airline":
        result = _gen_airline_cancel_eligible(fallback_rng)
        if result is not None:
            return result
        return _gen_airline_info_query(fallback_rng) or GeneratedScenario(
            domain="airline",
            scenario_type="policy_gated",
            user_system_prompt=build_user_system_prompt(
                customer_context="You are an airline customer.",
                goal="Ask about your flight.",
                approach="cooperative",
            ),
            initial_message="Hi, I have a question about my flight.",
            expected_actions=[],
            communicate_info=[],
            is_refusal=False,
            description="Fallback: airline inquiry",
        db={"users": {}, "reservations": {}, "flights": {}},
        )

    result = _gen_retail_cancel_pending(fallback_rng)
    if result is not None:
        return result

    return GeneratedScenario(
        domain="retail",
        scenario_type="policy_gated",
        user_system_prompt=build_user_system_prompt(
            customer_context="You are a customer.",
            goal="Ask about your account.",
            approach="cooperative",
        ),
        initial_message="Hi, I have a question about my account.",
        expected_actions=[],
        communicate_info=[],
        is_refusal=False,
        description="Fallback: general inquiry",
        db={"users": {}, "orders": {}, "products": {}},
    )
