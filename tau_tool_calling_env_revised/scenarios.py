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
    sample_retail_multi_orders,
    sample_airline_multi_reservations,
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




_PROGRESSIVE_DISCLOSURE_INSTRUCTIONS = [
    "Don't reveal your order ID right away. Wait for the agent to ask.",
    "Initially just describe what you want without providing specific identifiers.",
    "Start by describing your issue vaguely. Only provide order numbers or reservation IDs when the agent asks for them.",
    "Don't volunteer your account details upfront. Let the agent guide the conversation.",
]


def _maybe_add_progressive_disclosure(customer_context: str, rng: random.Random) -> str:
    """With 30% probability, append progressive disclosure instructions to customer_context.

    This forces the user simulator to withhold identifiers initially, testing
    the agent's ability to ask clarifying questions.
    """
    if rng.random() < 0.3:
        instruction = rng.choice(_PROGRESSIVE_DISCLOSURE_INSTRUCTIONS)
        return customer_context + " " + instruction
    return customer_context


# =====================================================================
# Cancel reasons
# =====================================================================

_CANCEL_REASONS = [
    "no longer needed",
    "ordered by mistake",
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
        communicate_info=["address"],
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
        communicate_info=["payment"],
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

    # Current baggages (top-level fields on reservation)
    current_total = res.get("total_baggages", 0)
    current_nonfree = res.get("nonfree_baggages", 0)

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
        communicate_info=["flight"],
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
        approach="adversarial",
        required_communication=(
            f"If the agent says the flight can't be changed because it's basic economy, "
            f"push back at least once before accepting."
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
        communicate_info=["passenger"],
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

        variant_list.sort(key=lambda x: (x[1].get("price", float("inf")), x[0]))
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
            communicate_info=[item_name.lower()],
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
        approach="adversarial",
        required_communication=(
            f"If the agent says the order can't be cancelled, push back at least once before accepting."
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
        approach="adversarial",
        required_communication=(
            f"If the agent says the reservation can't be cancelled, push back at least once before accepting."
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
# TYPE 4: Multi-step scenarios
# =====================================================================

def _gen_retail_multi_return(rng: random.Random) -> Optional[GeneratedScenario]:
    """Return items from 2 different delivered orders."""
    data = sample_retail_multi_orders(rng, {}, order_specs=[
        {"status": "delivered", "min_items": 1},
        {"status": "delivered", "min_items": 1},
    ])
    if data is None:
        return None

    user = data["user"]
    orders = data["orders"]
    identity = _user_identity_retail(data)
    db = build_retail_db(user, orders, data.get("products_db", {}))

    actions = []
    item_descs = []
    for order in orders:
        order_id = order["order_id"]
        items = order.get("items", [])
        if not items:
            continue
        item = rng.choice(items)
        item_id = item["item_id"]
        item_name = item["name"]

        payment_history = order.get("payment_history", [])
        payment_method_id = payment_history[0].get("payment_method_id", "") if payment_history else ""
        if not payment_method_id:
            pm = user.get("payment_methods", {})
            payment_method_id = list(pm.keys())[0] if pm else ""

        actions.append(ExpectedAction("return_delivered_order_items", {
            "order_id": order_id,
            "item_ids": [item_id],
            "payment_method_id": payment_method_id,
        }))
        item_descs.append(f"{item_name} from order {order_id}")

    if len(actions) < 2:
        return None

    items_text = " and ".join(item_descs)
    initial_msg = f"Hi, I'd like to return the {items_text}. Please refund to the original payment methods."

    prompt = build_user_system_prompt(
        customer_context=f"{identity} You have multiple delivered orders and want to return items from each.",
        goal=f"Return the {items_text}.",
        approach="cooperative",
        required_communication="Confirm when all returns are processed.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=actions,
        communicate_info=["return"],
        description=f"Return items from {len(actions)} orders",
        key_facts={"user_id": data["user_id"], "num_orders": len(actions)},
        db=db,
    )


def _gen_retail_cancel_and_return(rng: random.Random) -> Optional[GeneratedScenario]:
    """Cancel a pending order AND return items from a delivered order."""
    data = sample_retail_multi_orders(rng, {}, order_specs=[
        {"status": "pending", "min_items": 1},
        {"status": "delivered", "min_items": 1},
    ])
    if data is None:
        return None

    user = data["user"]
    orders = data["orders"]
    identity = _user_identity_retail(data)
    db = build_retail_db(user, orders, data.get("products_db", {}))

    pending_order = next((o for o in orders if o["status"] == "pending"), None)
    delivered_order = next((o for o in orders if o["status"] == "delivered"), None)
    if not pending_order or not delivered_order:
        return None

    cancel_reason = rng.choice(_CANCEL_REASONS)
    return_item = rng.choice(delivered_order.get("items", [{}]))
    return_item_id = return_item.get("item_id", "")
    return_item_name = return_item.get("name", "item")

    del_payment = delivered_order.get("payment_history", [{}])[0].get("payment_method_id", "")
    if not del_payment:
        pm = user.get("payment_methods", {})
        del_payment = list(pm.keys())[0] if pm else ""

    initial_msg = (
        f"Hi, I need to cancel my pending order {pending_order['order_id']} "
        f"({cancel_reason}), and also return the {return_item_name} from my "
        f"delivered order {delivered_order['order_id']}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have a pending order {pending_order['order_id']} you want to cancel, "
            f"and a delivered order {delivered_order['order_id']} from which you want to return "
            f"the {return_item_name}."
        ),
        goal=(
            f"Cancel order {pending_order['order_id']} ({cancel_reason}) and "
            f"return the {return_item_name} from order {delivered_order['order_id']}."
        ),
        approach="cooperative",
        required_communication="Confirm when both the cancellation and the return are processed.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("cancel_pending_order", {
                "order_id": pending_order["order_id"],
                "reason": cancel_reason,
            }),
            ExpectedAction("return_delivered_order_items", {
                "order_id": delivered_order["order_id"],
                "item_ids": [return_item_id],
                "payment_method_id": del_payment,
            }),
        ],
        communicate_info=["cancel", "return"],
        description=f"Cancel order {pending_order['order_id']} + return from {delivered_order['order_id']}",
        key_facts={"user_id": data["user_id"]},
        db=db,
    )


def _gen_retail_address_propagation(rng: random.Random) -> Optional[GeneratedScenario]:
    """Update user address AND pending order shipping addresses."""
    data = sample_retail_multi_orders(rng, {}, order_specs=[
        {"status": "pending", "min_items": 1},
        {"status": "pending", "min_items": 1},
    ])
    if data is None:
        return None

    user = data["user"]
    orders = data["orders"]
    identity = _user_identity_retail(data)
    db = build_retail_db(user, orders, data.get("products_db", {}))

    pending_orders = [o for o in orders if o["status"] == "pending"]
    if not pending_orders:
        return None

    streets = ["123 Oak Street", "456 Maple Avenue", "789 Pine Road", "321 Elm Drive"]
    cities_states = [
        ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
        ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
    ]
    street = rng.choice(streets)
    city, state, zipcode = rng.choice(cities_states)
    new_addr = {
        "address1": street, "address2": "", "city": city,
        "state": state, "country": "USA", "zip": zipcode,
    }

    order_ids_str = " and ".join(o["order_id"] for o in pending_orders)
    initial_msg = (
        f"Hi, I've moved to {street}, {city}, {state} {zipcode}. "
        f"Please update my profile address and also the shipping address "
        f"on my pending orders {order_ids_str}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You've moved and need to update your address everywhere. "
            f"Your new address is {street}, {city}, {state} {zipcode}, USA."
        ),
        goal=(
            f"Update your profile address AND the shipping address on all pending orders "
            f"({order_ids_str}) to: {street}, {city}, {state} {zipcode}, USA."
        ),
        approach="cooperative",
        required_communication="Confirm when all addresses are updated.",
    )

    actions = [
        ExpectedAction("modify_user_address", {
            "user_id": data["user_id"],
            **new_addr,
        }),
    ]
    for o in pending_orders:
        actions.append(ExpectedAction("modify_pending_order_address", {
            "order_id": o["order_id"],
            **new_addr,
        }))

    return GeneratedScenario(
        domain="retail",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=actions,
        communicate_info=["address"],
        description=f"Update user + {len(pending_orders)} order addresses",
        key_facts={"user_id": data["user_id"]},
        db=db,
    )


def _gen_retail_exchange_and_return(rng: random.Random) -> Optional[GeneratedScenario]:
    """Exchange from one order + return from another."""
    data = sample_retail_multi_orders(rng, {}, order_specs=[
        {"status": "delivered", "min_items": 2},
        {"status": "delivered", "min_items": 1},
    ])
    if data is None:
        return None

    user = data["user"]
    orders = data["orders"]
    products_db = data.get("products_db", {})
    identity = _user_identity_retail(data)
    db = build_retail_db(user, orders, products_db)

    exchange_order = orders[0]
    return_order = orders[1]

    # Pick exchange item
    ex_item = rng.choice(exchange_order.get("items", [{}]))
    ex_item_id = ex_item.get("item_id", "")
    ex_item_name = ex_item.get("name", "item")
    ex_product_id = ex_item.get("product_id", "")

    # Find a different available variant for exchange
    product = products_db.get(ex_product_id, {})
    variants = product.get("variants", {})
    other_variants = [
        vid for vid, v in variants.items()
        if vid != ex_item_id and v.get("available", True)
    ]
    if not other_variants:
        return None
    new_variant_id = rng.choice(other_variants)

    # Pick return item
    ret_item = rng.choice(return_order.get("items", [{}]))
    ret_item_id = ret_item.get("item_id", "")
    ret_item_name = ret_item.get("name", "item")

    ex_payment = exchange_order.get("payment_history", [{}])[0].get("payment_method_id", "")
    ret_payment = return_order.get("payment_history", [{}])[0].get("payment_method_id", "")
    if not ex_payment or not ret_payment:
        pm = user.get("payment_methods", {})
        fallback = list(pm.keys())[0] if pm else ""
        ex_payment = ex_payment or fallback
        ret_payment = ret_payment or fallback

    new_variant = variants[new_variant_id]
    new_opts = ", ".join(f"{k}: {v}" for k, v in new_variant.get("options", {}).items())

    initial_msg = (
        f"Hi, I'd like to exchange the {ex_item_name} in order {exchange_order['order_id']} "
        f"for the variant with {new_opts}, and also return the {ret_item_name} from "
        f"order {return_order['order_id']}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You want to exchange the {ex_item_name} in order "
            f"{exchange_order['order_id']} and return the {ret_item_name} from "
            f"order {return_order['order_id']}."
        ),
        goal=(
            f"Exchange the {ex_item_name} in order {exchange_order['order_id']} for "
            f"the variant with {new_opts}. Also return the {ret_item_name} from "
            f"order {return_order['order_id']}."
        ),
        approach="cooperative",
        required_communication="Confirm when both the exchange and return are processed.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("exchange_delivered_order_items", {
                "order_id": exchange_order["order_id"],
                "item_ids": [ex_item_id],
                "new_item_ids": [new_variant_id],
                "payment_method_id": ex_payment,
            }),
            ExpectedAction("return_delivered_order_items", {
                "order_id": return_order["order_id"],
                "item_ids": [ret_item_id],
                "payment_method_id": ret_payment,
            }),
        ],
        communicate_info=["exchange", "return"],
        description=f"Exchange from {exchange_order['order_id']} + return from {return_order['order_id']}",
        key_facts={"user_id": data["user_id"]},
        db=db,
    )


def _gen_retail_modify_items_and_address(rng: random.Random) -> Optional[GeneratedScenario]:
    """Modify items AND address on a pending order."""
    data = sample_retail_user(rng, {"status": "pending", "min_items": 1})
    if data is None:
        return None

    order = data["order"]
    order_id = data["order_id"]
    user = data["user"]
    products_db = data.get("products_db", {})
    identity = _user_identity_retail(data)
    db = build_retail_db(user, order, products_db)

    items = order.get("items", [])
    if not items:
        return None

    # Pick an item to modify
    item = rng.choice(items)
    product_id = item.get("product_id", "")
    product = products_db.get(product_id, {})
    variants = product.get("variants", {})
    other_variants = [
        vid for vid, v in variants.items()
        if vid != item["item_id"] and v.get("available", True)
    ]
    if not other_variants:
        return None
    new_variant_id = rng.choice(other_variants)
    new_variant = variants[new_variant_id]
    new_opts = ", ".join(f"{k}: {v}" for k, v in new_variant.get("options", {}).items())

    # Payment method
    pm = user.get("payment_methods", {})
    valid_pms = [pid for pid, pv in pm.items() if pv.get("source") in ("credit_card", "paypal")]
    payment_method_id = valid_pms[0] if valid_pms else (list(pm.keys())[0] if pm else "")
    if not payment_method_id:
        return None

    # New address
    streets = ["123 Oak Street", "456 Maple Avenue", "789 Pine Road"]
    cities_states = [("New York", "NY", "10001"), ("Chicago", "IL", "60601"), ("Houston", "TX", "77001")]
    street = rng.choice(streets)
    city, state, zipcode = rng.choice(cities_states)
    new_addr = {"address1": street, "address2": "", "city": city, "state": state, "country": "USA", "zip": zipcode}

    initial_msg = (
        f"Hi, I need to make two changes to my pending order {order_id}: "
        f"change the {item['name']} to the variant with {new_opts}, "
        f"and update the shipping address to {street}, {city}, {state} {zipcode}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You need to modify your pending order {order_id}: "
            f"change the {item['name']} and update the shipping address."
        ),
        goal=(
            f"1. Change the {item['name']} in order {order_id} to the variant with {new_opts}. "
            f"2. Update the shipping address to {street}, {city}, {state} {zipcode}, USA."
        ),
        approach="cooperative",
        required_communication=f"Confirm when both changes to order {order_id} are made.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("modify_pending_order_items", {
                "order_id": order_id,
                "item_ids": [item["item_id"]],
                "new_item_ids": [new_variant_id],
                "payment_method_id": payment_method_id,
            }),
            ExpectedAction("modify_pending_order_address", {
                "order_id": order_id,
                **new_addr,
            }),
        ],
        communicate_info=[item["name"].lower(), "address"],
        description=f"Modify items + address on order {order_id}",
        key_facts={"order_id": order_id, "user_id": data["user_id"]},
        db=db,
    )


def _gen_airline_search_and_change(rng: random.Random) -> Optional[GeneratedScenario]:
    """Search for flights then change reservation — model must search first."""
    cabin = rng.choice(["economy", "business"])
    data = sample_airline_user(rng, {"cabin": cabin, "has_unlimited_payment": True})
    if data is None:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    user = data["user"]
    identity = _user_identity_airline(data)
    flights_db = data.get("flights_db", {})
    db = build_airline_db(user, res, flights_db)

    flights = res.get("flights", [])
    if not flights:
        return None

    seg = flights[0]
    origin = seg["origin"]
    dest = seg["destination"]
    old_date = seg["date"]

    # Find alternative flights
    candidates = []
    for fn, flight_data in flights_db.items():
        if flight_data["origin"] != origin or flight_data["destination"] != dest:
            continue
        for date, date_info in flight_data.get("dates", {}).items():
            if date == old_date:
                continue
            if not isinstance(date_info, dict) or date_info.get("status") != "available":
                continue
            seats = date_info.get("available_seats", {})
            if seats.get(cabin, 0) < len(res.get("passengers", [1])):
                continue
            candidates.append({"flight_number": fn, "date": date,
                             "price": date_info.get("prices", {}).get(cabin, 0)})

    if not candidates:
        return None

    new_flight = rng.choice(candidates)

    # Payment
    pm = user.get("payment_methods", {})
    valid_pms = [pid for pid, v in pm.items() if v.get("source") in ("credit_card", "paypal")]
    if not valid_pms:
        return None
    payment_id = rng.choice(valid_pms)

    new_flights = []
    for i, s in enumerate(flights):
        if i == 0:
            new_flights.append({"flight_number": new_flight["flight_number"], "date": new_flight["date"]})
        else:
            new_flights.append({"flight_number": s["flight_number"], "date": s["date"]})

    initial_msg = (
        f"Hi, I need to change my flight in reservation {res_id}. "
        f"I'm currently booked from {origin} to {dest} on {old_date}. "
        f"Can you search for available flights on {new_flight['date']} and switch me to one? "
        f"I'm flexible on the exact flight."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have reservation {res_id} with a {cabin} flight "
            f"from {origin} to {dest} on {old_date}. You want to change to {new_flight['date']}."
        ),
        goal=(
            f"Change the flight in reservation {res_id} from {old_date} to {new_flight['date']}. "
            f"The agent should search for available flights. Use payment {payment_id}."
        ),
        approach="cooperative",
        required_communication=f"Confirm when the flight change is done.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="multi_step",
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
        communicate_info=["flight"],
        description=f"Search + change flight on reservation {res_id}",
        key_facts={"reservation_id": res_id, "user_id": data["user_id"]},
        db=db,
    )


def _gen_airline_compensation(rng: random.Random) -> Optional[GeneratedScenario]:
    """Send compensation certificate for delayed/cancelled flight."""
    data = sample_airline_user(rng, {})
    if data is None:
        return None

    res = data["reservation"]
    user = data["user"]
    identity = _user_identity_airline(data)
    flights_db = data.get("flights_db", {})

    # Make one flight delayed or cancelled
    flights = res.get("flights", [])
    if not flights:
        return None
    seg = flights[0]
    fn = seg["flight_number"]
    date = seg["date"]

    issue_type = rng.choice(["delayed", "cancelled"])
    # Update flight status in flights_db — replace the date entry with correct Pydantic schema
    if fn in flights_db and date in flights_db[fn].get("dates", {}):
        if issue_type == "delayed":
            flights_db[fn]["dates"][date] = {
                "status": "delayed",
                "estimated_departure_time_est": f"{date}T{rng.randint(10,20):02d}:00:00",
                "estimated_arrival_time_est": f"{date}T{rng.randint(20,23):02d}:00:00",
            }
        else:
            flights_db[fn]["dates"][date] = {"status": "cancelled"}

    db = build_airline_db(user, res, flights_db)

    num_passengers = len(res.get("passengers", []))
    amount_per_pax = 50 if issue_type == "delayed" else 100
    total_amount = amount_per_pax * num_passengers

    initial_msg = (
        f"Hi, my flight {fn} on {date} was {issue_type}. "
        f"I'd like to receive compensation for this."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} Your flight {fn} on {date} was {issue_type}. "
            f"You have {num_passengers} passenger(s) on the reservation."
        ),
        goal=(
            f"Get compensation for the {issue_type} flight {fn}. "
            f"The policy says ${amount_per_pax} per passenger for {issue_type} flights."
        ),
        approach="cooperative",
        required_communication=f"Confirm the compensation amount.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("send_certificate", {
                "user_id": data["user_id"],
                "amount": total_amount,
            }),
        ],
        communicate_info=[str(total_amount)],
        description=f"Compensate for {issue_type} flight {fn} (${total_amount})",
        key_facts={
            "user_id": data["user_id"],
            "flight_number": fn,
            "issue_type": issue_type,
            "num_passengers": num_passengers,
            "amount": total_amount,
        },
        db=db,
    )


def _gen_airline_multi_reservation_ops(rng: random.Random) -> Optional[GeneratedScenario]:
    """Operate on multiple reservations — cancel one + update another."""
    data = sample_airline_multi_reservations(rng, {}, min_reservations=2)
    if data is None:
        return None

    user = data["user"]
    reservations = data["reservations"]
    flights_db = data.get("flights_db", {})
    identity = _user_identity_airline(data)
    db = build_airline_db(user, reservations, flights_db)

    if len(reservations) < 2:
        return None

    # Cancel first, update bags on second
    cancel_res = reservations[0]
    bag_res = reservations[1]

    cancel_id = cancel_res["reservation_id"]
    bag_id = bag_res["reservation_id"]

    bags_to_add = rng.randint(1, 2)
    current_total = bag_res.get("total_baggages", 0)
    current_nonfree = bag_res.get("nonfree_baggages", 0)
    new_total = current_total + bags_to_add
    new_nonfree = current_nonfree + bags_to_add

    pm = user.get("payment_methods", {})
    valid_pms = [pid for pid, v in pm.items() if v.get("source") in ("credit_card", "paypal")]
    if not valid_pms:
        return None
    payment_id = rng.choice(valid_pms)

    initial_msg = (
        f"Hi, I need to cancel reservation {cancel_id} and also add "
        f"{bags_to_add} bag(s) to reservation {bag_id}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You want to cancel reservation {cancel_id} and "
            f"add {bags_to_add} checked bag(s) to reservation {bag_id}."
        ),
        goal=(
            f"Cancel reservation {cancel_id}. Also add {bags_to_add} bag(s) "
            f"to reservation {bag_id}."
        ),
        approach="cooperative",
        required_communication="Confirm both operations.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("cancel_reservation", {"reservation_id": cancel_id}),
            ExpectedAction("update_reservation_baggages", {
                "reservation_id": bag_id,
                "total_baggages": new_total,
                "nonfree_baggages": new_nonfree,
                "payment_id": payment_id,
            }),
        ],
        communicate_info=["cancel", "bag"],
        description=f"Cancel {cancel_id} + add bags to {bag_id}",
        key_facts={"user_id": data["user_id"]},
        db=db,
    )


def _gen_retail_calculate_and_communicate(rng: random.Random) -> Optional[GeneratedScenario]:
    """Info query requiring looking up order totals."""
    data = sample_retail_multi_orders(rng, {}, order_specs=[
        {"status": "delivered", "min_items": 2},
        {"status": "delivered", "min_items": 1},
    ])
    if data is None:
        return None

    user = data["user"]
    orders = data["orders"]
    identity = _user_identity_retail(data)
    db = build_retail_db(user, orders, data.get("products_db", {}))

    # Compute total spent across all orders
    total_spent = sum(
        sum(item["price"] for item in order.get("items", []))
        for order in orders
    )
    total_str = f"{total_spent:.2f}"

    initial_msg = (
        f"Hi, can you tell me the total amount I've spent across all my orders?"
    )

    prompt = build_user_system_prompt(
        customer_context=f"{identity} You want to know the total amount across all your orders.",
        goal="Ask the agent to calculate the total amount spent across all your orders.",
        approach="cooperative",
        required_communication=f"The agent should tell you the total is {total_str}.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[],
        communicate_info=[total_str],
        description=f"Info: total spent across {len(orders)} orders = ${total_str}",
        key_facts={"user_id": data["user_id"], "total": total_str},
        db=db,
    )


# =====================================================================
# TYPE 2 ENHANCEMENTS: Complex selection scenarios
# =====================================================================

def _gen_retail_exchange_most_expensive(rng: random.Random) -> Optional[GeneratedScenario]:
    """Exchange a delivered item for the most expensive variant of the same product."""
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
        variant_list = [
            (vid, v) for vid, v in variants.items()
            if vid != current_item_id and v.get("available", True)
        ]
        if not variant_list:
            continue

        # Sort descending by price, tiebreak by item_id
        variant_list.sort(key=lambda x: (-x[1].get("price", 0), x[0]))
        most_expensive_vid, most_expensive_variant = variant_list[0]
        most_expensive_price = most_expensive_variant.get("price", 0)

        item_name = item["name"]

        # Payment method
        payment_history = order.get("payment_history", [])
        if payment_history:
            payment_method_id = payment_history[0].get("payment_method_id", "")
        else:
            pm = user.get("payment_methods", {})
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
            f"for the most expensive available option of the same product. "
            f"I want to upgrade to the premium version."
        )

        prompt = build_user_system_prompt(
            customer_context=(
                f"{identity} You have a delivered order {order_id} with a {item_name}. "
                f"You want to exchange it for the most expensive available variant."
            ),
            goal=(
                f"Exchange the {item_name} (item {current_item_id}) in order {order_id} "
                f"for the most expensive available variant of the same product."
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
                    "new_item_ids": [most_expensive_vid],
                    "payment_method_id": payment_method_id,
                }),
            ],
            communicate_info=["exchange"],
            description=f"Exchange {item_name} for most expensive variant in order {order_id}",
            key_facts={
                "order_id": order_id,
                "item_id": current_item_id,
                "item_name": item_name,
                "product_id": product_id,
                "most_expensive_variant_id": most_expensive_vid,
                "most_expensive_price": most_expensive_price,
                "payment_method_id": payment_method_id,
                "user_id": data["user_id"],
            },
            db=db,
        )

    return None


def _gen_retail_exchange_multi_attr(rng: random.Random) -> Optional[GeneratedScenario]:
    """Exchange item for a variant matching multiple attributes (e.g. 'blue cotton T-Shirt in size M')."""
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
        if len(variants) < 4:
            continue

        current_item_id = item["item_id"]
        current_options = item.get("options", {})

        # Find variants with at least 2 attribute differences from current
        candidates = []
        for vid, v in variants.items():
            if vid == current_item_id:
                continue
            if not v.get("available", True):
                continue
            v_options = v.get("options", {})
            diffs = {k: v_options[k] for k in v_options
                     if k in current_options and v_options[k] != current_options[k]}
            if len(diffs) >= 2:
                candidates.append((vid, v, diffs))

        if not candidates:
            continue

        # Pick one target variant
        target_vid, target_variant, target_diffs = rng_copy.choice(candidates)

        # Verify uniqueness: only ONE variant should match ALL the target attributes
        target_options = target_variant.get("options", {})
        matching = []
        for vid, v in variants.items():
            if vid == current_item_id:
                continue
            if not v.get("available", True):
                continue
            v_opts = v.get("options", {})
            # Check if ALL target diff attributes match
            if all(v_opts.get(k) == target_diffs[k] for k in target_diffs):
                matching.append(vid)

        if len(matching) != 1:
            # Ambiguous — skip this item (multiple variants match)
            continue

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

        # Build multi-attribute description
        attr_desc = " and ".join(f"{k} {v}" for k, v in target_diffs.items())

        initial_msg = (
            f"Hi, I'd like to exchange the {item_name} in my order {order_id} "
            f"for one with {attr_desc}."
        )

        prompt = build_user_system_prompt(
            customer_context=(
                f"{identity} You have a delivered order {order_id} with a {item_name}. "
                f"You want to exchange it for a variant with {attr_desc}."
            ),
            goal=(
                f"Exchange the {item_name} in order {order_id} for a variant "
                f"with {attr_desc}."
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
                    "new_item_ids": [target_vid],
                    "payment_method_id": payment_method_id,
                }),
            ],
            communicate_info=["exchange"],
            description=f"Exchange {item_name} for multi-attr match ({attr_desc})",
            key_facts={
                "order_id": order_id,
                "item_id": current_item_id,
                "item_name": item_name,
                "target_variant_id": target_vid,
                "target_attributes": target_diffs,
                "payment_method_id": payment_method_id,
                "user_id": data["user_id"],
            },
            db=db,
        )

    return None


def _gen_airline_cheapest_flight(rng: random.Random) -> Optional[GeneratedScenario]:
    """Change to the cheapest available flight on a specific date.

    The model must search flights on that date, compare prices, and select the cheapest.
    Constrained to one date so the model can find it with a single search call.
    """
    cabin = rng.choice(["economy", "business"])
    data = sample_airline_user(rng, {"cabin": cabin, "has_unlimited_payment": True})
    if data is None:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    user = data["user"]
    identity = _user_identity_airline(data)
    flights_db = data.get("flights_db", {})
    db = build_airline_db(user, res, flights_db)

    flights = res.get("flights", [])
    if not flights:
        return None

    seg = flights[0]
    origin = seg["origin"]
    dest = seg["destination"]
    old_date = seg["date"]

    # Group available flights by date (same origin/dest)
    by_date = {}
    for fn, flight_data in flights_db.items():
        if flight_data["origin"] != origin or flight_data["destination"] != dest:
            continue
        for date, date_info in flight_data.get("dates", {}).items():
            if date == old_date:
                continue
            if not isinstance(date_info, dict) or date_info.get("status") != "available":
                continue
            seats = date_info.get("available_seats", {})
            n_pax = len(res.get("passengers", [1]))
            if seats.get(cabin, 0) < n_pax:
                continue
            price = date_info.get("prices", {}).get(cabin, 0)
            by_date.setdefault(date, []).append({
                "flight_number": fn, "date": date, "price": price,
            })

    # Find a date with 2+ flights (so there's a meaningful selection)
    valid_dates = [(d, flights_list) for d, flights_list in by_date.items() if len(flights_list) >= 2]
    if not valid_dates:
        return None

    target_date, date_flights = rng.choice(valid_dates)

    # Sort by price, tiebreak by flight number
    date_flights.sort(key=lambda x: (x["price"], x["flight_number"]))
    cheapest = date_flights[0]

    # Ensure unambiguous (no price tie with second cheapest)
    if date_flights[1]["price"] == cheapest["price"]:
        return None

    # Payment
    pm = user.get("payment_methods", {})
    valid_pms = [pid for pid, v in pm.items() if v.get("source") in ("credit_card", "paypal")]
    if not valid_pms:
        return None
    payment_id = rng.choice(valid_pms)

    new_flights = []
    for i, s in enumerate(flights):
        if i == 0:
            new_flights.append({"flight_number": cheapest["flight_number"], "date": cheapest["date"]})
        else:
            new_flights.append({"flight_number": s["flight_number"], "date": s["date"]})

    initial_msg = (
        f"Hi, I need to change my flight in reservation {res_id}. "
        f"I'm booked from {origin} to {dest} on {old_date} but need to fly on {target_date} instead. "
        f"Can you find me the cheapest available flight from {origin} to {dest} on {target_date}?"
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have reservation {res_id} with a {cabin} flight "
            f"from {origin} to {dest} on {old_date}. You want to change to the cheapest "
            f"available flight on {target_date}."
        ),
        goal=(
            f"Change the flight in reservation {res_id} to the cheapest available option "
            f"from {origin} to {dest} on {target_date}. Use payment {payment_id}."
        ),
        approach="cooperative",
        required_communication=f"Confirm when the flight change is done.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="selection",
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
        communicate_info=["flight"],
        description=f"Change to cheapest flight {origin}->{dest} on {target_date} for reservation {res_id}",
        key_facts={
            "reservation_id": res_id,
            "user_id": data["user_id"],
            "cabin": cabin,
            "cheapest_flight": cheapest["flight_number"],
            "cheapest_date": cheapest["date"],
            "cheapest_price": cheapest["price"],
            "payment_id": payment_id,
        },
        db=db,
    )


def _gen_airline_book_new_reservation(rng: random.Random) -> Optional[GeneratedScenario]:
    """Book a new reservation — user provides all details.

    Tests the book_reservation tool which is common in airline eval.
    """
    data = sample_airline_user(rng, {"has_unlimited_payment": True})
    if data is None:
        return None

    user = data["user"]
    identity = _user_identity_airline(data)
    flights_db = data.get("flights_db", {})

    # Find an available flight to book
    candidates = []
    for fn, flight_data in flights_db.items():
        for date, date_info in flight_data.get("dates", {}).items():
            if not isinstance(date_info, dict) or date_info.get("status") != "available":
                continue
            seats = date_info.get("available_seats", {})
            if seats.get("economy", 0) >= 1:
                candidates.append({
                    "flight_number": fn,
                    "date": date,
                    "origin": flight_data["origin"],
                    "destination": flight_data["destination"],
                    "price": date_info.get("prices", {}).get("economy", 0),
                })

    if not candidates:
        return None

    flight = rng.choice(candidates)

    # Payment
    pm = user.get("payment_methods", {})
    valid_pms = [pid for pid, v in pm.items() if v.get("source") in ("credit_card", "paypal")]
    if not valid_pms:
        return None
    payment_id = rng.choice(valid_pms)

    # Passenger info from user
    user_name = user["name"]
    passenger = {
        "first_name": user_name["first_name"],
        "last_name": user_name["last_name"],
        "dob": f"19{rng.randint(60,99)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
    }

    insurance = rng.choice(["yes", "no"])

    db = build_airline_db(user, data["reservation"], flights_db)

    initial_msg = (
        f"Hi, I'd like to book a new reservation. I need a flight from "
        f"{flight['origin']} to {flight['destination']} on {flight['date']}. "
        f"Flight {flight['flight_number']} in economy class. "
        f"Passenger: {passenger['first_name']} {passenger['last_name']}, "
        f"DOB: {passenger['dob']}. "
        f"{'With' if insurance == 'yes' else 'Without'} travel insurance. "
        f"No checked bags."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You want to book a new reservation. "
            f"Flight {flight['flight_number']} from {flight['origin']} to {flight['destination']} "
            f"on {flight['date']}, economy class. "
            f"Passenger: {passenger['first_name']} {passenger['last_name']}, DOB {passenger['dob']}. "
            f"Insurance: {insurance}. Payment: {payment_id}. No checked bags."
        ),
        goal=(
            f"Book flight {flight['flight_number']} from {flight['origin']} to {flight['destination']} "
            f"on {flight['date']} in economy with payment {payment_id}. No checked bags."
        ),
        approach="cooperative",
        required_communication=f"Confirm when the booking is complete.",
    )

    return GeneratedScenario(
        domain="airline",
        scenario_type="explicit",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("book_reservation", {
                "user_id": data["user_id"],
                "origin": flight["origin"],
                "destination": flight["destination"],
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": flight["flight_number"], "date": flight["date"]}],
                "passengers": [passenger],
                "payment_methods": [{"payment_id": payment_id, "amount": flight["price"]}],
                "insurance": insurance,
                "total_baggages": 0,
                "nonfree_baggages": 0,
            }),
        ],
        communicate_info=["book"],
        description=f"Book new reservation {flight['origin']}->{flight['destination']}",
        key_facts={
            "user_id": data["user_id"],
            "flight": flight,
            "payment_id": payment_id,
            "passenger": passenger,
        },
        db=db,
    )



# =====================================================================
# TYPE 6: Conditional fallback scenarios
# =====================================================================

def _gen_retail_conditional_exchange(rng: random.Random) -> Optional[GeneratedScenario]:
    """User wants to exchange to a specific variant; if unavailable, fall back to another.

    We generate DB where the primary target variant is marked unavailable, so the
    expected action uses the fallback variant.
    """
    data = sample_retail_user(rng, {"status": "delivered", "min_items": 1})
    if data is None:
        return None

    order = data["order"]
    order_id = data["order_id"]
    user = data["user"]
    products_db = data.get("products_db", {})

    items = order.get("items", [])
    if not items:
        return None

    rng_copy = random.Random(rng.random())
    shuffled_items = list(items)
    rng_copy.shuffle(shuffled_items)

    for item in shuffled_items:
        product_id = item.get("product_id", "")
        product = products_db.get(product_id, {})
        variants = product.get("variants", {})
        if len(variants) < 4:
            continue

        current_item_id = item["item_id"]
        other_variants = [
            (vid, v) for vid, v in variants.items()
            if vid != current_item_id
        ]
        if len(other_variants) < 2:
            continue

        # Pick primary target (make it unavailable) and fallback (keep available)
        rng_copy.shuffle(other_variants)
        primary_vid, primary_var = other_variants[0]
        fallback_vid, fallback_var = other_variants[1]

        # Make primary unavailable in the DB
        primary_var["available"] = False
        # Make fallback available
        fallback_var["available"] = True

        primary_opts = primary_var.get("options", {})
        fallback_opts = fallback_var.get("options", {})
        primary_desc = ", ".join(f"{k}: {v}" for k, v in primary_opts.items())
        fallback_desc = ", ".join(f"{k}: {v}" for k, v in fallback_opts.items())

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
        db = build_retail_db(user, order, products_db)

        initial_msg = (
            f"Hi, I'd like to exchange the {item_name} in my order {order_id} "
            f"for one with {primary_desc}. But if that's not available, "
            f"I'll take one with {fallback_desc} instead."
        )

        prompt = build_user_system_prompt(
            customer_context=(
                f"{identity} You have a delivered order {order_id} with a {item_name}. "
                f"You want to exchange it. Your first choice is {primary_desc}, "
                f"but if unavailable your fallback is {fallback_desc}."
            ),
            goal=(
                f"Exchange the {item_name} in order {order_id}. "
                f"Prefer variant with {primary_desc}. "
                f"If unavailable, accept variant with {fallback_desc}."
            ),
            approach="cooperative",
            required_communication="Confirm which variant the exchange was processed with.",
        )

        return GeneratedScenario(
            domain="retail",
            scenario_type="conditional_fallback",
            user_system_prompt=prompt,
            initial_message=initial_msg,
            expected_actions=[
                ExpectedAction("exchange_delivered_order_items", {
                    "order_id": order_id,
                    "item_ids": [current_item_id],
                    "new_item_ids": [fallback_vid],
                    "payment_method_id": payment_method_id,
                }),
            ],
            communicate_info=["exchange", "unavailable"],
            description=f"Conditional exchange: {item_name} -> fallback {fallback_desc} (primary {primary_desc} unavailable)",
            key_facts={
                "order_id": order_id,
                "item_id": current_item_id,
                "item_name": item_name,
                "primary_variant_id": primary_vid,
                "fallback_variant_id": fallback_vid,
                "payment_method_id": payment_method_id,
                "user_id": data["user_id"],
            },
            db=db,
        )

    return None


def _gen_retail_conditional_cancel_or_return(rng: random.Random) -> Optional[GeneratedScenario]:
    """User says: 'cancel if pending, otherwise return it.'

    Randomly generate a pending or delivered order to test both branches.
    """
    # Randomly pick status to determine which branch is tested
    is_pending = rng.random() < 0.5
    status = "pending" if is_pending else "delivered"

    data = sample_retail_user(rng, {"status": status, "min_items": 1})
    if data is None:
        return None

    order = data["order"]
    order_id = data["order_id"]
    user = data["user"]
    identity = _user_identity_retail(data)
    db = build_retail_db(user, order, data.get("products_db", {}))

    item = order.get("items", [{}])[0]
    item_id = item.get("item_id", "")
    item_name = item.get("name", "item")

    initial_msg = (
        f"Hi, I want to get rid of my order {order_id}. "
        f"If it's still pending, just cancel it. "
        f"If it's already been delivered, I'd like to return the {item_name}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You want to cancel or return order {order_id}. "
            f"Cancel if it's pending; return the {item_name} if it's delivered."
        ),
        goal=(
            f"Cancel order {order_id} if pending, or return the {item_name} if delivered."
        ),
        approach="cooperative",
        required_communication=f"Confirm whether the order was cancelled or returned.",
    )

    if is_pending:
        reason = rng.choice(_CANCEL_REASONS)
        expected = [
            ExpectedAction("cancel_pending_order", {
                "order_id": order_id,
                "reason": reason,
            }),
        ]
        communicate = ["cancel"]
    else:
        payment_history = order.get("payment_history", [])
        if payment_history:
            payment_method_id = payment_history[0].get("payment_method_id", "")
        else:
            pm = user.get("payment_methods", {})
            valid_pms = [pid for pid, pv in pm.items() if pv.get("source") in ("credit_card", "paypal")]
            payment_method_id = valid_pms[0] if valid_pms else (list(pm.keys())[0] if pm else "")
        expected = [
            ExpectedAction("return_delivered_order_items", {
                "order_id": order_id,
                "item_ids": [item_id],
                "payment_method_id": payment_method_id,
            }),
        ]
        communicate = ["return"]

    return GeneratedScenario(
        domain="retail",
        scenario_type="conditional_fallback",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=expected,
        communicate_info=communicate,
        description=f"Conditional cancel-or-return: order {order_id} (status={status})",
        key_facts={
            "order_id": order_id,
            "status": status,
            "user_id": data["user_id"],
        },
        db=db,
    )


def _gen_retail_conditional_modify_or_exchange(rng: random.Random) -> Optional[GeneratedScenario]:
    """User says: 'if order is pending, modify the item; if delivered, exchange it.'

    Randomly pick pending or delivered order.
    """
    is_pending = rng.random() < 0.5
    status = "pending" if is_pending else "delivered"

    data = sample_retail_user(rng, {"status": status, "min_items": 1})
    if data is None:
        return None

    order = data["order"]
    order_id = data["order_id"]
    user = data["user"]
    products_db = data.get("products_db", {})
    identity = _user_identity_retail(data)

    items = order.get("items", [])
    if not items:
        return None

    # Find item with alternative variant
    for item in items:
        product_id = item.get("product_id", "")
        product = products_db.get(product_id, {})
        variants = product.get("variants", {})
        current_item_id = item["item_id"]
        other_variants = [
            (vid, v) for vid, v in variants.items()
            if vid != current_item_id and v.get("available", True)
        ]
        if not other_variants:
            continue

        new_vid, new_var = rng.choice(other_variants)
        new_opts = new_var.get("options", {})
        opts_desc = ", ".join(f"{k}: {v}" for k, v in new_opts.items())
        item_name = item["name"]

        # Payment method
        pm = user.get("payment_methods", {})
        valid_pms = [pid for pid, pv in pm.items() if pv.get("source") in ("credit_card", "paypal")]
        payment_method_id = valid_pms[0] if valid_pms else (list(pm.keys())[0] if pm else "")
        if not payment_method_id:
            continue

        db = build_retail_db(user, order, products_db)

        initial_msg = (
            f"Hi, I want to change the {item_name} in order {order_id} to the "
            f"variant with {opts_desc}. If the order is still pending, just modify it. "
            f"If it's already delivered, exchange it instead."
        )

        prompt = build_user_system_prompt(
            customer_context=(
                f"{identity} You want to switch the {item_name} in order {order_id} "
                f"to the variant with {opts_desc}. Modify if pending, exchange if delivered."
            ),
            goal=(
                f"Change the {item_name} in order {order_id} to {opts_desc}. "
                f"Modify if pending, exchange if delivered."
            ),
            approach="cooperative",
            required_communication=f"Confirm the item change.",
        )

        if is_pending:
            expected = [
                ExpectedAction("modify_pending_order_items", {
                    "order_id": order_id,
                    "item_ids": [current_item_id],
                    "new_item_ids": [new_vid],
                    "payment_method_id": payment_method_id,
                }),
            ]
        else:
            expected = [
                ExpectedAction("exchange_delivered_order_items", {
                    "order_id": order_id,
                    "item_ids": [current_item_id],
                    "new_item_ids": [new_vid],
                    "payment_method_id": payment_method_id,
                }),
            ]

        return GeneratedScenario(
            domain="retail",
            scenario_type="conditional_fallback",
            user_system_prompt=prompt,
            initial_message=initial_msg,
            expected_actions=expected,
            communicate_info=[item_name.lower()],
            description=f"Conditional modify-or-exchange: {item_name} in order {order_id} (status={status})",
            key_facts={
                "order_id": order_id,
                "status": status,
                "item_id": current_item_id,
                "new_variant_id": new_vid,
                "user_id": data["user_id"],
            },
            db=db,
        )

    return None


def _gen_airline_conditional_modify_or_cancel(rng: random.Random) -> Optional[GeneratedScenario]:
    """User says: 'change flight to X; if basic economy can't change, cancel and rebook.'

    Generate with basic_economy cabin so the expected path is cancel + book.
    """
    data = sample_airline_user(rng, {"cabin": "basic_economy", "has_unlimited_payment": True})
    if data is None:
        return None

    res_id = data["reservation_id"]
    res = data["reservation"]
    user = data["user"]
    identity = _user_identity_airline(data)
    flights_db = data.get("flights_db", {})

    flights = res.get("flights", [])
    if not flights:
        return None

    seg = flights[0]
    origin = seg["origin"]
    dest = seg["destination"]
    old_date = seg["date"]

    # Find an alternative flight
    candidates = []
    for fn, flight_data in flights_db.items():
        if flight_data["origin"] != origin or flight_data["destination"] != dest:
            continue
        for date, date_info in flight_data.get("dates", {}).items():
            if date == old_date:
                continue
            if not isinstance(date_info, dict) or date_info.get("status") != "available":
                continue
            seats = date_info.get("available_seats", {})
            if seats.get("economy", 0) >= len(res.get("passengers", [1])):
                candidates.append({
                    "flight_number": fn, "date": date,
                    "price": date_info.get("prices", {}).get("economy", 0),
                })

    if not candidates:
        return None

    new_flight = rng.choice(candidates)

    # Payment
    pm = user.get("payment_methods", {})
    valid_pms = [pid for pid, v in pm.items() if v.get("source") in ("credit_card", "paypal")]
    if not valid_pms:
        return None
    payment_id = rng.choice(valid_pms)

    # Passenger info
    passengers = res.get("passengers", [])
    if not passengers:
        return None

    db = build_airline_db(user, res, flights_db)

    new_date = new_flight["date"]
    new_fn = new_flight["flight_number"]

    initial_msg = (
        f"Hi, I need to change my flight in reservation {res_id} from "
        f"{origin} to {dest} on {old_date} to {new_date}. "
        f"If that's not possible because of my fare class, "
        f"please cancel the reservation and book me a new one on flight "
        f"{new_fn} on {new_date} in economy."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have reservation {res_id} in basic economy from "
            f"{origin} to {dest} on {old_date}. You want to switch to "
            f"{new_date}. If basic economy can't be changed, cancel and rebook."
        ),
        goal=(
            f"Try to change reservation {res_id} to {new_date}. "
            f"If refused due to basic economy, cancel reservation {res_id} and "
            f"book a new one on flight {new_fn} on {new_date} "
            f"in economy with payment {payment_id}."
        ),
        approach="cooperative",
        required_communication="Confirm when the rebooking is complete.",
    )

    # Expected: cancel + book (since basic economy can't be modified)
    return GeneratedScenario(
        domain="airline",
        scenario_type="conditional_fallback",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("cancel_reservation", {"reservation_id": res_id}),
            ExpectedAction("book_reservation", {
                "user_id": data["user_id"],
                "origin": origin,
                "destination": dest,
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": new_flight["flight_number"], "date": new_flight["date"]}],
                "passengers": passengers,
                "payment_methods": [{"payment_id": payment_id, "amount": new_flight["price"]}],
                "insurance": res.get("insurance", "no"),
                "total_baggages": 0,
                "nonfree_baggages": 0,
            }),
        ],
        communicate_info=["cancel", "book"],
        description=f"Conditional modify-or-cancel+rebook: reservation {res_id} (basic economy)",
        key_facts={
            "reservation_id": res_id,
            "user_id": data["user_id"],
            "cabin": "basic_economy",
            "new_flight": new_flight,
            "payment_id": payment_id,
        },
        db=db,
    )


def _gen_airline_conditional_compensation(rng: random.Random) -> Optional[GeneratedScenario]:
    """User complains about delayed flight. If eligible for compensation, send certificate.
    If not eligible (regular member, basic economy, no insurance), agent should explain and
    the user asks for transfer to a supervisor.

    We generate an INELIGIBLE user so the expected action is just transfer.
    """
    data = sample_airline_user(rng, {
        "cabin": "basic_economy",
        "insurance": "no",
    })
    if data is None:
        return None

    res = data["reservation"]
    user = data["user"]
    identity = _user_identity_airline(data)
    flights_db = data.get("flights_db", {})

    flights = res.get("flights", [])
    if not flights:
        return None
    seg = flights[0]
    fn = seg["flight_number"]
    date = seg["date"]

    # Make the flight delayed
    if fn in flights_db and date in flights_db[fn].get("dates", {}):
        flights_db[fn]["dates"][date] = {
            "status": "delayed",
            "estimated_departure_time_est": f"{date}T{rng.randint(10,20):02d}:00:00",
            "estimated_arrival_time_est": f"{date}T{rng.randint(20,23):02d}:00:00",
        }

    db = build_airline_db(user, res, flights_db)

    initial_msg = (
        f"Hi, my flight {fn} on {date} was delayed significantly. "
        f"I'd like compensation for the inconvenience. "
        f"If you can't compensate me, please transfer me to a supervisor."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} Your flight {fn} on {date} was delayed. "
            f"You want compensation. If the agent says you're not eligible, "
            f"ask to be transferred to a supervisor."
        ),
        goal=(
            f"Get compensation for delayed flight {fn}. "
            f"If not eligible, request a transfer to a supervisor."
        ),
        approach="cooperative",
        required_communication="Find out if you can get compensation or need to be transferred.",
    )

    # Ineligible: agent should transfer to human
    return GeneratedScenario(
        domain="airline",
        scenario_type="conditional_fallback",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("transfer_to_human_agent", {
                "summary": (
                    f"Customer requesting compensation for delayed flight {fn} on {date}. "
                    f"Basic economy, no insurance. Not eligible for automatic compensation."
                ),
            }),
        ],
        communicate_info=["not eligible", "transfer"],
        description=f"Conditional compensation: flight {fn} delayed, ineligible -> transfer",
        key_facts={
            "user_id": data["user_id"],
            "flight_number": fn,
            "date": date,
            "cabin": "basic_economy",
            "insurance": "no",
        },
        db=db,
    )


# =====================================================================
# TYPE 4 ADDITION: Cross-entity reasoning
# =====================================================================

def _gen_retail_cross_entity_address(rng: random.Random) -> Optional[GeneratedScenario]:
    """User says: 'change my pending order's address to match my other order's address.'

    Generate 2 orders at different addresses. The user references the address of order B
    without stating it explicitly. Agent must look up order B's address and apply it to order A.
    """
    data = sample_retail_multi_orders(rng, {}, order_specs=[
        {"status": "pending", "min_items": 1},
        {"status": "delivered", "min_items": 1},
    ])
    if data is None:
        return None

    user = data["user"]
    orders = data["orders"]
    identity = _user_identity_retail(data)

    pending_order = next((o for o in orders if o["status"] == "pending"), None)
    delivered_order = next((o for o in orders if o["status"] == "delivered"), None)
    if not pending_order or not delivered_order:
        return None

    # Get the delivered order's address (the reference address)
    ref_addr = delivered_order.get("address", {})
    if not ref_addr or not ref_addr.get("address1"):
        return None

    # Make sure the pending order has a DIFFERENT address
    pending_addr = pending_order.get("address", {})
    if pending_addr == ref_addr:
        return None

    db = build_retail_db(user, orders, data.get("products_db", {}))

    pending_oid = pending_order["order_id"]
    delivered_oid = delivered_order["order_id"]

    initial_msg = (
        f"Hi, I need to update the shipping address on my pending order "
        f"{pending_oid}. Please change it to the same address "
        f"that's on my other order {delivered_oid}."
    )

    prompt = build_user_system_prompt(
        customer_context=(
            f"{identity} You have a pending order {pending_oid} and "
            f"a delivered order {delivered_oid}. You want the pending "
            f"order's shipping address changed to match the delivered order's address. "
            f"You don't remember the exact address -- the agent should look it up."
        ),
        goal=(
            f"Change the shipping address on order {pending_oid} to match "
            f"the address on order {delivered_oid}. Do NOT state the address "
            f"yourself -- let the agent look it up."
        ),
        approach="cooperative",
        required_communication="Confirm when the address is updated.",
    )

    return GeneratedScenario(
        domain="retail",
        scenario_type="multi_step",
        user_system_prompt=prompt,
        initial_message=initial_msg,
        expected_actions=[
            ExpectedAction("modify_pending_order_address", {
                "order_id": pending_oid,
                **ref_addr,
            }),
        ],
        communicate_info=["address"],
        description=f"Cross-entity: copy address from {delivered_oid} to {pending_oid}",
        key_facts={
            "user_id": data["user_id"],
            "pending_order_id": pending_oid,
            "reference_order_id": delivered_oid,
            "reference_address": ref_addr,
        },
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
    _gen_airline_book_new_reservation,
]

# Type 2: Selection generators
_TYPE2_GENERATORS = [
    _gen_retail_exchange_cheapest,
    _gen_retail_exchange_specific_attr,
    _gen_retail_exchange_most_expensive,
    _gen_retail_exchange_multi_attr,
    _gen_retail_modify_items,
    _gen_airline_cheapest_flight,
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
    _gen_airline_cancel_eligible,
    _gen_retail_cancel_pending,
]

# Type 4: Multi-step scenarios (2-4 actions)
_TYPE4_GENERATORS = [
    _gen_retail_multi_return,
    _gen_retail_cancel_and_return,
    _gen_retail_address_propagation,
    _gen_retail_exchange_and_return,
    _gen_retail_modify_items_and_address,
    _gen_airline_search_and_change,
    _gen_airline_compensation,
    _gen_airline_multi_reservation_ops,
    _gen_retail_cross_entity_address,
]

# Type 5: Info + communicate (numeric answers)
_TYPE5_GENERATORS = [
    _gen_retail_calculate_and_communicate,
    _gen_retail_info_query,
    _gen_airline_info_query,
]

# Type 6: Conditional fallback scenarios
_TYPE6_GENERATORS = [
    _gen_retail_conditional_exchange,
    _gen_retail_conditional_cancel_or_return,
    _gen_retail_conditional_modify_or_exchange,
    _gen_airline_conditional_modify_or_cancel,
    _gen_airline_conditional_compensation,
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

    # Rebalanced weights to include multi-step and info scenarios
    scenario_type = rng.choices(
        ["explicit", "selection", "policy_gated", "multi_step", "info_communicate", "conditional_fallback"],
        weights=[20, 15, 15, 30, 10, 10],
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
        elif stype == "multi_step":
            return list(_TYPE4_GENERATORS)
        elif stype == "info_communicate":
            return list(_TYPE5_GENERATORS)
        elif stype == "conditional_fallback":
            return list(_TYPE6_GENERATORS)
        return []

    # Try generators with retries (some may fail to find matching entities)
    max_retries = 10

    # Try preferred scenario type first, then fall back to others
    all_types = ["explicit", "selection", "policy_gated", "multi_step", "info_communicate", "conditional_fallback"]
    type_order = [scenario_type] + [t for t in all_types if t != scenario_type]

    def _strip_identifiers(msg: str) -> str:
        """Remove order IDs (#Wxxxxxxx), reservation IDs (6-char uppercase), and user IDs from a message.

        Replaces with vague descriptions so the initial message doesn't
        contradict progressive disclosure instructions.
        """
        import re
        # Strip retail order IDs (#W followed by digits)
        msg = re.sub(r'#W\d{7}', 'my recent order', msg)
        # Strip airline reservation IDs (standalone 6-char uppercase alphanum, e.g., EHGLP3)
        msg = re.sub(r'\b[A-Z0-9]{6}\b', 'my reservation', msg)
        # Clean up duplicate references
        msg = re.sub(r'(my recent order|my reservation)(,?\s*(my recent order|my reservation))+', r'\1', msg)
        return msg

    def _apply_progressive_disclosure(scenario, disc_rng):
        """Post-process: maybe inject progressive disclosure into user prompt."""
        if scenario is None:
            return scenario
        new_prompt = scenario.user_system_prompt
        if disc_rng.random() < 0.3:
            instruction = disc_rng.choice(_PROGRESSIVE_DISCLOSURE_INSTRUCTIONS)
            # Inject into <your_situation> block
            if "</your_situation>" in new_prompt:
                new_prompt = new_prompt.replace(
                    "</your_situation>",
                    " " + instruction + "\n</your_situation>",
                )
            else:
                # Fallback: prepend to prompt
                new_prompt = instruction + "\n" + new_prompt
            scenario.user_system_prompt = new_prompt
            # Also strip identifiers from initial_message to avoid contradiction
            scenario.initial_message = _strip_identifiers(scenario.initial_message)
        return scenario

    def _break_adversarial_coupling(scenario, mix_rng):
        """Inject adversarial tone into some non-refusal scenarios and cooperative tone
        into some refusal scenarios to prevent 'adversarial = refuse' shortcut.

        Target distribution:
          REFUSE: ~70% adversarial (already set by generators), 30% flipped to cooperative
          non-REFUSE: ~20% flipped to adversarial (legitimate but pushy users)
        """
        if scenario is None:
            return scenario
        prompt = scenario.user_system_prompt or ''
        is_adv = 'conflict with their policies' in prompt

        if scenario.is_refusal and is_adv and mix_rng.random() < 0.3:
            # Flip some adversarial refusals to cooperative (polite policy violators)
            from adversarial_policy_game.llm_user import _build_cooperative_prompt
            # Rebuild as cooperative by swapping the <role> and <approach> blocks
            prompt = prompt.replace(
                'You will try to get the agent to\nfulfill your request, even if it may conflict with their policies.',
                'You have a straightforward, valid request.')
            if '<approach>' in prompt:
                import re
                prompt = re.sub(r'<approach>.*?</approach>', '<approach>\nBe polite and cooperative. State your request clearly.\n</approach>', prompt, flags=re.DOTALL)
                prompt = re.sub(r'<rules>.*?</rules>', '<rules>\n- Stay in character as the customer at all times\n- Keep responses concise (1-2 sentences)\n- If the agent refuses, accept gracefully\n- End your FINAL message with [DONE] when the conversation is truly over\n- Never reveal you are an AI or break character\n</rules>', prompt, flags=re.DOTALL)
            scenario.user_system_prompt = prompt
        elif not scenario.is_refusal and not is_adv and mix_rng.random() < 0.2:
            # Flip some cooperative non-refusals to adversarial (pushy but legitimate)
            prompt = prompt.replace(
                'You have a straightforward, valid request.',
                'You will try to get the agent to\nfulfill your request, even if it may conflict with their policies.')
            if '<approach>' in prompt:
                import re
                prompt = re.sub(r'<approach>.*?</approach>', '<approach>\nBe persistent about getting your request handled quickly. Express impatience if the agent is slow.\n</approach>', prompt, flags=re.DOTALL)
            scenario.user_system_prompt = prompt
        return scenario

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
                    # Break adversarial-refusal coupling (deterministic per seed)
                    mix_rng = random.Random(seed + 55555)
                    result = _break_adversarial_coupling(result, mix_rng)
                    # Apply progressive disclosure (deterministic per seed)
                    disc_rng = random.Random(seed + 77777)
                    return _apply_progressive_disclosure(result, disc_rng)

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
