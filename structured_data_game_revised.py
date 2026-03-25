"""Structured Data Reasoning Game - tau2-bench Aligned (Airline + Retail)

Multi-turn game using EXACT tau2-bench tools, system prompt, and format.
Trains the model to reason over structured JSON data returned by tools
and take the correct action or communicate the correct answer.

Airline scenario types:
  1. flight_selection (30%):  User wants cheapest/shortest/earliest flight from
     search results. Agent must search, compare, and book or communicate.
  2. baggage_computation (20%):  User asks about free/paid baggage. Agent must
     look up reservation + user details, apply the policy table, and respond.
  3. reservation_comparison (20%):  User asks which reservation is cheapest,
     which has the most passengers, etc. Agent must look up multiple
     reservations and compare.
  4. cost_computation (15%):  User wants total cost of upgrading cabin or adding
     bags. Agent must look up details, search flights, compute differences.
  5. flight_status_check (15%):  User asks about flight status for a reservation.
     Agent must get reservation details, then check flight status, and report.

Retail scenario types:
  1. variant_selection (40%):  User wants to exchange an item for a specific
     variant. Agent must get_product_details, find matching variant from 10+
     variants, and identify the correct item_id.
  2. price_comparison (30%):  User asks which order is cheapest, which item
     costs most, total across orders, etc. Agent must get_order_details for
     multiple orders and compare.
  3. order_status_check (30%):  User asks about order status, item details,
     payment methods. Agent must look up and report correctly.

Uses LLM user simulator, ToolExecutor, and the same GameEnv protocol as
precondition_game and tau_tool_calling_env.

Reward: binary 0/1 based on whether the agent communicates the correct answer.
"""

import random
import copy
import json
import re
import sys
import pathlib
from typing import Dict, List, Any, Optional, Tuple

# Ensure tau2-bench source is in path for get_dict_hash
_TAU2_SRC = str(pathlib.Path(__file__).resolve().parent / "tau2-bench" / "src")
if _TAU2_SRC not in sys.path:
    sys.path.insert(0, _TAU2_SRC)

from tau2.utils.utils import get_dict_hash
from dataclasses import dataclass, field

from adversarial_policy_game.tools import ToolExecutor
from adversarial_policy_game.constants import (
    AIRLINE_POLICY,
    AIRLINE_TOOL_SCHEMAS,
    RETAIL_POLICY,
    RETAIL_TOOL_SCHEMAS,
)
from adversarial_policy_game.llm_user import LLMUser, UserLLMClient, build_user_system_prompt
from adversarial_policy_game.database import (
    sample_airline_multi_reservations,
    sample_retail_multi_orders,
)
from adversarial_policy_game.synthetic_db import build_airline_db, build_retail_db


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class SDRScenario:
    """A structured data reasoning scenario."""
    domain: str  # "airline" or "retail"
    scenario_type: str
    user_system_prompt: str
    initial_message: str
    expected_answer: Any  # the value(s) the agent must communicate
    communicate_info: List[str] = field(default_factory=list)
    description: str = ""
    key_facts: Dict[str, Any] = field(default_factory=dict)
    db: Dict[str, Any] = field(default_factory=dict)
    # For mutation scenarios: expected tool call the agent must make
    expected_tool_call: Optional[Dict[str, Any]] = None  # {"name": ..., "key_args": {...}}
    # Full args for gold DB replay (needed for DB hash verification)
    gold_tool_args: Optional[Dict[str, Any]] = None  # complete arguments for ToolExecutor.execute()


# =====================================================================
# Constants
# =====================================================================

IATA_CODES = [
    "JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "SEA",
    "MIA", "EWR", "PHX", "IAH", "LAS", "PHL", "DTW", "BOS",
    "MSP", "CLT", "MCO", "TPA",
]

CABIN_CLASSES = ["basic_economy", "economy", "business"]

# Free checked baggage per passenger by (membership, cabin) — airline only
FREE_BAGS = {
    ("regular", "basic_economy"): 0, ("regular", "economy"): 1, ("regular", "business"): 2,
    ("silver", "basic_economy"): 1, ("silver", "economy"): 2, ("silver", "business"): 3,
    ("gold", "basic_economy"): 2, ("gold", "economy"): 3, ("gold", "business"): 4,
}

_COOPERATIVE_STYLES = [
    "You are polite and cooperative.",
    "You are concise and business-like.",
    "You are friendly and patient.",
    "You are direct and efficient.",
    "You are calm and straightforward.",
]

SCENARIO_WEIGHTS = {
    # Exact match to eval SD failure distribution (excl. policy tasks):
    # 67% mutation, 33% report — from 18 non-policy airline SD failures
    "book_flight": 28,             # 5/18 eval failures: wrong book_reservation args
    "change_flight": 22,           # 4/18: wrong update_reservation_flights args
    "send_compensation": 17,       # 3/18: wrong send_certificate amount
    "reservation_comparison": 11,  # 2/18: multi-lookup comparison
    "flight_selection": 8,         # 1/18: search + report
    "baggage_computation": 6,      # 1/18: lookup + compute
    "cost_computation": 4,         # general report
    "flight_status_check": 4,      # general report
    "conditional_flight_change": 8,  # Gap 3: conditional fallback
}

RETAIL_SCENARIO_WEIGHTS = {
    # Exact match to eval SD failure distribution (excl. policy tasks):
    # 79% mutation, 21% report — from 38 non-policy retail SD failures
    "execute_exchange": 32,        # 12/38: wrong variant in exchange
    "execute_modify": 24,          # 9/38: wrong variant in modify
    "execute_return": 18,          # 7/38: wrong item in return
    "price_comparison": 8,         # 3/38: communicate fail on comparison
    "order_status_check": 8,       # 2/38: communicate fail on lookup
    "execute_cancel": 5,           # 2/38: wrong cancel args
    "variant_selection": 5,        # 1/38: report-only variant
    "conditional_exchange": 10,    # Gap 3: conditional fallback
    "cross_entity_exchange": 5,    # Gap 6: cross-entity reasoning
}


# =====================================================================
# Synthetic DB helpers (airline)
# =====================================================================

def _gen_flight(rng: random.Random, origin: str, dest: str, date: str,
                base_price: float, status: str = "available") -> Tuple[str, Dict]:
    """Generate a single flight entry for the flights DB."""
    fnum = f"SDR{rng.randint(100, 999)}"
    hour = rng.randint(6, 21)
    minute = rng.choice([0, 15, 30, 45])
    dep_time = f"{hour:02d}:{minute:02d}:00"
    duration_h = rng.randint(2, 8)
    duration_m = rng.choice([0, 15, 30, 45])
    arr_hour = (hour + duration_h + (minute + duration_m) // 60) % 24
    arr_min = (minute + duration_m) % 60
    arr_time = f"{arr_hour:02d}:{arr_min:02d}:00"

    econ_price = round(base_price + rng.uniform(-50, 50), 0)
    be_price = round(econ_price * rng.uniform(0.55, 0.75), 0)
    biz_price = round(econ_price * rng.uniform(2.0, 3.0), 0)

    flight_data = {
        "flight_number": fnum,
        "origin": origin,
        "destination": dest,
        "scheduled_departure_time_est": dep_time,
        "scheduled_arrival_time_est": arr_time,
        "dates": {
            date: {
                "status": status,
                "available_seats": {
                    "basic_economy": rng.randint(0, 30),
                    "economy": rng.randint(5, 50),
                    "business": rng.randint(2, 15),
                },
                "prices": {
                    "basic_economy": be_price,
                    "economy": econ_price,
                    "business": biz_price,
                },
            }
        },
    }

    if status == "delayed":
        flight_data["dates"][date] = {
            "status": "delayed",
            "estimated_departure_time_est": f"{date}T{(hour+2)%24:02d}:{minute:02d}:00",
            "estimated_arrival_time_est": f"{date}T{(arr_hour+2)%24:02d}:{arr_min:02d}:00",
        }
    elif status == "cancelled":
        flight_data["dates"][date] = {"status": "cancelled"}

    return fnum, flight_data


def _gen_user_with_reservations(rng: random.Random, n_reservations: int = 3,
                                 membership: Optional[str] = None) -> Tuple[Dict, List[Dict], Dict]:
    """Generate a user with multiple reservations and supporting flights DB."""
    from adversarial_policy_game.synthetic_db import (
        FIRST_NAMES, LAST_NAMES, CITIES_STATES_ZIPS, STREETS,
        _gen_dob, _gen_email, _gen_payment_id,
    )

    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    uid = f"{first.lower()}_{last.lower()}_{rng.randint(1000, 9999)}"
    membership = membership or rng.choice(["regular", "silver", "gold"])

    # Payment methods
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
        "amount": rng.choice([100, 200, 300, 500]),
    }

    city, state, zipcode = rng.choice(CITIES_STATES_ZIPS)
    dob = _gen_dob(rng)

    user = {
        "user_id": uid,
        "name": {"first_name": first, "last_name": last},
        "address": {
            "address1": f"{rng.randint(100, 999)} {rng.choice(STREETS)}",
            "address2": "", "city": city, "state": state, "zip": zipcode, "country": "USA",
        },
        "email": _gen_email(first, last, rng),
        "dob": dob,
        "payment_methods": payment_methods,
        "saved_passengers": [{"first_name": first, "last_name": last, "dob": dob}],
        "membership": membership,
        "reservations": [],
    }

    # Generate reservations
    flights_db = {}
    reservations = []
    used_routes = set()

    for i in range(n_reservations):
        # Pick unique route
        origin = rng.choice(IATA_CODES)
        dest = rng.choice([c for c in IATA_CODES if c != origin])
        while (origin, dest) in used_routes:
            origin = rng.choice(IATA_CODES)
            dest = rng.choice([c for c in IATA_CODES if c != origin])
        used_routes.add((origin, dest))

        day = rng.randint(15, 28)
        date = f"2024-05-{day:02d}"
        cabin = rng.choice(CABIN_CLASSES)
        base_price = rng.uniform(150, 600)

        fnum, flight_data = _gen_flight(rng, origin, dest, date, base_price)
        flights_db[fnum] = flight_data

        price = flight_data["dates"][date]["prices"][cabin]
        n_pax = rng.randint(1, 3)
        total_bags = rng.randint(0, n_pax * 3)
        free_per_pax = FREE_BAGS.get((membership, cabin), 0)
        nonfree = max(0, total_bags - free_per_pax * n_pax)

        passengers = [{"first_name": first, "last_name": last, "dob": dob}]
        for _ in range(n_pax - 1):
            pax_first = rng.choice(FIRST_NAMES)
            pax_last = last
            passengers.append({
                "first_name": pax_first, "last_name": pax_last,
                "dob": _gen_dob(rng),
            })

        res_id = f"SDR{rng.randint(10000, 99999)}"
        reservation = {
            "reservation_id": res_id,
            "user_id": uid,
            "origin": origin,
            "destination": dest,
            "flight_type": "one_way",
            "cabin": cabin,
            "flights": [{"flight_number": fnum, "date": date,
                         "origin": origin, "destination": dest, "price": price}],
            "passengers": passengers,
            "payment_history": [{"payment_id": cc_id, "amount": price * n_pax}],
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
# Progressive disclosure helper (Gap 7)
# =====================================================================

def _maybe_add_progressive_disclosure(rng: random.Random, user_sys: str) -> str:
    """For ~25% of scenarios, add progressive disclosure instruction to user system prompt."""
    if rng.random() < 0.25:
        user_sys += (
            "\n\nIMPORTANT: Don't reveal all details immediately. "
            "Wait for the agent to ask for specific information before providing it."
        )
    return user_sys


# =====================================================================
# Scenario generators
# =====================================================================

def _gen_flight_selection(rng: random.Random) -> SDRScenario:
    """User wants cheapest/shortest/earliest flight. Agent must search and report."""
    user, reservations, flights_db = _gen_user_with_reservations(rng, n_reservations=1)
    uid = user["user_id"]
    name = user["name"]

    origin = rng.choice(IATA_CODES)
    dest = rng.choice([c for c in IATA_CODES if c != origin])
    day = rng.randint(18, 25)
    date = f"2024-05-{day:02d}"

    # Pick selection criterion first so we can ensure no ties
    cabin = rng.choice(["economy", "business"])
    criterion = rng.choice(["cheapest", "earliest", "most_seats"])

    # Generate 4-8 flights for the search results, ensuring no ties on criterion
    n_flights = rng.randint(4, 8)
    search_flights = []
    used_dep_times = set()
    used_prices = set()
    used_seats = set()
    for _ in range(n_flights):
        fnum, fdata = _gen_flight(rng, origin, dest, date,
                                   base_price=rng.uniform(150, 600))
        # Break ties based on criterion
        if criterion == "earliest":
            dep = fdata["scheduled_departure_time_est"]
            while dep in used_dep_times:
                # Shift by 15 min
                h, m = int(dep[:2]), int(dep[3:5])
                m = (m + 15) % 60
                if m == 0: h = (h + 1) % 24
                dep = f"{h:02d}:{m:02d}:00"
                fdata["scheduled_departure_time_est"] = dep
            used_dep_times.add(dep)
        elif criterion == "cheapest":
            p = fdata["dates"][date]["prices"][cabin]
            while p in used_prices:
                p += 1
            used_prices.add(p)
            fdata["dates"][date]["prices"][cabin] = p
        else:  # most_seats
            s = fdata["dates"][date]["available_seats"][cabin]
            while s in used_seats:
                s += 1
            used_seats.add(s)
            fdata["dates"][date]["available_seats"][cabin] = s

        flights_db[fnum] = fdata
        search_flights.append((fnum, fdata))

    if criterion == "cheapest":
        best = min(search_flights,
                   key=lambda x: x[1]["dates"][date]["prices"][cabin])
        best_fnum = best[0]
        best_price = best[1]["dates"][date]["prices"][cabin]
        question = f"I need the cheapest {cabin} flight from {origin} to {dest} on May {day}."
        communicate = [best_fnum, str(int(best_price))]
        expected = {"flight": best_fnum, "price": best_price}
    elif criterion == "earliest":
        best = min(search_flights,
                   key=lambda x: x[1]["scheduled_departure_time_est"])
        best_fnum = best[0]
        dep_time = best[1]["scheduled_departure_time_est"]
        question = f"I need the earliest {cabin} flight from {origin} to {dest} on May {day}."
        communicate = [best_fnum, dep_time[:5]]
        expected = {"flight": best_fnum, "departure": dep_time}
    else:  # most_seats
        best = max(search_flights,
                   key=lambda x: x[1]["dates"][date]["available_seats"][cabin])
        best_fnum = best[0]
        seats = best[1]["dates"][date]["available_seats"][cabin]
        question = f"Which {cabin} flight from {origin} to {dest} on May {day} has the most available seats?"
        communicate = [best_fnum, str(seats)]
        expected = {"flight": best_fnum, "seats": seats}

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question} "
        f"Can you search and tell me which flight that is?"
    )

    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. {question}"
        ),
        goal=question,
        approach="cooperative",
        required_communication="Confirm when the agent gives you the flight number and details.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    return SDRScenario(
        domain="airline",
        scenario_type="flight_selection",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected,
        communicate_info=communicate,
        description=f"Flight selection: {criterion} {cabin} {origin}->{dest}",
        key_facts={"criterion": criterion, "cabin": cabin, "origin": origin,
                   "dest": dest, "date": date, "best_flight": best_fnum},
        db=db,
    )


def _gen_baggage_computation(rng: random.Random) -> SDRScenario:
    """User asks about baggage allowance. Agent must look up and compute."""
    membership = rng.choice(["regular", "silver", "gold"])
    user, reservations, flights_db = _gen_user_with_reservations(
        rng, n_reservations=rng.randint(2, 4), membership=membership)
    uid = user["user_id"]
    name = user["name"]

    # Pick a reservation to ask about
    res = rng.choice(reservations)
    cabin = res["cabin"]
    n_pax = len(res["passengers"])
    total_bags = res["total_baggages"]
    free_per_pax = FREE_BAGS.get((membership, cabin), 0)
    free_total = free_per_pax * n_pax
    nonfree = max(0, total_bags - free_total)

    question_type = rng.choice(["free_bags", "nonfree_bags", "can_add"])

    if question_type == "free_bags":
        question = (
            f"How many free checked bags am I allowed per passenger on "
            f"reservation {res['reservation_id']}?"
        )
        communicate = [str(free_per_pax)]
        expected = {"free_per_pax": free_per_pax}
    elif question_type == "nonfree_bags":
        question = (
            f"How many of my {total_bags} total bags on reservation "
            f"{res['reservation_id']} are paid (non-free) bags?"
        )
        communicate = [str(nonfree)]
        expected = {"nonfree_bags": nonfree}
    else:  # can_add
        max_bags_pp = 3 if cabin == "business" else 2
        max_total = max_bags_pp * n_pax
        can_add = max(0, max_total - total_bags)
        question = (
            f"How many more bags can I add to reservation {res['reservation_id']}? "
            f"I currently have {total_bags} bags."
        )
        communicate = [str(can_add)]
        expected = {"can_add": can_add}

    initial_msg = (
        f"Hi, I'm {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question}"
    )

    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. {question}"
        ),
        goal=question,
        approach="cooperative",
        required_communication="Wait for the agent's answer. Confirm if correct.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    return SDRScenario(
        domain="airline",
        scenario_type="baggage_computation",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected,
        communicate_info=communicate,
        description=f"Baggage computation: {question_type} ({membership}/{cabin})",
        key_facts={"membership": membership, "cabin": cabin, "n_pax": n_pax,
                   "total_bags": total_bags, "question_type": question_type},
        db=db,
    )


def _gen_reservation_comparison(rng: random.Random) -> SDRScenario:
    """User asks which reservation is cheapest/has most passengers/etc."""
    user, reservations, flights_db = _gen_user_with_reservations(
        rng, n_reservations=rng.randint(3, 5))
    uid = user["user_id"]
    name = user["name"]

    comparison = rng.choice(["cheapest", "most_passengers", "most_bags"])

    if comparison == "cheapest":
        # Total cost = sum of payment history amounts
        def cost(r):
            return sum(p["amount"] for p in r["payment_history"])
        best = min(reservations, key=cost)
        answer_val = cost(best)
        question = "Which of my reservations has the lowest total cost?"
        communicate = [best["reservation_id"], str(int(answer_val))]
        expected = {"reservation_id": best["reservation_id"], "cost": answer_val}
    elif comparison == "most_passengers":
        best = max(reservations, key=lambda r: len(r["passengers"]))
        n = len(best["passengers"])
        question = "Which of my reservations has the most passengers?"
        communicate = [best["reservation_id"], str(n)]
        expected = {"reservation_id": best["reservation_id"], "passengers": n}
    else:  # most_bags
        best = max(reservations, key=lambda r: r["total_baggages"])
        n = best["total_baggages"]
        question = "Which of my reservations has the most checked bags?"
        communicate = [best["reservation_id"], str(n)]
        expected = {"reservation_id": best["reservation_id"], "bags": n}

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question}"
    )

    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. {question}"
        ),
        goal=question,
        approach="cooperative",
        required_communication="Wait for the agent's answer.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    return SDRScenario(
        domain="airline",
        scenario_type="reservation_comparison",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected,
        communicate_info=communicate,
        description=f"Reservation comparison: {comparison}",
        key_facts={"comparison": comparison,
                   "best_res": best["reservation_id"]},
        db=db,
    )


def _gen_cost_computation(rng: random.Random) -> SDRScenario:
    """User asks about cost of upgrading or price difference."""
    user, reservations, flights_db = _gen_user_with_reservations(
        rng, n_reservations=rng.randint(2, 3))
    uid = user["user_id"]
    name = user["name"]

    # Pick a reservation to upgrade
    res = rng.choice(reservations)
    flight_info = res["flights"][0]
    fnum = flight_info["flight_number"]
    date = flight_info["date"]
    current_cabin = res["cabin"]
    current_price = flight_info["price"]

    # Find an upgrade target cabin
    cabin_order = {"basic_economy": 0, "economy": 1, "business": 2}
    possible_upgrades = [c for c in CABIN_CLASSES
                         if cabin_order[c] > cabin_order[current_cabin]]

    if not possible_upgrades:
        # Already in business - ask about downgrade saving instead
        possible_upgrades = [c for c in CABIN_CLASSES
                             if cabin_order[c] < cabin_order[current_cabin]]
        target_cabin = rng.choice(possible_upgrades) if possible_upgrades else "economy"
        target_price = flights_db[fnum]["dates"][date]["prices"][target_cabin]
        diff = current_price - target_price
        n_pax = len(res["passengers"])
        total_diff = diff * n_pax

        question = (
            f"How much would I save per passenger if I downgraded reservation "
            f"{res['reservation_id']} from {current_cabin} to {target_cabin}?"
        )
        communicate = [str(int(diff))]
        expected = {"saving_per_pax": diff, "total_saving": total_diff}
    else:
        target_cabin = rng.choice(possible_upgrades)
        target_price = flights_db[fnum]["dates"][date]["prices"][target_cabin]
        diff = target_price - current_price
        n_pax = len(res["passengers"])
        total_diff = diff * n_pax

        question = (
            f"How much extra per passenger would it cost to upgrade reservation "
            f"{res['reservation_id']} from {current_cabin} to {target_cabin}?"
        )
        communicate = [str(int(diff))]
        expected = {"cost_per_pax": diff, "total_cost": total_diff}

    initial_msg = (
        f"Hi, I'm {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question}"
    )

    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. {question}"
        ),
        goal=question,
        approach="cooperative",
        required_communication="Wait for the agent to calculate and tell you the cost.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    return SDRScenario(
        domain="airline",
        scenario_type="cost_computation",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected,
        communicate_info=communicate,
        description=f"Cost computation: {current_cabin}->{target_cabin}",
        key_facts={"reservation": res["reservation_id"], "current_cabin": current_cabin,
                   "target_cabin": target_cabin, "diff": diff},
        db=db,
    )


def _gen_flight_status_check(rng: random.Random) -> SDRScenario:
    """User asks about flight status. Agent must look up and report."""
    user, reservations, flights_db = _gen_user_with_reservations(
        rng, n_reservations=rng.randint(2, 3))
    uid = user["user_id"]
    name = user["name"]

    # Pick a reservation and set its flight to a status
    res = rng.choice(reservations)
    flight_info = res["flights"][0]
    fnum = flight_info["flight_number"]
    date = flight_info["date"]

    status = rng.choice(["available", "delayed", "cancelled"])
    old_data = flights_db[fnum]["dates"][date]

    if status == "delayed":
        dep = flights_db[fnum]["scheduled_departure_time_est"]
        arr = flights_db[fnum]["scheduled_arrival_time_est"]
        dep_h = int(dep[:2])
        arr_h = int(arr[:2])
        delay_h = rng.randint(1, 4)
        flights_db[fnum]["dates"][date] = {
            "status": "delayed",
            "estimated_departure_time_est": f"{date}T{(dep_h + delay_h) % 24:02d}:{dep[3:5]}:00",
            "estimated_arrival_time_est": f"{date}T{(arr_h + delay_h) % 24:02d}:{arr[3:5]}:00",
        }
        communicate = ["delayed"]
        expected = {"status": "delayed", "flight": fnum}
    elif status == "cancelled":
        flights_db[fnum]["dates"][date] = {"status": "cancelled"}
        communicate = ["cancelled"]
        expected = {"status": "cancelled", "flight": fnum}
    else:
        # on time / available — tool returns "available" so accept either phrasing
        communicate = ["available", fnum]
        expected = {"status": "available", "flight": fnum}

    question = (
        f"Can you check the status of my flight on reservation "
        f"{res['reservation_id']}? Is it on time?"
    )

    initial_msg = (
        f"Hi, I'm {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question}"
    )

    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. {question}"
        ),
        goal=question,
        approach="cooperative",
        required_communication="Wait for the agent to tell you the flight status.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    return SDRScenario(
        domain="airline",
        scenario_type="flight_status_check",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected,
        communicate_info=communicate,
        description=f"Flight status: {fnum} is {status}",
        key_facts={"flight": fnum, "status": status,
                   "reservation": res["reservation_id"]},
        db=db,
    )


def _gen_book_flight(rng: random.Random) -> SDRScenario:
    """User wants to book a flight with specific constraints.

    Agent must search flights, pick the right one meeting constraints
    (cheapest economy, specific date, payment method), and call book_reservation.
    Mirrors eval failure: wrong arguments on book_reservation (21% of airline SD failures).
    """
    user, reservations, flights_db = _gen_user_with_reservations(rng, n_reservations=1)
    uid = user["user_id"]
    name = user["name"]
    membership = user["membership"]

    origin = rng.choice(IATA_CODES)
    dest = rng.choice([c for c in IATA_CODES if c != origin])
    day = rng.randint(18, 25)
    date = f"2024-05-{day:02d}"
    cabin = rng.choice(["economy", "business"])

    # Generate 4-6 flights, ensure unique prices for the target cabin
    n_flights = rng.randint(4, 6)
    search_flights = []
    used_prices = set()
    for _ in range(n_flights):
        fnum, fdata = _gen_flight(rng, origin, dest, date,
                                   base_price=rng.uniform(150, 600))
        p = fdata["dates"][date]["prices"][cabin]
        while p in used_prices:
            p += 1
        used_prices.add(p)
        fdata["dates"][date]["prices"][cabin] = p
        flights_db[fnum] = fdata
        search_flights.append((fnum, fdata))

    # Pick cheapest flight as the expected booking
    best = min(search_flights, key=lambda x: x[1]["dates"][date]["prices"][cabin])
    best_fnum = best[0]
    best_price = best[1]["dates"][date]["prices"][cabin]

    # Pick payment method — sometimes specify credit card, sometimes gift card
    pm_ids = list(user["payment_methods"].keys())
    pm_id = rng.choice(pm_ids)
    pm_info = user["payment_methods"][pm_id]
    if pm_info["source"] == "credit_card":
        pm_desc = f"my credit card ending in {pm_info['last_four']}"
    elif pm_info["source"] == "gift_card":
        pm_desc = "my gift card"
    else:
        pm_desc = f"payment method {pm_id}"

    n_passengers = rng.randint(1, 2)
    pax_list = user["saved_passengers"][:n_passengers]
    if n_passengers > len(pax_list):
        pax_list = user["saved_passengers"][:1]
        n_passengers = 1

    question = (
        f"I need to book the cheapest {cabin} flight from {origin} to {dest} "
        f"on May {day}. Please use {pm_desc} for payment. "
        f"There will be {n_passengers} passenger{'s' if n_passengers > 1 else ''}."
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. {question}"
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm when the booking is made. Provide the flight number.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    expected_tool = {
        "name": "book_reservation",
        "key_args": {
            "user_id": uid,
            "origin": origin,
            "destination": dest,
            "flight_type": "one_way",
            "cabin": cabin,
            "payment_id": pm_id,
        },
    }

    return SDRScenario(
        domain="airline",
        scenario_type="book_flight",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"flight": best_fnum, "price": best_price},
        communicate_info=[best_fnum],
        description=f"Book flight: cheapest {cabin} {origin}->{dest} May {day}",
        key_facts={"origin": origin, "dest": dest, "date": date, "cabin": cabin,
                   "best_flight": best_fnum, "payment_id": pm_id},
        db=db,
        expected_tool_call=expected_tool,
    )


def _gen_change_flight(rng: random.Random) -> SDRScenario:
    """User wants to change their flight to a different one on the same route.

    Agent must look up reservation, search for alternatives, pick the right one,
    and call update_reservation_flights.
    Mirrors eval failure: wrong arguments on update_reservation_flights (12.5%).
    """
    user, reservations, flights_db = _gen_user_with_reservations(
        rng, n_reservations=rng.randint(2, 3))
    uid = user["user_id"]
    name = user["name"]

    # Pick a reservation to change
    res = rng.choice(reservations)
    flight_info = res["flights"][0]
    fnum = flight_info["flight_number"]
    origin = res["origin"]
    dest = res["destination"]
    date = flight_info["date"]
    cabin = res["cabin"]

    # Add 3-5 alternative flights on the same route
    alternatives = []
    used_prices = {flights_db[fnum]["dates"][date]["prices"][cabin]}
    for _ in range(rng.randint(3, 5)):
        alt_fnum, alt_fdata = _gen_flight(rng, origin, dest, date,
                                           base_price=rng.uniform(150, 600))
        p = alt_fdata["dates"][date]["prices"][cabin]
        while p in used_prices:
            p += 1
        used_prices.add(p)
        alt_fdata["dates"][date]["prices"][cabin] = p
        flights_db[alt_fnum] = alt_fdata
        alternatives.append((alt_fnum, alt_fdata))

    # User wants the cheapest alternative
    best_alt = min(alternatives, key=lambda x: x[1]["dates"][date]["prices"][cabin])
    new_fnum = best_alt[0]
    new_price = best_alt[1]["dates"][date]["prices"][cabin]

    pm_id = list(user["payment_methods"].keys())[0]

    question = (
        f"I'd like to change the flight on reservation {res['reservation_id']} "
        f"to the cheapest available {cabin} flight on the same route. "
        f"Please use my payment method on file."
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. {question}"
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm when the flight has been changed. Tell me the new flight number.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    expected_tool = {
        "name": "update_reservation_flights",
        "key_args": {
            "reservation_id": res["reservation_id"],
            "cabin": cabin,
        },
    }

    return SDRScenario(
        domain="airline",
        scenario_type="change_flight",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"new_flight": new_fnum, "new_price": new_price},
        communicate_info=[new_fnum],
        description=f"Change flight: {res['reservation_id']} to cheapest {cabin}",
        key_facts={"reservation": res["reservation_id"], "cabin": cabin,
                   "new_flight": new_fnum, "origin": origin, "dest": dest},
        db=db,
        expected_tool_call=expected_tool,
    )


def _gen_send_compensation(rng: random.Random) -> SDRScenario:
    """User has a cancelled/delayed flight and requests compensation certificate.

    Agent must look up reservation, check flight status, compute the correct amount,
    and call send_certificate.
    Mirrors eval failure: wrong send_certificate amount (8%).
    """
    membership = rng.choice(["silver", "gold"])  # must be eligible
    user, reservations, flights_db = _gen_user_with_reservations(
        rng, n_reservations=rng.randint(2, 4), membership=membership)
    uid = user["user_id"]
    name = user["name"]

    # Pick a reservation and set its flight to cancelled or delayed
    res = rng.choice(reservations)
    flight_info = res["flights"][0]
    fnum = flight_info["flight_number"]
    date = flight_info["date"]

    status = rng.choice(["cancelled", "delayed"])
    if status == "delayed":
        dep = flights_db[fnum]["scheduled_departure_time_est"]
        dep_h = int(dep[:2])
        delay_h = rng.randint(2, 5)
        flights_db[fnum]["dates"][date] = {
            "status": "delayed",
            "estimated_departure_time_est": f"{date}T{(dep_h + delay_h) % 24:02d}:{dep[3:5]}:00",
            "estimated_arrival_time_est": f"{date}T{(dep_h + delay_h + 3) % 24:02d}:{dep[3:5]}:00",
        }
    else:
        flights_db[fnum]["dates"][date] = {"status": "cancelled"}

    # Compute expected certificate amount: $100 per passenger for cancelled,
    # $50 per passenger for delayed (matching typical airline policy)
    n_pax = len(res["passengers"])
    cabin = res["cabin"]
    insurance = res.get("insurance", "no")

    # Amount depends on cabin and status per policy
    if status == "cancelled":
        per_pax = 100
    else:
        per_pax = 50
    amount = per_pax * n_pax

    question = (
        f"My flight {fnum} on reservation {res['reservation_id']} appears to be "
        f"{status}. I'd like to request a travel certificate as compensation."
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. {question}"
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm the certificate amount. Confirm it has been issued.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    expected_tool = {
        "name": "send_certificate",
        "key_args": {
            "user_id": uid,
            "amount": amount,
        },
    }

    return SDRScenario(
        domain="airline",
        scenario_type="send_compensation",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"amount": amount, "status": status},
        communicate_info=[str(amount)],
        description=f"Compensation: {status} flight, {n_pax} pax, ${amount}",
        key_facts={"flight": fnum, "status": status, "amount": amount,
                   "reservation": res["reservation_id"]},
        db=db,
        expected_tool_call=expected_tool,
    )


# =====================================================================
# Gap 3: Conditional flight change (airline)
# =====================================================================

def _gen_conditional_flight_change(rng: random.Random) -> SDRScenario:
    """User wants to change flight to cheapest option on a date.
    If all options are more expensive, they want to cancel instead.

    50% of time: cheaper alternatives exist -> expected: update_reservation_flights
    50% of time: all more expensive -> expected: cancel_reservation
    """
    user, reservations, flights_db = _gen_user_with_reservations(
        rng, n_reservations=rng.randint(2, 3))
    uid = user["user_id"]
    name = user["name"]

    res = rng.choice(reservations)
    flight_info = res["flights"][0]
    fnum = flight_info["flight_number"]
    origin = res["origin"]
    dest = res["destination"]
    # Use the actual date present in the flights_db (flight number collisions
    # can cause the reservation date to differ from the flights_db date)
    date = list(flights_db[fnum]["dates"].keys())[0]
    cabin = res["cabin"]
    current_price = flights_db[fnum]["dates"][date]["prices"][cabin]

    # Decide: cheaper available or all more expensive
    has_cheaper = rng.random() < 0.5

    # Generate 3-5 alternative flights
    alternatives = []
    used_prices = {current_price}
    for _ in range(rng.randint(3, 5)):
        if has_cheaper:
            base = rng.uniform(80, current_price * 1.5)
        else:
            # All more expensive than current
            base = rng.uniform(current_price + 50, current_price * 2.5)
        alt_fnum, alt_fdata = _gen_flight(rng, origin, dest, date, base_price=base)
        p = alt_fdata["dates"][date]["prices"][cabin]
        if not has_cheaper and p <= current_price:
            p = current_price + rng.uniform(10, 100)
        while p in used_prices:
            p += 1
        used_prices.add(p)
        alt_fdata["dates"][date]["prices"][cabin] = p
        flights_db[alt_fnum] = alt_fdata
        alternatives.append((alt_fnum, alt_fdata))

    # Ensure at least one is actually cheaper if has_cheaper
    if has_cheaper:
        cheapest_alt = min(alternatives, key=lambda x: x[1]["dates"][date]["prices"][cabin])
        cheapest_price = cheapest_alt[1]["dates"][date]["prices"][cabin]
        if cheapest_price >= current_price:
            # Force one to be cheaper
            cheapest_alt[1]["dates"][date]["prices"][cabin] = current_price - rng.uniform(20, 80)

    pm_id = list(user["payment_methods"].keys())[0]

    question = (
        f"I'd like to change my flight on reservation {res['reservation_id']} "
        f"to the cheapest option on {date}. If nothing cheaper than my current "
        f"flight is available, I'd rather cancel the reservation."
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My user ID is {uid}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your user id is {uid}. "
            f"Your current flight price is ${current_price:.0f} in {cabin}. "
            f"{question}"
        ),
        goal=question,
        approach=approach,
        required_communication=(
            "Confirm whether a cheaper flight was found or if the reservation was cancelled."
        ),
    )
    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_airline_db(user, reservations, flights_db)

    if has_cheaper:
        best_alt = min(alternatives, key=lambda x: x[1]["dates"][date]["prices"][cabin])
        new_fnum = best_alt[0]
        new_price = best_alt[1]["dates"][date]["prices"][cabin]
        expected_tool = {
            "name": "update_reservation_flights",
            "key_args": {
                "reservation_id": res["reservation_id"],
                "cabin": cabin,
            },
        }
        communicate = [new_fnum]
        expected_answer = {"new_flight": new_fnum, "new_price": new_price, "action": "change"}
        desc = f"Conditional flight change: cheaper found {new_fnum}"
    else:
        expected_tool = {
            "name": "cancel_reservation",
            "key_args": {
                "reservation_id": res["reservation_id"],
            },
        }
        communicate = [res["reservation_id"]]
        expected_answer = {"action": "cancel", "reservation": res["reservation_id"]}
        desc = f"Conditional flight change: no cheaper, cancel {res['reservation_id']}"

    return SDRScenario(
        domain="airline",
        scenario_type="conditional_flight_change",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected_answer,
        communicate_info=communicate,
        description=desc,
        key_facts={"reservation": res["reservation_id"], "cabin": cabin,
                   "current_price": current_price, "has_cheaper": has_cheaper},
        db=db,
        expected_tool_call=expected_tool,
    )


# =====================================================================
# Retail scenario generators
# =====================================================================

def _gen_retail_sample(rng: random.Random, n_orders: int = 3,
                       min_items: int = 2, need_delivered: bool = False,
                       many_variants: bool = False) -> Dict[str, Any]:
    """Helper: generate a retail user with multiple orders via synthetic_db.

    Returns the raw dict from generate_retail_multi_orders with keys:
      user, orders, order_ids, user_id, products_db
    """
    order_specs = []
    for i in range(n_orders):
        if need_delivered and i == 0:
            status = "delivered"
        else:
            status = rng.choice(["pending", "delivered"])
        order_specs.append({"status": status, "min_items": min_items})

    criteria = {
        "has_gift_card": rng.random() < 0.3,
        "has_multiple_payment_types": rng.random() < 0.4,
    }
    sample = sample_retail_multi_orders(rng, criteria, order_specs)

    # For variant_selection scenarios, bulk up the variant count to 10+
    if many_variants and sample:
        from adversarial_policy_game.synthetic_db import (
            PRODUCT_CATALOG, _gen_item_id,
        )
        for prod_id, prod_entry in sample["products_db"].items():
            variants = prod_entry["variants"]
            # Find the product template for option generation
            tmpl = None
            for p in PRODUCT_CATALOG:
                if p["name"] == prod_entry["name"]:
                    tmpl = p
                    break
            if tmpl is None:
                continue
            # Pad to at least 10 variants
            while len(variants) < 12:
                v_id = _gen_item_id(rng)
                v_opts = {}
                for opt_name, opt_values in tmpl["options_template"].items():
                    v_opts[opt_name] = rng.choice(opt_values)
                v_price = round(rng.uniform(15, 350), 2)
                variants[v_id] = {
                    "item_id": v_id,
                    "options": v_opts,
                    "available": rng.random() > 0.15,
                    "price": v_price,
                }

    return sample


def _gen_variant_selection(rng: random.Random) -> SDRScenario:
    """User wants to exchange a delivered item for a specific variant.

    Agent must get_product_details, scan 10+ variants, and find the one
    matching the requested options.  Key failure: picking wrong variant
    from a dense option space.
    """
    sample = _gen_retail_sample(rng, n_orders=2, min_items=2,
                                need_delivered=True, many_variants=True)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    # Pick a delivered order with items
    delivered = [o for o in orders if o["status"] == "delivered"]
    if not delivered:
        # Fallback: force one to be delivered
        orders[0]["status"] = "delivered"
        orders[0]["fulfillments"] = [{
            "tracking_id": [str(rng.randint(100000000000, 999999999999))],
            "item_ids": [it["item_id"] for it in orders[0]["items"]],
        }]
        delivered = [orders[0]]

    order = rng.choice(delivered)
    item = rng.choice(order["items"])
    prod_id = item["product_id"]
    prod_entry = products_db[prod_id]
    variants = prod_entry["variants"]

    # Pick a target variant that is available, different from current item,
    # and has UNIQUE options (no other available variant shares the same options)
    all_option_sets = {}  # frozen options -> list of variant ids
    for v_id, v in variants.items():
        key = tuple(sorted(v["options"].items()))
        all_option_sets.setdefault(key, []).append(v_id)

    available_variants = [
        v for v_id, v in variants.items()
        if v["available"] and v_id != item["item_id"]
        and len(all_option_sets[tuple(sorted(v["options"].items()))]) == 1  # unique options
    ]
    if not available_variants:
        # Fallback: deduplicate by removing extra variants with same options
        seen_opts = set()
        for v_id in list(variants.keys()):
            key = tuple(sorted(variants[v_id]["options"].items()))
            if key in seen_opts:
                del variants[v_id]
            else:
                seen_opts.add(key)
        available_variants = [
            v for v_id, v in variants.items()
            if v["available"] and v_id != item["item_id"]
        ]
        if not available_variants:
            for v_id, v in variants.items():
                if v_id != item["item_id"]:
                    v["available"] = True
                    available_variants = [v]
                    break

    target = rng.choice(available_variants)
    target_item_id = target["item_id"]
    target_options = target["options"]
    target_price = target["price"]

    # Build natural language description of desired options
    opt_desc = ", ".join(f"{k}: {v}" for k, v in target_options.items())

    question = (
        f"I'd like to exchange the {item['name']} (item {item['item_id']}) "
        f"from order {order['order_id']} for a different variant. "
        f"I want the one with {opt_desc}. "
        f"Can you find the right item ID for that variant?"
    )

    communicate = [target_item_id]
    expected = {
        "target_item_id": target_item_id,
        "target_options": target_options,
        "product_id": prod_id,
    }

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. "
            f"You want to exchange your {item['name']} for a variant with {opt_desc}."
        ),
        goal=question,
        approach="cooperative",
        required_communication="Wait for the agent to confirm the item ID of the variant.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    return SDRScenario(
        domain="retail",
        scenario_type="variant_selection",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected,
        communicate_info=communicate,
        description=f"Variant selection: {item['name']} -> {opt_desc}",
        key_facts={"product_id": prod_id, "target_item_id": target_item_id,
                   "target_options": target_options, "order_id": order["order_id"]},
        db=db,
    )


def _gen_price_comparison(rng: random.Random) -> SDRScenario:
    """User asks which order is cheapest, which item costs most, total across orders, etc.

    Agent must get_order_details for multiple orders and compare prices.
    """
    n_orders = rng.randint(3, 5)
    sample = _gen_retail_sample(rng, n_orders=n_orders, min_items=2)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    comparison = rng.choice([
        "cheapest_order", "most_expensive_order",
        "cheapest_item", "most_expensive_item", "total_all_orders",
    ])

    if comparison == "cheapest_order":
        def order_total(o):
            return sum(it["price"] for it in o["items"])
        # Ensure no ties by adding small perturbations
        totals = {}
        for o in orders:
            t = order_total(o)
            while t in totals.values():
                o["items"][0]["price"] = round(o["items"][0]["price"] + 0.01, 2)
                t = order_total(o)
            totals[o["order_id"]] = t
        best = min(orders, key=order_total)
        val = round(order_total(best), 2)
        question = "Which of my orders has the lowest total cost?"
        communicate = [best["order_id"], str(val)]
        expected = {"order_id": best["order_id"], "total": val}

    elif comparison == "most_expensive_order":
        def order_total(o):
            return sum(it["price"] for it in o["items"])
        totals = {}
        for o in orders:
            t = order_total(o)
            while t in totals.values():
                o["items"][0]["price"] = round(o["items"][0]["price"] + 0.01, 2)
                t = order_total(o)
            totals[o["order_id"]] = t
        best = max(orders, key=order_total)
        val = round(order_total(best), 2)
        question = "Which of my orders has the highest total cost?"
        communicate = [best["order_id"], str(val)]
        expected = {"order_id": best["order_id"], "total": val}

    elif comparison == "cheapest_item":
        all_items = []
        for o in orders:
            for it in o["items"]:
                all_items.append((o["order_id"], it))
        # Break ties
        prices = set()
        for oid, it in all_items:
            while it["price"] in prices:
                it["price"] = round(it["price"] + 0.01, 2)
            prices.add(it["price"])
        best_oid, best_item = min(all_items, key=lambda x: x[1]["price"])
        question = "Across all my orders, which single item is the cheapest? Tell me the item name, price, and which order it's in."
        communicate = [best_item["name"], str(best_item["price"])]
        expected = {"item_name": best_item["name"], "price": best_item["price"],
                    "order_id": best_oid}

    elif comparison == "most_expensive_item":
        all_items = []
        for o in orders:
            for it in o["items"]:
                all_items.append((o["order_id"], it))
        prices = set()
        for oid, it in all_items:
            while it["price"] in prices:
                it["price"] = round(it["price"] + 0.01, 2)
            prices.add(it["price"])
        best_oid, best_item = max(all_items, key=lambda x: x[1]["price"])
        question = "Across all my orders, which single item is the most expensive? Tell me the item name, price, and which order it's in."
        communicate = [best_item["name"], str(best_item["price"])]
        expected = {"item_name": best_item["name"], "price": best_item["price"],
                    "order_id": best_oid}

    else:  # total_all_orders
        grand_total = round(sum(
            sum(it["price"] for it in o["items"]) for o in orders
        ), 2)
        question = f"What is the total cost across all {len(orders)} of my orders combined?"
        communicate = [str(grand_total)]
        expected = {"grand_total": grand_total, "n_orders": len(orders)}

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. {question}"
        ),
        goal=question,
        approach="cooperative",
        required_communication="Wait for the agent's answer.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    return SDRScenario(
        domain="retail",
        scenario_type="price_comparison",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected,
        communicate_info=communicate,
        description=f"Price comparison: {comparison}",
        key_facts={"comparison": comparison, "n_orders": len(orders)},
        db=db,
    )


def _gen_order_status_check(rng: random.Random) -> SDRScenario:
    """User asks about order status, item details, or payment methods.

    Agent must look up and report correctly.
    """
    sample = _gen_retail_sample(rng, n_orders=rng.randint(2, 4), min_items=2)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    query_type = rng.choice([
        "order_status", "item_count", "payment_method", "item_options",
    ])

    order = rng.choice(orders)
    order_id = order["order_id"]

    if query_type == "order_status":
        question = f"What is the status of my order {order_id}?"
        communicate = [order["status"]]
        expected = {"order_id": order_id, "status": order["status"]}

    elif query_type == "item_count":
        n_items = len(order["items"])
        question = f"How many items are in my order {order_id}?"
        communicate = [str(n_items)]
        expected = {"order_id": order_id, "item_count": n_items}

    elif query_type == "payment_method":
        # Ask about payment method used for an order
        pay_hist = order["payment_history"]
        if pay_hist:
            pay_id = pay_hist[0]["payment_method_id"]
            pay_method = user["payment_methods"].get(pay_id, {})
            source = pay_method.get("source", "unknown")
            question = f"What payment method was used for my order {order_id}?"
            communicate = [source]
            expected = {"order_id": order_id, "payment_source": source,
                        "payment_id": pay_id}
        else:
            question = f"What is the status of my order {order_id}?"
            communicate = [order["status"]]
            expected = {"order_id": order_id, "status": order["status"]}

    else:  # item_options
        item = rng.choice(order["items"])
        opts = item["options"]
        opt_strs = [f"{v}" for v in opts.values()]
        question = (
            f"Can you tell me the options/details for the {item['name']} "
            f"in my order {order_id}?"
        )
        communicate = opt_strs
        expected = {"order_id": order_id, "item_name": item["name"],
                    "options": opts}

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. {question}"
        ),
        goal=question,
        approach="cooperative",
        required_communication="Wait for the agent to provide the information.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    return SDRScenario(
        domain="retail",
        scenario_type="order_status_check",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer=expected,
        communicate_info=communicate,
        description=f"Order status check: {query_type} on {order_id}",
        key_facts={"query_type": query_type, "order_id": order_id},
        db=db,
    )


def _gen_execute_exchange(rng: random.Random) -> SDRScenario:
    """User wants to exchange a delivered item for a different variant.

    Agent must find user, look up MULTIPLE orders to find the right one,
    get product details, scan 10+ variants for the right one, and call
    exchange_delivered_order_items.
    Mirrors eval: 32% of retail SD failures. Eval has mean 4+ lookups.
    """
    # More orders = more lookups needed (eval mean = 4.1 lookups)
    sample = _gen_retail_sample(rng, n_orders=rng.randint(3, 5), min_items=2,
                                need_delivered=True, many_variants=True)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    delivered = [o for o in orders if o["status"] == "delivered"]
    if not delivered:
        orders[0]["status"] = "delivered"
        orders[0]["fulfillments"] = [{
            "tracking_id": [str(rng.randint(100000000000, 999999999999))],
            "item_ids": [it["item_id"] for it in orders[0]["items"]],
        }]
        delivered = [orders[0]]

    order = rng.choice(delivered)
    item = rng.choice(order["items"])
    prod_id = item["product_id"]
    prod_entry = products_db[prod_id]
    variants = prod_entry["variants"]

    # Pick a target variant (available, different, unique options)
    all_option_sets = {}
    for v_id, v in variants.items():
        key = tuple(sorted(v["options"].items()))
        all_option_sets.setdefault(key, []).append(v_id)

    available_variants = [
        v for v_id, v in variants.items()
        if v["available"] and v_id != item["item_id"]
        and len(all_option_sets[tuple(sorted(v["options"].items()))]) == 1
    ]
    if not available_variants:
        seen_opts = set()
        for v_id in list(variants.keys()):
            key = tuple(sorted(variants[v_id]["options"].items()))
            if key in seen_opts:
                del variants[v_id]
            else:
                seen_opts.add(key)
        available_variants = [
            v for v_id, v in variants.items()
            if v["available"] and v_id != item["item_id"]
        ]
        if not available_variants:
            for v_id, v in variants.items():
                if v_id != item["item_id"]:
                    v["available"] = True
                    available_variants = [v]
                    break

    target = rng.choice(available_variants)
    target_item_id = target["item_id"]
    opt_desc = ", ".join(f"{k}: {v}" for k, v in target["options"].items())

    pm_id = list(user["payment_methods"].keys())[0]

    # Describe the item indirectly — by name + current options, NOT order_id/item_id.
    # Agent must look up orders to find which one has this item.
    current_opts = ", ".join(f"{v}" for v in item.get("options", {}).values())
    question = (
        f"I have a {item['name']} ({current_opts}) that was delivered. "
        f"I'd like to exchange it for the variant with {opt_desc}. "
        f"Can you find the order and process the exchange?"
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. {question}"
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm when the exchange has been processed.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    expected_tool = {
        "name": "exchange_delivered_order_items",
        "key_args": {
            "order_id": order["order_id"],
            "item_ids": [item["item_id"]],
            "new_item_ids": [target_item_id],
            "payment_method_id": pm_id,
        },
    }

    return SDRScenario(
        domain="retail",
        scenario_type="execute_exchange",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"target_item_id": target_item_id, "order_id": order["order_id"]},
        communicate_info=[target_item_id],
        description=f"Execute exchange: {item['name']} -> {opt_desc}",
        key_facts={"order_id": order["order_id"], "target_item_id": target_item_id},
        db=db,
        expected_tool_call=expected_tool,
    )


def _gen_execute_modify(rng: random.Random) -> SDRScenario:
    """User wants to modify an item in a pending order to a different variant.

    Agent must find user, look up multiple orders to find the pending one with
    this item, get product details, find the right variant, and call
    modify_pending_order_items.
    Mirrors eval: 24% of retail SD failures. Eval has mean 4+ lookups.
    """
    sample = _gen_retail_sample(rng, n_orders=rng.randint(3, 5), min_items=2,
                                many_variants=True)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    # Need a pending order
    pending = [o for o in orders if o["status"] == "pending"]
    if not pending:
        orders[0]["status"] = "pending"
        pending = [orders[0]]

    order = rng.choice(pending)
    item = rng.choice(order["items"])
    prod_id = item["product_id"]
    prod_entry = products_db[prod_id]
    variants = prod_entry["variants"]

    available_variants = [
        v for v_id, v in variants.items()
        if v["available"] and v_id != item["item_id"]
    ]
    if not available_variants:
        for v_id, v in variants.items():
            if v_id != item["item_id"]:
                v["available"] = True
                available_variants = [v]
                break

    target = rng.choice(available_variants)
    target_item_id = target["item_id"]
    opt_desc = ", ".join(f"{k}: {v}" for k, v in target["options"].items())

    pm_id = list(user["payment_methods"].keys())[0]

    # Indirect reference — describe item by name, not order_id
    current_opts = ", ".join(f"{v}" for v in item.get("options", {}).values())
    question = (
        f"I have a pending order with a {item['name']} ({current_opts}). "
        f"I'd like to change it to the variant with {opt_desc}. "
        f"Can you find the order and update it?"
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. {question}"
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm when the modification has been made.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    expected_tool = {
        "name": "modify_pending_order_items",
        "key_args": {
            "order_id": order["order_id"],
            "item_ids": [item["item_id"]],
            "new_item_ids": [target_item_id],
            "payment_method_id": pm_id,
        },
    }

    return SDRScenario(
        domain="retail",
        scenario_type="execute_modify",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"target_item_id": target_item_id, "order_id": order["order_id"]},
        communicate_info=[target_item_id],
        description=f"Execute modify: {item['name']} -> {opt_desc}",
        key_facts={"order_id": order["order_id"], "target_item_id": target_item_id},
        db=db,
        expected_tool_call=expected_tool,
    )


def _gen_execute_return(rng: random.Random) -> SDRScenario:
    """User wants to return an item from a delivered order.

    Agent must find user, look up multiple orders to find the delivered one,
    identify the correct item, and call return_delivered_order_items.
    Mirrors eval: 18% of retail SD failures. Eval has mean 4+ lookups.
    """
    sample = _gen_retail_sample(rng, n_orders=rng.randint(3, 5), min_items=3,
                                need_delivered=True)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    delivered = [o for o in orders if o["status"] == "delivered"]
    if not delivered:
        orders[0]["status"] = "delivered"
        orders[0]["fulfillments"] = [{
            "tracking_id": [str(rng.randint(100000000000, 999999999999))],
            "item_ids": [it["item_id"] for it in orders[0]["items"]],
        }]
        delivered = [orders[0]]

    order = rng.choice(delivered)
    item = rng.choice(order["items"])
    pm_id = list(user["payment_methods"].keys())[0]

    # Indirect reference — describe item by name and options, not order_id
    current_opts = ", ".join(f"{v}" for v in item.get("options", {}).values())
    question = (
        f"I received a {item['name']} ({current_opts}) recently and I'd like to "
        f"return it. Can you find the order and process the return? "
        f"Please refund to my payment method on file."
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. {question}"
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm when the return has been processed.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    expected_tool = {
        "name": "return_delivered_order_items",
        "key_args": {
            "order_id": order["order_id"],
            "item_ids": [item["item_id"]],
            "payment_method_id": pm_id,
        },
    }

    return SDRScenario(
        domain="retail",
        scenario_type="execute_return",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"item_id": item["item_id"], "order_id": order["order_id"]},
        communicate_info=[order["order_id"]],
        description=f"Execute return: {item['name']} from {order['order_id']}",
        key_facts={"order_id": order["order_id"], "item_id": item["item_id"]},
        db=db,
        expected_tool_call=expected_tool,
    )


def _gen_execute_cancel(rng: random.Random) -> SDRScenario:
    """User wants to cancel a pending order.

    Agent must find user, look up multiple orders to identify the pending one,
    and call cancel_pending_order.
    """
    sample = _gen_retail_sample(rng, n_orders=rng.randint(3, 5), min_items=2)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    pending = [o for o in orders if o["status"] == "pending"]
    if not pending:
        orders[0]["status"] = "pending"
        pending = [orders[0]]

    order = rng.choice(pending)
    reason = rng.choice(["no longer needed", "ordered by mistake"])

    # Describe by item contents, not order_id (forces lookup)
    item_names = [it["name"] for it in order["items"][:2]]
    items_desc = " and ".join(item_names)
    question = (
        f"I have a pending order with {items_desc} that I need to cancel. "
        f"The reason is: {reason}. Can you find it and cancel it?"
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. {question}"
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm when the order has been cancelled.",
    )

    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    expected_tool = {
        "name": "cancel_pending_order",
        "key_args": {
            "order_id": order["order_id"],
            "reason": reason,
        },
    }

    return SDRScenario(
        domain="retail",
        scenario_type="execute_cancel",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"order_id": order["order_id"]},
        communicate_info=[order["order_id"]],
        description=f"Execute cancel: {order['order_id']}",
        key_facts={"order_id": order["order_id"], "reason": reason},
        db=db,
        expected_tool_call=expected_tool,
    )


# =====================================================================
# Gap 3: Conditional exchange (retail)
# =====================================================================

def _gen_conditional_exchange(rng: random.Random) -> SDRScenario:
    """User wants to exchange item to a specific variant; if unavailable, fallback variant.

    Primary target variant has available=false, fallback IS available.
    Expected tool call uses the FALLBACK variant's item_id.
    """
    sample = _gen_retail_sample(rng, n_orders=rng.randint(3, 5), min_items=2,
                                need_delivered=True, many_variants=True)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    delivered = [o for o in orders if o["status"] == "delivered"]
    if not delivered:
        orders[0]["status"] = "delivered"
        orders[0]["fulfillments"] = [{
            "tracking_id": [str(rng.randint(100000000000, 999999999999))],
            "item_ids": [it["item_id"] for it in orders[0]["items"]],
        }]
        delivered = [orders[0]]

    order = rng.choice(delivered)
    item = rng.choice(order["items"])
    prod_id = item["product_id"]
    prod_entry = products_db[prod_id]
    variants = prod_entry["variants"]

    # We need at least 2 variants different from current item
    other_variants = [
        (v_id, v) for v_id, v in variants.items()
        if v_id != item["item_id"]
    ]
    if len(other_variants) < 2:
        # Add more variants
        from adversarial_policy_game.synthetic_db import PRODUCT_CATALOG, _gen_item_id
        tmpl = None
        for p in PRODUCT_CATALOG:
            if p["name"] == prod_entry["name"]:
                tmpl = p
                break
        if tmpl:
            while len(other_variants) < 3:
                v_id = _gen_item_id(rng)
                v_opts = {}
                for opt_name, opt_values in tmpl["options_template"].items():
                    v_opts[opt_name] = rng.choice(opt_values)
                v_price = round(rng.uniform(15, 350), 2)
                variants[v_id] = {
                    "item_id": v_id,
                    "options": v_opts,
                    "available": True,
                    "price": v_price,
                }
                other_variants.append((v_id, variants[v_id]))

    # Pick primary target (make unavailable) and fallback (make available)
    # Ensure primary and fallback have DIFFERENT options (otherwise scenario is nonsensical)
    rng.shuffle(other_variants)
    primary_vid, primary_v = other_variants[0]
    fallback_vid, fallback_v = None, None
    for vid, v in other_variants[1:]:
        if v.get("options") != primary_v.get("options"):
            fallback_vid, fallback_v = vid, v
            break
    if fallback_v is None:
        # All variants have same options — can't make a valid conditional scenario
        return None

    # 30% of the time, primary IS available (agent should use primary, not fallback).
    # 70% of the time, primary is unavailable (agent must fall back).
    # This prevents the "always pick fallback" shortcut.
    primary_available = rng.random() < 0.3
    if primary_available:
        primary_v["available"] = True
        fallback_v["available"] = True  # both available, but primary is preferred
    else:
        primary_v["available"] = False
        # Also mark any other variant with identical options as unavailable
        primary_opts_dict = primary_v["options"]
        for v_id, v in variants.items():
            if v_id == primary_vid or v_id == fallback_vid or v_id == item["item_id"]:
                continue
            if v.get("options") == primary_opts_dict:
                v["available"] = False
        fallback_v["available"] = True

    primary_opts = ", ".join(f"{k}: {v}" for k, v in primary_v["options"].items())
    fallback_opts = ", ".join(f"{k}: {v}" for k, v in fallback_v["options"].items())

    pm_id = list(user["payment_methods"].keys())[0]

    # Expected tool call uses PRIMARY when available, FALLBACK when not
    target_vid = primary_vid if primary_available else fallback_vid

    question = (
        f"I want to exchange my {item['name']} from order {order['order_id']} "
        f"to the variant with {primary_opts}. If that's not available, "
        f"I'd be fine with {fallback_opts}."
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. "
            f"You want to exchange your {item['name']}. "
            f"Primary choice: {primary_opts} (but it may be unavailable). "
            f"Fallback choice: {fallback_opts}."
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm when the exchange has been processed.",
    )
    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    expected_tool = {
        "name": "exchange_delivered_order_items",
        "key_args": {
            "order_id": order["order_id"],
            "item_ids": [item["item_id"]],
            "new_item_ids": [target_vid],
            "payment_method_id": pm_id,
        },
    }

    target_label = "primary" if primary_available else "fallback"
    target_opts = primary_opts if primary_available else fallback_opts

    return SDRScenario(
        domain="retail",
        scenario_type="conditional_exchange",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"target_item_id": target_vid, "order_id": order["order_id"],
                         "primary_unavailable": not primary_available},
        communicate_info=[target_vid],
        description=f"Conditional exchange: {item['name']} -> {target_label} ({target_opts})",
        key_facts={"order_id": order["order_id"], "primary_vid": primary_vid,
                   "fallback_vid": fallback_vid, "primary_available": primary_available},
        db=db,
        expected_tool_call=expected_tool,
    )


# =====================================================================
# Gap 6: Cross-entity exchange (retail)
# =====================================================================

def _gen_cross_entity_exchange(rng: random.Random) -> SDRScenario:
    """User has 2+ orders. Wants to exchange an item in order A to match
    an attribute from an item in order B.

    E.g., "exchange my blue t-shirt to the same color as the jacket in my other order"
    (jacket is red -> exchange to red t-shirt).
    """
    sample = _gen_retail_sample(rng, n_orders=rng.randint(3, 5), min_items=2,
                                need_delivered=True, many_variants=True)
    user = sample["user"]
    orders = sample["orders"]
    products_db = sample["products_db"]
    name = user["name"]
    zipcode = user["address"]["zip"]

    delivered = [o for o in orders if o["status"] == "delivered"]
    if not delivered:
        orders[0]["status"] = "delivered"
        orders[0]["fulfillments"] = [{
            "tracking_id": [str(rng.randint(100000000000, 999999999999))],
            "item_ids": [it["item_id"] for it in orders[0]["items"]],
        }]
        delivered = [orders[0]]

    # Pick the source order (order A) - must be delivered for exchange
    order_a = rng.choice(delivered)
    item_a = rng.choice(order_a["items"])

    # Pick a different order (order B) with an item that has a shared option key
    other_orders = [o for o in orders if o["order_id"] != order_a["order_id"]]
    if not other_orders:
        other_orders = [o for o in orders if o != order_a]
        if not other_orders:
            other_orders = [orders[-1]]

    order_b = rng.choice(other_orders)
    item_b = rng.choice(order_b["items"])

    # Find a shared option key between the two items
    opts_a = item_a.get("options", {})
    opts_b = item_b.get("options", {})
    shared_keys = set(opts_a.keys()) & set(opts_b.keys())

    if not shared_keys:
        if "color" in opts_b:
            shared_keys = {"color"}
        elif opts_b:
            shared_keys = {list(opts_b.keys())[0]}

    if not shared_keys:
        shared_keys = set(opts_a.keys()) if opts_a else {"color"}

    ref_key = rng.choice(list(shared_keys))
    ref_value = opts_b.get(ref_key, list(opts_a.values())[0] if opts_a else "blue")

    # Find a variant of item_a's product that matches the reference attribute
    prod_id_a = item_a["product_id"]
    prod_entry_a = products_db[prod_id_a]
    variants_a = prod_entry_a["variants"]

    matching_variants = [
        (v_id, v) for v_id, v in variants_a.items()
        if v["available"] and v_id != item_a["item_id"]
        and v["options"].get(ref_key) == ref_value
    ]

    if not matching_variants:
        # Force a variant to match
        for v_id, v in variants_a.items():
            if v_id != item_a["item_id"]:
                v["options"][ref_key] = ref_value
                v["available"] = True
                matching_variants = [(v_id, v)]
                break

    if not matching_variants:
        # Create one
        from adversarial_policy_game.synthetic_db import _gen_item_id
        v_id = _gen_item_id(rng)
        new_opts = dict(opts_a)
        new_opts[ref_key] = ref_value
        v_price = round(rng.uniform(15, 350), 2)
        variants_a[v_id] = {
            "item_id": v_id,
            "options": new_opts,
            "available": True,
            "price": v_price,
        }
        matching_variants = [(v_id, variants_a[v_id])]

    target_vid, target_v = rng.choice(matching_variants)

    pm_id = list(user["payment_methods"].keys())[0]

    question = (
        f"I'd like to exchange my {item_a['name']} from order {order_a['order_id']} "
        f"to the same {ref_key} as the {item_b['name']} in my order {order_b['order_id']}. "
        f"Can you find the right variant and process the exchange?"
    )

    initial_msg = (
        f"Hi, my name is {name['first_name']} {name['last_name']}. "
        f"My zip code is {zipcode}. {question}"
    )

    # Gap 1: adversarial approach for mutation scenarios
    approach = "adversarial" if rng.random() < 0.3 else "cooperative"
    user_sys = build_user_system_prompt(
        customer_context=(
            f"Your name is {name['first_name']} {name['last_name']}. "
            f"Your zip code is {zipcode}. "
            f"You want to exchange {item_a['name']} from order {order_a['order_id']} "
            f"to match the {ref_key} of {item_b['name']} in order {order_b['order_id']}. "
            f"The {item_b['name']}'s {ref_key} is {ref_value}."
        ),
        goal=question,
        approach=approach,
        required_communication="Confirm when the exchange has been processed.",
    )
    user_sys = _maybe_add_progressive_disclosure(rng, user_sys)

    db = build_retail_db(user, orders, products_db)

    expected_tool = {
        "name": "exchange_delivered_order_items",
        "key_args": {
            "order_id": order_a["order_id"],
            "item_ids": [item_a["item_id"]],
            "new_item_ids": [target_vid],
            "payment_method_id": pm_id,
        },
    }

    return SDRScenario(
        domain="retail",
        scenario_type="cross_entity_exchange",
        user_system_prompt=user_sys,
        initial_message=initial_msg,
        expected_answer={"target_item_id": target_vid, "order_a": order_a["order_id"],
                         "order_b": order_b["order_id"], "ref_key": ref_key,
                         "ref_value": ref_value},
        communicate_info=[target_vid],
        description=f"Cross-entity exchange: {item_a['name']} -> match {ref_key}={ref_value} from {item_b['name']}",
        key_facts={"order_a": order_a["order_id"], "order_b": order_b["order_id"],
                   "target_vid": target_vid, "ref_key": ref_key, "ref_value": ref_value},
        db=db,
        expected_tool_call=expected_tool,
    )


# =====================================================================
# Master scenario generator
# =====================================================================

_GENERATORS = {
    "flight_selection": _gen_flight_selection,
    "baggage_computation": _gen_baggage_computation,
    "reservation_comparison": _gen_reservation_comparison,
    "cost_computation": _gen_cost_computation,
    "flight_status_check": _gen_flight_status_check,
    "book_flight": _gen_book_flight,
    "change_flight": _gen_change_flight,
    "send_compensation": _gen_send_compensation,
    "conditional_flight_change": _gen_conditional_flight_change,
}

_RETAIL_GENERATORS = {
    "variant_selection": _gen_variant_selection,
    "price_comparison": _gen_price_comparison,
    "order_status_check": _gen_order_status_check,
    "execute_exchange": _gen_execute_exchange,
    "execute_modify": _gen_execute_modify,
    "execute_return": _gen_execute_return,
    "execute_cancel": _gen_execute_cancel,
    "conditional_exchange": _gen_conditional_exchange,
    "cross_entity_exchange": _gen_cross_entity_exchange,
}


def generate_scenario(seed: int, domain: Optional[str] = None) -> SDRScenario:
    """Generate a scenario from a seed. Deterministic.

    Args:
        seed: Random seed for deterministic generation.
        domain: "airline", "retail", or None (random 50/50).
    """
    rng = random.Random(seed)

    if domain is None:
        domain = rng.choice(["airline", "retail"])

    if domain == "retail":
        types = list(RETAIL_SCENARIO_WEIGHTS.keys())
        weights = list(RETAIL_SCENARIO_WEIGHTS.values())
        scenario_type = rng.choices(types, weights=weights)[0]
        return _RETAIL_GENERATORS[scenario_type](rng)
    else:
        types = list(SCENARIO_WEIGHTS.keys())
        weights = list(SCENARIO_WEIGHTS.values())
        scenario_type = rng.choices(types, weights=weights)[0]
        return _GENERATORS[scenario_type](rng)


# =====================================================================
# Reward computation
# =====================================================================

def _check_tool_call_match(conversation: List[Dict], expected: Dict) -> Tuple[bool, str]:
    """Check if the expected tool call was made in the conversation."""
    expected_name = expected["name"]
    key_args = expected.get("key_args", {})

    for msg in conversation:
        if msg.get("role") != "tool_call":
            continue
        try:
            tc = json.loads(msg["text"])
        except (json.JSONDecodeError, KeyError):
            continue

        if tc.get("name") != expected_name:
            continue

        args = tc.get("arguments", {})
        # Check key args match
        all_match = True
        for k, expected_val in key_args.items():
            actual_val = args.get(k)
            if actual_val is None:
                all_match = False
                break
            # Numeric comparison
            if isinstance(expected_val, (int, float)):
                try:
                    if abs(float(actual_val) - float(expected_val)) > 1:
                        all_match = False
                        break
                except (ValueError, TypeError):
                    all_match = False
                    break
            # List comparison (e.g. item_ids)
            elif isinstance(expected_val, list):
                if not isinstance(actual_val, list):
                    all_match = False
                    break
                if set(str(x) for x in expected_val) != set(str(x) for x in actual_val):
                    all_match = False
                    break
            # String comparison
            elif str(actual_val).strip() != str(expected_val).strip():
                all_match = False
                break

        if all_match:
            return True, f"Tool call matched: {expected_name}"

    return False, f"Expected {expected_name} not found or args mismatch"


def _check_tool_name_called(conversation: List[Dict], expected_name: str) -> bool:
    """Check if a tool with the given name was called at all in the conversation."""
    for msg in conversation:
        if msg.get("role") != "tool_call":
            continue
        try:
            tc = json.loads(msg["text"])
        except (json.JSONDecodeError, KeyError):
            continue
        if tc.get("name") == expected_name:
            return True
    return False


def _check_communicate(conversation: List[Dict], communicate_info: List[str]) -> Tuple[bool, str]:
    """Check if communicate_info strings appear in agent messages."""
    agent_text = ""
    for msg in conversation:
        if msg.get("role") == "assistant":
            agent_text += " " + (msg.get("text") or msg.get("content") or "")

    agent_text_lower = agent_text.lower()

    found = []
    missing = []
    for info in communicate_info:
        info_lower = info.lower().strip()
        if info_lower in agent_text_lower:
            found.append(info)
        else:
            try:
                num = float(info)
                agent_no_commas = agent_text.replace(",", "")
                if (str(int(num)) in agent_no_commas or
                    f"{num:.0f}" in agent_no_commas or
                    f"${int(num)}" in agent_no_commas or
                    f"${num:.2f}" in agent_no_commas or
                    f"{num:.2f}" in agent_no_commas or
                    info in agent_no_commas):
                    found.append(info)
                else:
                    missing.append(info)
            except ValueError:
                missing.append(info)

    if communicate_info:
        passed = not missing
        reason = f"Communicated: {found}" if passed else f"Missing: {missing}"
    else:
        passed = True
        reason = "No communicate_info required"

    return passed, reason


def compute_reward(conversation: List[Dict], scenario: SDRScenario,
                   tool_executor: Optional[ToolExecutor] = None) -> Tuple[float, str]:
    """Check if the agent communicated the correct answer AND made required tool calls.

    For mutation scenarios (expected_tool_call set):
      - Compute gold DB hash by replaying expected tool call on fresh DB copy
      - Compare with agent's actual DB hash
      - Reward: 1.0 if DB match + communicate pass, 0.3 if DB match only,
        0.1 if correct tool name but DB mismatch, 0.0 otherwise
      - Includes communicate_info check
    For report scenarios: checks communicate_info strings in agent messages.
    Returns (reward, reason).
    """
    # --- Check communicate_info ---
    comm_pass, comm_reason = _check_communicate(conversation, scenario.communicate_info)

    # --- Mutation scenario with DB hash verification ---
    if scenario.expected_tool_call is not None and tool_executor is not None:
        expected_name = scenario.expected_tool_call["name"]
        key_args = scenario.expected_tool_call.get("key_args", {})

        # DB hash verification: compare agent's DB against initial DB to confirm
        # mutation happened, then validate via key-args matching.
        # If gold_tool_args are available, replay them for gold comparison.
        initial_hash = get_dict_hash(copy.deepcopy(scenario.db))
        agent_hash = get_dict_hash(tool_executor.db)
        db_changed = (initial_hash != agent_hash)

        # If we have full gold args, do proper gold replay comparison
        if scenario.gold_tool_args:
            gold_executor = ToolExecutor(
                scenario.domain, copy.deepcopy(scenario.db))
            try:
                gold_executor.execute(expected_name, scenario.gold_tool_args)
            except Exception:
                pass
            gold_hash = get_dict_hash(gold_executor.db)
            db_match = (gold_hash == agent_hash)
        else:
            # No full args: use key-args match + DB-changed as proxy
            tool_match, _ = _check_tool_call_match(
                conversation, scenario.expected_tool_call)
            db_match = tool_match and db_changed

        tool_name_called = _check_tool_name_called(conversation, expected_name)

        if db_match and comm_pass:
            return 1.0, f"DB verified + communicate pass. {comm_reason}"
        elif db_match:
            return 0.3, f"DB verified but communicate fail. {comm_reason}"
        elif tool_name_called and db_changed:
            return 0.1, f"Correct tool ({expected_name}), DB changed but args mismatch. {comm_reason}"
        elif tool_name_called:
            return 0.0, f"Tool {expected_name} called but DB unchanged (likely error). {comm_reason}"
        else:
            return 0.0, f"Tool {expected_name} not called. {comm_reason}"

    # --- Mutation scenario without tool_executor (fallback to key-args matching) ---
    if scenario.expected_tool_call is not None:
        tool_match, tool_reason = _check_tool_call_match(
            conversation, scenario.expected_tool_call)
        if tool_match and comm_pass:
            return 1.0, f"{tool_reason}. {comm_reason}"
        elif tool_match:
            return 0.3, f"{tool_reason} but communicate fail. {comm_reason}"
        elif _check_tool_name_called(conversation, scenario.expected_tool_call["name"]):
            return 0.1, f"Correct tool name but args mismatch. {comm_reason}"
        else:
            return 0.0, f"Expected {scenario.expected_tool_call['name']} not found. {comm_reason}"

    # --- Report scenario: communicate_info only ---
    reward = 1.0 if comm_pass else 0.0
    return reward, comm_reason


# =====================================================================
# Game class
# =====================================================================

class StructuredDataGame:
    """Multi-turn structured data reasoning game in tau2-bench format.

    Supports both airline and retail domains.

    Implements the tool-calling game interface (supports_structured_messages):
    - get_system_prompt() / get_messages() / get_tool_schemas() / step()

    Also implements GameEnv protocol for GRPO training compatibility.
    """

    supports_structured_messages = True

    def __init__(self, user_client: Optional[UserLLMClient] = None,
                 domain: Optional[str] = None):
        self._user_client = user_client
        self._domain_filter = domain  # None=random 50/50, "airline", "retail"

        # GameEnv protocol
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        # Internal state
        self._scenario: Optional[SDRScenario] = None
        self._tools: Optional[ToolExecutor] = None
        self._llm_user: Optional[LLMUser] = None
        self._conversation: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._transferred: bool = False
        self._pending_stop: bool = False
        self._last_call_key: Optional[str] = None
        self._repeat_count: int = 0
        self.max_steps: int = 30

    def reset(self, seed: int) -> None:
        """Reset with new seed."""
        self._scenario = generate_scenario(seed, domain=self._domain_filter)
        self._tools = ToolExecutor(
            self._scenario.domain,
            copy.deepcopy(self._scenario.db),
        )

        if self._user_client is None:
            raise ValueError(
                "StructuredDataGame requires a UserLLMClient. "
                "Pass user_client= when constructing the environment."
            )

        self._llm_user = LLMUser(
            self._scenario.user_system_prompt,
            self._scenario.initial_message,
            self._user_client,
        )
        initial_msg = self._llm_user.get_initial_message()

        self._conversation = [{"role": "user", "text": initial_msg}]
        self._step_count = 0
        self._transferred = False
        self._pending_stop = False
        self._last_call_key = None
        self._repeat_count = 0

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    # -----------------------------------------------------------------
    # Structured message interface (tool-calling games)
    # -----------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Return full system prompt with domain-appropriate policy."""
        domain = self._scenario.domain if self._scenario else "airline"
        policy = AIRLINE_POLICY if domain == "airline" else RETAIL_POLICY
        return (
            "<instructions>\n"
            "You are a customer service agent that helps the user according to "
            "the <policy> provided below.\n"
            "In each turn you can either:\n"
            "- Send a message to the user.\n"
            "- Make a tool call.\n"
            "You cannot do both at the same time.\n"
            "\n"
            "Try to be helpful and always follow the policy. "
            "Always make sure you generate valid JSON only.\n"
            "</instructions>\n"
            "<policy>\n"
            f"{policy}\n"
            "</policy>"
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool schemas for the current domain."""
        if self._scenario is not None and self._scenario.domain == "retail":
            return RETAIL_TOOL_SCHEMAS
        return AIRLINE_TOOL_SCHEMAS

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return conversation as chat API format messages."""
        msgs = []
        for msg in self._conversation:
            role = msg["role"]
            text = msg["text"]
            if role == "user":
                msgs.append({"role": "user", "content": text})
            elif role == "assistant":
                msgs.append({"role": "assistant", "content": text})
            elif role == "tool_call":
                try:
                    tc = json.loads(text)
                    msgs.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": f"call_{len(msgs)}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }],
                    })
                except (json.JSONDecodeError, KeyError):
                    msgs.append({"role": "assistant", "content": text})
            elif role == "tool_result":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": f"call_{len(msgs) - 1}",
                    "content": text,
                })
        return msgs

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
        """Process one agent action (tool call or message)."""
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

        # --- Transfer to human ---
        if tool_name == "transfer_to_human_agents":
            self._transferred = True
            self._conversation.append({"role": "tool_call", "text": json.dumps(tool_call)})
            self._conversation.append({"role": "tool_result", "text": '{"transfer": "success"}'})
            reward, reason = compute_reward(self._conversation, self._scenario, tool_executor=self._tools)
            self._finalize(reward, f"Transferred. {reason}")
            return

        # --- Message to user ---
        if tool_name == "respond_to_user" or tool_name == "send_message":
            message = tool_args.get("message", tool_args.get("content", ""))
            self._conversation.append({"role": "assistant", "text": message})

            # Check if user simulator signals stop
            if self._pending_stop:
                reward, reason = compute_reward(self._conversation, self._scenario, tool_executor=self._tools)
                self._finalize(reward, reason)
                return

            # Get user response
            visible = self._get_visible_conversation()
            user_response = self._llm_user.get_response(visible)

            if user_response is None:
                reward, reason = compute_reward(self._conversation, self._scenario, tool_executor=self._tools)
                self._finalize(reward, reason)
                return
            elif "###STOP###" in user_response or "###TRANSFER###" in user_response:
                clean = user_response.replace("###STOP###", "").replace("###TRANSFER###", "").strip()
                if clean:
                    self._conversation.append({"role": "user", "text": clean})
                reward, reason = compute_reward(self._conversation, self._scenario, tool_executor=self._tools)
                self._finalize(reward, reason)
                return

            self._conversation.append({"role": "user", "text": user_response})

        # --- Tool call ---
        else:
            self._conversation.append({"role": "tool_call", "text": json.dumps(tool_call)})

            # Loop detection
            call_key = json.dumps(tool_call, sort_keys=True)
            if call_key == self._last_call_key:
                self._repeat_count += 1
                if self._repeat_count >= 3:
                    reward, reason = compute_reward(self._conversation, self._scenario, tool_executor=self._tools)
                    self._finalize(reward, f"Loop detected. {reason}")
                    return
            else:
                self._last_call_key = call_key
                self._repeat_count = 0

            # Execute tool
            try:
                result = self._tools.execute(tool_name, tool_args)
            except Exception as e:
                result = json.dumps({"error": str(e)})

            self._conversation.append({"role": "tool_result", "text": result})

        # Max steps check
        if self._step_count >= self.max_steps:
            reward, reason = compute_reward(self._conversation, self._scenario, tool_executor=self._tools)
            self._finalize(reward, f"Max steps. {reason}")

    def _get_visible_conversation(self) -> List[Dict[str, str]]:
        """Get conversation visible to the customer (text only, no tool calls)."""
        return [
            msg for msg in self._conversation
            if msg["role"] in ("user", "assistant")
        ]

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    def get_summary(self) -> Dict[str, Any]:
        return {
            "scenario_type": self._scenario.scenario_type if self._scenario else "",
            "description": self._scenario.description if self._scenario else "",
            "steps": self._step_count,
            "reward": self.rewards.get(0, 0.0),
            "reason": getattr(self, "_reason", ""),
            "expected_answer": self._scenario.expected_answer if self._scenario else None,
        }


# =====================================================================
# Action parsing
# =====================================================================

def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON tool call from model output."""
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
    """Extract action for game registry compatibility."""
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
    print("Testing Structured Data Reasoning Game (tau2-bench aligned)")
    print("=" * 70)

    for domain_label in ["airline", "retail", None]:
        print(f"\n{'=' * 70}")
        print(f"Domain filter: {domain_label}")
        print("=" * 70)

        type_counts = {}
        for seed in range(100):
            scenario = generate_scenario(seed, domain=domain_label)
            t = f"{scenario.domain}/{scenario.scenario_type}"
            type_counts[t] = type_counts.get(t, 0) + 1
            if seed < 3:
                print(f"\nSeed {seed}: {scenario.description}")
                print(f"  Domain: {scenario.domain}")
                print(f"  Initial msg: {scenario.initial_message[:120]}...")
                print(f"  Expected: {scenario.expected_answer}")
                print(f"  Communicate: {scenario.communicate_info}")
                print(f"  DB keys: {list(scenario.db.keys())}")

        print(f"\nScenario type distribution (100 seeds): {type_counts}")
