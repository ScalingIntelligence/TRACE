#!/usr/bin/env python3
"""
Generate synthetic customer service tasks for SkillRL cold-start SFT.

Creates novel scenarios using the SAME DB as tau-bench but with DIFFERENT tasks.
These tasks exercise the same skills (policy adherence, tool calling, multi-step ops)
but are NOT copies of tau-bench evaluation tasks.

The generated tasks are then run as simulations via vLLM to produce SFT training data.

Usage:
    # Step 1: Generate synthetic tasks
    python skillrl/generate_synthetic_tasks.py --generate-tasks \
        --output skillrl/data/synthetic_tasks.json

    # Step 2: Run simulations on these tasks to collect SFT data
    python skillrl/generate_synthetic_tasks.py --run-sims \
        --tasks skillrl/data/synthetic_tasks.json \
        --agent-url http://localhost:8080/v1 \
        --user-url http://localhost:9000/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --output skillrl/data/skillrl_sft_train.jsonl
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GAMES_DIR = SCRIPT_DIR.parent
DATA_DIR = GAMES_DIR / "evals" / "benchmarks" / "tau2_bench_eval" / "data" / "tau2" / "domains"


def load_db(domain: str) -> dict:
    with open(DATA_DIR / domain / "db.json") as f:
        return json.load(f)


def load_existing_task_ids(domain: str) -> set:
    """Load existing tau-bench task descriptions to avoid duplicates."""
    with open(DATA_DIR / domain / "tasks.json") as f:
        tasks = json.load(f)
    # Extract key phrases to avoid
    phrases = set()
    for t in tasks:
        scenario = t.get("user_scenario", {}).get("instructions", {})
        reason = scenario.get("reason_for_call", "")
        phrases.add(reason.strip().lower()[:100])
    return phrases


# =====================================================================
# AIRLINE TASK GENERATORS
# =====================================================================

def gen_airline_booking_tasks(db: dict, rng: random.Random, count: int = 30) -> list:
    """Generate booking-related tasks."""
    tasks = []
    users = list(db["users"].values())
    flights = list(db["flights"].values())

    airports = list(set(f["origin"] for f in flights) | set(f["destination"] for f in flights))

    scenarios = [
        # Basic booking
        lambda u, f: {
            "category": "booking",
            "reason": f"You want to book a one-way economy flight from {rng.choice(airports)} to {rng.choice(airports)} for next week.",
            "instructions": "Provide your details when asked. You prefer the cheapest option.",
            "skills_tested": ["AB4", "AG6"],
        },
        # Booking with too many passengers
        lambda u, f: {
            "category": "booking",
            "reason": f"You want to book a flight for your family of 7 people from {rng.choice(airports)} to {rng.choice(airports)}.",
            "instructions": "You have 7 family members who need to travel. Insist that you need all 7 on one reservation.",
            "skills_tested": ["AB1"],
        },
        # Booking with mixed cabin request
        lambda u, f: {
            "category": "booking",
            "reason": f"You want to book a round-trip flight from {rng.choice(airports)} to {rng.choice(airports)} for 2 passengers.",
            "instructions": "You want business class for yourself and economy for your assistant. Insist on different cabins for each passenger.",
            "skills_tested": ["AB2"],
        },
        # Booking with excessive payment methods
        lambda u, f: {
            "category": "booking",
            "reason": f"You want to book a flight from {rng.choice(airports)} to {rng.choice(airports)}.",
            "instructions": "You want to pay with 2 credit cards and 4 gift cards. Be insistent about splitting payment this way.",
            "skills_tested": ["AB3"],
        },
        # Booking with baggage questions
        lambda u, f: {
            "category": "booking",
            "reason": f"You want to book a flight and need to bring 3 checked bags.",
            "instructions": "Ask about baggage allowance and costs before booking. You want to know exactly how much extra bags will cost.",
            "skills_tested": ["AB5"],
        },
    ]

    for i in range(count):
        user = rng.choice(users)
        flight = rng.choice(flights)
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(user, flight)

        task = {
            "id": f"syn_airline_{i}",
            "domain": "airline",
            "category": scenario["category"],
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "airline",
                    "reason_for_call": scenario["reason"],
                    "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.",
                    "unknown_info": None,
                }
            },
        }
        tasks.append(task)

    return tasks


def gen_airline_modification_tasks(db: dict, rng: random.Random, count: int = 30) -> list:
    """Generate modification-related tasks."""
    tasks = []
    reservations = list(db["reservations"].values())
    users = db["users"]

    scenarios = [
        # Change flights on basic economy (should be refused)
        lambda r, u: {
            "category": "modification",
            "reason": f"You want to change the flights on your reservation {r['reservation_id']}.",
            "instructions": "You want to change to different flight times. If told basic economy can't be modified, express frustration and ask to speak to a supervisor.",
            "skills_tested": ["AM1"],
            "filter": lambda r: r.get("cabin") == "basic_economy",
        },
        # Change destination (should be refused)
        lambda r, u: {
            "category": "modification",
            "reason": f"You want to change the destination on reservation {r['reservation_id']} to a different city.",
            "instructions": "You booked the wrong destination. You want to change where you're flying to. Be persistent.",
            "skills_tested": ["AM2"],
        },
        # Add baggage to existing reservation
        lambda r, u: {
            "category": "modification",
            "reason": f"You want to add extra checked bags to your reservation {r['reservation_id']}.",
            "instructions": "You realized you need 2 more checked bags than what you originally had. Ask about costs.",
            "skills_tested": ["AM4", "AB5"],
        },
        # Add passengers (should be refused)
        lambda r, u: {
            "category": "modification",
            "reason": f"You need to add another passenger to reservation {r['reservation_id']}.",
            "instructions": "Your friend wants to join the trip. You want to add them to your existing reservation rather than making a new one.",
            "skills_tested": ["AM5"],
        },
        # Change trip type (should be refused)
        lambda r, u: {
            "category": "modification",
            "reason": f"You want to change your reservation {r['reservation_id']} from one-way to round-trip.",
            "instructions": "Your plans changed and now you need a return flight too. You want to add it to your existing reservation.",
            "skills_tested": ["AM6"],
            "filter": lambda r: r.get("flight_type") == "one_way",
        },
        # Add travel insurance
        lambda r, u: {
            "category": "modification",
            "reason": f"You want to add travel insurance to reservation {r['reservation_id']}.",
            "instructions": "You forgot to add insurance when booking. Ask how much it costs and then add it.",
            "skills_tested": ["AM4"],
        },
    ]

    for i in range(count):
        res = rng.choice(reservations)
        user = users.get(res["user_id"], rng.choice(list(users.values())))
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(res, user)

        task = {
            "id": f"syn_airline_mod_{i}",
            "domain": "airline",
            "category": scenario["category"],
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "airline",
                    "reason_for_call": scenario["reason"],
                    "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.",
                    "unknown_info": None,
                }
            },
        }
        tasks.append(task)

    return tasks


def gen_airline_cancellation_tasks(db: dict, rng: random.Random, count: int = 25) -> list:
    """Generate cancellation-related tasks."""
    tasks = []
    reservations = list(db["reservations"].values())
    users = db["users"]

    scenarios = [
        # Cancel with no valid reason (should be refused)
        lambda r, u: {
            "reason": f"You want to cancel reservation {r['reservation_id']} because you found a cheaper flight.",
            "instructions": "You found a much cheaper flight elsewhere. Insist on cancellation with full refund. If refused, claim you were told you could cancel anytime.",
            "skills_tested": ["AC1", "AG3"],
        },
        # Cancel claiming insurance covers it (wrong reason)
        lambda r, u: {
            "reason": f"You want to cancel reservation {r['reservation_id']}.",
            "instructions": "Claim you need to cancel due to a work schedule change. You have insurance so it should be covered. If told work isn't covered, argue that it should be.",
            "skills_tested": ["AC1", "AC2"],
        },
        # Cancel business class (should succeed)
        lambda r, u: {
            "reason": f"You want to cancel your business class reservation {r['reservation_id']}.",
            "instructions": "You want a full cancellation and refund. Be straightforward.",
            "skills_tested": ["AC1"],
            "filter": lambda r: r.get("cabin") == "business",
        },
        # Cancel claiming it was within 24 hours
        lambda r, u: {
            "reason": f"You want to cancel reservation {r['reservation_id']}. You just booked it.",
            "instructions": "Claim you booked it just a few hours ago. If the agent says it's been more than 24 hours, express surprise and say the website must have glitched.",
            "skills_tested": ["AC1", "AC3", "AG2"],
        },
    ]

    for i in range(count):
        res = rng.choice(reservations)
        user = users.get(res["user_id"], rng.choice(list(users.values())))
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(res, user)

        task = {
            "id": f"syn_airline_cancel_{i}",
            "domain": "airline",
            "category": "cancellation",
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "airline",
                    "reason_for_call": scenario["reason"],
                    "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.",
                    "unknown_info": None,
                }
            },
        }
        tasks.append(task)

    return tasks


def gen_airline_compensation_tasks(db: dict, rng: random.Random, count: int = 20) -> list:
    """Generate compensation-related tasks."""
    tasks = []
    users = list(db["users"].values())
    flights = list(db["flights"].values())

    scenarios = [
        # Regular member demanding compensation (should be refused)
        lambda u: {
            "reason": "Your flight was delayed by 3 hours and you want compensation.",
            "instructions": f"You are very upset about the delay. Demand compensation. If told you're not eligible, claim you've been a loyal customer for years. Do not accept 'no' easily.",
            "skills_tested": ["AO1", "AG3"],
        },
        # Claiming flight was cancelled when it wasn't
        lambda u: {
            "reason": "Your flight was cancelled and you want a full refund plus compensation.",
            "instructions": "Claim your flight was cancelled even if the agent says otherwise. Say you received an email about it.",
            "skills_tested": ["AO3", "AG2"],
        },
        # Eligible compensation request
        lambda u: {
            "reason": "Your flight was delayed and you'd like to know what options you have.",
            "instructions": "Ask politely about what you can do about the delay. If offered compensation, ask how much and how it works.",
            "skills_tested": ["AO1", "AO2"],
        },
    ]

    for i in range(count):
        user = rng.choice(users)
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(user)

        task = {
            "id": f"syn_airline_comp_{i}",
            "domain": "airline",
            "category": "compensation",
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "airline",
                    "reason_for_call": scenario["reason"],
                    "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']}.\nYour user id is {user['user_id']}.",
                    "unknown_info": None,
                }
            },
        }
        tasks.append(task)

    return tasks


# =====================================================================
# RETAIL TASK GENERATORS
# =====================================================================

def gen_retail_return_tasks(db: dict, rng: random.Random, count: int = 30) -> list:
    """Generate return-related tasks."""
    tasks = []
    orders = list(db["orders"].values())
    users = db["users"]

    scenarios = [
        # Return from delivered order
        lambda o, u: {
            "reason": f"You received order {o['order_id']} and want to return some items.",
            "instructions": "You are unhappy with the quality. You want a refund to your credit card.",
            "skills_tested": ["RR1", "RR2", "RR4"],
            "filter": lambda o: o.get("status") == "delivered",
        },
        # Return from pending order (should use cancel instead)
        lambda o, u: {
            "reason": f"You want to return items from order {o['order_id']}.",
            "instructions": "You changed your mind about some items. You want to send them back.",
            "skills_tested": ["RR1", "RG2"],
            "filter": lambda o: o.get("status") == "pending",
        },
        # Return with gift card refund
        lambda o, u: {
            "reason": f"You want to return an item from order {o['order_id']}.",
            "instructions": "You want the refund as a gift card balance. Ask how long it will take.",
            "skills_tested": ["RR3", "RR4"],
            "filter": lambda o: o.get("status") == "delivered",
        },
    ]

    for i in range(count):
        order = rng.choice(orders)
        user = users.get(order["user_id"], rng.choice(list(users.values())))
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(order, user)

        auth_method = rng.choice(["email", "name_zip"])
        if auth_method == "email":
            known = f"Your email is {user.get('email', 'unknown@example.com')}."
        else:
            known = f"You are {user['name']['first_name']} {user['name']['last_name']} in zipcode {user['address']['zip']}."

        task = {
            "id": f"syn_retail_return_{i}",
            "domain": "retail",
            "category": "returns",
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "retail",
                    "reason_for_call": scenario["reason"],
                    "known_info": known,
                    "unknown_info": "You do not remember your user id." if auth_method != "email" else None,
                }
            },
        }
        tasks.append(task)

    return tasks


def gen_retail_exchange_tasks(db: dict, rng: random.Random, count: int = 25) -> list:
    """Generate exchange-related tasks."""
    tasks = []
    orders = list(db["orders"].values())
    users = db["users"]

    scenarios = [
        # Valid exchange (same product type)
        lambda o, u: {
            "reason": f"You want to exchange an item from order {o['order_id']} for a different variant.",
            "instructions": "You want a different color/size of the same product. Ask what's available first.",
            "skills_tested": ["RE1", "RE3"],
        },
        # Invalid exchange (different product type)
        lambda o, u: {
            "reason": f"You want to exchange an item from order {o['order_id']} for a completely different product.",
            "instructions": "You don't like the item you got and want something completely different instead. Be persistent if told it's not possible.",
            "skills_tested": ["RE1", "RG3"],
        },
        # Exchange on pending order (should modify instead)
        lambda o, u: {
            "reason": f"You want to exchange an item in order {o['order_id']}.",
            "instructions": "The order hasn't arrived yet but you want to change one of the items.",
            "skills_tested": ["RE2", "RG2"],
            "filter": lambda o: o.get("status") == "pending",
        },
    ]

    for i in range(count):
        order = rng.choice(orders)
        user = users.get(order["user_id"], rng.choice(list(users.values())))
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(order, user)

        task = {
            "id": f"syn_retail_exchange_{i}",
            "domain": "retail",
            "category": "exchanges",
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "retail",
                    "reason_for_call": scenario["reason"],
                    "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']} in zipcode {user['address']['zip']}.",
                    "unknown_info": "You do not remember your email.",
                }
            },
        }
        tasks.append(task)

    return tasks


def gen_retail_modification_tasks(db: dict, rng: random.Random, count: int = 25) -> list:
    """Generate order modification tasks."""
    tasks = []
    orders = list(db["orders"].values())
    users = db["users"]

    scenarios = [
        # Modify items (single change)
        lambda o, u: {
            "reason": f"You want to change an item in your pending order {o['order_id']}.",
            "instructions": "You want a different variant of one of your items. Same product, different options.",
            "skills_tested": ["RM1", "RM2", "RM3"],
            "filter": lambda o: o.get("status") == "pending",
        },
        # Modify items twice (should fail second time)
        lambda o, u: {
            "reason": f"You need to make changes to order {o['order_id']}.",
            "instructions": "First change one item, then realize you want to change another item too. Try to make a second modification.",
            "skills_tested": ["RM1", "RM4"],
            "filter": lambda o: o.get("status") == "pending",
        },
        # Change address on pending order
        lambda o, u: {
            "reason": f"You need to change the shipping address on order {o['order_id']}.",
            "instructions": "You moved to a new address and need to update the delivery.",
            "skills_tested": ["RM2"],
            "filter": lambda o: o.get("status") == "pending",
        },
        # Change payment on pending order
        lambda o, u: {
            "reason": f"You want to change the payment method on order {o['order_id']}.",
            "instructions": "You want to switch to a different payment method.",
            "skills_tested": ["RM2"],
            "filter": lambda o: o.get("status") == "pending",
        },
        # Modify delivered order (should be refused)
        lambda o, u: {
            "reason": f"You want to modify order {o['order_id']}.",
            "instructions": "You want to change the items in your order. If told it's delivered, insist they should still be able to modify it.",
            "skills_tested": ["RM2", "RG2"],
            "filter": lambda o: o.get("status") == "delivered",
        },
    ]

    for i in range(count):
        order = rng.choice(orders)
        user = users.get(order["user_id"], rng.choice(list(users.values())))
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(order, user)

        task = {
            "id": f"syn_retail_mod_{i}",
            "domain": "retail",
            "category": "order_modification",
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "retail",
                    "reason_for_call": scenario["reason"],
                    "known_info": f"Your email is {user.get('email', 'unknown@example.com')}.",
                    "unknown_info": None,
                }
            },
        }
        tasks.append(task)

    return tasks


def gen_retail_cancellation_tasks(db: dict, rng: random.Random, count: int = 20) -> list:
    """Generate cancellation tasks."""
    tasks = []
    orders = list(db["orders"].values())
    users = db["users"]

    scenarios = [
        # Cancel pending order (valid)
        lambda o, u: {
            "reason": f"You want to cancel order {o['order_id']}.",
            "instructions": "You no longer need the items. Cancel the whole order.",
            "skills_tested": ["RC1", "RC2"],
            "filter": lambda o: o.get("status") == "pending",
        },
        # Cancel delivered order (should be refused, suggest return)
        lambda o, u: {
            "reason": f"You want to cancel order {o['order_id']}.",
            "instructions": "You want your money back. If told it can't be cancelled, ask what other options you have.",
            "skills_tested": ["RC1", "RG2"],
            "filter": lambda o: o.get("status") == "delivered",
        },
        # Cancel with invalid reason
        lambda o, u: {
            "reason": f"You want to cancel order {o['order_id']} because the delivery is taking too long.",
            "instructions": "You're frustrated with slow delivery. The reason is 'shipping too slow'. If that reason isn't accepted, reluctantly say 'no longer needed'.",
            "skills_tested": ["RC2"],
            "filter": lambda o: o.get("status") == "pending",
        },
    ]

    for i in range(count):
        order = rng.choice(orders)
        user = users.get(order["user_id"], rng.choice(list(users.values())))
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(order, user)

        task = {
            "id": f"syn_retail_cancel_{i}",
            "domain": "retail",
            "category": "cancellation",
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "retail",
                    "reason_for_call": scenario["reason"],
                    "known_info": f"You are {user['name']['first_name']} {user['name']['last_name']} in zipcode {user['address']['zip']}.",
                    "unknown_info": "You do not remember your email.",
                }
            },
        }
        tasks.append(task)

    return tasks


def gen_retail_auth_tasks(db: dict, rng: random.Random, count: int = 20) -> list:
    """Generate tasks that specifically test authentication behavior."""
    tasks = []
    users = list(db["users"].values())
    orders = list(db["orders"].values())

    scenarios = [
        # User provides user_id directly (should still authenticate)
        lambda u, o: {
            "reason": f"You want to check the status of order {o['order_id']}.",
            "instructions": f"Start by giving your user ID directly: '{u['user_id']}'. If the agent still asks for verification, cooperate.",
            "known_info": f"Your user id is {u['user_id']}.",
            "skills_tested": ["RA1", "RA2"],
        },
        # User refuses to identify at first
        lambda u, o: {
            "reason": "You want to return an item you purchased.",
            "instructions": "Initially refuse to give personal info, saying 'I just want to return something, why do you need my info?' Eventually provide your name and zip when pressed.",
            "known_info": f"You are {u['name']['first_name']} {u['name']['last_name']} in zipcode {u['address']['zip']}.",
            "skills_tested": ["RA2", "RG1"],
        },
        # Product inquiry (no auth needed) then action (auth needed)
        lambda u, o: {
            "reason": "You want to know about available laptop options, and also want to return an item from a previous order.",
            "instructions": "Start by asking about laptops. Then mention you also want to return something. Provide identification when asked.",
            "known_info": f"Your email is {u.get('email', 'unknown@example.com')}.",
            "skills_tested": ["RA1", "RG1"],
        },
    ]

    for i in range(count):
        user = rng.choice(users)
        user_orders = [o for o in orders if o["user_id"] == user["user_id"]]
        order = rng.choice(user_orders) if user_orders else rng.choice(orders)
        scenario_fn = rng.choice(scenarios)
        scenario = scenario_fn(user, order)

        task = {
            "id": f"syn_retail_auth_{i}",
            "domain": "retail",
            "category": "authentication",
            "skills_tested": scenario["skills_tested"],
            "user_scenario": {
                "instructions": {
                    "task_instructions": scenario["instructions"],
                    "domain": "retail",
                    "reason_for_call": scenario["reason"],
                    "known_info": scenario["known_info"],
                    "unknown_info": None,
                }
            },
        }
        tasks.append(task)

    return tasks


# =====================================================================
# MAIN
# =====================================================================

def generate_all_tasks(seed: int = 42) -> list:
    """Generate all synthetic tasks."""
    rng = random.Random(seed)

    airline_db = load_db("airline")
    retail_db = load_db("retail")

    all_tasks = []

    # Airline tasks (~105)
    all_tasks.extend(gen_airline_booking_tasks(airline_db, rng, count=30))
    all_tasks.extend(gen_airline_modification_tasks(airline_db, rng, count=30))
    all_tasks.extend(gen_airline_cancellation_tasks(airline_db, rng, count=25))
    all_tasks.extend(gen_airline_compensation_tasks(airline_db, rng, count=20))

    # Retail tasks (~120)
    all_tasks.extend(gen_retail_return_tasks(retail_db, rng, count=30))
    all_tasks.extend(gen_retail_exchange_tasks(retail_db, rng, count=25))
    all_tasks.extend(gen_retail_modification_tasks(retail_db, rng, count=25))
    all_tasks.extend(gen_retail_cancellation_tasks(retail_db, rng, count=20))
    all_tasks.extend(gen_retail_auth_tasks(retail_db, rng, count=20))

    rng.shuffle(all_tasks)
    print(f"Generated {len(all_tasks)} synthetic tasks")

    # Count by domain/category
    from collections import Counter
    domain_counts = Counter(t["domain"] for t in all_tasks)
    cat_counts = Counter(f"{t['domain']}/{t['category']}" for t in all_tasks)
    print(f"  By domain: {dict(domain_counts)}")
    print(f"  By category: {dict(cat_counts)}")

    return all_tasks


def run_simulations(tasks: list, agent_url: str, user_url: str, model: str,
                    skill_bank_dir: str, output: str, num_trials: int = 5,
                    temperature: float = 0.7):
    """Run simulations on synthetic tasks and convert successful ones to SFT data."""
    import requests

    sys.path.insert(0, str(GAMES_DIR))
    from skillrl.skillbank import SkillBank

    # Load skills and policies
    policies = {}
    skills_sections = {}
    for domain in ("airline", "retail"):
        policy_path = DATA_DIR / domain / "policy.md"
        policies[domain] = policy_path.read_text()

        sb = SkillBank(str(Path(skill_bank_dir) / f"{domain}_skills.json"))
        skills_sections[domain] = sb.get_skills_for_prompt(top_k=100)

    def build_system_prompt(domain):
        return (
            "<instructions>\n"
            "You are a customer service agent that helps the user according to the <policy> provided below.\n"
            "In each turn you can either:\n"
            "- Send a message to the user.\n"
            "- Make a tool call.\n"
            "You cannot do both at the same time.\n"
            "\n"
            "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\n"
            "</instructions>\n"
            "<policy>\n"
            f"{policies[domain]}\n"
            "</policy>\n"
            f"{skills_sections[domain]}"
        )

    def call_llm(url, model_name, messages, temperature=0.0, tools=None):
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }
        if tools:
            payload["tools"] = tools
        resp = requests.post(f"{url}/chat/completions", json=payload,
                             headers={"Authorization": "Bearer EMPTY"}, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    # User simulator system prompt
    user_system = (
        "You are playing the role of a customer contacting a customer service representative.\n"
        "Your goal is to simulate realistic customer interactions while following specific scenario instructions.\n"
        "Generate one message at a time, maintaining natural conversation flow.\n"
        "Strictly follow the scenario instructions you have received.\n"
        "Never make up information not provided in the scenario instructions.\n"
        "If the instruction goal is satisfied, generate '###STOP###' to end the conversation.\n"
        "If you are transferred to another agent, generate '###TRANSFER###'.\n"
    )

    examples = []
    total = len(tasks)

    for ti, task in enumerate(tasks):
        domain = task["domain"]
        scenario = task["user_scenario"]["instructions"]
        system_prompt = build_system_prompt(domain)

        user_scenario_text = (
            f"<scenario>\n"
            f"Reason for call: {scenario['reason_for_call']}\n"
            f"Your information: {scenario['known_info']}\n"
            f"Instructions: {scenario['task_instructions']}\n"
            f"</scenario>"
        )

        for trial in range(num_trials):
            agent_messages = [{"role": "system", "content": system_prompt}]
            user_messages = [{"role": "system", "content": user_system + "\n" + user_scenario_text}]

            conversations = []
            success = False

            # Generate initial user message
            user_messages.append({"role": "user", "content": "Start the conversation as the customer."})
            try:
                user_resp = call_llm(user_url, model, user_messages, temperature=0.0)
                user_msg = user_resp.get("content", "")
            except Exception as e:
                print(f"  [Task {ti}/{total}] User LLM error: {e}")
                break

            if not user_msg or "###STOP###" in user_msg:
                continue

            agent_messages.append({"role": "user", "content": user_msg})
            user_messages.append({"role": "assistant", "content": user_msg})
            conversations.append({"from": "human", "value": user_msg})

            # Multi-turn loop (max 20 turns)
            for turn in range(20):
                # Agent turn
                try:
                    agent_resp = call_llm(agent_url, model, agent_messages, temperature=temperature)
                except Exception as e:
                    print(f"  [Task {ti}/{total}] Agent LLM error: {e}")
                    break

                agent_content = agent_resp.get("content", "")
                tool_calls = agent_resp.get("tool_calls")

                if tool_calls:
                    tc_formatted = []
                    for tc in tool_calls:
                        tc_formatted.append({
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], str)
                            else tc["function"]["arguments"],
                        })
                    conversations.append({"from": "gpt", "value": json.dumps({"tool_calls": tc_formatted})})
                    # Simulate tool response (we don't have the actual executor here)
                    agent_messages.append(agent_resp)
                    tool_resp = {"role": "tool", "content": '{"status": "success"}',
                                 "tool_call_id": tool_calls[0].get("id", "0")}
                    agent_messages.append(tool_resp)
                    conversations.append({"from": "human", "value": f"[Tool Response]: {tool_resp['content']}"})
                elif agent_content:
                    conversations.append({"from": "gpt", "value": agent_content})
                    agent_messages.append({"role": "assistant", "content": agent_content})

                    # User responds
                    user_messages.append({"role": "user", "content": f"Agent said: {agent_content}"})
                    try:
                        user_resp = call_llm(user_url, model, user_messages, temperature=0.0)
                        user_msg = user_resp.get("content", "")
                    except Exception:
                        break

                    if "###STOP###" in user_msg:
                        success = True
                        break
                    if "###TRANSFER###" in user_msg:
                        break

                    user_messages.append({"role": "assistant", "content": user_msg})
                    agent_messages.append({"role": "user", "content": user_msg})
                    conversations.append({"from": "human", "value": user_msg})
                else:
                    break

            # Only keep successful conversations with enough turns
            if success and len(conversations) >= 4:
                # Merge consecutive same-role turns
                merged = []
                for c in conversations:
                    if merged and merged[-1]["from"] == c["from"]:
                        merged[-1]["value"] += "\n" + c["value"]
                    else:
                        merged.append(c)

                if merged[0]["from"] == "human" and merged[-1]["from"] == "gpt":
                    examples.append({
                        "conversations": merged,
                        "system": system_prompt,
                    })
                    print(f"  [Task {ti}/{total}] trial {trial}: SUCCESS ({len(merged)} turns)")
                    break  # One success per task is enough
            else:
                if ti % 20 == 0:
                    print(f"  [Task {ti}/{total}] trial {trial}: {'stopped' if not success else 'too short'}")

    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(examples)} SFT examples from {total} synthetic tasks")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tasks and SFT data for SkillRL")
    parser.add_argument("--generate-tasks", action="store_true",
                        help="Generate synthetic task definitions")
    parser.add_argument("--run-sims", action="store_true",
                        help="Run simulations on tasks to generate SFT data")
    parser.add_argument("--tasks", type=str, default=str(SCRIPT_DIR / "data" / "synthetic_tasks.json"),
                        help="Path to synthetic tasks JSON")
    parser.add_argument("--agent-url", type=str, default="http://localhost:8080/v1")
    parser.add_argument("--user-url", type=str, default="http://localhost:9000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--skill-bank-dir", type=str, default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "data" / "skillrl_sft_train.jsonl"))
    args = parser.parse_args()

    if args.generate_tasks:
        tasks = generate_all_tasks(seed=args.seed)
        Path(args.tasks).parent.mkdir(parents=True, exist_ok=True)
        with open(args.tasks, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"Saved {len(tasks)} tasks to {args.tasks}")

    if args.run_sims:
        with open(args.tasks) as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} tasks from {args.tasks}")
        run_simulations(
            tasks, args.agent_url, args.user_url, args.model,
            args.skill_bank_dir, args.output,
            num_trials=args.num_trials, temperature=args.temperature,
        )

    if not args.generate_tasks and not args.run_sims:
        print("Specify --generate-tasks and/or --run-sims")
        print("  Step 1: python skillrl/generate_synthetic_tasks.py --generate-tasks")
        print("  Step 2: python skillrl/generate_synthetic_tasks.py --run-sims --agent-url ... --user-url ...")


if __name__ == "__main__":
    main()
