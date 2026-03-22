"""
Structured Data Reasoning Game - Training Environment 1

Trains the model to parse JSON data, filter/sort by criteria, select correct
entities, and compute derived values. Single-turn, 3 questions per episode.
No LLM user needed.

Directly addresses ~44% of tau2-bench failures:
- 30 retail wrong item/variant selection failures
- 6 airline wrong flight selection failures
- 12 airline argument computation errors

Reward: fraction of correct answers -> {0, 0.33, 0.67, 1.0}
"""

import random
import json
import math
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Constants for data generation
# =============================================================================

COLORS = ["red", "blue", "green", "black", "white", "silver", "navy", "pink", "gray", "orange"]
SIZES = ["small", "medium", "large", "extra-large"]
MATERIALS = ["plastic", "metal", "wood", "fabric", "glass", "carbon-fiber"]
CONNECTIVITY_TYPES = ["wired", "wireless", "bluetooth"]
POWER_SOURCES = ["AC", "battery", "USB", "solar"]

PRODUCT_TYPES = [
    "Speaker", "Headphones", "Keyboard", "Mouse", "Monitor", "Laptop Stand",
    "USB Hub", "Webcam", "Desk Lamp", "Phone Charger", "Tablet Case",
    "Power Bank", "Smart Watch", "Camera", "Microphone",
]

AIRPORTS = [
    "JFK", "LAX", "ORD", "ATL", "DFW", "SFO", "SEA", "BOS", "DEN", "MIA",
    "PHL", "EWR", "IAH", "MSP", "DTW", "FLL", "SAN", "TPA", "PDX", "STL",
]

AIRLINES = ["HorizonAir", "SkyWest", "Atlantic", "Pacific", "United", "Delta"]

MEMBERSHIP_TIERS = ["regular", "silver", "gold"]
CABINS = ["basic_economy", "economy", "business"]

# Baggage allowance table (matching airline policy)
BAGGAGE_ALLOWANCE = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 0, "economy": 2, "business": 3},
    "gold": {"basic_economy": 0, "economy": 3, "business": 3},
}

# Difficulty presets: (min_items, max_items), max_constraints, question_pool
DIFFICULTY_CONFIG = {
    1: {"items": (3, 5), "constraints": 0,
        "types": ["select_cheapest", "select_max_attr", "count"]},
    2: {"items": (5, 10), "constraints": 1,
        "types": ["select_cheapest", "select_max_attr", "compute_sum", "count"]},
    3: {"items": (10, 15), "constraints": 2,
        "types": ["select_cheapest", "select_nth", "compute_sum", "multi_constraint", "compare"]},
    4: {"items": (15, 20), "constraints": 3,
        "types": ["select_nth", "compute_sum", "compute_derived", "multi_constraint", "compare"]},
    5: {"items": (20, 30), "constraints": 4,
        "types": ["select_nth", "compute_sum", "compute_derived", "multi_constraint", "compare"]},
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Question:
    text: str
    answer: Any           # str (item_id/flight_number), float, int, or "yes"/"no"
    answer_type: str      # "item_id", "number", "boolean"


@dataclass
class Episode:
    data_json: str        # JSON string of the dataset
    data_label: str       # e.g. "Product Catalog"
    data_type: str        # "products", "flights", "baggage"
    questions: List[Question]
    difficulty: int
    items: List[Dict]     # raw data for internal use
    baggage_context: Optional[Dict] = None


# =============================================================================
# Data generation
# =============================================================================

def _gen_product_catalog(rng: random.Random, n: int) -> List[Dict]:
    """Generate n products with unique prices and diverse attributes."""
    items = []
    used_prices = set()

    for i in range(n):
        # Unique price
        price = round(rng.uniform(15.0, 300.0), 2)
        while price in used_prices:
            price = round(price + 0.01, 2)
        used_prices.add(price)

        color = rng.choice(COLORS)
        size = rng.choice(SIZES)
        ptype = rng.choice(PRODUCT_TYPES)
        battery = rng.choice([4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48])

        items.append({
            "item_id": f"item_{1000 + i}",
            "name": f"{color.title()} {size.title()} {ptype}",
            "product_type": ptype,
            "price": price,
            "color": color,
            "size": size,
            "material": rng.choice(MATERIALS),
            "connectivity": rng.choice(CONNECTIVITY_TYPES),
            "power_source": rng.choice(POWER_SOURCES),
            "battery_life_hours": battery,
            "weight_oz": round(rng.uniform(2.0, 40.0), 1),
            "rating": round(rng.uniform(1.0, 5.0), 1),
            "warranty_months": rng.choice([6, 12, 18, 24, 36]),
            "in_stock": rng.random() > 0.1,
        })

    # Shuffle so position doesn't correlate with item_id order
    rng.shuffle(items)
    return items


def _gen_flight_list(rng: random.Random, n: int) -> List[Dict]:
    """Generate n flights with unique prices."""
    airports = list(AIRPORTS)
    rng.shuffle(airports)
    origins = airports[:3]
    destinations = airports[3:6]

    flights = []
    used_prices = set()

    for i in range(n):
        origin = rng.choice(origins)
        dest = rng.choice(destinations)
        while dest == origin:
            dest = rng.choice(destinations)

        price = round(rng.uniform(80.0, 800.0), 2)
        while price in used_prices:
            price = round(price + 0.01, 2)
        used_prices.add(price)

        stops = rng.choices([0, 1, 2], weights=[50, 35, 15])[0]
        base_dur = rng.randint(120, 360)
        duration = base_dur + stops * rng.randint(60, 120)

        cabin = rng.choice(["economy", "business"])
        day = rng.randint(16, 22)
        hour = rng.randint(6, 22)
        minute = rng.choice([0, 15, 30, 45])

        flights.append({
            "flight_number": f"HAT{100 + i}",
            "origin": origin,
            "destination": dest,
            "date": f"2024-05-{day:02d}",
            "departure_time": f"{hour:02d}:{minute:02d}",
            "cabin": cabin,
            "price": price,
            "stops": stops,
            "duration_minutes": duration,
            "airline": rng.choice(AIRLINES),
            "seats_available": rng.randint(1, 50),
        })

    rng.shuffle(flights)
    return flights


def _gen_baggage_context(rng: random.Random) -> Tuple[Dict, List[Dict]]:
    """Generate a membership/baggage scenario with reservations."""
    tier = rng.choice(MEMBERSHIP_TIERS)
    reservations = []
    for i in range(rng.randint(3, 6)):
        cabin = rng.choice(CABINS)
        reservations.append({
            "reservation_id": f"RES{10000 + i}",
            "cabin": cabin,
            "num_passengers": rng.randint(1, 4),
            "total_baggages": rng.randint(0, 8),
            "insurance": rng.choice(["yes", "no"]),
            "trip_price": round(rng.uniform(200, 3000), 2),
        })

    context = {
        "membership_tier": tier,
        "baggage_policy": BAGGAGE_ALLOWANCE,
        "reservations": reservations,
    }
    return context, reservations


# =============================================================================
# Question generators
# =============================================================================

def _q_select_cheapest(rng: random.Random, items: List[Dict], dtype: str,
                       constraints: int) -> Optional[Question]:
    """Which item is the cheapest (optionally with filter constraints)?"""
    filtered = list(items)
    desc_parts = []

    if dtype == "products":
        price_key = "price"
        id_key = "item_id"
        id_label = "item_id"
        noun = "product"
        filter_attrs = ["color", "size", "connectivity", "material"]
    elif dtype == "flights":
        price_key = "price"
        id_key = "flight_number"
        id_label = "flight_number"
        noun = "flight"
        filter_attrs = ["cabin", "origin", "destination"]
    else:
        return None

    for _ in range(min(constraints, len(filter_attrs))):
        if len(filtered) <= 2:
            break
        attr = rng.choice(filter_attrs)
        filter_attrs.remove(attr)
        values = list(set(item[attr] for item in filtered))
        if len(values) < 2:
            continue
        target = rng.choice(values)
        new_filtered = [x for x in filtered if x[attr] == target]
        if len(new_filtered) >= 1:
            filtered = new_filtered
            desc_parts.append(f"{attr}=\"{target}\"")

    if not filtered:
        return None

    cheapest = min(filtered, key=lambda x: x[price_key])
    if desc_parts:
        constraint_str = " with " + " and ".join(desc_parts)
    else:
        constraint_str = ""

    return Question(
        text=f"Which {noun} is the cheapest{constraint_str}? Answer with the {id_label}.",
        answer=cheapest[id_key],
        answer_type="item_id",
    )


def _q_select_max_attr(rng: random.Random, items: List[Dict],
                       dtype: str) -> Optional[Question]:
    """Which item has the max/min of a numeric attribute?"""
    if dtype == "products":
        choices = [
            ("battery_life_hours", "longest battery life", True),
            ("weight_oz", "heaviest weight", True),
            ("rating", "highest rating", True),
            ("warranty_months", "longest warranty", True),
        ]
        id_key = "item_id"
        noun = "product"
    elif dtype == "flights":
        choices = [
            ("duration_minutes", "longest duration", True),
            ("seats_available", "most available seats", True),
        ]
        id_key = "flight_number"
        noun = "flight"
    else:
        return None

    attr, desc, is_max = rng.choice(choices)
    # Sometimes flip to min
    if rng.random() < 0.3:
        desc = desc.replace("longest", "shortest").replace("heaviest", "lightest")
        desc = desc.replace("highest", "lowest").replace("most", "fewest")
        is_max = not is_max

    target = max(items, key=lambda x: x[attr]) if is_max else min(items, key=lambda x: x[attr])
    return Question(
        text=f"Which {noun} has the {desc}? Answer with the {id_key}.",
        answer=target[id_key],
        answer_type="item_id",
    )


def _q_select_nth(rng: random.Random, items: List[Dict],
                  dtype: str) -> Optional[Question]:
    """Which item is the Nth cheapest/most expensive?"""
    if len(items) < 3:
        return None

    n = rng.randint(2, min(5, len(items)))
    ordinal = {2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}.get(n, f"{n}th")

    if dtype == "products":
        id_key = "item_id"
        noun = "product"
    elif dtype == "flights":
        id_key = "flight_number"
        noun = "flight"
    else:
        return None

    ascending = rng.random() < 0.6
    sorted_items = sorted(items, key=lambda x: x["price"], reverse=not ascending)
    target = sorted_items[n - 1]
    direction = "cheapest" if ascending else "most expensive"

    return Question(
        text=f"Which {noun} is the {ordinal} {direction}? Answer with the {id_key}.",
        answer=target[id_key],
        answer_type="item_id",
    )


def _q_compute_sum(rng: random.Random, items: List[Dict],
                   dtype: str) -> Optional[Question]:
    """What is the total price of items X and Y?"""
    if len(items) < 2:
        return None

    k = rng.randint(2, min(3, len(items)))
    selected = rng.sample(items, k)

    if dtype == "products":
        ids = [x["item_id"] for x in selected]
        id_label = "items"
    elif dtype == "flights":
        ids = [x["flight_number"] for x in selected]
        id_label = "flights"
    else:
        return None

    total = round(sum(x["price"] for x in selected), 2)
    ids_str = ", ".join(ids)

    return Question(
        text=f"What is the total price of {id_label} {ids_str}? Answer with a number.",
        answer=total,
        answer_type="number",
    )


def _q_compute_derived(rng: random.Random, context: Dict) -> Optional[Question]:
    """Compute nonfree bags using tier table + reservation data."""
    if context is None:
        return None

    tier = context["membership_tier"]
    reservations = context["reservations"]
    policy = context["baggage_policy"]

    if not reservations:
        return None

    res = rng.choice(reservations)
    cabin = res["cabin"]
    total_bags = res["total_baggages"]
    free_bags = policy[tier][cabin]
    nonfree = max(0, total_bags - free_bags)

    return Question(
        text=(
            f"Reservation {res['reservation_id']} has {total_bags} total bags. "
            f"The member is {tier} tier in {cabin} cabin. "
            f"According to the baggage_policy table, how many nonfree (paid) bags "
            f"does this reservation have? Answer with a number."
        ),
        answer=nonfree,
        answer_type="number",
    )


def _q_multi_constraint(rng: random.Random, items: List[Dict],
                        dtype: str) -> Optional[Question]:
    """Cheapest item matching 2-3 attribute constraints."""
    if dtype != "products" or len(items) < 5:
        return None

    attrs = ["color", "size", "connectivity", "material", "power_source"]
    rng.shuffle(attrs)
    num_c = rng.randint(2, 3)

    filtered = list(items)
    desc_parts = []

    for attr in attrs[:num_c]:
        values = list(set(x[attr] for x in filtered))
        if len(values) < 2:
            continue
        target = rng.choice(values)
        new_filtered = [x for x in filtered if x[attr] == target]
        if new_filtered:
            filtered = new_filtered
            desc_parts.append(f"{attr}=\"{target}\"")

    if not filtered or len(desc_parts) < 2:
        return None

    cheapest = min(filtered, key=lambda x: x["price"])
    constraint_str = " AND ".join(desc_parts)

    return Question(
        text=f"Which is the cheapest product where {constraint_str}? Answer with the item_id.",
        answer=cheapest["item_id"],
        answer_type="item_id",
    )


def _q_compare(rng: random.Random, items: List[Dict],
               dtype: str) -> Optional[Question]:
    """Is item A cheaper than items B + C combined?"""
    if len(items) < 3:
        return None

    a, b, c = rng.sample(items, 3)

    if dtype == "products":
        id_key = "item_id"
    elif dtype == "flights":
        id_key = "flight_number"
    else:
        return None

    result = "yes" if a["price"] < (b["price"] + c["price"]) else "no"

    return Question(
        text=(
            f"Is {a[id_key]} (${a['price']}) cheaper than {b[id_key]} (${b['price']}) "
            f"and {c[id_key]} (${c['price']}) combined? Answer 'yes' or 'no'."
        ),
        answer=result,
        answer_type="boolean",
    )


def _q_count(rng: random.Random, items: List[Dict],
             dtype: str) -> Optional[Question]:
    """How many items match a given criterion?"""
    if dtype == "products":
        attr = rng.choice(["color", "size", "connectivity"])
        values = list(set(x[attr] for x in items))
        if len(values) < 2:
            return None
        target = rng.choice(values)
        count = sum(1 for x in items if x[attr] == target)
        return Question(
            text=f"How many products have {attr}=\"{target}\"? Answer with a number.",
            answer=count,
            answer_type="number",
        )
    elif dtype == "flights":
        attr = rng.choice(["cabin", "stops"])
        if attr == "cabin":
            target = rng.choice(["economy", "business"])
            count = sum(1 for f in items if f["cabin"] == target)
            return Question(
                text=f"How many flights have cabin=\"{target}\"? Answer with a number.",
                answer=count,
                answer_type="number",
            )
        else:
            target = rng.choice([0, 1])
            count = sum(1 for f in items if f["stops"] == target)
            label = "direct (0 stops)" if target == 0 else "1-stop"
            return Question(
                text=f"How many {label} flights are there? Answer with a number.",
                answer=count,
                answer_type="number",
            )
    return None


def _q_price_lookup(rng: random.Random, items: List[Dict],
                    dtype: str) -> Question:
    """Simple fallback: what is the price of item X?"""
    item = rng.choice(items)
    if dtype == "products":
        return Question(
            text=f"What is the price of {item['item_id']}? Answer with a number.",
            answer=item["price"],
            answer_type="number",
        )
    elif dtype == "flights":
        return Question(
            text=f"What is the price of flight {item['flight_number']}? Answer with a number.",
            answer=item["price"],
            answer_type="number",
        )
    else:  # baggage
        return Question(
            text=f"What is the trip_price of reservation {item['reservation_id']}? Answer with a number.",
            answer=item["trip_price"],
            answer_type="number",
        )


# =============================================================================
# Episode generation
# =============================================================================

def generate_episode(seed: int, difficulty: int = 3) -> Episode:
    """Generate a single episode with data + 3 questions."""
    rng = random.Random(seed)
    difficulty = max(1, min(5, difficulty))
    config = DIFFICULTY_CONFIG[difficulty]

    num_items = rng.randint(*config["items"])

    # Pick data type; baggage only at difficulty >= 4
    if difficulty >= 4 and rng.random() < 0.3:
        data_type = "baggage"
    else:
        data_type = rng.choice(["products", "flights"])

    # Generate data
    baggage_ctx = None
    if data_type == "products":
        items = _gen_product_catalog(rng, num_items)
        data_json = json.dumps(items, indent=2)
        data_label = "Product Catalog"
    elif data_type == "flights":
        items = _gen_flight_list(rng, num_items)
        data_json = json.dumps(items, indent=2)
        data_label = "Flight Search Results"
    else:
        baggage_ctx, items = _gen_baggage_context(rng)
        data_json = json.dumps(baggage_ctx, indent=2)
        data_label = "Membership & Reservations"

    constraints = config["constraints"]
    avail_types = list(config["types"])

    # Build generator map
    generators = {
        "select_cheapest": lambda: _q_select_cheapest(rng, items, data_type, constraints),
        "select_max_attr": lambda: _q_select_max_attr(rng, items, data_type),
        "select_nth": lambda: _q_select_nth(rng, items, data_type),
        "compute_sum": lambda: _q_compute_sum(rng, items, data_type),
        "compute_derived": lambda: _q_compute_derived(rng, baggage_ctx),
        "multi_constraint": lambda: _q_multi_constraint(rng, items, data_type),
        "compare": lambda: _q_compare(rng, items, data_type),
        "count": lambda: _q_count(rng, items, data_type),
    }

    questions: List[Question] = []
    used_answers = set()
    attempts = 0

    while len(questions) < 3 and attempts < 50:
        attempts += 1
        qtype = rng.choice(avail_types)
        gen = generators.get(qtype)
        if gen is None:
            continue
        q = gen()
        if q is None:
            continue
        # Avoid duplicate answers
        ans_key = str(q.answer)
        if ans_key in used_answers:
            continue
        used_answers.add(ans_key)
        questions.append(q)

    # Pad with simple price-lookup fallbacks if needed
    while len(questions) < 3:
        q = _q_price_lookup(rng, items, data_type)
        if str(q.answer) not in used_answers:
            used_answers.add(str(q.answer))
            questions.append(q)

    return Episode(
        data_json=data_json,
        data_label=data_label,
        data_type=data_type,
        questions=questions[:3],
        difficulty=difficulty,
        items=items,
        baggage_context=baggage_ctx,
    )


# =============================================================================
# Answer checking
# =============================================================================

def _check_answer(predicted: Any, expected: Any, answer_type: str) -> bool:
    """Check if a predicted answer matches the expected answer."""
    if predicted is None:
        return False

    if answer_type == "item_id":
        return str(predicted).strip() == str(expected).strip()

    elif answer_type == "number":
        try:
            # Strip $ and commas from string predictions
            pred_str = str(predicted).replace("$", "").replace(",", "").strip()
            pred_val = float(pred_str)
            exp_val = float(expected)
            return abs(pred_val - exp_val) < 0.02
        except (ValueError, TypeError):
            return False

    elif answer_type == "boolean":
        pred_str = str(predicted).strip().lower()
        exp_str = str(expected).strip().lower()
        return pred_str == exp_str

    return str(predicted).strip() == str(expected).strip()


# =============================================================================
# Game class
# =============================================================================

class StructuredDataGame:
    """Single-turn structured data reasoning game.

    The model sees a JSON dataset + 3 questions, outputs answers as JSON.
    Reward = fraction of correct answers (0, 0.33, 0.67, 1.0).
    """

    def __init__(self, difficulty: int = 3):
        self._difficulty = difficulty
        self._episode: Optional[Episode] = None

        # GameEnv protocol
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def reset(self, seed: int) -> None:
        self._episode = generate_episode(seed, self._difficulty)
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        ep = self._episode
        if ep is None:
            return "No episode loaded."

        lines = [
            "You are a customer service agent. Answer the following questions "
            "about the data provided below.",
            "Output your answers using the submit_answers tool.",
            "",
            f"<data type=\"{ep.data_label}\">",
            ep.data_json,
            "</data>",
            "",
            "Please answer ALL three questions based ONLY on the data above:",
            "",
        ]
        for i, q in enumerate(ep.questions, 1):
            lines.append(f"Question {i}: {q.text}")
        lines.extend([
            "",
            "Respond with exactly one JSON object:",
            '{"name": "submit_answers", "arguments": {"answer_1": ..., "answer_2": ..., "answer_3": ...}}',
            "",
            "Rules:",
            '- For item_id / flight_number answers, use the exact string (e.g., "item_1003")',
            "- For numeric answers, use a number (e.g., 145.50)",
            '- For yes/no answers, use a string (e.g., "yes" or "no")',
        ])
        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "submit_answers", "arguments": {"answer_1": ..., "answer_2": ..., "answer_3": ...}}']

    def step(self, action: Optional[str]) -> None:
        if self.done:
            return

        if action is None:
            self._finalize(0.0, "No action provided")
            return

        # Parse JSON
        tool_call = _parse_tool_call(action)
        if tool_call is None:
            self._finalize(0.0, "Invalid JSON format")
            self.invalid_player = 0
            return

        args = tool_call.get("arguments", {})

        # Check each answer
        ep = self._episode
        correct = 0
        details = []
        for i, q in enumerate(ep.questions, 1):
            predicted = args.get(f"answer_{i}")
            is_correct = _check_answer(predicted, q.answer, q.answer_type)
            if is_correct:
                correct += 1
            details.append(
                f"Q{i}: predicted={predicted!r} expected={q.answer!r} -> "
                f"{'CORRECT' if is_correct else 'WRONG'}"
            )

        reward = round(correct / 3, 2)
        reason = f"{correct}/3 correct. " + "; ".join(details)
        self._finalize(reward, reason)

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    def get_summary(self) -> Dict[str, Any]:
        ep = self._episode
        return {
            "data_type": ep.data_type if ep else "",
            "difficulty": ep.difficulty if ep else 0,
            "num_items": len(ep.items) if ep else 0,
            "questions": [
                {"text": q.text, "answer": q.answer, "type": q.answer_type}
                for q in (ep.questions if ep else [])
            ],
            "reward": self.rewards.get(0, 0.0),
            "reason": getattr(self, "_reason", ""),
        }


# =============================================================================
# Action parsing
# =============================================================================

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
                        # Also accept bare answer objects
                        if "answer_1" in obj:
                            return {"name": "submit_answers", "arguments": obj}
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass
    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action for game registry compatibility."""
    import re
    tool_call = _parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    # Try to find bare JSON answer object after stripping thinking tags
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    tool_call = _parse_tool_call(clean)
    if tool_call:
        return json.dumps(tool_call)
    return None


# =============================================================================
# System prompt
# =============================================================================

SYSTEM_PROMPT = (
    "You are a customer service agent skilled at analyzing structured data.\n"
    "Given a dataset (products, flights, or reservation info) and questions,\n"
    "analyze the data carefully and provide precise answers.\n"
    "Always respond with valid JSON using the submit_answers tool.\n"
)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Structured Data Reasoning Game")
    print("=" * 60)

    for diff in [1, 2, 3, 4, 5]:
        rewards = []
        dtypes = {"products": 0, "flights": 0, "baggage": 0}
        game = StructuredDataGame(difficulty=diff)

        for seed in range(100):
            game.reset(seed)
            ep = game._episode

            dtypes[ep.data_type] = dtypes.get(ep.data_type, 0) + 1

            # Simulate a "perfect" agent that gives correct answers
            answers = {}
            for i, q in enumerate(ep.questions, 1):
                answers[f"answer_{i}"] = q.answer

            action = json.dumps({
                "name": "submit_answers",
                "arguments": answers,
            })
            game.step(action)
            rewards.append(game.rewards[0])

        avg = sum(rewards) / len(rewards)
        print(f"\nDifficulty {diff}: avg_reward={avg:.2f} (should be ~1.0 with perfect answers)")
        print(f"  Data types: {dtypes}")
        print(f"  Items per episode: {game._episode and len(game._episode.items)}")

    # Show example observations
    print("\n" + "=" * 60)
    print("Example episode (difficulty=3, seed=42):")
    game = StructuredDataGame(difficulty=3)
    game.reset(42)
    obs = game.observe(0)
    print(obs[:2000])
    print("..." if len(obs) > 2000 else "")
    print(f"\nExpected answers:")
    for i, q in enumerate(game._episode.questions, 1):
        print(f"  answer_{i}: {q.answer!r} ({q.answer_type})")
