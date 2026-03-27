# Structured Data Game v5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `structured_data_new_game.py` with 5 new scenario types (3 airline, 2 retail) and adjusted weights to cover 100% of airline and 86% of retail SD reasoning eval failures.

**Architecture:** Single-turn game with pre-filled conversation prefix. Add new scenario generators following the exact same pattern as existing ones (generate data → build gold_action → build conversation prefix → return scenario dict). Update weights and `generate_scenario()` dispatch. Add `passengers`/`payment_methods` handling to `_score()`.

**Tech Stack:** Python, tau2-bench tool schemas, existing game infrastructure.

**Key files:**
- Modify: `/home/ubuntu/hangook/games/structured_data_new_game.py`

**Reference files (read-only):**
- Eval failures: `evals/benchmarks/tau2_bench_eval/data/simulations/qwen-3-30b-airline-qwen3-30b.json`
- Eval failures: `evals/benchmarks/tau2_bench_eval/data/simulations/qwen-3-30b-retail-qwen3-30b.json`
- Airline policy: `tau2-bench/data/tau2/domains/airline/policy.md`
- Existing game code: `structured_data_new_game.py` (all existing functions)

---

## Context: What SD Reasoning Failures Exist

### Airline (16 SD tasks / 38 total failures / 50 tasks)
- `flight_selection` (6 tasks): wrong flight_number from search results — **ALREADY COVERED**
- `book_flight` (5 tasks): wrong passengers/payment/baggages/flights in book_reservation — **NEW**
- `update_baggages` (3 tasks): wrong nonfree_baggages computation — **NEW**
- `send_compensation` (2 tasks): wrong certificate amount — **NEW**
- `payment_selection` (2 tasks): wrong payment_id — **ALREADY COVERED**
- `reservation_id` (1 task): wrong reservation_id — **ALREADY COVERED**

### Retail (29 SD tasks / 72 total failures / 114 tasks)
- `variant_selection` (18 tasks): wrong new_item_ids — **ALREADY COVERED**
- `item_identification` (11 tasks): wrong item_ids — **ALREADY COVERED** (+ return_items re-enable)
- `order_selection` (4 tasks): wrong order_id — **NEW**
- `payment_selection_retail` (4 tasks): wrong payment_method_id — **NEW**
- `address_extraction` (2 tasks): SKIPPED (too niche)
- `transfer_summary` (1 task): SKIPPED (not SD reasoning)
- `cancel_reason` (1 task): SKIPPED (not SD reasoning)

---

### Task 1: Update Weights and Domain Distribution

**Files:**
- Modify: `/home/ubuntu/hangook/games/structured_data_new_game.py` (lines 698-708, 1507-1511, 2031)

- [ ] **Step 1: Update RETAIL_SCENARIO_WEIGHTS**

Change lines 698-708 to:

```python
RETAIL_SCENARIO_WEIGHTS = {
    "single_exchange": 10,
    "multi_exchange": 10,
    "single_modify": 5,
    "extremal_exchange": 20,
    "conditional_exchange": 15,
    "cross_ref_exchange": 15,
    "return_items": 10,
    "order_selection": 10,       # NEW
    "payment_selection": 5,      # NEW
    "no_match": 0,
}
```

- [ ] **Step 2: Update AIRLINE_SCENARIO_WEIGHTS**

Change lines 1507-1511 to:

```python
AIRLINE_SCENARIO_WEIGHTS = {
    "flight_selection": 30,
    "book_flight": 25,           # NEW
    "update_baggages": 15,       # NEW
    "send_compensation": 10,     # NEW
    "payment_selection": 10,
    "reservation_id": 10,
}
```

- [ ] **Step 3: Update DOMAIN_WEIGHTS**

Change line 2031 to:

```python
DOMAIN_WEIGHTS = {"retail": 35, "airline": 65}
```

(No change needed — already 35/65.)

- [ ] **Step 4: Commit**

```bash
git add structured_data_new_game.py
git commit -m "chore: update SDR game weights for v5 scenario distribution"
```

---

### Task 2: Add Airline `book_flight` Scenario

Targets 5 eval SD failures (T8, T20, T24, T25, T35). The model must construct full `book_reservation` args from search results + user details. Common errors: wrong passengers (hallucinated extras), wrong payment split, wrong total_baggages (added when not requested), wrong flight selection.

**Files:**
- Modify: `/home/ubuntu/hangook/games/structured_data_new_game.py` (insert after `_gen_payment_selection_scenario`, ~line 1880)

- [ ] **Step 1: Add `_gen_book_flight_scenario` function**

Insert after `_gen_payment_selection_scenario` (around line 1880):

```python
def _gen_book_flight_scenario(rng: random.Random) -> Dict[str, Any]:
    """Book a new reservation: pick correct flight + construct all args.

    Trains: construct book_reservation arguments from user request + search
    results + user details. Model must NOT hallucinate extra passengers,
    must pick the correct payment method, compute baggage correctly, and
    respect user's stated preferences (no extras unless asked).

    Matches eval failures T8, T20, T24, T25, T35.
    """
    user_info = _gen_user(rng, domain="airline")
    user = user_info["user"]

    # Route
    codes = rng.sample(IATA_CODES, 2)
    origin, destination = codes[0], codes[1]
    date = f"2024-05-{rng.randint(15, 28):02d}"

    # Cabin
    cabin = rng.choice(["economy", "business"])

    # Flight search results (3-6 flights)
    n_flights = rng.randint(3, 6)
    flights = [_gen_flight(rng, origin, destination) for _ in range(n_flights)]

    # Pick target flight (cheapest in requested cabin)
    target_flight = min(flights, key=lambda f: f["prices"][cabin])

    # Passengers: user is always a passenger. Sometimes add saved passengers.
    first = user_info["first_name"]
    last = user_info["last_name"]
    dob = user["dob"]
    passengers = [{"first_name": first, "last_name": last, "dob": dob}]

    n_extra = rng.choices([0, 1, 2], weights=[60, 30, 10])[0]
    # Only add extras if explicitly mentioned in user message
    extra_passengers = []
    for _ in range(n_extra):
        ep_first = rng.choice(FIRST_NAMES)
        ep_last = last  # same family
        ep_dob = f"{rng.randint(1960, 2005)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        extra_passengers.append({"first_name": ep_first, "last_name": ep_last, "dob": ep_dob})
        passengers.append(extra_passengers[-1])

    n_passengers = len(passengers)

    # Baggage: explicit in user message
    total_baggages = rng.choices([0, 1, 2], weights=[50, 30, 20])[0]
    membership = user["membership"]
    free_per_pax = FREE_BAGS.get((membership, cabin), 0)
    nonfree = max(0, total_baggages - free_per_pax * n_passengers)

    # Insurance
    insurance = rng.choice(["yes", "no"])

    # Payment: user specifies one payment method
    payment_desc, payment_id = _describe_payment(rng, user_info, domain="airline")
    flight_price = target_flight["prices"][cabin]
    total_cost = flight_price * n_passengers + nonfree * 50
    if insurance == "yes":
        total_cost += 30 * n_passengers
    payment_methods = [{"payment_id": payment_id, "amount": total_cost}]

    # Build user message
    pax_desc = ""
    if extra_passengers:
        extras = ", ".join(f"{p['first_name']} {p['last_name']}" for p in extra_passengers)
        pax_desc = f" I'll be traveling with {extras}."

    bag_desc = ""
    if total_baggages > 0:
        bag_desc = f" I need {total_baggages} checked bag{'s' if total_baggages > 1 else ''}."

    ins_desc = ""
    if insurance == "yes":
        ins_desc = " Please add travel insurance."

    user_msg = (
        f"Hi, I'm {first} {last}, "
        f"zip code {user_info['zip']}. I'd like to book the cheapest "
        f"{cabin} flight from {origin} to {destination} on {date}.{pax_desc}"
        f"{bag_desc}{ins_desc} Use {payment_desc} for payment."
    )

    presentation = (
        f"I found several {cabin} flights from {origin} to {destination} "
        f"on {date}. Let me find the cheapest option and book it for you. "
        f"Shall I proceed?"
    )

    gold_action = {
        "name": "book_reservation",
        "arguments": {
            "user_id": user_info["user_id"],
            "origin": origin,
            "destination": destination,
            "flight_type": "one_way",
            "cabin": cabin,
            "flights": [{"flight_number": target_flight["flight_number"], "date": date}],
            "passengers": passengers,
            "payment_methods": payment_methods,
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree,
            "insurance": insurance,
        },
        "compare_args": ["origin", "destination", "cabin", "flights",
                         "passengers", "payment_methods", "total_baggages",
                         "nonfree_baggages", "insurance"],
    }

    return {
        "domain": "airline",
        "scenario_type": "book_flight",
        "user_info": user_info,
        "user": user,
        "reservation": None,
        "distractor_reservations": [],
        "search_results": flights,
        "search_origin": origin,
        "search_destination": destination,
        "search_date": date,
        "user_message": user_msg,
        "presentation": presentation,
        "gold_action": gold_action,
    }
```

- [ ] **Step 2: Add `book_flight` to generate_scenario dispatch**

In `generate_scenario()` (around line 2078-2083), add to the airline generators dict:

```python
            "book_flight": _gen_book_flight_scenario,
```

- [ ] **Step 3: Commit**

```bash
git add structured_data_new_game.py
git commit -m "feat: add book_flight airline scenario to SDR game"
```

---

### Task 3: Add Airline `update_baggages` Scenario

Targets 3 eval SD failures (T17, T21, T29). The model always gets `nonfree_baggages` wrong (puts >0 when should be 0). Must compute: `nonfree = max(0, total_bags - free_per_pax * n_passengers)` using the policy's FREE_BAGS table.

**Files:**
- Modify: `/home/ubuntu/hangook/games/structured_data_new_game.py` (insert after `_gen_book_flight_scenario`)

- [ ] **Step 1: Add `_gen_update_baggages_scenario` function**

```python
def _gen_update_baggages_scenario(rng: random.Random) -> Dict[str, Any]:
    """Add checked bags to a reservation: compute nonfree correctly.

    The model must look up membership + cabin → free bags per passenger,
    then compute nonfree_baggages = max(0, total - free_per_pax * n_pax).

    Matches eval failures T17, T21, T29 where nonfree is always wrong.
    """
    user_info = _gen_user(rng, domain="airline")
    user = user_info["user"]
    membership = user["membership"]

    # Route and reservation
    codes = rng.sample(IATA_CODES, 2)
    origin, destination = codes[0], codes[1]
    cabin = rng.choice(CABIN_CLASSES)
    n_passengers = rng.randint(1, 3)
    reservation = _gen_reservation(rng, user_info["user_id"], origin, destination,
                                    cabin, "one_way", n_passengers)

    # Current baggage state
    current_bags = reservation["total_baggages"]

    # User wants to ADD bags
    bags_to_add = rng.randint(1, 3)
    new_total = current_bags + bags_to_add

    # Compute free bags from policy
    free_per_pax = FREE_BAGS.get((membership, cabin), 0)
    free_total = free_per_pax * n_passengers
    nonfree = max(0, new_total - free_total)

    user["reservations"] = [reservation["reservation_id"]]
    payment_desc, payment_id = _describe_payment(rng, user_info, domain="airline")

    user_msg = (
        f"Hi, I'm {user_info['first_name']} {user_info['last_name']}, "
        f"zip code {user_info['zip']}. I'd like to add {bags_to_add} "
        f"checked bag{'s' if bags_to_add > 1 else ''} to my reservation "
        f"{reservation['reservation_id']}. Use {payment_desc} for payment."
    )

    presentation = (
        f"I found your reservation {reservation['reservation_id']} "
        f"({origin} to {destination}, {cabin} class, "
        f"{n_passengers} passenger{'s' if n_passengers > 1 else ''}, "
        f"currently {current_bags} checked bag{'s' if current_bags != 1 else ''}). "
        f"Shall I add {bags_to_add} more bag{'s' if bags_to_add > 1 else ''}?"
    )

    gold_action = {
        "name": "update_reservation_baggages",
        "arguments": {
            "reservation_id": reservation["reservation_id"],
            "total_baggages": new_total,
            "nonfree_baggages": nonfree,
            "payment_id": payment_id,
        },
        "compare_args": ["reservation_id", "total_baggages",
                         "nonfree_baggages", "payment_id"],
    }

    return {
        "domain": "airline",
        "scenario_type": "update_baggages",
        "user_info": user_info,
        "user": user,
        "reservation": reservation,
        "distractor_reservations": [],
        "search_results": [],
        "search_origin": origin,
        "search_destination": destination,
        "search_date": reservation["flights"][0]["date"],
        "user_message": user_msg,
        "presentation": presentation,
        "gold_action": gold_action,
    }
```

- [ ] **Step 2: Add `update_baggages` to generate_scenario dispatch**

Add to airline generators dict:

```python
            "update_baggages": _gen_update_baggages_scenario,
```

- [ ] **Step 3: Commit**

```bash
git add structured_data_new_game.py
git commit -m "feat: add update_baggages airline scenario to SDR game"
```

---

### Task 4: Add Airline `send_compensation` Scenario

Targets 2 eval SD failures (T2, T38). Model over-compensates (gives $300 or $150 instead of $50). Policy rules:
- Cancelled flight: $100 × n_passengers
- Delayed flight (with change/cancel): $50 × n_passengers
- Only if: silver/gold member OR has travel insurance OR flies business
- Regular member + no insurance + (basic) economy = NO compensation

**Files:**
- Modify: `/home/ubuntu/hangook/games/structured_data_new_game.py` (insert after `_gen_update_baggages_scenario`)

- [ ] **Step 1: Add `_gen_send_compensation_scenario` function**

```python
def _gen_send_compensation_scenario(rng: random.Random) -> Dict[str, Any]:
    """Issue travel certificate for delayed/cancelled flight.

    Amount depends on reason and number of passengers:
      - Cancelled: $100 * n_passengers
      - Delayed (with change/cancel): $50 * n_passengers

    Only eligible if: silver/gold member OR has insurance OR flies business.
    The scenario always places the user in an eligible state.

    Matches eval failures T2, T38 where model over-compensates.
    """
    user_info = _gen_user(rng, domain="airline")
    user = user_info["user"]

    # Make user eligible for compensation
    eligible_reason = rng.choice(["membership", "insurance", "cabin"])
    if eligible_reason == "membership":
        user["membership"] = rng.choice(["silver", "gold"])
    elif eligible_reason == "insurance":
        user["membership"] = "regular"  # force regular so insurance is the reason

    codes = rng.sample(IATA_CODES, 2)
    origin, destination = codes[0], codes[1]

    if eligible_reason == "cabin":
        cabin = "business"
    else:
        cabin = rng.choice(["basic_economy", "economy"])

    n_passengers = rng.randint(1, 3)
    reservation = _gen_reservation(rng, user_info["user_id"], origin, destination,
                                    cabin, "one_way", n_passengers)

    if eligible_reason == "insurance":
        reservation["insurance"] = "yes"
    elif eligible_reason != "insurance" and rng.random() < 0.5:
        reservation["insurance"] = "no"

    # Flight status
    flight_issue = rng.choice(["cancelled", "delayed"])
    flight_number = reservation["flights"][0]["flight_number"]
    flight_date = reservation["flights"][0]["date"]

    # Build flight status result
    if flight_issue == "cancelled":
        flight_status = {
            "flight_number": flight_number,
            "date": flight_date,
            "status": "cancelled",
        }
        amount = 100 * n_passengers
    else:
        flight_status = {
            "flight_number": flight_number,
            "date": flight_date,
            "status": "delayed",
            "estimated_departure_time_est": f"{flight_date}T{rng.randint(14,20):02d}:00:00",
            "estimated_arrival_time_est": f"{flight_date}T{rng.randint(20,23):02d}:00:00",
        }
        amount = 50 * n_passengers

    user["reservations"] = [reservation["reservation_id"]]

    if flight_issue == "cancelled":
        issue_desc = f"My flight {flight_number} on {flight_date} was cancelled"
    else:
        issue_desc = f"My flight {flight_number} on {flight_date} was delayed"

    user_msg = (
        f"Hi, I'm {user_info['first_name']} {user_info['last_name']}, "
        f"zip code {user_info['zip']}. {issue_desc}. "
        f"I'd like to request compensation."
    )

    presentation = (
        f"I've verified your reservation {reservation['reservation_id']} "
        f"and confirmed that flight {flight_number} was {flight_issue}. "
        f"You are eligible for a travel certificate. Shall I issue it?"
    )

    gold_action = {
        "name": "send_certificate",
        "arguments": {
            "user_id": user_info["user_id"],
            "amount": amount,
        },
        "compare_args": ["user_id", "amount"],
    }

    return {
        "domain": "airline",
        "scenario_type": "send_compensation",
        "user_info": user_info,
        "user": user,
        "reservation": reservation,
        "distractor_reservations": [],
        "search_results": [],
        "search_origin": origin,
        "search_destination": destination,
        "search_date": flight_date,
        "flight_status": flight_status,
        "user_message": user_msg,
        "presentation": presentation,
        "gold_action": gold_action,
    }
```

- [ ] **Step 2: Update `_build_airline_conversation` to include flight_status**

In `_build_airline_conversation` (around line 1950), after the flight search block (line ~2018), add:

```python
    # Flight status lookup (for compensation scenarios)
    flight_status = scenario.get("flight_status")
    if flight_status:
        fn = flight_status.get("flight_number", "")
        dt = flight_status.get("date", "")
        tc_id = add_tool_call("get_flight_status", {
            "flight_number": fn, "date": dt,
        })
        add_tool_result(tc_id, json.dumps(flight_status))
```

- [ ] **Step 3: Add `send_compensation` to generate_scenario dispatch**

Add to airline generators dict:

```python
            "send_compensation": _gen_send_compensation_scenario,
```

- [ ] **Step 4: Commit**

```bash
git add structured_data_new_game.py
git commit -m "feat: add send_compensation airline scenario to SDR game"
```

---

### Task 5: Add Retail `order_selection` Scenario

Targets 4 eval SD failures (T59, T72, T74, T93). The model picks the wrong `order_id` from multiple orders. User describes which order by item name or date, model must match.

**Files:**
- Modify: `/home/ubuntu/hangook/games/structured_data_new_game.py` (insert after `_gen_no_match_scenario`)

- [ ] **Step 1: Add `_gen_order_selection_scenario` function**

```python
def _gen_order_selection_scenario(rng: random.Random) -> Dict[str, Any]:
    """Pick the correct order from multiple orders based on user description.

    User describes an order by item name. Model must find the right order_id.
    Gold action is cancel_pending_order (for pending orders) or
    return_delivered_order_items (for delivered orders).

    Matches eval failures T59, T72, T74, T93.
    """
    user_info = _gen_user(rng)
    user = user_info["user"]

    # Target order
    target_tmpl = rng.choice(PRODUCT_CATALOG)
    target_options = {k: rng.choice(v) for k, v in target_tmpl["options_template"].items()}
    target_item_id = _gen_item_id(rng)
    target_price = round(rng.uniform(20, 300), 2)
    target_product = {
        "name": target_tmpl["name"], "product_id": _gen_product_id(rng),
        "item_id": target_item_id, "price": target_price,
        "options": target_options,
    }

    action_type = rng.choice(["return", "cancel"])
    target_status = "delivered" if action_type == "return" else "pending"
    target_order = _gen_order(rng, user_info["user_id"], target_status,
                              [target_product], user_info["cc_id"])

    # 2-3 distractor orders (some may contain similar product types)
    distractor_orders = []
    for _ in range(rng.randint(2, 3)):
        d = _gen_distractor_order(rng, user_info["user_id"])
        distractor_orders.append(d)

    all_order_ids = [target_order["order_id"]] + [d["order_id"] for d in distractor_orders]
    rng.shuffle(all_order_ids)
    user["orders"] = all_order_ids

    target_desc = _describe_options(target_options)
    payment_desc, payment_id = _describe_payment(rng, user_info)

    if action_type == "return":
        user_msg = (
            f"Hi, I'm {user_info['first_name']} {user_info['last_name']}, "
            f"zip code {user_info['zip']}. I'd like to return the "
            f"{target_tmpl['name']} ({target_desc}) I ordered. "
            f"Please refund to {payment_desc}."
        )
        presentation = (
            f"I found your orders and located the {target_tmpl['name']}. "
            f"Shall I proceed with the return?"
        )
        gold_action = {
            "name": "return_delivered_order_items",
            "arguments": {
                "order_id": target_order["order_id"],
                "item_ids": [target_item_id],
                "payment_method_id": payment_id,
            },
            "compare_args": ["order_id", "item_ids", "payment_method_id"],
        }
    else:
        reason = rng.choice(["no longer needed", "ordered by mistake", "found better price"])
        user_msg = (
            f"Hi, I'm {user_info['first_name']} {user_info['last_name']}, "
            f"zip code {user_info['zip']}. I'd like to cancel the order "
            f"with the {target_tmpl['name']} ({target_desc}). "
            f"Reason: {reason}."
        )
        presentation = (
            f"I found your orders and located the pending order with the "
            f"{target_tmpl['name']}. Shall I cancel it?"
        )
        gold_action = {
            "name": "cancel_pending_order",
            "arguments": {
                "order_id": target_order["order_id"],
                "reason": reason,
            },
            "compare_args": ["order_id", "reason"],
        }

    # Build conversation with ALL orders visible in prefix
    # (model must pick the right one)
    orders_for_conv = {target_order["order_id"]: target_order}
    for d in distractor_orders:
        orders_for_conv[d["order_id"]] = d

    return {
        "domain": "retail",
        "scenario_type": "order_selection",
        "user_info": user_info,
        "user": user,
        "order": target_order,
        "distractor_orders": distractor_orders,
        "products": {},
        "user_message": user_msg,
        "presentation": presentation,
        "gold_action": gold_action,
    }
```

- [ ] **Step 2: Update `_build_retail_conversation` to handle multi-order scenarios**

In `_build_retail_conversation` (around line 1887), after the `get_order_details` call for the main order (line ~1934), add handling for distractor orders:

```python
    # Distractor orders (for order_selection scenarios)
    for d_order in scenario.get("distractor_orders", []):
        tc_id = add_tool_call("get_order_details", {"order_id": d_order["order_id"]})
        add_tool_result(tc_id, json.dumps(d_order))
```

- [ ] **Step 3: Add `order_selection` to generate_scenario dispatch**

Add to retail generators dict:

```python
            "order_selection": _gen_order_selection_scenario,
```

- [ ] **Step 4: Commit**

```bash
git add structured_data_new_game.py
git commit -m "feat: add order_selection retail scenario to SDR game"
```

---

### Task 6: Add Retail `payment_selection` Scenario

Targets 4 eval SD failures (T16, T74, T83, T99). The model picks the wrong `payment_method_id`. User describes payment by brand/last4/type, model must resolve to correct ID.

**Files:**
- Modify: `/home/ubuntu/hangook/games/structured_data_new_game.py` (insert after `_gen_order_selection_scenario`)

- [ ] **Step 1: Add `_gen_payment_selection_retail_scenario` function**

```python
def _gen_payment_selection_retail_scenario(rng: random.Random) -> Dict[str, Any]:
    """Pick the correct payment method from user details.

    User describes a payment method (e.g., 'my visa card ending in 1234').
    Model must resolve to the correct payment_method_id and use it for
    a return or exchange operation.

    Matches eval failures T16, T74, T83, T99.
    """
    user_info = _gen_user(rng)
    user = user_info["user"]

    # Ensure user has multiple payment methods for ambiguity
    if not user_info["gc_id"]:
        gc_id = _gen_payment_id(rng, "gift_card")
        user["payment_methods"][gc_id] = {
            "source": "gift_card", "id": gc_id,
            "balance": round(rng.uniform(50, 500), 2),
        }
        user_info["gc_id"] = gc_id

    if not user_info["pp_id"]:
        pp_id = _gen_payment_id(rng, "paypal")
        user["payment_methods"][pp_id] = {"source": "paypal", "id": pp_id}
        user_info["pp_id"] = pp_id

    # Target product for return
    prod_tmpl = rng.choice(PRODUCT_CATALOG)
    target_options = {k: rng.choice(v) for k, v in prod_tmpl["options_template"].items()}
    target_item_id = _gen_item_id(rng)
    target_price = round(rng.uniform(20, 300), 2)
    target_product = {
        "name": prod_tmpl["name"], "product_id": _gen_product_id(rng),
        "item_id": target_item_id, "price": target_price,
        "options": target_options,
    }

    order = _gen_order(rng, user_info["user_id"], "delivered",
                       [target_product], user_info["cc_id"])
    user["orders"] = [order["order_id"]]

    # Payment constraint — user specifies a non-obvious payment method
    constraint = rng.choice(["gift_card", "paypal", "specific_cc"])
    if constraint == "gift_card" and user_info["gc_id"]:
        target_pay_id = user_info["gc_id"]
        pay_desc = "my gift card"
    elif constraint == "paypal" and user_info["pp_id"]:
        target_pay_id = user_info["pp_id"]
        pay_desc = "my PayPal account"
    else:
        target_pay_id = user_info["cc_id"]
        pay_desc = f"my {user_info['cc_brand']} card ending in {user_info['cc_last_four']}"

    target_desc = _describe_options(target_options)
    user_msg = (
        f"Hi, I'm {user_info['first_name']} {user_info['last_name']}, "
        f"zip code {user_info['zip']}. I'd like to return the "
        f"{prod_tmpl['name']} ({target_desc}) from order {order['order_id']}. "
        f"Please refund to {pay_desc}."
    )

    presentation = (
        f"I found your order with the {prod_tmpl['name']}. "
        f"Shall I proceed with the return?"
    )

    gold_action = {
        "name": "return_delivered_order_items",
        "arguments": {
            "order_id": order["order_id"],
            "item_ids": [target_item_id],
            "payment_method_id": target_pay_id,
        },
        "compare_args": ["order_id", "item_ids", "payment_method_id"],
    }

    return {
        "domain": "retail",
        "scenario_type": "payment_selection",
        "user_info": user_info,
        "user": user,
        "order": order,
        "products": {},
        "user_message": user_msg,
        "presentation": presentation,
        "gold_action": gold_action,
    }
```

- [ ] **Step 2: Add `payment_selection` to generate_scenario dispatch**

Add to retail generators dict:

```python
            "payment_selection": _gen_payment_selection_retail_scenario,
```

- [ ] **Step 3: Commit**

```bash
git add structured_data_new_game.py
git commit -m "feat: add payment_selection retail scenario to SDR game"
```

---

### Task 7: Update `_score()` for `book_reservation` Args

The `_score()` method needs to handle `passengers` and `payment_methods` (arrays of dicts) for `book_reservation`. Currently the generic list handler uses `sorted(str(x))` which works for simple lists but may fail on dicts with key ordering issues.

**Files:**
- Modify: `/home/ubuntu/hangook/games/structured_data_new_game.py` (in `_score` method, ~line 2191-2244)

- [ ] **Step 1: Add `passengers` and `payment_methods` handling to `_score()`**

In the `_score` method, after the `flights` special case (around line 2219) and before the `item_ids`/`new_item_ids` case, add:

```python
            elif key == "passengers":
                # Compare as sets of (first_name, last_name, dob) tuples
                exp_set = set()
                for p in (expected or []):
                    if isinstance(p, dict):
                        exp_set.add((p.get("first_name", ""), p.get("last_name", ""), p.get("dob", "")))
                    else:
                        exp_set.add(str(p))
                act_set = set()
                for p in (actual or []):
                    if isinstance(p, dict):
                        act_set.add((p.get("first_name", ""), p.get("last_name", ""), p.get("dob", "")))
                    else:
                        act_set.add(str(p))
                if exp_set != act_set:
                    return 0.0, f"Wrong {key}: {len(actual or [])} passengers (expected {len(expected or [])})"
            elif key == "payment_methods":
                # Compare as sets of (payment_id, amount) tuples
                exp_set = set()
                for p in (expected or []):
                    if isinstance(p, dict):
                        exp_set.add((p.get("payment_id", ""), round(float(p.get("amount", 0)), 2)))
                    else:
                        exp_set.add(str(p))
                act_set = set()
                for p in (actual or []):
                    if isinstance(p, dict):
                        act_set.add((p.get("payment_id", ""), round(float(p.get("amount", 0)), 2)))
                    else:
                        act_set.add(str(p))
                if exp_set != act_set:
                    return 0.0, f"Wrong {key}: {actual} (expected {expected})"
```

- [ ] **Step 2: Commit**

```bash
git add structured_data_new_game.py
git commit -m "feat: add passengers/payment_methods comparison to _score()"
```

---

### Task 8: Verify and Test

- [ ] **Step 1: Run the built-in verification**

```bash
cd /home/ubuntu/hangook/games && conda run -n games python structured_data_new_game.py
```

Expected: `Perfect agent: 200/200 (100.0%)` — all gold actions should produce reward=1.0.

Check the distribution output shows all new scenario types with reasonable counts.

- [ ] **Step 2: Spot-check new scenario types**

```bash
conda run -n games python -c "
import json
from structured_data_new_game import generate_scenario
for seed in range(500):
    s = generate_scenario(seed)
    if s['scenario_type'] in ('book_flight', 'update_baggages', 'send_compensation', 'order_selection', 'payment_selection'):
        print(f'seed={seed} {s[\"domain\"]}/{s[\"scenario_type\"]}')
        g = s['gold_action']
        print(f'  gold: {g[\"name\"]}({list(g[\"arguments\"].keys())})')
        if seed > 100:
            break
"
```

Expected: All 5 new types appear and produce valid gold actions.

- [ ] **Step 3: Verify format matches eval**

Check that the conversation prefix for new scenarios follows the same pattern as eval:
`assistant:greeting → user:request → tool_call:auth → tool_result → tool_call:lookup → tool_result → ... → assistant:presentation → user:confirmation`

- [ ] **Step 4: Final commit**

```bash
git add structured_data_new_game.py
git commit -m "feat: SDR game v5 — 5 new scenarios covering all SD reasoning eval failures"
```
