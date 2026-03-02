"""Synthetic database generation for adversarial policy game.

Generates fresh user/reservation/order/product data per seed so training
never uses real tau2-bench evaluation entities. Schema matches tau2-bench
exactly so existing tool implementations work unchanged.
"""

import random
import copy
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# =====================================================================
# Name pools (NOT from tau2-bench — generic names)
# =====================================================================
FIRST_NAMES = [
    "Alex", "Blake", "Casey", "Drew", "Ellis", "Finley", "Gray", "Harper",
    "Indigo", "Jordan", "Kai", "Lane", "Morgan", "Noel", "Oakley", "Parker",
    "Quinn", "Reese", "Sage", "Taylor", "Uri", "Val", "Wren", "Xen", "Yael", "Zion",
    "Ari", "Bay", "Cedar", "Dax", "Ember", "Frost", "Glen", "Haven",
    "Iris", "Jade", "Kit", "Lark", "Mira", "Nash", "Onyx", "Pax",
    "Rain", "Shea", "Teal", "Uma", "Vex", "Wade", "Xyla", "Yuki",
]

LAST_NAMES = [
    "Adler", "Banks", "Cole", "Drake", "Engel", "Frost", "Grant", "Hayes",
    "Irwin", "James", "Klein", "Lowe", "Marsh", "North", "Owen", "Price",
    "Reed", "Shaw", "Thorn", "Vale", "West", "York", "Zane", "Stone",
    "Brook", "Clark", "Dunn", "Ellis", "Ford", "Gray", "Hart", "Knox",
    "Lane", "Moon", "Nash", "Pace", "Ross", "Snow", "Troy", "Webb",
]

STREETS = [
    "Oak Lane", "Pine Road", "Elm Street", "Cedar Avenue", "Maple Drive",
    "Birch Court", "Willow Way", "Aspen Place", "Poplar Boulevard",
    "Spruce Circle", "Chestnut Lane", "Walnut Street", "Hickory Road",
    "Sycamore Drive", "Magnolia Avenue", "Redwood Court",
]

CITIES_STATES_ZIPS = [
    ("Portland", "OR", "97201"), ("Austin", "TX", "78701"),
    ("Denver", "CO", "80202"), ("Nashville", "TN", "37201"),
    ("Raleigh", "NC", "27601"), ("Tampa", "FL", "33602"),
    ("Tucson", "AZ", "85701"), ("Boise", "ID", "83702"),
    ("Madison", "WI", "53703"), ("Richmond", "VA", "23219"),
    ("Salem", "OR", "97301"), ("Omaha", "NE", "68102"),
    ("Fresno", "CA", "93721"), ("Tulsa", "OK", "74103"),
    ("Mesa", "AZ", "85201"), ("Oakland", "CA", "94612"),
]

# Airports for synthetic flights
AIRPORTS = [
    "SFO", "JFK", "LAX", "ORD", "DFW", "DEN", "SEA", "ATL", "MIA", "BOS",
    "PHX", "IAH", "LAS", "MCO", "EWR", "CLT", "MSP", "DTW", "PHL", "LGA",
]

PRODUCT_CATALOG = [
    {"name": "Water Bottle", "options_template": {"capacity": ["500ml", "750ml", "1000ml"], "material": ["plastic", "stainless steel", "glass"], "color": ["blue", "red", "black", "green"]}},
    {"name": "T-Shirt", "options_template": {"size": ["S", "M", "L", "XL"], "color": ["blue", "red", "black", "white"], "material": ["cotton", "polyester"], "style": ["crew neck", "v-neck"]}},
    {"name": "Desk Lamp", "options_template": {"color": ["black", "white", "silver"], "brightness": ["low", "medium", "high"], "style": ["modern", "classic"]}},
    {"name": "Backpack", "options_template": {"size": ["small", "medium", "large"], "color": ["black", "blue", "gray", "green"], "material": ["nylon", "canvas"]}},
    {"name": "Headphones", "options_template": {"type": ["over-ear", "on-ear", "in-ear"], "color": ["black", "white", "blue"], "connectivity": ["wired", "wireless"]}},
    {"name": "Notebook", "options_template": {"size": ["A4", "A5", "pocket"], "cover": ["hardcover", "softcover"], "pages": ["100", "200", "300"]}},
    {"name": "Coffee Mug", "options_template": {"size": ["small", "medium", "large"], "material": ["ceramic", "stainless steel"], "color": ["white", "black", "blue"]}},
    {"name": "Yoga Mat", "options_template": {"thickness": ["4mm", "6mm", "8mm"], "color": ["purple", "blue", "green", "black"], "material": ["PVC", "rubber", "TPE"]}},
    {"name": "Phone Case", "options_template": {"color": ["clear", "black", "blue", "red"], "material": ["silicone", "leather", "plastic"], "type": ["slim", "rugged"]}},
    {"name": "Sunglasses", "options_template": {"lens": ["polarized", "non-polarized"], "frame": ["metal", "plastic"], "color": ["black", "brown", "blue"]}},
]


# =====================================================================
# Synthetic ID generators
# =====================================================================

def _gen_user_id(rng: random.Random) -> str:
    first = rng.choice(FIRST_NAMES).lower()
    last = rng.choice(LAST_NAMES).lower()
    num = rng.randint(1000, 9999)
    return f"{first}_{last}_{num}"

def _gen_reservation_id(rng: random.Random) -> str:
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(chars) for _ in range(6))

def _gen_order_id(rng: random.Random) -> str:
    return f"#W{rng.randint(1000000, 9999999)}"

def _gen_flight_number(rng: random.Random) -> str:
    return f"SYN{rng.randint(100, 999)}"

def _gen_product_id(rng: random.Random) -> str:
    return str(rng.randint(1000000000, 9999999999))

def _gen_item_id(rng: random.Random) -> str:
    return str(rng.randint(1000000000, 9999999999))

def _gen_payment_id(rng: random.Random, source: str) -> str:
    num = rng.randint(1000000, 9999999)
    return f"{source}_{num}"

def _gen_email(first: str, last: str, rng: random.Random) -> str:
    num = rng.randint(100, 9999)
    domain = rng.choice(["example.com", "test.net", "mail.org"])
    return f"{first.lower()}.{last.lower()}{num}@{domain}"

def _gen_address(rng: random.Random) -> Dict[str, str]:
    num = rng.randint(100, 999)
    street = rng.choice(STREETS)
    city, state, zip_code = rng.choice(CITIES_STATES_ZIPS)
    return {
        "address1": f"{num} {street}",
        "address2": "",
        "city": city,
        "state": state,
        "zip": zip_code,
        "country": "USA",
    }

def _gen_dob(rng: random.Random) -> str:
    year = rng.randint(1955, 2003)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


# =====================================================================
# Airline synthetic generation
# =====================================================================

def generate_airline_episode(
    rng: random.Random,
    criteria: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Generate a synthetic airline user + reservation matching criteria.

    Returns dict with: user, reservation, reservation_id, user_id, flights_db
    matching the same shape as sample_airline_user() from the real DB.
    """
    cabin = criteria.get("cabin", rng.choice(["economy", "basic_economy", "business"]))
    insurance = criteria.get("insurance", rng.choice(["yes", "no"]))
    is_recent = criteria.get("is_recent")
    membership_options = criteria.get("membership")
    if isinstance(membership_options, str):
        membership_options = [membership_options]
    num_passengers_min = criteria.get("num_passengers_min", 1)

    # Generate user
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    uid = f"{first.lower()}_{last.lower()}_{rng.randint(1000, 9999)}"
    membership = rng.choice(membership_options) if membership_options else rng.choice(["regular", "silver", "gold"])

    # Payment methods
    payment_methods = {}
    cc_id = _gen_payment_id(rng, "credit_card")
    payment_methods[cc_id] = {
        "source": "credit_card",
        "id": cc_id,
        "brand": rng.choice(["visa", "mastercard", "amex"]),
        "last_four": str(rng.randint(1000, 9999)),
    }
    # Add gift card
    gc_id = _gen_payment_id(rng, "gift_card")
    payment_methods[gc_id] = {
        "source": "gift_card",
        "id": gc_id,
        "amount": round(rng.uniform(50, 500), 2),
    }
    # Maybe add certificate
    if rng.random() < 0.4:
        cert_id = _gen_payment_id(rng, "certificate")
        payment_methods[cert_id] = {
            "source": "certificate",
            "id": cert_id,
            "amount": rng.choice([50, 100, 200, 250, 500]),
        }

    has_unlimited = criteria.get("has_unlimited_payment", False)
    # credit_card already satisfies this

    # Generate reservation
    res_id = _gen_reservation_id(rng)
    origin = rng.choice(AIRPORTS)
    dest = rng.choice([a for a in AIRPORTS if a != origin])
    flight_type = rng.choice(["one_way", "round_trip"])

    # Generate flights
    base_date = datetime(2024, 5, rng.randint(17, 28))
    flights = []
    flights_db = {}

    price_map = {
        "basic_economy": (40, 150),
        "economy": (80, 300),
        "business": (300, 900),
    }
    lo, hi = price_map.get(cabin, (80, 300))

    # Outbound
    fn1 = _gen_flight_number(rng)
    p1 = int(round(rng.uniform(lo, hi)))
    flights.append({
        "flight_number": fn1,
        "origin": origin,
        "destination": dest if flight_type == "one_way" else rng.choice([a for a in AIRPORTS if a not in (origin, dest)]),
        "date": base_date.strftime("%Y-%m-%d"),
        "price": p1,
    })
    if flight_type == "one_way" and rng.random() < 0.4:
        # one-stop
        mid = flights[0]["destination"]
        flights[0]["destination"] = mid
        fn2 = _gen_flight_number(rng)
        p2 = int(round(rng.uniform(lo, hi)))
        flights.append({
            "flight_number": fn2,
            "origin": mid,
            "destination": dest,
            "date": base_date.strftime("%Y-%m-%d"),
            "price": p2,
        })

    if flight_type == "round_trip":
        ret_date = base_date + timedelta(days=rng.randint(2, 7))
        fn_r = _gen_flight_number(rng)
        p_r = int(round(rng.uniform(lo, hi)))
        flights.append({
            "flight_number": fn_r,
            "origin": dest,
            "destination": origin,
            "date": ret_date.strftime("%Y-%m-%d"),
            "price": p_r,
        })

    # Build flights_db entries for reservation flights
    for fl in flights:
        fn = fl["flight_number"]
        dep_h = rng.randint(6, 22)
        dep_m = rng.choice([0, 15, 30, 45])
        arr_h = min(dep_h + rng.randint(1, 5), 23)
        arr_m = rng.choice([0, 15, 30, 45])
        flights_db[fn] = {
            "flight_number": fn,
            "origin": fl["origin"],
            "destination": fl["destination"],
            "scheduled_departure_time_est": f"{dep_h:02d}:{dep_m:02d}:00",
            "scheduled_arrival_time_est": f"{arr_h:02d}:{arr_m:02d}:00",
            "dates": {
                fl["date"]: {
                    "status": "available",
                    "available_seats": {
                        "basic_economy": rng.randint(2, 15),
                        "economy": rng.randint(2, 15),
                        "business": rng.randint(1, 8),
                    },
                    "prices": {
                        "basic_economy": int(round(rng.uniform(40, 150))),
                        "economy": int(round(rng.uniform(80, 300))),
                        "business": int(round(rng.uniform(300, 900))),
                    },
                }
            },
        }

    # Generate additional searchable flights on nearby dates for same routes.
    # This gives the model alternative flights to find when modifying reservations.
    seen_routes = set()
    for fl in flights:
        route_key = (fl["origin"], fl["destination"])
        if route_key in seen_routes:
            continue
        seen_routes.add(route_key)
        for day_offset in [-3, -1, 1, 3, 5, 7]:
            alt_date = base_date + timedelta(days=day_offset)
            alt_date_str = alt_date.strftime("%Y-%m-%d")
            alt_fn = _gen_flight_number(rng)
            alt_dep_h = rng.randint(6, 22)
            alt_dep_m = rng.choice([0, 15, 30, 45])
            alt_arr_h = min(alt_dep_h + rng.randint(1, 5), 23)
            alt_arr_m = rng.choice([0, 15, 30, 45])
            flights_db[alt_fn] = {
                "flight_number": alt_fn,
                "origin": fl["origin"],
                "destination": fl["destination"],
                "scheduled_departure_time_est": f"{alt_dep_h:02d}:{alt_dep_m:02d}:00",
                "scheduled_arrival_time_est": f"{alt_arr_h:02d}:{alt_arr_m:02d}:00",
                "dates": {
                    alt_date_str: {
                        "status": "available",
                        "available_seats": {
                            "basic_economy": rng.randint(2, 15),
                            "economy": rng.randint(2, 15),
                            "business": rng.randint(1, 8),
                        },
                        "prices": {
                            "basic_economy": int(round(rng.uniform(40, 150))),
                            "economy": int(round(rng.uniform(80, 300))),
                            "business": int(round(rng.uniform(300, 900))),
                        },
                    }
                },
            }

    # Add extra flights on some existing alternative dates (so multiple flights per date)
    # This enables "cheapest flight on date X" scenarios
    existing_alt_dates = {}
    for fn, fd in flights_db.items():
        route = (fd["origin"], fd["destination"])
        for d in fd.get("dates", {}):
            existing_alt_dates.setdefault((route, d), []).append(fn)

    for (route, date_str), fns in list(existing_alt_dates.items()):
        if len(fns) >= 2:
            continue  # Already has multiple
        if rng.random() < 0.4:  # 40% chance to add a second flight on this date
            extra_fn = _gen_flight_number(rng)
            extra_dep_h = rng.randint(6, 22)
            extra_dep_m = rng.choice([0, 15, 30, 45])
            extra_arr_h = min(extra_dep_h + rng.randint(1, 5), 23)
            extra_arr_m = rng.choice([0, 15, 30, 45])
            flights_db[extra_fn] = {
                "flight_number": extra_fn,
                "origin": route[0],
                "destination": route[1],
                "scheduled_departure_time_est": f"{extra_dep_h:02d}:{extra_dep_m:02d}:00",
                "scheduled_arrival_time_est": f"{extra_arr_h:02d}:{extra_arr_m:02d}:00",
                "dates": {
                    date_str: {
                        "status": "available",
                        "available_seats": {
                            "basic_economy": rng.randint(2, 15),
                            "economy": rng.randint(2, 15),
                            "business": rng.randint(1, 8),
                        },
                        "prices": {
                            "basic_economy": int(round(rng.uniform(40, 150))),
                            "economy": int(round(rng.uniform(80, 300))),
                            "business": int(round(rng.uniform(300, 900))),
                        },
                    }
                },
            }

    # Passengers
    num_passengers = max(num_passengers_min, rng.randint(1, 3))
    passengers = []
    for _ in range(num_passengers):
        passengers.append({
            "first_name": rng.choice(FIRST_NAMES),
            "last_name": rng.choice(LAST_NAMES),
            "dob": _gen_dob(rng),
        })

    # Booking time
    if is_recent is not None:
        if is_recent:
            created = "2024-05-15T" + f"{rng.randint(0, 14):02d}:{rng.randint(0, 59):02d}:00"
        else:
            day = rng.randint(1, 13)
            created = f"2024-05-{day:02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00"
    else:
        day = rng.randint(1, 14)
        created = f"2024-05-{day:02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00"

    total_price = sum(fl["price"] for fl in flights) * num_passengers
    pay_id = cc_id
    total_bags = rng.randint(0, 3)
    nonfree = max(0, total_bags - (1 if cabin == "economy" else 2 if cabin == "business" else 0))

    total_price = int(total_price)
    reservation = {
        "reservation_id": res_id,
        "user_id": uid,
        "origin": origin,
        "destination": dest,
        "flight_type": flight_type,
        "cabin": cabin,
        "flights": flights,
        "passengers": passengers,
        "payment_history": [
            {"payment_id": pay_id, "amount": total_price},
        ],
        "created_at": created,
        "total_baggages": total_bags,
        "nonfree_baggages": nonfree,
        "insurance": insurance,
        "status": None,  # active
    }

    user = {
        "user_id": uid,
        "name": {"first_name": first, "last_name": last},
        "address": _gen_address(rng),
        "email": _gen_email(first, last, rng),
        "dob": _gen_dob(rng),
        "payment_methods": payment_methods,
        "saved_passengers": passengers[:1],
        "membership": membership,
        "reservations": [res_id],
    }

    return {
        "user": user,
        "reservation": reservation,
        "reservation_id": res_id,
        "user_id": uid,
        "flights_db": flights_db,
    }


def generate_airline_multi_reservations(
    rng: random.Random,
    criteria: Dict[str, Any],
    min_reservations: int = 2,
) -> Optional[Dict[str, Any]]:
    """Generate a synthetic user with multiple reservations."""
    membership_options = criteria.get("membership")
    if isinstance(membership_options, str):
        membership_options = [membership_options]

    # Generate base user
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    uid = f"{first.lower()}_{last.lower()}_{rng.randint(1000, 9999)}"
    membership = rng.choice(membership_options) if membership_options else rng.choice(["regular", "silver", "gold"])

    payment_methods = {}
    cc_id = _gen_payment_id(rng, "credit_card")
    payment_methods[cc_id] = {
        "source": "credit_card", "id": cc_id,
        "brand": rng.choice(["visa", "mastercard"]),
        "last_four": str(rng.randint(1000, 9999)),
    }
    gc_id = _gen_payment_id(rng, "gift_card")
    payment_methods[gc_id] = {
        "source": "gift_card", "id": gc_id,
        "amount": round(rng.uniform(100, 800), 2),
    }

    user = {
        "user_id": uid,
        "name": {"first_name": first, "last_name": last},
        "address": _gen_address(rng),
        "email": _gen_email(first, last, rng),
        "dob": _gen_dob(rng),
        "payment_methods": payment_methods,
        "saved_passengers": [],
        "membership": membership,
        "reservations": [],
    }

    reservations = []
    reservation_ids = []
    all_flights_db = {}

    cabins = ["basic_economy", "economy", "business"]
    for i in range(min_reservations):
        cabin = rng.choice(cabins)
        ins = rng.choice(["yes", "no"])
        ep = generate_airline_episode(rng, {
            "cabin": cabin,
            "insurance": ins,
            "is_recent": rng.choice([True, False]),
        })
        res = ep["reservation"]
        res["user_id"] = uid
        res["payment_history"][0]["payment_id"] = cc_id
        reservations.append(res)
        reservation_ids.append(res["reservation_id"])
        all_flights_db.update(ep["flights_db"])

    user["reservations"] = reservation_ids

    return {
        "user": user,
        "reservations": reservations,
        "reservation_ids": reservation_ids,
        "user_id": uid,
        "flights_db": all_flights_db,
    }


# =====================================================================
# Retail synthetic generation
# =====================================================================

def generate_retail_episode(
    rng: random.Random,
    criteria: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Generate a synthetic retail user + order matching criteria.

    Returns dict with: user, order, order_id, user_id, products_db
    """
    status = criteria.get("status", rng.choice(["pending", "delivered"]))
    min_items = criteria.get("min_items", 1)
    has_gift_card = criteria.get("has_gift_card", False)
    has_multiple_payment_types = criteria.get("has_multiple_payment_types", False)

    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    uid = f"{first.lower()}_{last.lower()}_{rng.randint(1000, 9999)}"

    # Payment methods
    payment_methods = {}
    cc_id = _gen_payment_id(rng, "credit_card")
    payment_methods[cc_id] = {
        "source": "credit_card", "id": cc_id,
        "brand": rng.choice(["visa", "mastercard", "amex"]),
        "last_four": str(rng.randint(1000, 9999)),
    }

    if has_gift_card or rng.random() < 0.5:
        gc_id = _gen_payment_id(rng, "gift_card")
        payment_methods[gc_id] = {
            "source": "gift_card", "id": gc_id,
            "balance": round(rng.uniform(50, 500), 2),
        }

    if has_multiple_payment_types or rng.random() < 0.4:
        pp_id = _gen_payment_id(rng, "paypal")
        payment_methods[pp_id] = {"source": "paypal", "id": pp_id}

    # Generate order
    order_id = _gen_order_id(rng)
    num_items = max(min_items, rng.randint(1, 4))

    items = []
    products_db = {}
    used_products = set()

    for _ in range(num_items):
        # Pick a product type we haven't used yet in this order
        available = [p for p in PRODUCT_CATALOG if p["name"] not in used_products]
        if not available:
            available = PRODUCT_CATALOG
        prod_tmpl = rng.choice(available)
        used_products.add(prod_tmpl["name"])

        prod_id = _gen_product_id(rng)
        item_id = _gen_item_id(rng)
        price = round(rng.uniform(15, 350), 2)

        # Generate options for this item
        options = {}
        for opt_name, opt_values in prod_tmpl["options_template"].items():
            options[opt_name] = rng.choice(opt_values)

        items.append({
            "name": prod_tmpl["name"],
            "product_id": prod_id,
            "item_id": item_id,
            "price": price,
            "options": options,
        })

        # Generate product catalog entry with variants
        variants = {}
        # The ordered item variant
        variants[item_id] = {
            "item_id": item_id,
            "options": copy.deepcopy(options),
            "available": True,
            "price": price,
        }
        # Generate 3-6 other variants
        for _ in range(rng.randint(3, 6)):
            v_id = _gen_item_id(rng)
            v_opts = {}
            for opt_name, opt_values in prod_tmpl["options_template"].items():
                v_opts[opt_name] = rng.choice(opt_values)
            v_price = round(rng.uniform(15, 350), 2)
            variants[v_id] = {
                "item_id": v_id,
                "options": v_opts,
                "available": rng.random() > 0.2,
                "price": v_price,
            }
        products_db[prod_id] = {
            "name": prod_tmpl["name"],
            "product_id": prod_id,
            "variants": variants,
        }

    total = sum(item["price"] for item in items)
    pay_id = list(payment_methods.keys())[0]

    order = {
        "order_id": order_id,
        "user_id": uid,
        "address": _gen_address(rng),
        "items": items,
        "status": status,
        "fulfillments": [{"tracking_id": [str(rng.randint(100000000000, 999999999999))],
                          "item_ids": [i["item_id"] for i in items]}] if status == "delivered" else [],
        "payment_history": [
            {"transaction_type": "payment", "amount": round(total, 2),
             "payment_method_id": pay_id},
        ],
        "cancel_reason": None,
        "exchange_items": None,
        "exchange_new_items": None,
        "exchange_payment_method_id": None,
        "exchange_price_difference": None,
        "return_items": None,
        "return_payment_method_id": None,
    }

    user = {
        "user_id": uid,
        "name": {"first_name": first, "last_name": last},
        "address": _gen_address(rng),
        "email": _gen_email(first, last, rng),
        "payment_methods": payment_methods,
        "orders": [order_id],
    }

    return {
        "user": user,
        "order": order,
        "order_id": order_id,
        "user_id": uid,
        "products_db": products_db,
    }


def generate_retail_multi_orders(
    rng: random.Random,
    criteria: Dict[str, Any],
    order_specs: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a synthetic retail user with multiple orders.

    Args:
        rng: Seeded random instance.
        criteria: Shared criteria for the user (has_gift_card, has_multiple_payment_types, etc.).
        order_specs: List of per-order specs. Each spec is a dict with:
            - "status": "pending" or "delivered" (default: random)
            - "min_items": minimum items in order (default: 1)
            Any other keys are ignored.
            If None, generates 2 orders with random statuses.

    Returns dict with: user, orders (list), order_ids (list), user_id, products_db
    """
    if order_specs is None:
        order_specs = [
            {"status": rng.choice(["pending", "delivered"])},
            {"status": rng.choice(["pending", "delivered"])},
        ]

    has_gift_card = criteria.get("has_gift_card", False)
    has_multiple_payment_types = criteria.get("has_multiple_payment_types", False)

    # Generate shared user identity
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    uid = f"{first.lower()}_{last.lower()}_{rng.randint(1000, 9999)}"

    # Shared payment methods
    payment_methods = {}
    cc_id = _gen_payment_id(rng, "credit_card")
    payment_methods[cc_id] = {
        "source": "credit_card", "id": cc_id,
        "brand": rng.choice(["visa", "mastercard", "amex"]),
        "last_four": str(rng.randint(1000, 9999)),
    }

    if has_gift_card or rng.random() < 0.5:
        gc_id = _gen_payment_id(rng, "gift_card")
        payment_methods[gc_id] = {
            "source": "gift_card", "id": gc_id,
            "balance": round(rng.uniform(50, 500), 2),
        }

    if has_multiple_payment_types or rng.random() < 0.4:
        pp_id = _gen_payment_id(rng, "paypal")
        payment_methods[pp_id] = {"source": "paypal", "id": pp_id}

    # Track used product names globally to avoid overlap across orders
    global_used_products: set = set()
    orders: List[Dict[str, Any]] = []
    order_ids: List[str] = []
    merged_products_db: Dict[str, Any] = {}
    pay_id = list(payment_methods.keys())[0]

    for spec in order_specs:
        status = spec.get("status", rng.choice(["pending", "delivered"]))
        min_items = spec.get("min_items", 1)

        order_id = _gen_order_id(rng)
        num_items = max(min_items, rng.randint(1, 4))

        items: List[Dict[str, Any]] = []
        order_products_db: Dict[str, Any] = {}

        for _ in range(num_items):
            # Pick a product type not yet used in ANY order
            available = [p for p in PRODUCT_CATALOG if p["name"] not in global_used_products]
            if not available:
                # Fallback: allow reuse if catalog exhausted
                available = PRODUCT_CATALOG
            prod_tmpl = rng.choice(available)
            global_used_products.add(prod_tmpl["name"])

            prod_id = _gen_product_id(rng)
            item_id = _gen_item_id(rng)
            price = round(rng.uniform(15, 350), 2)

            # Generate options for this item
            options = {}
            for opt_name, opt_values in prod_tmpl["options_template"].items():
                options[opt_name] = rng.choice(opt_values)

            items.append({
                "name": prod_tmpl["name"],
                "product_id": prod_id,
                "item_id": item_id,
                "price": price,
                "options": options,
            })

            # Generate product catalog entry with variants
            variants = {}
            variants[item_id] = {
                "item_id": item_id,
                "options": copy.deepcopy(options),
                "available": True,
                "price": price,
            }
            for _ in range(rng.randint(3, 6)):
                v_id = _gen_item_id(rng)
                v_opts = {}
                for opt_name, opt_values in prod_tmpl["options_template"].items():
                    v_opts[opt_name] = rng.choice(opt_values)
                v_price = round(rng.uniform(15, 350), 2)
                variants[v_id] = {
                    "item_id": v_id,
                    "options": v_opts,
                    "available": rng.random() > 0.2,
                    "price": v_price,
                }
            order_products_db[prod_id] = {
                "name": prod_tmpl["name"],
                "product_id": prod_id,
                "variants": variants,
            }

        total = sum(item["price"] for item in items)

        order = {
            "order_id": order_id,
            "user_id": uid,
            "address": _gen_address(rng),
            "items": items,
            "status": status,
            "fulfillments": [{"tracking_id": [str(rng.randint(100000000000, 999999999999))],
                              "item_ids": [i["item_id"] for i in items]}] if status == "delivered" else [],
            "payment_history": [
                {"transaction_type": "payment", "amount": round(total, 2),
                 "payment_method_id": pay_id},
            ],
            "cancel_reason": None,
            "exchange_items": None,
            "exchange_new_items": None,
            "exchange_payment_method_id": None,
            "exchange_price_difference": None,
            "return_items": None,
            "return_payment_method_id": None,
        }

        orders.append(order)
        order_ids.append(order_id)
        merged_products_db.update(order_products_db)

    user = {
        "user_id": uid,
        "name": {"first_name": first, "last_name": last},
        "address": _gen_address(rng),
        "email": _gen_email(first, last, rng),
        "payment_methods": payment_methods,
        "orders": order_ids,
    }

    return {
        "user": user,
        "orders": orders,
        "order_ids": order_ids,
        "user_id": uid,
        "products_db": merged_products_db,
    }


# =====================================================================
# Build full Pydantic-compatible DB from synthetic data
# =====================================================================

def build_airline_db(user: Dict, reservations: List[Dict], flights_db: Dict) -> Dict[str, Any]:
    """Build a minimal airline DB dict matching FlightDB schema."""
    res_map = {}
    for r in (reservations if isinstance(reservations, list) else [reservations]):
        res_map[r["reservation_id"]] = r
    return {
        "users": {user["user_id"]: user},
        "reservations": res_map,
        "flights": flights_db,
    }


def build_retail_db(user: Dict, orders: List[Dict], products_db: Dict) -> Dict[str, Any]:
    """Build a minimal retail DB dict matching RetailDB schema."""
    order_map = {}
    for o in (orders if isinstance(orders, list) else [orders]):
        order_map[o["order_id"]] = o
    return {
        "users": {user["user_id"]: user},
        "orders": order_map,
        "products": products_db,
    }
