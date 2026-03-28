"""Tau2-Bench Synthetic Game: GRPO-trainable wrapper with per-episode synthetic data.

Same mechanics as tau2_bench_game.py but generates a fresh synthetic DB and task
each episode, preventing memorization and maximizing training data diversity.

164 real tasks (50 airline + 114 retail) -> infinite synthetic tasks.
"""

import json
import random
import re
import string
import pathlib
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure tau2-bench src is importable
# ---------------------------------------------------------------------------
_GAME_DIR = pathlib.Path(__file__).resolve().parent
_TAU2_SRC = _GAME_DIR / "tau2-bench" / "src"
_TAU2_DATA = _GAME_DIR / "tau2-bench" / "data" / "tau2"
if str(_TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(_TAU2_SRC))

from tau2.domains.airline.data_model import (
    FlightDB, Flight, FlightDateStatusAvailable, FlightDateStatusDelayed,
    FlightDateStatusLanded,
    User as AirlineUser, Name, Address,
    Reservation, ReservationFlight, Payment, Passenger,
    CreditCard as AirlineCreditCard, GiftCard as AirlineGiftCard,
)
from tau2.domains.retail.data_model import (
    RetailDB, Product, Variant,
    User as RetailUser, UserName, UserAddress,
    Order, OrderItem, OrderFullfilment, OrderPayment,
    CreditCard as RetailCreditCard, GiftCard as RetailGiftCard, Paypal,
)
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.retail.tools import RetailTools
from tau2.environment.environment import Environment
from tau2.data_model.tasks import (
    Task, UserScenario, StructuredUserInstructions, Description,
    EvaluationCriteria, Action,
)
from tau2.data_model.message import ToolCall

# Reuse cached data from tau2_bench_game (avoids re-loading from disk)
from tau2_bench_game import (
    SYSTEM_PROMPT,
    extract_action,
    _airline_policy,
    _retail_policy,
    _airline_tool_schemas,
    _retail_tool_schemas,
    _user_sim_guidelines,
    _parse_tool_call,
    _is_user_stop,
    _strip_stop_tokens,
    _strip_descriptions,
)


# =========================================================================
# Data pools
# =========================================================================

_FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
    "Lucas", "Harper", "Henry", "Evelyn", "Alexander", "Luna", "Daniel",
    "Chloe", "Michael", "Penelope", "Sebastian", "Layla", "Jack", "Riley",
    "Owen", "Zoey", "Theodore", "Nora", "Aiden", "Lily", "Samuel", "Eleanor",
    "Ryan", "Hannah", "Leo", "Aria", "Nathan", "Grace", "Caleb", "Zoe",
    "Omar", "Yuki", "Raj", "Mei", "Carlos", "Sofia",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Kim", "Patel", "Chen",
    "Rossi",
]

_STREETS = [
    "Main Street", "Oak Avenue", "Maple Drive", "Cedar Lane", "Pine Road",
    "Elm Street", "Park Avenue", "Lake Drive", "River Road", "Hill Street",
    "Sunset Boulevard", "Forest Lane", "Meadow Drive", "Spring Road",
    "Valley Street", "Garden Avenue", "Brook Lane", "Ridge Drive",
    "Willow Road", "Birch Street",
]

_CITIES_STATES = [
    ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
    ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"), ("Philadelphia", "PA", "19101"),
    ("San Antonio", "TX", "78201"), ("San Diego", "CA", "92101"),
    ("Dallas", "TX", "75201"), ("San Jose", "CA", "95101"),
    ("Austin", "TX", "78701"), ("Jacksonville", "FL", "32099"),
    ("Fort Worth", "TX", "76101"), ("Columbus", "OH", "43085"),
    ("Charlotte", "NC", "28201"), ("Indianapolis", "IN", "46201"),
    ("San Francisco", "CA", "94101"), ("Seattle", "WA", "98101"),
    ("Denver", "CO", "80201"), ("Boston", "MA", "02101"),
]

_AIRPORTS = [
    "SFO", "JFK", "LAX", "ORD", "DFW", "SEA", "ATL", "BOS", "MIA", "IAH",
    "DEN", "PHX", "PHL", "LGA", "EWR", "CLT", "MSP", "DTW", "FLL", "BWI",
]

_PRODUCT_TYPES = {
    "T-Shirt": {"color": ["red", "blue", "green", "black", "white", "purple", "navy"],
                "size": ["XS", "S", "M", "L", "XL"],
                "material": ["cotton", "polyester", "blend"]},
    "Laptop": {"screen_size": ["13 inch", "14 inch", "15 inch", "17 inch"],
               "RAM": ["8GB", "16GB", "32GB"],
               "storage": ["256GB SSD", "512GB SSD", "1TB SSD"]},
    "Headphones": {"type": ["over-ear", "in-ear", "on-ear"],
                   "connectivity": ["wired", "wireless", "bluetooth"],
                   "color": ["black", "white", "silver", "red"]},
    "Water Bottle": {"capacity": ["500ml", "750ml", "1000ml"],
                     "material": ["stainless steel", "glass", "plastic"],
                     "color": ["blue", "green", "silver", "black"]},
    "Backpack": {"size": ["small", "medium", "large"],
                 "color": ["black", "navy", "gray", "green", "red"],
                 "material": ["nylon", "canvas", "polyester"]},
    "Running Shoes": {"size": ["7", "8", "9", "10", "11", "12"],
                      "color": ["black", "white", "blue", "red"],
                      "material": ["mesh", "leather", "synthetic"]},
    "Watch": {"style": ["analog", "digital", "smart"],
              "band": ["leather", "metal", "silicone"],
              "color": ["black", "silver", "gold"]},
    "Sunglasses": {"style": ["aviator", "wayfarer", "round", "sport"],
                   "color": ["black", "brown", "blue"],
                   "lens": ["polarized", "non-polarized", "mirrored"]},
    "Desk Lamp": {"style": ["LED", "fluorescent", "halogen"],
                  "color": ["white", "black", "silver"],
                  "brightness": ["low", "medium", "high", "adjustable"]},
    "Coffee Maker": {"type": ["drip", "espresso", "french press", "pod"],
                     "capacity": ["single", "4 cup", "8 cup", "12 cup"],
                     "color": ["black", "silver", "white"]},
    "Yoga Mat": {"thickness": ["3mm", "5mm", "6mm", "8mm"],
                 "color": ["purple", "blue", "green", "pink", "black"],
                 "material": ["PVC", "TPE", "natural rubber"]},
    "Bluetooth Speaker": {"size": ["mini", "portable", "home"],
                          "color": ["black", "blue", "red", "white"],
                          "waterproof": ["yes", "no"]},
    "Keyboard": {"type": ["mechanical", "membrane", "wireless"],
                 "layout": ["full", "tenkeyless", "60%"],
                 "color": ["black", "white"]},
    "Mouse": {"type": ["wired", "wireless", "bluetooth"],
              "color": ["black", "white", "gray"],
              "DPI": ["800", "1600", "3200"]},
    "Phone Case": {"material": ["silicone", "leather", "plastic"],
                   "color": ["clear", "black", "blue", "pink"],
                   "style": ["slim", "rugged", "wallet"]},
    "Wallet": {"material": ["leather", "faux leather", "canvas"],
               "color": ["black", "brown", "tan"],
               "style": ["bifold", "trifold", "slim"]},
    "Umbrella": {"size": ["compact", "standard", "golf"],
                 "color": ["black", "navy", "red", "blue"],
                 "type": ["manual", "automatic"]},
    "Thermos": {"capacity": ["350ml", "500ml", "750ml", "1000ml"],
                "color": ["silver", "black", "blue", "red"],
                "material": ["stainless steel", "glass"]},
    "Notebook": {"size": ["A4", "A5", "B5"],
                 "style": ["lined", "grid", "blank", "dotted"],
                 "cover": ["hardcover", "softcover", "spiral"]},
    "Candle": {"scent": ["vanilla", "lavender", "ocean", "cinnamon"],
               "size": ["small", "medium", "large"],
               "type": ["jar", "pillar", "votive"]},
}


# =========================================================================
# ID generation utilities
# =========================================================================

# Reserved IDs used by airline tools for new reservations/payments
_RESERVED_RES_IDS = {"HATHAT", "HATHAU", "HATHAV"}
_RESERVED_CERT_NUMS = {3221322, 3221323, 3221324}


def _gen_user_id(rng: random.Random, first: str, last: str) -> str:
    return f"{first.lower()}_{last.lower()}_{rng.randint(1000, 9999)}"


def _gen_reservation_id(rng: random.Random, existing: set) -> str:
    while True:
        rid = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
        if rid not in existing and rid not in _RESERVED_RES_IDS:
            return rid


def _gen_order_id(rng: random.Random, existing: set) -> str:
    while True:
        oid = f"#W{rng.randint(1000000, 9999999)}"
        if oid not in existing:
            return oid


def _gen_flight_number(rng: random.Random, existing: set) -> str:
    while True:
        fn = f"HAT{rng.randint(100, 999)}"
        if fn not in existing:
            return fn


def _gen_product_id(rng: random.Random, existing: set) -> str:
    while True:
        pid = str(rng.randint(1000000000, 9999999999))
        if pid not in existing:
            return pid


def _gen_item_id(rng: random.Random, existing: set) -> str:
    while True:
        iid = str(rng.randint(1000000000, 9999999999))
        if iid not in existing:
            return iid


def _gen_payment_method_id(rng: random.Random, prefix: str, existing: set) -> str:
    while True:
        num = rng.randint(1000000, 9999999)
        pid = f"{prefix}_{num}"
        # Avoid collisions with certificate IDs used by send_certificate
        if num not in _RESERVED_CERT_NUMS and pid not in existing:
            return pid


def _gen_email(first: str, last: str, rng: random.Random) -> str:
    num = rng.randint(100, 9999)
    domain = rng.choice(["example.com", "email.com", "test.com", "mail.com"])
    return f"{first.lower()}.{last.lower()}{num}@{domain}"


def _gen_address_tuple(rng: random.Random) -> Tuple[str, str, str, str, str, str]:
    """Returns (address1, address2, city, state, zip, country)."""
    num = rng.randint(100, 9999)
    street = rng.choice(_STREETS)
    address1 = f"{num} {street}"
    address2 = rng.choice(["", f"Apt {rng.randint(1, 500)}", f"Suite {rng.randint(100, 999)}"])
    city, state, zip_code = rng.choice(_CITIES_STATES)
    return address1, address2, city, state, zip_code, "USA"


def _gen_dob(rng: random.Random) -> str:
    year = rng.randint(1960, 2000)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


# =========================================================================
# Airline DB generation
# =========================================================================

_AIRLINE_FUTURE_DATES = [f"2024-05-{d:02d}" for d in range(16, 26)]
_AIRLINE_PAST_DATES = [f"2024-05-{d:02d}" for d in range(10, 15)]
_CABIN_CLASSES = ["basic_economy", "economy", "business"]


def _gen_airline_credit_card(rng: random.Random, existing: set) -> Tuple[str, AirlineCreditCard]:
    pid = _gen_payment_method_id(rng, "credit_card", existing)
    existing.add(pid)
    brand = rng.choice(["visa", "mastercard"])
    last_four = f"{rng.randint(1000, 9999)}"
    return pid, AirlineCreditCard(source="credit_card", id=pid, brand=brand, last_four=last_four)


def _gen_airline_gift_card(rng: random.Random, existing: set) -> Tuple[str, AirlineGiftCard]:
    pid = _gen_payment_method_id(rng, "gift_card", existing)
    existing.add(pid)
    amount = rng.randint(100, 1000)
    return pid, AirlineGiftCard(source="gift_card", id=pid, amount=amount)


def generate_airline_db(
    rng: random.Random, n_users: int = 10, n_flights: int = 20, n_reservations: int = 15,
) -> FlightDB:
    """Generate a synthetic airline DB with guaranteed diversity."""
    all_payment_ids: set = set()

    # --- Flights ---
    flights: Dict[str, Flight] = {}
    flight_ids: set = set()
    # Guarantee at least one delayed flight on 2024-05-15
    _forced_delayed = True

    for fi in range(n_flights):
        fn = _gen_flight_number(rng, flight_ids)
        flight_ids.add(fn)
        origin, dest = rng.sample(_AIRPORTS, 2)
        dep_hour = rng.randint(5, 22)
        dep_min = rng.choice([0, 15, 30, 45])
        dur_hours = rng.randint(1, 5)
        arr_hour = min(dep_hour + dur_hours, 23)
        arr_min = rng.choice([0, 15, 30, 45])

        dates: Dict[str, Any] = {}
        # Future dates: available
        for d in rng.sample(_AIRLINE_FUTURE_DATES, rng.randint(4, len(_AIRLINE_FUTURE_DATES))):
            dates[d] = FlightDateStatusAvailable(
                status="available",
                available_seats={
                    "basic_economy": rng.randint(5, 30),
                    "economy": rng.randint(5, 30),
                    "business": rng.randint(3, 15),
                },
                prices={
                    "basic_economy": rng.randint(50, 200),
                    "economy": rng.randint(100, 400),
                    "business": rng.randint(300, 800),
                },
            )
        # Past dates: landed
        for d in rng.sample(_AIRLINE_PAST_DATES, rng.randint(1, len(_AIRLINE_PAST_DATES))):
            offset_dep = rng.randint(0, 15)
            offset_arr = rng.randint(0, 15)
            dates[d] = FlightDateStatusLanded(
                status="landed",
                actual_departure_time_est=f"{d}T{dep_hour:02d}:{(dep_min + offset_dep) % 60:02d}:00",
                actual_arrival_time_est=f"{d}T{arr_hour:02d}:{(arr_min + offset_arr) % 60:02d}:00",
            )
        # Force at least one delayed on 2024-05-15
        if _forced_delayed and fi < 3:
            dates["2024-05-15"] = FlightDateStatusDelayed(
                status="delayed",
                estimated_departure_time_est=f"2024-05-15T{min(dep_hour + 2, 23):02d}:{dep_min:02d}:00",
                estimated_arrival_time_est=f"2024-05-15T{min(arr_hour + 2, 23):02d}:{arr_min:02d}:00",
            )
            _forced_delayed = False  # only need one

        flights[fn] = Flight(
            flight_number=fn, origin=origin, destination=dest,
            scheduled_departure_time_est=f"{dep_hour:02d}:{dep_min:02d}:00",
            scheduled_arrival_time_est=f"{arr_hour:02d}:{arr_min:02d}:00",
            dates=dates,
        )

    # --- Users ---
    users: Dict[str, AirlineUser] = {}
    user_ids: List[str] = []
    used_names: set = set()
    for _ in range(n_users):
        while True:
            first = rng.choice(_FIRST_NAMES)
            last = rng.choice(_LAST_NAMES)
            if f"{first}_{last}" not in used_names:
                used_names.add(f"{first}_{last}")
                break
        uid = _gen_user_id(rng, first, last)
        a1, a2, city, state, zc, country = _gen_address_tuple(rng)
        payment_methods: Dict[str, Any] = {}
        cc_id, cc = _gen_airline_credit_card(rng, all_payment_ids)
        payment_methods[cc_id] = cc
        if rng.random() < 0.3:
            cc2_id, cc2 = _gen_airline_credit_card(rng, all_payment_ids)
            payment_methods[cc2_id] = cc2
        if rng.random() < 0.4:
            gc_id, gc = _gen_airline_gift_card(rng, all_payment_ids)
            payment_methods[gc_id] = gc
        saved = []
        if rng.random() < 0.4:
            saved.append(Passenger(first_name=rng.choice(_FIRST_NAMES), last_name=last, dob=_gen_dob(rng)))
        users[uid] = AirlineUser(
            user_id=uid,
            name=Name(first_name=first, last_name=last),
            address=Address(address1=a1, address2=a2, city=city, country=country, state=state, zip=zc),
            email=_gen_email(first, last, rng),
            dob=_gen_dob(rng),
            payment_methods=payment_methods,
            saved_passengers=saved,
            membership=rng.choice(["gold", "silver", "regular"]),
            reservations=[],
        )
        user_ids.append(uid)

    # --- Reservations ---
    reservations: Dict[str, Reservation] = {}
    res_ids: set = set()
    flight_list = list(flights.keys())

    # Find delayed flights so we can force at least one reservation onto one
    _delayed_flights = [
        (fn, d)
        for fn, fl in flights.items()
        for d, s in fl.dates.items()
        if s.status == "delayed"
    ]
    _force_delayed_res = bool(_delayed_flights)

    for i in range(n_reservations):
        rid = _gen_reservation_id(rng, res_ids)
        res_ids.add(rid)
        uid = user_ids[i % len(user_ids)]
        user = users[uid]
        cabin = rng.choice(_CABIN_CLASSES)

        fn = None
        date = None
        price = None

        # Force the first reservation onto a delayed flight for compensation tasks
        if _force_delayed_res and i == 0:
            fn, date = rng.choice(_delayed_flights)
            price = rng.randint(100, 500)
            _force_delayed_res = False
        else:
            # Pick a flight with an available future date
            rng.shuffle(flight_list)
            for candidate_fn in flight_list:
                fl = flights[candidate_fn]
                avail = [d for d, s in fl.dates.items() if s.status == "available"]
                if avail:
                    fn = candidate_fn
                    date = rng.choice(avail)
                    price = fl.dates[date].prices.get(cabin, rng.randint(100, 500))
                    break
        if fn is None:
            # Fallback: use any flight/date
            fn = rng.choice(flight_list)
            date = rng.choice(list(flights[fn].dates.keys()))
            price = rng.randint(100, 500)

        flight_obj = flights[fn]
        res_flights = [ReservationFlight(
            flight_number=fn, origin=flight_obj.origin, destination=flight_obj.destination,
            date=date, price=price,
        )]

        passengers = [Passenger(first_name=user.name.first_name, last_name=user.name.last_name, dob=user.dob)]
        cc_ids = [k for k, v in user.payment_methods.items() if v.source == "credit_card"]
        pay_id = rng.choice(cc_ids) if cc_ids else list(user.payment_methods.keys())[0]
        payment_history = [Payment(payment_id=pay_id, amount=price)]

        baggages = rng.randint(0, 3)
        nonfree = max(0, baggages - 1)
        status = "cancelled" if i >= n_reservations - 2 else None

        reservations[rid] = Reservation(
            reservation_id=rid, user_id=uid,
            origin=flight_obj.origin, destination=flight_obj.destination,
            flight_type=rng.choice(["one_way", "round_trip"]),
            cabin=cabin, flights=res_flights, passengers=passengers,
            payment_history=payment_history,
            created_at=f"2024-05-{rng.randint(1, 14):02d}T{rng.randint(8, 20):02d}:00:00",
            total_baggages=baggages, nonfree_baggages=nonfree,
            insurance=rng.choice(["yes", "no"]), status=status,
        )
        user.reservations.append(rid)

    return FlightDB(flights=flights, users=users, reservations=reservations)


# =========================================================================
# Retail DB generation
# =========================================================================

def _gen_retail_credit_card(rng: random.Random, existing: set) -> Tuple[str, RetailCreditCard]:
    pid = _gen_payment_method_id(rng, "credit_card", existing)
    existing.add(pid)
    return pid, RetailCreditCard(source="credit_card", id=pid,
                                  brand=rng.choice(["visa", "mastercard", "amex"]),
                                  last_four=f"{rng.randint(1000, 9999)}")


def _gen_retail_gift_card(rng: random.Random, existing: set) -> Tuple[str, RetailGiftCard]:
    pid = _gen_payment_method_id(rng, "gift_card", existing)
    existing.add(pid)
    return pid, RetailGiftCard(source="gift_card", id=pid, balance=round(rng.uniform(50, 500), 2))


def generate_retail_db(
    rng: random.Random, n_products: int = 15, n_users: int = 10, n_orders: int = 18,
) -> RetailDB:
    """Generate a synthetic retail DB with guaranteed status diversity."""
    all_payment_ids: set = set()
    all_item_ids: set = set()

    # --- Products ---
    products: Dict[str, Product] = {}
    prod_ids: set = set()
    product_names = list(_PRODUCT_TYPES.keys())
    rng.shuffle(product_names)
    product_names = product_names[:n_products]

    for pname in product_names:
        pid = _gen_product_id(rng, prod_ids)
        prod_ids.add(pid)
        opts_spec = _PRODUCT_TYPES[pname]
        n_variants = rng.randint(3, 6)
        variants: Dict[str, Variant] = {}
        for _ in range(n_variants):
            iid = _gen_item_id(rng, all_item_ids)
            all_item_ids.add(iid)
            opts = {k: rng.choice(v) for k, v in opts_spec.items()}
            variants[iid] = Variant(
                item_id=iid, options=opts,
                available=rng.random() < 0.85,
                price=round(rng.uniform(10, 500), 2),
            )
        # Guarantee >= 2 available
        avail_count = sum(1 for v in variants.values() if v.available)
        if avail_count < 2:
            for v in list(variants.values())[:2]:
                v.available = True
        products[pid] = Product(name=pname, product_id=pid, variants=variants)

    # --- Users ---
    users: Dict[str, RetailUser] = {}
    user_ids: List[str] = []
    used_names: set = set()
    for _ in range(n_users):
        while True:
            first = rng.choice(_FIRST_NAMES)
            last = rng.choice(_LAST_NAMES)
            if f"{first}_{last}" not in used_names:
                used_names.add(f"{first}_{last}")
                break
        uid = _gen_user_id(rng, first, last)
        a1, a2, city, state, zc, country = _gen_address_tuple(rng)
        payment_methods: Dict[str, Any] = {}
        cc_id, cc = _gen_retail_credit_card(rng, all_payment_ids)
        payment_methods[cc_id] = cc
        if rng.random() < 0.4:
            gc_id, gc = _gen_retail_gift_card(rng, all_payment_ids)
            payment_methods[gc_id] = gc
        if rng.random() < 0.2:
            pp_id = _gen_payment_method_id(rng, "paypal", all_payment_ids)
            all_payment_ids.add(pp_id)
            payment_methods[pp_id] = Paypal(source="paypal", id=pp_id)
        users[uid] = RetailUser(
            user_id=uid,
            name=UserName(first_name=first, last_name=last),
            address=UserAddress(address1=a1, address2=a2, city=city, state=state, country=country, zip=zc),
            email=_gen_email(first, last, rng),
            payment_methods=payment_methods,
            orders=[],
        )
        user_ids.append(uid)

    # --- Orders (guaranteed status distribution) ---
    statuses = ["pending"] * 4 + ["delivered"] * 5 + ["processed"] * 3 + ["cancelled"] * 2
    while len(statuses) < n_orders:
        statuses.append(rng.choice(["pending", "delivered", "processed"]))
    rng.shuffle(statuses)

    orders: Dict[str, Order] = {}
    order_ids: set = set()
    product_list = list(products.keys())

    for i in range(n_orders):
        oid = _gen_order_id(rng, order_ids)
        order_ids.add(oid)
        uid = user_ids[i % len(user_ids)]
        user = users[uid]
        status = statuses[i]

        n_items = rng.randint(1, 3)
        items: List[OrderItem] = []
        total_amount = 0.0
        for _ in range(n_items):
            prod_id = rng.choice(product_list)
            prod = products[prod_id]
            variant_id = rng.choice(list(prod.variants.keys()))
            variant = prod.variants[variant_id]
            items.append(OrderItem(
                name=prod.name, product_id=prod_id, item_id=variant_id,
                price=variant.price, options=dict(variant.options),
            ))
            total_amount += variant.price

        a1, a2, city, state, zc, country = _gen_address_tuple(rng)
        order_address = UserAddress(address1=a1, address2=a2, city=city, state=state, country=country, zip=zc)

        pay_method_id = rng.choice(list(user.payment_methods.keys()))
        payment_history = [OrderPayment(
            transaction_type="payment", amount=round(total_amount, 2),
            payment_method_id=pay_method_id,
        )]

        fulfillments = []
        if status in ("delivered", "processed"):
            fulfillments = [OrderFullfilment(
                tracking_id=[f"TRK{rng.randint(100000, 999999)}"],
                item_ids=[item.item_id for item in items],
            )]

        orders[oid] = Order(
            order_id=oid, user_id=uid, address=order_address,
            items=items, status=status, fulfillments=fulfillments,
            payment_history=payment_history,
            cancel_reason=rng.choice(["no longer needed", "ordered by mistake"]) if status == "cancelled" else None,
        )
        user.orders.append(oid)

    return RetailDB(products=products, users=users, orders=orders)


# =========================================================================
# Task templates
# =========================================================================

class TaskTemplate:
    """Base class for synthetic task generators."""

    def can_generate(self, db) -> bool:
        raise NotImplementedError

    def generate(self, rng: random.Random, db, task_id: str) -> Task:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Airline templates
# ---------------------------------------------------------------------------

class AirlineCancelReservation(TaskTemplate):
    def can_generate(self, db: FlightDB) -> bool:
        return any(
            db.reservations[rid].status != "cancelled"
            for u in db.users.values() for rid in u.reservations
            if rid in db.reservations
        )

    def generate(self, rng, db: FlightDB, task_id: str) -> Task:
        candidates = [
            (u.user_id, rid)
            for u in db.users.values() for rid in u.reservations
            if rid in db.reservations and db.reservations[rid].status != "cancelled"
        ]
        uid, rid = rng.choice(candidates)
        user = db.users[uid]
        actions = [
            Action(action_id=f"{task_id}_0", name="get_user_details",
                   arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_1", name="get_reservation_details",
                   arguments={"reservation_id": rid}),
            Action(action_id=f"{task_id}_2", name="cancel_reservation",
                   arguments={"reservation_id": rid}),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Cancel an active reservation"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=f"You want to cancel reservation {rid}.",
                known_info=f"You are {user.name.first_name} {user.name.last_name}.\nYour user id is {uid}.",
                unknown_info=None,
                task_instructions=(
                    "You want to cancel your reservation because you changed your plans. "
                    "Answer any verification questions the agent asks. "
                    "If the agent successfully cancels the reservation, thank them and end the conversation."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(actions=actions, communicate_info=[], nl_assertions=None),
        )


class AirlineAddBaggages(TaskTemplate):
    def can_generate(self, db: FlightDB) -> bool:
        for u in db.users.values():
            cc_ids = [k for k, v in u.payment_methods.items() if v.source == "credit_card"]
            if not cc_ids:
                continue
            for rid in u.reservations:
                r = db.reservations.get(rid)
                if r and r.status != "cancelled" and r.total_baggages < 5:
                    return True
        return False

    def generate(self, rng, db: FlightDB, task_id: str) -> Task:
        candidates = []
        for u in db.users.values():
            cc_ids = [k for k, v in u.payment_methods.items() if v.source == "credit_card"]
            if not cc_ids:
                continue
            for rid in u.reservations:
                r = db.reservations.get(rid)
                if r and r.status != "cancelled" and r.total_baggages < 5:
                    candidates.append((u.user_id, rid, cc_ids))
        uid, rid, cc_ids = rng.choice(candidates)
        user = db.users[uid]
        res = db.reservations[rid]
        new_total = res.total_baggages + rng.randint(1, 2)
        new_nonfree = max(0, new_total - 1)
        pay_id = rng.choice(cc_ids)
        cost = 50 * max(0, new_nonfree - res.nonfree_baggages)

        actions = [
            Action(action_id=f"{task_id}_0", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_1", name="get_reservation_details", arguments={"reservation_id": rid}),
            Action(action_id=f"{task_id}_2", name="update_reservation_baggages", arguments={
                "reservation_id": rid, "total_baggages": new_total,
                "nonfree_baggages": new_nonfree, "payment_id": pay_id,
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Add extra baggages to a reservation"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=f"You want to increase baggages on reservation {rid} to {new_total} total.",
                known_info=f"You are {user.name.first_name} {user.name.last_name}.\nYour user id is {uid}.",
                unknown_info=None,
                task_instructions=(
                    f"You want {new_total} total bags on your reservation. If the agent mentions a cost, "
                    f"confirm and use your credit card. Answer any questions the agent asks."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(
                actions=actions, communicate_info=[str(cost)] if cost > 0 else [], nl_assertions=None,
            ),
        )


class AirlineCheckFlightStatus(TaskTemplate):
    def can_generate(self, db: FlightDB) -> bool:
        for u in db.users.values():
            for rid in u.reservations:
                r = db.reservations.get(rid)
                if r and r.status != "cancelled" and r.flights:
                    fl = r.flights[0]
                    f = db.flights.get(fl.flight_number)
                    if f and fl.date in f.dates:
                        return True
        return False

    def generate(self, rng, db: FlightDB, task_id: str) -> Task:
        candidates = []
        for u in db.users.values():
            for rid in u.reservations:
                r = db.reservations.get(rid)
                if r and r.status != "cancelled" and r.flights:
                    fl = r.flights[0]
                    f = db.flights.get(fl.flight_number)
                    if f and fl.date in f.dates:
                        candidates.append((u.user_id, rid, fl, f.dates[fl.date].status))
        uid, rid, fl, status_str = rng.choice(candidates)
        user = db.users[uid]
        actions = [
            Action(action_id=f"{task_id}_0", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_1", name="get_reservation_details", arguments={"reservation_id": rid}),
            Action(action_id=f"{task_id}_2", name="get_flight_status", arguments={
                "flight_number": fl.flight_number, "date": fl.date,
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Check flight status"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=f"You want to check the status of your flight for reservation {rid}.",
                known_info=f"You are {user.name.first_name} {user.name.last_name}.\nYour user id is {uid}.",
                unknown_info=None,
                task_instructions=(
                    "Ask the agent to check your flight status. Once you get the information, "
                    "thank them and end the conversation."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(
                actions=actions, communicate_info=[status_str], nl_assertions=None,
            ),
        )


class AirlineSendCompensation(TaskTemplate):
    def can_generate(self, db: FlightDB) -> bool:
        for u in db.users.values():
            for rid in u.reservations:
                r = db.reservations.get(rid)
                if not r or r.status == "cancelled":
                    continue
                for fl in r.flights:
                    f = db.flights.get(fl.flight_number)
                    if f and fl.date in f.dates and f.dates[fl.date].status in ("delayed", "cancelled"):
                        return True
        return False

    def generate(self, rng, db: FlightDB, task_id: str) -> Task:
        candidates = []
        for u in db.users.values():
            for rid in u.reservations:
                r = db.reservations.get(rid)
                if not r or r.status == "cancelled":
                    continue
                for fl in r.flights:
                    f = db.flights.get(fl.flight_number)
                    if f and fl.date in f.dates and f.dates[fl.date].status in ("delayed", "cancelled"):
                        candidates.append((u.user_id, rid))
                        break
        uid, rid = rng.choice(candidates)
        user = db.users[uid]
        amount = rng.choice([50, 100, 150, 200])
        actions = [
            Action(action_id=f"{task_id}_0", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_1", name="get_reservation_details", arguments={"reservation_id": rid}),
            Action(action_id=f"{task_id}_2", name="send_certificate", arguments={
                "user_id": uid, "amount": amount,
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Compensation for delayed flight"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=f"Your flight was delayed and you want compensation. Reservation: {rid}.",
                known_info=f"You are {user.name.first_name} {user.name.last_name}.\nYour user id is {uid}.",
                unknown_info=None,
                task_instructions=(
                    f"You want a ${amount} certificate as compensation for your delayed flight. "
                    f"If the agent offers a different amount, accept it. Answer any questions."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(
                actions=actions, communicate_info=[str(amount)], nl_assertions=None,
            ),
        )


class AirlineBookReservation(TaskTemplate):
    def can_generate(self, db: FlightDB) -> bool:
        has_avail = any(
            s.status == "available"
            for f in db.flights.values() for s in f.dates.values()
        )
        has_cc_user = any(
            any(v.source == "credit_card" for v in u.payment_methods.values())
            for u in db.users.values()
        )
        return has_avail and has_cc_user

    def generate(self, rng, db: FlightDB, task_id: str) -> Task:
        # Pick user with credit card
        cc_users = [
            (u.user_id, [k for k, v in u.payment_methods.items() if v.source == "credit_card"])
            for u in db.users.values()
            if any(v.source == "credit_card" for v in u.payment_methods.values())
        ]
        uid, cc_ids = rng.choice(cc_users)
        user = db.users[uid]
        pay_id = rng.choice(cc_ids)

        # Find available flight
        avail = []
        for fn, fl in db.flights.items():
            for d, s in fl.dates.items():
                if s.status == "available" and d >= "2024-05-16":
                    avail.append((fn, d, fl.origin, fl.destination, s))
        fn, date, origin, dest, fstatus = rng.choice(avail)
        cabin = rng.choice(["economy", "business"])
        price = fstatus.prices[cabin]

        actions = [
            Action(action_id=f"{task_id}_0", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_1", name="search_direct_flight", arguments={
                "origin": origin, "destination": dest, "date": date,
            }),
            Action(action_id=f"{task_id}_2", name="book_reservation", arguments={
                "user_id": uid, "origin": origin, "destination": dest,
                "flight_type": "one_way", "cabin": cabin,
                "flights": [{"flight_number": fn, "date": date}],
                "passengers": [{"first_name": user.name.first_name,
                                "last_name": user.name.last_name, "dob": user.dob}],
                "payment_methods": [{"payment_id": pay_id, "amount": price}],
                "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no",
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Book a new flight"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=f"You want to book a one-way {cabin} flight from {origin} to {dest} on {date}.",
                known_info=(
                    f"You are {user.name.first_name} {user.name.last_name}.\n"
                    f"Your user id is {uid}.\nYour date of birth is {user.dob}."
                ),
                unknown_info=None,
                task_instructions=(
                    f"Book a one-way {cabin} flight from {origin} to {dest} on {date}. "
                    f"No insurance, no extra baggage. Pay with your credit card. Answer any questions."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(
                actions=actions, communicate_info=[str(price)], nl_assertions=None,
            ),
        )


# ---------------------------------------------------------------------------
# Retail templates
# ---------------------------------------------------------------------------

def _retail_find_user_action(user: RetailUser, task_id: str, idx: int) -> Action:
    """Standard retail user-lookup action."""
    return Action(
        action_id=f"{task_id}_{idx}",
        name="find_user_id_by_name_zip",
        arguments={
            "first_name": user.name.first_name,
            "last_name": user.name.last_name,
            "zip": user.address.zip,
        },
    )


def _retail_known_info(user: RetailUser, uid: str, order_id: Optional[str] = None) -> str:
    parts = [
        f"You are {user.name.first_name} {user.name.last_name}.",
        f"Your zip code is {user.address.zip}.",
    ]
    if order_id:
        parts.append(f"Your order number is {order_id}.")
    return "\n".join(parts)


class RetailCancelPendingOrder(TaskTemplate):
    def can_generate(self, db: RetailDB) -> bool:
        return any(
            db.orders[oid].status == "pending"
            for u in db.users.values() for oid in u.orders if oid in db.orders
        )

    def generate(self, rng, db: RetailDB, task_id: str) -> Task:
        candidates = [
            (u.user_id, oid)
            for u in db.users.values() for oid in u.orders
            if oid in db.orders and db.orders[oid].status == "pending"
        ]
        uid, oid = rng.choice(candidates)
        user = db.users[uid]
        reason = rng.choice(["no longer needed", "ordered by mistake"])
        actions = [
            _retail_find_user_action(user, task_id, 0),
            Action(action_id=f"{task_id}_1", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_2", name="get_order_details", arguments={"order_id": oid}),
            Action(action_id=f"{task_id}_3", name="cancel_pending_order", arguments={
                "order_id": oid, "reason": reason,
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Cancel a pending order"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"You want to cancel order {oid}.",
                known_info=_retail_known_info(user, uid, oid),
                unknown_info="You don't remember your email.",
                task_instructions=(
                    f"You want to cancel order {oid}. The reason is: {reason}. "
                    f"Identify yourself by name and zip code. Answer any questions."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(actions=actions, communicate_info=[], nl_assertions=None),
        )


class RetailExchangeDeliveredItems(TaskTemplate):
    def can_generate(self, db: RetailDB) -> bool:
        for u in db.users.values():
            for oid in u.orders:
                order = db.orders.get(oid)
                if not order or order.status != "delivered":
                    continue
                for item in order.items:
                    prod = db.products.get(item.product_id)
                    if prod and any(v.item_id != item.item_id and v.available for v in prod.variants.values()):
                        return True
        return False

    def generate(self, rng, db: RetailDB, task_id: str) -> Task:
        candidates = []
        for u in db.users.values():
            for oid in u.orders:
                order = db.orders.get(oid)
                if not order or order.status != "delivered":
                    continue
                for item in order.items:
                    prod = db.products.get(item.product_id)
                    if not prod:
                        continue
                    alts = [v for v in prod.variants.values() if v.item_id != item.item_id and v.available]
                    if alts:
                        candidates.append((u.user_id, oid, item, prod, alts))
        uid, oid, item, prod, alts = rng.choice(candidates)
        user = db.users[uid]
        new_variant = rng.choice(alts)
        # Prefer credit card to avoid gift card balance issues
        cc_ids = [k for k, v in user.payment_methods.items() if v.source == "credit_card"]
        pay_id = rng.choice(cc_ids) if cc_ids else rng.choice(list(user.payment_methods.keys()))
        price_diff = round(new_variant.price - item.price, 2)
        new_opts = ", ".join(f"{k}: {v}" for k, v in new_variant.options.items())

        actions = [
            _retail_find_user_action(user, task_id, 0),
            Action(action_id=f"{task_id}_1", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_2", name="get_order_details", arguments={"order_id": oid}),
            Action(action_id=f"{task_id}_3", name="get_product_details", arguments={"product_id": item.product_id}),
            Action(action_id=f"{task_id}_4", name="exchange_delivered_order_items", arguments={
                "order_id": oid, "item_ids": [item.item_id],
                "new_item_ids": [new_variant.item_id], "payment_method_id": pay_id,
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Exchange item in a delivered order"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"You want to exchange the {prod.name} in order {oid} for a different variant ({new_opts}).",
                known_info=_retail_known_info(user, uid, oid),
                unknown_info="You don't remember your email.",
                task_instructions=(
                    f"Exchange the {prod.name} in order {oid} for the variant with: {new_opts}. "
                    f"Use payment method {pay_id} for any price difference. Answer any verification questions."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(
                actions=actions,
                communicate_info=[str(abs(price_diff))] if abs(price_diff) >= 0.01 else [],
                nl_assertions=None,
            ),
        )


class RetailReturnDeliveredItems(TaskTemplate):
    def can_generate(self, db: RetailDB) -> bool:
        return any(
            db.orders[oid].status == "delivered" and db.orders[oid].items
            for u in db.users.values() for oid in u.orders if oid in db.orders
        )

    def generate(self, rng, db: RetailDB, task_id: str) -> Task:
        candidates = [
            (u.user_id, oid)
            for u in db.users.values() for oid in u.orders
            if oid in db.orders and db.orders[oid].status == "delivered" and db.orders[oid].items
        ]
        uid, oid = rng.choice(candidates)
        user = db.users[uid]
        order = db.orders[oid]
        items_to_ret = order.items if len(order.items) == 1 else rng.sample(order.items, rng.randint(1, len(order.items)))
        item_ids = [it.item_id for it in items_to_ret]
        # Must use original payment method or a gift card
        original_pay = order.payment_history[0].payment_method_id
        pay_id = original_pay
        item_names = [it.name for it in items_to_ret]

        actions = [
            _retail_find_user_action(user, task_id, 0),
            Action(action_id=f"{task_id}_1", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_2", name="get_order_details", arguments={"order_id": oid}),
            Action(action_id=f"{task_id}_3", name="return_delivered_order_items", arguments={
                "order_id": oid, "item_ids": item_ids, "payment_method_id": pay_id,
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Return items from a delivered order"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"You want to return {', '.join(item_names)} from order {oid}.",
                known_info=_retail_known_info(user, uid, oid),
                unknown_info="You don't remember your email.",
                task_instructions=(
                    f"Return the following from order {oid}: {', '.join(item_names)}. "
                    f"Refund to your original payment method. Answer any verification questions."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(actions=actions, communicate_info=[], nl_assertions=None),
        )


class RetailModifyPendingOrderItems(TaskTemplate):
    def can_generate(self, db: RetailDB) -> bool:
        for u in db.users.values():
            for oid in u.orders:
                order = db.orders.get(oid)
                if not order or order.status != "pending":
                    continue
                for item in order.items:
                    prod = db.products.get(item.product_id)
                    if prod and any(v.item_id != item.item_id and v.available for v in prod.variants.values()):
                        return True
        return False

    def generate(self, rng, db: RetailDB, task_id: str) -> Task:
        candidates = []
        for u in db.users.values():
            for oid in u.orders:
                order = db.orders.get(oid)
                if not order or order.status != "pending":
                    continue
                for item in order.items:
                    prod = db.products.get(item.product_id)
                    if not prod:
                        continue
                    alts = [v for v in prod.variants.values() if v.item_id != item.item_id and v.available]
                    if alts:
                        candidates.append((u.user_id, oid, item, prod, alts))
        uid, oid, item, prod, alts = rng.choice(candidates)
        user = db.users[uid]
        new_variant = rng.choice(alts)
        cc_ids = [k for k, v in user.payment_methods.items() if v.source == "credit_card"]
        pay_id = rng.choice(cc_ids) if cc_ids else rng.choice(list(user.payment_methods.keys()))
        new_opts = ", ".join(f"{k}: {v}" for k, v in new_variant.options.items())

        actions = [
            _retail_find_user_action(user, task_id, 0),
            Action(action_id=f"{task_id}_1", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_2", name="get_order_details", arguments={"order_id": oid}),
            Action(action_id=f"{task_id}_3", name="get_product_details", arguments={"product_id": item.product_id}),
            Action(action_id=f"{task_id}_4", name="modify_pending_order_items", arguments={
                "order_id": oid, "item_ids": [item.item_id],
                "new_item_ids": [new_variant.item_id], "payment_method_id": pay_id,
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Modify items in a pending order"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"You want to change the {prod.name} in order {oid} to a different variant ({new_opts}).",
                known_info=_retail_known_info(user, uid, oid),
                unknown_info="You don't remember your email.",
                task_instructions=(
                    f"Change the {prod.name} in order {oid} to the variant with: {new_opts}. "
                    f"Use payment method {pay_id} for any price difference. Answer any questions."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(actions=actions, communicate_info=[], nl_assertions=None),
        )


class RetailModifyPendingOrderAddress(TaskTemplate):
    def can_generate(self, db: RetailDB) -> bool:
        return any(
            db.orders[oid].status == "pending"
            for u in db.users.values() for oid in u.orders if oid in db.orders
        )

    def generate(self, rng, db: RetailDB, task_id: str) -> Task:
        candidates = [
            (u.user_id, oid)
            for u in db.users.values() for oid in u.orders
            if oid in db.orders and db.orders[oid].status == "pending"
        ]
        uid, oid = rng.choice(candidates)
        user = db.users[uid]
        a1, a2, city, state, zc, country = _gen_address_tuple(rng)

        actions = [
            _retail_find_user_action(user, task_id, 0),
            Action(action_id=f"{task_id}_1", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_2", name="get_order_details", arguments={"order_id": oid}),
            Action(action_id=f"{task_id}_3", name="modify_pending_order_address", arguments={
                "order_id": oid, "address1": a1, "address2": a2,
                "city": city, "state": state, "country": country, "zip": zc,
            }),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Modify delivery address on a pending order"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"You want to change the delivery address for order {oid}.",
                known_info=_retail_known_info(user, uid, oid),
                unknown_info="You don't remember your email.",
                task_instructions=(
                    f"Change the delivery address for order {oid} to: {a1}, {a2}, {city}, {state} {zc}, {country}. "
                    f"Identify yourself by name and zip code. Answer any questions."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(actions=actions, communicate_info=[], nl_assertions=None),
        )


class RetailQueryProductInfo(TaskTemplate):
    def can_generate(self, db: RetailDB) -> bool:
        return len(db.products) > 0

    def generate(self, rng, db: RetailDB, task_id: str) -> Task:
        uid = rng.choice(list(db.users.keys()))
        user = db.users[uid]
        prod_id = rng.choice(list(db.products.keys()))
        prod = db.products[prod_id]
        n_available = sum(1 for v in prod.variants.values() if v.available)

        actions = [
            _retail_find_user_action(user, task_id, 0),
            Action(action_id=f"{task_id}_1", name="get_user_details", arguments={"user_id": uid}),
            Action(action_id=f"{task_id}_2", name="get_product_details", arguments={"product_id": prod_id}),
        ]
        return Task(
            id=task_id,
            description=Description(purpose="Query product information"),
            user_scenario=UserScenario(persona=None, instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"You want to know about the available variants of the {prod.name} (product ID: {prod_id}).",
                known_info=_retail_known_info(user, uid),
                unknown_info="You don't remember your email.",
                task_instructions=(
                    f"Ask about the {prod.name} (product {prod_id}). "
                    f"You want to know how many variants are available. Once informed, thank the agent."
                ),
            )),
            initial_state=None,
            evaluation_criteria=EvaluationCriteria(
                actions=actions, communicate_info=[str(n_available)], nl_assertions=None,
            ),
        )


# Template registries
_AIRLINE_TEMPLATES: List[TaskTemplate] = [
    AirlineCancelReservation(),
    AirlineAddBaggages(),
    AirlineCheckFlightStatus(),
    AirlineSendCompensation(),
    AirlineBookReservation(),
]

_RETAIL_TEMPLATES: List[TaskTemplate] = [
    RetailCancelPendingOrder(),
    RetailExchangeDeliveredItems(),
    RetailReturnDeliveredItems(),
    RetailModifyPendingOrderItems(),
    RetailModifyPendingOrderAddress(),
    RetailQueryProductInfo(),
]


def _generate_task(rng: random.Random, domain: str, db) -> Task:
    """Generate a synthetic task for the given domain and DB."""
    templates = list(_AIRLINE_TEMPLATES if domain == "airline" else _RETAIL_TEMPLATES)
    rng.shuffle(templates)
    for template in templates:
        if template.can_generate(db):
            task_id = f"syn_{rng.randint(0, 999999):06d}"
            return template.generate(rng, db, task_id)
    raise RuntimeError(f"No template could generate a task for domain={domain}")


# =========================================================================
# Compact schema cache (independent of tau2_bench_game)
# =========================================================================
_compact_schemas_cache: Dict[str, List[Dict]] = {}


# =========================================================================
# Main game class
# =========================================================================

class Tau2BenchSyntheticGame:
    """Tau2-bench game with per-episode synthetic data.

    Identical mechanics to Tau2BenchGame (same tools, policies, reward
    computation, user simulator) but generates a fresh synthetic DB and
    task each episode for maximum training data diversity.
    """

    supports_structured_messages = True

    def __init__(self, max_steps: int = 30, user_client=None, domain: Optional[str] = None):
        self.max_steps = max_steps
        self._user_client = user_client
        self._domain_filter = domain

        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        self._task: Optional[Task] = None
        self._domain: Optional[str] = None
        self._synthetic_db = None
        self._environment = None
        self._conversation: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._user_system_prompt: str = ""
        self._terminated_normally: bool = False
        self._last_call_key: Optional[str] = None
        self._repeat_count: int = 0

    def reset(self, seed: int) -> None:
        rng = random.Random(seed)

        if self._domain_filter:
            self._domain = self._domain_filter
        else:
            self._domain = rng.choice(["airline", "retail"])

        if self._domain == "airline":
            self._synthetic_db = generate_airline_db(rng)
        else:
            self._synthetic_db = generate_retail_db(rng)

        self._task = _generate_task(rng, self._domain, self._synthetic_db)
        self._environment = self._make_fresh_environment()

        self._user_system_prompt = (
            f"{_user_sim_guidelines}\n\n"
            f"<scenario>\n{self._task.user_scenario}\n</scenario>"
        )

        first_user_msg = self._call_user_llm(
            [{"role": "user", "content": "Hi! How can I help you today?"}]
        )

        self._conversation = [{"role": "user", "text": first_user_msg}]
        self._step_count = 0
        self._last_call_key = None
        self._repeat_count = 0
        self._terminated_normally = False

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    # -----------------------------------------------------------------
    # Structured access
    # -----------------------------------------------------------------

    def get_system_prompt(self) -> str:
        policy = _airline_policy if self._domain == "airline" else _retail_policy
        return (
            f"<instructions>\n"
            f"You are a customer service agent that helps the user according to the <policy> provided below.\n"
            f"In each turn you can either:\n"
            f"- Send a message to the user.\n"
            f"- Make a tool call.\n"
            f"You cannot do both at the same time.\n"
            f"\n"
            f"Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\n"
            f"</instructions>\n"
            f"<policy>\n"
            f"{policy}\n"
            f"</policy>"
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return _airline_tool_schemas if self._domain == "airline" else _retail_tool_schemas

    def get_tool_schemas_compact(self) -> List[Dict[str, Any]]:
        domain = self._domain or "airline"
        if domain not in _compact_schemas_cache:
            _compact_schemas_cache[domain] = _strip_descriptions(self.get_tool_schemas())
        return _compact_schemas_cache[domain]

    def get_messages(self) -> List[Dict[str, Any]]:
        messages = []
        tool_call_counter = 0
        i = 0
        while i < len(self._conversation):
            msg = self._conversation[i]
            role = msg["role"]
            text = msg["text"]
            if role == "user":
                messages.append({"role": "user", "content": text})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": text, "tool_calls": None})
            elif role == "tool_call":
                tc = json.loads(text)
                tc_id = f"tool-{tool_call_counter:04d}"
                tool_call_counter += 1
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": tc_id, "name": tc["name"],
                                    "function": {"name": tc["name"],
                                                 "arguments": json.dumps(tc.get("arguments", {}))},
                                    "type": "function"}],
                })
                if i + 1 < len(self._conversation) and self._conversation[i + 1]["role"] == "tool_result":
                    messages.append({
                        "role": "tool", "content": self._conversation[i + 1]["text"],
                        "tool_call_id": tc_id,
                    })
                    i += 1
            i += 1
        return messages

    # -----------------------------------------------------------------
    # Text observation
    # -----------------------------------------------------------------

    def observe(self, player_id: int) -> str:
        tool_schemas = json.dumps(self.get_tool_schemas(), indent=2)
        lines = [
            self.get_system_prompt(), "",
            "<available_tools>", tool_schemas, "</available_tools>", "",
            "<conversation>",
        ]
        for msg in self._conversation:
            lines.append(f"[{msg['role'].upper()}]: {msg['text']}")
        lines.extend([
            "</conversation>", "",
            "Respond with exactly one JSON object. Either:",
            '- Tool call: {"name": "<tool_name>", "arguments": {<args>}}',
            '- Message: {"name": "respond_to_user", "arguments": {"message": "<text>"}}',
            '- Transfer: {"name": "transfer_to_human_agents", "arguments": {"summary": "<text>"}}',
        ])
        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        return [] if self.done else ['{"name": "...", "arguments": {...}}']

    # -----------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------

    def step(self, action: Optional[str]) -> None:
        if self.done:
            return
        self._step_count += 1

        if action is None:
            self._finalize(0.0, "No action provided")
            self.invalid_player = 0
            return

        tool_call = _parse_tool_call(action)
        if tool_call is None:
            self._finalize(0.0, "Invalid action format")
            self.invalid_player = 0
            return

        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        if tool_name == "respond_to_user":
            message = tool_args.get("message", "")
            self._conversation.append({"role": "assistant", "text": message})
            self._last_call_key = None
            self._repeat_count = 0

            user_response = self._get_user_response()
            if user_response is None:
                self._finalize_with_reward("User LLM failure")
                return
            if _is_user_stop(user_response):
                clean = _strip_stop_tokens(user_response)
                self._conversation.append({"role": "user", "text": clean or "Thank you. Goodbye."})
                self._finalize_with_reward("User stop")
                return
            self._conversation.append({"role": "user", "text": user_response})
        else:
            self._conversation.append({"role": "tool_call", "text": json.dumps(tool_call)})
            tc = ToolCall(name=tool_name, arguments=tool_args)
            tool_msg = self._environment.get_response(tc)
            self._conversation.append({"role": "tool_result", "text": tool_msg.content})

            call_key = json.dumps({"name": tool_name, "arguments": tool_args}, sort_keys=True)
            if call_key == self._last_call_key:
                self._repeat_count += 1
            else:
                self._last_call_key = call_key
                self._repeat_count = 1
            if self._repeat_count >= 3:
                self._finalize(0.0, f"Loop detected: {tool_name} called 3x with same args")
                return

        if self._step_count >= self.max_steps:
            self._finalize(0.0, "Max steps reached")

    # -----------------------------------------------------------------
    # User simulator
    # -----------------------------------------------------------------

    def _get_user_response(self) -> Optional[str]:
        if self._user_client is None:
            return "###STOP###"
        flipped = []
        for msg in self._conversation:
            if msg["role"] == "assistant":
                flipped.append({"role": "user", "content": msg["text"]})
            elif msg["role"] == "user":
                flipped.append({"role": "assistant", "content": msg["text"]})
        try:
            raw = self._user_client.generate(self._user_system_prompt, flipped)
        except Exception:
            return None
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    def _call_user_llm(self, messages: List[Dict[str, str]]) -> str:
        if self._user_client is None:
            return "I need help with my account."
        try:
            raw = self._user_client.generate(self._user_system_prompt, messages)
        except Exception:
            return "I need help with my account."
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if _is_user_stop(raw):
            clean = _strip_stop_tokens(raw)
            return clean or "I need help with my account."
        return raw

    # -----------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------

    def _finalize_with_reward(self, reason: str) -> None:
        self._terminated_normally = True
        reward = self._compute_reward()
        self._finalize(reward, reason)

    def _compute_reward(self) -> float:
        task = self._task
        if task is None or task.evaluation_criteria is None:
            return 1.0

        gold_env = self._make_fresh_environment()
        golden_actions = task.evaluation_criteria.actions or []
        for action in golden_actions:
            try:
                gold_env.make_tool_call(
                    tool_name=action.name, requestor=action.requestor,
                    **action.arguments,
                )
            except Exception:
                pass

        gold_db_hash = gold_env.get_db_hash()
        predicted_db_hash = self._environment.get_db_hash()
        db_reward = 1.0 if gold_db_hash == predicted_db_hash else 0.0

        communicate_info = task.evaluation_criteria.communicate_info or []
        if not communicate_info:
            comm_reward = 1.0
        else:
            assistant_texts = [
                msg["text"] for msg in self._conversation if msg["role"] == "assistant"
            ]
            all_found = all(
                any(info_str.lower() in text.lower().replace(",", "") for text in assistant_texts)
                for info_str in communicate_info
            )
            comm_reward = 1.0 if all_found else 0.0

        return db_reward * comm_reward

    def _make_fresh_environment(self):
        db_copy = self._synthetic_db.model_copy(deep=True)
        if self._domain == "airline":
            return Environment(domain_name="airline", policy=_airline_policy, tools=AirlineTools(db_copy))
        return Environment(domain_name="retail", policy=_retail_policy, tools=RetailTools(db_copy))

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        return {
            "domain": self._domain or "",
            "task_id": self._task.id if self._task else "",
            "task_type": self._task.description.purpose if self._task and self._task.description else "",
            "steps": self._step_count,
            "reward": self.rewards.get(0, 0.0),
            "reason": getattr(self, "_reason", ""),
            "terminated_normally": self._terminated_normally,
            "conversation_length": len(self._conversation),
            "synthetic": True,
        }
