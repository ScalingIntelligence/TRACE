"""Database loading for airline and retail domains.

Loads the full tau2-bench database (exactly as used in evaluation).
No synthetic generation, no fallbacks.
"""

import sys
import random
import copy
import pathlib
from functools import lru_cache
from typing import Dict, Any, Optional

# Ensure tau2-bench source is in path
_TAU2_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "tau2-bench" / "src")
if _TAU2_SRC not in sys.path:
    sys.path.insert(0, _TAU2_SRC)

from tau2.domains.airline.data_model import FlightDB
from tau2.domains.retail.data_model import RetailDB


# =====================================================================
# Database loading (cached)
# =====================================================================

_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "tau2-bench" / "data" / "tau2" / "domains"


@lru_cache(maxsize=2)
def _load_pydantic_db(domain: str):
    """Load the full tau2-bench database as a Pydantic model. Cached."""
    path = str(_DATA_DIR / domain / "db.json")
    if domain == "airline":
        return FlightDB.load(path)
    else:
        return RetailDB.load(path)


def get_pydantic_db(domain: str):
    """Get a deep copy of the full tau2-bench database as a Pydantic model.

    This is what gets passed to ToolExecutor per episode.
    ~150ms for airline, ~100ms for retail.
    """
    return _load_pydantic_db(domain).model_copy(deep=True)


# =====================================================================
# Sampling functions (find real entities matching template criteria)
# =====================================================================

def _load_raw_db(domain: str) -> Dict[str, Any]:
    """Get the raw dict representation of the DB (for sampling)."""
    return _load_pydantic_db(domain).model_dump()


def sample_airline_user(
    rng: random.Random,
    criteria: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Find a real user+reservation matching template criteria.

    criteria keys:
        cabin: required cabin type (e.g. "economy", "basic_economy")
        insurance: "yes" or "no"
        is_recent: bool (within 24h = created_at >= 2024-05-14)
        membership: str or list of str
        min_reservations: int (default 1)
        flight_status: str (default "available")
        num_passengers_min: int
        status: None (active) or specific status

    Returns dict with: user, reservation, reservation_id, user_id
    or None if no match found.
    """
    full_db = _load_raw_db("airline")

    # Shuffle user IDs deterministically
    user_ids = list(full_db["users"].keys())
    rng.shuffle(user_ids)

    cabin = criteria.get("cabin")
    insurance = criteria.get("insurance")
    is_recent = criteria.get("is_recent")
    membership = criteria.get("membership")
    min_reservations = criteria.get("min_reservations", 1)
    num_passengers_min = criteria.get("num_passengers_min", 1)
    required_status = criteria.get("status")  # None = active (no status field)

    if isinstance(membership, str):
        membership = [membership]

    for uid in user_ids:
        user = full_db["users"][uid]

        if membership and user.get("membership") not in membership:
            continue

        res_ids = user.get("reservations", [])
        if len(res_ids) < min_reservations:
            continue

        # Check each reservation
        rng_copy = random.Random(rng.random())  # don't consume parent rng
        shuffled_res = list(res_ids)
        rng_copy.shuffle(shuffled_res)

        for rid in shuffled_res:
            res = full_db["reservations"].get(rid)
            if res is None:
                continue

            if cabin and res.get("cabin") != cabin:
                continue
            if insurance is not None and res.get("insurance") != insurance:
                continue
            if is_recent is not None:
                created = res.get("created_at", "")
                # Recent = created_at date >= 2024-05-14 (within ~24h of reference time 2024-05-15)
                recent = created >= "2024-05-14" if created else False
                if is_recent != recent:
                    continue
            if num_passengers_min > 1:
                if len(res.get("passengers", [])) < num_passengers_min:
                    continue
            if required_status is not None:
                if res.get("status") != required_status:
                    continue
            else:
                # Active = no status field or status is None
                if res.get("status") is not None:
                    continue

            # Check flight status if needed
            flight_status = criteria.get("flight_status")
            if flight_status:
                flights_ok = True
                for fleg in res.get("flights", []):
                    fn = fleg.get("flight_number")
                    date = fleg.get("date")
                    flight_data = full_db["flights"].get(fn, {}).get("dates", {}).get(date, {})
                    if flight_data.get("status") != flight_status:
                        flights_ok = False
                        break
                if not flights_ok:
                    continue

            return {
                "user": copy.deepcopy(user),
                "reservation": copy.deepcopy(res),
                "reservation_id": rid,
                "user_id": uid,
            }

    return None


def sample_airline_multi_reservations(
    rng: random.Random,
    criteria: Dict[str, Any],
    min_reservations: int = 2,
) -> Optional[Dict[str, Any]]:
    """Find a real user with multiple reservations matching criteria.

    Returns dict with: user, reservations (list), reservation_ids (list), user_id
    or None if no match found.
    """
    full_db = _load_raw_db("airline")
    user_ids = list(full_db["users"].keys())
    rng.shuffle(user_ids)

    membership = criteria.get("membership")
    if isinstance(membership, str):
        membership = [membership]

    for uid in user_ids:
        user = full_db["users"][uid]
        if membership and user.get("membership") not in membership:
            continue

        res_ids = user.get("reservations", [])
        # Filter to active reservations
        active = []
        for rid in res_ids:
            res = full_db["reservations"].get(rid)
            if res and res.get("status") is None:
                active.append((rid, res))

        if len(active) >= min_reservations:
            selected = active[:min_reservations]
            return {
                "user": copy.deepcopy(user),
                "reservations": [copy.deepcopy(r) for _, r in selected],
                "reservation_ids": [rid for rid, _ in selected],
                "user_id": uid,
            }

    return None


def sample_retail_user(
    rng: random.Random,
    criteria: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Find a real user+order matching template criteria.

    criteria keys:
        status: order status ("pending", "delivered", etc.)
        min_items: minimum items in order
        min_orders: minimum orders for user
        has_gift_card: bool (user has gift card payment method)
        has_multiple_payment_types: bool

    Returns dict with: user, order, order_id, user_id
    or None if no match found.
    """
    full_db = _load_raw_db("retail")

    user_ids = list(full_db["users"].keys())
    rng.shuffle(user_ids)

    status = criteria.get("status")
    min_items = criteria.get("min_items", 1)
    min_orders = criteria.get("min_orders", 1)
    has_gift_card = criteria.get("has_gift_card")
    has_multiple_payment_types = criteria.get("has_multiple_payment_types")

    for uid in user_ids:
        user = full_db["users"][uid]

        order_ids = user.get("orders", [])
        if len(order_ids) < min_orders:
            continue

        if has_gift_card:
            pm = user.get("payment_methods", {})
            if not any(v.get("source") == "gift_card" for v in pm.values()):
                continue

        if has_multiple_payment_types:
            pm = user.get("payment_methods", {})
            sources = {v.get("source") for v in pm.values()}
            if len(sources) < 2:
                continue

        rng_copy = random.Random(rng.random())
        shuffled_orders = list(order_ids)
        rng_copy.shuffle(shuffled_orders)

        for oid in shuffled_orders:
            order = full_db["orders"].get(oid)
            if order is None:
                continue

            if status and order.get("status") != status:
                continue
            if len(order.get("items", [])) < min_items:
                continue

            return {
                "user": copy.deepcopy(user),
                "order": copy.deepcopy(order),
                "order_id": oid,
                "user_id": uid,
            }

    return None
