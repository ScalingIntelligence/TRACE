"""Database functions for adversarial policy game.

Uses synthetic data generation — no tau2-bench DB entities.
Maintains the same API as the old module so scenarios.py changes are minimal.
"""

import random
from typing import Dict, Any, List, Optional

from .synthetic_db import (
    generate_airline_episode,
    generate_airline_multi_reservations,
    generate_retail_episode,
    generate_retail_multi_orders,
    build_airline_db,
    build_retail_db,
)


def sample_airline_user(
    rng: random.Random,
    criteria: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Generate a synthetic airline user + reservation matching criteria.

    Drop-in replacement for the old function that searched the real tau2-bench DB.
    Always succeeds (no "None if not found" since we generate to spec).

    Returns dict with: user, reservation, reservation_id, user_id, flights_db
    """
    return generate_airline_episode(rng, criteria)


def sample_airline_multi_reservations(
    rng: random.Random,
    criteria: Dict[str, Any],
    min_reservations: int = 2,
) -> Optional[Dict[str, Any]]:
    """Generate a synthetic user with multiple reservations.

    Returns dict with: user, reservations, reservation_ids, user_id, flights_db
    """
    return generate_airline_multi_reservations(rng, criteria, min_reservations)


def sample_retail_user(
    rng: random.Random,
    criteria: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Generate a synthetic retail user + order matching criteria.

    Returns dict with: user, order, order_id, user_id, products_db
    """
    return generate_retail_episode(rng, criteria)


def sample_retail_multi_orders(
    rng: random.Random,
    criteria: Dict[str, Any],
    order_specs: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a synthetic retail user with multiple orders.

    Args:
        rng: Seeded random instance.
        criteria: Shared criteria for the user (has_gift_card, etc.).
        order_specs: List of per-order specs (each with "status", optional "min_items").
            If None, generates 2 orders with random statuses.

    Returns dict with: user, orders, order_ids, user_id, products_db
    """
    return generate_retail_multi_orders(rng, criteria, order_specs)
