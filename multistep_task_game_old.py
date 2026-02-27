"""
Multi-Step Task Game - Training Environment for Sequential Task Completion

Teaches one core skill: completing a sequence of N tool-call operations,
gathering required information along the way.

The model must:
  1. Look up order/product details before acting (information gathering)
  2. Execute all N requested operations without skipping any (completeness)
  3. Batch items into single calls where required (one-shot constraint)
  4. Track what's been done across turns (state management)

Difficulty scales by the number of operations and orders, not by
reasoning complexity. All data is procedurally generated.

Reward: completion bonus (0.4) + linear partial credit (max 0.6) +
lookup penalty (0.15 per skip). Designed for GRPO signal quality —
0.5 gap between perfect and 1-miss, with gradient across partials.
Penalties sized so completion always dominates (all-ops+skips > miss-ops).
"""

import random
import json
import copy
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Constants
# =============================================================================

FIRST_NAMES = [
    "Liam", "Sophia", "Mateo", "Aria", "Kai", "Zara", "Ethan", "Mila",
    "Owen", "Luna", "Felix", "Ivy", "Jasper", "Nora", "Leo", "Elara",
]

PRODUCT_TYPES = [
    "Wireless Headphones", "USB-C Cable", "Phone Case", "Bluetooth Speaker",
    "Laptop Stand", "Wireless Mouse", "Keyboard", "Monitor Light", "Webcam",
    "Desk Lamp", "Power Bank", "Smart Watch", "Tablet Case", "Microphone",
    "External SSD", "Charger", "Mouse Pad", "Backpack", "Water Bottle",
    "Thermostat",
]

OPTION_COLORS = ["red", "blue", "green", "black", "white", "silver"]
OPTION_SIZES = ["small", "medium", "large"]
OPTION_MATERIALS = ["plastic", "metal", "fabric", "wood"]

STREETS = [
    "10 Maple Lane", "25 Oak Drive", "77 Pine Street",
    "33 Cedar Avenue", "88 Birch Road", "42 Elm Court",
    "55 Walnut Blvd", "19 Cherry Way", "64 Spruce Circle",
]

CITIES_STATES = [
    ("Portland", "OR", "97201"), ("Austin", "TX", "73301"),
    ("Denver", "CO", "80201"), ("Raleigh", "NC", "27601"),
    ("Nashville", "TN", "37201"), ("Minneapolis", "MN", "55401"),
    ("Tampa", "FL", "33601"), ("Boise", "ID", "83701"),
]

CANCEL_REASONS = ["no longer needed", "ordered by mistake"]

USER_CONFIRMATIONS = [
    "OK, please proceed with the next task.",
    "Got it, thanks. What about the remaining items?",
    "Great, please continue.",
    "Understood. Please handle the other requests too.",
    "Thanks! Please go ahead with the rest.",
]

INCREMENTAL_ADDITIONS = [
    "Actually, one more thing — {task_desc}",
    "Oh wait, I also need you to {task_desc}",
    "Before you wrap up, can you also {task_desc}",
    "Sorry, I forgot — please also {task_desc}",
]

# Difficulty scales by number of operations and orders (like structured_data_game
# scales by data size). Templates test pure multi-step execution, not reasoning.
DIFFICULTY_CONFIG = {
    1: {"orders": (2, 2), "items_per_order": (1, 1), "ops": (2, 2),
        "templates": ["cross_order_ops"]},
    2: {"orders": (2, 3), "items_per_order": (1, 2), "ops": (2, 3),
        "templates": ["cross_order_ops", "multi_item_batch", "address_propagation"]},
    3: {"orders": (3, 5), "items_per_order": (2, 3), "ops": (4, 6),
        "templates": ["cross_order_ops", "multi_item_batch", "address_propagation",
                      "status_dependent", "cross_ref_address"]},
    4: {"orders": (4, 6), "items_per_order": (2, 4), "ops": (5, 8),
        "templates": ["cross_order_ops", "multi_item_batch", "address_propagation",
                      "status_dependent", "cross_ref_address",
                      "incremental_reveal"]},
    5: {"orders": (5, 7), "items_per_order": (2, 4), "ops": (6, 9),
        "templates": ["cross_order_ops", "multi_item_batch", "address_propagation",
                      "status_dependent", "cross_ref_address",
                      "incremental_reveal"]},
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ItemVariant:
    item_id: str
    color: str
    size: str
    material: str
    price: float
    available: bool


@dataclass
class ProductEntity:
    product_id: str
    name: str
    variants: List[ItemVariant]


@dataclass
class OrderItem:
    item_id: str
    product_id: str
    name: str
    price: float


@dataclass
class Address:
    street: str
    city: str
    state: str
    zip_code: str

    def to_dict(self) -> Dict[str, str]:
        return {"address1": self.street, "city": self.city,
                "state": self.state, "zip": self.zip_code}

    def matches(self, args: Dict[str, Any]) -> bool:
        return (str(args.get("address1", "")).strip().lower() == self.street.lower()
                and str(args.get("city", "")).strip().lower() == self.city.lower()
                and str(args.get("state", "")).strip().lower() == self.state.lower()
                and str(args.get("zip", "")).strip().lower() == self.zip_code.lower())


@dataclass
class OrderEntity:
    order_id: str
    status: str  # "pending", "delivered"
    items: List[OrderItem]
    address: Address
    payment_method_id: str


@dataclass
class TaskOperation:
    description: str  # human-readable for user message
    order_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    requires_lookup: bool = True  # must call get_order_details first
    is_incremental: bool = False  # revealed mid-conversation


@dataclass
class EpisodeScenario:
    orders: List[OrderEntity]
    products: List[ProductEntity]
    operations: List[TaskOperation]
    initial_task_descriptions: List[str]
    incremental_task_descriptions: List[str]
    difficulty: int


# =============================================================================
# Tool schemas
# =============================================================================

TOOL_SCHEMAS = [
    {
        "name": "get_order_details",
        "description": "Look up the full details of an order including status, items, address, and payment method.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID to look up"},
        },
    },
    {
        "name": "get_product_details",
        "description": "Look up a product to see all available variant items with their options, prices, and availability.",
        "parameters": {
            "product_id": {"type": "string", "description": "The product ID to look up"},
        },
    },
    {
        "name": "cancel_pending_order",
        "description": "Cancel a pending order.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID to cancel"},
            "reason": {"type": "string", "description": "Reason for cancellation"},
        },
    },
    {
        "name": "modify_pending_order_items",
        "description": "Modify items in a pending order by replacing them with different variant items.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID"},
            "item_ids": {"type": "array", "description": "List of current item IDs to replace"},
            "new_item_ids": {"type": "array", "description": "List of new item IDs (same length as item_ids)"},
            "payment_method_id": {"type": "string", "description": "Payment method for price difference"},
        },
    },
    {
        "name": "modify_pending_order_address",
        "description": "Change the shipping address of a pending order.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID"},
            "address1": {"type": "string", "description": "Street address"},
            "city": {"type": "string", "description": "City"},
            "state": {"type": "string", "description": "State abbreviation"},
            "zip": {"type": "string", "description": "ZIP code"},
        },
    },
    {
        "name": "return_delivered_order_items",
        "description": "Return items from a delivered order for a refund.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID"},
            "item_ids": {"type": "array", "description": "List of item IDs to return"},
            "payment_method_id": {"type": "string", "description": "Payment method for refund"},
        },
    },
    {
        "name": "exchange_delivered_order_items",
        "description": "Exchange items in a delivered order for different variant items of the same product.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID"},
            "item_ids": {"type": "array", "description": "List of current item IDs to exchange"},
            "new_item_ids": {"type": "array", "description": "List of new item IDs"},
            "payment_method_id": {"type": "string", "description": "Payment method for price difference"},
        },
    },
    {
        "name": "respond_to_user",
        "description": "Send a final message to the user summarizing what was done.",
        "parameters": {
            "message": {"type": "string", "description": "Message to send"},
        },
    },
]


# =============================================================================
# Data generation helpers
# =============================================================================

def _gen_order_id(rng: random.Random) -> str:
    return f"#W{rng.randint(100000, 999999)}"


def _gen_product_id(rng: random.Random) -> str:
    return f"prod_{rng.randint(1000, 9999)}"


def _gen_item_id(rng: random.Random) -> str:
    return f"item_{rng.randint(10000, 99999)}"


def _gen_address(rng: random.Random) -> Address:
    city, state, zipcode = rng.choice(CITIES_STATES)
    return Address(rng.choice(STREETS), city, state, zipcode)


def _gen_product(rng: random.Random, name: str, num_variants: int = 4) -> ProductEntity:
    """Generate a product with several variants of different options/prices."""
    pid = _gen_product_id(rng)
    variants = []
    used_combos = set()
    for _ in range(num_variants):
        color = rng.choice(OPTION_COLORS)
        size = rng.choice(OPTION_SIZES)
        material = rng.choice(OPTION_MATERIALS)
        combo = (color, size, material)
        if combo in used_combos:
            continue
        used_combos.add(combo)
        variants.append(ItemVariant(
            item_id=_gen_item_id(rng),
            color=color, size=size, material=material,
            price=round(rng.uniform(20, 300), 2),
            available=rng.random() > 0.15,
        ))
    # Ensure at least 2 variants
    while len(variants) < 2:
        variants.append(ItemVariant(
            item_id=_gen_item_id(rng),
            color=rng.choice(OPTION_COLORS),
            size=rng.choice(OPTION_SIZES),
            material=rng.choice(OPTION_MATERIALS),
            price=round(rng.uniform(20, 300), 2),
            available=True,
        ))
    return ProductEntity(product_id=pid, name=name, variants=variants)


def _gen_order(rng: random.Random, status: str, products: List[ProductEntity],
               n_items: int) -> OrderEntity:
    """Generate an order with n_items drawn from the product catalog."""
    items = []
    used_products = set()
    for _ in range(n_items):
        # Pick a product not yet in this order
        candidates = [p for p in products if p.product_id not in used_products]
        if not candidates:
            candidates = products
        product = rng.choice(candidates)
        used_products.add(product.product_id)
        variant = rng.choice(product.variants)
        items.append(OrderItem(
            item_id=variant.item_id,
            product_id=product.product_id,
            name=product.name,
            price=variant.price,
        ))
    return OrderEntity(
        order_id=_gen_order_id(rng),
        status=status,
        items=items,
        address=_gen_address(rng),
        payment_method_id=f"pay_{rng.randint(1000, 9999)}",
    )


# =============================================================================
# Template-based operation generators
# =============================================================================

def _tmpl_cross_order_ops(rng: random.Random, orders: List[OrderEntity],
                          products: List[ProductEntity]) -> List[TaskOperation]:
    """Simple independent operations across different orders."""
    ops = []
    for order in orders:
        if order.status == "pending":
            if rng.random() < 0.5:
                reason = rng.choice(CANCEL_REASONS)
                ops.append(TaskOperation(
                    description=f"cancel order {order.order_id} (reason: {reason})",
                    order_id=order.order_id,
                    tool_name="cancel_pending_order",
                    tool_args={"order_id": order.order_id, "reason": reason},
                ))
            else:
                new_addr = _gen_address(rng)
                while new_addr.street == order.address.street:
                    new_addr = _gen_address(rng)
                ops.append(TaskOperation(
                    description=(
                        f"change the shipping address of order {order.order_id} to "
                        f"{new_addr.street}, {new_addr.city}, {new_addr.state} {new_addr.zip_code}"
                    ),
                    order_id=order.order_id,
                    tool_name="modify_pending_order_address",
                    tool_args={"order_id": order.order_id, **new_addr.to_dict()},
                ))
        else:  # delivered
            item = rng.choice(order.items)
            ops.append(TaskOperation(
                description=(
                    f"return the {item.name} (item {item.item_id}) from order "
                    f"{order.order_id}, refund to {order.payment_method_id}"
                ),
                order_id=order.order_id,
                tool_name="return_delivered_order_items",
                tool_args={
                    "order_id": order.order_id,
                    "item_ids": [item.item_id],
                    "payment_method_id": order.payment_method_id,
                },
            ))
    return ops


def _tmpl_multi_item_batch(rng: random.Random, orders: List[OrderEntity],
                           products: List[ProductEntity]) -> Optional[TaskOperation]:
    """Exchange/return/modify MULTIPLE items from ONE order in a single call."""
    # Find an order with 2+ items
    candidates = [o for o in orders if len(o.items) >= 2]
    if not candidates:
        return None

    order = rng.choice(candidates)
    items_to_act = rng.sample(order.items, min(len(order.items), rng.randint(2, 3)))

    if order.status == "delivered":
        if rng.random() < 0.6:
            # Multi-item exchange — specify target variants so model knows what to change to
            new_item_ids = []
            swap_descs = []
            for oi in items_to_act:
                product = next((p for p in products if p.product_id == oi.product_id), None)
                if product:
                    other_variants = [v for v in product.variants
                                      if v.item_id != oi.item_id and v.available]
                    if other_variants:
                        chosen = rng.choice(other_variants)
                        new_item_ids.append(chosen.item_id)
                        swap_descs.append(
                            f"{oi.name} ({oi.item_id}) to the "
                            f"{chosen.color} {chosen.size} variant ({chosen.item_id})"
                        )
                    else:
                        new_item_ids.append(oi.item_id)
                        swap_descs.append(f"{oi.name} ({oi.item_id}) (keep same)")
                else:
                    new_item_ids.append(oi.item_id)
                    swap_descs.append(f"{oi.name} ({oi.item_id}) (keep same)")

            swap_list = "; ".join(swap_descs)
            return TaskOperation(
                description=(
                    f"exchange ALL of these items from order {order.order_id} in one request: "
                    f"{swap_list}"
                ),
                order_id=order.order_id,
                tool_name="exchange_delivered_order_items",
                tool_args={
                    "order_id": order.order_id,
                    "item_ids": [i.item_id for i in items_to_act],
                    "new_item_ids": new_item_ids,
                    "payment_method_id": order.payment_method_id,
                },
            )
        else:
            # Multi-item return — no new_item_ids needed
            item_descs = ", ".join(f"{i.name} ({i.item_id})" for i in items_to_act)
            return TaskOperation(
                description=(
                    f"return ALL of these items from order {order.order_id} in one request: "
                    f"{item_descs}"
                ),
                order_id=order.order_id,
                tool_name="return_delivered_order_items",
                tool_args={
                    "order_id": order.order_id,
                    "item_ids": [i.item_id for i in items_to_act],
                    "payment_method_id": order.payment_method_id,
                },
            )
    else:  # pending — multi-item modify — specify target variants
        new_item_ids = []
        swap_descs = []
        for oi in items_to_act:
            product = next((p for p in products if p.product_id == oi.product_id), None)
            if product:
                other_variants = [v for v in product.variants
                                  if v.item_id != oi.item_id and v.available]
                if other_variants:
                    chosen = rng.choice(other_variants)
                    new_item_ids.append(chosen.item_id)
                    swap_descs.append(
                        f"{oi.name} ({oi.item_id}) to the "
                        f"{chosen.color} {chosen.size} variant ({chosen.item_id})"
                    )
                else:
                    new_item_ids.append(oi.item_id)
                    swap_descs.append(f"{oi.name} ({oi.item_id}) (keep same)")
            else:
                new_item_ids.append(oi.item_id)
                swap_descs.append(f"{oi.name} ({oi.item_id}) (keep same)")

        swap_list = "; ".join(swap_descs)
        return TaskOperation(
            description=(
                f"modify ALL of these items in order {order.order_id} in one request: "
                f"{swap_list}"
            ),
            order_id=order.order_id,
            tool_name="modify_pending_order_items",
            tool_args={
                "order_id": order.order_id,
                "item_ids": [i.item_id for i in items_to_act],
                "new_item_ids": new_item_ids,
                "payment_method_id": order.payment_method_id,
            },
        )


def _tmpl_address_propagation(rng: random.Random, orders: List[OrderEntity],
                              products: List[ProductEntity]) -> List[TaskOperation]:
    """Update address on ALL pending orders."""
    pending = [o for o in orders if "pending" in o.status]
    if len(pending) < 2:
        return []

    new_addr = _gen_address(rng)
    ops = []
    for order in pending:
        ops.append(TaskOperation(
            description=f"update address on order {order.order_id}",
            order_id=order.order_id,
            tool_name="modify_pending_order_address",
            tool_args={"order_id": order.order_id, **new_addr.to_dict()},
        ))

    # The user description is a single instruction for all
    addr_str = f"{new_addr.street}, {new_addr.city}, {new_addr.state} {new_addr.zip_code}"
    for op in ops:
        op.description = (
            f"update the shipping address on ALL pending orders to {addr_str}"
        )
    return ops


def _tmpl_cross_ref_address(rng: random.Random, orders: List[OrderEntity],
                            products: List[ProductEntity]) -> Optional[TaskOperation]:
    """Change one order's address to match another order's address."""
    pending = [o for o in orders if "pending" in o.status]
    others = [o for o in orders if o not in pending]
    if not pending or not others:
        # Fall back: use any two different orders
        if len(orders) < 2:
            return None
        source, target = rng.sample(orders, 2)
        if "pending" not in target.status:
            source, target = target, source
        if "pending" not in target.status:
            return None
    else:
        target = rng.choice(pending)
        source = rng.choice(others)

    addr = source.address
    return TaskOperation(
        description=(
            f"change order {target.order_id}'s shipping address to match "
            f"the address on order {source.order_id} "
            f"(you'll need to look up order {source.order_id} first to get the address)"
        ),
        order_id=target.order_id,
        tool_name="modify_pending_order_address",
        tool_args={"order_id": target.order_id, **addr.to_dict()},
        requires_lookup=True,
    )


def _tmpl_status_dependent(rng: random.Random, orders: List[OrderEntity],
                           products: List[ProductEntity]) -> Optional[TaskOperation]:
    """User says 'change item X' — agent must check status to pick the right tool."""
    order = rng.choice(orders)
    item = rng.choice(order.items)

    product = next((p for p in products if p.product_id == item.product_id), None)
    if not product:
        return None
    other_variants = [v for v in product.variants
                      if v.item_id != item.item_id and v.available]
    if not other_variants:
        return None
    new_variant = rng.choice(other_variants)

    # Use ambiguous wording — agent must determine tool from status
    if order.status == "pending":
        tool_name = "modify_pending_order_items"
    else:
        tool_name = "exchange_delivered_order_items"

    return TaskOperation(
        description=(
            f"change the {item.name} ({item.item_id}) in order {order.order_id} "
            f"to the {new_variant.color} {new_variant.size} variant ({new_variant.item_id})"
        ),
        order_id=order.order_id,
        tool_name=tool_name,
        tool_args={
            "order_id": order.order_id,
            "item_ids": [item.item_id],
            "new_item_ids": [new_variant.item_id],
            "payment_method_id": order.payment_method_id,
        },
        requires_lookup=True,
    )


# =============================================================================
# Episode generation
# =============================================================================

def generate_episode(seed: int, difficulty: int = 3) -> EpisodeScenario:
    rng = random.Random(seed)
    difficulty = max(1, min(5, difficulty))
    config = DIFFICULTY_CONFIG[difficulty]

    n_orders = rng.randint(*config["orders"])
    n_items_per = config["items_per_order"]
    target_ops = rng.randint(*config["ops"])

    # Generate product catalog (enough for all orders)
    n_products = max(n_orders * 3, 8)
    product_names = rng.sample(PRODUCT_TYPES, min(n_products, len(PRODUCT_TYPES)))
    products = [_gen_product(rng, name, num_variants=rng.randint(3, 6))
                for name in product_names]

    # Generate orders with mixed statuses
    orders = []
    for _ in range(n_orders):
        status = rng.choice(["pending", "delivered"])
        n_items = rng.randint(*n_items_per)
        orders.append(_gen_order(rng, status, products, n_items))

    # Ensure at least one pending and one delivered for interesting scenarios
    if difficulty >= 3:
        statuses = [o.status for o in orders]
        if "pending" not in statuses:
            orders[0].status = "pending"
        if "delivered" not in statuses and len(orders) > 1:
            orders[1].status = "delivered"

    # Select operations from available templates
    available_templates = list(config["templates"])
    operations: List[TaskOperation] = []
    used_orders: Dict[str, str] = {}  # order_id -> tool used (for one-shot tracking)
    incremental_ops: List[TaskOperation] = []

    rng.shuffle(available_templates)

    for template_name in available_templates:
        if len(operations) + len(incremental_ops) >= target_ops:
            break

        if template_name == "cross_order_ops":
            # Pick one or two orders not yet used
            free_orders = [o for o in orders if o.order_id not in used_orders]
            if free_orders:
                subset = rng.sample(free_orders, min(len(free_orders), 2))
                for op in _tmpl_cross_order_ops(rng, subset, products):
                    if len(operations) + len(incremental_ops) < target_ops:
                        operations.append(op)
                        used_orders[op.order_id] = op.tool_name

        elif template_name == "multi_item_batch":
            free_orders = [o for o in orders if o.order_id not in used_orders
                           and len(o.items) >= 2]
            if free_orders:
                op = _tmpl_multi_item_batch(rng, free_orders, products)
                if op:
                    operations.append(op)
                    used_orders[op.order_id] = op.tool_name

        elif template_name == "address_propagation":
            pending_free = [o for o in orders if "pending" in o.status
                            and o.order_id not in used_orders]
            if len(pending_free) >= 2:
                addr_ops = _tmpl_address_propagation(rng, orders, products)
                # Only add ops for orders not yet used
                for op in addr_ops:
                    if op.order_id not in used_orders:
                        if len(operations) + len(incremental_ops) < target_ops:
                            operations.append(op)
                            used_orders[op.order_id] = op.tool_name

        elif template_name == "cross_ref_address":
            pending_free = [o for o in orders if "pending" in o.status
                            and o.order_id not in used_orders]
            if pending_free:
                op = _tmpl_cross_ref_address(rng, orders, products)
                if op and op.order_id not in used_orders:
                    operations.append(op)
                    used_orders[op.order_id] = op.tool_name

        elif template_name == "status_dependent":
            free_orders = [o for o in orders if o.order_id not in used_orders]
            if free_orders:
                # Temporarily override to use a free order
                saved = orders
                op = _tmpl_status_dependent(rng, free_orders, products)
                if op and op.order_id not in used_orders:
                    operations.append(op)
                    used_orders[op.order_id] = op.tool_name

        elif template_name == "incremental_reveal":
            free_orders = [o for o in orders if o.order_id not in used_orders]
            if free_orders and len(operations) >= 2:
                # Generate one op to be revealed mid-conversation
                subset = [rng.choice(free_orders)]
                cross_ops = _tmpl_cross_order_ops(rng, subset, products)
                if cross_ops:
                    op = cross_ops[0]
                    op.is_incremental = True
                    incremental_ops.append(op)
                    used_orders[op.order_id] = op.tool_name

    # Pad with simple cross-order ops if we haven't hit target
    while len(operations) + len(incremental_ops) < max(2, target_ops):
        free_orders = [o for o in orders if o.order_id not in used_orders]
        if not free_orders:
            break
        subset = [rng.choice(free_orders)]
        for op in _tmpl_cross_order_ops(rng, subset, products):
            operations.append(op)
            used_orders[op.order_id] = op.tool_name
            break

    # Build user message descriptions
    # Deduplicate address propagation descriptions
    seen_descs = set()
    initial_descs = []
    for i, op in enumerate(operations):
        if op.description not in seen_descs:
            seen_descs.add(op.description)
            initial_descs.append(f"Task {len(initial_descs) + 1}: {op.description}")

    incr_descs = []
    for op in incremental_ops:
        incr_descs.append(op.description)

    all_ops = operations + incremental_ops

    return EpisodeScenario(
        orders=orders,
        products=products,
        operations=all_ops,
        initial_task_descriptions=initial_descs,
        incremental_task_descriptions=incr_descs,
        difficulty=difficulty,
    )


# =============================================================================
# Tool simulation (stateful)
# =============================================================================

def _serialize_order(order: OrderEntity) -> str:
    return json.dumps({
        "order_id": order.order_id,
        "status": order.status,
        "items": [{"item_id": i.item_id, "product_id": i.product_id,
                    "name": i.name, "price": i.price} for i in order.items],
        "address": order.address.to_dict(),
        "payment_method_id": order.payment_method_id,
    }, indent=2)


def _serialize_product(product: ProductEntity) -> str:
    return json.dumps({
        "product_id": product.product_id,
        "name": product.name,
        "variants": [{"item_id": v.item_id, "color": v.color, "size": v.size,
                       "material": v.material, "price": v.price,
                       "available": v.available} for v in product.variants],
    }, indent=2)


def _normalize_order_id(oid: str, orders: List[OrderEntity]) -> Optional[OrderEntity]:
    """Find order by ID, tolerant of missing '#' prefix."""
    for o in orders:
        if o.order_id == oid or o.order_id == f"#{oid}" or o.order_id.lstrip("#") == oid.lstrip("#"):
            return o
    return None


def execute_tool(tool_name: str, args: Dict[str, Any],
                 orders: List[OrderEntity],
                 products: List[ProductEntity]) -> Tuple[bool, str]:
    """Execute a tool call against mutable state. Returns (success, result_json)."""

    if tool_name == "get_order_details":
        oid = args.get("order_id", "")
        order = _normalize_order_id(oid, orders)
        if order is None:
            return False, json.dumps({"error": f"Order {oid} not found."})
        return True, _serialize_order(order)

    if tool_name == "get_product_details":
        pid = args.get("product_id", "")
        product = next((p for p in products if p.product_id == pid), None)
        if product is None:
            return False, json.dumps({"error": f"Product {pid} not found."})
        return True, _serialize_product(product)

    if tool_name == "cancel_pending_order":
        oid = args.get("order_id", "")
        order = _normalize_order_id(oid, orders)
        if order is None:
            return False, json.dumps({"error": f"Order {oid} not found."})
        if order.status != "pending":
            return False, json.dumps({
                "error": f"Cannot cancel: status is '{order.status}', must be 'pending'."
            })
        reason = args.get("reason", "")
        if reason not in CANCEL_REASONS:
            return False, json.dumps({
                "error": f"Invalid reason. Must be one of: {CANCEL_REASONS}"
            })
        order.status = "cancelled"
        return True, json.dumps({"success": True,
                                  "message": f"Order {oid} cancelled. Reason: {reason}."})

    if tool_name == "modify_pending_order_items":
        oid = args.get("order_id", "")
        order = _normalize_order_id(oid, orders)
        if order is None:
            return False, json.dumps({"error": f"Order {oid} not found."})
        # EXACT match — "pending (modified)" will fail
        if order.status != "pending":
            return False, json.dumps({
                "error": f"Cannot modify items: status is '{order.status}', must be exactly 'pending'."
            })
        item_ids = args.get("item_ids", [])
        new_item_ids = args.get("new_item_ids", [])
        if len(item_ids) != len(new_item_ids):
            return False, json.dumps({"error": "item_ids and new_item_ids must have same length."})
        valid_ids = {i.item_id for i in order.items}
        invalid = [iid for iid in item_ids if iid not in valid_ids]
        if invalid:
            return False, json.dumps({"error": f"Invalid item_ids: {invalid}. Valid: {sorted(valid_ids)}"})
        order.status = "pending (modified)"
        return True, json.dumps({"success": True,
                                  "message": f"Items modified in order {oid}. Status is now 'pending (modified)'."})

    if tool_name == "modify_pending_order_address":
        oid = args.get("order_id", "")
        order = _normalize_order_id(oid, orders)
        if order is None:
            return False, json.dumps({"error": f"Order {oid} not found."})
        # SUBSTRING match — works for "pending" and "pending (modified)"
        if "pending" not in order.status:
            return False, json.dumps({
                "error": f"Cannot modify address: status is '{order.status}', must contain 'pending'."
            })
        required = ["address1", "city", "state", "zip"]
        missing = [k for k in required if not args.get(k)]
        if missing:
            return False, json.dumps({"error": f"Missing required fields: {missing}"})
        order.address = Address(args["address1"], args["city"], args["state"], args["zip"])
        return True, json.dumps({"success": True, "message": f"Address updated for order {oid}."})

    if tool_name == "return_delivered_order_items":
        oid = args.get("order_id", "")
        order = _normalize_order_id(oid, orders)
        if order is None:
            return False, json.dumps({"error": f"Order {oid} not found."})
        # EXACT match
        if order.status != "delivered":
            return False, json.dumps({
                "error": f"Cannot return: status is '{order.status}', must be exactly 'delivered'."
            })
        item_ids = args.get("item_ids", [])
        valid_ids = {i.item_id for i in order.items}
        invalid = [iid for iid in item_ids if iid not in valid_ids]
        if invalid:
            return False, json.dumps({"error": f"Invalid item_ids: {invalid}. Valid: {sorted(valid_ids)}"})
        if not item_ids:
            return False, json.dumps({"error": "item_ids cannot be empty."})
        order.status = "return requested"
        return True, json.dumps({"success": True,
                                  "message": f"Return requested for items {item_ids} from order {oid}."})

    if tool_name == "exchange_delivered_order_items":
        oid = args.get("order_id", "")
        order = _normalize_order_id(oid, orders)
        if order is None:
            return False, json.dumps({"error": f"Order {oid} not found."})
        # EXACT match
        if order.status != "delivered":
            return False, json.dumps({
                "error": f"Cannot exchange: status is '{order.status}', must be exactly 'delivered'."
            })
        item_ids = args.get("item_ids", [])
        new_item_ids = args.get("new_item_ids", [])
        if len(item_ids) != len(new_item_ids):
            return False, json.dumps({"error": "item_ids and new_item_ids must have same length."})
        valid_ids = {i.item_id for i in order.items}
        invalid = [iid for iid in item_ids if iid not in valid_ids]
        if invalid:
            return False, json.dumps({"error": f"Invalid item_ids: {invalid}. Valid: {sorted(valid_ids)}"})
        if not item_ids:
            return False, json.dumps({"error": "item_ids cannot be empty."})
        order.status = "exchange requested"
        return True, json.dumps({"success": True,
                                  "message": f"Exchange requested for order {oid}: {item_ids} -> {new_item_ids}."})

    return False, json.dumps({"error": f"Unknown tool: {tool_name}"})


# =============================================================================
# Reward computation
# =============================================================================

def _normalize_oid_str(oid: str) -> str:
    return oid.lstrip("#").strip()


def _match_tool_call(call_name: str, call_args: Dict, op: TaskOperation) -> bool:
    if call_name != op.tool_name:
        return False
    if _normalize_oid_str(call_args.get("order_id", "")) != _normalize_oid_str(op.tool_args.get("order_id", "")):
        return False

    if call_name == "cancel_pending_order":
        return bool(call_args.get("reason"))

    if call_name in ("return_delivered_order_items", "exchange_delivered_order_items",
                      "modify_pending_order_items"):
        expected_ids = set(op.tool_args.get("item_ids", []))
        provided_ids = set(call_args.get("item_ids", []))
        if expected_ids != provided_ids:
            return False
        if call_name != "return_delivered_order_items":
            expected_new = set(op.tool_args.get("new_item_ids", []))
            provided_new = set(call_args.get("new_item_ids", []))
            if expected_new != provided_new:
                return False
        return bool(call_args.get("payment_method_id"))

    if call_name == "modify_pending_order_address":
        expected_addr = Address(
            op.tool_args.get("address1", ""), op.tool_args.get("city", ""),
            op.tool_args.get("state", ""), op.tool_args.get("zip", ""),
        )
        return expected_addr.matches(call_args)

    return False


def compute_reward(
    tool_calls: List[Dict[str, Any]],
    operations: List[TaskOperation],
    one_shot_violations: int,
    wrong_tool_type_count: int,
    skipped_lookups: int,
) -> Tuple[float, str]:
    n_ops = len(operations)
    if n_ops == 0:
        return 0.0, "No operations expected."

    completed = [False] * n_ops
    wrong_calls = 0

    for call in tool_calls:
        call_name = call.get("name", "")
        call_args = call.get("arguments", {})
        if call_name in ("respond_to_user", "get_order_details", "get_product_details"):
            continue
        matched = False
        for i, op in enumerate(operations):
            if not completed[i] and _match_tool_call(call_name, call_args, op):
                completed[i] = True
                matched = True
                break
        if not matched:
            wrong_calls += 1

    correct_count = sum(completed)

    # --- Reward in [0, 1]: completion bonus + linear partial credit ---
    #
    # Linear partial credit (max 0.6) with a 0.4 completion bonus for all ops.
    # Creates strong GRPO signal (0.5 gap between perfect and 1-miss)
    # while providing gradient across partial completions.
    #
    # Penalties are sized so completion always dominates: all-ops + N skips
    # always beats miss-1-op + 0 skips. Skip penalty at 0.15 prevents
    # stacking from wiping out perfect completions (old 0.30 did).
    #
    # Reward landscape (5-op example):
    #   5/5 ops, 0 skips: 1.00   |  4/5 ops, 0 skips: 0.48
    #   5/5 ops, 1 skip:  0.85   |  4/5 ops, 1 skip:  0.33
    #   5/5 ops, 2 skips: 0.70   |  3/5 ops, 0 skips: 0.36
    #   5/5 ops, 4 skips: 0.40   |  (completion always > skip penalty)

    if correct_count == n_ops:
        reward = 1.0
    else:
        reward = 0.6 * (correct_count / n_ops)

    # Penalties (applied subtractively, then clamped to [0, 1])
    reward -= 0.15 * skipped_lookups
    reward -= 0.15 * one_shot_violations
    reward -= 0.1 * wrong_tool_type_count
    reward -= 0.1 * wrong_calls

    reward = max(0.0, min(1.0, round(reward, 3)))

    details = []
    for i, (op, done) in enumerate(zip(operations, completed)):
        details.append(f"Op{i+1} ({op.tool_name}): {'DONE' if done else 'MISSED'}")
    if one_shot_violations:
        details.append(f"One-shot violations: {one_shot_violations}")
    if wrong_tool_type_count:
        details.append(f"Wrong tool type: {wrong_tool_type_count}")
    if wrong_calls:
        details.append(f"Wrong calls: {wrong_calls}")
    if skipped_lookups:
        details.append(f"Skipped lookups: {skipped_lookups}")

    reason = f"{correct_count}/{n_ops} ops completed. " + "; ".join(details)
    return reward, reason



# =============================================================================
# Game class
# =============================================================================

class RealisticMultiStepGame:
    """Multi-turn task game for sequential task completion.

    Tests:
    - Information must be retrieved via tool calls (not given upfront)
    - One-shot constraints on modify/exchange/return (status changes block re-calls)
    - Multi-item batching required
    - Cross-entity referencing
    - Incremental user reveals
    - Completing all N operations without stopping early
    """

    def __init__(self, max_steps: int = 30, difficulty: int = 3):
        self._max_steps = max_steps
        self._difficulty = difficulty
        self._episode: Optional[EpisodeScenario] = None
        self._conversation: List[Dict[str, str]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._rng: Optional[random.Random] = None

        # Tracking for reward penalties
        self._one_shot_violations: int = 0
        self._wrong_tool_type_count: int = 0
        self._orders_looked_up: set = set()
        self._orders_written_to: set = set()  # orders that had a write tool called
        self._one_shot_used: Dict[str, str] = {}  # order_id -> tool that used it

        # Incremental reveal state
        self._incremental_revealed: bool = False
        self._ops_completed_before_reveal: int = 0

        # GameEnv protocol
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed + 999)
        self._episode = generate_episode(seed, self._difficulty)

        user_name = random.Random(seed).choice(FIRST_NAMES)
        task_list = "\n".join(self._episode.initial_task_descriptions)
        order_ids = ", ".join(o.order_id for o in self._episode.orders)
        initial_msg = (
            f"Hi, I'm {user_name}. I need help with my orders ({order_ids}).\n\n"
            f"{task_list}\n\n"
            f"Please complete all of these tasks."
        )

        self._conversation = [{"role": "user", "text": initial_msg}]
        self._tool_calls = []
        self._step_count = 0
        self._one_shot_violations = 0
        self._wrong_tool_type_count = 0
        self._orders_looked_up = set()
        self._orders_written_to = set()
        self._one_shot_used = {}
        self._incremental_revealed = False
        self._ops_completed_before_reveal = 0

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        ep = self._episode
        if ep is None:
            return "No episode loaded."

        tools_json = json.dumps(TOOL_SCHEMAS, indent=2)

        lines = [
            "You are a customer service agent. You can help users cancel or modify "
            "pending orders, return or exchange delivered orders, and look up order "
            "and product information.",
            "",
            "<available_tools>",
            tools_json,
            "</available_tools>",
            "",
            "<conversation>",
        ]

        for msg in self._conversation:
            role = msg["role"].upper()
            text = msg["text"]
            lines.append(f"[{role}]: {text}")

        lines.extend([
            "</conversation>",
            "",
            "Execute ONE tool call at a time. Respond with JSON:",
            '{"name": "<tool_name>", "arguments": {<args>}}',
        ])
        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]) -> None:
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

        # respond_to_user → end episode
        if tool_name == "respond_to_user":
            message = tool_args.get("message", "")
            self._conversation.append({"role": "assistant", "text": message})
            reward, reason = compute_reward(
                self._tool_calls, self._episode.operations,
                self._one_shot_violations, self._wrong_tool_type_count,
                self._count_skipped_lookups(),
            )
            self._finalize(reward, reason)
            return

        # Record tool call
        self._tool_calls.append(tool_call)
        self._conversation.append({"role": "tool_call", "text": json.dumps(tool_call)})

        order_id = tool_args.get("order_id", "")

        # Track lookups
        if tool_name == "get_order_details":
            self._orders_looked_up.add(order_id)

        # Track one-shot violations
        ONE_SHOT_TOOLS = {
            "modify_pending_order_items", "exchange_delivered_order_items",
            "return_delivered_order_items",
        }
        if tool_name in ONE_SHOT_TOOLS:
            self._orders_written_to.add(order_id)
            if order_id in self._one_shot_used:
                self._one_shot_violations += 1
            else:
                self._one_shot_used[order_id] = tool_name

        # Track wrong tool type (exchange on pending, modify on delivered)
        if tool_name == "exchange_delivered_order_items":
            order = _normalize_order_id(order_id, self._episode.orders)
            if order and "pending" in order.status:
                self._wrong_tool_type_count += 1
        elif tool_name == "modify_pending_order_items":
            order = _normalize_order_id(order_id, self._episode.orders)
            if order and order.status == "delivered":
                self._wrong_tool_type_count += 1

        # Execute tool
        success, result = execute_tool(
            tool_name, tool_args,
            self._episode.orders, self._episode.products,
        )
        self._conversation.append({"role": "tool_result", "text": result})

        # Simulated user confirmation after successful write
        if success and tool_name not in ("get_order_details", "get_product_details"):
            self._maybe_reveal_incremental()
            if self._rng:
                confirmation = self._rng.choice(USER_CONFIRMATIONS)
                self._conversation.append({"role": "user", "text": confirmation})

        # Loop detection
        if len(self._tool_calls) >= 3:
            last3 = self._tool_calls[-3:]
            keys = [json.dumps(tc, sort_keys=True) for tc in last3]
            if keys[0] == keys[1] == keys[2]:
                reward, reason = compute_reward(
                    self._tool_calls, self._episode.operations,
                    self._one_shot_violations, self._wrong_tool_type_count,
                    self._count_skipped_lookups(),
                )
                self._finalize(reward, f"Loop detected. {reason}")
                return

        # Max steps
        if self._step_count >= self._max_steps:
            reward, reason = compute_reward(
                self._tool_calls, self._episode.operations,
                self._one_shot_violations, self._wrong_tool_type_count,
                self._count_skipped_lookups(),
            )
            self._finalize(reward, f"Max steps reached. {reason}")

    def _maybe_reveal_incremental(self) -> None:
        """Check if it's time to reveal the incremental task."""
        if self._incremental_revealed:
            return
        ep = self._episode
        if not ep.incremental_task_descriptions:
            return

        # Count how many non-incremental ops are done
        non_incr_ops = [op for op in ep.operations if not op.is_incremental]
        completed = 0
        for op in non_incr_ops:
            for call in self._tool_calls:
                if _match_tool_call(call.get("name", ""), call.get("arguments", {}), op):
                    completed += 1
                    break

        # Reveal when at least half of initial ops are done
        if completed >= max(1, len(non_incr_ops) // 2):
            self._incremental_revealed = True
            for desc in ep.incremental_task_descriptions:
                template = self._rng.choice(INCREMENTAL_ADDITIONS)
                msg = template.format(task_desc=desc)
                self._conversation.append({"role": "user", "text": msg})

    def _count_skipped_lookups(self) -> int:
        """Count orders that were written to without being looked up first."""
        return len(self._orders_written_to - self._orders_looked_up)

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    def get_summary(self) -> Dict[str, Any]:
        ep = self._episode
        return {
            "n_ops": len(ep.operations) if ep else 0,
            "difficulty": ep.difficulty if ep else 0,
            "task_descriptions": ep.initial_task_descriptions + ep.incremental_task_descriptions if ep else [],
            "steps": self._step_count,
            "tool_calls": self._tool_calls,
            "reward": self.rewards.get(0, 0.0),
            "reason": getattr(self, "_reason", ""),
            "one_shot_violations": self._one_shot_violations,
            "wrong_tool_type": self._wrong_tool_type_count,
            "skipped_lookups": self._count_skipped_lookups(),
        }


# =============================================================================
# Action parsing (shared with v1)
# =============================================================================

def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
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
    import re
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


# =============================================================================
# System prompt
# =============================================================================

SYSTEM_PROMPT = (
    "You are a customer service agent. You can help users cancel or modify "
    "pending orders, return or exchange delivered orders, and look up order "
    "and product information.\n"
    "Respond with one JSON tool call at a time.\n"
)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Testing Realistic Multi-Step Task Game")
    print("=" * 70)

    for diff in [1, 2, 3, 4, 5]:
        rewards = []
        template_counts: Dict[str, int] = {}
        game = RealisticMultiStepGame(max_steps=30, difficulty=diff)

        for seed in range(100):
            game.reset(seed)
            ep = game._episode

            # Perfect agent: look up all orders, then execute all ops
            for order in ep.orders:
                action = json.dumps({
                    "name": "get_order_details",
                    "arguments": {"order_id": order.order_id},
                })
                game.step(action)
                if game.done:
                    break

            if not game.done:
                # Look up products needed for conditional/exchange ops
                looked_up_products = set()
                for op in ep.operations:
                    if op.requires_lookup and op.tool_name in (
                        "exchange_delivered_order_items", "modify_pending_order_items"
                    ):
                        for item_id in op.tool_args.get("item_ids", []):
                            for o in ep.orders:
                                for oi in o.items:
                                    if oi.item_id == item_id and oi.product_id not in looked_up_products:
                                        action = json.dumps({
                                            "name": "get_product_details",
                                            "arguments": {"product_id": oi.product_id},
                                        })
                                        game.step(action)
                                        looked_up_products.add(oi.product_id)

            if not game.done:
                for op in ep.operations:
                    action = json.dumps({
                        "name": op.tool_name,
                        "arguments": op.tool_args,
                    })
                    game.step(action)
                    if game.done:
                        break

            if not game.done:
                action = json.dumps({
                    "name": "respond_to_user",
                    "arguments": {"message": "All tasks completed."},
                })
                game.step(action)

            rewards.append(game.rewards[0])

            # Count templates used
            for op in ep.operations:
                tname = op.tool_name
                template_counts[tname] = template_counts.get(tname, 0) + 1

        avg = sum(rewards) / len(rewards)
        perfect = sum(1 for r in rewards if r >= 1.0)
        non_perfect = [(seed, r) for seed, r in enumerate(rewards) if r < 1.0]
        print(f"\nDifficulty {diff}: avg_reward={avg:.3f}, perfect={perfect}/100")
        print(f"  Tool distribution: {template_counts}")
        if non_perfect:
            print(f"  Non-perfect seeds: {non_perfect[:5]}")

    # Show example observation
    print("\n" + "=" * 70)
    print("Example episode (difficulty=3, seed=42):")
    game = RealisticMultiStepGame(max_steps=30, difficulty=3)
    game.reset(42)
    obs = game.observe(0)
    print(obs[:3000])
    print("..." if len(obs) > 3000 else "")
    print(f"\nExpected operations:")
    for i, op in enumerate(game._episode.operations):
        print(f"  {i+1}. {op.tool_name}({json.dumps(op.tool_args)[:100]})")
        print(f"     requires_lookup={op.requires_lookup}, incremental={op.is_incremental}")
