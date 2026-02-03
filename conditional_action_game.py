"""
Conditional Action Game - Trains conditional reasoning with strict constraint adherence.

Key insight from tau-bench failure analysis:
- Models fail when user says "If X unavailable, ONLY do Y"
- Model incorrectly does both X-similar AND Y instead of ONLY Y

This game trains:
1. Understanding conditional requirements ("if... then ONLY...")
2. Checking conditions via tool calls
3. Executing EXACTLY the action specified by the condition result
4. NOT doing extra actions beyond what's specified

Example scenario:
- User: "If product A is in stock, buy A. Otherwise, ONLY buy B."
- Model must: Check stock → Buy ONLY the correct item(s) based on condition
"""

import random
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Data Models
# =============================================================================

class ActionType(str, Enum):
    EXCHANGE = "exchange"
    CANCEL = "cancel"
    MODIFY = "modify"


@dataclass
class Product:
    product_id: str
    name: str
    variant: str
    in_stock: bool
    price: float


@dataclass
class OrderItem:
    item_id: str
    product: Product


@dataclass
class Order:
    order_id: str
    items: List[OrderItem]
    status: str  # "pending" or "delivered"


@dataclass
class ConditionalTask:
    """
    A task with conditional logic:
    - If primary_condition is True: do primary_action with primary_items
    - Else: do fallback_action with fallback_items ONLY
    """
    user_name: str
    order: Order

    # The condition to check (e.g., "is product X in stock?")
    primary_product_id: str  # Product to check availability for
    primary_condition_met: bool  # Whether the product is in stock

    # Primary action (if condition met)
    primary_action: ActionType
    primary_item_ids: List[str]
    primary_new_item_ids: List[str]  # For exchanges

    # Fallback action (if condition NOT met) - ONLY this should be done
    fallback_action: ActionType
    fallback_item_ids: List[str]
    fallback_new_item_ids: List[str]

    # Ground truth (which action should actually be taken)
    expected_action: ActionType
    expected_item_ids: List[str]
    expected_new_item_ids: List[str]


# =============================================================================
# Data Generation
# =============================================================================

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery"]

PRODUCTS = [
    ("Wireless Headphones", ["black", "white", "blue"]),
    ("Mechanical Keyboard", ["RGB backlight", "white backlight", "no backlight"]),
    ("Smart Watch", ["leather band", "metal band", "silicone band"]),
    ("Laptop Stand", ["aluminum", "wooden", "plastic"]),
    ("USB-C Hub", ["4-port", "7-port", "10-port"]),
]


def generate_product_id(rng: random.Random) -> str:
    return str(rng.randint(1000000, 9999999))


def generate_order_id(rng: random.Random) -> str:
    return f"#W{rng.randint(100000, 999999)}"


def generate_item_id(rng: random.Random) -> str:
    return f"item_{rng.randint(100000, 999999)}"


def generate_task(rng: random.Random) -> ConditionalTask:
    """Generate a conditional action task."""

    user_name = rng.choice(FIRST_NAMES)

    # Create order with 2-3 items
    order_id = generate_order_id(rng)
    num_items = rng.randint(2, 3)

    items = []
    for _ in range(num_items):
        product_name, variants = rng.choice(PRODUCTS)
        variant = rng.choice(variants)
        product = Product(
            product_id=generate_product_id(rng),
            name=product_name,
            variant=variant,
            in_stock=True,  # Original products are in stock
            price=round(rng.uniform(30, 200), 2)
        )
        items.append(OrderItem(
            item_id=generate_item_id(rng),
            product=product
        ))

    order = Order(
        order_id=order_id,
        items=items,
        status="delivered"  # For exchanges
    )

    # Decide condition (50/50 whether primary condition is met)
    primary_condition_met = rng.random() < 0.5

    # Primary target: first item
    # Fallback target: second item
    primary_item = items[0]
    fallback_item = items[1]

    # Generate alternate product (for exchanges)
    primary_product_name, primary_variants = rng.choice(PRODUCTS)
    requested_variant = rng.choice(primary_variants)

    # The primary product they want
    primary_new_product = Product(
        product_id=generate_product_id(rng),
        name=primary_product_name,
        variant=requested_variant,
        in_stock=primary_condition_met,  # This is the KEY - determines condition
        price=round(rng.uniform(30, 200), 2)
    )

    # Fallback product (always available)
    fallback_product_name, fallback_variants = rng.choice(PRODUCTS)
    fallback_variant = rng.choice(fallback_variants)
    fallback_new_product = Product(
        product_id=generate_product_id(rng),
        name=fallback_product_name,
        variant=fallback_variant,
        in_stock=True,
        price=round(rng.uniform(30, 200), 2)
    )

    # Both actions are exchanges for simplicity
    primary_action = ActionType.EXCHANGE
    fallback_action = ActionType.EXCHANGE

    # Determine expected outcome
    if primary_condition_met:
        expected_action = primary_action
        expected_item_ids = [primary_item.item_id]
        expected_new_item_ids = [primary_new_product.product_id]
    else:
        expected_action = fallback_action
        expected_item_ids = [fallback_item.item_id]
        expected_new_item_ids = [fallback_new_product.product_id]

    return ConditionalTask(
        user_name=user_name,
        order=order,
        primary_product_id=primary_new_product.product_id,
        primary_condition_met=primary_condition_met,
        primary_action=primary_action,
        primary_item_ids=[primary_item.item_id],
        primary_new_item_ids=[primary_new_product.product_id],
        fallback_action=fallback_action,
        fallback_item_ids=[fallback_item.item_id],
        fallback_new_item_ids=[fallback_new_product.product_id],
        expected_action=expected_action,
        expected_item_ids=expected_item_ids,
        expected_new_item_ids=expected_new_item_ids,
    )


# =============================================================================
# Game State
# =============================================================================

@dataclass
class GameState:
    task: ConditionalTask
    step: int = 0
    checked_order: bool = False
    checked_stock: bool = False
    stock_result: Optional[bool] = None
    final_action: Optional[Dict] = None
    done: bool = False
    reason: str = ""


# =============================================================================
# Main Game Class
# =============================================================================

class ConditionalActionGame:
    """
    Conditional action game that trains strict constraint adherence.

    The model must:
    1. Get order details (to see items)
    2. Check product availability (to determine condition)
    3. Execute EXACTLY the action specified by the condition

    Key failure modes trained against:
    - Doing BOTH primary and fallback actions
    - Doing primary-similar when condition is NOT met
    - Ignoring the "ONLY" constraint
    """

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self._rng = random.Random()
        self._state: Optional[GameState] = None

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def reset(self, seed: int):
        """Reset with new seed."""
        self._rng = random.Random(seed)
        task = generate_task(self._rng)
        self._state = GameState(task=task)

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        """Get observation - uses natural language like real tau-bench."""
        state = self._state
        task = state.task
        order = task.order

        # Format items list
        items_str = "\n".join([
            f"    - {item.item_id}: {item.product.name} ({item.product.variant}) - ${item.product.price}"
            for item in order.items
        ])

        # Get product names for more natural language
        primary_item = order.items[0]
        fallback_item = order.items[1]

        lines = [
            "You are a customer service agent processing an exchange request.",
            "",
            "=== USER REQUEST ===",
            f"User {task.user_name} says:",
        ]

        # Natural language conditional - more like real tau-bench
        # This is where the model must understand the implicit "only" constraint
        lines.extend([
            f'"Hi, I have order {order.order_id} and I\'d like to make an exchange.',
            f'I want to swap my {primary_item.product.name} ({primary_item.product.variant}) for product {task.primary_product_id}.',
            f'However, if that product is not available, then I\'d be okay with just exchanging my {fallback_item.product.name} for product {task.fallback_new_item_ids[0]} instead.',
            f'Please check availability first."',
        ])

        lines.extend([
            "",
            "=== ORDER DETAILS ===",
            f"Order ID: {order.order_id}",
            f"Status: {order.status.upper()}",
            f"Items:",
            items_str,
            "",
            "=== AVAILABLE TOOLS ===",
            '1. check_product_stock: {"name": "check_product_stock", "arguments": {"product_id": "..."}}',
            '2. exchange_items: {"name": "exchange_items", "arguments": {"order_id": "...", "item_ids": [...], "new_item_ids": [...]}}',
            '3. respond_to_user: {"name": "respond_to_user", "arguments": {"message": "..."}}',
            "",
        ])

        # Show stock check result without explicit instructions
        if state.checked_stock:
            lines.append("=== STOCK CHECK RESULT ===")
            if state.stock_result:
                lines.append(f"Product {task.primary_product_id} is IN STOCK and available.")
            else:
                lines.append(f"Product {task.primary_product_id} is OUT OF STOCK.")
            lines.append("")

        lines.append("Respond with a JSON tool call:")

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]):
        """Process action."""
        if self.done:
            return

        state = self._state
        task = state.task
        state.step += 1

        # Parse action
        if action is None:
            self._finalize(-1.0, "No action")
            return

        tool_call = parse_tool_call(action)
        if tool_call is None:
            self._finalize(-1.0, "Invalid JSON")
            self.invalid_player = 0
            return

        tool_name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        # Handle different tools
        if tool_name == "get_order_details":
            order_id = args.get("order_id", "")
            if normalize_order_id(order_id) == task.order.order_id:
                state.checked_order = True
                # Continue - this is fine but not required
                return
            else:
                self._finalize(-1.0, f"Wrong order_id: {order_id}")
                return

        elif tool_name == "check_product_stock":
            product_id = str(args.get("product_id", ""))
            if product_id == task.primary_product_id:
                state.checked_stock = True
                state.stock_result = task.primary_condition_met
                # Continue to next step
                return
            else:
                # Wrong product checked - partial penalty
                self._finalize(-0.5, f"Checked wrong product: {product_id}")
                return

        elif tool_name == "exchange_items":
            if not state.checked_stock:
                # Acted without checking stock!
                self._finalize(-1.0, "Exchanged without checking stock")
                return

            state.final_action = tool_call

            # Extract parameters
            order_id = args.get("order_id", "")
            item_ids = args.get("item_ids", [])
            new_item_ids = args.get("new_item_ids", [])

            # Normalize
            if normalize_order_id(order_id) != task.order.order_id:
                self._finalize(-1.0, "Wrong order_id in exchange")
                return

            # Normalize item_ids (be lenient about "item_" prefix)
            def normalize_item_id(item_id: str) -> str:
                item_id = str(item_id)
                if item_id.startswith("item_"):
                    return item_id
                return f"item_{item_id}"

            # Convert to sets for comparison (with normalization)
            item_ids_set = set(normalize_item_id(x) for x in item_ids)
            new_item_ids_set = set(str(x) for x in new_item_ids)  # new_item_ids are product IDs, no prefix

            expected_item_ids_set = set(task.expected_item_ids)
            expected_new_item_ids_set = set(task.expected_new_item_ids)

            # Check if EXACTLY correct
            items_match = item_ids_set == expected_item_ids_set
            new_items_match = new_item_ids_set == expected_new_item_ids_set

            if items_match and new_items_match:
                self._finalize(1.0, "Correct conditional action")
            elif item_ids_set.issuperset(expected_item_ids_set) and len(item_ids_set) > len(expected_item_ids_set):
                # Did extra items (the main failure mode!)
                self._finalize(-1.0, "Did BOTH actions instead of only the conditional one")
            elif item_ids_set != expected_item_ids_set:
                self._finalize(-1.0, f"Wrong item_ids: expected {expected_item_ids_set}, got {item_ids_set}")
            else:
                self._finalize(-1.0, f"Wrong new_item_ids: expected {expected_new_item_ids_set}, got {new_item_ids_set}")

        elif tool_name == "respond_to_user":
            # Should have done an exchange, not just responded
            self._finalize(-1.0, "Responded instead of executing exchange")

        else:
            self._finalize(-1.0, f"Unknown tool: {tool_name}")

        # Check max steps
        if state.step >= self.max_steps and not self.done:
            self._finalize(-1.0, "Max steps exceeded")

    def _finalize(self, reward: float, reason: str):
        self.done = True
        self._state.done = True
        self._state.reason = reason
        self.rewards = {0: reward}

    def get_summary(self) -> Dict[str, Any]:
        state = self._state
        task = state.task
        return {
            "condition_met": task.primary_condition_met,
            "checked_stock": state.checked_stock,
            "steps": state.step,
            "reward": self.rewards.get(0, 0),
            "reason": state.reason,
            "expected_items": task.expected_item_ids,
            "expected_new_items": task.expected_new_item_ids,
        }


# =============================================================================
# Parsing Utilities
# =============================================================================

def normalize_order_id(order_id: str) -> str:
    """Normalize order ID to have # prefix."""
    if order_id.startswith("#"):
        return order_id
    return f"#{order_id}"


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from model output."""
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
                        obj = json.loads(text[start:i+1])
                        if "name" in obj:
                            return {
                                "name": obj["name"],
                                "arguments": obj.get("arguments", obj.get("parameters", {}))
                            }
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass
    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action for game registry."""
    tool_call = parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    return None


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT_CONDITIONAL_ACTION = """You are a customer service agent processing conditional requests.

The user's request has a conditional structure:
- IF condition X is true, do action A
- ELSE, ONLY do action B (not both!)

You must:
1. Check the condition using the appropriate tool
2. Execute EXACTLY the action specified by the condition result
3. NOT do both actions - only the one that applies

This is critical: if the condition is NOT met, do ONLY the fallback action.
Do not do "something similar" to the primary action.

Respond with JSON tool calls.
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Conditional Action Game")
    print("=" * 60)

    game = ConditionalActionGame()

    # Test distribution
    condition_met = 0
    condition_not_met = 0
    for seed in range(500):
        game.reset(seed)
        if game._state.task.primary_condition_met:
            condition_met += 1
        else:
            condition_not_met += 1

    print(f"Distribution: {condition_met} condition met, {condition_not_met} condition not met")
    print(f"Ratio: {condition_met/5:.1f}% / {condition_not_met/5:.1f}%")

    # Show examples
    print("\n" + "=" * 60)
    for seed in [0, 1, 2]:
        game.reset(seed)
        task = game._state.task
        print(f"\n--- Seed {seed} ---")
        print(f"Primary product in stock: {task.primary_condition_met}")
        print(f"Expected action: exchange items {task.expected_item_ids} → {task.expected_new_item_ids}")
        print(f"\nObservation:\n{game.observe(0)[:1200]}...")

    # Simulate correct behavior
    print("\n" + "=" * 60)
    print("\nSimulating correct sequence for seed 0:")
    game.reset(0)
    task = game._state.task

    # Step 1: Check stock
    print(f"Step 1: Check stock for {task.primary_product_id}")
    game.step(json.dumps({
        "name": "check_product_stock",
        "arguments": {"product_id": task.primary_product_id}
    }))
    print(f"  Stock result: {game._state.stock_result}")
    print(f"  Done: {game.done}")

    if not game.done:
        # Step 2: Execute correct exchange
        print(f"Step 2: Execute exchange")
        game.step(json.dumps({
            "name": "exchange_items",
            "arguments": {
                "order_id": task.order.order_id,
                "item_ids": task.expected_item_ids,
                "new_item_ids": task.expected_new_item_ids
            }
        }))
        print(f"  Reward: {game.rewards.get(0, 0)}")
        print(f"  Reason: {game._state.reason}")

    # Simulate WRONG behavior (doing both actions)
    print("\n" + "=" * 60)
    print("\nSimulating WRONG behavior (doing both) for seed 1:")
    game.reset(1)
    task = game._state.task

    # Step 1: Check stock
    game.step(json.dumps({
        "name": "check_product_stock",
        "arguments": {"product_id": task.primary_product_id}
    }))
    print(f"Stock result: {game._state.stock_result}")

    if not game.done:
        # Step 2: WRONG - do both items
        both_item_ids = task.primary_item_ids + task.fallback_item_ids
        both_new_item_ids = task.primary_new_item_ids + task.fallback_new_item_ids
        print(f"Attempting BOTH exchanges: {both_item_ids} → {both_new_item_ids}")
        game.step(json.dumps({
            "name": "exchange_items",
            "arguments": {
                "order_id": task.order.order_id,
                "item_ids": both_item_ids,
                "new_item_ids": both_new_item_ids
            }
        }))
        print(f"  Reward: {game.rewards.get(0, 0)}")
        print(f"  Reason: {game._state.reason}")
