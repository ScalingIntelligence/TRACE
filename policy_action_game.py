"""
Policy Action Game - Single-step tool calling with policy reasoning.

The model sees ALL information upfront (order details, status, items).
It must make ONE correct decision:
1. If action is allowed by policy → call the action with correct parameters
2. If action is NOT allowed → call respond_to_user to refuse

This trains:
- Tool calling JSON format
- Parameter extraction from observation
- Policy rule checking
- Decision making (act vs refuse)

NO tool chaining required - everything needed is in the observation.
"""

import random
import json
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Data Models
# =============================================================================

class OrderStatus(str, Enum):
    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"


@dataclass
class OrderItem:
    item_id: str
    name: str
    price: float


@dataclass
class Order:
    order_id: str
    status: OrderStatus
    items: List[OrderItem]
    total: float


@dataclass
class Task:
    """A task with all information provided."""
    user_name: str
    order: Order
    requested_action: str  # "cancel", "refund"
    is_possible: bool
    ground_truth: Dict[str, Any]  # The correct tool call


# =============================================================================
# Policy Rules
# =============================================================================

POLICY_RULES = {
    "cancel": {
        "allowed_statuses": [OrderStatus.PENDING],
        "tool_name": "cancel_order",
        "description": "Cancel order - only for PENDING orders",
    },
    "refund": {
        "allowed_statuses": [OrderStatus.DELIVERED],
        "tool_name": "refund_order",
        "description": "Refund order - only for DELIVERED orders",
    },
}

VALID_REASONS = ["no longer needed", "ordered by mistake"]


# =============================================================================
# Data Generation
# =============================================================================

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
               "Avery", "Parker", "Blake", "Drew", "Jamie", "Reese", "Cameron"]

PRODUCT_NAMES = [
    "Wireless Headphones", "USB-C Cable", "Phone Case", "Bluetooth Speaker",
    "Laptop Stand", "Wireless Mouse", "Keyboard", "Monitor Light", "Webcam"
]


def generate_order_id(rng: random.Random) -> str:
    return f"#W{rng.randint(100000, 999999)}"


def generate_item_id(rng: random.Random) -> str:
    return f"item_{rng.randint(100000, 999999)}"


def generate_task(rng: random.Random) -> Task:
    """Generate a single-step policy task."""

    user_name = rng.choice(FIRST_NAMES)

    # Pick action type (only cancel and refund - simpler)
    action = rng.choice(["cancel", "refund"])

    # Decide if task should be possible (50/50 split for balanced learning)
    make_possible = rng.random() < 0.5

    # Set order status based on action and possibility
    if action == "cancel":
        if make_possible:
            status = OrderStatus.PENDING
        else:
            status = rng.choice([OrderStatus.SHIPPED, OrderStatus.DELIVERED])
    else:  # refund
        if make_possible:
            status = OrderStatus.DELIVERED
        else:
            status = rng.choice([OrderStatus.PENDING, OrderStatus.SHIPPED])

    # Generate order
    order_id = generate_order_id(rng)
    num_items = rng.randint(1, 3)
    items = []
    total = 0
    for _ in range(num_items):
        price = round(rng.uniform(10, 100), 2)
        items.append(OrderItem(
            item_id=generate_item_id(rng),
            name=rng.choice(PRODUCT_NAMES),
            price=price
        ))
        total += price

    order = Order(
        order_id=order_id,
        status=status,
        items=items,
        total=round(total, 2)
    )

    # Determine actual possibility
    allowed_statuses = POLICY_RULES[action]["allowed_statuses"]
    is_possible = status in allowed_statuses

    # Compute ground truth
    if is_possible:
        tool_name = POLICY_RULES[action]["tool_name"]
        if action == "cancel":
            ground_truth = {
                "name": tool_name,
                "arguments": {
                    "order_id": order_id,
                    "reason": "no longer needed"
                }
            }
        else:  # refund
            ground_truth = {
                "name": tool_name,
                "arguments": {
                    "order_id": order_id,
                    "item_ids": [item.item_id for item in items],
                    "payment_method": "original"
                }
            }
    else:
        ground_truth = {
            "name": "respond_to_user",
            "arguments": {
                "message": "cannot"  # Will check for "cannot" keyword
            }
        }

    return Task(
        user_name=user_name,
        order=order,
        requested_action=action,
        is_possible=is_possible,
        ground_truth=ground_truth
    )


# =============================================================================
# Game Class
# =============================================================================

@dataclass
class GameState:
    task: Task
    action_taken: Optional[Dict[str, Any]] = None
    done: bool = False


class PolicyActionGame:
    """
    Single-step policy action game.

    Model sees complete order information and must:
    1. Check if requested action is allowed by policy
    2. If YES: call the action tool with correct parameters
    3. If NO: call respond_to_user to refuse

    This is ONE decision - no tool chaining.
    """

    def __init__(self):
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
        """Get observation with ALL information provided."""
        task = self._state.task
        order = task.order

        # Format items
        items_str = "\n".join([
            f"    - {item.item_id}: {item.name} (${item.price})"
            for item in order.items
        ])

        lines = [
            "You are a customer service agent. A user has made a request.",
            "",
            "=== POLICY RULES ===",
            "- cancel_order: Can ONLY cancel orders with status 'pending'",
            "- refund_order: Can ONLY refund orders with status 'delivered'",
            "- If action is NOT allowed, use respond_to_user to politely refuse",
            "",
            "=== ORDER DETAILS ===",
            f"Order ID: {order.order_id}",
            f"Status: {order.status.value.upper()}",
            f"Total: ${order.total}",
            f"Items:",
            items_str,
            "",
            "=== USER REQUEST ===",
            f"User {task.user_name} wants to {task.requested_action} this order.",
            "",
            "=== AVAILABLE TOOLS ===",
            '- cancel_order: {"name": "cancel_order", "arguments": {"order_id": "...", "reason": "no longer needed"}}',
            '- refund_order: {"name": "refund_order", "arguments": {"order_id": "...", "item_ids": [...], "payment_method": "original"}}',
            '- respond_to_user: {"name": "respond_to_user", "arguments": {"message": "..."}}',
            "",
            "=== INSTRUCTIONS ===",
            "1. Check if the requested action is allowed by policy rules",
            "2. If ALLOWED: Call the appropriate action tool with correct parameters",
            "3. If NOT ALLOWED: Call respond_to_user explaining why (include 'cannot' in message)",
            "",
            "Respond with a JSON tool call:",
        ]

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]):
        """Process the model's action."""
        if self.done:
            return

        task = self._state.task

        # Parse action
        if action is None:
            self._finalize(-1.0, "No action")
            return

        tool_call = parse_tool_call(action)
        if tool_call is None:
            self._finalize(-1.0, "Invalid JSON")
            self.invalid_player = 0
            return

        self._state.action_taken = tool_call

        # Evaluate
        tool_name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        if task.is_possible:
            # Task IS possible - model should call the action
            expected_tool = task.ground_truth["name"]

            if tool_name == expected_tool:
                # Check parameters
                if self._check_params(tool_name, args, task):
                    self._finalize(1.0, "Correct action with correct params")
                else:
                    self._finalize(-1.0, "Correct action but wrong params")
            elif tool_name == "respond_to_user":
                # Incorrectly refused
                self._finalize(-1.0, "Refused a possible task")
            else:
                # Wrong action type
                self._finalize(-1.0, "Wrong action type")
        else:
            # Task is NOT possible - model should refuse
            if tool_name == "respond_to_user":
                message = args.get("message", "").lower()
                if any(kw in message for kw in ["cannot", "can't", "unable", "not allowed", "not possible", "only"]):
                    self._finalize(1.0, "Correctly refused")
                else:
                    self._finalize(-1.0, "Refused but unclear message")
            else:
                # Tried to execute impossible action
                self._finalize(-1.0, "Attempted impossible action")

    def _check_params(self, tool_name: str, args: Dict, task: Task) -> bool:
        """Check if parameters are correct."""
        order = task.order

        # Check order_id (lenient about # prefix)
        order_id = args.get("order_id", "")
        order_id_norm = order_id if order_id.startswith("#") else f"#{order_id}"
        if order_id_norm != order.order_id:
            return False

        if tool_name == "cancel_order":
            # Check reason
            reason = args.get("reason", "").lower()
            if not any(r in reason for r in ["no longer needed", "ordered by mistake", "don't need", "mistake"]):
                return False
            return True

        elif tool_name == "refund_order":
            # Check item_ids - must include all items
            provided_ids = set(args.get("item_ids", []))
            expected_ids = set(item.item_id for item in order.items)

            # Be lenient - if they provided at least one correct item
            if not provided_ids:
                return False
            if not provided_ids.issubset(expected_ids):
                return False
            return True

        return False

    def _finalize(self, reward: float, reason: str):
        self.done = True
        self._state.done = True
        self.rewards = {0: reward}
        self._state.reason = reason

    def get_summary(self) -> Dict[str, Any]:
        task = self._state.task
        return {
            "action": task.requested_action,
            "status": task.order.status.value,
            "is_possible": task.is_possible,
            "reward": self.rewards.get(0, 0),
            "reason": getattr(self._state, "reason", "unknown"),
        }


# =============================================================================
# Action Parsing
# =============================================================================

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from model output."""
    # Try to find JSON object
    try:
        # Find first { and matching }
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

SYSTEM_PROMPT_POLICY_ACTION = """You are a customer service agent that processes order requests.

Given order details and a user request, you must:
1. Check if the action is allowed by policy rules
2. If allowed, call the action tool with correct parameters
3. If not allowed, call respond_to_user to explain why

Policy rules:
- cancel_order: Only for orders with status 'pending'
- refund_order: Only for orders with status 'delivered'

Respond with a single JSON tool call.
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Policy Action Game")
    print("=" * 60)

    game = PolicyActionGame()

    # Test distribution
    possible = 0
    impossible = 0
    for seed in range(1000):
        game.reset(seed)
        if game._state.task.is_possible:
            possible += 1
        else:
            impossible += 1

    print(f"Distribution: {possible} possible, {impossible} impossible")
    print(f"Ratio: {possible/10:.1f}% / {impossible/10:.1f}%")

    # Show examples
    print("\n" + "=" * 60)
    for seed in [0, 1, 2, 3]:
        game.reset(seed)
        task = game._state.task
        print(f"\n--- Seed {seed} ---")
        print(f"Action: {task.requested_action}, Status: {task.order.status.value}")
        print(f"Possible: {task.is_possible}")
        print(f"Ground truth: {task.ground_truth['name']}")
        print(f"\nObservation:\n{game.observe(0)[:800]}...")
