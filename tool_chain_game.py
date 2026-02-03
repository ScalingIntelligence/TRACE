"""
Tool Chain Game - Tests 2-step tool chaining.

The model must:
1. Call a lookup tool to get information
2. Use that information to call an action tool

This trains the CORE skill that's failing in tau-bench:
- Extracting information from tool responses
- Using extracted info in subsequent calls
- NOT getting stuck in loops

Key design:
- Only 2 steps required (lookup → action)
- Clear success criteria
- Information from step 1 is REQUIRED for step 2
"""

import random
import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Data Models
# =============================================================================

class OrderStatus(str, Enum):
    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"


@dataclass
class Order:
    order_id: str
    status: OrderStatus
    item_id: str
    item_name: str
    price: float


@dataclass
class Task:
    user_name: str
    order: Order
    action: str  # "cancel" or "refund"
    is_possible: bool


@dataclass
class GameState:
    task: Task
    step: int = 0
    lookup_done: bool = False
    lookup_result: Optional[Dict] = None
    final_action: Optional[str] = None
    final_params: Optional[Dict] = None
    done: bool = False


# =============================================================================
# Data Generation
# =============================================================================

NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Blake"]
ITEMS = ["Headphones", "Keyboard", "Mouse", "Monitor", "Laptop Stand", "Webcam", "USB Hub"]


def generate_task(rng: random.Random) -> Task:
    user_name = rng.choice(NAMES)
    order_id = f"#W{rng.randint(100000, 999999)}"
    item_id = f"item_{rng.randint(100000, 999999)}"
    item_name = rng.choice(ITEMS)
    price = round(rng.uniform(20, 200), 2)

    action = rng.choice(["cancel", "refund"])

    # 50/50 possible vs impossible
    if rng.random() < 0.5:
        # Make it possible
        if action == "cancel":
            status = OrderStatus.PENDING
        else:
            status = OrderStatus.DELIVERED
        is_possible = True
    else:
        # Make it impossible
        if action == "cancel":
            status = rng.choice([OrderStatus.SHIPPED, OrderStatus.DELIVERED])
        else:
            status = rng.choice([OrderStatus.PENDING, OrderStatus.SHIPPED])
        is_possible = False

    order = Order(
        order_id=order_id,
        status=status,
        item_id=item_id,
        item_name=item_name,
        price=price
    )

    return Task(user_name=user_name, order=order, action=action, is_possible=is_possible)


# =============================================================================
# Game Class
# =============================================================================

class ToolChainGame:
    """
    2-step tool chaining game.

    Step 1: Model must call get_order_details to learn the status
    Step 2: Model must either:
        - Call cancel_order/refund_order if allowed
        - Call respond_to_user if not allowed

    The model does NOT know the status until it calls get_order_details.
    This forces it to chain tools.
    """

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self._rng = random.Random()
        self._state: Optional[GameState] = None

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def reset(self, seed: int):
        self._rng = random.Random(seed)
        task = generate_task(self._rng)
        self._state = GameState(task=task)

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        state = self._state
        task = state.task

        lines = [
            "You are a customer service agent.",
            "",
            "=== POLICY RULES ===",
            "- cancel_order: ONLY for orders with status 'pending'",
            "- refund_order: ONLY for orders with status 'delivered'",
            "",
            "=== USER REQUEST ===",
            f"User {task.user_name} wants to {task.action} order {task.order.order_id}",
            "",
            "=== TOOLS ===",
            "1. get_order_details - Get order status (CALL THIS FIRST)",
            '   {"name": "get_order_details", "arguments": {"order_id": "..."}}',
            "",
            "2. cancel_order - Cancel a pending order",
            '   {"name": "cancel_order", "arguments": {"order_id": "...", "reason": "no longer needed"}}',
            "",
            "3. refund_order - Refund a delivered order",
            '   {"name": "refund_order", "arguments": {"order_id": "...", "item_id": "..."}}',
            "",
            "4. respond_to_user - Refuse if action not allowed",
            '   {"name": "respond_to_user", "arguments": {"message": "cannot..."}}',
            "",
        ]

        # Show previous actions and results
        if state.lookup_done and state.lookup_result:
            lines.append("=== PREVIOUS ACTION ===")
            lines.append(f"You called get_order_details and got:")
            lines.append(f"  Order ID: {state.lookup_result['order_id']}")
            lines.append(f"  Status: {state.lookup_result['status']}")
            lines.append(f"  Item ID: {state.lookup_result['item_id']}")
            lines.append(f"  Item: {state.lookup_result['item_name']}")
            lines.append("")
            lines.append("Now decide: execute the action OR refuse if not allowed.")
            lines.append("")
        else:
            lines.append("=== INSTRUCTIONS ===")
            lines.append("First, call get_order_details to check the order status.")
            lines.append("")

        lines.append("Respond with a JSON tool call:")

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]):
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

        # Handle based on current state
        if not state.lookup_done:
            # Step 1: Should call get_order_details
            if tool_name == "get_order_details":
                order_id = args.get("order_id", "")
                # Lenient about # prefix
                order_id_norm = order_id if order_id.startswith("#") else f"#{order_id}"

                if order_id_norm == task.order.order_id:
                    # Correct lookup
                    state.lookup_done = True
                    state.lookup_result = {
                        "order_id": task.order.order_id,
                        "status": task.order.status.value,
                        "item_id": task.order.item_id,
                        "item_name": task.order.item_name,
                        "price": task.order.price
                    }
                    # Continue to step 2
                    return
                else:
                    self._finalize(-1.0, f"Wrong order_id: {order_id}")
                    return

            elif tool_name in ["cancel_order", "refund_order"]:
                # Tried to act without checking status first
                self._finalize(-1.0, "Acted without checking status")
                return

            elif tool_name == "respond_to_user":
                # Refused without even checking - might be lucky
                if not task.is_possible:
                    # Lucky guess - but we want them to check first
                    self._finalize(-0.5, "Refused without checking (lucky)")
                    return
                else:
                    self._finalize(-1.0, "Refused possible task without checking")
                    return
            else:
                self._finalize(-1.0, f"Unknown tool: {tool_name}")
                return

        else:
            # Step 2: Should call action or refuse
            state.final_action = tool_name
            state.final_params = args

            if task.is_possible:
                # Should execute the action
                expected_tool = "cancel_order" if task.action == "cancel" else "refund_order"

                if tool_name == expected_tool:
                    if self._check_action_params(tool_name, args, task):
                        self._finalize(1.0, "Correct action after lookup")
                    else:
                        self._finalize(-1.0, "Right action, wrong params")
                elif tool_name == "respond_to_user":
                    self._finalize(-1.0, "Refused a possible task")
                elif tool_name == "get_order_details":
                    # Called lookup again - loop behavior
                    self._finalize(-1.0, "Repeated lookup (loop)")
                else:
                    self._finalize(-1.0, "Wrong action type")

            else:
                # Should refuse
                if tool_name == "respond_to_user":
                    message = args.get("message", "").lower()
                    if any(kw in message for kw in ["cannot", "can't", "unable", "not allowed", "only"]):
                        self._finalize(1.0, "Correctly refused after lookup")
                    else:
                        self._finalize(-0.5, "Refused but unclear message")
                elif tool_name == "get_order_details":
                    self._finalize(-1.0, "Repeated lookup (loop)")
                else:
                    self._finalize(-1.0, "Attempted impossible action")

        # Check max steps
        if state.step >= self.max_steps and not self.done:
            self._finalize(-1.0, "Max steps exceeded")

    def _check_action_params(self, tool_name: str, args: Dict, task: Task) -> bool:
        order_id = args.get("order_id", "")
        order_id_norm = order_id if order_id.startswith("#") else f"#{order_id}"

        if order_id_norm != task.order.order_id:
            return False

        if tool_name == "cancel_order":
            reason = args.get("reason", "").lower()
            return any(r in reason for r in ["no longer", "needed", "mistake"])

        elif tool_name == "refund_order":
            item_id = args.get("item_id", "")
            return item_id == task.order.item_id

        return False

    def _finalize(self, reward: float, reason: str):
        self.done = True
        self._state.done = True
        self.rewards = {0: reward}
        self._state.reason = reason

    def get_summary(self) -> Dict[str, Any]:
        state = self._state
        task = state.task
        return {
            "action": task.action,
            "status": task.order.status.value,
            "is_possible": task.is_possible,
            "steps": state.step,
            "lookup_done": state.lookup_done,
            "reward": self.rewards.get(0, 0),
            "reason": getattr(state, "reason", "unknown"),
        }


# =============================================================================
# Parsing
# =============================================================================

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
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
                    except:
                        pass
                    start = -1
    except:
        pass
    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    tool_call = parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    return None


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT_TOOL_CHAIN = """You are a customer service agent.

To process a request:
1. First call get_order_details to check the order status
2. Then either execute the action (if allowed) or refuse (if not allowed)

Policy rules:
- cancel_order: Only for 'pending' orders
- refund_order: Only for 'delivered' orders

Respond with JSON tool calls.
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Tool Chain Game")
    print("=" * 60)

    game = ToolChainGame()

    # Distribution test
    possible = sum(1 for s in range(500) if (game.reset(s), game._state.task.is_possible)[1])
    print(f"Possible tasks: {possible}/500 = {possible/5:.1f}%")

    # Show examples
    for seed in [0, 1, 2]:
        print(f"\n--- Seed {seed} ---")
        game.reset(seed)
        task = game._state.task
        print(f"Action: {task.action}, Status: {task.order.status.value}")
        print(f"Possible: {task.is_possible}")
        print(f"Order: {task.order.order_id}, Item: {task.order.item_id}")
        print(f"\nObservation:\n{game.observe(0)}")
