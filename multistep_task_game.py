"""
Multi-Step Task Tracking Game - Training Environment 2

Trains the model to plan, execute, and track N independent sub-tasks,
each requiring a specific tool call. Multi-turn episodes with simulated
tool execution and user confirmations.

Directly addresses ~32% of tau2-bench failures:
- 28 retail multi-step incompleteness failures
- 7 airline multi-reservation orchestration failures

Reward: partial credit per correct task + all-complete bonus - wrong-call penalty.
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
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
    "Avery", "Parker", "Blake", "Drew", "Jamie", "Reese", "Cameron",
    "Harper", "Emery", "Sage", "Dakota", "Skyler", "Phoenix",
]

PRODUCT_NAMES = [
    "Wireless Headphones", "USB-C Cable", "Phone Case", "Bluetooth Speaker",
    "Laptop Stand", "Wireless Mouse", "Keyboard", "Monitor Light", "Webcam",
    "Desk Lamp", "Power Bank", "Smart Watch", "Tablet Case", "Microphone",
    "External SSD", "Charger", "HDMI Cable", "Mouse Pad", "Cable Organizer",
]

STREETS = [
    "123 Oak Street", "456 Maple Avenue", "789 Pine Road",
    "321 Elm Drive", "654 Cedar Lane", "987 Birch Court",
    "111 Walnut Blvd", "222 Cherry Way", "333 Spruce Circle",
]

CITIES_STATES = [
    ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
    ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"), ("Seattle", "WA", "98101"),
    ("Boston", "MA", "02101"), ("Denver", "CO", "80201"),
    ("Atlanta", "GA", "30301"), ("Portland", "OR", "97201"),
]

CANCEL_REASONS = [
    "no longer needed", "ordered by mistake", "found a better price",
    "changed my mind", "duplicate order",
]

# Template user confirmations between tool calls
USER_CONFIRMATIONS = [
    "OK, please proceed with the next task.",
    "Got it, thanks. What about the remaining items?",
    "Great, please continue.",
    "Understood. Please handle the other requests too.",
    "Thanks! Please go ahead with the rest.",
    "OK, that's done. What's next?",
    "Good. Please take care of the other tasks as well.",
]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class OrderEntity:
    order_id: str
    status: str                       # "pending" or "delivered"
    items: List[Dict[str, Any]]       # [{item_id, name, price, product_id}]
    address: Dict[str, str]           # {street, city, state, zip}
    payment_method_id: str
    exchange_variants: Dict[str, str] # {old_item_id: new_item_id} (for exchange tasks)


@dataclass
class TaskOperation:
    description: str
    order_id: str
    tool_name: str
    tool_args: Dict[str, Any]


@dataclass
class MultiStepEpisode:
    entities: List[OrderEntity]
    operations: List[TaskOperation]
    n_tasks: int
    task_descriptions: List[str]      # human-readable task list


# =============================================================================
# Tool definitions (for the observation prompt)
# =============================================================================

TOOL_SCHEMAS = [
    {
        "name": "cancel_order",
        "description": "Cancel a pending order. Only works for orders with status 'pending'.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID to cancel"},
            "reason": {"type": "string", "description": "Reason for cancellation"},
        },
    },
    {
        "name": "return_items",
        "description": "Return items from a delivered order for refund. Only works for orders with status 'delivered'.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID"},
            "item_ids": {"type": "array", "description": "List of item IDs to return"},
            "payment_method_id": {"type": "string", "description": "Payment method for refund"},
        },
    },
    {
        "name": "exchange_items",
        "description": "Exchange items in a delivered order for different variants. Only works for orders with status 'delivered'.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID"},
            "item_ids": {"type": "array", "description": "List of item IDs to exchange"},
            "new_item_ids": {"type": "array", "description": "List of new item IDs to receive"},
            "payment_method_id": {"type": "string", "description": "Payment method for price difference"},
        },
    },
    {
        "name": "modify_address",
        "description": "Change the shipping address of a pending order. Only works for orders with status 'pending'.",
        "parameters": {
            "order_id": {"type": "string", "description": "The order ID"},
            "address1": {"type": "string", "description": "Street address"},
            "city": {"type": "string", "description": "City"},
            "state": {"type": "string", "description": "State abbreviation"},
            "zip": {"type": "string", "description": "ZIP code"},
        },
    },
    {
        "name": "respond_to_user",
        "description": "Send a message to the user. Use this ONLY after ALL tasks are complete to provide a summary.",
        "parameters": {
            "message": {"type": "string", "description": "Message to send to the user"},
        },
    },
]


# =============================================================================
# Entity and operation generation
# =============================================================================

def _gen_order_id(rng: random.Random) -> str:
    return f"#W{rng.randint(100000, 999999)}"


def _gen_item(rng: random.Random, idx: int) -> Dict[str, Any]:
    return {
        "item_id": f"item_{5000 + idx}",
        "name": rng.choice(PRODUCT_NAMES),
        "price": round(rng.uniform(10, 200), 2),
        "product_id": f"prod_{rng.randint(100, 999)}",
    }


def _gen_address(rng: random.Random) -> Dict[str, str]:
    city, state, zipcode = rng.choice(CITIES_STATES)
    return {
        "street": rng.choice(STREETS),
        "city": city,
        "state": state,
        "zip": zipcode,
    }


def _gen_order(rng: random.Random, status: str, order_idx: int) -> OrderEntity:
    n_items = rng.randint(1, 3)
    items = [_gen_item(rng, order_idx * 10 + i) for i in range(n_items)]

    # Generate exchange variants for each item
    exchange_variants = {}
    for item in items:
        exchange_variants[item["item_id"]] = f"item_{9000 + rng.randint(0, 999)}"

    return OrderEntity(
        order_id=_gen_order_id(rng),
        status=status,
        items=items,
        address=_gen_address(rng),
        payment_method_id=f"pay_{rng.randint(1000, 9999)}",
        exchange_variants=exchange_variants,
    )


def _gen_operation_for_order(rng: random.Random, order: OrderEntity) -> TaskOperation:
    """Generate a valid operation for a given order based on its status."""
    if order.status == "pending":
        op_type = rng.choice(["cancel", "modify_address"])
    else:  # delivered
        op_type = rng.choice(["return", "exchange"])

    if op_type == "cancel":
        reason = rng.choice(CANCEL_REASONS)
        return TaskOperation(
            description=f"Cancel order {order.order_id} (reason: {reason})",
            order_id=order.order_id,
            tool_name="cancel_order",
            tool_args={"order_id": order.order_id, "reason": reason},
        )

    elif op_type == "modify_address":
        new_addr = _gen_address(rng)
        # Ensure it's different from current
        while new_addr["street"] == order.address["street"]:
            new_addr = _gen_address(rng)
        return TaskOperation(
            description=(
                f"Change the shipping address of order {order.order_id} to "
                f"{new_addr['street']}, {new_addr['city']}, {new_addr['state']} {new_addr['zip']}"
            ),
            order_id=order.order_id,
            tool_name="modify_address",
            tool_args={
                "order_id": order.order_id,
                "address1": new_addr["street"],
                "city": new_addr["city"],
                "state": new_addr["state"],
                "zip": new_addr["zip"],
            },
        )

    elif op_type == "return":
        # Return one item from the order
        item = rng.choice(order.items)
        return TaskOperation(
            description=(
                f"Return the {item['name']} (item {item['item_id']}) from order "
                f"{order.order_id}. Refund to {order.payment_method_id}."
            ),
            order_id=order.order_id,
            tool_name="return_items",
            tool_args={
                "order_id": order.order_id,
                "item_ids": [item["item_id"]],
                "payment_method_id": order.payment_method_id,
            },
        )

    else:  # exchange
        item = rng.choice(order.items)
        new_item_id = order.exchange_variants[item["item_id"]]
        return TaskOperation(
            description=(
                f"Exchange the {item['name']} (item {item['item_id']}) in order "
                f"{order.order_id} for item {new_item_id}. "
                f"Use payment method {order.payment_method_id} for any price difference."
            ),
            order_id=order.order_id,
            tool_name="exchange_items",
            tool_args={
                "order_id": order.order_id,
                "item_ids": [item["item_id"]],
                "new_item_ids": [new_item_id],
                "payment_method_id": order.payment_method_id,
            },
        )


def generate_episode(seed: int, n_tasks: int = 3) -> MultiStepEpisode:
    """Generate a multi-step episode with n_tasks independent sub-tasks."""
    rng = random.Random(seed)
    n_tasks = max(2, min(5, n_tasks))

    # Generate orders with mixed statuses
    entities = []
    for i in range(n_tasks):
        status = rng.choice(["pending", "delivered"])
        entities.append(_gen_order(rng, status, i))

    # Generate one operation per order
    operations = []
    task_descriptions = []
    for i, entity in enumerate(entities):
        op = _gen_operation_for_order(rng, entity)
        operations.append(op)
        task_descriptions.append(f"Task {i + 1}: {op.description}")

    return MultiStepEpisode(
        entities=entities,
        operations=operations,
        n_tasks=n_tasks,
        task_descriptions=task_descriptions,
    )


# =============================================================================
# Tool simulation
# =============================================================================

def _execute_tool(tool_name: str, args: Dict[str, Any],
                  entities: List[OrderEntity]) -> Tuple[bool, str]:
    """Simulate tool execution against entity state.

    Returns (success, result_message).
    """
    order_id = args.get("order_id", "")

    # Find the target entity
    entity = None
    for e in entities:
        if e.order_id == order_id:
            entity = e
            break

    if entity is None:
        return False, json.dumps({"error": f"Order {order_id} not found."})

    if tool_name == "cancel_order":
        if entity.status != "pending":
            return False, json.dumps({
                "error": f"Cannot cancel order {order_id}: status is '{entity.status}', must be 'pending'."
            })
        reason = args.get("reason", "")
        if not reason:
            return False, json.dumps({"error": "Reason is required for cancellation."})
        entity.status = "cancelled"
        return True, json.dumps({
            "success": True,
            "message": f"Order {order_id} has been cancelled. Reason: {reason}.",
        })

    elif tool_name == "return_items":
        if entity.status != "delivered":
            return False, json.dumps({
                "error": f"Cannot return items from order {order_id}: status is '{entity.status}', must be 'delivered'."
            })
        item_ids = args.get("item_ids", [])
        valid_ids = {item["item_id"] for item in entity.items}
        if not item_ids or not all(iid in valid_ids for iid in item_ids):
            return False, json.dumps({
                "error": f"Invalid item_ids. Valid items: {sorted(valid_ids)}."
            })
        pm = args.get("payment_method_id", "")
        if not pm:
            return False, json.dumps({"error": "payment_method_id is required."})
        return True, json.dumps({
            "success": True,
            "message": f"Return initiated for items {item_ids} from order {order_id}. Refund to {pm}.",
        })

    elif tool_name == "exchange_items":
        if entity.status != "delivered":
            return False, json.dumps({
                "error": f"Cannot exchange items from order {order_id}: status is '{entity.status}', must be 'delivered'."
            })
        item_ids = args.get("item_ids", [])
        new_item_ids = args.get("new_item_ids", [])
        valid_ids = {item["item_id"] for item in entity.items}
        if not item_ids or not all(iid in valid_ids for iid in item_ids):
            return False, json.dumps({
                "error": f"Invalid item_ids. Valid items: {sorted(valid_ids)}."
            })
        if not new_item_ids or len(new_item_ids) != len(item_ids):
            return False, json.dumps({
                "error": "new_item_ids must have same length as item_ids."
            })
        pm = args.get("payment_method_id", "")
        if not pm:
            return False, json.dumps({"error": "payment_method_id is required."})
        return True, json.dumps({
            "success": True,
            "message": f"Exchange processed for order {order_id}: {item_ids} -> {new_item_ids}.",
        })

    elif tool_name == "modify_address":
        if entity.status != "pending":
            return False, json.dumps({
                "error": f"Cannot modify address for order {order_id}: status is '{entity.status}', must be 'pending'."
            })
        required = ["address1", "city", "state", "zip"]
        missing = [k for k in required if not args.get(k)]
        if missing:
            return False, json.dumps({
                "error": f"Missing required fields: {missing}."
            })
        entity.address = {
            "street": args["address1"],
            "city": args["city"],
            "state": args["state"],
            "zip": args["zip"],
        }
        return True, json.dumps({
            "success": True,
            "message": f"Address updated for order {order_id}.",
        })

    return False, json.dumps({"error": f"Unknown tool: {tool_name}"})


# =============================================================================
# Reward computation
# =============================================================================

def _match_tool_call(call_name: str, call_args: Dict, op: TaskOperation) -> bool:
    """Check if a tool call matches an expected operation."""
    if call_name != op.tool_name:
        return False

    # Check order_id
    if call_args.get("order_id") != op.tool_args.get("order_id"):
        return False

    # Tool-specific argument checks
    if call_name == "cancel_order":
        # Just need correct order_id and a non-empty reason
        return bool(call_args.get("reason"))

    elif call_name == "return_items":
        expected_ids = set(op.tool_args.get("item_ids", []))
        provided_ids = set(call_args.get("item_ids", []))
        return expected_ids == provided_ids and bool(call_args.get("payment_method_id"))

    elif call_name == "exchange_items":
        expected_ids = set(op.tool_args.get("item_ids", []))
        provided_ids = set(call_args.get("item_ids", []))
        expected_new = set(op.tool_args.get("new_item_ids", []))
        provided_new = set(call_args.get("new_item_ids", []))
        return (expected_ids == provided_ids and expected_new == provided_new
                and bool(call_args.get("payment_method_id")))

    elif call_name == "modify_address":
        for key in ["address1", "city", "state", "zip"]:
            if str(call_args.get(key, "")).strip().lower() != str(op.tool_args.get(key, "")).strip().lower():
                return False
        return True

    return False


def compute_reward(
    tool_calls: List[Dict[str, Any]],
    operations: List[TaskOperation],
    n_tasks: int,
) -> Tuple[float, str]:
    """Compute reward based on completed tasks.

    Returns (reward, reason_string).
    """
    # Track which operations were completed
    completed = [False] * n_tasks
    wrong_calls = 0

    for call in tool_calls:
        call_name = call.get("name", "")
        call_args = call.get("arguments", {})

        # Skip non-action calls
        if call_name in ("respond_to_user", "done"):
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

    if correct_count == n_tasks:
        reward = 1.0
    else:
        reward = (correct_count / n_tasks) * 0.8

    # Penalty for wrong calls (capped)
    reward -= 0.2 * wrong_calls
    reward = max(-1.0, min(1.0, round(reward, 2)))

    details = []
    for i, (op, done) in enumerate(zip(operations, completed)):
        details.append(f"Task {i+1} ({op.tool_name}): {'DONE' if done else 'MISSED'}")
    if wrong_calls:
        details.append(f"Wrong calls: {wrong_calls}")

    reason = f"{correct_count}/{n_tasks} tasks completed. " + "; ".join(details)
    return reward, reason


# =============================================================================
# Game class
# =============================================================================

class MultiStepTaskGame:
    """Multi-turn task tracking game.

    The model must complete N independent sub-tasks, each requiring a
    specific tool call. Tracks state, executes tools, and provides
    simulated user confirmations between steps.
    """

    def __init__(self, max_steps: int = 15, n_tasks: int = 3):
        self._max_steps = max_steps
        self._n_tasks = n_tasks
        self._episode: Optional[MultiStepEpisode] = None
        self._conversation: List[Dict[str, str]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._rng: Optional[random.Random] = None

        # GameEnv protocol
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed + 777)  # offset for user confirmations
        self._episode = generate_episode(seed, self._n_tasks)

        # Build initial user message with all task descriptions
        user_name = random.Random(seed).choice(FIRST_NAMES)
        task_list = "\n".join(self._episode.task_descriptions)
        initial_msg = (
            f"Hi, I'm {user_name}. I need help with multiple things:\n\n"
            f"{task_list}\n\n"
            f"Please complete all of these tasks."
        )

        self._conversation = [{"role": "user", "text": initial_msg}]
        self._tool_calls = []
        self._step_count = 0

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        ep = self._episode
        if ep is None:
            return "No episode loaded."

        # Entity details as JSON
        entities_json = json.dumps([
            {
                "order_id": e.order_id,
                "status": e.status,
                "items": e.items,
                "address": e.address,
                "payment_method_id": e.payment_method_id,
                **({"exchange_variants": e.exchange_variants} if e.status == "delivered" else {}),
            }
            for e in ep.entities
        ], indent=2)

        # Tool schemas
        tools_json = json.dumps(TOOL_SCHEMAS, indent=2)

        lines = [
            "You are a customer service agent. The user has requested multiple tasks.",
            "Complete ALL tasks listed below. Execute ONE tool call at a time.",
            "When ALL tasks are complete, use respond_to_user to summarize what was done.",
            "",
            "<orders>",
            entities_json,
            "</orders>",
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
            if role == "USER":
                lines.append(f"[USER]: {text}")
            elif role == "ASSISTANT":
                lines.append(f"[ASSISTANT]: {text}")
            elif role == "TOOL_CALL":
                lines.append(f"[TOOL_CALL]: {text}")
            elif role == "TOOL_RESULT":
                lines.append(f"[TOOL_RESULT]: {text}")

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
                self._tool_calls, self._episode.operations, self._episode.n_tasks,
            )
            self._finalize(reward, reason)
            return

        # Record and execute tool call
        self._tool_calls.append(tool_call)
        self._conversation.append({"role": "tool_call", "text": json.dumps(tool_call)})

        success, result = _execute_tool(
            tool_name, tool_args,
            self._episode.entities,  # mutable state
        )
        self._conversation.append({"role": "tool_result", "text": result})

        # Add simulated user confirmation
        if success and self._rng:
            confirmation = self._rng.choice(USER_CONFIRMATIONS)
            self._conversation.append({"role": "user", "text": confirmation})

        # Loop detection: same call 3 times
        if len(self._tool_calls) >= 3:
            last3 = self._tool_calls[-3:]
            keys = [json.dumps(tc, sort_keys=True) for tc in last3]
            if keys[0] == keys[1] == keys[2]:
                reward, reason = compute_reward(
                    self._tool_calls, self._episode.operations, self._episode.n_tasks,
                )
                self._finalize(reward, f"Loop detected. {reason}")
                return

        # Max steps
        if self._step_count >= self._max_steps:
            reward, reason = compute_reward(
                self._tool_calls, self._episode.operations, self._episode.n_tasks,
            )
            self._finalize(reward, f"Max steps reached. {reason}")

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    def get_summary(self) -> Dict[str, Any]:
        ep = self._episode
        return {
            "n_tasks": ep.n_tasks if ep else 0,
            "task_descriptions": ep.task_descriptions if ep else [],
            "steps": self._step_count,
            "tool_calls": self._tool_calls,
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
    # Plain text → respond_to_user
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"</?tool_call>", "", clean).strip()
    tool_call = _parse_tool_call(clean)
    if tool_call:
        return json.dumps(tool_call)
    if clean:
        return json.dumps({
            "name": "respond_to_user",
            "arguments": {"message": clean},
        })
    return None


# =============================================================================
# System prompt
# =============================================================================

SYSTEM_PROMPT = (
    "You are a customer service agent that handles multiple tasks.\n"
    "Given a list of tasks and order details, complete ALL tasks by making "
    "the appropriate tool calls one at a time.\n"
    "After completing all tasks, use respond_to_user to summarize what was done.\n"
    "Always respond with valid JSON.\n"
)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Multi-Step Task Tracking Game")
    print("=" * 60)

    for n_tasks in [2, 3, 4, 5]:
        rewards = []
        game = MultiStepTaskGame(max_steps=15, n_tasks=n_tasks)

        for seed in range(100):
            game.reset(seed)
            ep = game._episode

            # Simulate a "perfect" agent that executes all operations in order
            for op in ep.operations:
                action = json.dumps({
                    "name": op.tool_name,
                    "arguments": op.tool_args,
                })
                game.step(action)
                if game.done:
                    break

            if not game.done:
                # Signal completion
                action = json.dumps({
                    "name": "respond_to_user",
                    "arguments": {"message": "All tasks completed."},
                })
                game.step(action)

            rewards.append(game.rewards[0])

        avg = sum(rewards) / len(rewards)
        perfect = sum(1 for r in rewards if r == 1.0)
        print(f"\nn_tasks={n_tasks}: avg_reward={avg:.2f}, perfect={perfect}/100")

    # Simulate an agent that only completes first task
    print("\n" + "=" * 60)
    print("Partial completion test (n_tasks=3, only completes first task):")
    game = MultiStepTaskGame(max_steps=15, n_tasks=3)
    partial_rewards = []
    for seed in range(100):
        game.reset(seed)
        ep = game._episode

        # Only complete first operation
        op = ep.operations[0]
        action = json.dumps({"name": op.tool_name, "arguments": op.tool_args})
        game.step(action)

        # Then signal done
        action = json.dumps({
            "name": "respond_to_user",
            "arguments": {"message": "Done."},
        })
        game.step(action)
        partial_rewards.append(game.rewards[0])

    avg = sum(partial_rewards) / len(partial_rewards)
    print(f"Avg reward (1/3 tasks): {avg:.2f} (expected ~0.27)")

    # Simulate an agent that makes wrong calls
    print("\nWrong call penalty test:")
    game = MultiStepTaskGame(max_steps=15, n_tasks=2)
    game.reset(42)
    ep = game._episode

    # Make 2 wrong calls then signal done
    game.step(json.dumps({"name": "cancel_order", "arguments": {"order_id": "WRONG", "reason": "test"}}))
    game.step(json.dumps({"name": "cancel_order", "arguments": {"order_id": "WRONG2", "reason": "test"}}))
    game.step(json.dumps({"name": "respond_to_user", "arguments": {"message": "Done."}}))
    print(f"Reward with 0 correct + 2 wrong calls: {game.rewards[0]:.2f} (expected -0.4)")

    # Show example observation
    print("\n" + "=" * 60)
    print("Example episode (n_tasks=3, seed=42):")
    game = MultiStepTaskGame(max_steps=15, n_tasks=3)
    game.reset(42)
    obs = game.observe(0)
    print(obs[:3000])
    print("..." if len(obs) > 3000 else "")
    print(f"\nExpected operations:")
    for i, op in enumerate(game._episode.operations):
        print(f"  {i+1}. {op.tool_name}({json.dumps(op.tool_args)})")
