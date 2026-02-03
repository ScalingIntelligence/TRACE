"""
Policy-Gated Action Game - Trains state verification and constraint compliance.

This environment targets the core tau-bench failure modes:
1. Tool call loops (37.9%) - via repetition penalties
2. Incorrect parameters (21.4%) - via strict parameter validation
3. Policy violations - via state-dependent constraints

Key design principles:
- BINARY rewards only (no partial credit, no shaped rewards)
- Implicit task requirements (must infer from natural language)
- State-dependent constraints (must verify before acting)
- No tool call repetition (tracks what's been queried)

Distribution matches tau-bench complexity levels.
"""

import random
import json
import re
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Complexity Distribution (matching tau-bench)
# =============================================================================

# Task complexity levels
COMPLEXITY_LEVELS = [1, 2, 3, 4, 5]
COMPLEXITY_WEIGHTS = [15, 25, 30, 20, 10]  # Weighted toward medium difficulty


# =============================================================================
# Data Models
# =============================================================================

class OrderStatus(str, Enum):
    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class UserTier(str, Enum):
    BASIC = "basic"
    SILVER = "silver"
    GOLD = "gold"


@dataclass
class User:
    user_id: str
    name: str
    email: str
    tier: UserTier
    balance: float


@dataclass
class Item:
    item_id: str
    product_id: str
    name: str
    price: float
    available: bool


@dataclass
class OrderItem:
    item_id: str
    product_id: str
    name: str
    price: float
    quantity: int


@dataclass
class Order:
    order_id: str
    user_id: str
    status: OrderStatus
    items: List[OrderItem]
    total: float
    created_days_ago: int  # For time-based constraints


@dataclass
class Task:
    """A generated task with ground truth."""
    request: str
    action_type: str
    user: User
    order: Order
    available_items: Dict[str, Item]  # item_id -> Item
    is_possible: bool
    ground_truth_action: Optional[str]
    ground_truth_params: Optional[Dict[str, Any]]
    required_lookups: Set[str]  # What tools must be called
    constraints_to_check: List[Tuple[str, str, Any]]  # Constraints that must be verified


# =============================================================================
# Policy Constraints
# =============================================================================

POLICY_RULES = {
    "cancel_order": {
        "description": "Cancel a pending order",
        "constraints": [
            ("order.status", "==", OrderStatus.PENDING),
        ],
        "required_params": ["order_id", "reason"],
        "valid_reasons": ["no longer needed", "ordered by mistake"],
    },
    "refund_order": {
        "description": "Refund a delivered order",
        "constraints": [
            ("order.status", "==", OrderStatus.DELIVERED),
        ],
        "required_params": ["order_id", "item_ids", "payment_method"],
    },
    "exchange_item": {
        "description": "Exchange items in a delivered order",
        "constraints": [
            ("order.status", "==", OrderStatus.DELIVERED),
            ("new_item.available", "==", True),
        ],
        "required_params": ["order_id", "old_item_id", "new_item_id"],
    },
    "modify_order": {
        "description": "Modify items in a pending order",
        "constraints": [
            ("order.status", "==", OrderStatus.PENDING),
            ("new_item.available", "==", True),
        ],
        "required_params": ["order_id", "old_item_id", "new_item_id"],
    },
}


# =============================================================================
# Task Templates (Natural Language)
# =============================================================================

TASK_TEMPLATES = {
    "cancel_order": [
        "User {name} (ID: {user_id}) wants to cancel order {order_id}.",
        "Please help {name} cancel their order {order_id}.",
        "Customer {name} requested cancellation of order {order_id}.",
        "{name} needs to cancel order {order_id} because they no longer need it.",
    ],
    "refund_order": [
        "User {name} (ID: {user_id}) wants a refund for order {order_id}.",
        "Please process a refund for {name}'s order {order_id}.",
        "Customer {name} is requesting a refund on order {order_id}.",
        "{name} would like to return items from order {order_id} for a refund.",
    ],
    "exchange_item": [
        "User {name} (ID: {user_id}) wants to exchange an item from order {order_id}.",
        "Please help {name} exchange item {old_item_name} for {new_item_name} in order {order_id}.",
        "Customer {name} wants to swap {old_item_name} for a different variant in order {order_id}.",
        "{name} needs to exchange their {old_item_name} from order {order_id}.",
    ],
    "modify_order": [
        "User {name} (ID: {user_id}) wants to modify order {order_id}.",
        "Please help {name} change item {old_item_name} to {new_item_name} in order {order_id}.",
        "{name} wants to update their pending order {order_id}.",
        "Customer {name} needs to modify the items in order {order_id}.",
    ],
}

IMPOSSIBLE_TASK_TEMPLATES = {
    "cancel_shipped": [
        "User {name} wants to cancel order {order_id}. Note: The order has already shipped.",
        "{name} is asking to cancel order {order_id} which is currently in transit.",
    ],
    "cancel_delivered": [
        "User {name} wants to cancel order {order_id}. Note: The order was already delivered.",
        "{name} is requesting cancellation of delivered order {order_id}.",
    ],
    "refund_pending": [
        "User {name} wants a refund for order {order_id}. Note: The order hasn't shipped yet.",
        "{name} is requesting a refund on pending order {order_id}.",
    ],
    "exchange_unavailable": [
        "User {name} wants to exchange {old_item_name} for {new_item_name} in order {order_id}. Note: {new_item_name} may not be available.",
        "{name} wants to swap to {new_item_name} in order {order_id}.",
    ],
}


# =============================================================================
# Tool Definitions
# =============================================================================

TOOL_SPECS = [
    {
        "name": "get_user_details",
        "description": "Get user information by user ID",
        "parameters": ["user_id"],
    },
    {
        "name": "get_order_details",
        "description": "Get order information including status and items",
        "parameters": ["order_id"],
    },
    {
        "name": "get_item_details",
        "description": "Get item availability and pricing information",
        "parameters": ["item_id"],
    },
    {
        "name": "find_user_by_name",
        "description": "Find user ID by name",
        "parameters": ["name"],
    },
    {
        "name": "cancel_order",
        "description": "Cancel a pending order. Requires order.status == 'pending'.",
        "parameters": ["order_id", "reason"],
    },
    {
        "name": "refund_order",
        "description": "Refund items from a delivered order. Requires order.status == 'delivered'.",
        "parameters": ["order_id", "item_ids", "payment_method"],
    },
    {
        "name": "exchange_item",
        "description": "Exchange an item in a delivered order. Requires order.status == 'delivered' and new item available.",
        "parameters": ["order_id", "old_item_id", "new_item_id"],
    },
    {
        "name": "modify_order",
        "description": "Modify items in a pending order. Requires order.status == 'pending' and new item available.",
        "parameters": ["order_id", "old_item_id", "new_item_id"],
    },
    {
        "name": "respond_to_user",
        "description": "Send a response to the user (use when task cannot be completed)",
        "parameters": ["message"],
    },
]


# =============================================================================
# Data Generation
# =============================================================================

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
               "Parker", "Sage", "Blake", "Drew", "Jamie", "Reese", "Cameron", "Dakota"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson"]

PRODUCT_NAMES = [
    ("Wireless Headphones", "PROD001"),
    ("USB-C Cable", "PROD002"),
    ("Phone Case", "PROD003"),
    ("Bluetooth Speaker", "PROD004"),
    ("Laptop Stand", "PROD005"),
    ("Wireless Mouse", "PROD006"),
    ("Keyboard", "PROD007"),
    ("Monitor Light", "PROD008"),
    ("Webcam", "PROD009"),
    ("Desk Organizer", "PROD010"),
]


def generate_id(rng: random.Random, prefix: str) -> str:
    """Generate a random ID with prefix."""
    return f"{prefix}_{rng.randint(100000, 999999)}"


def generate_user(rng: random.Random) -> User:
    """Generate a random user."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return User(
        user_id=generate_id(rng, "user"),
        name=f"{first} {last}",
        email=f"{first.lower()}.{last.lower()}@email.com",
        tier=rng.choice(list(UserTier)),
        balance=round(rng.uniform(0, 500), 2),
    )


def generate_items(rng: random.Random, num_items: int) -> Dict[str, Item]:
    """Generate available items catalog."""
    items = {}
    selected = rng.sample(PRODUCT_NAMES, min(num_items, len(PRODUCT_NAMES)))
    for name, product_id in selected:
        for variant in ["Standard", "Premium", "Basic"]:
            item_id = generate_id(rng, "item")
            items[item_id] = Item(
                item_id=item_id,
                product_id=product_id,
                name=f"{name} ({variant})",
                price=round(rng.uniform(10, 200), 2),
                available=rng.random() > 0.2,  # 80% available
            )
    return items


def generate_order(rng: random.Random, user: User, status: OrderStatus,
                   items_catalog: Dict[str, Item]) -> Order:
    """Generate an order with given status."""
    # Select 1-3 items for the order
    num_order_items = rng.randint(1, 3)
    selected_items = rng.sample(list(items_catalog.values()),
                                 min(num_order_items, len(items_catalog)))

    order_items = []
    total = 0
    for item in selected_items:
        qty = rng.randint(1, 2)
        order_items.append(OrderItem(
            item_id=item.item_id,
            product_id=item.product_id,
            name=item.name,
            price=item.price,
            quantity=qty,
        ))
        total += item.price * qty

    return Order(
        order_id=f"#W{rng.randint(100000, 999999)}",
        user_id=user.user_id,
        status=status,
        items=order_items,
        total=round(total, 2),
        created_days_ago=rng.randint(1, 60),
    )


# =============================================================================
# Task Generation
# =============================================================================

def generate_task(rng: random.Random, complexity: int) -> Task:
    """Generate a task with specified complexity level."""
    # Generate base data
    user = generate_user(rng)
    items_catalog = generate_items(rng, 5 + complexity)

    # Decide if task should be possible (70% possible, 30% impossible)
    make_possible = rng.random() < 0.7

    # Select action type
    action_type = rng.choice(["cancel_order", "refund_order", "exchange_item", "modify_order"])

    # Determine order status based on action and possibility
    if action_type == "cancel_order":
        if make_possible:
            status = OrderStatus.PENDING
        else:
            status = rng.choice([OrderStatus.SHIPPED, OrderStatus.DELIVERED])
    elif action_type == "refund_order":
        if make_possible:
            status = OrderStatus.DELIVERED
        else:
            status = rng.choice([OrderStatus.PENDING, OrderStatus.SHIPPED])
    elif action_type in ["exchange_item", "modify_order"]:
        if action_type == "exchange_item":
            status = OrderStatus.DELIVERED if make_possible else OrderStatus.PENDING
        else:  # modify_order
            status = OrderStatus.PENDING if make_possible else OrderStatus.DELIVERED

    order = generate_order(rng, user, status, items_catalog)

    # For exchange/modify, select items
    old_item = order.items[0] if order.items else None
    new_item = None
    new_item_available = True

    if action_type in ["exchange_item", "modify_order"] and old_item:
        # Find a different item of same product type
        candidates = [i for i in items_catalog.values()
                     if i.product_id == old_item.product_id and i.item_id != old_item.item_id]
        if candidates:
            new_item = rng.choice(candidates)
            # For impossible tasks, might make new item unavailable
            if not make_possible and rng.random() < 0.5:
                new_item.available = False
                new_item_available = False

    # Determine if task is actually possible
    is_possible = check_task_possibility(action_type, status, new_item_available)

    # Generate natural language request
    request = generate_request(rng, action_type, user, order, old_item, new_item, is_possible)

    # Compute ground truth
    if is_possible:
        ground_truth_action = action_type
        ground_truth_params = compute_ground_truth_params(
            action_type, order, old_item, new_item, user
        )
    else:
        ground_truth_action = "respond_to_user"
        ground_truth_params = {"message": "cannot"}  # Will check for "cannot" keyword

    # Determine required lookups based on complexity
    required_lookups = determine_required_lookups(action_type, complexity)

    # Constraints to verify
    constraints = get_constraints_for_action(action_type)

    return Task(
        request=request,
        action_type=action_type,
        user=user,
        order=order,
        available_items=items_catalog,
        is_possible=is_possible,
        ground_truth_action=ground_truth_action,
        ground_truth_params=ground_truth_params,
        required_lookups=required_lookups,
        constraints_to_check=constraints,
    )


def check_task_possibility(action_type: str, status: OrderStatus,
                           new_item_available: bool) -> bool:
    """Check if task is possible given constraints."""
    if action_type == "cancel_order":
        return status == OrderStatus.PENDING
    elif action_type == "refund_order":
        return status == OrderStatus.DELIVERED
    elif action_type == "exchange_item":
        return status == OrderStatus.DELIVERED and new_item_available
    elif action_type == "modify_order":
        return status == OrderStatus.PENDING and new_item_available
    return False


def generate_request(rng: random.Random, action_type: str, user: User,
                     order: Order, old_item: Optional[OrderItem],
                     new_item: Optional[Item], is_possible: bool) -> str:
    """Generate natural language request."""
    if is_possible:
        templates = TASK_TEMPLATES.get(action_type, TASK_TEMPLATES["cancel_order"])
    else:
        # Use templates that hint at impossibility
        if action_type == "cancel_order" and order.status == OrderStatus.SHIPPED:
            templates = IMPOSSIBLE_TASK_TEMPLATES.get("cancel_shipped", TASK_TEMPLATES[action_type])
        elif action_type == "cancel_order" and order.status == OrderStatus.DELIVERED:
            templates = IMPOSSIBLE_TASK_TEMPLATES.get("cancel_delivered", TASK_TEMPLATES[action_type])
        elif action_type == "refund_order" and order.status == OrderStatus.PENDING:
            templates = IMPOSSIBLE_TASK_TEMPLATES.get("refund_pending", TASK_TEMPLATES[action_type])
        else:
            templates = TASK_TEMPLATES.get(action_type, TASK_TEMPLATES["cancel_order"])

    template = rng.choice(templates)

    return template.format(
        name=user.name,
        user_id=user.user_id,
        order_id=order.order_id,
        old_item_name=old_item.name if old_item else "item",
        new_item_name=new_item.name if new_item else "different item",
    )


def compute_ground_truth_params(action_type: str, order: Order,
                                 old_item: Optional[OrderItem],
                                 new_item: Optional[Item],
                                 user: User) -> Dict[str, Any]:
    """Compute correct parameters for the action."""
    if action_type == "cancel_order":
        return {
            "order_id": order.order_id,
            "reason": "no longer needed",  # Default valid reason
        }
    elif action_type == "refund_order":
        return {
            "order_id": order.order_id,
            "item_ids": [item.item_id for item in order.items],
            "payment_method": "original",  # Refund to original payment
        }
    elif action_type == "exchange_item":
        return {
            "order_id": order.order_id,
            "old_item_id": old_item.item_id if old_item else "",
            "new_item_id": new_item.item_id if new_item else "",
        }
    elif action_type == "modify_order":
        return {
            "order_id": order.order_id,
            "old_item_id": old_item.item_id if old_item else "",
            "new_item_id": new_item.item_id if new_item else "",
        }
    return {}


def determine_required_lookups(action_type: str, complexity: int) -> Set[str]:
    """Determine what tools must be called based on action and complexity."""
    # Base lookups
    lookups = {"get_order_details"}

    if complexity >= 2:
        lookups.add("get_user_details")

    if complexity >= 3 and action_type in ["exchange_item", "modify_order"]:
        lookups.add("get_item_details")

    return lookups


def get_constraints_for_action(action_type: str) -> List[Tuple[str, str, Any]]:
    """Get constraints that must be checked for an action."""
    rules = POLICY_RULES.get(action_type, {})
    return rules.get("constraints", [])


# =============================================================================
# Game State
# =============================================================================

@dataclass
class GameState:
    """Complete state of a game episode."""
    task: Task
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tools_called: Set[str] = field(default_factory=set)  # Track unique tool calls
    repeated_calls: int = 0  # Count repeated tool calls
    current_step: int = 0
    final_action: Optional[str] = None
    final_params: Optional[Dict[str, Any]] = None
    done: bool = False


# =============================================================================
# Tool Execution
# =============================================================================

def execute_tool(state: GameState, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call and return result."""
    task = state.task

    # Track repetition
    call_key = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
    if call_key in state.tools_called:
        state.repeated_calls += 1
    state.tools_called.add(call_key)

    # Execute tool
    if tool_name == "get_user_details":
        user_id = params.get("user_id", "")
        if user_id == task.user.user_id:
            return {
                "user_id": task.user.user_id,
                "name": task.user.name,
                "email": task.user.email,
                "tier": task.user.tier.value,
                "balance": task.user.balance,
            }
        return {"error": f"User '{user_id}' not found"}

    elif tool_name == "get_order_details":
        order_id = params.get("order_id", "")
        # Be lenient about # prefix - accept both "W123456" and "#W123456"
        normalized_input = order_id if order_id.startswith("#") else f"#{order_id}"
        if normalized_input == task.order.order_id:
            return {
                "order_id": task.order.order_id,
                "user_id": task.order.user_id,
                "status": task.order.status.value,
                "items": [
                    {"item_id": i.item_id, "name": i.name, "price": i.price, "quantity": i.quantity}
                    for i in task.order.items
                ],
                "total": task.order.total,
                "created_days_ago": task.order.created_days_ago,
            }
        return {"error": f"Order '{order_id}' not found"}

    elif tool_name == "get_item_details":
        item_id = params.get("item_id", "")
        if item_id in task.available_items:
            item = task.available_items[item_id]
            return {
                "item_id": item.item_id,
                "product_id": item.product_id,
                "name": item.name,
                "price": item.price,
                "available": item.available,
            }
        return {"error": f"Item '{item_id}' not found"}

    elif tool_name == "find_user_by_name":
        name = params.get("name", "")
        if name.lower() == task.user.name.lower():
            return {"user_id": task.user.user_id, "name": task.user.name}
        return {"error": f"User '{name}' not found"}

    elif tool_name in ["cancel_order", "refund_order", "exchange_item", "modify_order"]:
        # Final action - mark completion
        state.final_action = tool_name
        state.final_params = params
        state.done = True

        # Check if action was valid (lenient about # prefix)
        order_id = params.get("order_id", "")
        normalized_input = order_id if order_id.startswith("#") else f"#{order_id}"
        if normalized_input != task.order.order_id:
            return {"error": f"Order '{order_id}' not found"}

        # Validate against constraints
        constraints = POLICY_RULES.get(tool_name, {}).get("constraints", [])
        for field, op, expected in constraints:
            if field == "order.status":
                actual = task.order.status
                if op == "==" and actual != expected:
                    return {"error": f"Cannot {tool_name.replace('_', ' ')}: order status is '{actual.value}', expected '{expected.value}'"}
            elif field == "new_item.available" and tool_name in ["exchange_item", "modify_order"]:
                new_item_id = params.get("new_item_id", "")
                if new_item_id in task.available_items:
                    actual = task.available_items[new_item_id].available
                    if op == "==" and actual != expected:
                        return {"error": f"Cannot {tool_name.replace('_', ' ')}: new item is not available"}

        return {"success": True, "action": tool_name}

    elif tool_name == "respond_to_user":
        state.final_action = "respond_to_user"
        state.final_params = params
        state.done = True
        return {"success": True, "message_sent": True}

    return {"error": f"Unknown tool: {tool_name}"}


# =============================================================================
# Action Parsing
# =============================================================================

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from model output."""
    # Try to find JSON with name and arguments
    patterns = [
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters|params)"\s*:\s*(\{[^}]+\})\s*\}',
        r'\{\s*"(?:arguments|parameters|params)"\s*:\s*(\{[^}]+\})\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 2:
                    if '"name"' in pattern[:30]:
                        name = groups[0]
                        args = json.loads(groups[1])
                    else:
                        args = json.loads(groups[0])
                        name = groups[1]
                    return {"name": name, "arguments": args}
            except json.JSONDecodeError:
                continue

    # Try simpler JSON extraction
    try:
        # Find any JSON object with "name" key
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
                                "arguments": obj.get("arguments", obj.get("parameters", obj.get("params", {})))
                            }
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass

    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action from model output."""
    tool_call = parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    return None


# =============================================================================
# Main Game Class
# =============================================================================

class PolicyGatedActionGame:
    """
    Policy-Gated Action Game.

    Trains agents to:
    1. Verify state before taking actions
    2. Respect policy constraints
    3. Gather necessary information without repetition
    4. Execute correct action OR refuse impossible tasks

    BINARY REWARDS ONLY - no partial credit.
    """

    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self._rng = random.Random()
        self._state: Optional[GameState] = None

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def reset(self, seed: int):
        """Reset game with new seed."""
        self._rng = random.Random(seed)

        # Sample complexity from distribution
        complexity = self._rng.choices(COMPLEXITY_LEVELS, weights=COMPLEXITY_WEIGHTS, k=1)[0]

        # Generate task
        task = generate_task(self._rng, complexity)

        self._state = GameState(task=task)

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        """Get observation for the agent."""
        state = self._state
        task = state.task

        lines = []
        lines.append("You are a customer service agent. Help users with their orders.")
        lines.append("")
        lines.append("POLICY RULES:")
        lines.append("- cancel_order: Only for PENDING orders")
        lines.append("- refund_order: Only for DELIVERED orders")
        lines.append("- exchange_item: Only for DELIVERED orders with available replacement")
        lines.append("- modify_order: Only for PENDING orders with available replacement")
        lines.append("")
        lines.append("AVAILABLE TOOLS (use JSON format: {\"name\": \"...\", \"arguments\": {...}}):")
        for tool in TOOL_SPECS:
            params_str = ", ".join(tool["parameters"])
            lines.append(f"- {tool['name']}: {tool['description']}. Args: {{{params_str}}}")
        lines.append("")
        lines.append("TASK:")
        lines.append(task.request)
        lines.append("")

        # Show previous tool calls
        if state.tool_calls:
            lines.append("PREVIOUS ACTIONS:")
            for tc in state.tool_calls[-5:]:
                result_str = json.dumps(tc['result'])
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                lines.append(f"  {tc['tool']}({tc['params']}) -> {result_str}")
            lines.append("")

        lines.append("INSTRUCTIONS:")
        lines.append("1. First verify the order status using get_order_details")
        lines.append("2. Check if the requested action is allowed by policy rules")
        lines.append("3. If allowed, execute the action with correct parameters")
        lines.append("4. If NOT allowed, use respond_to_user to explain why")
        lines.append("")
        lines.append("Respond with a single tool call.")

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        """Return valid action formats."""
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]):
        """Process agent action."""
        if self.done:
            return

        state = self._state

        # Parse action
        if action is None:
            self._finalize(-1.0, "No action provided")
            return

        tool_call = None
        try:
            tool_call = json.loads(action)
        except json.JSONDecodeError:
            tool_call = parse_tool_call(action)

        if tool_call is None:
            self._finalize(-1.0, "Could not parse tool call")
            return

        tool_name = tool_call.get("name", "")
        params = tool_call.get("arguments", tool_call.get("parameters", {}))

        # Execute tool
        result = execute_tool(state, tool_name, params)

        # Track
        state.tool_calls.append({
            "tool": tool_name,
            "params": params,
            "result": result,
        })
        state.current_step += 1

        # Check termination conditions
        if state.done:
            self._evaluate_final_action()
        elif state.current_step >= self.max_steps:
            self._finalize(-1.0, "Max steps reached without completing task")
        elif state.repeated_calls >= 3:
            # Stuck in a loop
            self._finalize(-1.0, "Stuck in tool call loop")

    def _evaluate_final_action(self):
        """Evaluate if the final action was correct."""
        state = self._state
        task = state.task

        # BINARY REWARD: +1 for correct, -1 for incorrect

        if task.is_possible:
            # Task was possible - agent should have executed the correct action
            if state.final_action == task.ground_truth_action:
                # Check parameters
                if self._params_match(state.final_params, task.ground_truth_params, task):
                    self._finalize(1.0, "Correct action executed")
                else:
                    self._finalize(-1.0, "Wrong parameters")
            elif state.final_action == "respond_to_user":
                # Incorrectly refused a possible task
                self._finalize(-1.0, "Incorrectly refused possible task")
            else:
                self._finalize(-1.0, "Wrong action type")
        else:
            # Task was NOT possible - agent should have refused
            if state.final_action == "respond_to_user":
                # Check if refusal message indicates inability
                message = state.final_params.get("message", "").lower()
                if any(kw in message for kw in ["cannot", "can't", "unable", "not possible", "not allowed"]):
                    self._finalize(1.0, "Correctly refused impossible task")
                else:
                    self._finalize(-1.0, "Refusal message unclear")
            else:
                # Tried to execute impossible action
                self._finalize(-1.0, "Attempted impossible action")

    def _params_match(self, actual: Dict[str, Any], expected: Dict[str, Any], task: Task) -> bool:
        """Check if parameters match (with some flexibility)."""
        if actual is None or expected is None:
            return False

        # Check order_id (lenient about # prefix)
        if "order_id" in expected:
            actual_order = actual.get("order_id", "")
            expected_order = expected["order_id"]
            # Normalize both to have # prefix
            actual_norm = actual_order if actual_order.startswith("#") else f"#{actual_order}"
            if actual_norm != expected_order:
                return False

        # Check item_ids for refund
        if "item_ids" in expected:
            actual_ids = set(actual.get("item_ids", []))
            expected_ids = set(expected["item_ids"])
            if actual_ids != expected_ids:
                return False

        # Check old_item_id and new_item_id for exchange/modify
        if "old_item_id" in expected:
            if actual.get("old_item_id") != expected["old_item_id"]:
                return False

        if "new_item_id" in expected:
            # For exchange/modify, just check that new_item_id is a valid available item
            new_id = actual.get("new_item_id", "")
            if new_id and new_id in task.available_items:
                if not task.available_items[new_id].available:
                    return False
            else:
                return False

        # Check reason for cancel
        if "reason" in expected:
            reason = actual.get("reason", "").lower()
            valid_reasons = ["no longer needed", "ordered by mistake"]
            if not any(r in reason for r in valid_reasons):
                return False

        return True

    def _finalize(self, reward: float, reason: str):
        """Finalize the episode with reward."""
        self.done = True
        self._state.done = True
        self.rewards = {0: reward}
        self._state.final_reason = reason

    def get_summary(self) -> Dict[str, Any]:
        """Get game summary for logging."""
        state = self._state
        task = state.task

        return {
            "action_type": task.action_type,
            "is_possible": task.is_possible,
            "expected_action": task.ground_truth_action,
            "actual_action": state.final_action,
            "steps": state.current_step,
            "repeated_calls": state.repeated_calls,
            "reward": self.rewards.get(0, 0.0),
            "reason": getattr(state, 'final_reason', 'unknown'),
            "tool_sequence": [tc["tool"] for tc in state.tool_calls],
        }


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT_POLICY_GATED = """You are a customer service agent with access to order and user management tools.

Your job is to help customers with their requests while following policy rules:
- Only cancel PENDING orders
- Only refund DELIVERED orders
- Only exchange items in DELIVERED orders (if replacement available)
- Only modify PENDING orders (if replacement available)

IMPORTANT: Always check the order status BEFORE attempting any action.
If the action is not allowed by policy, politely explain why using respond_to_user.

Respond with a single tool call in JSON format: {"name": "tool_name", "arguments": {...}}
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Policy-Gated Action Game")
    print("=" * 60)

    game = PolicyGatedActionGame()

    # Test complexity distribution
    print("\nTesting complexity distribution (1000 samples):")
    from collections import Counter
    complexities = []
    possible_count = 0
    for seed in range(1000):
        game.reset(seed)
        task = game._state.task
        # Complexity is implicit in required_lookups
        complexities.append(len(task.required_lookups))
        if task.is_possible:
            possible_count += 1

    counts = Counter(complexities)
    print(f"Possible tasks: {possible_count}/1000 ({possible_count/10:.1f}%)")
    for c in sorted(counts.keys()):
        print(f"  {c} required lookups: {counts[c]} ({counts[c]/10:.1f}%)")

    # Test a few games
    print("\n" + "=" * 60)
    print("Sample games:")

    for seed in [0, 1, 2, 42, 100]:
        print(f"\n--- Seed {seed} ---")
        game.reset(seed)
        task = game._state.task

        print(f"Action: {task.action_type}")
        print(f"Order status: {task.order.status.value}")
        print(f"Possible: {task.is_possible}")
        print(f"Ground truth: {task.ground_truth_action}")
        print(f"\nRequest: {task.request}")
        print(f"\nObservation preview:\n{game.observe(0)[:600]}...")
