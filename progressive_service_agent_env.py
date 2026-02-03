"""
Progressive Service Agent Environment

A training environment designed to teach the core skills needed for tau-bench
and similar agentic benchmarks, WITHOUT directly copying from them.

=============================================================================
DESIGN PHILOSOPHY
=============================================================================

This environment addresses the 7 KEY SKILL GAPS identified in tau-bench analysis:

1. AUTHENTICATION PROTOCOL
   - Must verify user identity before ANY action
   - Choice between email vs name+zip methods
   - Cannot skip even if user provides ID

2. PROGRESSIVE INFORMATION DISCOVERY
   - Nothing given upfront - must query to learn
   - Order details, product variants, user info all hidden
   - Tracks discovered vs unknown information

3. ONE-SHOT ACTION CONSTRAINTS
   - Critical actions (modify, exchange) can only be called ONCE per entity
   - Forces batching and planning BEFORE acting
   - State transitions that block subsequent actions

4. CONDITIONAL LOGIC WITH FALLBACKS
   - Tasks have "if X available, do A; else do B" structure
   - Nested conditionals up to 3 levels deep
   - Must check conditions BEFORE committing

5. MULTI-ITEM BATCHING
   - Must collect ALL items into single tool call
   - Cannot do sequential modifications
   - Requires understanding the one-shot constraint

6. ENTITY DISAMBIGUATION
   - product_id vs item_id vs variant_id
   - Similar-looking IDs that must be distinguished
   - Validates correct ID types in tool calls

7. CONFIRMATION PROTOCOL
   - Must summarize planned actions
   - Must get explicit user confirmation ("yes")
   - Handle user adding/changing requests

=============================================================================
CURRICULUM LEVELS
=============================================================================

Level 1: Simple Query
   - Authenticate + single lookup + respond
   - Trains: auth protocol, basic tool usage

Level 2: Single Action
   - Authenticate + discover + single action + confirm
   - Trains: info discovery, confirmation protocol

Level 3: Conditional Action
   - If condition met, do A; else do B
   - Trains: conditional reasoning, fallback logic

Level 4: Multi-Item Batch
   - Multiple items must be batched into one call
   - Trains: batching, one-shot constraint

Level 5: User Dynamics
   - User changes mind, adds requests, provides wrong info
   - Trains: adaptation, re-planning, validation

Level 6: Full Complexity
   - Combines all above with nested conditionals
   - Trains: complete agentic reasoning

=============================================================================
"""

import random
import json
import re
import hashlib
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class OrderStatus(str, Enum):
    PENDING = "pending"
    PENDING_MODIFIED = "pending_modified"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class UserPersonality(str, Enum):
    HELPFUL = "helpful"           # Provides info when asked
    TERSE = "terse"               # Minimal responses
    CONFUSED = "confused"         # Sometimes provides wrong info
    IMPATIENT = "impatient"       # Changes mind if too many questions
    ADVERSARIAL = "adversarial"   # Triggers edge cases


class TaskComplexity(int, Enum):
    SIMPLE_QUERY = 1
    SINGLE_ACTION = 2
    CONDITIONAL = 3
    MULTI_ITEM = 4
    USER_DYNAMICS = 5
    FULL_COMPLEXITY = 6


# ID prefixes for disambiguation training
ID_PREFIXES = {
    "user": "USR_",
    "order": "#ORD",
    "product": "PROD_",
    "item": "ITEM_",
    "variant": "VAR_",
    "payment": "PAY_",
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PaymentMethod:
    payment_id: str
    method_type: str  # "credit_card", "gift_card", "paypal"
    last_four: str
    balance: Optional[float] = None  # For gift cards


@dataclass
class ProductVariant:
    variant_id: str
    product_id: str
    name: str
    attributes: Dict[str, str]  # e.g., {"color": "red", "size": "M"}
    price: float
    in_stock: bool
    stock_quantity: int = 0


@dataclass
class OrderItem:
    item_id: str
    variant_id: str
    product_id: str
    name: str
    attributes: Dict[str, str]
    price: float
    quantity: int


@dataclass
class Order:
    order_id: str
    user_id: str
    status: OrderStatus
    items: List[OrderItem]
    payment_method_id: str
    total: float
    address: Dict[str, str]
    created_days_ago: int
    # Tracking one-shot constraints
    items_modified: bool = False
    payment_modified: bool = False


@dataclass
class User:
    user_id: str
    first_name: str
    last_name: str
    email: str
    zip_code: str
    payment_methods: List[PaymentMethod]
    address: Dict[str, str]


@dataclass
class Product:
    product_id: str
    name: str
    category: str
    variants: List[ProductVariant]


# =============================================================================
# USER SIMULATOR
# =============================================================================

@dataclass
class UserState:
    """Tracks user simulator state across conversation."""
    known_to_agent: Set[str] = field(default_factory=set)  # What agent has learned
    messages_sent: int = 0
    confirmations_given: int = 0
    mind_changes: int = 0
    current_request: Optional[str] = None
    pending_additions: List[str] = field(default_factory=list)


class UserSimulator:
    """
    Simulates user behavior with configurable personality.

    Key behaviors:
    - Progressive information disclosure
    - Mind changes based on agent behavior
    - Confirmation/rejection of proposed actions
    """

    def __init__(
        self,
        user: User,
        personality: UserPersonality,
        task_instructions: str,
        hidden_preferences: Dict[str, Any],
        mind_change_triggers: List[Dict[str, Any]],
        rng: random.Random,
    ):
        self.user = user
        self.personality = personality
        self.task_instructions = task_instructions
        self.hidden_preferences = hidden_preferences
        self.mind_change_triggers = mind_change_triggers
        self.rng = rng
        self.state = UserState()

    def get_initial_message(self) -> str:
        """Generate opening message based on personality."""
        base = self.task_instructions

        if self.personality == UserPersonality.HELPFUL:
            # Provides name AND email upfront (makes auth easy)
            return f"Hi, I'm {self.user.first_name} {self.user.last_name}. My email is {self.user.email}. {base}"
        elif self.personality == UserPersonality.TERSE:
            # Provides name + zip (alternative auth path)
            return f"{self.user.first_name} {self.user.last_name}, {self.user.address['zip']}. {base}"
        elif self.personality == UserPersonality.CONFUSED:
            # Provides email but might be uncertain
            return f"Hi, I think my email is {self.user.email}? {base}"
        elif self.personality == UserPersonality.IMPATIENT:
            # Provides email, wants fast service
            return f"My email is {self.user.email}. {base} I'm in a hurry, please be quick."
        else:  # ADVERSARIAL
            # Provides user_id directly (agent should still verify via email/name)
            return f"My user ID is {self.user.user_id}. My email is {self.user.email}. {base}"

    def respond_to_agent(self, agent_message: str, agent_action: Optional[Dict] = None) -> str:
        """Generate response based on agent's message/action."""
        self.state.messages_sent += 1

        # Check for mind change triggers
        for trigger in self.mind_change_triggers:
            if self._check_trigger(trigger, agent_message, agent_action):
                self.state.mind_changes += 1
                return self._generate_mind_change_response(trigger)

        # Handle different agent intents
        lower_msg = agent_message.lower()

        # Authentication requests
        if "email" in lower_msg:
            return self._respond_to_email_request()
        if "zip" in lower_msg or "postal" in lower_msg:
            return self._respond_to_zip_request()
        if "name" in lower_msg and "?" in agent_message:
            return self._respond_to_name_request()

        # Confirmation requests
        if "confirm" in lower_msg or "proceed" in lower_msg or "is that correct" in lower_msg:
            return self._respond_to_confirmation(agent_message)

        # Information requests
        if "order" in lower_msg and ("which" in lower_msg or "what" in lower_msg or "number" in lower_msg):
            return self._respond_to_order_request()

        # Default response
        return self._generate_default_response(agent_message)

    def _check_trigger(self, trigger: Dict, message: str, action: Optional[Dict]) -> bool:
        """Check if a mind change trigger condition is met."""
        trigger_type = trigger.get("type")

        if trigger_type == "confirmation_count":
            return self.state.confirmations_given >= trigger.get("threshold", 1)
        elif trigger_type == "message_count":
            return self.state.messages_sent >= trigger.get("threshold", 5)
        elif trigger_type == "keyword":
            return trigger.get("keyword", "").lower() in message.lower()
        elif trigger_type == "action":
            return action and action.get("name") == trigger.get("action_name")

        return False

    def _generate_mind_change_response(self, trigger: Dict) -> str:
        """Generate response when user changes mind."""
        change_type = trigger.get("change_type", "add_request")

        if change_type == "add_request":
            addition = trigger.get("addition", "Actually, I'd also like to do something else.")
            self.state.pending_additions.append(addition)
            return f"Wait, {addition}"
        elif change_type == "cancel_request":
            return "Actually, never mind. I don't want to do this anymore."
        elif change_type == "change_preference":
            new_pref = trigger.get("new_preference", "something different")
            return f"Actually, I changed my mind. I'd prefer {new_pref} instead."
        else:
            return "Hold on, let me think about this..."

    def _respond_to_email_request(self) -> str:
        if self.personality == UserPersonality.CONFUSED:
            # Give slightly wrong email sometimes
            if self.rng.random() < 0.3:
                wrong_email = self.user.email.replace("@", "_at_")
                return f"I think it's {wrong_email}... or maybe {self.user.email}"

        self.state.known_to_agent.add("email")
        return f"It's {self.user.email}"

    def _respond_to_zip_request(self) -> str:
        self.state.known_to_agent.add("zip")
        return f"My zip code is {self.user.zip_code}"

    def _respond_to_name_request(self) -> str:
        self.state.known_to_agent.add("name")
        return f"My name is {self.user.first_name} {self.user.last_name}"

    def _respond_to_confirmation(self, agent_message: str) -> str:
        """Handle confirmation requests."""
        self.state.confirmations_given += 1

        # Check for pending additions first
        if self.state.pending_additions:
            addition = self.state.pending_additions.pop(0)
            return f"Wait, before you do that - {addition}"

        # Personality-based confirmation
        if self.personality == UserPersonality.IMPATIENT and self.state.confirmations_given > 1:
            return "Yes, yes, just do it already!"
        elif self.personality == UserPersonality.ADVERSARIAL:
            return "Yes, but make sure you do it exactly as I said."
        else:
            return "Yes, that's correct. Please proceed."

    def _respond_to_order_request(self) -> str:
        """Respond when agent asks about order."""
        # Could return order ID or say they don't remember
        if self.personality == UserPersonality.CONFUSED:
            return "I'm not sure of the exact order number. Can you look it up?"
        return "Let me check... I should have ordered recently."

    def _generate_default_response(self, agent_message: str) -> str:
        """Generate default response."""
        if self.personality == UserPersonality.TERSE:
            return "Okay."
        elif self.personality == UserPersonality.HELPFUL:
            return "Thank you for helping me with this."
        else:
            return "I understand."


# =============================================================================
# TASK GENERATION
# =============================================================================

@dataclass
class ConditionalBranch:
    """Represents a conditional branch in task logic."""
    condition_type: str  # "product_available", "order_status", "price_threshold"
    condition_params: Dict[str, Any]
    if_true_action: str
    if_true_params: Dict[str, Any]
    if_false_action: Optional[str] = None
    if_false_params: Optional[Dict[str, Any]] = None
    nested_branch: Optional['ConditionalBranch'] = None


@dataclass
class TaskSpec:
    """Complete specification of a task."""
    complexity: TaskComplexity
    user: User
    orders: List[Order]
    products: List[Product]

    # User simulator config
    personality: UserPersonality
    task_instructions: str
    hidden_preferences: Dict[str, Any]
    mind_change_triggers: List[Dict[str, Any]]

    # Ground truth
    required_auth_method: str  # "email" or "name_zip"
    required_lookups: Set[str]  # Tool names that must be called
    conditional_logic: Optional[ConditionalBranch] = None
    expected_actions: List[Dict[str, Any]] = field(default_factory=list)
    items_to_batch: List[str] = field(default_factory=list)  # Item IDs that must be batched


class TaskGenerator:
    """Generates tasks of varying complexity."""

    # Product templates
    PRODUCTS = [
        ("Wireless Headphones", "electronics", ["color", "noise_cancelling"]),
        ("Mechanical Keyboard", "electronics", ["switch_type", "size", "backlight"]),
        ("Smart Watch", "electronics", ["band_type", "screen_size"]),
        ("Running Shoes", "apparel", ["size", "color", "cushioning"]),
        ("Winter Jacket", "apparel", ["size", "color", "material"]),
        ("Backpack", "accessories", ["size", "color", "material"]),
        ("Water Bottle", "accessories", ["size", "material", "insulated"]),
        ("Desk Lamp", "home", ["brightness", "color_temp", "power_source"]),
        ("Office Chair", "furniture", ["material", "armrests", "lumbar_support"]),
        ("Standing Desk", "furniture", ["size", "motor_type", "color"]),
    ]

    FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
                   "Avery", "Harper", "Sage", "Blake", "Drew", "Jamie", "Reese"]
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                  "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor"]

    ATTRIBUTE_VALUES = {
        "color": ["black", "white", "blue", "red", "silver", "green"],
        "size": ["S", "M", "L", "XL"],
        "switch_type": ["linear", "tactile", "clicky"],
        "backlight": ["RGB", "white", "none"],
        "noise_cancelling": ["yes", "no"],
        "band_type": ["silicone", "leather", "metal"],
        "screen_size": ["40mm", "44mm", "45mm"],
        "cushioning": ["minimal", "moderate", "maximum"],
        "material": ["synthetic", "leather", "cotton", "polyester", "aluminum", "wood"],
        "brightness": ["low", "medium", "high"],
        "color_temp": ["warm", "neutral", "cool"],
        "power_source": ["USB", "AC", "battery"],
        "armrests": ["fixed", "adjustable", "none"],
        "lumbar_support": ["yes", "no"],
        "motor_type": ["single", "dual"],
        "insulated": ["yes", "no"],
    }

    def __init__(self, rng: random.Random):
        self.rng = rng

    def generate_task(self, complexity: TaskComplexity) -> TaskSpec:
        """Generate a task of specified complexity."""
        # Generate base entities
        user = self._generate_user()
        products = self._generate_products(3 + complexity.value)
        orders = self._generate_orders(user, products, 1 + complexity.value // 2)

        # Generate task-specific configuration
        if complexity == TaskComplexity.SIMPLE_QUERY:
            return self._generate_simple_query_task(user, orders, products)
        elif complexity == TaskComplexity.SINGLE_ACTION:
            return self._generate_single_action_task(user, orders, products)
        elif complexity == TaskComplexity.CONDITIONAL:
            return self._generate_conditional_task(user, orders, products)
        elif complexity == TaskComplexity.MULTI_ITEM:
            return self._generate_multi_item_task(user, orders, products)
        elif complexity == TaskComplexity.USER_DYNAMICS:
            return self._generate_user_dynamics_task(user, orders, products)
        else:  # FULL_COMPLEXITY
            return self._generate_full_complexity_task(user, orders, products)

    def _generate_user(self) -> User:
        """Generate a random user."""
        first = self.rng.choice(self.FIRST_NAMES)
        last = self.rng.choice(self.LAST_NAMES)
        user_id = f"{ID_PREFIXES['user']}{self.rng.randint(100000, 999999)}"

        # Generate payment methods
        payment_methods = [
            PaymentMethod(
                payment_id=f"{ID_PREFIXES['payment']}{self.rng.randint(10000, 99999)}",
                method_type="credit_card",
                last_four=str(self.rng.randint(1000, 9999)),
            ),
            PaymentMethod(
                payment_id=f"{ID_PREFIXES['payment']}{self.rng.randint(10000, 99999)}",
                method_type="gift_card",
                last_four=str(self.rng.randint(1000, 9999)),
                balance=round(self.rng.uniform(10, 200), 2),
            ),
        ]

        return User(
            user_id=user_id,
            first_name=first,
            last_name=last,
            email=f"{first.lower()}.{last.lower()}@email.com",
            zip_code=str(self.rng.randint(10000, 99999)),
            payment_methods=payment_methods,
            address={
                "street": f"{self.rng.randint(100, 9999)} Main St",
                "city": self.rng.choice(["New York", "Los Angeles", "Chicago", "Houston"]),
                "state": self.rng.choice(["NY", "CA", "IL", "TX"]),
                "zip": str(self.rng.randint(10000, 99999)),
            }
        )

    def _generate_products(self, count: int) -> List[Product]:
        """Generate products with variants."""
        products = []
        selected = self.rng.sample(self.PRODUCTS, min(count, len(self.PRODUCTS)))

        for name, category, attr_names in selected:
            product_id = f"{ID_PREFIXES['product']}{self.rng.randint(100000, 999999)}"

            # Generate 2-4 variants per product
            variants = []
            num_variants = self.rng.randint(2, 4)
            for i in range(num_variants):
                variant_id = f"{ID_PREFIXES['variant']}{self.rng.randint(100000, 999999)}"
                attributes = {
                    attr: self.rng.choice(self.ATTRIBUTE_VALUES.get(attr, ["default"]))
                    for attr in attr_names
                }
                variants.append(ProductVariant(
                    variant_id=variant_id,
                    product_id=product_id,
                    name=f"{name} ({', '.join(f'{k}={v}' for k, v in attributes.items())})",
                    attributes=attributes,
                    price=round(self.rng.uniform(20, 300), 2),
                    in_stock=self.rng.random() > 0.2,  # 80% in stock
                    stock_quantity=self.rng.randint(0, 50) if self.rng.random() > 0.2 else 0,
                ))

            products.append(Product(
                product_id=product_id,
                name=name,
                category=category,
                variants=variants,
            ))

        return products

    def _generate_orders(self, user: User, products: List[Product], count: int) -> List[Order]:
        """Generate orders for the user."""
        orders = []
        statuses = [OrderStatus.PENDING, OrderStatus.DELIVERED, OrderStatus.SHIPPED]

        for i in range(count):
            order_id = f"{ID_PREFIXES['order']}{self.rng.randint(100000, 999999)}"
            status = self.rng.choice(statuses)

            # Generate 1-3 items per order
            items = []
            num_items = self.rng.randint(1, 3)
            selected_products = self.rng.sample(products, min(num_items, len(products)))

            total = 0
            for product in selected_products:
                if not product.variants:
                    continue
                variant = self.rng.choice(product.variants)
                item_id = f"{ID_PREFIXES['item']}{self.rng.randint(100000, 999999)}"
                qty = self.rng.randint(1, 2)

                items.append(OrderItem(
                    item_id=item_id,
                    variant_id=variant.variant_id,
                    product_id=product.product_id,
                    name=variant.name,
                    attributes=variant.attributes,
                    price=variant.price,
                    quantity=qty,
                ))
                total += variant.price * qty

            orders.append(Order(
                order_id=order_id,
                user_id=user.user_id,
                status=status,
                items=items,
                payment_method_id=user.payment_methods[0].payment_id,
                total=round(total, 2),
                address=user.address.copy(),
                created_days_ago=self.rng.randint(1, 30),
            ))

        return orders

    def _generate_simple_query_task(self, user: User, orders: List[Order],
                                     products: List[Product]) -> TaskSpec:
        """Level 1: Simple lookup task."""
        order = orders[0]

        return TaskSpec(
            complexity=TaskComplexity.SIMPLE_QUERY,
            user=user,
            orders=orders,
            products=products,
            personality=UserPersonality.HELPFUL,
            task_instructions=f"I want to know the status of my recent order.",
            hidden_preferences={},
            mind_change_triggers=[],
            required_auth_method=self.rng.choice(["email", "name_zip"]),
            required_lookups={"authenticate_user", "get_order_details"},
            expected_actions=[
                {"name": "respond_to_user", "params": {"contains": order.status.value}}
            ],
        )

    def _generate_single_action_task(self, user: User, orders: List[Order],
                                      products: List[Product]) -> TaskSpec:
        """Level 2: Single action with confirmation."""
        # Find a pending order for cancellation
        pending_orders = [o for o in orders if o.status == OrderStatus.PENDING]
        if not pending_orders:
            orders[0].status = OrderStatus.PENDING
            pending_orders = [orders[0]]

        order = pending_orders[0]

        return TaskSpec(
            complexity=TaskComplexity.SINGLE_ACTION,
            user=user,
            orders=orders,
            products=products,
            personality=UserPersonality.HELPFUL,
            task_instructions=f"I want to cancel order {order.order_id}. I no longer need it.",
            hidden_preferences={},
            mind_change_triggers=[],
            required_auth_method="name_zip",
            required_lookups={"authenticate_user", "get_order_details"},
            expected_actions=[
                {"name": "cancel_order", "params": {"order_id": order.order_id, "reason": "no longer needed"}}
            ],
        )

    def _generate_conditional_task(self, user: User, orders: List[Order],
                                    products: List[Product]) -> TaskSpec:
        """Level 3: Conditional logic task."""
        # Find a delivered order for exchange
        delivered_orders = [o for o in orders if o.status == OrderStatus.DELIVERED]
        if not delivered_orders:
            orders[0].status = OrderStatus.DELIVERED
            delivered_orders = [orders[0]]

        order = delivered_orders[0]
        if not order.items:
            return self._generate_single_action_task(user, orders, products)

        item = order.items[0]
        product = next((p for p in products if p.product_id == item.product_id), None)

        if not product or len(product.variants) < 2:
            return self._generate_single_action_task(user, orders, products)

        # Create primary preference (might not be available)
        other_variants = [v for v in product.variants if v.variant_id != item.variant_id]
        primary_variant = self.rng.choice(other_variants)
        primary_variant.in_stock = self.rng.random() < 0.5  # 50% chance available

        # Create fallback preference (always available)
        # Need a variant different from both item's variant and primary variant
        fallback_candidates = [v for v in product.variants
                               if v.variant_id not in [item.variant_id, primary_variant.variant_id]]
        if fallback_candidates:
            fallback_variant = self.rng.choice(fallback_candidates)
        else:
            # If only 2 variants exist, reuse primary as fallback but make it always available
            # This creates a simpler task where primary preference always works
            fallback_variant = primary_variant
            primary_variant.in_stock = True  # Force available since no real fallback
        fallback_variant.in_stock = True

        primary_attr = list(primary_variant.attributes.items())[0] if primary_variant.attributes else ("type", "primary")
        fallback_attr = list(fallback_variant.attributes.items())[0] if fallback_variant.attributes else ("type", "fallback")

        return TaskSpec(
            complexity=TaskComplexity.CONDITIONAL,
            user=user,
            orders=orders,
            products=products,
            personality=UserPersonality.HELPFUL,
            task_instructions=(
                f"I want to exchange my {item.name} from order {order.order_id}. "
                f"I'd prefer one with {primary_attr[0]}={primary_attr[1]}. "
                f"But if that's not available, I'll take one with {fallback_attr[0]}={fallback_attr[1]} instead."
            ),
            hidden_preferences={
                "primary": {"attribute": primary_attr[0], "value": primary_attr[1]},
                "fallback": {"attribute": fallback_attr[0], "value": fallback_attr[1]},
            },
            mind_change_triggers=[],
            required_auth_method="email",
            required_lookups={"authenticate_user", "get_order_details", "get_product_details"},
            conditional_logic=ConditionalBranch(
                condition_type="variant_available",
                condition_params={"attribute": primary_attr[0], "value": primary_attr[1]},
                if_true_action="exchange_items",
                if_true_params={"item_ids": [item.item_id], "new_variant_id": primary_variant.variant_id},
                if_false_action="exchange_items",
                if_false_params={"item_ids": [item.item_id], "new_variant_id": fallback_variant.variant_id},
            ),
            expected_actions=[
                {"name": "exchange_items", "params": {
                    "order_id": order.order_id,
                    "item_ids": [item.item_id],
                    "new_variant_id": primary_variant.variant_id if primary_variant.in_stock else fallback_variant.variant_id,
                }}
            ],
        )

    def _generate_multi_item_task(self, user: User, orders: List[Order],
                                   products: List[Product]) -> TaskSpec:
        """Level 4: Multi-item batching task."""
        # Find a delivered order with multiple items
        multi_item_orders = [o for o in orders if o.status == OrderStatus.DELIVERED and len(o.items) >= 2]
        if not multi_item_orders:
            # Create one
            orders[0].status = OrderStatus.DELIVERED
            if len(orders[0].items) < 2:
                # Add another item
                product = products[0]
                if product.variants:
                    variant = product.variants[0]
                    orders[0].items.append(OrderItem(
                        item_id=f"{ID_PREFIXES['item']}{self.rng.randint(100000, 999999)}",
                        variant_id=variant.variant_id,
                        product_id=product.product_id,
                        name=variant.name,
                        attributes=variant.attributes,
                        price=variant.price,
                        quantity=1,
                    ))
            multi_item_orders = [orders[0]]

        order = multi_item_orders[0]
        items_to_batch = [item.item_id for item in order.items[:2]]

        return TaskSpec(
            complexity=TaskComplexity.MULTI_ITEM,
            user=user,
            orders=orders,
            products=products,
            personality=UserPersonality.HELPFUL,
            task_instructions=(
                f"I want to return multiple items from order {order.order_id}: "
                f"the {order.items[0].name} and the {order.items[1].name}. "
                f"Please process them together."
            ),
            hidden_preferences={},
            mind_change_triggers=[],
            required_auth_method="name_zip",
            required_lookups={"authenticate_user", "get_order_details"},
            expected_actions=[
                {"name": "return_items", "params": {
                    "order_id": order.order_id,
                    "item_ids": items_to_batch,  # MUST be batched!
                }}
            ],
            items_to_batch=items_to_batch,
        )

    def _generate_user_dynamics_task(self, user: User, orders: List[Order],
                                      products: List[Product]) -> TaskSpec:
        """Level 5: User changes mind during conversation."""
        order = orders[0]
        order.status = OrderStatus.DELIVERED

        if not order.items:
            return self._generate_multi_item_task(user, orders, products)

        item = order.items[0]

        return TaskSpec(
            complexity=TaskComplexity.USER_DYNAMICS,
            user=user,
            orders=orders,
            products=products,
            personality=UserPersonality.IMPATIENT,
            task_instructions=f"I want to return my {item.name} from order {order.order_id}.",
            hidden_preferences={},
            mind_change_triggers=[
                {
                    "type": "confirmation_count",
                    "threshold": 1,
                    "change_type": "change_preference",
                    "new_preference": "exchange it for the same item instead of returning",
                }
            ],
            required_auth_method="email",
            required_lookups={"authenticate_user", "get_order_details", "get_product_details"},
            expected_actions=[
                # After mind change, should exchange instead of return
                {"name": "exchange_items", "params": {
                    "order_id": order.order_id,
                    "item_ids": [item.item_id],
                }}
            ],
        )

    def _generate_full_complexity_task(self, user: User, orders: List[Order],
                                        products: List[Product]) -> TaskSpec:
        """Level 6: Full complexity combining all challenges."""
        # Ensure we have appropriate orders
        if len(orders) < 2:
            orders.append(self._generate_orders(user, products, 1)[0])

        orders[0].status = OrderStatus.DELIVERED
        orders[1].status = OrderStatus.PENDING

        # Ensure multiple items in first order
        while len(orders[0].items) < 2:
            product = self.rng.choice(products)
            if product.variants:
                variant = product.variants[0]
                orders[0].items.append(OrderItem(
                    item_id=f"{ID_PREFIXES['item']}{self.rng.randint(100000, 999999)}",
                    variant_id=variant.variant_id,
                    product_id=product.product_id,
                    name=variant.name,
                    attributes=variant.attributes,
                    price=variant.price,
                    quantity=1,
                ))

        item1 = orders[0].items[0]
        item2 = orders[0].items[1]
        pending_item = orders[1].items[0] if orders[1].items else None

        # Complex nested conditional
        return TaskSpec(
            complexity=TaskComplexity.FULL_COMPLEXITY,
            user=user,
            orders=orders,
            products=products,
            personality=UserPersonality.ADVERSARIAL,
            task_instructions=(
                f"I have two things to do. First, from order {orders[0].order_id}, I want to exchange "
                f"both the {item1.name} and {item2.name} for different variants - preferably in black. "
                f"If black isn't available for either, just exchange that one for the cheapest available. "
                f"Second, {'I want to cancel order ' + orders[1].order_id + ' if possible, otherwise just leave it' if pending_item else 'thats all'}."
            ),
            hidden_preferences={
                "primary_color": "black",
                "fallback": "cheapest",
            },
            mind_change_triggers=[
                {
                    "type": "confirmation_count",
                    "threshold": 2,
                    "change_type": "add_request",
                    "addition": "also update my address to the one on my most recent order",
                }
            ],
            required_auth_method="name_zip",
            required_lookups={"authenticate_user", "get_order_details", "get_product_details", "get_user_details"},
            conditional_logic=ConditionalBranch(
                condition_type="variant_available",
                condition_params={"attribute": "color", "value": "black"},
                if_true_action="exchange_items",
                if_true_params={"item_ids": [item1.item_id, item2.item_id]},  # BATCHED
                if_false_action="exchange_items",
                if_false_params={"item_ids": [item1.item_id, item2.item_id], "use_fallback": True},
            ),
            expected_actions=[
                {"name": "exchange_items", "params": {"order_id": orders[0].order_id, "item_ids": [item1.item_id, item2.item_id]}},
                {"name": "cancel_order", "params": {"order_id": orders[1].order_id}} if pending_item else None,
            ],
            items_to_batch=[item1.item_id, item2.item_id],
        )


# =============================================================================
# GAME STATE AND ENVIRONMENT
# =============================================================================

@dataclass
class GameState:
    """Complete state of a game episode."""
    task: TaskSpec
    user_simulator: UserSimulator

    # Discovery tracking
    authenticated: bool = False
    auth_method_used: Optional[str] = None
    discovered_user_id: Optional[str] = None
    discovered_orders: Set[str] = field(default_factory=set)
    discovered_products: Set[str] = field(default_factory=set)

    # Action tracking
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    one_shot_actions_used: Dict[str, Set[str]] = field(default_factory=lambda: {
        "exchange": set(),
        "modify": set(),
        "return": set(),
    })

    # Confirmation tracking
    pending_confirmation: Optional[Dict[str, Any]] = None
    confirmed_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Conversation
    conversation: List[Dict[str, str]] = field(default_factory=list)

    # Termination
    current_step: int = 0
    done: bool = False
    success: bool = False
    failure_reason: Optional[str] = None


class ProgressiveServiceAgentEnv:
    """
    Main environment class.

    Implements the GameEnv protocol for compatibility with training infrastructure.
    """

    def __init__(
        self,
        max_steps: int = 20,
        complexity: Optional[TaskComplexity] = None,
        curriculum: bool = False,
        curriculum_threshold: float = 0.7,
        games_per_iter: int = 512,
        curriculum_iters: int = 3,
    ):
        self.max_steps = max_steps
        self.fixed_complexity = complexity
        self._rng = random.Random()
        self._state: Optional[GameState] = None
        self._task_generator: Optional[TaskGenerator] = None

        # Curriculum learning settings
        self.curriculum_enabled = curriculum
        self.curriculum_threshold = curriculum_threshold
        self.games_per_iter = games_per_iter
        self.curriculum_iters = curriculum_iters
        # Window in episodes = games_per_iter * curriculum_iters
        self.curriculum_window = games_per_iter * curriculum_iters

        # Curriculum state (persists across resets)
        self._curriculum_level = 0  # Index into TaskComplexity
        self._level_history: Dict[int, List[float]] = {i: [] for i in range(len(TaskComplexity))}
        self._total_episodes = 0
        self._level_episodes = 0

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def _get_curriculum_complexity(self) -> TaskComplexity:
        """Get complexity level based on curriculum state."""
        complexities = list(TaskComplexity)
        return complexities[self._curriculum_level]

    def _get_level_win_rate(self, level: int) -> float:
        """Calculate win rate for a specific level."""
        history = self._level_history[level]
        if not history:
            return 0.0
        # Use last N episodes for this level
        recent = history[-self.curriculum_window:]
        return sum(recent) / len(recent)

    def _update_curriculum(self, success: bool):
        """Update curriculum state after an episode ends."""
        if not self.curriculum_enabled:
            return

        level = self._curriculum_level
        self._level_history[level].append(1.0 if success else 0.0)
        self._total_episodes += 1
        self._level_episodes += 1

        # Check for level advancement
        # Require at least 1 full iteration of games before considering advancement
        win_rate = self._get_level_win_rate(level)
        min_episodes = self.games_per_iter  # At least 1 iteration

        if (len(self._level_history[level]) >= min_episodes and
            win_rate >= self.curriculum_threshold and
            level < len(TaskComplexity) - 1):

            old_level = level
            self._curriculum_level += 1
            self._level_episodes = 0
            complexity_names = [c.name for c in TaskComplexity]
            iters_at_level = len(self._level_history[old_level]) / self.games_per_iter
            print(f"[Curriculum] Level up! {complexity_names[old_level]} -> {complexity_names[self._curriculum_level]} "
                  f"(win_rate={win_rate:.2f}, threshold={self.curriculum_threshold}, iters={iters_at_level:.1f})")

    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum learning status."""
        complexities = list(TaskComplexity)
        level_episodes = len(self._level_history[self._curriculum_level])
        return {
            "enabled": self.curriculum_enabled,
            "current_level": self._curriculum_level,
            "current_complexity": complexities[self._curriculum_level].name,
            "level_win_rate": self._get_level_win_rate(self._curriculum_level),
            "total_episodes": self._total_episodes,
            "level_episodes": level_episodes,
            "level_iterations": level_episodes / self.games_per_iter,
            "games_per_iter": self.games_per_iter,
            "curriculum_iters_window": self.curriculum_iters,
            "threshold": self.curriculum_threshold,
            "all_level_win_rates": {
                complexities[i].name: self._get_level_win_rate(i)
                for i in range(len(complexities))
            },
        }

    def reset(self, seed: int):
        """Reset environment with new seed."""
        self._rng = random.Random(seed)
        self._task_generator = TaskGenerator(self._rng)

        # Determine complexity
        if self.fixed_complexity:
            complexity = self.fixed_complexity
        elif self.curriculum_enabled:
            # Use curriculum-based complexity selection
            complexity = self._get_curriculum_complexity()
        else:
            # Random with fixed weights (no curriculum)
            weights = [30, 25, 20, 12, 8, 5]  # Heavier on simpler tasks
            complexity = self._rng.choices(list(TaskComplexity), weights=weights, k=1)[0]

        # Generate task
        task = self._task_generator.generate_task(complexity)

        # Create user simulator
        user_sim = UserSimulator(
            user=task.user,
            personality=task.personality,
            task_instructions=task.task_instructions,
            hidden_preferences=task.hidden_preferences,
            mind_change_triggers=task.mind_change_triggers,
            rng=self._rng,
        )

        # Initialize state
        self._state = GameState(task=task, user_simulator=user_sim)

        # Add initial user message to conversation
        initial_msg = user_sim.get_initial_message()
        self._state.conversation.append({"role": "user", "content": initial_msg})

        # Reset protocol fields
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        """Get observation for the agent."""
        state = self._state
        task = state.task

        lines = []

        # System instructions
        lines.append("=" * 60)
        lines.append("CUSTOMER SERVICE AGENT ENVIRONMENT")
        lines.append("=" * 60)
        lines.append("")
        lines.append("CRITICAL RULES:")
        lines.append("1. You MUST authenticate the user before any action (even if they give you their ID)")
        lines.append("2. You MUST get explicit 'yes' confirmation before any state-changing action")
        lines.append("3. Exchange/Modify/Return can only be called ONCE per order - batch all items!")
        lines.append("4. Check order status before attempting actions:")
        lines.append("   - cancel_order: only PENDING orders")
        lines.append("   - modify_items: only PENDING orders (ONE SHOT)")
        lines.append("   - exchange_items: only DELIVERED orders (ONE SHOT)")
        lines.append("   - return_items: only DELIVERED orders (ONE SHOT)")
        lines.append("")

        # Available tools
        lines.append("AVAILABLE TOOLS:")
        lines.append("Authentication:")
        lines.append('  - find_user_by_email: {"name": "find_user_by_email", "arguments": {"email": "..."}}')
        lines.append('  - find_user_by_name_zip: {"name": "find_user_by_name_zip", "arguments": {"first_name": "...", "last_name": "...", "zip": "..."}}')
        lines.append("Lookup:")
        lines.append('  - get_user_details: {"name": "get_user_details", "arguments": {"user_id": "..."}}')
        lines.append('  - get_order_details: {"name": "get_order_details", "arguments": {"order_id": "..."}}')
        lines.append('  - get_product_details: {"name": "get_product_details", "arguments": {"product_id": "..."}}')
        lines.append("Actions (require confirmation):")
        lines.append('  - cancel_order: {"name": "cancel_order", "arguments": {"order_id": "...", "reason": "..."}}')
        lines.append('  - modify_items: {"name": "modify_items", "arguments": {"order_id": "...", "item_ids": [...], "new_variant_ids": [...]}}')
        lines.append('  - exchange_items: {"name": "exchange_items", "arguments": {"order_id": "...", "item_ids": [...], "new_variant_ids": [...]}}')
        lines.append('  - return_items: {"name": "return_items", "arguments": {"order_id": "...", "item_ids": [...]}}')
        lines.append("Communication:")
        lines.append('  - respond_to_user: {"name": "respond_to_user", "arguments": {"message": "..."}}')
        lines.append('  - request_confirmation: {"name": "request_confirmation", "arguments": {"action_summary": "..."}}')
        lines.append("")

        # Current state
        lines.append("CURRENT STATE:")
        lines.append(f"  Authenticated: {state.authenticated}")
        if state.authenticated:
            lines.append(f"  User ID: {state.discovered_user_id}")
        lines.append(f"  Discovered Orders: {list(state.discovered_orders)}")
        lines.append(f"  Discovered Products: {list(state.discovered_products)}")
        lines.append(f"  One-shot actions used: {dict(state.one_shot_actions_used)}")
        lines.append("")

        # Conversation history (last 10 messages)
        lines.append("CONVERSATION:")
        for msg in state.conversation[-10:]:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"  [{role}]: {content}")
        lines.append("")

        lines.append("Respond with a JSON tool call:")

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
        state.current_step += 1

        # Parse action
        if action is None:
            self._finalize(False, "No action provided")
            return

        tool_call = self._parse_tool_call(action)
        if tool_call is None:
            self._finalize(False, "Invalid JSON format")
            self.invalid_player = 0
            return

        tool_name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        # Record tool call
        state.tool_calls.append({"name": tool_name, "arguments": args})

        # Execute tool
        result = self._execute_tool(tool_name, args)

        # Add result to conversation
        state.conversation.append({
            "role": "system",
            "content": f"Tool result for {tool_name}: {json.dumps(result)}"
        })

        # Check for user response if needed
        if tool_name == "respond_to_user" or tool_name == "request_confirmation":
            user_response = state.user_simulator.respond_to_agent(
                args.get("message", args.get("action_summary", "")),
                tool_call
            )
            state.conversation.append({"role": "user", "content": user_response})

            # Check for confirmation
            if tool_name == "request_confirmation":
                if "yes" in user_response.lower() and "wait" not in user_response.lower():
                    state.confirmed_actions.append(state.pending_confirmation)
                    state.pending_confirmation = None

        # Check termination conditions
        if state.current_step >= self.max_steps:
            self._finalize(False, "Max steps exceeded")
        elif self._check_task_completion():
            self._finalize(True, "Task completed successfully")

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from model output."""
        try:
            # Find JSON object
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

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return result."""
        state = self._state
        task = state.task

        # Authentication tools
        if tool_name == "find_user_by_email":
            email = args.get("email", "")
            if email.lower() == task.user.email.lower():
                state.authenticated = True
                state.auth_method_used = "email"
                state.discovered_user_id = task.user.user_id
                # Return user info including their order IDs for easy lookup
                return {
                    "success": True,
                    "user_id": task.user.user_id,
                    "name": f"{task.user.first_name} {task.user.last_name}",
                    "orders": [{"order_id": o.order_id, "status": o.status.value} for o in task.orders],
                }
            return {"error": "User not found"}

        if tool_name == "find_user_by_name_zip":
            first = args.get("first_name", "")
            last = args.get("last_name", "")
            zip_code = args.get("zip", "")
            if (first.lower() == task.user.first_name.lower() and
                last.lower() == task.user.last_name.lower() and
                zip_code == task.user.zip_code):
                state.authenticated = True
                state.auth_method_used = "name_zip"
                state.discovered_user_id = task.user.user_id
                # Return user info including their order IDs for easy lookup
                return {
                    "success": True,
                    "user_id": task.user.user_id,
                    "name": f"{task.user.first_name} {task.user.last_name}",
                    "orders": [{"order_id": o.order_id, "status": o.status.value} for o in task.orders],
                }
            return {"error": "User not found"}

        # Check authentication for all other tools
        if not state.authenticated:
            return {"error": "User must be authenticated first"}

        # Lookup tools
        if tool_name == "get_user_details":
            user_id = args.get("user_id", "")
            if user_id == task.user.user_id:
                return {
                    "user_id": task.user.user_id,
                    "name": f"{task.user.first_name} {task.user.last_name}",
                    "email": task.user.email,
                    "orders": [
                        {"order_id": o.order_id, "status": o.status.value, "item_count": len(o.items)}
                        for o in task.orders
                    ],
                    "payment_methods": [
                        {"id": pm.payment_id, "type": pm.method_type, "last_four": pm.last_four}
                        for pm in task.user.payment_methods
                    ],
                    "address": task.user.address,
                }
            return {"error": "User not found"}

        if tool_name == "get_order_details":
            order_id = args.get("order_id", "")
            # Normalize order ID
            if not order_id.startswith("#"):
                order_id = "#" + order_id

            order = next((o for o in task.orders if o.order_id == order_id), None)
            if order:
                state.discovered_orders.add(order_id)
                return {
                    "order_id": order.order_id,
                    "status": order.status.value,
                    "items": [
                        {
                            "item_id": item.item_id,
                            "product_id": item.product_id,
                            "variant_id": item.variant_id,
                            "name": item.name,
                            "price": item.price,
                            "quantity": item.quantity,
                        }
                        for item in order.items
                    ],
                    "total": order.total,
                    "payment_method_id": order.payment_method_id,
                }
            return {"error": f"Order {order_id} not found"}

        if tool_name == "get_product_details":
            product_id = args.get("product_id", "")
            product = next((p for p in task.products if p.product_id == product_id), None)
            if product:
                state.discovered_products.add(product_id)
                return {
                    "product_id": product.product_id,
                    "name": product.name,
                    "category": product.category,
                    "variants": [
                        {
                            "variant_id": v.variant_id,
                            "name": v.name,
                            "attributes": v.attributes,
                            "price": v.price,
                            "in_stock": v.in_stock,
                        }
                        for v in product.variants
                    ],
                }
            return {"error": f"Product {product_id} not found"}

        # Action tools (require confirmation)
        if tool_name in ["cancel_order", "modify_items", "exchange_items", "return_items"]:
            return self._execute_action_tool(tool_name, args)

        # Communication tools
        if tool_name == "respond_to_user":
            return {"success": True, "message_sent": True}

        if tool_name == "request_confirmation":
            state.pending_confirmation = {
                "action_summary": args.get("action_summary", ""),
                "timestamp": state.current_step,
            }
            return {"success": True, "awaiting_confirmation": True}

        return {"error": f"Unknown tool: {tool_name}"}

    def _execute_action_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a state-changing action tool."""
        state = self._state
        task = state.task

        order_id = args.get("order_id", "")
        if not order_id.startswith("#"):
            order_id = "#" + order_id

        order = next((o for o in task.orders if o.order_id == order_id), None)
        if not order:
            return {"error": f"Order {order_id} not found"}

        # Check order status constraints
        if tool_name == "cancel_order":
            if order.status != OrderStatus.PENDING:
                return {"error": f"Cannot cancel order with status '{order.status.value}'. Only PENDING orders can be cancelled."}
            # Execute cancellation
            order.status = OrderStatus.CANCELLED
            return {"success": True, "order_cancelled": order_id}

        if tool_name == "modify_items":
            if order.status not in [OrderStatus.PENDING, OrderStatus.PENDING_MODIFIED]:
                return {"error": f"Cannot modify order with status '{order.status.value}'. Only PENDING orders can be modified."}
            # Check one-shot constraint
            if order_id in state.one_shot_actions_used["modify"]:
                return {"error": f"Order {order_id} has already been modified. Modify can only be called once per order!"}

            state.one_shot_actions_used["modify"].add(order_id)
            order.status = OrderStatus.PENDING_MODIFIED
            order.items_modified = True
            return {"success": True, "items_modified": args.get("item_ids", [])}

        if tool_name == "exchange_items":
            if order.status != OrderStatus.DELIVERED:
                return {"error": f"Cannot exchange items in order with status '{order.status.value}'. Only DELIVERED orders can have items exchanged."}
            # Check one-shot constraint
            if order_id in state.one_shot_actions_used["exchange"]:
                return {"error": f"Order {order_id} has already had items exchanged. Exchange can only be called once per order!"}

            state.one_shot_actions_used["exchange"].add(order_id)
            return {"success": True, "items_exchanged": args.get("item_ids", [])}

        if tool_name == "return_items":
            if order.status != OrderStatus.DELIVERED:
                return {"error": f"Cannot return items from order with status '{order.status.value}'. Only DELIVERED orders can have items returned."}
            # Check one-shot constraint
            if order_id in state.one_shot_actions_used["return"]:
                return {"error": f"Order {order_id} has already had items returned. Return can only be called once per order!"}

            state.one_shot_actions_used["return"].add(order_id)
            return {"success": True, "items_returned": args.get("item_ids", [])}

        return {"error": f"Unknown action: {tool_name}"}

    def _check_task_completion(self) -> bool:
        """Check if task has been completed successfully."""
        state = self._state
        task = state.task

        # Must be authenticated
        if not state.authenticated:
            return False

        # Check all expected actions were taken
        for expected in task.expected_actions:
            if expected is None:
                continue
            expected_name = expected.get("name")
            expected_params = expected.get("params", {})

            # Find matching tool call
            found = False
            for call in state.tool_calls:
                if call["name"] == expected_name:
                    # Check key parameters match
                    call_args = call.get("arguments", {})
                    match = True
                    for key, value in expected_params.items():
                        if key == "item_ids":
                            # For batched items, check all required items are included
                            call_items = set(call_args.get("item_ids", []))
                            expected_items = set(value) if isinstance(value, list) else {value}
                            if not expected_items.issubset(call_items):
                                match = False
                        elif key == "contains":
                            # Special check: value should be contained in the message
                            message = call_args.get("message", "")
                            if value.lower() not in message.lower():
                                match = False
                        elif key not in call_args:
                            match = False
                        elif call_args[key] != value:
                            match = False
                    if match:
                        found = True
                        break

            if not found:
                return False

        return True

    def _finalize(self, success: bool, reason: str):
        """Finalize the episode."""
        self.done = True
        self._state.done = True
        self._state.success = success
        self._state.failure_reason = None if success else reason

        # BINARY REWARD: 1.0 for success, 0.0 for failure
        self.rewards = {0: 1.0 if success else 0.0}

        # Update curriculum learning state
        self._update_curriculum(success)

    def get_summary(self) -> Dict[str, Any]:
        """Get episode summary for logging."""
        state = self._state
        task = state.task

        return {
            "complexity": task.complexity.name,
            "personality": task.personality.value,
            "authenticated": state.authenticated,
            "auth_method": state.auth_method_used,
            "steps": state.current_step,
            "success": state.success,
            "failure_reason": state.failure_reason,
            "tool_calls": [tc["name"] for tc in state.tool_calls],
            "one_shot_used": {k: list(v) for k, v in state.one_shot_actions_used.items()},
            "reward": self.rewards.get(0, 0.0),
        }


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a customer service agent with access to order and user management tools.

CRITICAL PROTOCOL - FOLLOW EXACTLY:

1. AUTHENTICATION (MANDATORY - DO FIRST):
   - You MUST verify user identity before ANY other action
   - Use find_user_by_email OR find_user_by_name_zip
   - Even if user gives you their ID, you must still authenticate

2. INFORMATION DISCOVERY:
   - Use get_order_details to see order status and items
   - Use get_product_details to see available variants
   - You cannot act on information you haven't discovered

3. ONE-SHOT ACTIONS (CRITICAL):
   - modify_items, exchange_items, return_items can only be called ONCE per order
   - If you need to change multiple items, collect them ALL into a single call
   - After calling one of these, you CANNOT call it again on the same order

4. STATUS CONSTRAINTS:
   - cancel_order: only works on PENDING orders
   - modify_items: only works on PENDING orders
   - exchange_items: only works on DELIVERED orders
   - return_items: only works on DELIVERED orders

5. CONFIRMATION PROTOCOL:
   - Before any state-changing action, use request_confirmation
   - Summarize exactly what you will do
   - Wait for user to say "yes" before proceeding

6. CONDITIONAL LOGIC:
   - Users often have preferences with fallbacks
   - Check availability BEFORE committing to an action
   - If primary preference unavailable, use fallback

Respond with a single JSON tool call: {"name": "tool_name", "arguments": {...}}
"""


# =============================================================================
# HELPER FUNCTIONS FOR GAME REGISTRY
# =============================================================================

def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action for game registry."""
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
                            return json.dumps(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass
    return None


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Progressive Service Agent Environment")
    print("=" * 60)

    env = ProgressiveServiceAgentEnv()

    # Test each complexity level
    for complexity in TaskComplexity:
        print(f"\n--- Testing {complexity.name} ---")

        # Run a few seeds
        successes = 0
        for seed in range(10):
            env.fixed_complexity = complexity
            env.reset(seed)

            task = env._state.task
            print(f"  Seed {seed}: {task.personality.value} user")
            print(f"    Task: {task.task_instructions[:80]}...")
            print(f"    Expected: {[a['name'] if a else None for a in task.expected_actions]}")

            # Show first observation
            if seed == 0:
                obs = env.observe(0)
                print(f"    Observation preview:\n{obs[:500]}...")

    print("\n" + "=" * 60)
    print("Environment initialized successfully!")
