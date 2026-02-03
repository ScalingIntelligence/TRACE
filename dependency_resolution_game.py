"""
Conversation Termination Game - Trains agents to know when to stop.

Targets the tau-bench failure mode where agents:
1. Don't recognize when they have enough information
2. Get stuck in conversation loops (responding to "thanks" endlessly)
3. Don't recognize when conversation is complete

Key design principles:
- Formula is HIDDEN - model sees natural language request only
- Sufficiency is IMPLICIT - tracked internally, model must reason
- Answer structure VARIES by seed - prevents pattern learning
- User responses are SEMANTIC - can't keyword match satisfaction
- Ground truth is fully PROGRAMMATIC and VERIFIABLE

Reward structure:
- +1.0 for correct answer AND correct termination timing
- -1.0 for wrong answer
- -1.0 for premature termination (user had follow-up)
- Graduated penalty for looping after user satisfied
- Soft penalty for over-gathering after sufficiency
"""

import re
import random
import json
from typing import List, Optional, Dict, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


# =============================================================================
# Conversation Phase Tracking
# =============================================================================

class ConversationPhase(Enum):
    GATHERING = "gathering"           # Agent gathering info
    RESPONDED = "responded"           # Agent provided answer
    USER_SATISFIED = "user_satisfied" # User expressed satisfaction - should END
    USER_FOLLOWUP = "user_followup"   # User has follow-up - should CONTINUE
    ENDED = "ended"                   # Conversation properly ended
    LOOP = "loop"                     # Agent is looping (bad)


# =============================================================================
# Hidden Formula System (for ground truth computation)
# =============================================================================

class Expr:
    """Base class for hidden formula expressions."""

    def evaluate(self, context: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        raise NotImplementedError


@dataclass
class Literal(Expr):
    value: Union[int, float]

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.value

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        return set()


@dataclass
class FieldRef(Expr):
    record_id: str
    field_name: str

    def evaluate(self, context: Dict[str, Any]) -> Any:
        key = f"{self.record_id}.{self.field_name}"
        return context.get(key)

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        return {(self.record_id, self.field_name)}


@dataclass
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr

    def evaluate(self, context: Dict[str, Any]) -> Any:
        l = self.left.evaluate(context)
        r = self.right.evaluate(context)
        if l is None or r is None:
            return None
        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b if b != 0 else 0,
        }
        return ops.get(self.op, lambda a, b: None)(l, r)

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        return self.left.get_dependencies() | self.right.get_dependencies()


@dataclass
class Conditional(Expr):
    """IF condition THEN true_expr ELSE false_expr - creates conditional dependencies."""
    condition_field: Tuple[str, str]  # (record_id, field_name)
    threshold: int
    comparison: str  # '>=', '<', etc.
    true_expr: Expr
    false_expr: Expr

    def evaluate(self, context: Dict[str, Any]) -> Any:
        key = f"{self.condition_field[0]}.{self.condition_field[1]}"
        cond_value = context.get(key)
        if cond_value is None:
            return None

        comparisons = {
            '>=': lambda a, b: a >= b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '<': lambda a, b: a < b,
            '==': lambda a, b: a == b,
        }

        takes_true = comparisons.get(self.comparison, lambda a, b: False)(cond_value, self.threshold)

        if takes_true:
            return self.true_expr.evaluate(context)
        else:
            return self.false_expr.evaluate(context)

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        # Always need the condition field
        return {self.condition_field}

    def get_full_dependencies(self, context: Dict[str, Any]) -> Set[Tuple[str, str]]:
        """Get dependencies including the taken branch."""
        deps = {self.condition_field}

        key = f"{self.condition_field[0]}.{self.condition_field[1]}"
        cond_value = context.get(key)
        if cond_value is None:
            # Need both branches if we don't know condition yet
            return deps | self.true_expr.get_dependencies() | self.false_expr.get_dependencies()

        comparisons = {
            '>=': lambda a, b: a >= b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '<': lambda a, b: a < b,
            '==': lambda a, b: a == b,
        }

        takes_true = comparisons.get(self.comparison, lambda a, b: False)(cond_value, self.threshold)

        if takes_true:
            deps |= self.true_expr.get_dependencies()
        else:
            deps |= self.false_expr.get_dependencies()

        return deps


# =============================================================================
# Record / Database
# =============================================================================

@dataclass
class Record:
    id: str
    record_type: str
    fields: Dict[str, Any]


# =============================================================================
# Programmatic User Response Generator
# =============================================================================

class UserResponseGenerator:
    """Generates deterministic user responses based on seed."""

    # Satisfied responses - conversation should END after these
    SATISFIED_TEMPLATES = [
        "Perfect, thank you!",
        "Great, that's exactly what I needed.",
        "Thanks, that answers my question!",
        "Got it, thanks for your help!",
        "Wonderful, thank you so much!",
        "That's all I needed, thanks!",
        "Perfect, I appreciate your help!",
        "Great, thank you for checking!",
    ]

    # Follow-up responses - conversation should CONTINUE
    FOLLOWUP_TEMPLATES = [
        "Thanks! But can you also tell me about the shipping charges?",
        "Got it. Actually, can you also check if my discount was applied?",
        "Thanks! One more thing - what's my membership status?",
        "Great. By the way, how many items are in this order?",
        "Thanks! Also, when was this order placed?",
        "Perfect. Can you also confirm my account balance?",
    ]

    # Pleasantry loop responses - if agent doesn't end, user keeps going
    PLEASANTRY_TEMPLATES = [
        "You're welcome!",
        "Have a great day!",
        "Thanks again!",
        "Take care!",
        "Appreciate your help!",
        "No problem!",
        "Glad I could help!",
        "Anytime!",
    ]

    def __init__(self, rng: random.Random):
        self.rng = rng

    def get_satisfied_response(self) -> str:
        return self.rng.choice(self.SATISFIED_TEMPLATES)

    def get_followup_response(self) -> str:
        return self.rng.choice(self.FOLLOWUP_TEMPLATES)

    def get_pleasantry_response(self) -> str:
        return self.rng.choice(self.PLEASANTRY_TEMPLATES)


# =============================================================================
# Ground Truth Specification
# =============================================================================

@dataclass
class GroundTruth:
    """Fully programmatic, verifiable ground truth."""

    # Hidden formula and answer
    formula: Expr
    correct_answer: int
    required_fields: Set[Tuple[str, str]]

    # Conversation flow (determined at reset)
    has_followup: bool
    followup_field: Optional[str]  # What the follow-up asks about
    followup_answer: Optional[Any]  # Answer to follow-up

    # Pre-generated user responses (deterministic)
    initial_request: str
    satisfied_response: str
    followup_response: Optional[str]
    pleasantry_responses: List[str]


# =============================================================================
# Game State
# =============================================================================

@dataclass
class GameState:
    """Complete state of a game episode."""

    # Database
    records: Dict[str, Record]
    field_values: Dict[Tuple[str, str], Any]

    # Ground truth (hidden from model)
    ground_truth: GroundTruth

    # IDs
    user_id: str
    order_id: str
    user_name: str

    # Progress tracking
    acquired_info: Set[Tuple[str, str]]
    phase: ConversationPhase
    current_turn: int

    # Sufficiency tracking (hidden from model)
    sufficiency_turn: Optional[int]

    # Response tracking
    agent_responses: List[str]
    response_turns: List[int]

    # Termination tracking
    user_satisfied_turn: Optional[int]
    ended_turn: Optional[int]

    # Conversation history
    conversation: List[Dict[str, Any]]

    # Completion
    done: bool


# =============================================================================
# Task Generator
# =============================================================================

class TaskGenerator:
    """Generates tasks with variable answer structures."""

    REQUEST_TEMPLATES = [
        "Hi, I'm {name}. Can you tell me the total for my order?",
        "Hello, this is {name}. I need to know what I owe on my order.",
        "Hey, I'm {name}. Can you check my order total?",
        "Hi there, {name} here. What's the total amount for my order?",
        "Hello, my name is {name}. Could you look up my order total?",
    ]

    NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Quinn"]

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.user_gen = UserResponseGenerator(rng)

    def generate(self, complexity: int) -> Dict[str, Any]:
        """Generate a complete task specification."""

        # Generate IDs and name
        user_id = f"user_{self.rng.randint(100, 999)}"
        order_id = f"order_{self.rng.randint(100, 999)}"
        user_name = self.rng.choice(self.NAMES)

        # Generate database
        records, field_values = self._generate_database(user_id, order_id, user_name)

        # Generate formula and answer (varies by seed)
        formula, correct_answer, required_fields = self._generate_formula(
            records, field_values, user_id, order_id, complexity
        )

        # Determine conversation flow
        has_followup = self.rng.random() < 0.3  # 30% have follow-up

        if has_followup:
            followup_field, followup_answer = self._generate_followup(records, field_values, user_id, order_id)
            followup_response = self.user_gen.get_followup_response()
        else:
            followup_field = None
            followup_answer = None
            followup_response = None

        # Generate user messages
        initial_request = self.rng.choice(self.REQUEST_TEMPLATES).format(name=user_name)
        satisfied_response = self.user_gen.get_satisfied_response()
        pleasantry_responses = [self.user_gen.get_pleasantry_response() for _ in range(5)]

        ground_truth = GroundTruth(
            formula=formula,
            correct_answer=correct_answer,
            required_fields=required_fields,
            has_followup=has_followup,
            followup_field=followup_field,
            followup_answer=followup_answer,
            initial_request=initial_request,
            satisfied_response=satisfied_response,
            followup_response=followup_response,
            pleasantry_responses=pleasantry_responses,
        )

        return {
            'records': records,
            'field_values': field_values,
            'ground_truth': ground_truth,
            'user_id': user_id,
            'order_id': order_id,
            'user_name': user_name,
        }

    def _generate_database(
        self, user_id: str, order_id: str, user_name: str
    ) -> Tuple[Dict[str, Record], Dict[Tuple[str, str], Any]]:
        """Generate database with records."""
        records = {}
        field_values = {}
        rng = self.rng

        # User record
        membership_level = rng.randint(1, 5)
        user_fields = {
            "name": user_name,
            "email": f"{user_name.lower()}@example.com",
            "membership_level": membership_level,
            "store_credit": rng.randint(0, 50),
            "points": rng.randint(0, 5000),
            "status": rng.choice(["active", "pending"]),
            "created_date": f"2023-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        }
        records[user_id] = Record(user_id, "user", user_fields)
        for fn, val in user_fields.items():
            field_values[(user_id, fn)] = val

        # Order record
        subtotal = rng.randint(100, 500)
        tax = rng.randint(10, 50)
        discount = rng.randint(0, 30)
        shipping = rng.randint(0, 25)

        order_fields = {
            "user_ref": user_id,
            "subtotal": subtotal,
            "tax": tax,
            "discount": discount,
            "shipping": shipping,
            "items_count": rng.randint(1, 5),
            "status": rng.choice(["pending", "processing", "shipped"]),
            "created_date": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        }

        # Sometimes include total field, sometimes not (varies by seed)
        include_total = rng.random() < 0.4  # 40% have direct total
        if include_total:
            # Sometimes total is correct, sometimes stale/wrong
            total_is_correct = rng.random() < 0.7  # 70% correct when exists
            if total_is_correct:
                order_fields["total"] = subtotal + tax + shipping - discount
            else:
                # Stale/wrong total - model should compute from components
                order_fields["total"] = subtotal + tax + shipping - discount + rng.randint(10, 50)

        records[order_id] = Record(order_id, "order", order_fields)
        for fn, val in order_fields.items():
            field_values[(order_id, fn)] = val

        # Discount rates record (sometimes needed for membership discounts)
        rates_id = "discount_rates"
        discount_rates = {
            "level_1": 0,
            "level_2": 5,
            "level_3": 10,
            "level_4": 15,
            "level_5": 20,
        }
        records[rates_id] = Record(rates_id, "rates", discount_rates)
        for fn, val in discount_rates.items():
            field_values[(rates_id, fn)] = val

        # Distractor records
        self._add_distractors(records, field_values, order_id, user_id, rng)

        return records, field_values

    def _add_distractors(
        self, records: Dict, field_values: Dict,
        order_id: str, user_id: str, rng: random.Random
    ):
        """Add distractor records with tempting but wrong values."""

        # Pricing config - has tempting field names
        pricing_id = f"pricing_{rng.randint(100, 999)}"
        records[pricing_id] = Record(pricing_id, "config", {
            "total_amount": rng.randint(200, 600),  # Tempting but wrong!
            "final_cost": rng.randint(200, 600),    # Very tempting name!
            "base_rate": rng.randint(10, 50),
            "order_ref": order_id,  # References order to seem relevant
        })

        # Shipping config
        shipping_id = f"shipping_{rng.randint(100, 999)}"
        records[shipping_id] = Record(shipping_id, "shipping", {
            "total_cost": rng.randint(50, 150),  # Tempting!
            "carrier": rng.choice(["FedEx", "UPS", "USPS"]),
            "estimated_days": rng.randint(2, 7),
        })

        # Audit log - seems important but isn't
        audit_id = f"audit_{rng.randint(100, 999)}"
        records[audit_id] = Record(audit_id, "audit", {
            "last_total": rng.randint(200, 600),  # Old value, wrong!
            "verified": rng.choice([True, False]),
            "order_ref": order_id,
        })

        # Add items
        for i in range(rng.randint(2, 4)):
            item_id = f"item_{rng.randint(100, 999)}"
            records[item_id] = Record(item_id, "item", {
                "name": rng.choice(["Widget", "Gadget", "Tool", "Device"]),
                "price": rng.randint(20, 100),
                "quantity": rng.randint(1, 3),
                "subtotal": rng.randint(50, 200),  # Item subtotal, not order total!
            })

    def _generate_formula(
        self, records: Dict, field_values: Dict,
        user_id: str, order_id: str, complexity: int
    ) -> Tuple[Expr, int, Set[Tuple[str, str]]]:
        """Generate hidden formula with variable structure."""
        rng = self.rng

        # Get base values
        subtotal = field_values[(order_id, "subtotal")]
        tax = field_values[(order_id, "tax")]
        discount = field_values[(order_id, "discount")]
        shipping = field_values[(order_id, "shipping")]
        store_credit = field_values[(user_id, "store_credit")]
        membership_level = field_values[(user_id, "membership_level")]

        # Determine formula type based on seed (creates unpredictability)
        formula_type = rng.randint(0, 4)

        if formula_type == 0 and (order_id, "total") in field_values:
            # Direct lookup if total exists and is correct
            total_val = field_values[(order_id, "total")]
            computed = subtotal + tax + shipping - discount
            if total_val == computed:
                formula = FieldRef(order_id, "total")
                correct_answer = total_val
                required_fields = {(order_id, "total")}
            else:
                # Total is wrong, must compute
                formula = BinaryOp('-',
                    BinaryOp('+',
                        BinaryOp('+', FieldRef(order_id, "subtotal"), FieldRef(order_id, "tax")),
                        FieldRef(order_id, "shipping")
                    ),
                    FieldRef(order_id, "discount")
                )
                correct_answer = computed
                required_fields = {
                    (order_id, "subtotal"), (order_id, "tax"),
                    (order_id, "shipping"), (order_id, "discount")
                }

        elif formula_type == 1:
            # Compute from components
            formula = BinaryOp('-',
                BinaryOp('+',
                    BinaryOp('+', FieldRef(order_id, "subtotal"), FieldRef(order_id, "tax")),
                    FieldRef(order_id, "shipping")
                ),
                FieldRef(order_id, "discount")
            )
            correct_answer = subtotal + tax + shipping - discount
            required_fields = {
                (order_id, "subtotal"), (order_id, "tax"),
                (order_id, "shipping"), (order_id, "discount")
            }

        elif formula_type == 2:
            # Include store credit deduction
            base = subtotal + tax + shipping - discount
            formula = BinaryOp('-',
                BinaryOp('-',
                    BinaryOp('+',
                        BinaryOp('+', FieldRef(order_id, "subtotal"), FieldRef(order_id, "tax")),
                        FieldRef(order_id, "shipping")
                    ),
                    FieldRef(order_id, "discount")
                ),
                FieldRef(user_id, "store_credit")
            )
            correct_answer = base - store_credit
            required_fields = {
                (order_id, "subtotal"), (order_id, "tax"),
                (order_id, "shipping"), (order_id, "discount"),
                (user_id, "store_credit")
            }

        elif formula_type == 3 and complexity >= 2:
            # Membership discount - conditional dependency
            # If membership >= 3, get 10% off subtotal
            membership_discount = int(subtotal * 0.1) if membership_level >= 3 else 0
            base = subtotal + tax + shipping - discount - membership_discount

            # Formula with conditional
            formula = Conditional(
                condition_field=(user_id, "membership_level"),
                threshold=3,
                comparison='>=',
                true_expr=BinaryOp('-',
                    BinaryOp('-',
                        BinaryOp('+',
                            BinaryOp('+', FieldRef(order_id, "subtotal"), FieldRef(order_id, "tax")),
                            FieldRef(order_id, "shipping")
                        ),
                        FieldRef(order_id, "discount")
                    ),
                    BinaryOp('*', FieldRef(order_id, "subtotal"), Literal(0.1))
                ),
                false_expr=BinaryOp('-',
                    BinaryOp('+',
                        BinaryOp('+', FieldRef(order_id, "subtotal"), FieldRef(order_id, "tax")),
                        FieldRef(order_id, "shipping")
                    ),
                    FieldRef(order_id, "discount")
                )
            )
            correct_answer = base

            # Required fields depend on membership level
            required_fields = {(user_id, "membership_level")}
            if membership_level >= 3:
                required_fields |= {
                    (order_id, "subtotal"), (order_id, "tax"),
                    (order_id, "shipping"), (order_id, "discount")
                }
            else:
                required_fields |= {
                    (order_id, "subtotal"), (order_id, "tax"),
                    (order_id, "shipping"), (order_id, "discount")
                }

        else:
            # Default: simple computation
            formula = BinaryOp('-',
                BinaryOp('+',
                    BinaryOp('+', FieldRef(order_id, "subtotal"), FieldRef(order_id, "tax")),
                    FieldRef(order_id, "shipping")
                ),
                FieldRef(order_id, "discount")
            )
            correct_answer = subtotal + tax + shipping - discount
            required_fields = {
                (order_id, "subtotal"), (order_id, "tax"),
                (order_id, "shipping"), (order_id, "discount")
            }

        return formula, int(correct_answer), required_fields

    def _generate_followup(
        self, records: Dict, field_values: Dict,
        user_id: str, order_id: str
    ) -> Tuple[str, Any]:
        """Generate a follow-up question and its answer."""
        options = [
            ("membership_level", field_values.get((user_id, "membership_level"))),
            ("store_credit", field_values.get((user_id, "store_credit"))),
            ("items_count", field_values.get((order_id, "items_count"))),
            ("shipping", field_values.get((order_id, "shipping"))),
        ]
        return self.rng.choice(options)


# =============================================================================
# Tool Definitions
# =============================================================================

TOOL_SPECS = [
    {
        "name": "search_user",
        "description": "Find a user by name",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name to search for"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_record",
        "description": "Retrieve all fields of a record by its ID",
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string", "description": "The record ID"}
            },
            "required": ["record_id"]
        }
    },
    {
        "name": "get_field",
        "description": "Retrieve a specific field from a record",
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string", "description": "The record ID"},
                "field_name": {"type": "string", "description": "The field name"}
            },
            "required": ["record_id", "field_name"]
        }
    },
    {
        "name": "list_records",
        "description": "List all record IDs of a specific type",
        "parameters": {
            "type": "object",
            "properties": {
                "record_type": {"type": "string", "description": "The type of records"}
            },
            "required": ["record_type"]
        }
    },
    {
        "name": "respond_to_user",
        "description": "Send a response message to the user",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Your response message to the user"}
            },
            "required": ["message"]
        }
    },
    {
        "name": "end_conversation",
        "description": "End the conversation when the user is satisfied and has no more questions",
        "parameters": {
            "type": "object",
            "properties": {
                "closing_message": {"type": "string", "description": "A brief closing message"}
            },
            "required": ["closing_message"]
        }
    }
]


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from model output."""

    # Try JSON format
    json_patterns = [
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})',
        r'\[\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return {"name": match.group(1), "arguments": json.loads(match.group(2))}
            except json.JSONDecodeError:
                continue

    # Try function call format
    tool_names = "search_user|get_record|get_field|list_records|respond_to_user|end_conversation"
    func_pattern = rf'\b({tool_names})\s*\(\s*(\{{[^}}]+\}})\s*\)'
    match = re.search(func_pattern, text, re.IGNORECASE)
    if match:
        try:
            return {"name": match.group(1).lower(), "arguments": json.loads(match.group(2))}
        except json.JSONDecodeError:
            pass

    # Try to extract from natural text
    for tool in ["search_user", "get_record", "get_field", "list_records",
                 "respond_to_user", "end_conversation"]:
        if tool in text.lower():
            # Look for arguments
            arg_match = re.search(rf'{tool}\s*\(\s*(\{{.*?\}})\s*\)', text, re.IGNORECASE | re.DOTALL)
            if arg_match:
                try:
                    return {"name": tool, "arguments": json.loads(arg_match.group(1))}
                except json.JSONDecodeError:
                    pass

    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action from model output for game registry compatibility."""
    tool_call = parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    return None


# =============================================================================
# Main Game Class
# =============================================================================

class DependencyResolutionGame:
    """
    Conversation Termination Game.

    Trains agents to:
    1. Gather information efficiently (no formula shown)
    2. Respond to users appropriately
    3. Know when to END vs CONTINUE a conversation
    """

    def __init__(
        self,
        min_complexity: int = 1,
        max_complexity: int = 3,
        max_steps: int = 25,
    ):
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.max_steps = max_steps

        self._rng = random.Random()
        self._state: Optional[GameState] = None
        self._task_gen: Optional[TaskGenerator] = None

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def reset(self, seed: int):
        """Reset game with new seed."""
        self._rng = random.Random(seed)
        self._task_gen = TaskGenerator(self._rng)

        # Generate task
        complexity = self._rng.randint(self.min_complexity, self.max_complexity)
        task = self._task_gen.generate(complexity)

        self._state = GameState(
            records=task['records'],
            field_values=task['field_values'],
            ground_truth=task['ground_truth'],
            user_id=task['user_id'],
            order_id=task['order_id'],
            user_name=task['user_name'],
            acquired_info=set(),
            phase=ConversationPhase.GATHERING,
            current_turn=0,
            sufficiency_turn=None,
            agent_responses=[],
            response_turns=[],
            user_satisfied_turn=None,
            ended_turn=None,
            conversation=[],
            done=False,
        )

        # Add initial user message
        self._state.conversation.append({
            "role": "user",
            "content": self._state.ground_truth.initial_request,
        })

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        """Get observation for the agent."""
        state = self._state

        lines = []
        lines.append("=" * 60)
        lines.append("CUSTOMER SERVICE TASK")
        lines.append("=" * 60)
        lines.append("")
        lines.append("You are a customer service agent. Help the user with their request.")
        lines.append("")

        # Show available data types only
        record_types = sorted(set(r.record_type for r in state.records.values()))
        lines.append(f"Available data types: {', '.join(record_types)}")
        lines.append("")

        # Show conversation
        lines.append("CONVERSATION:")
        lines.append("-" * 40)
        for msg in state.conversation:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}]: {content}")
        lines.append("-" * 40)
        lines.append("")

        # Show available actions
        lines.append("ACTIONS (use JSON format):")
        lines.append('{"name": "search_user", "arguments": {"name": "..."}}')
        lines.append('{"name": "get_record", "arguments": {"record_id": "..."}}')
        lines.append('{"name": "get_field", "arguments": {"record_id": "...", "field_name": "..."}}')
        lines.append('{"name": "list_records", "arguments": {"record_type": "..."}}')
        lines.append('{"name": "respond_to_user", "arguments": {"message": "..."}}')
        lines.append('{"name": "end_conversation", "arguments": {"closing_message": "..."}}')
        lines.append("")

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        """Return valid action formats."""
        if self.done:
            return []
        return [
            '{"name": "search_user", "arguments": {"name": "..."}}',
            '{"name": "get_record", "arguments": {"record_id": "..."}}',
            '{"name": "get_field", "arguments": {"record_id": "...", "field_name": "..."}}',
            '{"name": "list_records", "arguments": {"record_type": "..."}}',
            '{"name": "respond_to_user", "arguments": {"message": "..."}}',
            '{"name": "end_conversation", "arguments": {"closing_message": "..."}}',
        ]

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return result."""
        state = self._state

        if name == "search_user":
            search_name = args.get("name", "").lower()
            for rid, rec in state.records.items():
                if rec.record_type == "user":
                    rec_name = str(rec.fields.get("name", "")).lower()
                    if search_name in rec_name or rec_name in search_name:
                        # Track that we found the user
                        return json.dumps({"user_id": rid, "name": rec.fields.get("name")})
            return json.dumps({"error": f"User '{search_name}' not found"})

        elif name == "get_record":
            rid = args.get("record_id", "")
            if rid in state.records:
                rec = state.records[rid]
                # Track acquired info
                for fn in rec.fields:
                    state.acquired_info.add((rid, fn))
                self._check_sufficiency()
                return json.dumps({"id": rec.id, "type": rec.record_type, **rec.fields}, indent=2)
            return json.dumps({"error": f"Record '{rid}' not found"})

        elif name == "get_field":
            rid = args.get("record_id", "")
            fn = args.get("field_name", "")
            if rid in state.records:
                rec = state.records[rid]
                if fn in rec.fields:
                    state.acquired_info.add((rid, fn))
                    self._check_sufficiency()
                    return json.dumps({fn: rec.fields[fn]})
                return json.dumps({"error": f"Field '{fn}' not found"})
            return json.dumps({"error": f"Record '{rid}' not found"})

        elif name == "list_records":
            rtype = args.get("record_type", "")
            matching = [r.id for r in state.records.values() if r.record_type == rtype]
            return json.dumps({"records": matching})

        elif name == "respond_to_user":
            message = args.get("message", "")
            state.agent_responses.append(message)
            state.response_turns.append(state.current_turn)

            # Generate user reaction
            user_response = self._generate_user_response(len(state.agent_responses))

            # Add to conversation
            state.conversation.append({"role": "assistant", "content": message})
            state.conversation.append({"role": "user", "content": user_response})

            return json.dumps({"sent": True, "user_response": user_response})

        elif name == "end_conversation":
            closing = args.get("closing_message", "Goodbye!")
            state.conversation.append({"role": "assistant", "content": closing})
            state.ended_turn = state.current_turn

            # IMPORTANT: Save phase before changing it for _finalize to check
            phase_at_termination = state.phase
            state.phase = ConversationPhase.ENDED

            # Finalize with the phase at termination time
            self._finalize(phase_at_termination)
            return json.dumps({"ended": True})

        return json.dumps({"error": f"Unknown tool: {name}"})

    def _check_sufficiency(self):
        """Check if agent has gathered enough information."""
        state = self._state
        if state.sufficiency_turn is None:
            gt = state.ground_truth

            # For conditional formulas, we need to compute required fields dynamically
            if isinstance(gt.formula, Conditional):
                # Build context from acquired info
                context = {f"{r}.{f}": state.field_values.get((r, f))
                          for r, f in state.acquired_info}
                required = gt.formula.get_full_dependencies(context)
            else:
                required = gt.required_fields

            if required <= state.acquired_info:
                state.sufficiency_turn = state.current_turn

    def _generate_user_response(self, response_count: int) -> str:
        """Generate deterministic user response."""
        state = self._state
        gt = state.ground_truth

        if response_count == 1:
            # First response from agent
            if gt.has_followup:
                state.phase = ConversationPhase.USER_FOLLOWUP
                return gt.followup_response
            else:
                state.phase = ConversationPhase.USER_SATISFIED
                state.user_satisfied_turn = state.current_turn
                return gt.satisfied_response

        elif response_count == 2 and gt.has_followup:
            # Second response (after handling follow-up)
            state.phase = ConversationPhase.USER_SATISFIED
            state.user_satisfied_turn = state.current_turn
            return gt.satisfied_response

        else:
            # Loop territory - user keeps responding with pleasantries
            state.phase = ConversationPhase.LOOP
            idx = min(response_count - 2, len(gt.pleasantry_responses) - 1)
            return gt.pleasantry_responses[idx]

    def _check_answer(self, response: str) -> bool:
        """Check if response contains correct answer."""
        gt = self._state.ground_truth
        answer = gt.correct_answer

        # Check if number appears in response
        response_lower = response.lower()

        # Try exact match
        if str(answer) in response:
            return True

        # Try with common formats
        if f"${answer}" in response or f"$ {answer}" in response:
            return True

        # Try extracting numbers
        numbers = re.findall(r'\b\d+\b', response)
        return str(answer) in numbers

    def _finalize(self, phase_at_termination: Optional[ConversationPhase] = None):
        """Compute final reward.

        Simplified reward structure focused on termination skill:
        - +1.0: Correct answer + correct termination (after user satisfied)
        -  0.0: Wrong answer + correct termination
        - -0.1 * N: Looping N turns after user satisfied
        - -1.0: Premature termination (before user satisfied / during followup)

        Args:
            phase_at_termination: The conversation phase when end_conversation was called.
                                  If None, uses current phase (for max_steps timeout).
        """
        state = self._state

        self.done = True
        state.done = True

        # Check if agent ever responded
        if not state.agent_responses:
            self.rewards = {0: -1.0}
            self.invalid_player = 0
            return

        # Use the phase at termination time, not the current phase (which is ENDED)
        check_phase = phase_at_termination if phase_at_termination else state.phase

        # PREMATURE TERMINATION: Ended before user was satisfied
        # This includes: during GATHERING, during USER_FOLLOWUP, or never reaching satisfaction
        if check_phase == ConversationPhase.USER_FOLLOWUP:
            self.rewards = {0: -1.0}
            return

        if check_phase == ConversationPhase.GATHERING:
            # Ended without even responding to user
            self.rewards = {0: -1.0}
            return

        # CORRECT TERMINATION: User was satisfied at some point
        # Now check answer correctness and looping
        answer_correct = self._check_answer(state.agent_responses[0])

        # Calculate loop penalty: -0.1 per excess turn after satisfaction
        loop_penalty = 0.0
        if state.user_satisfied_turn is not None and state.ended_turn is not None:
            excess_turns = state.ended_turn - state.user_satisfied_turn - 1
            if excess_turns > 0:
                loop_penalty = -0.1 * excess_turns

        if answer_correct:
            # Correct answer + correct termination: 1.0 - 0.1*N
            reward = 1.0 + loop_penalty
        else:
            # Wrong answer + correct termination: 0.0 - 0.1*N
            reward = 0.0 + loop_penalty

        # Clamp to [-1, 1]
        self.rewards = {0: max(-1.0, min(1.0, reward))}

    def step(self, action: Optional[str]):
        """Process agent action."""
        if self.done:
            return

        state = self._state

        # Parse action
        if action is None:
            self.done = True
            state.done = True
            self.rewards = {0: -1.0}
            self.invalid_player = 0
            return

        tool_call = None
        try:
            tool_call = json.loads(action)
        except json.JSONDecodeError:
            tool_call = parse_tool_call(action)

        if tool_call is None:
            self.done = True
            state.done = True
            self.rewards = {0: -1.0}
            self.invalid_player = 0
            return

        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        # Execute tool
        result = self._execute_tool(name, args)

        # Add tool call to conversation (except for respond/end which handle their own)
        if name not in ["respond_to_user", "end_conversation"]:
            state.conversation.append({
                "role": "assistant",
                "content": f"[Tool: {name}] Result: {result[:200]}{'...' if len(result) > 200 else ''}"
            })

        # Increment turn
        state.current_turn += 1

        # Check max steps
        if state.current_turn >= self.max_steps and not self.done:
            self.done = True
            state.done = True

            # Penalize based on state
            if not state.agent_responses:
                self.rewards = {0: -1.0}  # Never responded
            elif state.user_satisfied_turn is not None:
                self.rewards = {0: -0.3}  # Had chance to end but looped
            else:
                self.rewards = {0: -0.5}  # At least tried

    def get_summary(self) -> Dict[str, Any]:
        """Get game summary for logging."""
        state = self._state
        gt = state.ground_truth

        return {
            "correct_answer": gt.correct_answer,
            "has_followup": gt.has_followup,
            "agent_responses": state.agent_responses,
            "answer_correct": self._check_answer(state.agent_responses[0]) if state.agent_responses else False,
            "required_fields": [list(x) for x in gt.required_fields],
            "acquired_fields": [list(x) for x in state.acquired_info],
            "sufficiency_turn": state.sufficiency_turn,
            "response_turns": state.response_turns,
            "user_satisfied_turn": state.user_satisfied_turn,
            "ended_turn": state.ended_turn,
            "final_phase": state.phase.value,
            "total_turns": state.current_turn,
            "reward": self.rewards.get(0, 0.0),
        }


# =============================================================================
# System Prompt for Game Registry
# =============================================================================

SYSTEM_PROMPT_DEPENDENCY_RESOLUTION = """You are a customer service agent helping users with their requests.

TOOLS (use JSON format):
- {"name": "search_user", "arguments": {"name": "..."}}
- {"name": "get_record", "arguments": {"record_id": "..."}}
- {"name": "get_field", "arguments": {"record_id": "...", "field_name": "..."}}
- {"name": "list_records", "arguments": {"record_type": "..."}}
- {"name": "respond_to_user", "arguments": {"message": "..."}}
- {"name": "end_conversation", "arguments": {"closing_message": "..."}}
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Conversation Termination Game")
    print("=" * 60)

    game = DependencyResolutionGame(min_complexity=1, max_complexity=3)

    for seed in range(5):
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        game.reset(seed)
        state = game._state
        gt = state.ground_truth

        print(f"User: {state.user_name}")
        print(f"Order: {state.order_id}")
        print(f"Correct Answer: {gt.correct_answer}")
        print(f"Required Fields: {gt.required_fields}")
        print(f"Has Followup: {gt.has_followup}")
        print(f"Total field exists: {'total' in state.records[state.order_id].fields}")
        if 'total' in state.records[state.order_id].fields:
            actual_total = state.field_values[(state.order_id, "subtotal")] + \
                          state.field_values[(state.order_id, "tax")] + \
                          state.field_values[(state.order_id, "shipping")] - \
                          state.field_values[(state.order_id, "discount")]
            stored_total = state.records[state.order_id].fields['total']
            print(f"  Stored total: {stored_total}, Computed: {actual_total}, Match: {stored_total == actual_total}")

        print(f"\nInitial observation:\n{game.observe(0)[:500]}...")

        # Simulate basic interaction
        print("\n--- Simulating interaction ---")

        # Search user
        action = f'{{"name": "search_user", "arguments": {{"name": "{state.user_name}"}}}}'
        game.step(action)
        print(f"1. Searched user: {state.user_name}")

        if not game.done:
            # Get order
            action = f'{{"name": "get_record", "arguments": {{"record_id": "{state.order_id}"}}}}'
            game.step(action)
            print(f"2. Got order record")
            print(f"   Sufficiency reached: {state.sufficiency_turn is not None}")

        print(f"\nGame summary: {game.get_summary()}")
