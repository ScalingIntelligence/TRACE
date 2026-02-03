"""
Dependency Resolution Game - Trains agents to know when to stop gathering information.

The agent must compute "final_cost" by discovering and resolving a dependency graph.
Each record lookup returns many fields (few needed, many distractors).
The agent must recognize when it has sufficient information and submit.

This targets the tau-bench failure mode where agents don't know when to stop.

Key features:
- Diverse formula generation (operations, conditions, structures vary by seed)
- Tool execution matches tau2-bench pattern (JSON tool calls)
- Programmatic sufficiency detection

Reward structure:
- +1.0 for correct answer submitted right after sufficiency
- +1.0 - 0.1*N for correct answer submitted N turns after sufficiency
- -1.0 for submitting before acquiring all required info
- -1.0 for wrong answer
- -1.0 for hitting max_steps without submitting
"""

import re
import random
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


# =============================================================================
# Formula AST - Enables diverse programmatic formula generation
# =============================================================================

class Expr(ABC):
    """Base class for formula expressions."""

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Evaluate the expression given a context of resolved values."""
        pass

    @abstractmethod
    def get_dependencies(self) -> Set[Tuple[str, str]]:
        """Get all (record_id, field_name) dependencies."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string."""
        pass


@dataclass
class Literal(Expr):
    """A constant value."""
    value: Union[int, float]

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.value

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        return set()

    def to_string(self) -> str:
        return str(self.value)


@dataclass
class FieldRef(Expr):
    """Reference to a field in a record."""
    record_id: str
    field_name: str

    def evaluate(self, context: Dict[str, Any]) -> Any:
        key = f"{self.record_id}.{self.field_name}"
        return context.get(key)

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        return {(self.record_id, self.field_name)}

    def to_string(self) -> str:
        return f"{self.record_id}.{self.field_name}"


@dataclass
class BinaryOp(Expr):
    """Binary operation: +, -, *, /, max, min."""
    op: str
    left: Expr
    right: Expr

    def evaluate(self, context: Dict[str, Any]) -> Any:
        l = self.left.evaluate(context)
        r = self.right.evaluate(context)
        if l is None or r is None:
            return None
        if self.op == '+':
            return l + r
        elif self.op == '-':
            return l - r
        elif self.op == '*':
            return l * r
        elif self.op == '/':
            return l / r if r != 0 else 0
        elif self.op == 'max':
            return max(l, r)
        elif self.op == 'min':
            return min(l, r)
        return None

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        return self.left.get_dependencies() | self.right.get_dependencies()

    def to_string(self) -> str:
        if self.op in ['max', 'min']:
            return f"{self.op}({self.left.to_string()}, {self.right.to_string()})"
        return f"({self.left.to_string()} {self.op} {self.right.to_string()})"


@dataclass
class Aggregation(Expr):
    """Aggregation over multiple values: sum, avg, max, min."""
    op: str
    operands: List[Expr]

    def evaluate(self, context: Dict[str, Any]) -> Any:
        values = [e.evaluate(context) for e in self.operands]
        if any(v is None for v in values):
            return None
        if self.op == 'sum':
            return sum(values)
        elif self.op == 'avg':
            return sum(values) / len(values) if values else 0
        elif self.op == 'max':
            return max(values)
        elif self.op == 'min':
            return min(values)
        elif self.op == 'product':
            result = 1
            for v in values:
                result *= v
            return result
        return None

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        deps = set()
        for e in self.operands:
            deps |= e.get_dependencies()
        return deps

    def to_string(self) -> str:
        args = ", ".join(e.to_string() for e in self.operands)
        return f"{self.op}({args})"


@dataclass
class Condition:
    """A boolean condition for conditionals."""
    left: Expr
    op: str  # >=, <=, >, <, ==, !=
    right: Expr

    def evaluate(self, context: Dict[str, Any]) -> Optional[bool]:
        l = self.left.evaluate(context)
        r = self.right.evaluate(context)
        if l is None or r is None:
            return None
        if self.op == '>=':
            return l >= r
        elif self.op == '<=':
            return l <= r
        elif self.op == '>':
            return l > r
        elif self.op == '<':
            return l < r
        elif self.op == '==':
            return l == r
        elif self.op == '!=':
            return l != r
        return None

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        return self.left.get_dependencies() | self.right.get_dependencies()

    def to_string(self) -> str:
        return f"{self.left.to_string()} {self.op} {self.right.to_string()}"


@dataclass
class Conditional(Expr):
    """IF condition THEN true_expr ELSE false_expr."""
    condition: Condition
    true_expr: Expr
    false_expr: Expr

    def evaluate(self, context: Dict[str, Any]) -> Any:
        cond_result = self.condition.evaluate(context)
        if cond_result is None:
            return None
        if cond_result:
            return self.true_expr.evaluate(context)
        else:
            return self.false_expr.evaluate(context)

    def get_dependencies(self) -> Set[Tuple[str, str]]:
        # Need condition deps to evaluate which branch
        return self.condition.get_dependencies()

    def get_branch_dependencies(self, take_true: bool) -> Set[Tuple[str, str]]:
        """Get dependencies for a specific branch."""
        if take_true:
            return self.true_expr.get_dependencies()
        else:
            return self.false_expr.get_dependencies()

    def to_string(self) -> str:
        return f"IF {self.condition.to_string()} THEN {self.true_expr.to_string()} ELSE {self.false_expr.to_string()}"


# =============================================================================
# Record and Database
# =============================================================================

@dataclass
class Record:
    """A database record with multiple fields."""
    id: str
    record_type: str
    fields: Dict[str, Any]


# =============================================================================
# Tool Definitions (tau2-bench style)
# =============================================================================

TOOL_SPECS = [
    {
        "name": "get_record",
        "description": "Retrieve all fields of a record by its ID",
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The ID of the record to retrieve"
                }
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
                "record_id": {
                    "type": "string",
                    "description": "The ID of the record"
                },
                "field_name": {
                    "type": "string",
                    "description": "The name of the field to retrieve"
                }
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
                "record_type": {
                    "type": "string",
                    "description": "The type of records to list"
                }
            },
            "required": ["record_type"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate (e.g., '(72 + 30) * 36')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "submit_answer",
        "description": "Submit the final computed answer",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "integer",
                    "description": "The computed value of final_cost"
                }
            },
            "required": ["answer"]
        }
    }
]


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from model output (supports multiple formats)."""

    # Format 1: [Tool Call] name({"arg": "value"}) - model's natural format
    tool_call_pattern = r'\[Tool Call\]\s*(\w+)\s*\(\s*(\{[^)]+\})\s*\)'
    match = re.search(tool_call_pattern, text, re.IGNORECASE)
    if match:
        try:
            name = match.group(1)
            args = json.loads(match.group(2))
            return {"name": name, "arguments": args}
        except json.JSONDecodeError:
            pass

    # Format 2: name({"arg": "value"}) - function call style (most common from model)
    func_call_pattern = r'\b(get_record|get_field|list_records|calculate|submit_answer|submit)\s*\(\s*(\{[^)]+\})\s*\)'
    match = re.search(func_call_pattern, text, re.IGNORECASE)
    if match:
        try:
            name = match.group(1).lower()
            if name == 'submit':
                name = 'submit_answer'
            args = json.loads(match.group(2))
            return {"name": name, "arguments": args}
        except json.JSONDecodeError:
            pass

    # Format 3: JSON object with name and arguments
    # {"name": "...", "arguments": {...}} or [{"name": "...", ...}]
    json_patterns = [
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})',
        r'\[\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                name = match.group(1)
                args_str = match.group(2)
                args = json.loads(args_str)
                return {"name": name, "arguments": args}
            except json.JSONDecodeError:
                continue

    # Format 4: Bracket format [tool_name: arg1, arg2]
    bracket_patterns = [
        (r'\[get_record\s*:\s*([^\],]+)\]', 'get_record', lambda m: {"record_id": m.group(1).strip().strip('"')}),
        (r'\[get_field\s*:\s*([^,\]]+)\s*,\s*([^\]]+)\]', 'get_field', lambda m: {"record_id": m.group(1).strip().strip('"'), "field_name": m.group(2).strip().strip('"')}),
        (r'\[list_records\s*:\s*([^\]]+)\]', 'list_records', lambda m: {"record_type": m.group(1).strip().strip('"')}),
        (r'\[submit_answer\s*:\s*(-?\d+)\]', 'submit_answer', lambda m: {"answer": int(m.group(1))}),
        (r'\[submit\s*:\s*(-?\d+)\]', 'submit_answer', lambda m: {"answer": int(m.group(1))}),
    ]

    for pattern, name, args_fn in bracket_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {"name": name, "arguments": args_fn(match)}

    # Format 5: Shorthand {"answer": <number>} for submit
    answer_pattern = r'\{\s*"answer"\s*:\s*(-?\d+(?:\.\d+)?)\s*\}'
    match = re.search(answer_pattern, text)
    if match:
        try:
            val = float(match.group(1))
            if val == int(val):
                val = int(val)
            return {"name": "submit_answer", "arguments": {"answer": val}}
        except ValueError:
            pass

    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action from model output for game registry compatibility."""
    tool_call = parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    return None


# =============================================================================
# Formula Generator
# =============================================================================

class FormulaGenerator:
    """Generates diverse formulas programmatically from a seed."""

    # Field name pools for variety
    FINANCIAL_FIELDS = [
        "subtotal", "base_amount", "principal", "gross_total", "net_amount",
        "list_price", "unit_cost", "base_rate", "standard_fee", "core_value"
    ]

    MODIFIER_FIELDS = [
        "tax", "fee", "surcharge", "premium", "adjustment", "markup",
        "handling", "processing", "service_charge", "admin_fee"
    ]

    DISCOUNT_FIELDS = [
        "discount", "rebate", "credit", "reduction", "savings",
        "promo_value", "loyalty_bonus", "member_discount", "bulk_discount"
    ]

    TIER_FIELDS = [
        "membership", "tier", "level", "rank", "status_code",
        "priority", "grade", "classification", "category_id"
    ]

    OPERATIONS = ['+', '-', '*']
    AGGREGATIONS = ['sum', 'product', 'max', 'min']
    COMPARISONS = ['>=', '<=', '>', '<', '==']

    def __init__(self, rng: random.Random):
        self.rng = rng
        self._field_counter = 0

    def _pick_field_name(self, pool: List[str]) -> str:
        """Pick a field name, ensuring variety."""
        return self.rng.choice(pool)

    def generate_formula(
        self,
        record_ids: Dict[str, str],  # role -> record_id mapping
        field_values: Dict[Tuple[str, str], int],  # (record_id, field) -> value
        complexity: int = 2  # 1=simple, 2=conditional, 3=nested
    ) -> Tuple[Expr, Set[Tuple[str, str]], int]:
        """
        Generate a formula, return (formula, required_dependencies, correct_answer).

        Args:
            record_ids: Maps roles like 'order', 'user' to actual record IDs
            field_values: Maps (record_id, field_name) to actual values
            complexity: How complex the formula should be

        Returns:
            (formula_expr, required_deps, correct_answer)
        """
        order_id = record_ids['order']
        user_id = record_ids['user']

        if complexity == 1:
            return self._generate_simple_formula(order_id, field_values)
        elif complexity == 2:
            return self._generate_conditional_formula(order_id, user_id, field_values)
        else:
            return self._generate_nested_formula(order_id, user_id, field_values)

    def _generate_simple_formula(
        self,
        order_id: str,
        field_values: Dict[Tuple[str, str], int]
    ) -> Tuple[Expr, Set[Tuple[str, str]], int]:
        """Generate a simple formula with 2 fields."""

        # Pick operation (simple binary op)
        op = self.rng.choice(['+', '-', '*'])

        # Pick 2 fields
        field1 = self.rng.choice(self.FINANCIAL_FIELDS)
        field2 = self.rng.choice(self.MODIFIER_FIELDS)

        formula = BinaryOp(op, FieldRef(order_id, field1), FieldRef(order_id, field2))
        deps = {(order_id, field1), (order_id, field2)}

        # Build context and evaluate
        context = {f"{rid}.{fn}": v for (rid, fn), v in field_values.items()}
        answer = formula.evaluate(context)

        return formula, deps, int(answer)

    def _generate_conditional_formula(
        self,
        order_id: str,
        user_id: str,
        field_values: Dict[Tuple[str, str], int]
    ) -> Tuple[Expr, Set[Tuple[str, str]], int]:
        """Generate a formula with one conditional (shorter)."""

        # Pick condition field and threshold
        cond_field = self.rng.choice(self.TIER_FIELDS)
        cond_op = self.rng.choice(['>', '<', '>=', '<='])
        threshold = self.rng.randint(2, 4)

        # Get actual condition value
        cond_value = field_values.get((user_id, cond_field), 0)

        # Evaluate condition
        if cond_op == '>=':
            takes_true = cond_value >= threshold
        elif cond_op == '<=':
            takes_true = cond_value <= threshold
        elif cond_op == '>':
            takes_true = cond_value > threshold
        else:  # <
            takes_true = cond_value < threshold

        # Pick operation for each branch
        true_op = self.rng.choice(['+', '-'])
        false_op = self.rng.choice(['+', '*'])

        # Pick 2 fields for true branch
        true_field1 = self.rng.choice(self.FINANCIAL_FIELDS)
        true_field2 = self.rng.choice(self.MODIFIER_FIELDS)

        # Pick 2 fields for false branch
        false_field1 = self.rng.choice(self.FINANCIAL_FIELDS)
        false_field2 = self.rng.choice(self.DISCOUNT_FIELDS)

        # Build expressions
        condition = Condition(
            FieldRef(user_id, cond_field),
            cond_op,
            Literal(threshold)
        )

        true_expr = BinaryOp(true_op, FieldRef(order_id, true_field1), FieldRef(order_id, true_field2))
        false_expr = BinaryOp(false_op, FieldRef(order_id, false_field1), FieldRef(order_id, false_field2))

        formula = Conditional(condition, true_expr, false_expr)

        # Compute required dependencies
        deps = {(user_id, cond_field)}

        if takes_true:
            deps |= {(order_id, true_field1), (order_id, true_field2)}
        else:
            deps |= {(order_id, false_field1), (order_id, false_field2)}

        # Evaluate
        context = {f"{rid}.{fn}": v for (rid, fn), v in field_values.items()}
        answer = formula.evaluate(context)

        return formula, deps, int(answer)

    def _generate_nested_formula(
        self,
        order_id: str,
        user_id: str,
        field_values: Dict[Tuple[str, str], int]
    ) -> Tuple[Expr, Set[Tuple[str, str]], int]:
        """Generate a formula with nested conditionals (simplified)."""

        # Outer condition
        outer_cond_field = self.rng.choice(self.TIER_FIELDS[:3])
        outer_op = self.rng.choice(['>=', '>'])
        outer_threshold = self.rng.randint(2, 4)

        # Inner condition (for true branch)
        inner_cond_field = self.rng.choice(self.TIER_FIELDS[3:])
        inner_op = self.rng.choice(['>=', '<='])
        inner_threshold = self.rng.randint(2, 4)

        # Get actual values
        outer_value = field_values.get((user_id, outer_cond_field), 0)
        inner_value = field_values.get((user_id, inner_cond_field), 0)

        # Evaluate conditions
        outer_true = (outer_value >= outer_threshold) if outer_op == '>=' else (outer_value > outer_threshold)
        inner_true = (inner_value >= inner_threshold) if inner_op == '>=' else (inner_value <= inner_threshold)

        # Pick fields for each branch (2 fields each for shorter formulas)
        field_a1 = self.rng.choice(self.FINANCIAL_FIELDS)
        field_a2 = self.rng.choice(self.MODIFIER_FIELDS)
        field_b1 = self.rng.choice(self.FINANCIAL_FIELDS)
        field_b2 = self.rng.choice(self.DISCOUNT_FIELDS)
        field_c = self.rng.choice(self.FINANCIAL_FIELDS)

        # Simple expressions: a+b, c-d, e*2
        expr_a = BinaryOp('+', FieldRef(order_id, field_a1), FieldRef(order_id, field_a2))
        expr_b = BinaryOp('-', FieldRef(order_id, field_b1), FieldRef(order_id, field_b2))
        expr_c = BinaryOp('*', FieldRef(order_id, field_c), Literal(2))

        # Build nested conditional
        inner_cond = Condition(FieldRef(user_id, inner_cond_field), inner_op, Literal(inner_threshold))
        inner_conditional = Conditional(inner_cond, expr_a, expr_b)

        outer_cond = Condition(FieldRef(user_id, outer_cond_field), outer_op, Literal(outer_threshold))
        formula = Conditional(outer_cond, inner_conditional, expr_c)

        # Compute dependencies based on actual path
        deps = {(user_id, outer_cond_field)}

        if outer_true:
            deps.add((user_id, inner_cond_field))
            if inner_true:
                deps |= {(order_id, field_a1), (order_id, field_a2)}
            else:
                deps |= {(order_id, field_b1), (order_id, field_b2)}
        else:
            deps.add((order_id, field_c))

        # Evaluate
        context = {f"{rid}.{fn}": v for (rid, fn), v in field_values.items()}
        answer = formula.evaluate(context)

        return formula, deps, int(answer)


# =============================================================================
# Game State
# =============================================================================

@dataclass
class GameState:
    """Tracks the state of a game episode."""
    records: Dict[str, Record]
    formula: Expr
    formula_string: str
    correct_answer: int
    required_info: Set[Tuple[str, str]]
    acquired_info: Set[Tuple[str, str]]
    sufficiency_turn: Optional[int]
    current_turn: int
    conversation: List[Dict[str, Any]]  # tau2-bench style conversation
    done: bool
    submitted_answer: Optional[int]


# =============================================================================
# Main Game Class
# =============================================================================

class DependencyResolutionGame:
    """
    Dependency Resolution Game - single player learns when to stop gathering info.

    Follows tau2-bench patterns:
    - JSON tool calls
    - Conversation-style interaction
    - Tool results as separate messages
    """

    def __init__(
        self,
        min_complexity: int = 1,
        max_complexity: int = 3,
        min_distractors: int = 3,
        max_distractors: int = 8,
        num_distractor_records: int = 4,
        max_steps: int = 15,
    ):
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.min_distractors = min_distractors
        self.max_distractors = max_distractors
        self.num_distractor_records = num_distractor_records
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

        records, formula, formula_str, required_info, answer = self._generate_task()

        self._state = GameState(
            records=records,
            formula=formula,
            formula_string=formula_str,
            correct_answer=answer,
            required_info=required_info,
            acquired_info=set(),
            sufficiency_turn=None,
            current_turn=0,
            conversation=[],
            done=False,
            submitted_answer=None,
        )

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def _generate_task(self) -> Tuple[Dict[str, Record], Expr, str, Set[Tuple[str, str]], int]:
        """Generate database, formula, and ground truth."""
        rng = self._rng
        records: Dict[str, Record] = {}
        field_values: Dict[Tuple[str, str], int] = {}

        # Generate IDs
        order_id = f"order_{rng.randint(100, 999)}"
        user_id = f"user_{rng.randint(100, 999)}"

        # Create formula generator
        gen = FormulaGenerator(rng)

        # Generate all possible field values first
        all_fields = (
            gen.FINANCIAL_FIELDS + gen.MODIFIER_FIELDS +
            gen.DISCOUNT_FIELDS + gen.TIER_FIELDS
        )

        # Order fields
        order_fields = {"user_ref": user_id}
        for fn in gen.FINANCIAL_FIELDS:
            val = rng.randint(50, 500)
            order_fields[fn] = val
            field_values[(order_id, fn)] = val
        for fn in gen.MODIFIER_FIELDS:
            val = rng.randint(5, 50)
            order_fields[fn] = val
            field_values[(order_id, fn)] = val
        for fn in gen.DISCOUNT_FIELDS:
            val = rng.randint(5, 30)
            order_fields[fn] = val
            field_values[(order_id, fn)] = val

        # Add distractors to order
        for _ in range(rng.randint(self.min_distractors, self.max_distractors)):
            fn = f"meta_{rng.randint(1000, 9999)}"
            order_fields[fn] = rng.choice([
                rng.randint(0, 100),
                f"ref_{rng.randint(100, 999)}",
                rng.choice(["pending", "active", "completed"]),
            ])

        # Add tempting cross-references
        item_ids = [f"item_{rng.randint(100, 999)}" for _ in range(rng.randint(2, 4))]
        order_fields["items"] = item_ids
        if rng.random() < 0.5:
            order_fields["coupon_ref"] = f"coupon_{rng.randint(100, 999)}"

        records[order_id] = Record(order_id, "order", order_fields)
        field_values[(order_id, "user_ref")] = user_id

        # User fields
        user_fields = {}
        for fn in gen.TIER_FIELDS:
            val = rng.randint(0, 5)
            user_fields[fn] = val
            field_values[(user_id, fn)] = val

        # Add user distractors
        user_fields["balance"] = rng.randint(0, 1000)
        user_fields["points"] = rng.randint(0, 5000)
        user_fields["region"] = rng.choice(["US", "EU", "APAC"])
        user_fields["created"] = f"2023-{rng.randint(1,12):02d}"

        records[user_id] = Record(user_id, "user", user_fields)

        # Generate distractor records (items, coupons, etc.)
        for item_id in item_ids:
            records[item_id] = Record(item_id, "item", {
                "name": rng.choice(["Widget", "Gadget", "Module", "Unit"]),
                "price": rng.randint(10, 200),
                "qty": rng.randint(1, 5),
                "supplier": f"sup_{rng.randint(100, 999)}",
                # Add tempting fields that mirror formula fields
                "base_cost": rng.randint(10, 100),
                "discount_rate": rng.randint(5, 25),
            })

        # Create distractor records that form a tempting web of references
        distractor_ids = []
        distractor_types = [
            ("pricing_rule", ["base_rate", "markup_pct", "discount_threshold", "applies_to"]),
            ("validation_config", ["required_fields", "min_amount", "max_amount", "status"]),
            ("discount_policy", ["discount_rate", "min_order", "expires", "policy_ref"]),
            ("rate_table", ["base_rate", "tier_1_rate", "tier_2_rate", "effective_date"]),
            ("audit_log", ["last_check", "validation_status", "reviewer", "notes"]),
            ("shipping_rule", ["base_fee", "weight_factor", "zone_ref", "carrier"]),
        ]

        for _ in range(self.num_distractor_records):
            dtype, field_templates = rng.choice(distractor_types)
            did = f"{dtype.split('_')[0]}_{rng.randint(100, 999)}"

            # Build fields with tempting names and cross-references
            fields = {}
            for fn in field_templates:
                if fn.endswith("_ref") or fn == "applies_to":
                    # Reference another distractor or order
                    if distractor_ids and rng.random() < 0.7:
                        fields[fn] = rng.choice(distractor_ids)
                    else:
                        fields[fn] = rng.choice([order_id, user_id])
                elif fn in ["status", "validation_status"]:
                    fields[fn] = rng.choice(["active", "pending", "verified", "requires_review"])
                elif fn in ["required_fields", "notes"]:
                    fields[fn] = rng.choice(["base_rate,discount", "subtotal,tax", "all_fields"])
                else:
                    fields[fn] = rng.randint(5, 500)

            # Add cross-reference to make it seem connected
            if rng.random() < 0.5:
                fields["order_ref"] = order_id
            if rng.random() < 0.3:
                fields["user_ref"] = user_id

            records[did] = Record(did, dtype.replace("_", ""), fields)
            distractor_ids.append(did)

        # Add tempting references FROM order to distractor records
        if distractor_ids:
            order_fields["pricing_rule_ref"] = rng.choice(distractor_ids)
            order_fields["validation_ref"] = rng.choice(distractor_ids)
            if rng.random() < 0.5:
                order_fields["requires_validation"] = rng.choice(["yes", "no", "pending"])
            if rng.random() < 0.5:
                order_fields["discount_policy_ref"] = rng.choice(distractor_ids)

        # Update the order record with new fields
        records[order_id] = Record(order_id, "order", order_fields)

        # Generate formula
        complexity = rng.randint(self.min_complexity, self.max_complexity)
        formula, required_deps, answer = gen.generate_formula(
            {"order": order_id, "user": user_id},
            field_values,
            complexity
        )

        # Create final_cost record with optional misleading verification hints
        verification_hints = [
            ("validation_required", "Check validation_config status before computing"),
            ("pre_submit_check", "Verify pricing_rule constraints are satisfied"),
            ("audit_note", "Cross-reference with audit_log for accuracy"),
            ("rate_dependency", "Ensure rate_table values are current"),
            ("policy_check", "Confirm discount_policy eligibility first"),
        ]
        # Pick 0-1 misleading hints (50% chance of having one)
        num_hints = 1 if rng.random() < 0.5 else 0
        selected_hints = rng.sample(verification_hints, num_hints) if num_hints > 0 else []

        formula_fields = {
            "description": "Compute the final cost",
            "expression": formula.to_string(),
            "order_ref": order_id,
            "version": f"{rng.randint(1, 5)}.{rng.randint(0, 9)}",
        }
        for hint_key, hint_val in selected_hints:
            formula_fields[hint_key] = hint_val

        # Sometimes add a reference to a distractor that seems important
        if distractor_ids and rng.random() < 0.3:
            formula_fields["config_ref"] = rng.choice(distractor_ids)

        records["final_cost"] = Record("final_cost", "formula", formula_fields)

        return records, formula, formula.to_string(), required_deps, answer

    def _build_system_message(self) -> str:
        """Build the system prompt."""
        tools_desc = json.dumps(TOOL_SPECS, indent=2)

        return f"""You are solving a data retrieval and computation task.

GOAL: Compute the value of "final_cost" by querying the database records.

AVAILABLE TOOLS:
{tools_desc}

To call a tool, output JSON: {{"name": "tool_name", "arguments": {{...}}}}
"""

    def observe(self, player_id: int) -> str:
        """Get observation string for the agent."""
        state = self._state

        lines = []
        lines.append("=" * 60)
        lines.append("TASK: Compute final_cost")
        lines.append("=" * 60)
        lines.append("")

        # List available records
        lines.append("AVAILABLE RECORDS:")
        for rid, rec in sorted(state.records.items()):
            lines.append(f"  - {rid} (type: {rec.record_type})")
        lines.append("")

        # Show conversation history (tool calls and results)
        if state.conversation:
            lines.append("CONVERSATION HISTORY:")
            lines.append("-" * 40)
            for msg in state.conversation:
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]
                    lines.append(f"[Tool Call] {tc['name']}({json.dumps(tc['arguments'])})")
                elif msg["role"] == "tool":
                    content = msg["content"]
                    # Allow longer tool results so model sees all fields
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    lines.append(f"[Tool Result] {content}")
            lines.append("-" * 40)
            lines.append("")

        lines.append("Enter your next action (tool call or submit):")

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        """Return valid action formats."""
        if self.done:
            return []
        return ['{"name": "get_record", "arguments": {"record_id": "..."}}',
                '{"name": "get_field", "arguments": {"record_id": "...", "field_name": "..."}}',
                '{"name": "list_records", "arguments": {"record_type": "..."}}',
                '{"name": "calculate", "arguments": {"expression": "..."}}',
                '{"name": "submit_answer", "arguments": {"answer": ...}}']

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Tuple[str, Set[Tuple[str, str]]]:
        """Execute a tool and return (result, acquired_info)."""
        state = self._state
        acquired = set()

        if name == "get_record":
            rid = args.get("record_id", "")
            if rid in state.records:
                rec = state.records[rid]
                for fn in rec.fields:
                    acquired.add((rid, fn))
                result = json.dumps({"id": rec.id, "type": rec.record_type, **rec.fields}, indent=2)
            else:
                result = json.dumps({"error": f"Record '{rid}' not found"})

        elif name == "get_field":
            rid = args.get("record_id", "")
            fn = args.get("field_name", "")
            if rid in state.records:
                rec = state.records[rid]
                if fn in rec.fields:
                    acquired.add((rid, fn))
                    result = json.dumps({fn: rec.fields[fn]})
                else:
                    result = json.dumps({"error": f"Field '{fn}' not found"})
            else:
                result = json.dumps({"error": f"Record '{rid}' not found"})

        elif name == "list_records":
            rtype = args.get("record_type", "")
            matching = [r.id for r in state.records.values() if r.record_type == rtype]
            result = json.dumps({"records": matching})

        elif name == "calculate":
            expr = args.get("expression", "")
            try:
                # Safe eval: only allow math operations
                allowed_names = {"sum": sum, "min": min, "max": max, "abs": abs}
                # Remove anything that's not digits, operators, parens, spaces, commas
                safe_expr = re.sub(r'[^0-9+\-*/().,%s ]' % ''.join(allowed_names.keys()), '', expr)
                val = eval(safe_expr, {"__builtins__": {}}, allowed_names)
                result = json.dumps({"result": val})
            except Exception as e:
                result = json.dumps({"error": f"Could not evaluate: {str(e)}"})

        elif name == "submit_answer":
            answer = args.get("answer")
            if isinstance(answer, (int, float)):
                state.submitted_answer = int(answer)
                result = json.dumps({"submitted": int(answer)})
            else:
                result = json.dumps({"error": "Invalid answer format"})
        else:
            result = json.dumps({"error": f"Unknown tool: {name}"})

        return result, acquired

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

        # Try to parse as JSON or extract tool call
        tool_call = None
        try:
            tool_call = json.loads(action)
        except json.JSONDecodeError:
            tool_call = parse_tool_call(action)

        if tool_call is None:
            # Invalid action
            self.done = True
            state.done = True
            self.rewards = {0: -1.0}
            self.invalid_player = 0
            return

        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        # Execute tool
        result, acquired = self._execute_tool(name, args)

        # Update acquired info
        state.acquired_info.update(acquired)

        # Check sufficiency
        if state.sufficiency_turn is None:
            if state.required_info <= state.acquired_info:
                state.sufficiency_turn = state.current_turn

        # Record in conversation
        state.conversation.append({
            "role": "assistant",
            "tool_calls": [{"name": name, "arguments": args}],
        })
        state.conversation.append({
            "role": "tool",
            "content": result,
        })

        # Check for submission
        if state.submitted_answer is not None:
            self._finalize()
            return

        # Increment turn
        state.current_turn += 1

        # Check max steps
        if state.current_turn >= self.max_steps:
            self.done = True
            state.done = True
            self.rewards = {0: -1.0}

    def _finalize(self):
        """Compute final reward after submission."""
        state = self._state
        self.done = True
        state.done = True

        # Submitted before sufficient info
        if state.sufficiency_turn is None:
            self.rewards = {0: -1.0}
            return

        # Wrong answer
        if state.submitted_answer != state.correct_answer:
            self.rewards = {0: -1.0}
            return

        # Correct - compute reward based on efficiency
        unnecessary = max(0, state.current_turn - state.sufficiency_turn - 1)
        reward = max(-1.0, 1.0 - 0.1 * unnecessary)
        self.rewards = {0: reward}

    def get_summary(self) -> Dict[str, Any]:
        """Get game summary for logging."""
        state = self._state
        return {
            "formula": state.formula_string,
            "correct_answer": state.correct_answer,
            "submitted_answer": state.submitted_answer,
            "required_info": [list(x) for x in state.required_info],
            "acquired_info": [list(x) for x in state.acquired_info],
            "sufficiency_turn": state.sufficiency_turn,
            "total_turns": state.current_turn,
            "reward": self.rewards.get(0, 0.0),
        }


# =============================================================================
# System Prompt for Game Registry
# =============================================================================

SYSTEM_PROMPT_DEPENDENCY_RESOLUTION = """You are solving a data retrieval and computation task.

GOAL: Compute the value of "final_cost" by querying database records.

TOOLS (use JSON format):
- {"name": "get_record", "arguments": {"record_id": "<id>"}}
- {"name": "get_field", "arguments": {"record_id": "<id>", "field_name": "<field>"}}
- {"name": "list_records", "arguments": {"record_type": "<type>"}}
- {"name": "calculate", "arguments": {"expression": "<math expression>"}}
- {"name": "submit_answer", "arguments": {"answer": <number>}}
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    game = DependencyResolutionGame(min_complexity=2, max_complexity=3)

    for seed in range(3):
        print(f"\n{'='*70}")
        print(f"GAME SEED={seed}")
        print(f"{'='*70}")

        game.reset(seed)
        state = game._state

        print(f"Formula: {state.formula_string}")
        print(f"Correct answer: {state.correct_answer}")
        print(f"Required info: {state.required_info}")
        print()
        print(game.observe(0))

        # Simulate getting final_cost
        action = '{"name": "get_record", "arguments": {"record_id": "final_cost"}}'
        print(f"\n>>> {action}")
        game.step(action)
        print(game.observe(0)[-800:])

        print(f"\nSummary: {game.get_summary()}")
