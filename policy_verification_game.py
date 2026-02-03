"""
Policy Verification Game - A SIMPLER environment for learning state-dependent actions.

This is a simplified version that trains ONE core skill:
- Given COMPLETE information, decide the correct action.

The model sees ALL the information upfront (no tool calling needed).
It just needs to:
1. Check if the action is allowed by policy
2. Output the correct action OR refuse

This trains the REASONING skill without the tool-calling complexity.
Once this is learned, we can add tool-calling back.
"""

import random
import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Data Models (simplified)
# =============================================================================

class OrderStatus(str, Enum):
    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A verification task with all info provided."""
    order_id: str
    order_status: OrderStatus
    requested_action: str  # What user wants: "cancel", "refund", "exchange"
    is_possible: bool
    correct_response: str  # "approve" or "deny"
    reason: str  # Why it's possible/impossible


# =============================================================================
# Policy Rules
# =============================================================================

POLICY_RULES = {
    "cancel": {
        "allowed_statuses": [OrderStatus.PENDING],
        "description": "Can only cancel PENDING orders",
    },
    "refund": {
        "allowed_statuses": [OrderStatus.DELIVERED],
        "description": "Can only refund DELIVERED orders",
    },
    "exchange": {
        "allowed_statuses": [OrderStatus.DELIVERED],
        "description": "Can only exchange items from DELIVERED orders",
    },
    "modify": {
        "allowed_statuses": [OrderStatus.PENDING],
        "description": "Can only modify PENDING orders",
    },
}


# =============================================================================
# Task Generation
# =============================================================================

def generate_task(rng: random.Random) -> Task:
    """Generate a policy verification task."""
    # Pick random action and status
    action = rng.choice(["cancel", "refund", "exchange", "modify"])
    status = rng.choice(list(OrderStatus))

    # Skip cancelled status (always impossible for all actions)
    if status == OrderStatus.CANCELLED:
        status = rng.choice([OrderStatus.PENDING, OrderStatus.SHIPPED, OrderStatus.DELIVERED])

    order_id = f"#W{rng.randint(100000, 999999)}"

    # Determine if action is allowed
    allowed_statuses = POLICY_RULES[action]["allowed_statuses"]
    is_possible = status in allowed_statuses

    correct_response = "approve" if is_possible else "deny"

    if is_possible:
        reason = f"{action} is allowed for {status.value} orders"
    else:
        reason = f"{action} requires {[s.value for s in allowed_statuses]}, but order is {status.value}"

    return Task(
        order_id=order_id,
        order_status=status,
        requested_action=action,
        is_possible=is_possible,
        correct_response=correct_response,
        reason=reason,
    )


# =============================================================================
# Game State
# =============================================================================

@dataclass
class GameState:
    task: Task
    response: Optional[str] = None
    done: bool = False


# =============================================================================
# Main Game Class
# =============================================================================

class PolicyVerificationGame:
    """
    Simplified policy verification game.

    The model sees complete information and must decide: approve or deny.

    This trains the core skill of checking state against policy rules.
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
        """Reset game with new seed."""
        self._rng = random.Random(seed)
        task = generate_task(self._rng)
        self._state = GameState(task=task)

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def observe(self, player_id: int) -> str:
        """Get observation - ALL information is provided upfront."""
        task = self._state.task

        lines = [
            "POLICY VERIFICATION TASK",
            "",
            "RULES:",
            "- cancel: Only allowed for PENDING orders",
            "- refund: Only allowed for DELIVERED orders",
            "- exchange: Only allowed for DELIVERED orders",
            "- modify: Only allowed for PENDING orders",
            "",
            "CURRENT SITUATION:",
            f"- Order ID: {task.order_id}",
            f"- Order Status: {task.order_status.value.upper()}",
            f"- Requested Action: {task.requested_action.upper()}",
            "",
            "QUESTION: Should this request be approved or denied?",
            "",
            "Respond with exactly one word: [approve] or [deny]",
        ]

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        """Return valid actions."""
        if self.done:
            return []
        return ["[approve]", "[deny]"]

    def step(self, action: Optional[str]):
        """Process agent action."""
        if self.done:
            return

        task = self._state.task

        # Parse response
        if action is None:
            self._finalize(-1.0)
            return

        action_lower = action.lower().strip()

        # Extract approve/deny from response
        if "[approve]" in action_lower or action_lower == "approve":
            response = "approve"
        elif "[deny]" in action_lower or action_lower == "deny":
            response = "deny"
        elif "approve" in action_lower and "deny" not in action_lower:
            response = "approve"
        elif "deny" in action_lower and "approve" not in action_lower:
            response = "deny"
        else:
            # Can't parse - invalid
            self._finalize(-1.0)
            self.invalid_player = 0
            return

        self._state.response = response

        # Check correctness - BINARY reward
        if response == task.correct_response:
            self._finalize(1.0)
        else:
            self._finalize(-1.0)

    def _finalize(self, reward: float):
        """End the game with reward."""
        self.done = True
        self._state.done = True
        self.rewards = {0: reward}

    def get_summary(self) -> Dict[str, Any]:
        """Get game summary for logging."""
        task = self._state.task
        return {
            "order_status": task.order_status.value,
            "requested_action": task.requested_action,
            "is_possible": task.is_possible,
            "correct_response": task.correct_response,
            "actual_response": self._state.response,
            "reward": self.rewards.get(0, 0.0),
        }


# =============================================================================
# Action Extraction
# =============================================================================

def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action from model output."""
    text_lower = text.lower()

    if "[approve]" in text_lower:
        return "[approve]"
    elif "[deny]" in text_lower:
        return "[deny]"
    elif "approve" in text_lower and "deny" not in text_lower:
        return "[approve]"
    elif "deny" in text_lower and "approve" not in text_lower:
        return "[deny]"

    return None


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT_POLICY_VERIFICATION = """You are a policy verification assistant.
Given an order status and requested action, determine if the action is allowed.

Rules:
- cancel: Only for PENDING orders
- refund: Only for DELIVERED orders
- exchange: Only for DELIVERED orders
- modify: Only for PENDING orders

Respond with exactly: [approve] or [deny]
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Policy Verification Game")
    print("=" * 50)

    game = PolicyVerificationGame()

    # Test distribution
    approve_count = 0
    deny_count = 0

    for seed in range(1000):
        game.reset(seed)
        task = game._state.task
        if task.is_possible:
            approve_count += 1
        else:
            deny_count += 1

    print(f"Distribution over 1000 tasks:")
    print(f"  Approve (possible): {approve_count} ({approve_count/10:.1f}%)")
    print(f"  Deny (impossible): {deny_count} ({deny_count/10:.1f}%)")

    # Show some examples
    print("\n" + "=" * 50)
    print("Sample tasks:")

    for seed in [0, 1, 2, 3, 4]:
        game.reset(seed)
        task = game._state.task
        print(f"\n--- Seed {seed} ---")
        print(f"Status: {task.order_status.value}, Action: {task.requested_action}")
        print(f"Correct: {task.correct_response} ({task.reason})")
        print(f"\nObservation:\n{game.observe(0)}")
