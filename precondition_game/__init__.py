"""Precondition Verification Microenvironment.

Tests the model's ability to check policy preconditions before executing
state-changing actions. Uses the EXACT same airline tools, DB format,
policy, and system prompt as tau2-bench.

Each scenario presents a user request that is either policy-allowed or
policy-forbidden. The model must retrieve data, check the applicable rule,
and correctly allow or refuse the action.

Implements the GameEnv protocol for GRPO/PPO training.
"""

from .game import PreconditionGame, extract_action, SYSTEM_PROMPT

__all__ = ["PreconditionGame", "extract_action", "SYSTEM_PROMPT"]
