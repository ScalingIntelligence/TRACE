"""Tau Bench Tool-Calling Microenvironment.

Simplified tau bench environment for training tool-calling competence.
Uses the EXACT same tools, DB, policy, and user LLM as tau2-bench.
Differs only in scenario complexity (single-action, clearer intent).

Implements the GameEnv protocol for GRPO/PPO training.
"""

from .game import TauToolCallingEnv, extract_action, SYSTEM_PROMPT

__all__ = ["TauToolCallingEnv", "extract_action", "SYSTEM_PROMPT"]
