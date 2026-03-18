"""Verification for the precondition verification microenvironment.

Reward computation isolates the precondition go/no-go decision:

REFUSE tasks (is_refusal=True):
  - Agent makes a write action → reward = 0.0 (failed to refuse)
  - Agent correctly refuses (no writes) → reward = 1.0

ALLOW tasks (is_refusal=False, has expected_actions):
  - Agent called the expected core action tool → reward = 1.0
  - Agent refused (no core action call) → reward = 0.0
  - Agent transferred → reward = 0.0

The ALLOW check uses ACTION matching (was the expected tool called on the
right entity?) NOT DB hash comparison. This prevents false negatives when
the agent correctly decides to allow but also makes extra writes (e.g.
proactive compensation).

All rewards are bounded [0, 1] for GRPO compatibility.
"""

import sys
import pathlib
from typing import Any, Dict, List, Tuple

# Ensure tau2-bench source is in path
_TAU2_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "tau2-bench" / "src")
if _TAU2_SRC not in sys.path:
    sys.path.insert(0, _TAU2_SRC)

from .scenarios import ExpectedAction, PreconditionScenario

# Write tools that modify DB state
_WRITE_TOOLS = {
    "cancel_reservation", "update_reservation_flights",
    "update_reservation_passengers", "update_reservation_baggages",
    "book_reservation", "send_certificate",
}


def check_communicate_info(
    expected_info: List[str],
    conversation: List[Dict[str, Any]],
) -> bool:
    """Check if the agent communicated all required information."""
    if not expected_info:
        return True

    assistant_text = " ".join(
        msg.get("text", "") or str(msg.get("content", ""))
        for msg in conversation
        if msg.get("role") == "assistant"
    ).lower().replace(",", "")

    for info in expected_info:
        info_clean = str(info).lower().replace(",", "")
        if info_clean not in assistant_text:
            return False

    return True


def _check_action_called(
    expected_actions: List[ExpectedAction],
    tool_executor,
) -> Tuple[bool, str]:
    """Check if the expected core action was called on the right entity.

    Uses ACTION matching with key args only (reservation_id or user_id).
    This tests the go/no-go DECISION, not execution correctness.
    """
    tool_calls = tool_executor.tool_calls if tool_executor else []

    for expected in expected_actions:
        # Find a matching tool call by name
        found = False
        for tc in tool_calls:
            if tc.get("name") != expected.name:
                continue

            # Check key arguments (reservation_id, user_id)
            tc_args = tc.get("arguments", {})
            match = True
            for key in ("reservation_id", "user_id"):
                if key in expected.arguments:
                    if str(tc_args.get(key, "")) != str(expected.arguments[key]):
                        match = False
                        break
            if match:
                found = True
                break

        if not found:
            return False, f"Expected {expected.name}({expected.arguments}) not found"

    return True, "All expected actions found"


def compute_reward(
    scenario: PreconditionScenario,
    tool_executor,
    conversation: List[Dict[str, Any]],
    transferred: bool = False,
) -> Tuple[float, str]:
    """Compute reward for precondition verification.

    Returns:
        (reward, reason_string) where reward is in [0.0, 1.0]
    """
    if scenario.is_refusal:
        return _compute_refusal_reward(scenario, tool_executor, conversation, transferred)

    if scenario.expected_actions:
        return _compute_action_reward(scenario, tool_executor, conversation, transferred)

    # Info-only (shouldn't happen in this env, but handle gracefully)
    return _compute_info_reward(scenario, conversation, transferred)


def _compute_refusal_reward(
    scenario: PreconditionScenario,
    tool_executor,
    conversation: List[Dict[str, Any]],
    transferred: bool,
) -> Tuple[float, str]:
    """Reward for REFUSE tasks: agent should NOT take write actions."""
    tool_calls = tool_executor.tool_calls if tool_executor else []
    writes = [tc for tc in tool_calls if tc.get("name") in _WRITE_TOOLS]

    if writes:
        write_names = ", ".join(tc["name"] for tc in writes)
        return 0.0, f"REFUSE_FAIL: Made write action(s) [{write_names}] on refusal task"

    # Correctly refused — no writes made
    return 1.0, "REFUSE_PASS: Correctly refused (no write actions)"


def _compute_action_reward(
    scenario: PreconditionScenario,
    tool_executor,
    conversation: List[Dict[str, Any]],
    transferred: bool,
) -> Tuple[float, str]:
    """Reward for ALLOW tasks: agent should execute expected actions.

    Uses ACTION matching (was the core tool called on the right entity?)
    instead of DB hash comparison. This prevents false negatives when
    the agent correctly allows the action but also makes extra writes
    (e.g., proactive compensation).
    """
    if transferred:
        return 0.0, "ALLOW_FAIL: Unnecessarily transferred on allow task"

    # Check if expected action was called (key args only)
    action_found, action_detail = _check_action_called(
        scenario.expected_actions, tool_executor,
    )

    if not action_found:
        # Check if agent made any writes at all
        tool_calls = tool_executor.tool_calls if tool_executor else []
        writes = [tc for tc in tool_calls if tc.get("name") in _WRITE_TOOLS]
        if not writes:
            return 0.0, "ALLOW_FAIL: Agent refused (no writes) when action was allowed"
        return 0.0, f"ALLOW_FAIL: Wrong action executed — {action_detail}"

    # Expected action was called — check communication
    comm_pass = check_communicate_info(scenario.communicate_info, conversation)
    if comm_pass:
        return 1.0, "ALLOW_PASS: Correct action taken + communication complete"
    else:
        # Action correct but missing communication — partial credit
        return 0.5, "ALLOW_PARTIAL: Correct action but missing communication"


def _compute_info_reward(
    scenario: PreconditionScenario,
    conversation: List[Dict[str, Any]],
    transferred: bool,
) -> Tuple[float, str]:
    """Reward for info-only scenarios."""
    if transferred:
        return 0.0, "INFO_FAIL: Unnecessarily transferred"

    comm_pass = check_communicate_info(scenario.communicate_info, conversation)
    if comm_pass:
        return 1.0, "INFO_PASS: Communicated requested information"

    return 0.0, "INFO_FAIL: Failed to communicate requested information"
