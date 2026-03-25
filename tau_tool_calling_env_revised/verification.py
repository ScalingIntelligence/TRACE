"""Verification for the tau bench tool-calling microenvironment.

Uses DB hash comparison + communicate check (based on tau2-bench):
  1. DB hash comparison: replay expected actions on a gold environment,
     compare DB hash with the agent's environment DB hash.
  2. Communicate check: substring matching on agent messages.

Reward tiers for action tasks:
  1.0  = DB hash match + communication complete (full success)
  0.3  = DB hash match but missing communication
  0.1  = DB mismatch but called correct expected tool names (right approach, wrong args)
  0.0  = didn't attempt expected tools, or transferred, or info-only failure

Refusal tasks:
  1.0  = correctly refused + communicated
  0.0  = refused but missing communication
 -1.0  = made write action on refusal task (penalty)

The 0.1 partial credit for "right tools, wrong DB state" gives GRPO signal
to improve argument precision, preventing the dead zone where "almost correct"
and "didn't try" both get 0.0 (identical reward → zero gradient).
"""

import sys
import pathlib
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

# Ensure tau2-bench source is in path
_TAU2_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "tau2-bench" / "src")
if _TAU2_SRC not in sys.path:
    sys.path.insert(0, _TAU2_SRC)

from tau2.utils.utils import get_dict_hash

from .scenarios import ExpectedAction, GeneratedScenario


def compute_gold_db_hash(
    domain: str,
    initial_db: Dict[str, Any],
    expected_actions: List[ExpectedAction],
) -> str:
    """Compute the expected DB hash by replaying gold actions on a fresh DB copy.

    Args:
        domain: "airline" or "retail"
        initial_db: The scenario's initial DB state (will be deep-copied).
        expected_actions: List of tool calls to replay.

    Returns:
        SHA256 hash of the expected DB state.
    """
    import copy
    from adversarial_policy_game.tools import ToolExecutor

    # Create fresh environment with clean DB copy
    db = copy.deepcopy(initial_db)
    gold_tools = ToolExecutor(domain, db)

    # Replay expected actions
    for action in expected_actions:
        try:
            gold_tools.execute(action.name, action.arguments)
        except Exception:
            pass  # If gold action fails, the hash will reflect the partial state

    # Get hash of resulting DB
    return get_dict_hash(gold_tools.db)


def compute_predicted_db_hash(tool_executor) -> str:
    """Compute DB hash from the agent's tool executor (after episode).

    Args:
        tool_executor: ToolExecutor instance from the episode.

    Returns:
        SHA256 hash of the agent's DB state.
    """
    return get_dict_hash(tool_executor.db)


def check_communicate_info(
    expected_info: List[str],
    conversation: List[Dict[str, Any]],
) -> bool:
    """Check if the agent communicated all required information.

    Uses case-insensitive substring matching on assistant messages,
    matching tau2-bench's communicate check.

    Args:
        expected_info: List of strings that must appear in agent messages.
        conversation: List of conversation messages.

    Returns:
        True if all expected_info strings found in assistant messages.
    """
    if not expected_info:
        return True

    # Collect all assistant text
    assistant_text = " ".join(
        msg.get("text", "") or str(msg.get("content", ""))
        for msg in conversation
        if msg.get("role") == "assistant"
    ).lower()

    # Remove commas for number matching
    assistant_text_clean = assistant_text.replace(",", "")

    for info in expected_info:
        info_clean = str(info).lower().replace(",", "")
        if info_clean not in assistant_text and info_clean not in assistant_text_clean:
            return False

    return True


def compute_reward(
    scenario: GeneratedScenario,
    tool_executor,
    conversation: List[Dict[str, Any]],
    transferred: bool = False,
) -> Tuple[float, str]:
    """Compute reward using tau bench verification: DB_check × COMMUNICATE_check.

    Args:
        scenario: The generated scenario with expected actions and communicate_info.
        tool_executor: The ToolExecutor from the episode (has current DB state).
        conversation: Full conversation for communicate check.
        transferred: Whether the agent transferred to human.

    Returns:
        (reward, reason_string)
    """
    # For refusal tasks: DB should NOT change, agent should communicate refusal
    if scenario.is_refusal:
        return _compute_refusal_reward(scenario, tool_executor, conversation, transferred)

    # For info-only tasks (no expected actions, not refusal)
    if not scenario.expected_actions:
        return _compute_info_reward(scenario, conversation, transferred)

    # For action tasks: compare DB hashes + communicate check
    return _compute_action_reward(scenario, tool_executor, conversation, transferred)


def _check_expected_tool_names_called(
    expected_actions: List[ExpectedAction],
    tool_executor,
) -> Tuple[bool, int, int]:
    """Check if the agent called the expected tool names (ignoring arguments).

    Returns (all_found, n_matched, n_expected).
    """
    tool_calls = tool_executor.tool_calls if tool_executor else []
    agent_tool_names = [tc.get("name") for tc in tool_calls]

    n_expected = len(expected_actions)
    # For each expected action, check if the tool name appears in agent's calls
    # Use greedy matching: each agent call can match at most one expected action
    matched = [False] * n_expected
    used_agent = [False] * len(agent_tool_names)

    for i, expected in enumerate(expected_actions):
        for j, agent_name in enumerate(agent_tool_names):
            if not used_agent[j] and agent_name == expected.name:
                matched[i] = True
                used_agent[j] = True
                break

    n_matched = sum(matched)
    return all(matched), n_matched, n_expected


def _compute_action_reward(
    scenario: GeneratedScenario,
    tool_executor,
    conversation: List[Dict[str, Any]],
    transferred: bool,
) -> Tuple[float, str]:
    """Reward for action scenarios (has expected_actions)."""
    if transferred:
        return 0.0, "Unnecessarily transferred on action task"

    # Compute gold and predicted DB hashes
    gold_hash = compute_gold_db_hash(scenario.domain, scenario.db, scenario.expected_actions)
    predicted_hash = compute_predicted_db_hash(tool_executor)

    db_pass = gold_hash == predicted_hash
    comm_pass = check_communicate_info(scenario.communicate_info, conversation)

    if db_pass and comm_pass:
        return 1.0, "Success: DB correct + communication complete"
    elif db_pass and not comm_pass:
        return 0.3, "Partial: DB correct but missing communication"

    # DB mismatch — check if the agent at least called the right tool names
    all_names_found, n_matched, n_expected = _check_expected_tool_names_called(
        scenario.expected_actions, tool_executor,
    )
    if all_names_found:
        return 0.1, f"Partial: called correct tools ({n_matched}/{n_expected}) but DB mismatch"

    if n_matched > 0:
        return 0.0, f"Failure: DB mismatch, only {n_matched}/{n_expected} expected tools called"

    return 0.0, "Failure: DB mismatch, no expected tools called"


def _compute_refusal_reward(
    scenario: GeneratedScenario,
    tool_executor,
    conversation: List[Dict[str, Any]],
    transferred: bool,
) -> Tuple[float, str]:
    """Reward for refusal scenarios (agent should NOT take write actions)."""
    # Check if agent made any write operations
    write_tools = {
        "cancel_reservation", "update_reservation_flights", "update_reservation_passengers",
        "update_reservation_baggages", "book_reservation", "send_certificate",
        "cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
        "modify_pending_order_payment",
        "return_delivered_order_items", "exchange_delivered_order_items",
        "modify_user_address", "modify_user_email",
    }

    tool_calls = tool_executor.tool_calls if tool_executor else []
    writes = [tc for tc in tool_calls if tc.get("name") in write_tools]

    if writes:
        return -1.0, f"Failure: Made write action ({writes[0]['name']}) on refusal task"

    # No writes — correct refusal. Check communication.
    comm_pass = check_communicate_info(scenario.communicate_info, conversation)

    if comm_pass:
        return 1.0, "Success: Correct refusal with explanation"

    return 0.0, "Failure: Refused correctly but missing communication"


def _compute_info_reward(
    scenario: GeneratedScenario,
    conversation: List[Dict[str, Any]],
    transferred: bool,
) -> Tuple[float, str]:
    """Reward for info-only scenarios (no write actions expected)."""
    if transferred:
        return 0.0, "Unnecessarily transferred on info task"

    comm_pass = check_communicate_info(scenario.communicate_info, conversation)

    if comm_pass:
        return 1.0, "Success: Communicated requested information"

    return 0.0, "Failed to communicate requested information"
