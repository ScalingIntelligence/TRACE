"""Main game class for the tau bench tool-calling microenvironment.

Single-player game (player 0 = agent) with LLM user simulator.
Uses the EXACT same tools, DB, policy, and message format as tau2-bench.
Implements the GameEnv protocol for GRPO/PPO training.
"""

import json
import re
from typing import Dict, List, Any, Optional

from .scenarios import GeneratedScenario, generate_scenario
from .verification import compute_reward

import copy

from adversarial_policy_game.tools import ToolExecutor
from adversarial_policy_game.constants import (
    AIRLINE_POLICY, RETAIL_POLICY,
    AIRLINE_TOOL_DEFS, RETAIL_TOOL_DEFS,
    AIRLINE_TOOL_SCHEMAS, RETAIL_TOOL_SCHEMAS,
)
from adversarial_policy_game.llm_user import LLMUser, UserLLMClient


# =====================================================================
# Main Game Class
# =====================================================================

class TauToolCallingEnv:
    """Tau bench tool-calling microenvironment.

    Simplified tau bench with LLM user and programmatic verification.
    Uses the EXACT same tau2-bench infrastructure:
    - Same tools (via ToolExecutor wrapping tau2-bench AirlineTools/RetailTools)
    - Same DB (via tau2-bench db.json)
    - Same policy (via tau2-bench policy.md)
    - Same message format (matches tau2-bench agent format)
    - Same verification (DB hash + communicate check)

    Differs only in task scenarios (simpler, single-action tasks).

    Implements the GameEnv protocol for GRPO training.
    """

    supports_structured_messages = True

    def __init__(
        self,
        max_steps: int = 30,
        user_client: Optional[UserLLMClient] = None,
        domain: Optional[str] = None,
    ):
        self.max_steps = max_steps
        self._user_client = user_client
        self._domain_filter = domain  # Optional: "airline", "retail", or None (both)

        # GameEnv protocol attributes
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        # Internal state
        self._scenario: Optional[GeneratedScenario] = None
        self._tools: Optional[ToolExecutor] = None
        self._llm_user: Optional[LLMUser] = None
        self._conversation: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._transferred: bool = False
        self._pending_stop: bool = False
        self._last_call_key: Optional[str] = None
        self._repeat_count: int = 0

    def reset(self, seed: int) -> None:
        """Reset with new seed. Deterministic scenario generation."""
        self._scenario = generate_scenario(seed, domain=self._domain_filter)
        self._tools = ToolExecutor(
            self._scenario.domain,
            copy.deepcopy(self._scenario.db),
        )

        # Initialize LLM user
        if self._user_client is None:
            raise ValueError(
                "TauToolCallingEnv requires a UserLLMClient. "
                "Pass user_client= when constructing the environment."
            )

        self._llm_user = LLMUser(
            self._scenario.user_system_prompt,
            self._scenario.initial_message,
            self._user_client,
        )
        initial_msg = self._llm_user.get_initial_message()

        self._conversation = [{"role": "user", "text": initial_msg}]

        self._step_count = 0
        self._transferred = False
        self._pending_stop = False
        self._last_call_key = None
        self._repeat_count = 0

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    # -----------------------------------------------------------------
    # Structured access for function-calling eval
    # -----------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Return full system prompt with policy (matches tau2-bench format)."""
        sc = self._scenario
        if sc is None:
            return ""
        policy = AIRLINE_POLICY if sc.domain == "airline" else RETAIL_POLICY
        return (
            "<instructions>\n"
            "You are a customer service agent that helps the user according to "
            "the <policy> provided below.\n"
            "In each turn you can either:\n"
            "- Send a message to the user.\n"
            "- Make a tool call.\n"
            "You cannot do both at the same time.\n"
            "\n"
            "Try to be helpful and always follow the policy. "
            "Always make sure you generate valid JSON only.\n"
            "</instructions>\n"
            "<policy>\n"
            f"{policy}\n"
            "</policy>"
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool schemas for the current domain."""
        sc = self._scenario
        if sc is None:
            return []
        if sc.domain == "airline":
            return AIRLINE_TOOL_SCHEMAS
        return RETAIL_TOOL_SCHEMAS

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return conversation as list of chat-API-format messages.

        Returns messages compatible with OpenAI chat completions API:
        - {"role": "user", "content": "..."}
        - {"role": "assistant", "content": "...", "tool_calls": null}
        - {"role": "assistant", "content": null, "tool_calls": [...]}
        - {"role": "tool", "content": "...", "tool_call_id": "..."}
        """
        messages = []
        tool_call_counter = 0

        i = 0
        while i < len(self._conversation):
            msg = self._conversation[i]
            role = msg["role"]
            text = msg["text"]

            if role == "user":
                messages.append({"role": "user", "content": text})
            elif role == "assistant":
                messages.append({
                    "role": "assistant", "content": text, "tool_calls": None,
                })
            elif role == "tool_call":
                tc = json.loads(text)
                tc_id = f"tool-{tool_call_counter:04d}"
                tool_call_counter += 1
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc_id,
                        "name": tc["name"],
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                        "type": "function",
                    }],
                })
                if (i + 1 < len(self._conversation)
                        and self._conversation[i + 1]["role"] == "tool_result"):
                    messages.append({
                        "role": "tool",
                        "content": self._conversation[i + 1]["text"],
                        "tool_call_id": tc_id,
                    })
                    i += 1
            i += 1

        return messages

    # -----------------------------------------------------------------
    # Text-based observation (GameEnv protocol)
    # -----------------------------------------------------------------

    def observe(self, player_id: int) -> str:
        """Return observation for the agent."""
        sc = self._scenario
        if sc is None:
            return "No scenario loaded."

        tool_schemas = json.dumps(self.get_tool_schemas(), indent=2)

        lines = [
            self.get_system_prompt(),
            "",
            "<available_tools>",
            tool_schemas,
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
            "Respond with exactly one JSON object. Either:",
            '- Tool call: {"name": "<tool_name>", "arguments": {<args>}}',
            '- Message: {"name": "respond_to_user", "arguments": {"message": "<text>"}}',
            '- Transfer: {"name": "transfer_to_human_agents", "arguments": {"summary": "<text>"}}',
        ])

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]) -> None:
        """Process agent's action."""
        if self.done:
            return

        self._step_count += 1

        # Parse action
        if action is None:
            self._finalize(0.0, "No action provided")
            return

        tool_call = _parse_tool_call(action)
        if tool_call is None:
            self._finalize(0.0, "Invalid action format")
            self.invalid_player = 0
            return

        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        if tool_name == "respond_to_user":
            message = tool_args.get("message", "")
            self._conversation.append({"role": "assistant", "text": message})
            self._last_call_key = None
            self._repeat_count = 0

            if self._pending_stop:
                pass  # Agent's extra turn after user ###STOP###
            else:
                visible = self._get_visible_conversation()
                user_response = self._llm_user.get_response(visible)

                if user_response is None:
                    pass
                elif "###STOP###" in user_response:
                    clean = user_response.replace("###STOP###", "").strip()
                    self._conversation.append({
                        "role": "user",
                        "text": clean or "Thank you. Goodbye.",
                    })
                    self._pending_stop = True
                    return
                elif "###TRANSFER###" in user_response:
                    clean = user_response.replace("###TRANSFER###", "").strip()
                    self._conversation.append({
                        "role": "user",
                        "text": clean or "OK, please transfer me.",
                    })
                else:
                    self._conversation.append({"role": "user", "text": user_response})

        elif tool_name == "transfer_to_human_agents":
            self._transferred = True
            self._conversation.append({
                "role": "tool_call", "text": json.dumps(tool_call),
            })
            result = self._tools.execute(tool_name, tool_args)
            self._conversation.append({"role": "tool_result", "text": result})
            self._conversation.append({
                "role": "assistant",
                "text": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
            })
            self._finalize_with_verification()
            return

        elif tool_name == "end_conversation":
            self._conversation.append({
                "role": "assistant",
                "text": tool_args.get("closing_message", "Thank you for contacting us."),
            })
            self._finalize_with_verification()
            return

        else:
            # Tool call — execute against database
            self._conversation.append({
                "role": "tool_call", "text": json.dumps(tool_call),
            })
            try:
                result = self._tools.execute(tool_name, tool_args)
            except Exception as e:
                result = f"Error: {e}"
            self._conversation.append({"role": "tool_result", "text": result})

            # Loop detection
            call_key = json.dumps(
                {"name": tool_name, "arguments": tool_args}, sort_keys=True,
            )
            if call_key == self._last_call_key:
                self._repeat_count += 1
            else:
                self._last_call_key = call_key
                self._repeat_count = 1
            if self._repeat_count >= 3:
                self._finalize(0.0, f"Loop detected: {tool_name} called 3x")
                return

        # Check pending stop
        if self._pending_stop:
            self._finalize_with_verification()
            return

        # Check max steps
        if self._step_count >= self.max_steps:
            self._finalize_with_verification()

    def _get_visible_conversation(self) -> List[Dict[str, str]]:
        """Get conversation visible to the customer (text only)."""
        return [
            msg for msg in self._conversation
            if msg["role"] in ("user", "assistant")
        ]

    def _finalize_with_verification(self) -> None:
        """End game and compute reward via DB hash + communicate check."""
        reward, reason = compute_reward(
            scenario=self._scenario,
            tool_executor=self._tools,
            conversation=self._conversation,
            transferred=self._transferred,
        )
        self._finalize(reward, reason)

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    def get_summary(self) -> Dict[str, Any]:
        """Get episode summary for debugging/analysis."""
        sc = self._scenario
        return {
            "scenario_type": sc.scenario_type if sc else "",
            "domain": sc.domain if sc else "",
            "description": sc.description if sc else "",
            "is_refusal": sc.is_refusal if sc else False,
            "steps": self._step_count,
            "reward": self.rewards.get(0, 0.0),
            "reason": getattr(self, "_reason", ""),
            "transferred": self._transferred,
            "tool_calls": self._tools.tool_calls if self._tools else [],
            "conversation_length": len(self._conversation),
            "expected_actions": [
                {"name": a.name, "arguments": a.arguments}
                for a in (sc.expected_actions if sc else [])
            ],
            "communicate_info": sc.communicate_info if sc else [],
            "key_facts": sc.key_facts if sc else {},
        }


# =====================================================================
# Action parsing (reused from adversarial_policy_game)
# =====================================================================

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
                                "arguments": obj.get(
                                    "arguments", obj.get("parameters", {}),
                                ),
                            }
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass
    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action for game registry compatibility.

    Matches tau2-bench eval behavior:
    - JSON tool call found → return it
    - Plain text → wrap as respond_to_user
    """
    tool_call = _parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    # Plain text → respond_to_user
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"</?tool_call>", "", clean).strip()
    if clean:
        return json.dumps({
            "name": "respond_to_user",
            "arguments": {"message": clean},
        })
    return None


# =====================================================================
# System prompt (static, for GameSpec — policy injected dynamically)
# =====================================================================

SYSTEM_PROMPT = (
    "<instructions>\n"
    "You are a customer service agent that helps the user according to "
    "the <policy> provided below.\n"
    "In each turn you can either:\n"
    "- Send a message to the user.\n"
    "- Make a tool call.\n"
    "You cannot do both at the same time.\n"
    "\n"
    "Try to be helpful and always follow the policy. "
    "Always make sure you generate valid JSON only.\n"
    "</instructions>"
)
