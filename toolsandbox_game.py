"""ToolSandbox Game: GRPO-trainable wrapper around Apple's ToolSandbox benchmark.

Wraps the 129 base ToolSandbox scenarios as a GameEnv-protocol game for training
with train_grpo_optimized.py. Uses the exact same:
  - ExecutionContext (tools, state DBs, tool tracing)
  - ExecutionEnvironment (Python REPL for tool execution)
  - Evaluation (milestone DAG matching + minefield checking)

The model plays the AGENT role. An LLM user simulator (via UserLLMClient)
plays the user role with ToolSandbox's user instruction prompt and role-flipping.
"""

import copy
import json
import re
import pathlib
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure ToolSandbox is importable
# ---------------------------------------------------------------------------
_GAME_DIR = pathlib.Path(__file__).resolve().parent
_TSB_DIR = _GAME_DIR / "evals" / "benchmarks" / "ToolSandbox"

if str(_TSB_DIR) not in sys.path:
    sys.path.insert(0, str(_TSB_DIR))

import attrs
import polars as pl

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    ScenarioCategories,
    get_current_context,
    set_current_context,
)
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.roles.execution_environment import ExecutionEnvironment

# ---------------------------------------------------------------------------
# Lazy-loaded scenario cache (loaded once, shared across episodes)
# ---------------------------------------------------------------------------
_scenarios: Optional[Dict[str, Scenario]] = None
_scenario_names: Optional[List[str]] = None
_all_scenarios: Optional[Dict[str, Scenario]] = None
_all_scenario_names: Optional[List[str]] = None


def _ensure_scenarios_loaded() -> None:
    global _scenarios, _scenario_names, _all_scenarios, _all_scenario_names
    if _scenarios is not None:
        return

    from tool_sandbox.scenarios import named_scenarios
    from tool_sandbox.common.tool_discovery import ToolBackend

    all_scen = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)

    # Base scenarios only (no distraction tools, no scrambling)
    base = {
        name: s for name, s in all_scen.items()
        if ScenarioCategories.NO_DISTRACTION_TOOLS in s.categories
    }
    _scenarios = base
    _scenario_names = sorted(base.keys())

    # All scenarios (including augmentations)
    _all_scenarios = all_scen
    _all_scenario_names = sorted(all_scen.keys())


# ---------------------------------------------------------------------------
# System prompt (matches ToolSandbox's SYSTEM->AGENT message)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Don't make assumptions about what values to plug into functions. "
    "Ask for clarification if a user request is ambiguous."
)

# ---------------------------------------------------------------------------
# Compact tool schemas cache
# ---------------------------------------------------------------------------
_compact_schemas_cache: Dict[str, List[Dict]] = {}


def _strip_descriptions(obj):
    """Recursively strip 'description' and 'title' keys from a schema dict."""
    if isinstance(obj, dict):
        return {k: _strip_descriptions(v) for k, v in obj.items()
                if k not in ("description", "title")}
    if isinstance(obj, list):
        return [_strip_descriptions(item) for item in obj]
    return obj


# =====================================================================
# Main Game Class
# =====================================================================

class ToolSandboxGame:
    """ToolSandbox game environment for GRPO training.

    Wraps ToolSandbox's 129 base scenarios with:
    - Exact same tools, state DBs, evaluation as ToolSandbox
    - LLM user simulator via UserLLMClient with role-flipping
    - Milestone/minefield reward matching ToolSandbox evaluator

    Implements the GameEnv protocol for GRPO training.
    """

    supports_structured_messages = True

    def __init__(self, max_steps: int = 30, user_client=None,
                 include_augmentations: bool = False):
        """
        Args:
            max_steps: Maximum agent actions before forced termination.
            user_client: UserLLMClient for the user simulator.
            include_augmentations: If True, include distraction/scrambled variants.
        """
        self.max_steps = max_steps
        self._user_client = user_client
        self._include_augmentations = include_augmentations

        # GameEnv protocol attributes
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        # Internal state (set in reset)
        self._scenario: Optional[Scenario] = None
        self._scenario_name: str = ""
        self._execution_context: Optional[ExecutionContext] = None
        self._exec_env: Optional[ExecutionEnvironment] = None
        self._conversation: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._agent_system_prompt: str = SYSTEM_PROMPT
        self._user_sim_system_prompt: str = ""
        self._tool_schemas: List[Dict] = []
        self._available_tool_names: set = set()
        self._terminated_normally: bool = False
        self._last_call_key: Optional[str] = None
        self._repeat_count: int = 0
        self._reason: str = ""

    def reset(self, seed: int) -> None:
        """Reset with a new seed. Picks scenario deterministically."""
        _ensure_scenarios_loaded()

        # Pick scenario pool
        if self._include_augmentations:
            pool = _all_scenarios
            names = _all_scenario_names
        else:
            pool = _scenarios
            names = _scenario_names

        # Pick scenario
        scenario_name = names[seed % len(names)]
        scenario = pool[scenario_name]

        self._scenario = scenario
        self._scenario_name = scenario_name

        # Deep copy starting_context and set as global
        # (matches scenario.play() at scenario.py:75-77)
        self._execution_context = copy.deepcopy(scenario.starting_context)
        set_current_context(self._execution_context)

        # Create ExecutionEnvironment and process system messages
        # (matches scenario.play() at scenario.py:79-96)
        self._exec_env = ExecutionEnvironment()
        sandbox_db = self._execution_context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        max_idx = self._execution_context.max_sandbox_message_index
        for msg_idx in range(max_idx + 1):
            if (sandbox_db["recipient"][msg_idx] == RoleType.EXECUTION_ENVIRONMENT
                    and sandbox_db["sender"][msg_idx] == RoleType.SYSTEM):
                self._exec_env.respond(ending_index=msg_idx)

        # Re-read sandbox DB after system message processing
        sandbox_db = self._execution_context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )

        # Extract agent system prompt from SYSTEM->AGENT message
        for i in range(len(sandbox_db)):
            if (sandbox_db["sender"][i] == RoleType.SYSTEM
                    and sandbox_db["recipient"][i] == RoleType.AGENT):
                self._agent_system_prompt = sandbox_db["content"][i]
                break

        # Extract user sim instruction from SYSTEM->USER message
        # Filter: visible_to != [USER] (skip few-shot examples)
        self._user_sim_system_prompt = ""
        for i in range(len(sandbox_db)):
            if (sandbox_db["sender"][i] == RoleType.SYSTEM
                    and sandbox_db["recipient"][i] == RoleType.USER):
                vis = sandbox_db["visible_to"][i]
                # Real instruction has visible_to=null or not [USER]-only
                if vis is None or vis.to_list() != [RoleType.USER]:
                    self._user_sim_system_prompt = sandbox_db["content"][i]
                    # Don't break — take the last matching one (the real one
                    # is after few-shot examples)

        # Append stop token instruction for UserLLMClient compatibility
        if self._user_sim_system_prompt:
            self._user_sim_system_prompt += (
                "\n\nIMPORTANT: When User B has completed the task for you, "
                "or when User B cannot complete the task after 5 tries, "
                "respond with ###STOP### to end the conversation."
            )

        # Extract initial USER->AGENT message
        # Use first_user_sandbox_message_index (skips few-shot examples)
        first_user_idx = self._execution_context.first_user_sandbox_message_index
        first_user_msg = ""
        if first_user_idx >= 0:
            row = self._execution_context.get_database(
                DatabaseNamespace.SANDBOX,
                sandbox_message_index=first_user_idx + 1,
                drop_sandbox_message_index=False,
                get_all_history_snapshots=True,
            ).filter(pl.col("sandbox_message_index") == first_user_idx)
            if len(row) > 0:
                first_user_msg = row["content"][0]

        # Compute tool schemas for this scenario
        available_tools = self._execution_context.get_available_tools(
            scrambling_allowed=False
        )
        # Filter to AGENT-visible tools only
        self._available_tool_names = set()
        agent_tools = {}
        for name, tool in available_tools.items():
            if RoleType.AGENT in getattr(tool, "visible_to", (RoleType.AGENT,)):
                agent_tools[name] = tool
                self._available_tool_names.add(name)
        self._tool_schemas = convert_to_openai_tools(agent_tools)

        # Initialize conversation
        self._conversation = [{"role": "user", "text": first_user_msg}]

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None
        self._step_count = 0
        self._last_call_key = None
        self._repeat_count = 0
        self._terminated_normally = False
        self._reason = ""

    # -----------------------------------------------------------------
    # Structured access (for training with supports_structured_messages)
    # -----------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Return system prompt WITHOUT tool schemas."""
        return self._agent_system_prompt

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool schemas for the current scenario."""
        return self._tool_schemas

    def get_tool_schemas_compact(self) -> List[Dict[str, Any]]:
        """Return compressed tool schemas (descriptions stripped)."""
        key = self._scenario_name
        if key not in _compact_schemas_cache:
            _compact_schemas_cache[key] = _strip_descriptions(self._tool_schemas)
        return _compact_schemas_cache[key]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return conversation as OpenAI chat-API-format messages.

        Format matches tau2_bench_game.py get_messages() exactly:
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
                messages.append({"role": "assistant", "content": text, "tool_calls": None})
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
    # Text-based observation (fallback for non-structured path)
    # -----------------------------------------------------------------

    def observe(self, player_id: int) -> str:
        """Return text observation for the agent."""
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
            lines.append(f"[{role}]: {text}")
        lines.extend([
            "</conversation>",
            "",
            "Respond with exactly one JSON object. Either:",
            '- Tool call: {"name": "<tool_name>", "arguments": {<args>}}',
            '- Message: {"name": "respond_to_user", "arguments": {"message": "<text>"}}',
        ])
        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    # -----------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------

    def step(self, action: Optional[str]) -> None:
        """Process agent's action."""
        if self.done:
            return

        self._step_count += 1

        # Ensure our context is the global one
        set_current_context(self._execution_context)

        # Parse action
        if action is None:
            self._finalize(0.0, "No action provided")
            self.invalid_player = 0
            return

        tool_call = _parse_tool_call(action)
        if tool_call is None:
            self._finalize(0.0, "Invalid action format")
            self.invalid_player = 0
            return

        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        # Handle respond_to_user
        if tool_name == "respond_to_user":
            message = tool_args.get("message", "")
            self._conversation.append({"role": "assistant", "text": message})

            # Add AGENT->USER message to sandbox DB
            self._add_sandbox_message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content=message,
            )

            self._last_call_key = None
            self._repeat_count = 0

            # Call user simulator
            user_response = self._get_user_response()
            if user_response is None:
                self._finalize_with_reward("User LLM failure")
                return

            # Check for stop signals
            if _is_user_stop(user_response):
                clean = _strip_stop_tokens(user_response)
                if clean:
                    self._conversation.append({"role": "user", "text": clean})
                    self._add_sandbox_message(
                        sender=RoleType.USER,
                        recipient=RoleType.AGENT,
                        content=clean,
                    )
                else:
                    self._conversation.append({"role": "user", "text": "Thank you."})
                    self._add_sandbox_message(
                        sender=RoleType.USER,
                        recipient=RoleType.AGENT,
                        content="Thank you.",
                    )
                self._end_conversation_and_finalize()
                return

            self._conversation.append({"role": "user", "text": user_response})
            self._add_sandbox_message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content=user_response,
            )

        else:
            # Regular tool call — execute against ToolSandbox environment
            tc_id = f"tool_{self._step_count:04d}"

            # Convert arguments to Python code (ToolSandbox format)
            # Use the dict directly in f-string (not json.dumps) so Python
            # booleans are True/False, not JSON true/false.
            # This matches openai_tool_call_to_python_code in message_conversion.py:89
            python_code = (
                f"{tc_id}_parameters = {tool_args}\n"
                f"{tc_id}_response = {tool_name}(**{tc_id}_parameters)\n"
                f"print(repr({tc_id}_response))"
            )

            self._conversation.append({
                "role": "tool_call",
                "text": json.dumps(tool_call),
            })

            # Add AGENT->EXEC_ENV message to sandbox DB
            self._add_sandbox_message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=python_code,
                openai_tool_call_id=tc_id,
                openai_function_name=tool_name,
            )

            # Execute via ToolSandbox's ExecutionEnvironment
            try:
                self._exec_env.respond()
            except Exception as e:
                self._conversation.append({
                    "role": "tool_result",
                    "text": f"Error: {e}",
                })
                self._finalize(0.0, f"Execution error: {e}")
                return

            # Read tool result from sandbox DB (last message)
            sandbox_db = self._execution_context.get_database(
                DatabaseNamespace.SANDBOX,
                get_all_history_snapshots=True,
            )
            tool_result_content = sandbox_db["content"][-1] or ""
            self._conversation.append({
                "role": "tool_result",
                "text": tool_result_content,
            })

            # Loop detection: 3 consecutive identical tool calls
            call_key = json.dumps(
                {"name": tool_name, "arguments": tool_args}, sort_keys=True
            )
            if call_key == self._last_call_key:
                self._repeat_count += 1
            else:
                self._last_call_key = call_key
                self._repeat_count = 1
            if self._repeat_count >= 3:
                self._finalize(0.0, f"Loop detected: {tool_name} called 3x with same args")
                return

        # Check if conversation was ended (conversation_active flag)
        sandbox_db = self._execution_context.get_database(DatabaseNamespace.SANDBOX)
        if not sandbox_db["conversation_active"][-1]:
            self._finalize_with_reward("Conversation ended")
            return

        # Check max steps
        if self._step_count >= self.max_steps:
            self._finalize(0.0, "Max steps reached")

    # -----------------------------------------------------------------
    # Sandbox DB helpers
    # -----------------------------------------------------------------

    def _add_sandbox_message(self, sender: RoleType, recipient: RoleType,
                             content: str, openai_tool_call_id: Optional[str] = None,
                             openai_function_name: Optional[str] = None) -> None:
        """Add a message to the sandbox DB via the execution context."""
        row = {
            "sender": sender,
            "recipient": recipient,
            "content": content,
        }
        if openai_tool_call_id is not None:
            row["openai_tool_call_id"] = openai_tool_call_id
        if openai_function_name is not None:
            row["openai_function_name"] = openai_function_name
        self._execution_context.add_to_database(
            namespace=DatabaseNamespace.SANDBOX,
            rows=[row],
        )

    # -----------------------------------------------------------------
    # User simulator
    # -----------------------------------------------------------------

    def _get_user_response(self) -> Optional[str]:
        """Call user LLM with role-flipped messages (ToolSandbox format).

        Role flipping (verified against OpenAIAPIUser.to_openai_messages):
          AGENT->USER messages → role="user" (from user sim perspective)
          USER->AGENT messages → role="assistant"
          tool_call/tool_result → skipped (invisible to user)
        """
        if self._user_client is None:
            return "###STOP###"

        flipped = []
        for msg in self._conversation:
            role = msg["role"]
            text = msg["text"]
            if role == "assistant":
                flipped.append({"role": "user", "content": text})
            elif role == "user":
                flipped.append({"role": "assistant", "content": text})
            # Skip tool_call and tool_result

        try:
            raw = self._user_client.generate(self._user_sim_system_prompt, flipped)
        except Exception:
            return None

        # Strip thinking tags (Qwen3)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw

    # -----------------------------------------------------------------
    # Reward computation
    # -----------------------------------------------------------------

    def _finalize_with_reward(self, reason: str) -> None:
        """Compute reward and finalize."""
        self._terminated_normally = True
        reward = self._compute_reward()
        self._finalize(reward, reason)

    def _compute_reward(self) -> float:
        """Compute reward using ToolSandbox's evaluation system.

        EvaluationResult.similarity:
        = 0 if minefield_similarity > 0 (hit a forbidden state)
        = milestone_similarity otherwise (mean across all milestones)
        """
        try:
            eval_result = self._scenario.evaluation.evaluate(
                execution_context=self._execution_context,
                max_turn_count=self._scenario.max_messages,
            )
            return eval_result.similarity
        except Exception:
            return 0.0

    def _end_conversation_and_finalize(self) -> None:
        """Programmatically end conversation and compute reward.

        Replicates end_conversation() logic from user_tools.py:
        flips conversation_active to False in sandbox DB.
        """
        sandbox_db = self._execution_context.get_database(DatabaseNamespace.SANDBOX)
        if sandbox_db["conversation_active"][-1]:
            self._execution_context.update_database(
                DatabaseNamespace.SANDBOX,
                sandbox_db.with_columns(~pl.col("conversation_active")),
            )
        self._finalize_with_reward("User ended conversation")

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Episode summary for logging."""
        return {
            "scenario": self._scenario_name,
            "steps": self._step_count,
            "reward": self.rewards.get(0, 0.0),
            "reason": self._reason,
            "terminated_normally": self._terminated_normally,
            "conversation_length": len(self._conversation),
        }


# =====================================================================
# Action parsing (matches tau2_bench_game.py)
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
                                "arguments": obj.get("arguments", obj.get("parameters", {})),
                            }
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass
    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action for game registry compatibility.

    - JSON tool call found → return it
    - Plain text (no tool call) → wrap as respond_to_user
    - Invalid → return None
    """
    tool_call = _parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    # Plain text → respond_to_user
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"</?tool_call>", "", clean).strip()
    if clean:
        return json.dumps({"name": "respond_to_user", "arguments": {"message": clean}})
    return None


# =====================================================================
# Stop signal helpers
# =====================================================================

_STOP_TOKENS = {"###STOP###"}


def _is_user_stop(text: str) -> bool:
    """Check if user response contains a stop signal."""
    return any(token in text for token in _STOP_TOKENS)


def _strip_stop_tokens(text: str) -> str:
    """Remove stop tokens from text."""
    for token in _STOP_TOKENS:
        text = text.replace(token, "")
    return text.strip()
