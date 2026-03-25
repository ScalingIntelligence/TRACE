"""Tau2-Bench Game: GRPO-trainable wrapper around the real tau2-bench evaluation.

Wraps the 50 airline + 114 retail tau2-bench tasks as a GameEnv-protocol game
for training with train_grpo_optimized.py. Uses the exact same:
  - Environment (tools, DB, policy)
  - User simulator prompt format (simulation_guidelines.md + scenario)
  - Reward computation (DB hash comparison + communicate_info substring match)
  - Agent system prompt (<instructions> + <policy>)

The model plays the agent role. An LLM user simulator (via UserLLMClient)
plays the customer role with tau2-bench's exact prompt format and role-flipping.
"""

import json
import re
import pathlib
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure tau2-bench src is importable
# ---------------------------------------------------------------------------
_GAME_DIR = pathlib.Path(__file__).resolve().parent
_TAU2_SRC = _GAME_DIR / "tau2-bench" / "src"
_TAU2_DATA = _GAME_DIR / "tau2-bench" / "data" / "tau2"

if str(_TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(_TAU2_SRC))

from tau2.domains.airline.environment import get_environment as _get_airline_env, get_tasks as _get_airline_tasks
from tau2.domains.retail.environment import get_environment as _get_retail_env, get_tasks as _get_retail_tasks
from tau2.data_model.tasks import Task
from tau2.data_model.message import ToolCall

# ---------------------------------------------------------------------------
# Module-level caching (loaded once, shared across episodes)
# ---------------------------------------------------------------------------

# Tasks (immutable)
_airline_tasks: List[Task] = _get_airline_tasks(task_split_name=None)
_retail_tasks: List[Task] = _get_retail_tasks(task_split_name=None)
_all_tasks: List[Task] = _airline_tasks + _retail_tasks

# Task domain lookup
_task_domain: Dict[str, str] = {}
for _t in _airline_tasks:
    _task_domain[_t.id + "_airline"] = "airline"
for _t in _retail_tasks:
    _task_domain[_t.id + "_retail"] = "retail"

# Policies (immutable strings)
_airline_policy: str = (_TAU2_DATA / "domains" / "airline" / "policy.md").read_text()
_retail_policy: str = (_TAU2_DATA / "domains" / "retail" / "policy.md").read_text()

# Tool schemas (computed once from fresh environments)
_airline_env_for_schemas = _get_airline_env()
_retail_env_for_schemas = _get_retail_env()
_airline_tool_schemas: List[Dict] = [t.openai_schema for t in _airline_env_for_schemas.get_tools()]
_retail_tool_schemas: List[Dict] = [t.openai_schema for t in _retail_env_for_schemas.get_tools()]

# Cache default DBs for fast deepcopy (avoid disk I/O per episode)
_airline_default_db = _airline_env_for_schemas.tools.db
_retail_default_db = _retail_env_for_schemas.tools.db
del _airline_env_for_schemas, _retail_env_for_schemas

# User simulator guidelines
_user_sim_guidelines: str = (_TAU2_DATA / "user_simulator" / "simulation_guidelines.md").read_text()

# Compact tool schemas cache
_compact_schemas_cache: Dict[str, List[Dict]] = {}


def _strip_descriptions(obj):
    """Recursively strip 'description' and 'title' keys from a schema dict."""
    if isinstance(obj, dict):
        return {k: _strip_descriptions(v) for k, v in obj.items() if k not in ("description", "title")}
    if isinstance(obj, list):
        return [_strip_descriptions(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# System prompt (matches tau2-bench LLMAgent and AdversarialPolicyGame)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "<instructions>\n"
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n"
    "- Send a message to the user.\n"
    "- Make a tool call.\n"
    "You cannot do both at the same time.\n"
    "\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\n"
    "</instructions>"
)


# =====================================================================
# Main Game Class
# =====================================================================

class Tau2BenchGame:
    """Tau2-bench game environment for GRPO training.

    Wraps the real tau2-bench airline/retail tasks with:
    - Exact same tools, DB, policy as tau2-bench evaluation
    - LLM user simulator with tau2-bench's prompt format
    - DB hash + communicate_info reward matching tau2-bench evaluator

    Implements the GameEnv protocol for GRPO training.
    """

    supports_structured_messages = True

    def __init__(self, max_steps: int = 30, user_client=None, domain: Optional[str] = None):
        """
        Args:
            max_steps: Maximum agent actions before forced termination.
            user_client: UserLLMClient for the user simulator.
            domain: "airline", "retail", or None (both, selected by seed).
        """
        self.max_steps = max_steps
        self._user_client = user_client
        self._domain_filter = domain

        # Build task pool based on domain filter
        if domain == "airline":
            self._task_pool = _airline_tasks
        elif domain == "retail":
            self._task_pool = _retail_tasks
        else:
            self._task_pool = _all_tasks

        # GameEnv protocol attributes
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        # Internal state (set in reset)
        self._task: Optional[Task] = None
        self._domain: Optional[str] = None
        self._environment = None
        self._conversation: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._user_system_prompt: str = ""
        self._terminated_normally: bool = False
        self._last_call_key: Optional[str] = None
        self._repeat_count: int = 0

    def reset(self, seed: int) -> None:
        """Reset with a new seed. Picks task deterministically."""
        # Pick task
        task_idx = seed % len(self._task_pool)
        self._task = self._task_pool[task_idx]

        # Determine domain
        if self._domain_filter:
            self._domain = self._domain_filter
        else:
            # Determine from task pool position
            if task_idx < len(_airline_tasks) and self._task_pool is _all_tasks:
                self._domain = "airline"
            elif self._task_pool is _all_tasks:
                self._domain = "retail"
            else:
                self._domain = self._domain_filter or "airline"

        # Create fresh environment with deepcopied DB
        self._environment = self._make_fresh_environment()

        # Build user simulator system prompt
        self._user_system_prompt = (
            f"{_user_sim_guidelines}\n\n"
            f"<scenario>\n{self._task.user_scenario}\n</scenario>"
        )

        # Generate first user message
        # In tau2-bench, the agent says "Hi! How can I help you today?" and the
        # user simulator responds. We replicate this by calling the user LLM.
        first_user_msg = self._call_user_llm(
            [{"role": "user", "content": "Hi! How can I help you today?"}]
        )

        self._conversation = [{"role": "user", "text": first_user_msg}]
        self._step_count = 0
        self._last_call_key = None
        self._repeat_count = 0
        self._terminated_normally = False

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    # -----------------------------------------------------------------
    # Structured access (for training with supports_structured_messages)
    # -----------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Return system prompt WITHOUT tool schemas (tokenizer handles tools)."""
        policy = _airline_policy if self._domain == "airline" else _retail_policy
        return (
            f"<instructions>\n"
            f"You are a customer service agent that helps the user according to the <policy> provided below.\n"
            f"In each turn you can either:\n"
            f"- Send a message to the user.\n"
            f"- Make a tool call.\n"
            f"You cannot do both at the same time.\n"
            f"\n"
            f"Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\n"
            f"</instructions>\n"
            f"<policy>\n"
            f"{policy}\n"
            f"</policy>"
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool schemas for the current domain."""
        if self._domain == "airline":
            return _airline_tool_schemas
        return _retail_tool_schemas

    def get_tool_schemas_compact(self) -> List[Dict[str, Any]]:
        """Return compressed tool schemas (descriptions stripped)."""
        domain = self._domain or "airline"
        if domain not in _compact_schemas_cache:
            _compact_schemas_cache[domain] = _strip_descriptions(self.get_tool_schemas())
        return _compact_schemas_cache[domain]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return conversation as OpenAI chat-API-format messages.

        Format matches AdversarialPolicyGame.get_messages() exactly:
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
                if i + 1 < len(self._conversation) and self._conversation[i + 1]["role"] == "tool_result":
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
            '- Transfer: {"name": "transfer_to_human_agents", "arguments": {"summary": "<text>"}}',
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

        # Handle respond_to_user (training-side abstraction for text messages)
        if tool_name == "respond_to_user":
            message = tool_args.get("message", "")
            self._conversation.append({"role": "assistant", "text": message})
            self._last_call_key = None
            self._repeat_count = 0

            # Call user simulator
            user_response = self._get_user_response()
            if user_response is None:
                # User LLM failure — finalize gracefully
                self._finalize_with_reward("User LLM failure")
                return

            # Check for stop signals
            if _is_user_stop(user_response):
                clean = _strip_stop_tokens(user_response)
                self._conversation.append({"role": "user", "text": clean or "Thank you. Goodbye."})
                self._finalize_with_reward("User stop")
                return

            self._conversation.append({"role": "user", "text": user_response})

        else:
            # Regular tool call — execute against tau2-bench environment
            self._conversation.append({"role": "tool_call", "text": json.dumps(tool_call)})

            tc = ToolCall(name=tool_name, arguments=tool_args)
            tool_msg = self._environment.get_response(tc)
            self._conversation.append({"role": "tool_result", "text": tool_msg.content})

            # Loop detection: 3 consecutive identical tool calls
            call_key = json.dumps({"name": tool_name, "arguments": tool_args}, sort_keys=True)
            if call_key == self._last_call_key:
                self._repeat_count += 1
            else:
                self._last_call_key = call_key
                self._repeat_count = 1
            if self._repeat_count >= 3:
                self._finalize(0.0, f"Loop detected: {tool_name} called 3x with same args")
                return

        # Check max steps
        if self._step_count >= self.max_steps:
            # MAX_STEPS → reward 0.0 (matches tau2-bench: non-normal termination)
            self._finalize(0.0, "Max steps reached")

    # -----------------------------------------------------------------
    # User simulator
    # -----------------------------------------------------------------

    def _get_user_response(self) -> Optional[str]:
        """Call user LLM with role-flipped messages (tau2-bench format)."""
        if self._user_client is None:
            return "###STOP###"

        # Build role-flipped messages (agent text → user role, user text → assistant role)
        # Tool calls and tool results are NOT visible to the user
        flipped = []
        for msg in self._conversation:
            role = msg["role"]
            text = msg["text"]
            if role == "assistant":
                flipped.append({"role": "user", "content": text})
            elif role == "user":
                flipped.append({"role": "assistant", "content": text})
            # Skip tool_call and tool_result (invisible to user)

        try:
            raw = self._user_client.generate(self._user_system_prompt, flipped)
        except Exception:
            return None

        # Strip thinking tags (Qwen3 thinking mode)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw

    def _call_user_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call user LLM for initial message generation."""
        if self._user_client is None:
            return "I need help with my account."

        try:
            raw = self._user_client.generate(self._user_system_prompt, messages)
        except Exception:
            return "I need help with my account."

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip stop tokens from initial message (user shouldn't stop immediately)
        if _is_user_stop(raw):
            clean = _strip_stop_tokens(raw)
            return clean or "I need help with my account."
        return raw

    # -----------------------------------------------------------------
    # Reward computation
    # -----------------------------------------------------------------

    def _finalize_with_reward(self, reason: str) -> None:
        """Compute reward (DB hash + communicate_info) and finalize."""
        self._terminated_normally = True
        reward = self._compute_reward()
        self._finalize(reward, reason)

    def _compute_reward(self) -> float:
        """Compute reward matching tau2-bench evaluator logic.

        reward = db_reward * communicate_reward

        DB reward: compare live environment DB hash vs gold environment
        (fresh env + gold actions executed). 1.0 if match, 0.0 if not.

        Communicate reward: check if all communicate_info strings appear
        as substrings in any assistant text message. 1.0 if all found, 0.0 if not.
        """
        task = self._task
        if task is None or task.evaluation_criteria is None:
            return 1.0

        # --- DB reward ---
        gold_env = self._make_fresh_environment()
        golden_actions = task.evaluation_criteria.actions or []
        for action in golden_actions:
            try:
                gold_env.make_tool_call(
                    tool_name=action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
            except Exception:
                pass

        gold_db_hash = gold_env.get_db_hash()
        predicted_db_hash = self._environment.get_db_hash()
        db_reward = 1.0 if gold_db_hash == predicted_db_hash else 0.0

        # --- Communicate reward ---
        communicate_info = task.evaluation_criteria.communicate_info or []
        if not communicate_info:
            comm_reward = 1.0
        else:
            assistant_texts = [
                msg["text"] for msg in self._conversation if msg["role"] == "assistant"
            ]
            all_found = all(
                any(
                    info_str.lower() in text.lower().replace(",", "")
                    for text in assistant_texts
                )
                for info_str in communicate_info
            )
            comm_reward = 1.0 if all_found else 0.0

        return db_reward * comm_reward

    def _make_fresh_environment(self):
        """Create a fresh environment with deepcopied default DB."""
        if self._domain == "airline":
            db_copy = _airline_default_db.model_copy(deep=True)
            from tau2.domains.airline.tools import AirlineTools
            tools = AirlineTools(db_copy)
            from tau2.environment.environment import Environment
            return Environment(domain_name="airline", policy=_airline_policy, tools=tools)
        else:
            db_copy = _retail_default_db.model_copy(deep=True)
            from tau2.domains.retail.tools import RetailTools
            tools = RetailTools(db_copy)
            from tau2.environment.environment import Environment
            return Environment(domain_name="retail", policy=_retail_policy, tools=tools)

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
            "domain": self._domain or "",
            "task_id": self._task.id if self._task else "",
            "steps": self._step_count,
            "reward": self.rewards.get(0, 0.0),
            "reason": getattr(self, "_reason", ""),
            "terminated_normally": self._terminated_normally,
            "conversation_length": len(self._conversation),
        }


# =====================================================================
# Action parsing (matches AdversarialPolicyGame)
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

_STOP_TOKENS = {"###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"}


def _is_user_stop(text: str) -> bool:
    """Check if user response contains a stop signal."""
    return any(token in text for token in _STOP_TOKENS)


def _strip_stop_tokens(text: str) -> str:
    """Remove stop tokens from text."""
    for token in _STOP_TOKENS:
        text = text.replace(token, "")
    return text.strip()
