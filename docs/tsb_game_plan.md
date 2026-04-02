# ToolSandbox Game Environment: Implementation Plan

## Overview

Wrap Apple's ToolSandbox benchmark (129 stateful, conversational tool-use scenarios) as a `GameEnv`-protocol game for GRPO training, following the same pattern as `tau2_bench_game.py`.

**File to create:** `/home/ubuntu/hangook/games/toolsandbox_game.py` (~400-450 lines)
**File to modify:** `/home/ubuntu/hangook/games/game_registry.py` (~30 lines added)

---

## Architecture

The model plays the AGENT role. Tool calls are executed by ToolSandbox's real `ExecutionEnvironment` (Python REPL + InteractiveConsole). A `UserLLMClient`-backed user simulator replaces ToolSandbox's GPT-based `OpenAIAPIUser`. Reward comes from ToolSandbox's milestone/minefield evaluation system.

### Key Architectural Mapping

| GameEnv Concept | ToolSandbox Equivalent |
|---|---|
| `reset(seed)` | `Scenario.starting_context` deep copy + system message processing |
| `step(action)` - tool call | `AGENT->EXEC_ENV` Message + `ExecutionEnvironment.respond()` |
| `step(action)` - text response | `AGENT->USER` Message + `UserLLMClient.generate()` |
| `rewards` | `Evaluation.evaluate(execution_context).similarity` |
| `done` | `conversation_active==False` in sandbox DB, or max_steps |

---

## Critical Design Decisions

### 1. Action Format

The model outputs JSON tool calls: `{"name": "...", "arguments": {...}}`, identical to `tau2_bench_game.py`. Text responses are wrapped as `{"name": "respond_to_user", "arguments": {"message": "..."}}`.

For tool execution, we convert this to Python code matching ToolSandbox's expected format:
```python
tool_0001_parameters = {"enabled": False}
tool_0001_response = set_cellular_service_status(**tool_0001_parameters)
print(repr(tool_0001_response))
```
This is the format that `openai_tool_call_to_python_code()` produces (verified at `message_conversion.py:51-93`).

### 2. Tool Execution via Real ExecutionEnvironment

We reuse ToolSandbox's `ExecutionEnvironment` role directly. It handles:
- `InteractiveConsole` code execution (`execution_environment.py:88`)
- Tool tracing for evaluation (`execution_environment.py:304-328`)
- Parallel tool call permutation checking (`execution_environment.py:140-228`)
- State DB updates via the global `ExecutionContext` singleton

We do NOT re-implement any tool execution logic.

### 3. User Simulation via UserLLMClient (NOT ToolSandbox's OpenAIAPIUser)

**Problem:** ToolSandbox's `OpenAIAPIUser` uses OpenAI function calling to invoke `end_conversation`. When the user sim LLM returns a `tool_calls` response for `end_conversation`, the message goes to `EXECUTION_ENVIRONMENT` which runs it, flipping `conversation_active` to `False` (verified at `user_tools.py:17-35`, `openai_api_user.py:92-120`).

`UserLLMClient.generate()` returns plain text only -- no tool calling support.

**Solution:** Same pattern as `tau2_bench_game.py`:
- Include an instruction in the user sim system prompt telling it to output `###STOP###` when the task is complete or the agent fails after 5 tries.
- Detect `###STOP###` in the response and programmatically call `end_conversation()` logic on the `ExecutionContext`.
- The user sim system prompt is built from the scenario's `SYSTEM->USER` message content (which contains `USER_INSTRUCTION` + task description).

### 4. Reward = `EvaluationResult.similarity`

ToolSandbox's evaluation system (verified at `evaluation.py:1216-1277`):
- `milestone_similarity`: arithmetic mean across all milestones (DAG-matched against execution context snapshots)
- `minefield_similarity`: arithmetic mean across all minefields (forbidden states)
- `similarity = 0 if minefield_similarity > 0, else milestone_similarity` (line 980-982)

This is a float in [0, 1], directly usable as GRPO reward.

### 5. Global ExecutionContext Singleton

ToolSandbox uses a module-level global `_global_execution_context` (verified at `execution_context.py:777-806`). All tool functions read/write state through `get_current_context()`.

**Thread safety:** Not thread-safe (documented at `execution_context.py:3-8`). GRPO training uses multiprocessing (not threading), so each process gets its own global. We call `set_current_context()` at the start of each `reset()` and `step()` to ensure correctness.

### 6. Scenario Selection: Base Only (No Augmentations)

The 129 base scenarios (tagged `NO_DISTRACTION_TOOLS`) are the training set by default. The 1,032 augmented variants (distraction tools, scrambled names/descriptions) can be optionally enabled via constructor parameter.

**Rationale:** Augmentation variants test robustness but would dilute the core skill-learning signal during training. They're better suited for evaluation.

---

## Sandbox DB Message Structure at Scenario Start

Verified by tracing through `base_scenarios.py:22-155` and `ScenarioExtension.get_extended_scenario()` at `scenario.py:246-274`:

### Base scenario messages (shared by all scenarios):

| Index | Sender | Recipient | Content | visible_to |
|-------|--------|-----------|---------|------------|
| 0 | SYSTEM | EXEC_ENV | `import json; from tool_sandbox.tools.setting import ...` | null (default) |
| 1 | SYSTEM | AGENT | "Don't make assumptions about what values to plug into functions..." | null (default) |
| 2-13 | mixed | mixed | Few-shot user sim example (12 msgs, scenario: `send_message_with_contact_content_cellular_off_multiple_user_turn`) | `[USER]` |

### Extension messages (scenario-specific, appended after base):

**Single-tool-call scenario (e.g., `cellular_off`):**

| Index | Sender | Recipient | Content | visible_to |
|-------|--------|-----------|---------|------------|
| 14 | SYSTEM | USER | `USER_INSTRUCTION + "Turn off cellular service"` | null |
| 15 | USER | AGENT | "Turn off cellular" | null |

**Multi-user-turn scenario (e.g., `search_message_with_recency_latest_multiple_user_turn`):**

| Index | Sender | Recipient | Content | visible_to |
|-------|--------|-----------|---------|------------|
| 14-19 | mixed | mixed | Scenario-specific few-shot example (6 msgs) | `[USER]` |
| 20 | SYSTEM | USER | `USER_INSTRUCTION + "Find the content of your most recent message..."` | null |
| 21 | USER | AGENT | "I wanna find a message" | null |

### How to identify the "real" initial messages:

- **Real USER->AGENT message:** Use `execution_context.first_user_sandbox_message_index` property (verified at `execution_context.py:466-489`). Filters for `sender==USER AND (visible_to != [USER] OR visible_to IS NULL)`.
- **Real SYSTEM->USER message (user sim instruction):** Filter sandbox DB for `sender==SYSTEM AND recipient==USER AND (visible_to != [USER] OR visible_to IS NULL)`.
- **Agent system prompt:** The `SYSTEM->AGENT` message at index 1.

---

## Detailed Implementation: `reset(seed)`

```
def reset(self, seed: int) -> None:
    # 1. Ensure scenarios are loaded (lazy, cached globally)
    _ensure_scenarios_loaded()

    # 2. Pick scenario deterministically
    scenario_name = _scenario_names[seed % len(_scenario_names)]
    scenario = _scenarios[scenario_name]

    # 3. Deep copy starting_context, set as global
    #    (Matches scenario.play() at scenario.py:75-77)
    self._execution_context = copy.deepcopy(scenario.starting_context)
    set_current_context(self._execution_context)

    # 4. Store scenario reference for evaluation later
    self._scenario = scenario
    self._scenario_name = scenario_name

    # 5. Create ExecutionEnvironment and process system messages
    #    (Matches scenario.play() at scenario.py:79-96)
    self._exec_env = ExecutionEnvironment()
    sandbox_db = self._execution_context.get_database(
        DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    )
    max_idx = self._execution_context.max_sandbox_message_index
    for msg_idx in range(max_idx + 1):
        if (sandbox_db["recipient"][msg_idx] == RoleType.EXECUTION_ENVIRONMENT
            and sandbox_db["sender"][msg_idx] == RoleType.SYSTEM):
            self._exec_env.respond(ending_index=msg_idx)

    # 6. Extract agent system prompt from SYSTEM->AGENT message
    #    Content: "Don't make assumptions about what values to plug into functions..."
    sandbox_db = self._execution_context.get_database(
        DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    )
    # Filter: sender==SYSTEM, recipient==AGENT
    self._agent_system_prompt = <extracted content>

    # 7. Extract user sim instruction from SYSTEM->USER message
    #    Filter: sender==SYSTEM, recipient==USER, visible_to != [USER]
    #    Content: USER_INSTRUCTION + task description
    self._user_sim_system_prompt = <extracted content>
    # Append stop token instruction for UserLLMClient compatibility

    # 8. Extract initial USER->AGENT message
    #    Use first_user_sandbox_message_index or same filter as above
    first_user_msg_content = <extracted content>

    # 9. Compute tool schemas for this scenario
    #    (scenario may have different tool_allow_list)
    available_tools = self._execution_context.get_available_tools(
        scrambling_allowed=False  # no scrambling for training
    )
    # Filter to AGENT-visible tools only
    self._available_tools = {
        name: tool for name, tool in available_tools.items()
        if RoleType.AGENT in getattr(tool, "visible_to", (RoleType.AGENT,))
    }
    self._tool_schemas = convert_to_openai_tools(self._available_tools)

    # 10. Initialize conversation tracking
    #     Internal format matches tau2_bench_game.py:
    #     [{"role": "user"|"assistant"|"tool_call"|"tool_result", "text": "..."}]
    self._conversation = [{"role": "user", "text": first_user_msg_content}]

    # 11. Initialize GameEnv protocol attributes
    self.done = False
    self.current_player = 0
    self.rewards = {0: 0.0}
    self.invalid_player = None
    self._step_count = 0
    self._last_call_key = None
    self._repeat_count = 0
    self._terminated_normally = False
```

### Verification notes for reset():

- Step 5 exactly replicates `scenario.play()` lines 79-96. The assertion at line 95 (`max_sandbox_message_index unchanged after system message processing`) should hold because system messages to EXEC_ENV don't produce response messages (verified at `execution_environment.py:89-91`: "If this message is from system, do not respond").
- Step 9: `get_available_tools(scrambling_allowed=False)` returns tools filtered by `tool_allow_list` and `tool_deny_list` (verified at `execution_context.py:308-343`). We further filter to `visible_to` containing `AGENT` (verified: `base_role.py:112-121` does this in `get_available_tools()`).
- Step 8: The `first_user_sandbox_message_index` property correctly skips few-shot messages by filtering `visible_to != [USER]` (verified at `execution_context.py:476-486`).

---

## Detailed Implementation: `step(action)`

```
def step(self, action: Optional[str]) -> None:
    if self.done:
        return
    self._step_count += 1

    # CRITICAL: Ensure our context is the global one
    set_current_context(self._execution_context)

    # Parse action (reuse _parse_tool_call from tau2_bench_game.py pattern)
    if action is None:
        self._finalize(0.0, "No action provided")
        self.invalid_player = 0
        return

    tool_call = _parse_tool_call(action)
    if tool_call is None:
        self._finalize(0.0, "Invalid action format")
        self.invalid_player = 0
        return

    tool_name = tool_call["name"]
    tool_args = tool_call["arguments"]

    if tool_name == "respond_to_user":
        # --- CASE A: Agent sends text message to user ---
        message_text = tool_args.get("message", "")

        # A1. Add AGENT->USER message to sandbox DB
        agent_msg = Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content=message_text,
        )
        self._execution_context.add_to_database(
            namespace=DatabaseNamespace.SANDBOX,
            rows=[attrs.asdict(agent_msg)],
        )
        self._conversation.append({"role": "assistant", "text": message_text})

        # A2. Call UserLLMClient for user response
        user_response = self._get_user_response()
        if user_response is None:
            self._finalize_with_reward("User LLM failure")
            return

        # A3. Check for stop signal
        if _is_user_stop(user_response):
            clean = _strip_stop_tokens(user_response)
            if clean:
                # Add cleaned user message before ending
                user_msg = Message(
                    sender=RoleType.USER, recipient=RoleType.AGENT,
                    content=clean,
                )
                self._execution_context.add_to_database(
                    DatabaseNamespace.SANDBOX, [attrs.asdict(user_msg)]
                )
                self._conversation.append({"role": "user", "text": clean})
            # Programmatically end conversation
            self._end_conversation_and_finalize()
            return

        # A4. Add USER->AGENT message to sandbox DB
        user_msg = Message(
            sender=RoleType.USER, recipient=RoleType.AGENT,
            content=user_response,
        )
        self._execution_context.add_to_database(
            DatabaseNamespace.SANDBOX, [attrs.asdict(user_msg)]
        )
        self._conversation.append({"role": "user", "text": user_response})

    else:
        # --- CASE B: Agent makes a tool call ---

        # B1. Convert to Python code (ToolSandbox format)
        tc_id = f"tool_{self._step_count:04d}"
        # Must use repr() for string args to avoid injection
        python_code = (
            f"{tc_id}_parameters = {tool_args}\n"
            f"{tc_id}_response = {tool_name}(**{tc_id}_parameters)\n"
            f"print(repr({tc_id}_response))"
        )

        # B2. Add AGENT->EXEC_ENV message to sandbox DB
        agent_msg = Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content=python_code,
            openai_tool_call_id=tc_id,
            openai_function_name=tool_name,
        )
        self._execution_context.add_to_database(
            DatabaseNamespace.SANDBOX, [attrs.asdict(agent_msg)]
        )
        self._conversation.append({
            "role": "tool_call",
            "text": json.dumps(tool_call),
        })

        # B3. Call ExecutionEnvironment.respond()
        #     Reads last message from sandbox DB, executes code,
        #     writes EXEC_ENV->AGENT response back to sandbox DB
        #     (verified: execution_environment.py:279-329)
        self._exec_env.respond()

        # B4. Read tool result from sandbox DB (last message)
        sandbox_db = self._execution_context.get_database(
            DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
        )
        tool_result_content = sandbox_db["content"][-1]
        self._conversation.append({
            "role": "tool_result",
            "text": tool_result_content,
        })

        # B5. Loop detection (3x identical tool call)
        call_key = json.dumps(
            {"name": tool_name, "arguments": tool_args}, sort_keys=True
        )
        if call_key == self._last_call_key:
            self._repeat_count += 1
        else:
            self._last_call_key = call_key
            self._repeat_count = 1
        if self._repeat_count >= 3:
            self._finalize(0.0, f"Loop: {tool_name} called 3x")
            return

    # Check if conversation was ended by a tool (e.g., end_conversation
    # called by some tool chain, though unlikely in agent path)
    sandbox_db = self._execution_context.get_database(DatabaseNamespace.SANDBOX)
    if not sandbox_db["conversation_active"][-1]:
        self._finalize_with_reward("Conversation ended")
        return

    # Check max steps
    if self._step_count >= self.max_steps:
        self._finalize(0.0, "Max steps reached")
```

### Verification notes for step():

- **B2: Message format.** `openai_tool_call_id` and `openai_function_name` must be set for the evaluation's tool trace matching to work. Verified: `execution_environment.py:105-111` copies these fields to the response message, which is used by `tool_trace_dependent_similarity` evaluators.
- **B3: messages_validation.** `ExecutionEnvironment.respond()` calls `self.messages_validation(messages)` (line 280) which checks `messages[-1].recipient == EXECUTION_ENVIRONMENT`. Since we just added an `AGENT->EXEC_ENV` message, this passes.
- **B4: Reading result.** After `respond()`, the last message in sandbox DB is `EXEC_ENV->AGENT`. Its `content` field contains stdout + stderr from execution (verified at `execution_environment.py:96-108`).
- **B1: Python code format.** The `tool_args` dict is inserted literally. This matches how `openai_tool_call_to_python_code` works (line 89 in `message_conversion.py`): `f"{tool_id}_parameters = {json.loads(tool_call.function.arguments)}"`. We use `tool_args` which is already a dict, so `{tool_args}` produces the same Python literal.

**Potential issue with B1:** If `tool_args` contains strings with special characters, the Python literal repr might differ from what `json.loads` would produce. For safety, we should use `json.loads(json.dumps(tool_args))` to ensure consistent formatting, or better yet, just assign the dict directly since Python can handle it.

---

## Detailed Implementation: User Simulator

```
def _get_user_response(self) -> Optional[str]:
    """Call UserLLMClient with ToolSandbox's user sim prompt."""
    if self._user_client is None:
        return "###STOP###"  # No user sim -> immediate stop

    # Build role-flipped messages
    # (verified against OpenAIAPIUser.to_openai_messages at
    #  openai_api_user.py:146-188)
    #
    # Role mapping:
    #   AGENT->USER  content -> {"role": "user", "content": ...}
    #   USER->AGENT  content -> {"role": "assistant", "content": ...}
    #   SYSTEM->USER content -> skipped (already in system prompt)
    #   USER<->EXEC_ENV      -> skipped
    #
    # We use self._conversation which only tracks agent-visible messages
    flipped = []
    for msg in self._conversation:
        role = msg["role"]
        text = msg["text"]
        if role == "assistant":
            # Agent spoke to user -> from user sim's perspective, this is "user" role
            flipped.append({"role": "user", "content": text})
        elif role == "user":
            # User spoke to agent -> from user sim's perspective, this is "assistant" role
            flipped.append({"role": "assistant", "content": text})
        # Skip tool_call and tool_result (invisible to user)

    try:
        raw = self._user_client.generate(self._user_sim_system_prompt, flipped)
    except Exception:
        return None

    # Strip thinking tags (Qwen3)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw
```

### User sim system prompt construction:

```
# In reset(), after extracting SYSTEM->USER message content:
user_instruction_content = <SYSTEM->USER message content>

self._user_sim_system_prompt = (
    user_instruction_content + "\n\n"
    "IMPORTANT: When the agent (User B) has completed the task, or "
    "when the agent cannot complete the task after 5 tries, output "
    "###STOP### to end the conversation."
)
```

The `USER_INSTRUCTION` text already contains the full role-play instructions (verified at `user_simulator_few_shot_examples.py:13-35`):
- "You are no longer an assistant. From now on role play as a user (User A)..."
- Rules about answering accurately, using casual language, etc.
- The task description is appended at the end

We append the `###STOP###` instruction to make it compatible with `UserLLMClient` (text-only, no tool calling).

---

## Detailed Implementation: Reward and Finalization

```
def _compute_reward(self) -> float:
    """Use ToolSandbox's evaluation system.

    Verified at evaluation.py:1226-1277:
    - milestone_matcher.compute_mapping_and_similarity()
    - minefield_matcher.compute_mapping_and_similarity()
    - Returns EvaluationResult with .similarity

    EvaluationResult.similarity (line 980-982):
    = 0 if minefield_similarity > 0
    = milestone_similarity otherwise
    """
    eval_result = self._scenario.evaluation.evaluate(
        execution_context=self._execution_context,
        max_turn_count=self._scenario.max_messages,
    )
    return eval_result.similarity


def _end_conversation_and_finalize(self):
    """Programmatically end conversation and compute reward.

    Replicates end_conversation() logic from user_tools.py:17-35:
    - Flips conversation_active to False in sandbox DB
    """
    sandbox_db = self._execution_context.get_database(DatabaseNamespace.SANDBOX)
    self._execution_context.update_database(
        DatabaseNamespace.SANDBOX,
        sandbox_db.with_columns(~pl.col("conversation_active")),
    )
    self._terminated_normally = True
    reward = self._compute_reward()
    self._finalize(reward, "User ended conversation")


def _finalize_with_reward(self, reason: str):
    """Compute reward and finalize."""
    self._terminated_normally = True
    reward = self._compute_reward()
    self._finalize(reward, reason)


def _finalize(self, reward: float, reason: str):
    self.done = True
    self.rewards = {0: reward}
    self._reason = reason
```

---

## Structured Message Methods

```
def get_system_prompt(self) -> str:
    """Agent system prompt extracted from SYSTEM->AGENT message.
    Content: 'Don't make assumptions about what values to plug into functions...'
    """
    return self._agent_system_prompt

def get_tool_schemas(self) -> List[Dict]:
    """OpenAI-format tool schemas from ExecutionContext.
    Computed via convert_to_openai_tools() (tool_conversion.py:402-405).
    Respects scenario's tool_allow_list.
    """
    return self._tool_schemas

def get_tool_schemas_compact(self) -> List[Dict]:
    """Tool schemas with descriptions stripped (same as tau2_bench_game.py)."""
    # Cache and return _strip_descriptions(self._tool_schemas)

def get_messages(self) -> List[Dict]:
    """OpenAI chat-API-format messages.
    Same format as tau2_bench_game.py:234-280:
    - {"role": "user", "content": "..."}
    - {"role": "assistant", "content": "...", "tool_calls": null}
    - {"role": "assistant", "content": null, "tool_calls": [...]}
    - {"role": "tool", "content": "...", "tool_call_id": "..."}
    """
    # Convert self._conversation to OpenAI format
    # Same logic as tau2_bench_game.py get_messages()

def observe(self, player_id: int) -> str:
    """Text-based observation (fallback for non-structured path).
    System prompt + tool schemas + conversation + action format instructions.
    """
```

---

## Module-Level Caching

```python
# Path setup
_GAME_DIR = pathlib.Path(__file__).resolve().parent
_TSB_DIR = _GAME_DIR / "evals" / "benchmarks" / "ToolSandbox"

# Add to sys.path for tool_sandbox imports
if str(_TSB_DIR) not in sys.path:
    sys.path.insert(0, str(_TSB_DIR))

# Lazy-loaded scenario cache (loaded once on first reset())
_scenarios: Optional[Dict[str, Scenario]] = None
_scenario_names: Optional[List[str]] = None

def _ensure_scenarios_loaded():
    global _scenarios, _scenario_names
    if _scenarios is not None:
        return
    from tool_sandbox.scenarios import named_scenarios
    from tool_sandbox.common.tool_discovery import ToolBackend
    all_scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
    # Filter to base scenarios only (NO_DISTRACTION_TOOLS)
    _scenarios = {
        name: s for name, s in all_scenarios.items()
        if ScenarioCategories.NO_DISTRACTION_TOOLS in s.categories
    }
    _scenario_names = sorted(_scenarios.keys())
```

---

## Registration in game_registry.py

```python
# At top of file, with other imports:
try:
    from toolsandbox_game import (
        ToolSandboxGame,
        extract_action as extract_action_toolsandbox,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TOOLSANDBOX,
    )
except ImportError:
    ToolSandboxGame = None

# In _register_builtin_games():
if ToolSandboxGame is not None:
    def make_toolsandbox(user_client=None) -> ToolSandboxGame:
        return ToolSandboxGame(max_steps=30, user_client=user_client)

    register_game(GameSpec(
        name="toolsandbox",
        make_env=make_toolsandbox,
        extract_action=extract_action_toolsandbox,
        action_space=[],
        stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
        system_prompt=SYSTEM_PROMPT_TOOLSANDBOX,
        max_gen_tokens=1024,
    ))
```

---

## extract_action() and SYSTEM_PROMPT

```python
# Module-level constants
SYSTEM_PROMPT = (
    "Don't make assumptions about what values to plug into functions. "
    "Ask for clarification if a user request is ambiguous."
)

_STOP_TOKENS = {"###STOP###"}

def _is_user_stop(text: str) -> bool:
    return any(token in text for token in _STOP_TOKENS)

def _strip_stop_tokens(text: str) -> str:
    for token in _STOP_TOKENS:
        text = text.replace(token, "")
    return text.strip()

def _parse_tool_call(text: str) -> Optional[Dict]:
    """Parse JSON tool call from model output.
    Same implementation as tau2_bench_game.py:540-565.
    Finds innermost {...} JSON object with 'name' key.
    """

def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action for game registry compatibility.
    Same implementation as tau2_bench_game.py:568-583.
    JSON tool call -> return it.
    Plain text -> wrap as respond_to_user.
    Invalid -> return None.
    """
```

---

## Known Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Global ExecutionContext not thread-safe** | Corrupted state if threads share context | GRPO uses multiprocessing. `set_current_context()` at start of `reset()` and `step()`. |
| **UserLLMClient can't call end_conversation tool** | User sim can't signal task completion via tool calling | Detect `###STOP###` token; programmatically flip `conversation_active`. |
| **`datetime.datetime.now()` in tools** | Timestamps vary between runs; some scenarios compare timestamps | Milestone evaluation mostly uses tool traces (function name + args), not absolute timestamps. Few scenarios may be noisy. |
| **Python code injection via tool_args** | Malicious args could execute arbitrary code | Training model is our own; args come from model output parsed as JSON, not user input. Same risk level as ToolSandbox's own agent roles. |
| **Scenario loading is slow (imports all tools)** | First `reset()` takes several seconds | One-time cost, cached globally. Same pattern as tau2_bench_game.py module-level caching. |
| **Tool schemas differ per scenario** | Can't cache globally like tau2_bench | Recompute `convert_to_openai_tools()` in each `reset()`. Cheap operation. |
| **Few-shot examples in sandbox DB** | Must not leak to agent | All few-shot messages have `visible_to=[USER]`. Our `get_messages()` only tracks messages we explicitly add to `self._conversation`. Agent never sees few-shot context. |
| **Polars import** | New dependency | Already present via ToolSandbox. Only used for `end_conversation` DB update and reading sandbox DB. |

---

## Scenario Statistics

| Category | Base Count | With Augmentations |
|----------|-----------|-------------------|
| Single tool call | 19 | 152 |
| Multiple tool calls | 54 | 432 |
| Multiple user turns | 28 | 224 |
| Insufficient information | 28 | 224 |
| **Total** | **129** | **1,032** |

---

## Dependencies

**Already available (no new installs needed):**
- `polars` - used by ToolSandbox for all DB operations
- `attrs` - used by ToolSandbox for Message/Scenario classes
- `networkx` - used by evaluation DAG
- `scipy` - used by evaluation (Hungarian algorithm)
- `rouge-score` - used by text similarity evaluation

**ToolSandbox package itself:** Imported via `sys.path` manipulation (same pattern as tau2-bench). NOT installed as a pip package.

---

## Implementation Order

1. Module-level setup: path manipulation, imports, lazy scenario caching
2. `ToolSandboxGame.__init__()` and `reset()`
3. `step()` - tool call path (Case B)
4. `step()` - respond_to_user path (Case A) + `_get_user_response()`
5. Reward: `_compute_reward()`, `_end_conversation_and_finalize()`, `_finalize()`
6. Structured message methods: `get_system_prompt()`, `get_tool_schemas()`, `get_messages()`, `observe()`
7. Module-level: `extract_action()`, `SYSTEM_PROMPT`, `_parse_tool_call()`
8. Registration in `game_registry.py`
