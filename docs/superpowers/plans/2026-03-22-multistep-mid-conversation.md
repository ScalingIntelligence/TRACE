# Multi-Step Task Game: Mid-Conversation Rewrite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `multistep_task_game.py` to place the model mid-conversation (after auth + lookups), eliminating the LLM user simulator and matching the exact tau2-bench eval format. Reweight operation distributions to match verified eval failure data.

**Architecture:** The game pre-fills the conversation with greeting, authentication, all lookups, agent summary, and user confirmation. The model then produces tool calls freely (writes and/or lookups). Each tool call is executed via ToolExecutor with the result appended. Game ends when the model sends text (respond_to_user) or hits max_steps. Reward is based on how many expected write-actions were completed. Follows the `structured_data_new_game.py` pattern (single-turn with pre-filled context) but allows multiple steps for the write phase.

**Tech Stack:** Python, existing `ToolExecutor`, existing `synthetic_db` generators, existing tau2-bench tool schemas/policies.

---

### Task 1: Create the new game file with scenario generation

**Files:**
- Create: `multistep_task_game.py` (overwrite existing)

This is the core implementation. The file keeps the existing operation generators (`_op_cancel_reservation`, `_op_change_flight`, etc.) and retail generators, but replaces the game class and scenario generation.

#### Key design decisions:

1. **Pre-filled conversation format** (matches tau2-bench eval exactly):
   ```
   [0] assistant: "Welcome to customer service. How can I help you today?"
   [1] user: "Hi, I'm {name}, zip {zip}. I need help with: {task_list}"
   [2] assistant: tool_calls=[find_user_id_by_name_zip(...)]
   [3] tool: {user_id}
   [4] assistant: tool_calls=[get_user_details(...)]
   [5] tool: {user_details_json}
   [6] assistant: tool_calls=[get_reservation_details(...)]  # or get_order_details
   [7] tool: {reservation_json}
   ... (more lookups as needed)
   [N] assistant: "I've reviewed your information. I'll process: {summary}. Shall I proceed?"
   [N+1] user: "Yes, please proceed."
   ```

2. **Game loop** (matches tau2-bench orchestrator):
   - Model produces tool call → ToolExecutor executes → result appended → model called again
   - Model produces text (respond_to_user) → game ends
   - max_steps = n_ops + 5 (buffer for optional lookups)

3. **Updated distributions** (from verified eval failure data):

   **Airline n-ops:** `[2, 3, 4, 5] @ [50, 29, 14, 7]` (unchanged — already exact match)

   **Airline tool weights:** `[flight_change=44, cancel=28, baggages=13, book=10, passenger=5]` (unchanged — already exact match)

   **Retail n-ops:** `[2, 3, 4, 5] @ [48, 39, 10, 3]` (was `[58, 35, 3, 3]`)

   **Retail tool weights:**
   - `modify_items=27` (was 28)
   - `return=20` (was 23)
   - `modify_address=20` (was 21)
   - `exchange=13` (was 14)
   - `cancel=12` (was 13)
   - `modify_user_address=7` (NEW — was missing)

4. **Reward** (same formula as before):
   - 0.6 * (correct_ops / total_ops) + 0.4 completion bonus if all correct
   - Penalty for wrong write calls: -0.1 per wrong call
   - Lookup calls are ignored in reward (neither penalized nor rewarded)

- [ ] **Step 1: Write the failing test for scenario generation**

Create `tests/test_multistep_mid_convo.py`:

```python
"""Tests for the mid-conversation multistep task game."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_generate_scenario_airline():
    """Scenario generates valid pre-filled conversation for airline."""
    from multistep_task_game import generate_scenario
    scenario = generate_scenario(42, domain="airline")

    assert scenario.domain == "airline"
    assert len(scenario.operations) >= 2
    assert len(scenario.operations) <= 5

    # Must have pre-filled messages
    msgs = scenario.messages
    assert len(msgs) >= 6, f"Expected at least 6 pre-filled messages, got {len(msgs)}"

    # First message is assistant greeting
    assert msgs[0]["role"] == "assistant"

    # Second message is user with task list
    assert msgs[1]["role"] == "user"

    # Must have tool_calls (lookups) in the pre-filled messages
    has_tool_call = any(m.get("tool_calls") for m in msgs if m["role"] == "assistant")
    assert has_tool_call, "Pre-filled messages must include lookup tool calls"

    # Must have tool results
    has_tool_result = any(m["role"] == "tool" for m in msgs)
    assert has_tool_result, "Pre-filled messages must include tool results"

    # Last user message should be confirmation
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) >= 2, "Need initial request + confirmation"

    # DB must be populated
    assert scenario.db, "DB must be populated"


def test_generate_scenario_retail():
    """Scenario generates valid pre-filled conversation for retail."""
    from multistep_task_game import generate_scenario
    scenario = generate_scenario(100, domain="retail")

    assert scenario.domain == "retail"
    assert len(scenario.operations) >= 2

    msgs = scenario.messages
    assert len(msgs) >= 6

    # Check for lookup tool calls
    tool_call_names = []
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tool_call_names.append(tc["function"]["name"])

    assert "find_user_id_by_name_zip" in tool_call_names, "Must include auth lookup"


def test_scenario_deterministic():
    """Same seed produces same scenario."""
    from multistep_task_game import generate_scenario
    s1 = generate_scenario(42, domain="airline")
    s2 = generate_scenario(42, domain="airline")
    assert len(s1.operations) == len(s2.operations)
    assert s1.messages == s2.messages


def test_retail_has_modify_user_address():
    """Retail scenarios can generate modify_user_address operations."""
    from multistep_task_game import generate_scenario
    found = False
    for seed in range(500):
        s = generate_scenario(seed, domain="retail")
        for op in s.operations:
            if op.tool_name == "modify_user_address":
                found = True
                break
        if found:
            break
    assert found, "modify_user_address should appear in retail scenarios"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/hangook/games && conda run -n games python -m pytest tests/test_multistep_mid_convo.py -v`
Expected: FAIL (tests reference new `scenario.messages` attribute that doesn't exist yet)

- [ ] **Step 3: Implement the scenario generation changes**

In `multistep_task_game.py`, make these changes:

**3a. Add `messages` field to `MultiStepScenario`:**
```python
@dataclass
class MultiStepScenario:
    domain: str
    messages: List[Dict[str, Any]]  # Pre-filled conversation (OpenAI format)
    operations: List[TaskOperation]
    description: str = ""
    db: Dict[str, Any] = field(default_factory=dict)
```
Remove `user_system_prompt` and `initial_message` fields (no longer needed).

**3b. Add `_retail_op_modify_user_address` generator:**
```python
def _retail_op_modify_user_address(rng, order, user, products_db):
    """Change the user's profile address."""
    city, state, zipcode = rng.choice(CITIES_STATES_ZIPS)
    new_addr = {
        "address1": f"{rng.randint(100, 999)} {rng.choice(STREETS)}",
        "address2": "",
        "city": city,
        "state": state,
        "country": "USA",
        "zip": zipcode,
    }
    return TaskOperation(
        description=f"update my profile address to {new_addr['address1']}, {city}, {state} {zipcode}",
        tool_name="modify_user_address",
        tool_args={"user_id": user["user_id"], **new_addr},
        key_args=["user_id"],
    )
```

**3c. Update `_RETAIL_OP_GENERATORS`:**
```python
_RETAIL_OP_GENERATORS = [
    ("cancel", _retail_op_cancel, "pending", 12),
    ("exchange", _retail_op_exchange, "delivered", 13),
    ("modify_items", _retail_op_modify_items, "pending", 27),
    ("modify_address", _retail_op_modify_address, "pending", 20),
    ("return", _retail_op_return, "delivered", 20),
    ("modify_user_address", _retail_op_modify_user_address, "any", 7),
]
```
Note: `modify_user_address` doesn't depend on order status, so use "any".

**3d. Update retail n-ops weights:**
```python
n_ops_target = rng.choices([2, 3, 4, 5], weights=[48, 39, 10, 3])[0]
```

**3e. Add `_build_prefilled_conversation` function:**

This is the core new function. It builds the pre-filled conversation by:
1. Creating greeting
2. Building user message with all task descriptions
3. Executing auth lookups via ToolExecutor
4. Executing data lookups (reservations/orders, products, flights)
5. Building agent summary
6. Adding user confirmation

```python
def _build_prefilled_conversation(
    scenario_domain: str,
    user: Dict,
    operations: List[TaskOperation],
    db: Dict,
) -> List[Dict[str, Any]]:
    """Build pre-filled conversation in OpenAI message format.

    Executes auth + data lookups via ToolExecutor, building the same
    message sequence the model would see mid-conversation in tau2-bench eval.
    """
    te = ToolExecutor(scenario_domain, copy.deepcopy(db))
    msgs = []
    call_id = 0

    def _add_tool_call(name, args):
        nonlocal call_id
        cid = f"call_{call_id:04d}"
        call_id += 1
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }],
        })
        result = te.execute(name, args)
        msgs.append({
            "role": "tool",
            "content": result,
            "tool_call_id": cid,
        })
        return result

    # Phase 1: Greeting
    if scenario_domain == "retail":
        msgs.append({"role": "assistant", "content": "Hi! How can I help you today?"})
    else:
        msgs.append({"role": "assistant", "content": "Welcome to our airline customer service. How can I assist you today?"})

    # Phase 2: User request with all tasks
    name = user["name"]
    task_lines = [op.description for op in operations]
    task_text = "; ".join(task_lines)

    if scenario_domain == "retail":
        zipcode = user["address"]["zip"]
        user_msg = (
            f"Hi, I'm {name['first_name']} {name['last_name']}, "
            f"zip code {zipcode}. I need help with the following: {task_text}."
        )
    else:
        uid = user["user_id"]
        user_msg = (
            f"Hi, I'm {name['first_name']} {name['last_name']}. "
            f"My user ID is {uid}. I need help with: {task_text}."
        )
    msgs.append({"role": "user", "content": user_msg})

    # Phase 3: Auth lookups
    if scenario_domain == "retail":
        _add_tool_call("find_user_id_by_name_zip", {
            "first_name": name["first_name"],
            "last_name": name["last_name"],
            "zip": user["address"]["zip"],
        })
    _add_tool_call("get_user_details", {"user_id": user["user_id"]})

    # Phase 4: Data lookups
    if scenario_domain == "retail":
        # Look up all orders
        for order_id in user.get("orders", []):
            _add_tool_call("get_order_details", {"order_id": order_id})
        # Look up products referenced in operations
        seen_products = set()
        for op in operations:
            if op.tool_name in ("exchange_delivered_order_items", "modify_pending_order_items"):
                # Need product details for variant selection
                for order_id in user.get("orders", []):
                    order_data = db.get("orders", {}).get(order_id, {})
                    for item in order_data.get("items", []):
                        pid = item.get("product_id")
                        if pid and pid not in seen_products:
                            _add_tool_call("get_product_details", {"product_id": pid})
                            seen_products.add(pid)
    else:
        # Airline: look up all reservations
        for res_id in user.get("reservations", []):
            _add_tool_call("get_reservation_details", {"reservation_id": res_id})
        # Search for flights if any flight change operations
        for op in operations:
            if op.tool_name in ("update_reservation_flights", "book_reservation"):
                flight_args = op.tool_args.get("flights", [])
                if flight_args:
                    f = flight_args[0]
                    origin = f.get("origin", op.tool_args.get("origin", ""))
                    dest = f.get("destination", op.tool_args.get("destination", ""))
                    if origin and dest:
                        _add_tool_call("search_direct_flight", {
                            "origin": origin, "destination": dest,
                        })

    # Phase 5: Agent summary + Phase 6: User confirmation
    op_summaries = []
    for i, op in enumerate(operations, 1):
        op_summaries.append(f"{i}. {op.description}")
    summary = "\n".join(op_summaries)

    msgs.append({
        "role": "assistant",
        "content": (
            f"I've reviewed your account and found the relevant information. "
            f"I'll process the following for you:\n{summary}\n\n"
            f"Shall I proceed with all of these?"
        ),
    })
    msgs.append({"role": "user", "content": "Yes, please proceed with all of them."})

    return msgs
```

**3f. Update `generate_scenario` to build pre-filled messages:**

In both the airline and retail paths of `generate_scenario()`, after building operations and DB, call `_build_prefilled_conversation` and store the result in `scenario.messages`.

**3g. Update `_generate_retail_scenario` to handle `modify_user_address`:**

The `modify_user_address` op doesn't need a specific order status. Update the generation loop to handle `required_status="any"`.

- [ ] **Step 4: Run scenario generation tests**

Run: `cd /home/ubuntu/hangook/games && conda run -n games python -m pytest tests/test_multistep_mid_convo.py::test_generate_scenario_airline tests/test_multistep_mid_convo.py::test_generate_scenario_retail tests/test_multistep_mid_convo.py::test_scenario_deterministic tests/test_multistep_mid_convo.py::test_retail_has_modify_user_address -v`
Expected: PASS

- [ ] **Step 5: Commit scenario generation**

```bash
git add multistep_task_game.py tests/test_multistep_mid_convo.py
git commit -m "feat: multistep game mid-conversation scenario generation

Pre-fills auth + lookups in conversation, matching tau2-bench eval format.
Adds modify_user_address for retail. Updates retail n-ops weights to [48,39,10,3]."
```

---

### Task 2: Rewrite the game class

**Files:**
- Modify: `multistep_task_game.py` (RealisticMultiStepGame class)

The game class changes from multi-turn (with LLM user sim) to mid-conversation (pre-filled context, model produces tool calls).

- [ ] **Step 1: Write failing tests for the game class**

Add to `tests/test_multistep_mid_convo.py`:

```python
def test_game_reset_and_messages():
    """Game reset produces valid messages for the model."""
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame(domain="airline")
    game.reset(42)

    assert not game.done
    msgs = game.get_messages()
    assert len(msgs) >= 6

    # System prompt must contain policy
    sys_prompt = game.get_system_prompt()
    assert "<policy>" in sys_prompt
    assert "<instructions>" in sys_prompt

    # Tools must be available
    tools = game.get_tool_schemas()
    assert len(tools) > 0
    tool_names = [t["function"]["name"] for t in tools]
    assert "get_user_details" in tool_names


def test_game_step_write_action():
    """Game accepts write actions and tracks them."""
    import json
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame(domain="retail")
    game.reset(100)

    # Get the expected operations
    ops = game._scenario.operations
    assert len(ops) >= 2

    # Submit the first expected write action
    op = ops[0]
    action = json.dumps({"name": op.tool_name, "arguments": op.tool_args})
    game.step(action)

    # Game should NOT be done after first write (more ops expected)
    # (unless it was a 1-op game, but we enforce >= 2)
    # The tool result should be in the conversation
    msgs = game.get_messages()
    last_tool = [m for m in msgs if m["role"] == "tool"]
    assert len(last_tool) > 0, "Tool result should be appended"


def test_game_step_text_ends_game():
    """Sending text (respond_to_user) ends the game."""
    import json
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame(domain="airline")
    game.reset(42)

    action = json.dumps({"name": "respond_to_user", "arguments": {"message": "All done!"}})
    game.step(action)

    assert game.done, "respond_to_user should end the game"


def test_game_full_correct_sequence():
    """Submitting all correct write actions gives reward=1.0."""
    import json
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame(domain="retail")
    game.reset(100)

    ops = game._scenario.operations
    for op in ops:
        assert not game.done
        action = json.dumps({"name": op.tool_name, "arguments": op.tool_args})
        game.step(action)

    # Now send text to end
    action = json.dumps({"name": "respond_to_user", "arguments": {"message": "Done!"}})
    game.step(action)

    assert game.done
    assert game.rewards[0] == 1.0, f"Expected reward 1.0, got {game.rewards[0]}"


def test_game_no_user_client_needed():
    """Game should NOT require user_client (no LLM user sim)."""
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame(domain="airline")
    # Should not raise even without user_client
    game.reset(42)
    assert not game.done


def test_game_max_steps():
    """Game ends at max_steps with partial reward."""
    import json
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame(max_steps=2, domain="retail")
    game.reset(100)

    # Do 2 lookups (wasting steps)
    for _ in range(2):
        if game.done:
            break
        action = json.dumps({"name": "get_user_details", "arguments": {"user_id": "fake"}})
        game.step(action)

    assert game.done, "Should be done at max_steps"
    assert game.rewards[0] == 0.0, "No writes completed, reward should be 0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/hangook/games && conda run -n games python -m pytest tests/test_multistep_mid_convo.py -v -k "test_game"`
Expected: FAIL

- [ ] **Step 3: Implement the new game class**

Replace `RealisticMultiStepGame` in `multistep_task_game.py`:

```python
class RealisticMultiStepGame:
    """Mid-conversation multi-step task game in tau2-bench format.

    Pre-fills conversation with auth + lookups (Phase 1-2).
    Model produces tool calls freely in Phase 3 (writes + optional lookups).
    No LLM user simulator needed.
    """

    supports_structured_messages = True

    def __init__(self, max_steps: int = 15,
                 user_client=None,  # Accepted but ignored for backward compat
                 domain: Optional[str] = None):
        self._max_steps = max_steps
        self._domain = domain

        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        self._scenario: Optional[MultiStepScenario] = None
        self._tools: Optional[ToolExecutor] = None
        self._conversation: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._call_id_counter: int = 0
        self._last_call_key: Optional[str] = None
        self._repeat_count: int = 0

    def reset(self, seed: int) -> None:
        self._scenario = generate_scenario(seed, domain=self._domain)
        self._tools = ToolExecutor(
            self._scenario.domain,
            copy.deepcopy(self._scenario.db),
        )
        self._conversation = copy.deepcopy(self._scenario.messages)
        self._tool_calls = []
        self._step_count = 0
        # Start call_id counter after pre-filled messages
        self._call_id_counter = sum(
            1 for m in self._conversation
            if m.get("tool_calls")
        )
        self._last_call_key = None
        self._repeat_count = 0

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def get_system_prompt(self) -> str:
        if self._scenario and self._scenario.domain == "retail":
            policy = RETAIL_POLICY
        else:
            policy = AIRLINE_POLICY
        return (
            "<instructions>\n"
            f"{AGENT_INSTRUCTION}\n"
            "</instructions>\n"
            "<policy>\n"
            f"{policy}\n"
            "</policy>"
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._scenario and self._scenario.domain == "retail":
            return RETAIL_TOOL_SCHEMAS
        return AIRLINE_TOOL_SCHEMAS

    def get_messages(self) -> List[Dict[str, Any]]:
        return list(self._conversation)

    def step(self, action: Optional[str]) -> None:
        if self.done:
            return

        self._step_count += 1

        if action is None:
            self._finalize(0.0, "No action provided")
            return

        tool_call = _parse_tool_call(action)
        if tool_call is None:
            self._finalize(0.0, "Invalid JSON format")
            self.invalid_player = 0
            return

        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        # Text response to user -> end game
        if tool_name == "respond_to_user" or tool_name == "send_message":
            reward, reason = compute_reward(self._tool_calls, self._scenario.operations)
            self._finalize(reward, reason)
            return

        # Transfer -> end game
        if tool_name == "transfer_to_human_agents":
            reward, reason = compute_reward(self._tool_calls, self._scenario.operations)
            self._finalize(reward, f"Transferred. {reason}")
            return

        # Loop detection
        call_key = json.dumps(tool_call, sort_keys=True)
        if call_key == self._last_call_key:
            self._repeat_count += 1
            if self._repeat_count >= 3:
                reward, reason = compute_reward(self._tool_calls, self._scenario.operations)
                self._finalize(reward, f"Loop detected. {reason}")
                return
        else:
            self._last_call_key = call_key
            self._repeat_count = 0

        # Track write calls for reward
        self._tool_calls.append({"name": tool_name, "arguments": tool_args})

        # Execute tool via ToolExecutor
        cid = f"call_{self._call_id_counter:04d}"
        self._call_id_counter += 1

        self._conversation.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args),
                },
            }],
        })

        try:
            result = self._tools.execute(tool_name, tool_args)
        except Exception as e:
            result = json.dumps({"error": str(e)})

        self._conversation.append({
            "role": "tool",
            "content": result,
            "tool_call_id": cid,
        })

        # Max steps check
        if self._step_count >= self._max_steps:
            reward, reason = compute_reward(self._tool_calls, self._scenario.operations)
            self._finalize(reward, f"Max steps. {reason}")

    def _finalize(self, reward: float, reason: str) -> None:
        self.done = True
        self.rewards = {0: reward}
        self._reason = reason

    def observe(self, player_id: int) -> str:
        return "This game uses tool-calling interface, not observe()."

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def get_summary(self) -> Dict[str, Any]:
        return {
            "n_ops": len(self._scenario.operations) if self._scenario else 0,
            "domain": self._scenario.domain if self._scenario else "",
            "steps": self._step_count,
            "reason": getattr(self, "_reason", ""),
        }
```

Key changes from old class:
- `__init__` accepts `user_client` but ignores it (backward compat with game_registry)
- `reset` builds pre-filled conversation, no LLM user
- `get_messages` returns the growing conversation (pre-filled + model's tool calls + results)
- `step` executes tool calls directly (no user sim interaction)
- `max_steps` default reduced from 40 to 15 (n_ops + buffer)
- `respond_to_user` ends game (no user sim to respond)

- [ ] **Step 4: Run game class tests**

Run: `cd /home/ubuntu/hangook/games && conda run -n games python -m pytest tests/test_multistep_mid_convo.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit game class**

```bash
git add multistep_task_game.py tests/test_multistep_mid_convo.py
git commit -m "feat: rewrite multistep game class for mid-conversation format

No LLM user simulator needed. Model produces tool calls freely.
Matches tau2-bench eval loop exactly. max_steps reduced to 15."
```

---

### Task 3: Update game_registry and run integration test

**Files:**
- Modify: `game_registry.py:229-230` (update make_multistep)

- [ ] **Step 1: Write integration test**

Add to `tests/test_multistep_mid_convo.py`:

```python
def test_game_registry_integration():
    """Game works through the registry without user_client."""
    from game_registry import get_game_spec
    spec = get_game_spec("multistep_task")
    game = spec.make_env(domain="airline")
    game.reset(42)

    assert not game.done
    assert game.get_system_prompt()
    assert len(game.get_messages()) >= 6
    assert len(game.get_tool_schemas()) > 0
```

- [ ] **Step 2: Update game_registry.py**

Change `make_multistep`:
```python
def make_multistep(user_client=None, domain=None) -> RealisticMultiStepGame:
    return RealisticMultiStepGame(max_steps=15, domain=domain)
```

- [ ] **Step 3: Run all tests**

Run: `cd /home/ubuntu/hangook/games && conda run -n games python -m pytest tests/test_multistep_mid_convo.py -v`
Expected: ALL PASS

- [ ] **Step 4: Run the self-test in multistep_task_game.py**

Run: `cd /home/ubuntu/hangook/games && conda run -n games python multistep_task_game.py`
Expected: Generates scenarios for both domains, shows pre-filled messages and operations.

- [ ] **Step 5: Commit**

```bash
git add game_registry.py tests/test_multistep_mid_convo.py
git commit -m "feat: update game_registry for mid-conversation multistep game

No longer requires user_client. max_steps=15."
```

---

### Task 4: Verify distributions and format match

**Files:**
- Create: `tests/test_multistep_distributions.py`

- [ ] **Step 1: Write distribution verification tests**

```python
"""Verify scenario distributions match eval failure data."""
import json
import sys
import os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_airline_nops_distribution():
    """Airline n-ops matches eval: [2,3,4,5] @ [50,29,14,7]."""
    from multistep_task_game import generate_scenario
    counts = Counter()
    N = 2000
    for seed in range(N):
        s = generate_scenario(seed, domain="airline")
        counts[len(s.operations)] += 1

    total = sum(counts.values())
    for n_ops, expected_pct in [(2, 50), (3, 29), (4, 14), (5, 7)]:
        actual_pct = counts[n_ops] / total * 100
        assert abs(actual_pct - expected_pct) < 5, \
            f"Airline {n_ops}-ops: {actual_pct:.0f}% (expected ~{expected_pct}%)"


def test_retail_nops_distribution():
    """Retail n-ops matches eval: [2,3,4,5] @ [48,39,10,3]."""
    from multistep_task_game import generate_scenario
    counts = Counter()
    N = 2000
    for seed in range(N):
        s = generate_scenario(seed, domain="retail")
        counts[len(s.operations)] += 1

    total = sum(counts.values())
    for n_ops, expected_pct in [(2, 48), (3, 39), (4, 10), (5, 3)]:
        actual_pct = counts[n_ops] / total * 100
        assert abs(actual_pct - expected_pct) < 5, \
            f"Retail {n_ops}-ops: {actual_pct:.0f}% (expected ~{expected_pct}%)"


def test_retail_tool_distribution():
    """Retail tool weights include modify_user_address."""
    from multistep_task_game import generate_scenario
    tool_counts = Counter()
    N = 2000
    for seed in range(N):
        s = generate_scenario(seed, domain="retail")
        for op in s.operations:
            tool_counts[op.tool_name] += 1

    total = sum(tool_counts.values())
    assert "modify_user_address" in tool_counts, "modify_user_address must appear"
    mua_pct = tool_counts["modify_user_address"] / total * 100
    assert 3 < mua_pct < 12, f"modify_user_address: {mua_pct:.1f}% (expected ~7%)"


def test_message_format_matches_eval():
    """Pre-filled messages use exact tau2-bench OpenAI format."""
    from multistep_task_game import generate_scenario
    s = generate_scenario(42, domain="retail")

    for m in s.messages:
        assert "role" in m, f"Message missing 'role': {m}"

        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                assert "id" in tc, "tool_call missing id"
                assert "type" in tc, "tool_call missing type"
                assert tc["type"] == "function"
                assert "function" in tc
                assert "name" in tc["function"]
                assert "arguments" in tc["function"]
                # arguments must be a JSON string (not dict)
                assert isinstance(tc["function"]["arguments"], str), \
                    f"arguments should be JSON string, got {type(tc['function']['arguments'])}"

        elif m["role"] == "tool":
            assert "tool_call_id" in m, "tool result missing tool_call_id"
            assert "content" in m, "tool result missing content"

        elif m["role"] in ("user", "assistant"):
            assert "content" in m, f"Message missing content: {m}"


def test_system_prompt_matches_eval():
    """System prompt matches tau2-bench exactly."""
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame(domain="airline")
    game.reset(42)

    prompt = game.get_system_prompt()
    assert "<instructions>" in prompt
    assert "In each turn you can either:" in prompt
    assert "- Send a message to the user." in prompt
    assert "- Make a tool call." in prompt
    assert "<policy>" in prompt
```

- [ ] **Step 2: Run distribution tests**

Run: `cd /home/ubuntu/hangook/games && conda run -n games python -m pytest tests/test_multistep_distributions.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_multistep_distributions.py
git commit -m "test: verify multistep distributions and format match eval"
```
