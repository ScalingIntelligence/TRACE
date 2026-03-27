# ToolSandbox Training Environment Proposal

## Failure Analysis Summary

From analyzing 129 base model trajectories on ToolSandbox, we identified 6 distinct failure patterns:

| Pattern | Count | Trainable? | Description |
|---------|-------|-----------|-------------|
| **Tool-then-communicate** | 26 | **Yes (high priority)** | Agent makes correct tool call but conversation ends without reporting result to user. Scores 0.5 (milestone 0 passes, milestone 1 fails). |
| **Error recovery chain** | 15 | **Yes** | Agent hits PermissionError/ConnectionError and either gives up or doesn't chain the fix (e.g., low battery → disable low battery → enable WiFi → retry). |
| **Zero-turn no response** | 15 | Unclear | Agent produces no response at all. May be user simulator issue. |
| **Partial multi-step** | 8 | **Yes** | Agent completes some milestones but not all in multi-tool scenarios. |
| **Tool argument errors** | 6 | **Yes** | Wrong parameters (e.g., wrong timestamp bounds, wrong phone format). |
| **Multi-turn coordination** | 5 | Moderate | Agent can't handle vague initial request + clarification flow. |

## Proposed Training Environment: Tool-Execute-Communicate (TEC)

### Why This Skill?

The **tool-then-communicate** pattern is the single largest fixable failure mode (26 scenarios, 20% of all tasks). The pattern is identical across all 26 cases:

1. User asks a question or requests an action
2. Agent correctly calls the right tool with right arguments ✓
3. Tool returns the correct result ✓
4. **Agent fails to send a response back to the user** ✗
5. Conversation ends → milestone 0 (tool execution) passes, milestone 1 (user communication) fails → score 0.5

This happens because the model generates only a tool call in its response and the framework treats that as the full turn. The model needs to learn: **after receiving a tool result, always generate a natural language response to the user**.

### Milestone Structure

Every ToolSandbox scenario has at least 2 milestones:
- **Milestone 0**: Database/state change (did the tool execute correctly?)
- **Milestone 1**: Communication (did the agent tell the user the result?)

The base model consistently passes milestone 0 but fails milestone 1. Training needs to reinforce the full loop.

### Environment Design

**Format**: Single-turn (user request → agent tool call → tool result → agent response to user)

This matches ToolSandbox's structure: the tools are Python functions executed in a sandbox, with OpenAI-format tool calling.

**Task types** (matching ToolSandbox's 5 tool categories):

1. **Contact lookup**: "What is X's phone number?" → search_contacts → "+1234567890" → "X's phone number is +1234567890"
2. **Settings query**: "Is my WiFi on?" → get_wifi_status → True → "Yes, your WiFi is currently on"
3. **Settings change**: "Turn off cellular" → set_cellular_status(on=False) → None → "Done, I've turned off your cellular service"
4. **Reminder creation**: "Remind me to buy milk at 3pm tomorrow" → datetime_info_to_timestamp + add_reminder → ID → "I've set a reminder for 'buy milk' at 3:00 PM tomorrow"
5. **Message search**: "What did my last text say?" → get_current_timestamp + search_messages → message content → "Your last message was from X saying '...'"
6. **Information lookup**: "How many days until Christmas?" → search_holiday + get_current_timestamp + timestamp_diff → days → "Christmas is 274 days away"

**Difficulty calibration for GRPO**:

To ensure the reward distribution has variance (some pass, some fail within a group):

- **Easy (30%)**: Simple single-tool tasks — lookup contact, check setting, add contact. Most rollouts should succeed (reward ~0.7-1.0).
- **Medium (50%)**: Tasks requiring 2 tools — search + communicate result, or timestamp conversion + reminder creation. Moderate success rate (~0.4-0.6).
- **Hard (20%)**: Tasks requiring 3+ tools with intermediate reasoning — find most recent message sender's phone number (search_messages → extract sender_id → search_contacts), or compute time until a holiday (search_holiday → get_timestamp → timestamp_diff → communicate). Low success rate (~0.1-0.3).

This distribution ensures GRPO gets meaningful advantage scores: within a group of N rollouts, some will complete the full loop (tool call + communication) and some won't.

### Reward Function

Binary reward with two components matching ToolSandbox's milestone evaluation:

```
reward = 0.5 * tool_correct + 0.5 * communication_correct
```

- **tool_correct** (0 or 1): Did the agent call the correct tool with correct arguments? Verified by comparing the tool call against expected action.
- **communication_correct** (0 or 1): After receiving the tool result, did the agent generate a response to the user containing the key information? Verified by checking that the agent's final message contains the expected value (phone number, status, reminder confirmation, etc.).

This means:
- Tool call only (no response) → reward 0.5
- Wrong tool call + response → reward 0.0 or 0.5
- Correct tool call + correct response → reward 1.0

The 0.5 partial reward for tool-only is important — it tells GRPO "you're halfway there, now learn to also communicate."

### Tool Interface

Use the exact same tool schemas as ToolSandbox to maximize transfer:

```python
# Contact tools
def add_contact(name: str, phone_number: str, relationship: str = None) -> str
def search_contacts(name: str = None, phone_number: str = None, relationship: str = None) -> list
def modify_contact(person_id: str, name: str = None, phone_number: str = None) -> None
def remove_contact(person_id: str) -> None

# Setting tools
def get_wifi_status() -> bool
def set_wifi_status(on: bool) -> None
def get_cellular_service_status() -> bool
def set_cellular_service_status(on: bool) -> None
def get_location_service_status() -> bool
def set_location_service_status(on: bool) -> None
def get_low_battery_mode_status() -> bool
def set_low_battery_mode_status(on: bool) -> None
def get_current_location() -> dict

# Messaging tools
def send_message_with_phone_number(phone_number: str, content: str) -> str
def search_messages(content: str = None, sender_phone_number: str = None, ...) -> list

# Reminder tools
def add_reminder(content: str, reminder_timestamp: int, ...) -> str
def search_reminder(content: str = None, ...) -> list
def modify_reminder(reminder_id: str, ...) -> None
def remove_reminder(reminder_id: str) -> None

# Utility tools
def get_current_timestamp() -> float
def datetime_info_to_timestamp(year, month, day, hour, minute, second) -> float
def timestamp_to_datetime_info(timestamp: float) -> dict
def search_holiday(holiday_name: str, year: int) -> float
```

### Database State

Each episode starts with a pre-populated state (like ToolSandbox):
- 3-5 contacts with names, phone numbers, relationships
- 2-4 messages with timestamps and content
- 1-3 reminders with content and timestamps
- Random settings state (WiFi on/off, cellular on/off, low battery on/off)

This is generated procedurally with random values each episode.

### Conversation Format

Uses OpenAI chat format with tool calling, matching vLLM's `--enable-auto-tool-choice --tool-call-parser hermes`:

```json
{"role": "system", "content": "You are a helpful phone assistant. Use the provided tools to help the user. After using a tool, always tell the user the result."}
{"role": "user", "content": "What is Homer's phone number?"}
{"role": "assistant", "tool_calls": [{"function": {"name": "search_contacts", "arguments": {"name": "Homer"}}}]}
{"role": "tool", "content": "[{\"name\": \"Homer S\", \"phone_number\": \"+10000000000\"}]"}
{"role": "assistant", "content": "Homer S's phone number is +10000000000."}
```

The reward checks both the tool call (step 4) and the final assistant message (step 5).

### Expected Impact

If this environment successfully trains the model to complete the tool→communicate loop:

- **26 scenarios at 0.5 → ~1.0**: Direct improvement on the exact failure pattern
- **Estimated score lift**: 0.430 → ~0.530 (26 × 0.5 improvement / 129 scenarios = +0.10)
- **Transfer to tau2-bench**: The communicate-after-tool pattern also applies to tau2-bench's COMMUNICATE score, where the model sometimes takes the right action but fails to state specific values

### Implementation Notes

- Use ToolSandbox's actual tool implementations as a starting point (copy from `tool_sandbox/tools/`)
- Database can be simplified (in-memory dicts instead of polars DataFrames) for faster rollout generation
- System prompt should match ToolSandbox's: "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
- Tool call format must match vLLM's hermes parser output for zero transfer gap
