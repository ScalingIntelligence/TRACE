# TEC Environment v2 Design — Based on Trajectory Analysis

## Root Cause Analysis (110 failed scenarios)

| Root Cause | Count | Single Skill? | Trainable? |
|-----------|-------|---------------|-----------|
| **COMPLETE_FAILURE** | 26 | Mixed | Partially |
| **TOOL_OK_NO_COMMUNICATE** | 23 | **YES — single skill** | **YES** |
| **PARTIAL_SUCCESS** | 21 | Mixed | Partially |
| **ZERO_TURN** | 15 | No (user sim issue) | No |
| **ERROR_RECOVERY** | 13 | **YES — single skill** | **YES** |
| **PARTIAL_CHAIN** | 12 | Mixed | Partially |

## The Two Clean, Trainable Skills

### Skill A: Tool-then-Communicate (23 scenarios, 0.5 → 1.0)

**What happens**: Agent calls the correct tool, gets the correct result, but the conversation ENDS without the agent telling the user the answer.

**Examples**:
- `get_wifi`: User asks "Is my wifi on?" → Agent calls `get_wifi_status()` → Returns `True` → **conversation ends** (no "Yes, your WiFi is on")
- `search_phone_number_with_name`: User asks "What is Homer S's phone number?" → Agent calls `search_contacts(name="Homer S")` → Returns phone number → **conversation ends**
- `cellular_off`: User says "Turn off cellular" → Agent calls `set_cellular_service_status(on=False)` → Returns `None` → **conversation ends** (no "Done, cellular is off")

**This is a SINGLE, clean skill**: After receiving a tool result, generate a text response to the user.

**Current score**: All 23 get exactly 0.50 (milestone 0 passes, milestone 1 fails).

### Skill B: Error Recovery Chain (13+ scenarios, 0.0 → 0.67+)

**What happens**: Agent calls a tool, gets a `PermissionError`, and STOPS instead of diagnosing and fixing the prerequisite.

**Examples**:
- `turn_on_wifi_low_battery_mode`: User says "Turn on wifi" → Agent calls `set_wifi_status(on=True)` → `PermissionError: Wifi cannot be turned on in low battery mode` → **agent stops**
- The correct chain: error → `get_low_battery_mode_status()` → True → `set_low_battery_mode_status(on=False)` → `set_wifi_status(on=True)` → communicate success

**This is a SINGLE, clean skill**: When a tool returns an error mentioning a prerequisite, diagnose and fix it before retrying.

## Environment Design

### CRITICAL: Format must match ToolSandbox EXACTLY

- Same tool schemas (same function names, same parameter names)
- Same error messages (`PermissionError: Wifi cannot be turned on in low battery mode`)
- Same tool result formats (Python repr strings, JSON lists)
- Same system prompt ("Don't make assumptions about what values to plug into functions")
- Only the DATA (contacts, settings state) is synthetic

### Option 1: Single Environment with Both Skills (recommended)

Scenario distribution:
- 40% Tool-then-Communicate (simple lookup/action → must respond to user)
- 40% Error Recovery Chain (blocked action → diagnose → fix → retry → communicate)
- 20% Combined (error recovery AND must communicate result)

### Option 2: Two Separate Environments

**Env A — communicate_after_tool**:
- Single-turn: user asks → agent calls tool → tool returns → agent MUST respond
- Reward: 1.0 if agent responds with correct info, 0.5 if tool only, 0.0 if wrong tool

**Env B — error_recovery**:
- Multi-step: user requests → agent tries → gets error → must fix prerequisite → retry → succeed → communicate
- Reward: 1.0 if full chain completed, partial credit for each step

## Key Changes from v1

1. **Remove complex multi-tool scenarios** (temporal reasoning, batch modify, find_sender_name) — these combine multiple skills and the model already does well on them. Focus on the TWO clean skills.

2. **Make Skill A scenarios exactly match the 23 ToolSandbox failures**: same tool functions, same user prompts, same expected behavior.

3. **Make Skill B scenarios exactly match the 13 error recovery failures**: same error messages, same recovery chain logic.

4. **DO NOT change the system prompt or tool schemas** — format must be identical to ToolSandbox.
