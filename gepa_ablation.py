"""
GEPA Ablation: Skill Prompt Optimization vs LoRA Training

Ablation: Is skill knowledge better encoded in optimized prompt text or in weights?

Uses GEPA's optimize_anything to find optimal per-skill instruction prompts
for the base model (no LoRA), optimized on the same synthetic environments
used to train LoRA adapters. Both methods transfer cold to tau2-bench.

Controlled variables:
  - Same skill decomposition (tool_calling, precondition, structured_data, multistep)
  - Same synthetic environments for optimization/training
  - Same base model (Qwen3-30B-A3B-Instruct-2507)
  - Same eval (tau2-bench, unseen by both)

Independent variable:
  - GEPA optimizes text injected into the system prompt
  - GRPO optimizes LoRA adapter weights

Usage:
    # Optimize one skill (deprecated envs — no user LLM needed):
    python gepa_ablation.py \
        --skill structured_data \
        --base-url http://localhost:8080/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --reflection-base-url http://localhost:9000/v1 \
        --reflection-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --max-metric-calls 300

    # Tool-calling / precondition envs (need separate user LLM):
    python gepa_ablation.py \
        --skill precondition \
        --base-url http://localhost:8080/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --user-base-url http://localhost:9000/v1 \
        --user-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --reflection-base-url http://localhost:9000/v1 \
        --reflection-model Qwen/Qwen3-30B-A3B-Instruct-2507

    # Run all skills:
    for skill in tool_calling precondition structured_data multistep; do
        python gepa_ablation.py --skill $skill \
            --base-url http://localhost:8080/v1 \
            --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
            --reflection-base-url http://localhost:9000/v1 \
            --reflection-model Qwen/Qwen3-30B-A3B-Instruct-2507 &
    done
"""

import sys
import os
import json
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add GEPA to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "gepa" / "src"))

import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    MergeConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)
from gepa.lm import LM
from gepa.core.result import GEPAResult

from game_registry import get_game_spec


# ===========================================================================
# Per-skill tau2-bench task IDs from scaling_skills.json
# These are the tasks the base model fails on due to lacking each skill.
# Format: "A21" = airline seed 21, "R5" = retail seed 5.
# ===========================================================================

SKILL_TASK_IDS = {
    "tool_calling": [  # S2: Precise tool calling (22 tasks)
        "A8", "A17", "A20", "A24", "A25", "A29", "A30", "A33",
        "R1", "R5", "R6", "R7", "R8", "R9", "R29", "R49",
        "R52", "R60", "R64", "R79", "R91", "R93",
    ],
    "structured_data": [  # S1: Structured data reasoning (3 tasks)
        "A21", "A42", "A44",
    ],
    "multistep": [  # S3: Multi-step task completion (15 tasks)
        "A12", "A14", "A18", "A22", "A23",
        "R3", "R4", "R16", "R20", "R28", "R33", "R34", "R59", "R92", "R100",
    ],
    "precondition": [  # S4: Precondition verification (20 tasks)
        "A7", "A13", "A15", "A16", "A19", "A26", "A31", "A32",
        "A35", "A37", "A39", "A41", "A43", "A47",
        "R10", "R57", "R74", "R82", "R83", "R90",
    ],
}


def _task_ids_to_valset(task_ids: List[str]) -> List[Dict]:
    """Convert task ID strings ('A21', 'R5') to valset dicts."""
    valset = []
    for tid in task_ids:
        domain = "airline" if tid.startswith("A") else "retail"
        seed = int(tid[1:])
        valset.append({"seed": seed, "env": "tau2_bench", "domain": domain})
    return valset


# ===========================================================================
# Per-skill ToolSandbox failed scenario names → seed indices.
# Seeds are indices into the sorted base scenario list (129 scenarios).
# ===========================================================================

TOOLSANDBOX_FAILED_SCENARIOS = {
    "ts_multiturn": [  # Multi-turn clarification failures (21 scenarios)
        "add_reminder_content_and_date_and_time_multiple_user_turn",
        "add_reminder_content_and_week_delta_and_time_and_location_multiple_user_turn",
        "add_reminder_content_and_week_delta_and_time_multiple_user_turn",
        "add_reminder_content_and_weekday_delta_and_time_multiple_user_turn",
        "find_days_till_holiday_multiple_user_turn",
        "find_days_till_holiday_wifi_off_multiple_user_turn",
        "find_distance_with_location_name_multiple_user_turn",
        "find_temperature_f_with_location_and_time_diff_multiple_user_turn",
        "find_temperature_f_with_location_and_time_diff_wifi_off_multiple_user_turn",
        "modify_contact_with_message_recency_multiple_user_turn",
        "modify_contact_with_message_recency_multiple_user_turn_alt",
        "remove_contact_by_phone_multiple_user_turn",
        "remove_contact_by_phone_multiple_user_turn_alt",
        "search_message_with_recency_latest_multiple_user_turn",
        "search_message_with_recency_latest_multiple_user_turn_alt",
        "search_message_with_recency_oldest_multiple_user_turn",
        "search_message_with_recency_oldest_multiple_user_turn_alt",
        "send_message_with_contact_content_cellular_off_multiple_user_turn",
        "send_message_with_contact_content_cellular_off_multiple_user_turn_alt",
        "update_contact_relationship_with_relationship_multiple_user_turn",
        "update_contact_relationship_with_relationship_twice_multiple_user_turn",
    ],
    "ts_tec": [  # TEC / error recovery failures (9 scenarios)
        "add_reminder_content_and_week_delta_and_time_and_location_low_battery_mode_multiple_user_turn_alt",
        "find_current_city_low_battery_mode",
        "find_current_city_low_battery_mode_alt",
        "find_temperature_f_with_location_and_time_diff_low_battery_mode_multiple_user_turn",
        "turn_on_cellular_low_battery_mode",
        "turn_on_cellular_low_battery_mode_implicit",
        "turn_on_location_low_battery_mode",
        "turn_on_wifi_low_battery_mode",
        "turn_on_wifi_low_battery_mode_implicit",
    ],
}

# Scenario name → seed index (built lazily since it requires loading toolsandbox data)
_toolsandbox_name_to_seed: Optional[Dict[str, int]] = None


def _get_toolsandbox_name_to_seed() -> Dict[str, int]:
    """Lazily build scenario name → seed index mapping."""
    global _toolsandbox_name_to_seed
    if _toolsandbox_name_to_seed is None:
        import toolsandbox_game as tsg
        tsg._ensure_scenarios_loaded()
        _toolsandbox_name_to_seed = {
            name: idx for idx, name in enumerate(tsg._scenario_names)
        }
    return _toolsandbox_name_to_seed


def _toolsandbox_scenarios_to_valset(scenario_names: List[str]) -> List[Dict]:
    """Convert ToolSandbox scenario names to valset dicts."""
    name_to_seed = _get_toolsandbox_name_to_seed()
    valset = []
    for name in scenario_names:
        if name not in name_to_seed:
            print(f"WARNING: scenario '{name}' not found in toolsandbox, skipping")
            continue
        valset.append({
            "seed": name_to_seed[name],
            "env": "toolsandbox",
            "scenario": name,
        })
    return valset


# ===========================================================================
# Routing descriptions from the orchestrator config — these are what the
# classifier sees to decide which skill to route to. They describe WHEN
# to use a skill, not HOW to execute it.
# ===========================================================================

ROUTING_DESCRIPTIONS = {
    "tool_calling": (
        "Single-action execution. Use when the customer states a clear action: "
        "book a flight, cancel a reservation, process a refund, update baggage, "
        "change a passenger name, or modify payment."
    ),
    "structured_data": (
        "Data lookup, comparison, and computation. Use when the task requires "
        "searching for options, comparing values, or computing results: finding "
        "the cheapest or best-matching flight, upgrading or downgrading cabin "
        "class, calculating baggage allowances, checking flight delay status, "
        "or choosing which of several bookings to act on."
    ),
    "multistep": (
        "Multiple write operations. Use ONLY when the customer lists multiple "
        "independent tasks in one request."
    ),
    "precondition": (
        "Policy compliance and refusal decisions. Use when the action may need "
        "to be refused or its eligibility verified: whether a booking can be "
        "replicated from a past trip, whether ticket restrictions prevent the "
        "requested change, refund eligibility based on insurance or ticket type, "
        "or verifying a prior agent's commitment."
    ),
    "ts_multiturn": (
        "Multi-turn clarification. Use when the user's request is incomplete or "
        "ambiguous and requires asking follow-up questions before taking action."
    ),
    "ts_tec": (
        "Error recovery. Use when a tool call fails with a PermissionError due "
        "to low battery mode or disabled services, requiring automatic fix and retry."
    ),
}


# ===========================================================================
# Skill configurations
# ===========================================================================

SKILL_CONFIG = {
    "tool_calling": {
        "env_name": "tau_tool_calling",
        "env_type": "tool_calling",
        "needs_user_client": True,
        "seed_prompt": (
            "When calling tools, you MUST copy argument values exactly as they "
            "appear in prior tool outputs or the user's message. Never paraphrase, "
            "abbreviate, or translate values. Use the exact string for airport codes, "
            "reservation IDs, flight numbers, and payment method IDs. If a tool "
            "returned 'HAM', use 'HAM' — not 'Hamburg'. If the user said reservation "
            "'ABC123', use 'ABC123' verbatim."
        ),
        "objective": (
            "Optimize the instruction text to maximize task success rate on "
            "customer service tool-calling tasks. The model must select the correct "
            "tool and provide exact argument values copied from prior context. "
            "A score of 1.0 means the database state matches expected AND the agent "
            "communicated correctly. 0.3 means DB correct but communication incomplete. "
            "0.1 means right tools called but DB mismatch. 0.0 means total failure."
        ),
        "background": (
            "Domain: airline and retail customer service. The agent calls tools like "
            "update_reservation_flights, calculate_baggage, book_reservation, etc. "
            "Common failure modes:\n"
            "- Paraphrasing airport codes ('Hamburg' instead of 'HAM')\n"
            "- Using wrong reservation or flight IDs\n"
            "- Incorrect payment method IDs\n"
            "- Adding extra arguments not present in user request\n"
            "- Failing to communicate results back to user after tool call\n"
            "- On refusal tasks: calling write tools when policy prohibits the action\n"
            "The instruction text is appended to the system prompt inside "
            "<skill_instructions> tags. The agent also sees tool schemas, policy "
            "documents, and multi-turn conversation with a simulated user."
        ),
    },
    "precondition": {
        "env_name": "precondition_check",
        "env_type": "tool_calling",
        "needs_user_client": True,
        "seed_prompt": (
            "Before executing ANY state-changing action (cancel_reservation, "
            "update_reservation_flights, book_reservation, send_certificate, etc.), "
            "you MUST verify all preconditions by checking the airline policy. "
            "If any precondition fails, REFUSE the request: explain which policy "
            "rule prevents it and do NOT call any write tools for the rest of the "
            "conversation. Common preconditions to check:\n"
            "- Ticket class restrictions (basic economy cannot be cancelled)\n"
            "- Insurance requirements\n"
            "- Membership tier eligibility\n"
            "- Time-based restrictions"
        ),
        "objective": (
            "Optimize the instruction text to maximize policy compliance accuracy. "
            "~60% of tasks should be REFUSED under airline policy, ~40% are valid. "
            "Score 1.0 = correct decision (allow or refuse). Score 0.0 = wrong. "
            "On refusal tasks, calling ANY write tool is an automatic failure."
        ),
        "background": (
            "Domain: airline customer service with strict policy rules. The agent "
            "must check preconditions before acting. 20 scenarios (12 REFUSE + 8 ALLOW). "
            "Common failure modes:\n"
            "- Allowing cancellation of basic economy tickets (policy forbids)\n"
            "- Calling write tools before checking eligibility\n"
            "- Not recognizing that insurance is required for certain operations\n"
            "- Granting compensation when membership tier doesn't qualify\n"
            "- On valid tasks: refusing when action is actually permitted\n"
            "The instruction text is appended to the system prompt. The agent sees "
            "airline tools, full policy document, and converses with a simulated user."
        ),
    },
    "structured_data": {
        "env_name": "structured_data_reasoning_deprecated",
        "env_type": "observe",
        "needs_user_client": False,
        "seed_prompt": (
            "When answering questions about structured data:\n"
            "1. Read ALL the data carefully before answering any question.\n"
            "2. For filtering questions, apply each constraint one at a time and "
            "verify the filtered set.\n"
            "3. For 'cheapest' or 'nth cheapest' questions, sort the relevant items "
            "by price and pick the correct position.\n"
            "4. For sum/arithmetic questions, identify the exact items referenced and "
            "compute precisely.\n"
            "5. For baggage questions, look up the policy table using the exact "
            "membership tier and cabin class.\n"
            "6. Double-check your answer against the raw data before submitting.\n"
            "7. Use the exact item_id or flight_number string from the data."
        ),
        "objective": (
            "Optimize the instruction text to maximize the fraction of correctly "
            "answered questions (3 per episode). Score is 0/3, 1/3, 2/3, or 3/3. "
            "The model receives JSON data and must answer questions about it."
        ),
        "background": (
            "Single-turn task. The model receives a JSON dataset with 10-15 items "
            "(products, flights, or reservation/baggage info) and 3 questions. "
            "Question types: find cheapest with constraints, nth-cheapest, max/min "
            "attribute, compute price sums, count items matching criteria, compare "
            "prices, compute non-free baggage from policy table.\n"
            "Common failure modes:\n"
            "- Wrong filtering (missing a constraint or applying it incorrectly)\n"
            "- Off-by-one in nth-cheapest sorting\n"
            "- Arithmetic errors in sums\n"
            "- Selecting item name instead of item_id\n"
            "- Misreading the baggage allowance table\n"
            "The instruction text is injected into the system prompt. The model "
            "sees the data in <data> tags and must respond with a JSON submit_answers "
            "tool call."
        ),
    },
    "multistep": {
        "env_name": "multistep_task_deprecated",
        "env_type": "observe",
        "needs_user_client": False,
        "seed_prompt": (
            "You must complete ALL requested operations. Follow this protocol:\n"
            "1. Authenticate first: find_user_id_by_name_zip, then get_user_details.\n"
            "2. Before acting on any order, call get_order_details to discover item IDs.\n"
            "3. Execute operations ONE AT A TIME in order.\n"
            "4. Use the correct tool for each order status: modify/cancel for pending, "
            "return/exchange for delivered.\n"
            "5. Never call the same write operation on the same order twice.\n"
            "6. After completing ALL operations, call respond_to_user to confirm.\n"
            "7. If new tasks are revealed mid-conversation, complete those too."
        ),
        "objective": (
            "Optimize the instruction text to maximize task completion rate. "
            "5-8 operations per episode. Score = 0.4 completion bonus + 0.6 * "
            "(completed / total). Penalties for one-shot violations (repeating "
            "write ops on same order) and wrong tool type."
        ),
        "background": (
            "Multi-turn retail customer service. The model authenticates a user, "
            "then executes 5-8 operations: cancel pending orders, modify items, "
            "change addresses, return/exchange delivered items. Information is gated: "
            "item IDs only discovered via get_order_details, payment method IDs via "
            "get_user_details. Mid-conversation, incremental tasks may be revealed.\n"
            "Common failure modes:\n"
            "- Skipping operations (not completing all tasks)\n"
            "- Calling exchange on pending orders (wrong tool type)\n"
            "- Repeating modify/return/exchange on same order (one-shot violation)\n"
            "- Guessing item IDs instead of looking them up\n"
            "- Calling respond_to_user before all operations complete\n"
            "The instruction text is injected into the system prompt. Each turn, "
            "the model sees the full conversation history and available tools."
        ),
    },
    "ts_multiturn": {
        "env_name": "toolsandbox_multiturn",
        "env_type": "tool_calling",
        "needs_user_client": False,
        "seed_prompt": (
            "When a user's request is incomplete or ambiguous, you MUST ask "
            "clarifying questions before taking action.\n"
            "1. If the user mentions a contact by name, call search_contacts first "
            "to get their phone number — never ask the user for it directly.\n"
            "2. If the user wants to set a reminder but doesn't specify the time, "
            "date, or content, ask for the missing details.\n"
            "3. After completing any tool call, communicate the result back to the "
            "user clearly.\n"
            "4. If the user mentions a relative date (e.g., 'next Tuesday'), use "
            "get_current_timestamp and timestamp_to_datetime_info to compute it."
        ),
        "objective": (
            "Optimize the instruction text to maximize task success on multi-turn "
            "ToolSandbox scenarios where the agent must ask clarifying questions, "
            "look up contacts, and communicate results. Score combines: did the "
            "agent ask for clarification (0.6 weight), call correct tools (0.2), "
            "and communicate results (0.2)."
        ),
        "background": (
            "Domain: smartphone assistant (contacts, reminders, messaging, weather, "
            "settings). The agent has 27 tools. Scenarios require multi-turn "
            "interaction where the user provides information incrementally.\n"
            "Common failure modes:\n"
            "- Acting without asking for missing information (e.g., adding reminder "
            "without asking time)\n"
            "- Asking the user for a phone number instead of looking it up via "
            "search_contacts\n"
            "- Not communicating tool results back to the user\n"
            "- Incorrect datetime computation (guessing instead of using tools)\n"
            "The instruction text is appended to the system prompt."
        ),
    },
    "ts_tec": {
        "env_name": "tec_v2",
        "env_type": "tool_calling",
        "needs_user_client": False,
        "seed_prompt": (
            "If a tool call returns a PermissionError, do NOT report the error to "
            "the user. Instead:\n"
            "1. Check if low_battery_mode is enabled by calling "
            "get_low_battery_mode_status.\n"
            "2. If low_battery_mode is on, turn it off with "
            "set_low_battery_mode_status(False).\n"
            "3. Then retry the original action.\n"
            "4. After successfully completing the action, communicate the result "
            "to the user.\n"
            "Common blocked services: wifi, cellular, location services — all are "
            "disabled when low_battery_mode is on."
        ),
        "objective": (
            "Optimize the instruction text to maximize error recovery success. "
            "The agent encounters PermissionError from tools blocked by low_battery_mode "
            "and must automatically diagnose and fix the issue, then retry. "
            "Score: 0.6 * action_success + 0.4 * communication."
        ),
        "background": (
            "Domain: smartphone assistant. When low_battery_mode is enabled, "
            "wifi/cellular/location services are blocked, causing PermissionError. "
            "The agent must: detect the error, disable low_battery_mode, retry.\n"
            "Common failure modes:\n"
            "- Reporting the PermissionError to the user instead of fixing it\n"
            "- Not recognizing that low_battery_mode blocks services\n"
            "- Forgetting to retry the original action after fixing\n"
            "- Not communicating the final result to the user\n"
            "The instruction text is appended to the system prompt."
        ),
    },
}


# ===========================================================================
# Lightweight vLLM client (matches collect_rollouts.py interface)
# ===========================================================================

import requests


class VLLMClient:
    """Thread-safe vLLM client for agent model inference."""

    def __init__(self, base_url: str, model: str, max_tokens: int = 1024,
                 temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_maxsize=128, pool_connections=128,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def generate_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """OpenAI-compatible tool-calling API."""
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": self.model,
            "messages": all_messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=120,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        return {
            "content": choice.get("content"),
            "tool_calls": choice.get("tool_calls"),
        }

    def generate_text(self, system_prompt: str, user_content: str) -> str:
        """Plain text generation for observe-based games."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ===========================================================================
# Episode runners (mirror collect_rollouts.py logic)
# ===========================================================================

def run_toolcall_episode(
    game, client: VLLMClient, seed: int, skill_prompt: str,
) -> Dict[str, Any]:
    """Run one tool-calling episode with skill prompt injected."""
    try:
        game.reset(seed)
    except Exception as e:
        return {"reward": 0.0, "reason": f"reset error: {e}", "steps": 0}

    system_prompt = game.get_system_prompt()
    system_prompt += (
        f"\n\n<skill_instructions>\n{skill_prompt}\n</skill_instructions>"
    )
    tools = game.get_tool_schemas()

    max_steps = getattr(game, "max_steps", getattr(game, "_max_steps", 30))
    step = 0
    while not game.done and step < max_steps:
        messages = game.get_messages()

        try:
            result = client.generate_with_tools(system_prompt, messages, tools)
        except Exception:
            # Finalize on API error so we still get partial reward
            if not game.done and hasattr(game, "_finalize_with_verification"):
                game._finalize_with_verification()
            break

        content = result.get("content")
        tool_calls = result.get("tool_calls")

        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "")
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            action = json.dumps({"name": name, "arguments": arguments})
        elif content:
            action = json.dumps({
                "name": "respond_to_user",
                "arguments": {"message": content},
            })
        else:
            action = json.dumps({
                "name": "respond_to_user",
                "arguments": {"message": "Let me help you with that."},
            })

        game.step(action)
        step += 1

    # Ensure finalization even if max_steps hit
    if not game.done and hasattr(game, "_finalize_with_verification"):
        game._finalize_with_verification()

    if hasattr(game, "get_summary"):
        return game.get_summary()
    # Fallback for games without get_summary (e.g., toolsandbox training envs)
    return {
        "reward": game.rewards.get(0, 0.0),
        "reason": "max_steps" if not game.done else "done",
        "steps": step,
    }


def run_observe_episode(
    game, game_spec, client: VLLMClient, seed: int, skill_prompt: str,
) -> Dict[str, Any]:
    """Run one observe-based episode with skill prompt injected."""
    try:
        game.reset(seed)
    except Exception as e:
        return {"reward": 0.0, "reason": f"reset error: {e}", "steps": 0}

    system_prompt = game_spec.system_prompt
    system_prompt += (
        f"\n\n<skill_instructions>\n{skill_prompt}\n</skill_instructions>"
    )

    turns_trace: List[Dict[str, str]] = []
    step = 0
    max_steps = getattr(game, "_max_steps", 30)

    while not game.done and step < max_steps:
        obs = game.observe(0)

        try:
            response = client.generate_text(system_prompt, obs)
        except Exception:
            game.step(None)
            break

        action = game_spec.extract_action(response, game.legal_actions())

        # Keep truncated trace for ASI (avoid huge context in reflection)
        turns_trace.append({
            "observation": obs[:500],
            "response": response[:500],
            "action": str(action)[:300] if action else "None",
        })

        game.step(action)
        step += 1

    summary = game.get_summary()
    summary["_turns_trace"] = turns_trace
    return summary


# ===========================================================================
# Checkpoint stopper — saves best prompt every N metric calls
# ===========================================================================

class CheckpointStopper:
    """StopperProtocol-compatible callback that saves the best prompt
    every ``interval`` metric calls. Never actually stops the optimization."""

    def __init__(self, run_dir: str, interval: int = 500,
                 str_candidate_key: str | None = None):
        self.run_dir = run_dir
        self.interval = interval
        self.str_candidate_key = str_candidate_key
        self._last_checkpoint_at = 0

    def __call__(self, gepa_state) -> bool:
        total = gepa_state.total_num_evals
        if total - self._last_checkpoint_at >= self.interval:
            self._last_checkpoint_at = total

            # Build result to extract best candidate
            result = GEPAResult.from_state(
                gepa_state,
                run_dir=self.run_dir,
                str_candidate_key=self.str_candidate_key,
            )
            best = result.best_candidate
            score = result.val_aggregate_scores[result.best_idx]

            ckpt_path = os.path.join(
                self.run_dir, f"best_prompt_at_{total}.txt",
            )
            with open(ckpt_path, "w") as f:
                f.write(best if isinstance(best, str) else json.dumps(best))

            print(
                f"\n[Checkpoint] {total} metric calls | "
                f"best val={score:.3f} | saved to {ckpt_path}"
            )

        return False  # never stop


# ===========================================================================
# GEPA evaluator factory
# ===========================================================================

def _build_synthetic_side_info(
    config: Dict, summary: Dict, seed: int,
) -> SideInfo:
    """Build rich ASI from a synthetic env episode for GEPA reflection."""
    reward = summary.get("reward", 0.0)
    reason = summary.get("reason", "")

    side_info: SideInfo = {
        "score": reward,
        "Seed": seed,
        "Reward": f"{reward:.2f}",
        "Reason": reason[:500],
    }

    if config["env_type"] == "tool_calling":
        tool_calls = summary.get("tool_calls", [])
        side_info["Tool Calls Made"] = json.dumps(
            tool_calls, indent=1,
        )[:2000]
        if "expected_actions" in summary:
            side_info["Expected Actions"] = json.dumps(
                summary["expected_actions"],
            )[:1000]
        if "is_refusal" in summary:
            side_info["Task Type"] = (
                "REFUSE" if summary["is_refusal"] else "ALLOW"
            )
        if "communicate_info" in summary:
            side_info["Expected Communication"] = str(
                summary["communicate_info"],
            )[:500]
    else:
        turns = summary.get("_turns_trace", [])
        if turns:
            last = turns[-1]
            side_info["Last Turn Response"] = last.get(
                "response", "",
            )[:1000]
            side_info["Last Turn Action"] = last.get(
                "action", "",
            )[:500]
        tool_calls = summary.get("tool_calls", [])
        if tool_calls:
            side_info["Tool Calls Made"] = json.dumps(
                tool_calls, indent=1,
            )[:2000]
        if "n_ops" in summary:
            side_info["Total Operations"] = summary["n_ops"]
            side_info["One-Shot Violations"] = summary.get(
                "one_shot_violations", 0,
            )
            side_info["Wrong Tool Type"] = summary.get(
                "wrong_tool_type", 0,
            )
        if "questions" in summary:
            side_info["Questions"] = json.dumps(
                summary["questions"],
            )[:1500]

    return side_info


def _build_tau2bench_side_info(summary: Dict, seed: int) -> SideInfo:
    """Build minimal ASI from a tau2-bench episode.

    Intentionally sparse: tau2-bench examples are used for GEPA's val set
    (candidate selection only), NOT for reflection. Keeping ASI minimal
    prevents instance-specific data from leaking into prompt proposals.
    """
    return {
        "score": summary.get("reward", 0.0),
        "Task ID": summary.get("task_id", f"seed-{seed}"),
        "Domain": summary.get("domain", ""),
        "Reward": f"{summary.get('reward', 0.0):.1f}",
        "Reason": summary.get("reason", "")[:200],
        "Steps": summary.get("steps", 0),
    }


def _build_toolsandbox_side_info(summary: Dict, seed: int) -> SideInfo:
    """Build minimal ASI from a ToolSandbox eval episode.

    Same rationale as tau2-bench: val set only, no trace leakage.
    """
    return {
        "score": summary.get("reward", 0.0),
        "Scenario": summary.get("scenario", f"seed-{seed}"),
        "Reward": f"{summary.get('reward', 0.0):.2f}",
        "Reason": summary.get("reason", "")[:200],
        "Steps": summary.get("steps", 0),
    }


def make_evaluator(skill_name: str, args: argparse.Namespace):
    """Build a GEPA-compatible evaluator for the given skill.

    Returns a callable(candidate, example) -> (score, side_info).
    Thread-safe: creates fresh game instances per call, shares HTTP clients.

    In oracle mode, the evaluator dispatches based on example["env"]:
      - "synthetic": run the skill's synthetic environment (for dataset/reflection)
      - "tau2_bench": run a tau2-bench task (for valset/selection)
      - "toolsandbox": run a ToolSandbox eval task (for valset/selection)
    In normal mode, all examples use the synthetic environment.
    """
    config = SKILL_CONFIG[skill_name]
    game_spec = get_game_spec(config["env_name"])

    # Shared HTTP clients (thread-safe via connection pooling)
    agent_client = VLLMClient(
        base_url=args.base_url,
        model=args.model,
        max_tokens=1024,
        temperature=args.temperature,
    )

    # User client — needed for tool-calling synthetic envs AND tau2-bench
    user_client = None
    if config["needs_user_client"] or getattr(args, "oracle", False):
        from adversarial_policy_game.llm_user import UserLLMClient
        user_client = UserLLMClient(
            base_url=args.user_base_url or args.base_url,
            model=args.user_model or args.model,
            max_tokens=256,
            temperature=0.7,
        )

    # Pre-load eval game specs if oracle mode
    tau2_game_specs = {}
    toolsandbox_game_spec = None
    is_oracle = getattr(args, "oracle", False)
    if is_oracle:
        if skill_name in SKILL_TASK_IDS:
            tau2_game_specs["airline"] = get_game_spec("tau2_bench_airline")
            tau2_game_specs["retail"] = get_game_spec("tau2_bench_retail")
        if skill_name in TOOLSANDBOX_FAILED_SCENARIOS:
            toolsandbox_game_spec = get_game_spec("toolsandbox")

    def evaluator(candidate: str, example: Dict) -> Tuple[float, SideInfo]:
        seed = example["seed"]
        env_type = example.get("env", "synthetic")

        try:
            if env_type == "tau2_bench":
                # --- tau2-bench evaluation (valset only) ---
                domain = example.get("domain", "airline")
                game = tau2_game_specs[domain].make_env(user_client=user_client)
                summary = run_toolcall_episode(
                    game, agent_client, seed, candidate,
                )
                reward = summary.get("reward", 0.0)
                side_info = _build_tau2bench_side_info(summary, seed)
                oa.log(
                    f"[tau2] Task {summary.get('task_id', seed)}: "
                    f"reward={reward:.1f} | {summary.get('reason', '')[:100]}"
                )
                return reward, side_info
            elif env_type == "toolsandbox":
                # --- ToolSandbox evaluation (valset only) ---
                game = toolsandbox_game_spec.make_env(user_client=user_client)
                summary = run_toolcall_episode(
                    game, agent_client, seed, candidate,
                )
                reward = summary.get("reward", 0.0)
                side_info = _build_toolsandbox_side_info(summary, seed)
                oa.log(
                    f"[toolsandbox] {summary.get('scenario', seed)}: "
                    f"reward={reward:.2f} | {summary.get('reason', '')[:100]}"
                )
                return reward, side_info
            else:
                # --- Synthetic env evaluation (dataset for reflection) ---
                if config["needs_user_client"]:
                    game = game_spec.make_env(user_client=user_client)
                else:
                    game = game_spec.make_env()

                if config["env_type"] == "tool_calling":
                    summary = run_toolcall_episode(
                        game, agent_client, seed, candidate,
                    )
                else:
                    summary = run_observe_episode(
                        game, game_spec, agent_client, seed, candidate,
                    )

                reward = summary.get("reward", 0.0)
                side_info = _build_synthetic_side_info(
                    config, summary, seed,
                )
                oa.log(
                    f"[synth] Seed {seed}: reward={reward:.2f} | "
                    f"{summary.get('reason', '')[:200]}"
                )
                return reward, side_info

        except Exception as e:
            oa.log(f"Seed {seed} ({env_type}): ERROR - {e}")
            return 0.0, {
                "error": str(e),
                "traceback": traceback.format_exc()[:1000],
            }

    return evaluator


# ===========================================================================
# Baseline evaluation
# ===========================================================================

def evaluate_baseline(
    skill_name: str,
    prompt: str,
    seeds: List[int],
    args: argparse.Namespace,
) -> float:
    """Evaluate a prompt on a set of seeds and return mean reward."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config = SKILL_CONFIG[skill_name]
    game_spec = get_game_spec(config["env_name"])

    agent_client = VLLMClient(
        base_url=args.base_url,
        model=args.model,
        max_tokens=1024,
        temperature=args.temperature,
    )

    user_client = None
    if config["needs_user_client"]:
        from adversarial_policy_game.llm_user import UserLLMClient
        user_client = UserLLMClient(
            base_url=args.user_base_url or args.base_url,
            model=args.user_model or args.model,
            max_tokens=256,
            temperature=0.7,
        )

    def _eval_seed(seed: int) -> float:
        try:
            if config["needs_user_client"]:
                game = game_spec.make_env(user_client=user_client)
            else:
                game = game_spec.make_env()

            if config["env_type"] == "tool_calling":
                summary = run_toolcall_episode(
                    game, agent_client, seed, prompt,
                )
            else:
                summary = run_observe_episode(
                    game, game_spec, agent_client, seed, prompt,
                )

            return summary.get("reward", 0.0)
        except Exception as e:
            print(f"  Seed {seed}: ERROR - {e}")
            return 0.0

    rewards = []
    max_workers = min(args.max_workers, len(seeds))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_seed, s): s for s in seeds}
        for future in as_completed(futures):
            rewards.append(future.result())

    return sum(rewards) / len(rewards) if rewards else 0.0


def evaluate_tau2bench(
    prompt: str,
    tasks: List[Dict],
    args: argparse.Namespace,
) -> Tuple[float, List[Dict]]:
    """Evaluate a prompt on tau2-bench tasks.

    Args:
        prompt: Skill instruction text to inject.
        tasks: List of {"seed": int, "domain": str} dicts.
        args: CLI args.

    Returns:
        (mean_reward, per_task_results)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tau2_specs = {
        "airline": get_game_spec("tau2_bench_airline"),
        "retail": get_game_spec("tau2_bench_retail"),
    }

    agent_client = VLLMClient(
        base_url=args.base_url,
        model=args.model,
        max_tokens=1024,
        temperature=args.temperature,
    )

    from adversarial_policy_game.llm_user import UserLLMClient
    user_client = UserLLMClient(
        base_url=args.user_base_url or args.base_url,
        model=args.user_model or args.model,
        max_tokens=256,
        temperature=0.7,
    )

    results = []

    def _eval_task(task: Dict) -> Dict:
        seed = task["seed"]
        domain = task["domain"]
        try:
            game = tau2_specs[domain].make_env(user_client=user_client)
            summary = run_toolcall_episode(game, agent_client, seed, prompt)
            return {
                "seed": seed,
                "domain": domain,
                "task_id": summary.get("task_id", f"{domain}-{seed}"),
                "reward": summary.get("reward", 0.0),
                "reason": summary.get("reason", ""),
                "steps": summary.get("steps", 0),
            }
        except Exception as e:
            print(f"  Task {domain}/{seed}: ERROR - {e}")
            return {
                "seed": seed, "domain": domain,
                "reward": 0.0, "error": str(e),
            }

    max_workers = min(args.max_workers, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_task, t): t for t in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    rewards = [r["reward"] for r in results]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return mean_reward, sorted(results, key=lambda r: (r["domain"], r["seed"]))


def evaluate_toolsandbox(
    prompt: str,
    tasks: List[Dict],
    args: argparse.Namespace,
) -> Tuple[float, List[Dict]]:
    """Evaluate a prompt on ToolSandbox tasks. Returns (mean_reward, per_task_results)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ts_game_spec = get_game_spec("toolsandbox")

    agent_client = VLLMClient(
        base_url=args.base_url,
        model=args.model,
        max_tokens=1024,
        temperature=args.temperature,
    )

    from adversarial_policy_game.llm_user import UserLLMClient
    user_client = UserLLMClient(
        base_url=args.user_base_url or args.base_url,
        model=args.user_model or args.model,
        max_tokens=256,
        temperature=0.7,
    )

    results = []

    def _eval_task(task: Dict) -> Dict:
        seed = task["seed"]
        scenario = task.get("scenario", f"seed-{seed}")
        try:
            game = ts_game_spec.make_env(user_client=user_client)
            summary = run_toolcall_episode(game, agent_client, seed, prompt)
            return {
                "seed": seed,
                "scenario": summary.get("scenario", scenario),
                "reward": summary.get("reward", 0.0),
                "reason": summary.get("reason", ""),
                "steps": summary.get("steps", 0),
            }
        except Exception as e:
            print(f"  Task {scenario}: ERROR - {e}")
            return {
                "seed": seed, "scenario": scenario,
                "reward": 0.0, "error": str(e),
            }

    max_workers = min(args.max_workers, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_task, t): t for t in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    rewards = [r["reward"] for r in results]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return mean_reward, sorted(results, key=lambda r: r["seed"])


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GEPA ablation: optimize skill prompts on synthetic environments",
    )
    # Skill selection
    parser.add_argument(
        "--skill", required=True, choices=list(SKILL_CONFIG.keys()),
        help="Which skill to optimize",
    )
    # Model serving
    parser.add_argument(
        "--base-url", required=True,
        help="vLLM base URL for agent model (e.g., http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--model", required=True,
        help="Agent model name (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)",
    )
    parser.add_argument(
        "--user-base-url", default=None,
        help="vLLM base URL for user simulator (tool-calling envs only). "
             "Defaults to --base-url.",
    )
    parser.add_argument(
        "--user-model", default=None,
        help="User simulator model name. Defaults to --model.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Agent model temperature (default: 0.0 for deterministic eval)",
    )
    # GEPA reflection LM (must be a vLLM-served model)
    parser.add_argument(
        "--reflection-model", required=True,
        help="Model name for GEPA reflection LLM "
             "(e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)",
    )
    parser.add_argument(
        "--reflection-base-url", required=True,
        help="vLLM base URL for reflection LLM "
             "(e.g., http://localhost:9000/v1)",
    )
    parser.add_argument(
        "--reflection-temperature", type=float, default=0.7,
        help="Temperature for reflection LLM (default: 0.7)",
    )
    parser.add_argument(
        "--reflection-max-tokens", type=int, default=4096,
        help="Max tokens for reflection LLM (default: 4096)",
    )
    parser.add_argument(
        "--max-metric-calls", type=int, default=300,
        help="Total GEPA evaluation budget (default: 300)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=16,
        help="Parallel evaluation workers (default: 16)",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=500,
        help="Save best prompt every N metric calls (default: 500). 0 to disable.",
    )
    parser.add_argument(
        "--reflection-minibatch-size", type=int, default=3,
        help="Examples shown per GEPA reflection step (default: 3)",
    )
    # Seed splits
    parser.add_argument(
        "--train-seeds", type=int, default=200,
        help="Number of training seeds (default: 200)",
    )
    parser.add_argument(
        "--val-seeds", type=int, default=50,
        help="Number of validation seeds (default: 50)",
    )
    parser.add_argument(
        "--test-seeds", type=int, default=50,
        help="Number of test seeds for final eval (default: 50)",
    )
    # Output
    parser.add_argument(
        "--output-dir", default="outputs/gepa_ablation",
        help="Output directory (default: outputs/gepa_ablation)",
    )
    # Oracle mode
    parser.add_argument(
        "--oracle", action="store_true",
        help="Oracle mode: reflect on synthetic envs, validate on tau2-bench. "
             "Requires --domain and --user-base-url.",
    )
    parser.add_argument(
        "--domain", default=None, choices=["airline", "retail"],
        help="Filter tau2-bench to one domain (default: both).",
    )
    # Flags
    parser.add_argument(
        "--seed-prompt-mode", default="execution",
        choices=["execution", "routing"],
        help="Which seed prompt to use: 'execution' (hand-written how-to-do-it "
             "instructions) or 'routing' (orchestrator skill descriptions that "
             "describe when-to-use-it). Default: execution.",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip baseline evaluation of seed prompt",
    )
    parser.add_argument(
        "--eval-only", default=None,
        help="Path to a prompt file to evaluate (skip optimization)",
    )

    args = parser.parse_args()

    skill_name = args.skill
    config = SKILL_CONFIG[skill_name]

    # Select seed prompt mode
    if args.seed_prompt_mode == "routing":
        config = dict(config)  # shallow copy so we don't mutate the original
        config["seed_prompt"] = ROUTING_DESCRIPTIONS[skill_name]

    # Validate oracle mode args
    if args.oracle:
        if not args.user_base_url:
            parser.error("--oracle requires --user-base-url for tau2-bench user sim")

    run_dir = os.path.join(args.output_dir, skill_name)
    os.makedirs(run_dir, exist_ok=True)

    # Build dataset and valset
    if args.oracle:
        # Oracle mode:
        #   dataset = synthetic env seeds (GEPA reflects on these)
        #   valset  = skill-specific tau2-bench tasks from scaling_skills.json
        train_seeds = args.train_seeds

        dataset = [
            {"seed": i, "env": "synthetic"} for i in range(train_seeds)
        ]
        if skill_name in SKILL_TASK_IDS:
            valset = _task_ids_to_valset(SKILL_TASK_IDS[skill_name])
        elif skill_name in TOOLSANDBOX_FAILED_SCENARIOS:
            valset = _toolsandbox_scenarios_to_valset(
                TOOLSANDBOX_FAILED_SCENARIOS[skill_name],
            )
        else:
            parser.error(
                f"No oracle valset defined for skill '{skill_name}'"
            )
        test_seed_list = None  # test eval uses valset directly
    else:
        # Normal mode: both dataset and valset use synthetic envs
        train_seeds = args.train_seeds
        val_seeds = args.val_seeds
        test_seeds = args.test_seeds

        dataset = [{"seed": i} for i in range(train_seeds)]
        valset = [{"seed": train_seeds + i} for i in range(val_seeds)]
        test_seed_list = [
            train_seeds + val_seeds + i for i in range(test_seeds)
        ]

    print(f"{'=' * 60}")
    print(f"GEPA Ablation: {skill_name}")
    print(f"{'=' * 60}")
    print(f"Environment:      {config['env_name']}")
    print(f"Interface:        {config['env_type']}")
    print(f"Seed prompt mode: {args.seed_prompt_mode}")
    print(f"Oracle mode:      {args.oracle}")
    if args.oracle:
        n_air = sum(1 for v in valset if v.get("domain") == "airline")
        n_ret = sum(1 for v in valset if v.get("domain") == "retail")
        print(f"Val tasks:        {len(valset)} tau2-bench ({n_air} airline + {n_ret} retail)")
    print(f"Agent model:      {args.model}")
    print(f"Reflection LM:    {args.reflection_model} @ {args.reflection_base_url}")
    print(f"Train (synth):    {len(dataset)} seeds")
    print(f"Val:              {len(valset)} {'tau2-bench tasks' if args.oracle else 'synthetic seeds'}")
    print(f"Max metric calls: {args.max_metric_calls}")
    print(f"Max workers:      {args.max_workers}")
    print(f"Output:           {run_dir}")
    print()

    # ------------------------------------------------------------------
    # Eval-only mode: evaluate a saved prompt without optimization
    # ------------------------------------------------------------------
    if args.eval_only:
        with open(args.eval_only) as f:
            prompt = f.read().strip()
        print(f"Evaluating prompt from {args.eval_only}")
        print(f"Prompt:\n{prompt}\n")
        if args.oracle:
            if skill_name in TOOLSANDBOX_FAILED_SCENARIOS:
                score, results = evaluate_toolsandbox(prompt, valset, args)
            else:
                score, results = evaluate_tau2bench(prompt, valset, args)
            passed = sum(1 for r in results if r["reward"] >= 1.0)
            print(f"Eval: {passed}/{len(results)} ({score:.3f})")
        else:
            score = evaluate_baseline(
                skill_name, prompt, test_seed_list, args,
            )
            print(f"Test score: {score:.3f}")
        return

    # ------------------------------------------------------------------
    # Baseline: evaluate seed prompt before optimization
    # ------------------------------------------------------------------
    seed_prompt = config["seed_prompt"]

    if not args.skip_baseline:
        if args.oracle:
            eval_type = "ToolSandbox" if skill_name in TOOLSANDBOX_FAILED_SCENARIOS else "tau2-bench"
            print(f"Evaluating baseline on {len(valset)} {eval_type} tasks...")
            if skill_name in TOOLSANDBOX_FAILED_SCENARIOS:
                baseline_score, _ = evaluate_toolsandbox(
                    seed_prompt, valset, args,
                )
            else:
                baseline_score, _ = evaluate_tau2bench(
                    seed_prompt, valset, args,
                )
        else:
            print("Evaluating baseline (seed prompt) on val seeds...")
            baseline_score = evaluate_baseline(
                skill_name, seed_prompt,
                [e["seed"] for e in valset[:30]],
                args,
            )
        print(f"Baseline val score: {baseline_score:.3f}\n")
    else:
        baseline_score = None

    # ------------------------------------------------------------------
    # GEPA optimization
    # ------------------------------------------------------------------
    evaluator = make_evaluator(skill_name, args)

    reflection_lm = LM(
        model=f"openai/{args.reflection_model}",
        api_base=args.reflection_base_url,
        api_key="not-needed",
        temperature=args.reflection_temperature,
        max_tokens=args.reflection_max_tokens,
    )

    # Checkpoint stopper: saves best prompt periodically
    stop_callbacks = []
    if args.checkpoint_interval > 0:
        stop_callbacks.append(CheckpointStopper(
            run_dir=run_dir,
            interval=args.checkpoint_interval,
            str_candidate_key="current_candidate",
        ))

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir,
            max_metric_calls=args.max_metric_calls,
            parallel=True,
            max_workers=args.max_workers,
            cache_evaluation=True,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            reflection_minibatch_size=args.reflection_minibatch_size,
        ),
        merge=MergeConfig(),
        stop_callbacks=stop_callbacks if stop_callbacks else None,
    )

    print("Starting GEPA optimization...")
    result = optimize_anything(
        seed_candidate=seed_prompt,
        evaluator=evaluator,
        dataset=dataset,
        valset=valset,
        objective=config["objective"],
        background=config["background"],
        config=gepa_config,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    best_prompt = result.best_candidate
    best_val_score = result.val_aggregate_scores[result.best_idx]

    print(f"\n{'=' * 60}")
    print(f"Optimization complete: {skill_name}")
    print(f"{'=' * 60}")
    print(f"Candidates explored: {result.num_candidates}")
    print(f"Total metric calls:  {result.total_metric_calls}")
    print(f"Best val score:      {best_val_score:.3f}")
    if baseline_score is not None:
        print(f"Baseline val score:  {baseline_score:.3f}")
        print(f"Improvement:         {best_val_score - baseline_score:+.3f}")
    print(f"\nBest Prompt:\n{best_prompt}\n")

    # Evaluate best prompt on test set
    if args.oracle:
        if skill_name in TOOLSANDBOX_FAILED_SCENARIOS:
            print(f"Evaluating best prompt on {len(valset)} ToolSandbox tasks...")
            test_score, test_results = evaluate_toolsandbox(
                best_prompt, valset, args,
            )
            passed = sum(1 for r in test_results if r["reward"] >= 1.0)
            print(f"ToolSandbox: {passed}/{len(test_results)} ({test_score:.3f})")
        else:
            print(f"Evaluating best prompt on {len(valset)} tau2-bench tasks...")
            test_score, test_results = evaluate_tau2bench(
                best_prompt, valset, args,
            )
            passed = sum(1 for r in test_results if r["reward"] >= 1.0)
            n_air = sum(1 for r in test_results if r["domain"] == "airline")
            n_ret = sum(1 for r in test_results if r["domain"] == "retail")
            air_passed = sum(
                1 for r in test_results
                if r["reward"] >= 1.0 and r["domain"] == "airline"
            )
            ret_passed = sum(
                1 for r in test_results
                if r["reward"] >= 1.0 and r["domain"] == "retail"
            )
            parts = [f"tau2-bench: {passed}/{len(test_results)}"]
            if n_air:
                parts.append(f"airline {air_passed}/{n_air}")
            if n_ret:
                parts.append(f"retail {ret_passed}/{n_ret}")
            print(" | ".join(parts))
    else:
        print(f"Evaluating best prompt on {len(test_seed_list)} test seeds...")
        test_score = evaluate_baseline(
            skill_name, best_prompt, test_seed_list, args,
        )
        test_results = None
        print(f"Test score: {test_score:.3f}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    with open(os.path.join(run_dir, "best_prompt.txt"), "w") as f:
        f.write(best_prompt)

    with open(os.path.join(run_dir, "seed_prompt.txt"), "w") as f:
        f.write(seed_prompt)

    summary = {
        "skill": skill_name,
        "env_name": config["env_name"],
        "model": args.model,
        "seed_prompt_mode": args.seed_prompt_mode,
        "oracle": args.oracle,
        "domain": args.domain,
        "reflection_model": args.reflection_model,
        "reflection_base_url": args.reflection_base_url,
        "max_metric_calls": args.max_metric_calls,
        "train_seeds_synthetic": len(dataset),
        "val_tasks": len(valset),
        "val_type": "tau2_bench" if args.oracle else "synthetic",
        "baseline_val_score": baseline_score,
        "best_val_score": best_val_score,
        "test_score": test_score,
        "candidates_explored": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "seed_prompt": seed_prompt,
        "best_prompt": best_prompt,
    }
    if args.oracle and test_results:
        summary["tau2_bench_per_task"] = test_results
        summary["tau2_bench_passed"] = sum(
            1 for r in test_results if r["reward"] >= 1.0
        )
        summary["tau2_bench_total"] = len(test_results)
        summary["tau2_bench_airline_passed"] = sum(
            1 for r in test_results
            if r["reward"] >= 1.0 and r["domain"] == "airline"
        )
        summary["tau2_bench_retail_passed"] = sum(
            1 for r in test_results
            if r["reward"] >= 1.0 and r["domain"] == "retail"
        )
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, "gepa_result.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    try:
        with open(os.path.join(run_dir, "candidate_tree.html"), "w") as f:
            f.write(result.candidate_tree_html())
    except Exception:
        pass  # Visualization is optional

    print(f"\nArtifacts saved to {run_dir}/")
    print(f"  best_prompt.txt     — optimized skill prompt")
    print(f"  seed_prompt.txt     — initial seed prompt")
    print(f"  summary.json        — scores and config")
    print(f"  gepa_result.json    — full GEPA result (candidates, lineage)")
    print(f"  candidate_tree.html — evolution visualization")


if __name__ == "__main__":
    main()
