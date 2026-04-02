"""Datetime Reasoning Game — Teaches correct Unix timestamp handling.

The base model skips `timestamp_to_datetime_info` and hallucinates the year
(usually 2024 instead of 2026). AWM always calls this tool, gaining ~15 scenarios.

This environment trains the model to:
1. Call `get_current_timestamp` → `timestamp_to_datetime_info` to learn the date
2. Use `datetime_info_to_timestamp` or `shift_timestamp` for date arithmetic
3. Pass correct timestamps to `add_reminder` / `search_reminder`

Format matches ToolSandbox EXACTLY: same tool schemas, same function signatures,
same return formats. Only the synthetic data differs.
"""

import random
import json
import re
import copy
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# =====================================================================
# System prompt — EXACT match to ToolSandbox
# =====================================================================

SYSTEM_PROMPT = (
    "Don't make assumptions about what values to plug into functions. "
    "Ask for clarification if a user request is ambiguous."
)


# =====================================================================
# Tool schemas — EXACT match to ToolSandbox utilities + reminder tools
# =====================================================================

TOOL_SCHEMAS = [
    # Datetime tools
    {"type": "function", "function": {
        "name": "get_current_timestamp",
        "description": "Get current POSIX timestamp",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "timestamp_to_datetime_info",
        "description": 'Convert POSIX timestamp to a dictionary of date time information, including ["year", "month", "day", "hour", "minute", "second", "isoweekday"]. isoweekday ranges from [1, 7]',
        "parameters": {"type": "object", "properties": {
            "timestamp": {"type": "number", "description": "Float representing POSIX timestamp"},
        }, "required": ["timestamp"]},
    }},
    {"type": "function", "function": {
        "name": "datetime_info_to_timestamp",
        "description": "Convert date time information to POSIX timestamp",
        "parameters": {"type": "object", "properties": {
            "year": {"type": "integer"}, "month": {"type": "integer"},
            "day": {"type": "integer"}, "hour": {"type": "integer"},
            "minute": {"type": "integer"}, "second": {"type": "integer"},
        }, "required": ["year", "month", "day", "hour", "minute", "second"]},
    }},
    {"type": "function", "function": {
        "name": "shift_timestamp",
        "description": "Shifts a POSIX timestamp by provided deltas. Delta can be positive or negative.",
        "parameters": {"type": "object", "properties": {
            "timestamp": {"type": "number"}, "weeks": {"type": "integer"},
            "days": {"type": "integer"}, "hours": {"type": "integer"},
            "minutes": {"type": "integer"}, "seconds": {"type": "integer"},
        }, "required": ["timestamp"]},
    }},
    {"type": "function", "function": {
        "name": "timestamp_diff",
        "description": "Calculates the difference between two timestamps, timestamp_1 - timestamp_0, return the difference represented in days and seconds",
        "parameters": {"type": "object", "properties": {
            "timestamp_0": {"type": "number"}, "timestamp_1": {"type": "number"},
        }, "required": ["timestamp_0", "timestamp_1"]},
    }},
    {"type": "function", "function": {
        "name": "search_holiday",
        "description": "Search for a US holiday by name. Returns POSIX timestamp or null.",
        "parameters": {"type": "object", "properties": {
            "holiday_name": {"type": "string"},
            "year": {"type": "integer"},
        }, "required": ["holiday_name"]},
    }},
    # Reminder tools
    {"type": "function", "function": {
        "name": "add_reminder",
        "description": "Add a reminder with content and timestamp.",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string"},
            "reminder_timestamp": {"type": "number"},
        }, "required": ["content", "reminder_timestamp"]},
    }},
    {"type": "function", "function": {
        "name": "search_reminder",
        "description": "Search reminders by content and/or timestamp bounds.",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string"},
            "reminder_timestamp_lowerbound": {"type": "number"},
            "reminder_timestamp_upperbound": {"type": "number"},
        }},
    }},
    {"type": "function", "function": {
        "name": "remove_reminder",
        "description": "Remove a reminder by ID.",
        "parameters": {"type": "object", "properties": {
            "reminder_id": {"type": "string"},
        }, "required": ["reminder_id"]},
    }},
]


# =====================================================================
# US Holidays (2026)
# =====================================================================

HOLIDAYS_2026 = {
    "New Year's Day": datetime.datetime(2026, 1, 1),
    "Martin Luther King Jr. Day": datetime.datetime(2026, 1, 19),
    "Presidents' Day": datetime.datetime(2026, 2, 16),
    "Memorial Day": datetime.datetime(2026, 5, 25),
    "Independence Day": datetime.datetime(2026, 7, 4),
    "Labor Day": datetime.datetime(2026, 9, 7),
    "Columbus Day": datetime.datetime(2026, 10, 12),
    "Veterans Day": datetime.datetime(2026, 11, 11),
    "Thanksgiving": datetime.datetime(2026, 11, 26),
    "Christmas Day": datetime.datetime(2026, 12, 25),
}

HOLIDAY_NAMES = list(HOLIDAYS_2026.keys())


# =====================================================================
# Data generation
# =====================================================================

REMINDER_CONTENTS = [
    "Buy groceries", "Call dentist", "Pick up kids", "Submit report",
    "Pay rent", "Gym session", "Team meeting", "Buy birthday gift",
    "Walk the dog", "Return library books", "Renew insurance",
    "Schedule oil change", "Buy chocolate milk", "Clean the house",
]


def _rid():
    return f"{random.randint(10000000,99999999):08x}-{random.randint(1000,9999)}"


def _make_db(rng, base_ts):
    """Create synthetic DB with reminders around the base timestamp."""
    reminders = []
    for i in range(rng.randint(2, 5)):
        delta_days = rng.randint(-3, 14)
        delta_hours = rng.randint(8, 20)
        ts = base_ts + delta_days * 86400 + delta_hours * 3600
        reminders.append({
            "reminder_id": _rid(),
            "content": rng.choice(REMINDER_CONTENTS),
            "reminder_timestamp": ts,
            "creation_timestamp": base_ts - rng.randint(86400, 86400 * 7),
        })
    return {"reminders": reminders, "current_timestamp": base_ts}


# =====================================================================
# Tool executor — EXACT same signatures/returns as ToolSandbox
# =====================================================================

class ToolExecutor:
    def __init__(self, db):
        self.db = db
        self._base_ts = db["current_timestamp"]

    def execute(self, name, args):
        fn = getattr(self, f"_t_{name}", None)
        if not fn:
            return f"Error: Unknown tool '{name}'"
        try:
            return fn(**args)
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def _t_get_current_timestamp(self):
        return self._base_ts

    def _t_timestamp_to_datetime_info(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        return {
            "year": dt.year, "month": dt.month, "day": dt.day,
            "hour": dt.hour, "minute": dt.minute, "second": dt.second,
            "isoweekday": dt.isoweekday(),
        }

    def _t_datetime_info_to_timestamp(self, year, month, day, hour, minute, second):
        return datetime.datetime(year, month, day, hour, minute, second).timestamp()

    def _t_shift_timestamp(self, timestamp, weeks=0, days=0, hours=0, minutes=0, seconds=0):
        dt = datetime.datetime.fromtimestamp(timestamp) + datetime.timedelta(
            weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)
        return dt.timestamp()

    def _t_timestamp_diff(self, timestamp_0, timestamp_1):
        td = datetime.datetime.fromtimestamp(timestamp_1) - datetime.datetime.fromtimestamp(timestamp_0)
        return {"days": td.days, "seconds": td.seconds}

    def _t_search_holiday(self, holiday_name, year=None):
        if year is None:
            year = datetime.datetime.fromtimestamp(self._base_ts).year
        # Fuzzy match
        best_match = None
        best_score = 0
        for name, dt in HOLIDAYS_2026.items():
            if dt.year != year:
                continue
            score = sum(1 for a, b in zip(holiday_name.lower(), name.lower()) if a == b)
            if holiday_name.lower() in name.lower() or name.lower() in holiday_name.lower():
                score = 100
            if score > best_score:
                best_score = score
                best_match = name
        if best_match and best_score > 3:
            dt = HOLIDAYS_2026[best_match]
            if year != 2026:
                # Adjust to requested year
                try:
                    dt = dt.replace(year=year)
                except ValueError:
                    return None
            return dt.timestamp()
        return None

    def _t_add_reminder(self, content, reminder_timestamp):
        rid = _rid()
        self.db["reminders"].append({
            "reminder_id": rid, "content": content,
            "reminder_timestamp": reminder_timestamp,
            "creation_timestamp": self._base_ts,
        })
        return rid

    def _t_search_reminder(self, content=None, reminder_timestamp_lowerbound=None,
                           reminder_timestamp_upperbound=None):
        results = []
        for r in self.db["reminders"]:
            if content and content.lower() not in r["content"].lower():
                continue
            if reminder_timestamp_lowerbound and r["reminder_timestamp"] < reminder_timestamp_lowerbound:
                continue
            if reminder_timestamp_upperbound and r["reminder_timestamp"] > reminder_timestamp_upperbound:
                continue
            results.append(r)
        return results

    def _t_remove_reminder(self, reminder_id):
        self.db["reminders"] = [r for r in self.db["reminders"] if r["reminder_id"] != reminder_id]
        return None


# =====================================================================
# Scenario generation
# =====================================================================

@dataclass
class Scenario:
    skill: str  # "relative_reminder", "holiday", "search_recency"
    user_message: str
    expected_timestamp: float  # ground truth target timestamp
    current_year: int
    verify_keywords: List[str]
    description: str = ""


WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def generate_scenario(seed: int) -> Tuple[Scenario, Dict]:
    rng = random.Random(seed)

    # Base time: March 2026, varying day/hour for diversity
    base_day = rng.randint(20, 28)
    base_hour = rng.randint(9, 16)
    try:
        base_dt = datetime.datetime(2026, 3, base_day, base_hour, rng.randint(0, 59), rng.randint(0, 59))
    except ValueError:
        base_dt = datetime.datetime(2026, 3, 25, base_hour, rng.randint(0, 59), rng.randint(0, 59))
    base_ts = base_dt.timestamp()

    db = _make_db(rng, base_ts)

    # Per-rollout noise
    noise_rng = random.Random(seed * 1000003 + int(time.time() * 1000) % 1000000 + id(db))
    # Add 0-2 noise reminders
    for _ in range(noise_rng.randint(0, 2)):
        db["reminders"].append({
            "reminder_id": _rid(),
            "content": noise_rng.choice(REMINDER_CONTENTS),
            "reminder_timestamp": base_ts + noise_rng.randint(-86400*3, 86400*14),
            "creation_timestamp": base_ts - noise_rng.randint(86400, 86400*5),
        })

    roll = rng.random()

    if roll < 0.35:
        # RELATIVE REMINDER: "tomorrow at 5PM", "in 3 days at 10AM"
        variant = rng.choice(["tomorrow", "in_n_days", "next_week"])
        target_hour = rng.choice([9, 10, 12, 14, 15, 17, 18, 20])
        content = rng.choice(REMINDER_CONTENTS)

        if variant == "tomorrow":
            target_dt = base_dt.replace(hour=target_hour, minute=0, second=0) + datetime.timedelta(days=1)
            hr_str = f"{target_hour % 12 or 12}{'AM' if target_hour < 12 else 'PM'}"
            msg = rng.choice([
                f"Remind me to {content.lower()} tomorrow at {hr_str}",
                f"Set a reminder for tomorrow {hr_str}: {content}",
                f"Tomorrow at {hr_str}, remind me to {content.lower()}",
            ])

        elif variant == "in_n_days":
            n = rng.randint(2, 7)
            target_dt = base_dt.replace(hour=target_hour, minute=0, second=0) + datetime.timedelta(days=n)
            hr_str = f"{target_hour % 12 or 12}{'AM' if target_hour < 12 else 'PM'}"
            msg = rng.choice([
                f"Remind me to {content.lower()} in {n} days at {hr_str}",
                f"Set a reminder for {n} days from now at {hr_str}: {content}",
            ])

        else:  # next_week
            target_dt = base_dt.replace(hour=target_hour, minute=0, second=0) + datetime.timedelta(weeks=1)
            hr_str = f"{target_hour % 12 or 12}{'AM' if target_hour < 12 else 'PM'}"
            msg = rng.choice([
                f"Remind me to {content.lower()} next week at {hr_str}",
                f"Set a reminder for next week {hr_str}: {content}",
            ])

        expected_ts = target_dt.timestamp()
        return Scenario("relative_reminder", msg, expected_ts, 2026,
                        ["reminder"], f"Relative: {variant} {hr_str}"), db

    elif roll < 0.60:
        # WEEKDAY REMINDER: "next Friday at 3PM"
        content = rng.choice(REMINDER_CONTENTS)
        target_hour = rng.choice([9, 10, 12, 14, 15, 17, 18])

        current_weekday = base_dt.isoweekday()  # 1=Mon, 7=Sun
        target_weekday = rng.randint(1, 7)
        days_ahead = (target_weekday - current_weekday) % 7
        if days_ahead == 0:
            days_ahead = 7  # "next X" means next week if today is X

        target_dt = base_dt.replace(hour=target_hour, minute=0, second=0) + datetime.timedelta(days=days_ahead)
        weekday_name = WEEKDAY_NAMES[target_weekday - 1]
        hr_str = f"{target_hour % 12 or 12}{'AM' if target_hour < 12 else 'PM'}"

        msg = rng.choice([
            f"Remind me to {content.lower()} next {weekday_name} at {hr_str}",
            f"Set a reminder for next {weekday_name} {hr_str}: {content}",
            f"Next {weekday_name} at {hr_str}, remind me to {content.lower()}",
        ])

        expected_ts = target_dt.timestamp()
        return Scenario("weekday_reminder", msg, expected_ts, 2026,
                        ["reminder"], f"Weekday: next {weekday_name} {hr_str}"), db

    elif roll < 0.80:
        # DAYS UNTIL HOLIDAY
        holiday = rng.choice(HOLIDAY_NAMES)
        holiday_dt = HOLIDAYS_2026[holiday]
        expected_days = (holiday_dt - base_dt).days
        if expected_days < 0:
            # Holiday already passed, pick one in the future
            future = [(n, d) for n, d in HOLIDAYS_2026.items() if d > base_dt]
            if future:
                holiday, holiday_dt = rng.choice(future)
                expected_days = (holiday_dt - base_dt).days
            else:
                holiday = "Christmas Day"
                holiday_dt = HOLIDAYS_2026[holiday]
                expected_days = (holiday_dt - base_dt).days

        msg = rng.choice([
            f"How many days until {holiday}?",
            f"When is {holiday} and how many days away is it?",
            f"How far away is {holiday}?",
        ])

        return Scenario("holiday", msg, holiday_dt.timestamp(), 2026,
                        [str(expected_days)], f"Holiday: {holiday} ({expected_days} days)"), db

    else:
        # SEARCH UPCOMING REMINDERS
        future_reminders = [r for r in db["reminders"] if r["reminder_timestamp"] > base_ts]
        if not future_reminders:
            # Ensure at least one future reminder
            rid = _rid()
            content = rng.choice(REMINDER_CONTENTS)
            future_ts = base_ts + rng.randint(3600, 86400 * 7)
            db["reminders"].append({
                "reminder_id": rid, "content": content,
                "reminder_timestamp": future_ts,
                "creation_timestamp": base_ts - 86400,
            })
            future_reminders = [db["reminders"][-1]]

        target_reminder = min(future_reminders, key=lambda r: r["reminder_timestamp"])

        msg = rng.choice([
            "What's my next upcoming reminder?",
            "Do I have any reminders coming up?",
            "What reminders do I have scheduled?",
            "Show me my upcoming reminders",
        ])

        return Scenario("search_recency", msg, base_ts, 2026,
                        [target_reminder["content"].split()[0].lower()],
                        f"Search upcoming"), db


# =====================================================================
# Game
# =====================================================================

class DatetimeGame:
    supports_structured_messages = True

    def __init__(self):
        self.done = False
        self.current_player = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player = None
        self._scenario = None
        self._db = None
        self._tools = None
        self._conversation = []
        self._step_count = 0
        self._tool_result = None
        self._all_tools = []
        self._all_tool_args = []
        self.max_steps = 15

    def reset(self, seed: int) -> None:
        self._scenario, self._db = generate_scenario(seed)

        # Per-rollout noise for GRPO variance
        noise_rng = random.Random(seed * 1000003 + int(time.time() * 1000) % 1000000 + id(self))
        # Shuffle existing reminders
        noise_rng.shuffle(self._db["reminders"])
        # Add 0-1 more noise reminders
        for _ in range(noise_rng.randint(0, 1)):
            self._db["reminders"].append({
                "reminder_id": _rid(),
                "content": noise_rng.choice(REMINDER_CONTENTS),
                "reminder_timestamp": self._db["current_timestamp"] + noise_rng.randint(-86400*3, 86400*10),
                "creation_timestamp": self._db["current_timestamp"] - noise_rng.randint(3600, 86400*3),
            })

        self._tools = ToolExecutor(copy.deepcopy(self._db))
        self._conversation = [{"role": "user", "content": self._scenario.user_message}]
        self._step_count = 0
        self._tool_result = None
        self._all_tools = []
        self._all_tool_args = []
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tool_schemas(self) -> List[Dict]:
        return TOOL_SCHEMAS

    def get_messages(self) -> List[Dict]:
        return list(self._conversation)

    def observe(self, player_id: int) -> str:
        if self.done:
            return ""
        return self._conversation[-1].get("content", "") if self._conversation else ""

    def legal_actions(self) -> List[str]:
        return []

    def step(self, action: Optional[str]) -> None:
        if self.done or action is None:
            return
        self._step_count += 1

        tc = self._parse_tool_call(action)

        if tc:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            result = self._tools.execute(name, args)
            # Use repr() to match ToolSandbox's Python-style output:
            # None/True/False instead of json's null/true/false
            result_str = repr(result) if not isinstance(result, str) else result
            self._tool_result = result_str
            self._all_tools.append(name)
            self._all_tool_args.append(args)
            tool_call_id = f"call_{self._step_count}"
            self._conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }],
            })
            self._conversation.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_str,
            })
        else:
            self._conversation.append({"role": "assistant", "content": action})
            self._evaluate(action)

        if self._step_count >= self.max_steps and not self.done:
            self.done = True
            if self._all_tools:
                self.rewards = {0: 0.15}
            else:
                self.rewards = {0: 0.0}

    def _evaluate(self, response: str):
        """Score based on scenario type."""
        response_lower = response.lower()
        tools = self._all_tools
        tool_args = self._all_tool_args
        scenario = self._scenario

        tool_chain_score = 0.0
        result_score = 0.0
        comm_score = 0.0

        # --- Tool chain score (0.50) ---
        # Key skill: did the model call timestamp_to_datetime_info?
        has_get_ts = "get_current_timestamp" in tools
        has_ts_to_dt = "timestamp_to_datetime_info" in tools
        has_dt_to_ts = "datetime_info_to_timestamp" in tools
        has_shift = "shift_timestamp" in tools
        has_search_holiday = "search_holiday" in tools
        has_ts_diff = "timestamp_diff" in tools
        has_add_reminder = "add_reminder" in tools
        has_search_reminder = "search_reminder" in tools

        if scenario.skill in ("relative_reminder", "weekday_reminder"):
            if has_get_ts:
                tool_chain_score += 0.10
            if has_ts_to_dt:
                tool_chain_score += 0.25  # THE critical skill
            if has_dt_to_ts or has_shift:
                tool_chain_score += 0.10
            if has_add_reminder:
                tool_chain_score += 0.05

        elif scenario.skill == "holiday":
            if has_get_ts:
                tool_chain_score += 0.10
            if has_ts_to_dt:
                tool_chain_score += 0.15
            if has_search_holiday:
                tool_chain_score += 0.10
                # Check if called with correct year
                for i, t in enumerate(tools):
                    if t == "search_holiday":
                        yr = tool_args[i].get("year")
                        if yr == scenario.current_year:
                            tool_chain_score += 0.10
                        break
            if has_ts_diff:
                tool_chain_score += 0.05

        elif scenario.skill == "search_recency":
            if has_get_ts:
                tool_chain_score += 0.20
            if has_search_reminder:
                tool_chain_score += 0.20
                # Check if used current timestamp as lowerbound
                for i, t in enumerate(tools):
                    if t == "search_reminder":
                        lb = tool_args[i].get("reminder_timestamp_lowerbound", 0)
                        if lb and abs(lb - scenario.expected_timestamp) < 120:
                            tool_chain_score += 0.10
                        break

        # --- Result accuracy score (0.30) ---
        if scenario.skill in ("relative_reminder", "weekday_reminder"):
            for i, t in enumerate(tools):
                if t == "add_reminder":
                    actual_ts = tool_args[i].get("reminder_timestamp", 0)
                    if isinstance(actual_ts, (int, float)):
                        diff = abs(actual_ts - scenario.expected_timestamp)
                        if diff < 120:        # within 2 minutes
                            result_score = 0.30
                        elif diff < 3600:     # within 1 hour
                            result_score = 0.20
                        elif diff < 86400:    # within 1 day
                            result_score = 0.10
                        # Past timestamp = zero
                        if actual_ts < self._tools._base_ts:
                            result_score = 0.0
                    break

        elif scenario.skill == "holiday":
            # Check if the response contains the correct day count
            if scenario.verify_keywords:
                expected_days = scenario.verify_keywords[0]
                if expected_days in response:
                    result_score = 0.30
                else:
                    # Try to extract a number from the response
                    nums = re.findall(r'\b(\d+)\b', response)
                    if nums:
                        for n in nums:
                            if abs(int(n) - int(expected_days)) <= 1:
                                result_score = 0.20
                                break

        elif scenario.skill == "search_recency":
            # Check if found the right reminder
            if scenario.verify_keywords:
                matched = sum(1 for kw in scenario.verify_keywords if kw in response_lower)
                result_score = 0.30 * (matched / len(scenario.verify_keywords))

        # --- Communication score (0.20) ---
        if response and len(response.strip()) > 10:
            comm_score = 0.20

        total = min(1.0, tool_chain_score + result_score + comm_score)
        self.rewards = {0: total}
        self.done = True

    def _parse_tool_call(self, action):
        if not action:
            return None
        try:
            p = json.loads(action)
            if isinstance(p, dict):
                if "name" in p and "arguments" in p:
                    return p
                if "function" in p:
                    fn = p["function"]
                    return {"name": fn.get("name", ""), "arguments": fn.get("arguments", {})}
        except Exception:
            pass
        try:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', action)
            if m:
                name = m.group(1)
                am = re.search(r'"arguments"\s*:\s*(\{[^}]*\})', action)
                args = json.loads(am.group(1)) if am else {}
                return {"name": name, "arguments": args}
        except Exception:
            pass
        return None


def extract_action(text, legal_actions):
    return text.strip() if text else None


# =====================================================================
# Test
# =====================================================================

if __name__ == "__main__":
    from collections import Counter

    skill_counts = Counter()

    for seed in range(50):
        game = DatetimeGame()
        game.reset(seed)
        s = game._scenario
        skill_counts[s.skill] += 1
        base_dt = datetime.datetime.fromtimestamp(game._db["current_timestamp"])
        print(f"Seed {seed:>2d} [{s.skill:>20s}] {s.description:<45s} | {s.user_message[:60]}")

    print(f"\nSkill distribution: {dict(skill_counts)}")
