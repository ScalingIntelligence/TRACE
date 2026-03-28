"""TEC Game v2 — Focused on two clean skills from ToolSandbox failure analysis.

Skill A: Tool-then-Communicate (40%) — agent calls tool correctly but must also
         respond to user with the result. Targets the 23 scenarios scoring 0.50.

Skill B: Error Recovery Chain (40%) — agent hits PermissionError and must diagnose,
         fix the prerequisite, retry, and communicate. Targets 13 scenarios scoring 0.0.

Combined (20%) — error recovery + must communicate the final result.

Format matches ToolSandbox EXACTLY: same tool schemas, same error messages,
same system prompt. Only the synthetic data differs.
"""

import random
import json
import re
import copy
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# =====================================================================
# System prompt — EXACT match to ToolSandbox
# =====================================================================

SYSTEM_PROMPT_BASE = (
    "Don't make assumptions about what values to plug into functions. "
    "Ask for clarification if a user request is ambiguous."
)

SYSTEM_PROMPT_HINT = (
    "Don't make assumptions about what values to plug into functions. "
    "Ask for clarification if a user request is ambiguous.\n"
    "If a tool returns a PermissionError, fix the underlying issue automatically "
    "without asking the user, then retry the original action."
)

# Default for registration
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE


# =====================================================================
# Tool schemas — EXACT match to ToolSandbox tools
# =====================================================================

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "search_contacts", "description": "Search for contacts by name, phone number, or relationship.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "phone_number": {"type": "string"}, "relationship": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "add_contact", "description": "Add a new contact.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "phone_number": {"type": "string"}, "relationship": {"type": "string"}}, "required": ["name", "phone_number"]}}},
    {"type": "function", "function": {"name": "modify_contact", "description": "Modify an existing contact by person_id.", "parameters": {"type": "object", "properties": {"person_id": {"type": "string"}, "name": {"type": "string"}, "phone_number": {"type": "string"}, "relationship": {"type": "string"}}, "required": ["person_id"]}}},
    {"type": "function", "function": {"name": "remove_contact", "description": "Remove a contact by person_id.", "parameters": {"type": "object", "properties": {"person_id": {"type": "string"}}, "required": ["person_id"]}}},
    {"type": "function", "function": {"name": "get_wifi_status", "description": "Check if WiFi is currently on or off.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_wifi_status", "description": "Turn WiFi on or off.", "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}}},
    {"type": "function", "function": {"name": "get_cellular_service_status", "description": "Check if cellular service is on or off.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_cellular_service_status", "description": "Turn cellular service on or off.", "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}}},
    {"type": "function", "function": {"name": "get_location_service_status", "description": "Check if location service is on or off.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_location_service_status", "description": "Turn location service on or off.", "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}}},
    {"type": "function", "function": {"name": "get_low_battery_mode_status", "description": "Check if low battery mode is on or off.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_low_battery_mode_status", "description": "Turn low battery mode on or off.", "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}}},
    {"type": "function", "function": {"name": "get_current_location", "description": "Get current GPS coordinates. Requires location service enabled.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "add_reminder", "description": "Add a reminder.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "reminder_timestamp": {"type": "integer"}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "search_reminder", "description": "Search reminders by content.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "remove_reminder", "description": "Remove a reminder by ID.", "parameters": {"type": "object", "properties": {"reminder_id": {"type": "string"}}, "required": ["reminder_id"]}}},
    {"type": "function", "function": {"name": "send_message_with_phone_number", "description": "Send a text message to a phone number.", "parameters": {"type": "object", "properties": {"phone_number": {"type": "string"}, "content": {"type": "string"}}, "required": ["phone_number", "content"]}}},
    {"type": "function", "function": {"name": "search_messages", "description": "Search messages by content or sender.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "sender_phone_number": {"type": "string"}}}}},
]


# =====================================================================
# Data generation
# =====================================================================

NAMES = ["Homer S", "Marge B", "Lisa T", "Bart F", "Ned V", "Maude K",
         "Apu P", "Krusty L", "Moe M", "Carl N", "Alice W", "Bob R",
         "Charlie D", "Diana E", "Eve G", "Frank H", "Grace J", "Henry Q"]
RELATIONSHIPS = ["friend", "boss", "colleague", "family", "neighbor"]
MESSAGES = ["Hey, want to grab lunch?", "Don't forget the meeting at 3pm",
            "Can you pick up some milk?", "Happy birthday!", "Want some GPUs?",
            "The deadline is tomorrow", "Are you free this weekend?", "I sent the document"]
REMINDERS = ["Buy groceries", "Call dentist", "Pick up kids", "Submit report",
             "Pay rent", "Gym session", "Team meeting", "Buy birthday gift"]


def _pid():
    return f"{random.randint(10000000,99999999):08x}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(100000000000,999999999999):012x}"

def _phone():
    return f"+1{random.randint(1000000000,9999999999)}"


def _make_db(rng):
    contacts = []
    for _ in range(rng.randint(3, 5)):
        contacts.append({"person_id": _pid(), "name": rng.choice(NAMES),
                          "phone_number": _phone(), "relationship": rng.choice(RELATIONSHIPS), "is_self": False})
    contacts.append({"person_id": _pid(), "name": "Me", "phone_number": _phone(),
                      "relationship": "self", "is_self": True})
    messages = [{"message_id": _pid(), "sender_phone_number": rng.choice(contacts[:-1])["phone_number"],
                 "sender_name": rng.choice(contacts[:-1])["name"], "content": rng.choice(MESSAGES),
                 "creation_timestamp": 1711000000 + i * 3600} for i in range(rng.randint(2, 4))]
    reminders = [{"reminder_id": _pid(), "content": rng.choice(REMINDERS),
                  "reminder_timestamp": 1711000000 + (i+1) * 86400, "creation_timestamp": 1711000000 - 86400}
                 for i in range(rng.randint(1, 3))]
    return {"contacts": contacts, "messages": messages, "reminders": reminders,
            "settings": {"wifi": True, "cellular": True, "location": True, "low_battery_mode": False},
            "current_timestamp": 1711000000 + 36000}


# =====================================================================
# Tool executor — EXACT same error messages as ToolSandbox
# =====================================================================

class ToolExecutor:
    def __init__(self, db):
        self.db = db

    def execute(self, name, args):
        fn = getattr(self, f"_t_{name}", None)
        if not fn:
            return f"Error: Unknown tool '{name}'"
        try:
            return fn(**args)
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def _t_search_contacts(self, name=None, phone_number=None, relationship=None):
        r = []
        for c in self.db["contacts"]:
            if name and name.lower() not in c["name"].lower(): continue
            if phone_number and phone_number != c["phone_number"]: continue
            if relationship and relationship.lower() != c["relationship"].lower(): continue
            r.append(c)
        return r

    def _t_add_contact(self, name, phone_number, relationship=None):
        pid = _pid()
        self.db["contacts"].append({"person_id": pid, "name": name, "phone_number": phone_number,
                                     "relationship": relationship or "", "is_self": False})
        return pid

    def _t_modify_contact(self, person_id, name=None, phone_number=None, relationship=None):
        for c in self.db["contacts"]:
            if c["person_id"] == person_id:
                if name: c["name"] = name
                if phone_number: c["phone_number"] = phone_number
                if relationship: c["relationship"] = relationship
                return None
        return "Error: Contact not found"

    def _t_remove_contact(self, person_id):
        self.db["contacts"] = [c for c in self.db["contacts"] if c["person_id"] != person_id]
        return None

    def _t_get_wifi_status(self): return self.db["settings"]["wifi"]
    def _t_set_wifi_status(self, on):
        if on and self.db["settings"]["low_battery_mode"]:
            raise PermissionError("Wifi cannot be turned on in low battery mode")
        self.db["settings"]["wifi"] = on; return None

    def _t_get_cellular_service_status(self): return self.db["settings"]["cellular"]
    def _t_set_cellular_service_status(self, on):
        if on and self.db["settings"]["low_battery_mode"]:
            raise PermissionError("Cellular service cannot be turned on in low battery mode")
        self.db["settings"]["cellular"] = on; return None

    def _t_get_location_service_status(self): return self.db["settings"]["location"]
    def _t_set_location_service_status(self, on):
        if on and self.db["settings"]["low_battery_mode"]:
            raise PermissionError("Location service cannot be turned on in low battery mode")
        self.db["settings"]["location"] = on; return None

    def _t_get_low_battery_mode_status(self): return self.db["settings"]["low_battery_mode"]
    def _t_set_low_battery_mode_status(self, on):
        self.db["settings"]["low_battery_mode"] = on; return None

    def _t_get_current_location(self):
        if not self.db["settings"]["location"]:
            raise PermissionError("Location service is not enabled.")
        return {"latitude": 37.334606, "longitude": -122.009102}

    def _t_add_reminder(self, content, reminder_timestamp=None):
        rid = _pid()
        self.db["reminders"].append({"reminder_id": rid, "content": content,
                                      "reminder_timestamp": reminder_timestamp or self.db["current_timestamp"] + 3600})
        return rid

    def _t_search_reminder(self, content=None):
        return [r for r in self.db["reminders"] if not content or content.lower() in r["content"].lower()]

    def _t_remove_reminder(self, reminder_id):
        self.db["reminders"] = [r for r in self.db["reminders"] if r["reminder_id"] != reminder_id]
        return None

    def _t_send_message_with_phone_number(self, phone_number, content):
        if not self.db["settings"]["cellular"]:
            raise ConnectionError("Cellular service is not enabled")
        return _pid()

    def _t_search_messages(self, content=None, sender_phone_number=None):
        r = []
        for m in self.db["messages"]:
            if content and content.lower() not in m["content"].lower(): continue
            if sender_phone_number and sender_phone_number != m.get("sender_phone_number"): continue
            r.append(m)
        return r


# =====================================================================
# Scenario generation
# =====================================================================

@dataclass
class Scenario:
    skill: str  # "communicate" or "recovery" or "combined"
    user_message: str
    verify_keywords: List[str]  # must appear in agent's final response
    description: str = ""


def generate_scenario(seed: int) -> Tuple[Scenario, Dict]:
    rng = random.Random(seed)
    db = _make_db(rng)

    # Add per-rollout noise for GRPO variance
    noise_rng = random.Random(seed * 1000003 + int(time.time() * 1000) % 1000000 + id(db))
    noise_rng.shuffle(db["contacts"][:-1])
    for _ in range(noise_rng.randint(0, 2)):
        db["contacts"].insert(-1, {"person_id": _pid(), "name": noise_rng.choice(NAMES),
                                    "phone_number": _phone(), "relationship": noise_rng.choice(RELATIONSHIPS), "is_self": False})

    # Skill distribution: 40% communicate, 40% recovery, 20% combined
    roll = rng.random()

    if roll < 0.40:
        # SKILL A: Tool-then-Communicate
        scenario_type = rng.choice([
            "lookup_phone", "lookup_relationship", "check_setting",
            "toggle_setting", "add_contact", "search_reminder",
            "add_reminder_simple", "search_message",
        ])

        if scenario_type == "lookup_phone":
            c = rng.choice(db["contacts"][:-1])
            return Scenario("communicate", f"What is {c['name']}'s phone number?",
                          [c["phone_number"]], "Lookup phone number"), db

        elif scenario_type == "lookup_relationship":
            c = rng.choice(db["contacts"][:-1])
            return Scenario("communicate", f"What is my relationship with {c['name']}?",
                          [c["relationship"]], "Lookup relationship"), db

        elif scenario_type == "check_setting":
            setting = rng.choice(["wifi", "cellular"])
            val = db["settings"][setting]
            q = f"Is my {'WiFi' if setting == 'wifi' else 'cellular service'} on?"
            kw = ["on" if val else "off"]
            return Scenario("communicate", q, kw, f"Check {setting}"), db

        elif scenario_type == "toggle_setting":
            setting = rng.choice(["wifi", "cellular"])
            action = rng.choice(["on", "off"])
            db["settings"]["low_battery_mode"] = False  # ensure success
            label = "WiFi" if setting == "wifi" else "cellular service"
            return Scenario("communicate", f"Turn {action} my {label}",
                          [], f"Toggle {setting} {action}"), db

        elif scenario_type == "add_contact":
            name = rng.choice(NAMES)
            phone = _phone()
            return Scenario("communicate", f"Add {name} to my contacts, phone number {phone}",
                          [name], "Add contact"), db

        elif scenario_type == "search_reminder":
            if db["reminders"]:
                r = rng.choice(db["reminders"])
                kw = r["content"].split()[0].lower()
                return Scenario("communicate", f"Do I have a reminder about {kw}?",
                              [kw], "Search reminder"), db
            else:
                c = rng.choice(db["contacts"][:-1])
                return Scenario("communicate", f"What is {c['name']}'s phone number?",
                              [c["phone_number"]], "Lookup phone number"), db

        elif scenario_type == "add_reminder_simple":
            content = rng.choice(REMINDERS)
            return Scenario("communicate", f"Remind me to {content.lower()}",
                          ["reminder"], "Add reminder"), db

        elif scenario_type == "search_message":
            if db["messages"]:
                m = rng.choice(db["messages"])
                kw = m["content"].split()[0]
                return Scenario("communicate", f"Find messages containing '{kw}'",
                              [kw.lower()], "Search messages"), db
            else:
                c = rng.choice(db["contacts"][:-1])
                return Scenario("communicate", f"What is {c['name']}'s phone number?",
                              [c["phone_number"]], "Lookup phone"), db

    elif roll < 0.80:
        # SKILL B: Error Recovery Chain
        # KEY TRICK: The seed picks the SCENARIO (which service to enable),
        # but the per-rollout noise randomizes WHETHER low_battery blocks it.
        # This gives GRPO variance: same user request, sometimes easy (no blocker),
        # sometimes hard (blocked). The model learns: "when it fails, fix the prereq."
        target = rng.choice(["wifi", "cellular", "location"])
        db["settings"][target] = False

        # LOW_BATTERY IS SET IN reset() VIA NOISE — NOT HERE
        # This is critical: the seed determines the scenario type,
        # but reset()'s noise_rng determines if low_battery is on or off.
        # So within the same GRPO group (same seed), some rollouts have
        # low_battery=True (hard, needs recovery) and some have
        # low_battery=False (easy, direct success).
        # DO NOT set low_battery_mode here — let reset() handle it.

        prompts = {
            "wifi": [
                "Turn on wifi",
                "I need WiFi turned on.",
                "Get me connected to the internet.",
                "Enable my WiFi please.",
            ],
            "cellular": [
                "Turn on cellular",
                "I don't have cellphone signal. Can you get it on?",
                "Enable cellular service.",
                "Turn on my cellular service please.",
            ],
            "location": [
                "Turn on location service",
                "Enable location services.",
                "I need location turned on.",
                "Turn on my GPS.",
            ],
        }
        msg = rng.choice(prompts[target])
        label = {"wifi": "WiFi", "cellular": "cellular service", "location": "location service"}[target]
        return Scenario("recovery", msg, [label.split()[0].lower(), "on"],
                        f"Recovery: enable {target}"), db

    else:
        # COMBINED: Get location (may or may not need recovery)
        # Same trick: low_battery randomized per rollout in reset()
        db["settings"]["location"] = False
        prompts = [
            "What are my current GPS coordinates?",
            "Where am I right now?",
            "Get my current location.",
        ]
        return Scenario("combined", rng.choice(prompts),
                        ["latitude", "longitude"],
                        "Combined: get location + communicate"), db


# =====================================================================
# Game
# =====================================================================

class TECGameV2:
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
        self._tool_called = False
        self._tool_result = None
        self._all_tools = []
        self._use_hint = False
        self.max_steps = 12

    def reset(self, seed: int) -> None:
        self._scenario, self._db = generate_scenario(seed)

        # PER-ROLLOUT RANDOMIZATION — creates GRPO variance.
        noise_rng = random.Random(seed * 1000003 + int(time.time() * 1000) % 1000000 + id(self))

        if self._scenario.skill in ("recovery", "combined"):
            # Always HARD (low_battery=True), randomly add hint ~40% of the time.
            # HARD without hint → model reports error → r=0.55
            # HARD with hint → model fixes blocker and retries → r=1.0
            # Clean contrast: same scenario, same blocker, only the hint differs.
            # GRPO reinforces the recovery action (set_low_battery_mode → retry).
            self._db["settings"]["low_battery_mode"] = True
            self._use_hint = noise_rng.random() < 0.40  # 40% get hint
        else:
            # Communicate scenarios: randomize settings for noise
            self._db["settings"]["low_battery_mode"] = noise_rng.choice([True, False])
            self._use_hint = False

        # Shuffle contacts for extra noise
        noise_rng.shuffle(self._db["contacts"][:-1])

        # Add 0-2 noise contacts
        for _ in range(noise_rng.randint(0, 2)):
            self._db["contacts"].insert(-1, {
                "person_id": _pid(), "name": noise_rng.choice(NAMES),
                "phone_number": _phone(), "relationship": noise_rng.choice(RELATIONSHIPS),
                "is_self": False,
            })

        self._tools = ToolExecutor(copy.deepcopy(self._db))
        self._conversation = [{"role": "user", "content": self._scenario.user_message}]
        self._step_count = 0
        self._tool_called = False
        self._tool_result = None
        self._all_tools = []
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def get_system_prompt(self) -> str:
        if getattr(self, '_use_hint', False):
            return SYSTEM_PROMPT_HINT
        return SYSTEM_PROMPT_BASE

    def get_tool_schemas(self) -> List[Dict]:
        return TOOL_SCHEMAS

    def get_messages(self) -> List[Dict]:
        return list(self._conversation)

    def observe(self, player_id: int) -> str:
        if self.done: return ""
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
            result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
            self._tool_called = True
            self._tool_result = result_str
            self._all_tools.append(name)
            self._conversation.append({"role": "assistant", "content": action,
                                        "tool_call": {"name": name, "arguments": args}})
            self._conversation.append({"role": "tool", "content": result_str})
            # Don't end — wait for communication or more tool calls
        else:
            # Text response to user
            self._conversation.append({"role": "assistant", "content": action})
            self._evaluate(action)

        if self._step_count >= self.max_steps and not self.done:
            # Timed out — partial credit for tools called
            self.done = True
            if self._tool_called:
                self.rewards = {0: 0.25}  # did something but didn't finish
            else:
                self.rewards = {0: 0.0}

    def _evaluate(self, response: str):
        """Score based on skill type."""
        response_lower = response.lower()

        if self._scenario.skill == "communicate":
            # Skill A: Did agent call a tool AND communicate result?
            tool_score = 1.0 if self._tool_called else 0.0

            if self._scenario.verify_keywords:
                matched = sum(1 for kw in self._scenario.verify_keywords if kw.lower() in response_lower)
                comm_score = matched / len(self._scenario.verify_keywords)
            else:
                comm_score = 1.0 if len(response.strip()) > 10 else 0.0

            self.rewards = {0: 0.5 * tool_score + 0.5 * comm_score}

        elif self._scenario.skill == "recovery":
            # Skill B: Did the agent successfully enable the target service?
            # Check if the service is actually on now in the DB
            target_service = None
            for svc in ["wifi", "cellular", "location"]:
                if svc in self._scenario.user_message.lower() or \
                   svc in self._scenario.description.lower():
                    target_service = svc
                    break
            if "internet" in self._scenario.user_message.lower() or "wifi" in self._scenario.user_message.lower():
                target_service = "wifi"
            if "signal" in self._scenario.user_message.lower() or "cellular" in self._scenario.user_message.lower():
                target_service = "cellular"
            if "gps" in self._scenario.user_message.lower() or "location" in self._scenario.user_message.lower():
                target_service = "location"

            # Did the service actually get turned on?
            service_on = self._tools.db["settings"].get(target_service, False) if target_service else False

            if service_on:
                action_score = 1.0  # service is on — success regardless of how
            else:
                # Service still off — check if agent at least tried
                has_set_low_battery = "set_low_battery_mode_status" in self._all_tools
                if has_set_low_battery:
                    action_score = 0.5  # diagnosed the blocker
                elif self._tool_called:
                    action_score = 0.25  # tried but didn't recover
                else:
                    action_score = 0.0

            if self._scenario.verify_keywords:
                matched = sum(1 for kw in self._scenario.verify_keywords if kw.lower() in response_lower)
                comm_score = matched / len(self._scenario.verify_keywords)
            else:
                comm_score = 1.0 if len(response.strip()) > 10 else 0.0

            self.rewards = {0: 0.6 * action_score + 0.4 * comm_score}

        elif self._scenario.skill == "combined":
            # Did agent get the location successfully?
            location_on = self._tools.db["settings"].get("location", False)
            has_location_call = "get_current_location" in self._all_tools

            if location_on and has_location_call:
                action_score = 1.0  # got location
            elif has_location_call:
                action_score = 0.25  # tried but location was off
            elif "set_low_battery_mode_status" in self._all_tools:
                action_score = 0.25  # started recovery but didn't finish
            else:
                action_score = 0.0

            if self._scenario.verify_keywords:
                matched = sum(1 for kw in self._scenario.verify_keywords if kw.lower() in response_lower)
                comm_score = matched / len(self._scenario.verify_keywords)
            else:
                comm_score = 1.0 if len(response.strip()) > 10 else 0.0

            self.rewards = {0: 0.5 * action_score + 0.5 * comm_score}

        self.done = True

    def _parse_tool_call(self, action):
        if not action: return None
        try:
            p = json.loads(action)
            if isinstance(p, dict):
                if "name" in p and "arguments" in p: return p
                if "function" in p:
                    fn = p["function"]
                    return {"name": fn.get("name", ""), "arguments": fn.get("arguments", {})}
        except: pass
        try:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', action)
            if m:
                name = m.group(1)
                am = re.search(r'"arguments"\s*:\s*(\{[^}]*\})', action)
                args = json.loads(am.group(1)) if am else {}
                return {"name": name, "arguments": args}
        except: pass
        return None


def extract_action(text, legal_actions):
    return text.strip() if text else None


# =====================================================================
# Test
# =====================================================================

if __name__ == "__main__":
    from collections import Counter

    skill_counts = Counter()
    reward_by_skill = {"communicate": [], "recovery": [], "combined": []}

    for seed in range(50):
        game = TECGameV2()
        game.reset(seed)
        s = game._scenario
        skill_counts[s.skill] += 1
        print(f"Seed {seed:>2d} [{s.skill:>12s}] {s.description:<50s} | {s.user_message[:60]}")

    print(f"\nSkill distribution: {dict(skill_counts)}")
