"""Tool-Execute-Communicate (TEC) Game — ToolSandbox-aligned training environment.

Trains the model to complete the full tool-calling loop:
  1. Call the right tool with right arguments
  2. Read the tool result
  3. Communicate the result back to the user

Designed to address the #1 failure mode on ToolSandbox: agent scores 0.5 because
it executes the tool correctly but never tells the user the answer.

Uses ToolSandbox's actual tool schemas and data patterns. Single-turn format
for efficient GRPO training.

Usage:
    from tec_game import TECGame, TEC_SYSTEM_PROMPT, extract_action

    game = TECGame()
    game.reset(seed=42)
    prompt = game.observe(0)
    game.step(action)  # agent's response (tool call JSON or text)
"""

import random
import json
import re
import copy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# =====================================================================
# System prompt (matches ToolSandbox format)
# =====================================================================

SYSTEM_PROMPT = (
    "You are a helpful phone assistant. You have access to tools to help the user.\n"
    "Don't make assumptions about what values to plug into functions. "
    "Ask for clarification if a user request is ambiguous.\n\n"
    "IMPORTANT: After using a tool and receiving its result, you MUST respond to "
    "the user with the information they asked for. Do not just call the tool — "
    "always follow up with a natural language response containing the answer."
)


# =====================================================================
# Tool schemas (matching ToolSandbox exactly)
# =====================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_contacts",
            "description": "Search for contacts by name, phone number, or relationship.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Contact name to search for"},
                    "phone_number": {"type": "string", "description": "Phone number to search for"},
                    "relationship": {"type": "string", "description": "Relationship to search for"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_contact",
            "description": "Add a new contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Contact name"},
                    "phone_number": {"type": "string", "description": "Phone number"},
                    "relationship": {"type": "string", "description": "Relationship (e.g., friend, boss, family)"},
                },
                "required": ["name", "phone_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_contact",
            "description": "Modify an existing contact by person_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "string", "description": "The contact's person_id"},
                    "name": {"type": "string"},
                    "phone_number": {"type": "string"},
                    "relationship": {"type": "string"},
                },
                "required": ["person_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_contact",
            "description": "Remove a contact by person_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "string", "description": "The contact's person_id"},
                },
                "required": ["person_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wifi_status",
            "description": "Check if WiFi is currently on or off.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_wifi_status",
            "description": "Turn WiFi on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {"type": "boolean", "description": "True to turn on, False to turn off"},
                },
                "required": ["on"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cellular_service_status",
            "description": "Check if cellular service is on or off.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_cellular_service_status",
            "description": "Turn cellular service on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {"type": "boolean", "description": "True to turn on, False to turn off"},
                },
                "required": ["on"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_reminder",
            "description": "Add a reminder with content and optional timestamp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Reminder content"},
                    "reminder_timestamp": {"type": "integer", "description": "Unix timestamp for the reminder"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reminder",
            "description": "Search for reminders by content or time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to search for"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_with_phone_number",
            "description": "Send a text message to a phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "Recipient phone number"},
                    "content": {"type": "string", "description": "Message content"},
                },
                "required": ["phone_number", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_messages",
            "description": "Search messages by content or sender.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to search for"},
                    "sender_phone_number": {"type": "string", "description": "Sender's phone number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_timestamp",
            "description": "Get the current Unix timestamp.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timestamp_to_datetime_info",
            "description": "Convert a Unix timestamp to year, month, day, hour, minute, second, weekday.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "number", "description": "Unix timestamp"},
                },
                "required": ["timestamp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "datetime_info_to_timestamp",
            "description": "Convert date/time components to a Unix timestamp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"}, "month": {"type": "integer"},
                    "day": {"type": "integer"}, "hour": {"type": "integer"},
                    "minute": {"type": "integer"}, "second": {"type": "integer"},
                },
                "required": ["year", "month", "day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shift_timestamp",
            "description": "Shift a timestamp by a given number of days, hours, minutes, seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "number"}, "days": {"type": "integer"},
                    "hours": {"type": "integer"}, "minutes": {"type": "integer"},
                },
                "required": ["timestamp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_location_service_status",
            "description": "Check if location service is on or off.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_location_service_status",
            "description": "Turn location service on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {"type": "boolean", "description": "True to turn on, False to turn off"},
                },
                "required": ["on"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_low_battery_mode_status",
            "description": "Check if low battery mode is on or off.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_low_battery_mode_status",
            "description": "Turn low battery mode on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {"type": "boolean", "description": "True to turn on, False to turn off"},
                },
                "required": ["on"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_location",
            "description": "Get the current GPS coordinates (latitude and longitude). Requires location service to be enabled.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# =====================================================================
# Simulated database
# =====================================================================

FIRST_NAMES = ["Homer", "Marge", "Lisa", "Bart", "Maggie", "Ned", "Maude",
               "Apu", "Krusty", "Moe", "Lenny", "Carl", "Barney", "Smithers",
               "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
LAST_NAMES = ["S", "B", "T", "F", "N", "V", "K", "P", "L", "M"]
RELATIONSHIPS = ["friend", "boss", "colleague", "family", "neighbor"]
REMINDER_CONTENTS = [
    "Buy groceries", "Call dentist", "Pick up kids", "Submit report",
    "Pay rent", "Gym session", "Team meeting", "Buy birthday gift",
    "Car maintenance", "Renew subscription", "Book flights",
    "Doctor appointment", "Clean house", "Prepare presentation",
]
MESSAGE_CONTENTS = [
    "Hey, want to grab lunch?", "Don't forget the meeting at 3pm",
    "Can you pick up some milk?", "Happy birthday!",
    "Want some GPUs?", "The project deadline is tomorrow",
    "Are you free this weekend?", "I sent the document",
    "Running late, be there in 10", "Great job on the presentation!",
]


def _gen_phone():
    return f"+1{random.randint(1000000000, 9999999999)}"


def _gen_person_id():
    return f"{random.randint(10000000, 99999999):08x}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(100000000000, 999999999999):012x}"


def generate_db(rng: random.Random) -> Dict[str, Any]:
    """Generate a random phone assistant database state."""
    n_contacts = rng.randint(3, 6)
    contacts = []
    for _ in range(n_contacts):
        contacts.append({
            "person_id": _gen_person_id(),
            "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "phone_number": _gen_phone(),
            "relationship": rng.choice(RELATIONSHIPS),
            "is_self": False,
        })

    # Add "self" contact
    self_contact = {
        "person_id": _gen_person_id(),
        "name": "Me",
        "phone_number": _gen_phone(),
        "relationship": "self",
        "is_self": True,
    }
    contacts.append(self_contact)

    messages = []
    base_ts = 1711000000  # ~March 2024
    for i in range(rng.randint(2, 5)):
        sender = rng.choice(contacts[:n_contacts])
        messages.append({
            "message_id": _gen_person_id(),
            "sender_phone_number": sender["phone_number"],
            "sender_name": sender["name"],
            "content": rng.choice(MESSAGE_CONTENTS),
            "creation_timestamp": base_ts + i * 3600,
        })

    reminders = []
    for i in range(rng.randint(1, 3)):
        reminders.append({
            "reminder_id": _gen_person_id(),
            "content": rng.choice(REMINDER_CONTENTS),
            "reminder_timestamp": base_ts + (i + 1) * 86400,
            "creation_timestamp": base_ts - 86400,
        })

    settings = {
        "wifi": rng.choice([True, False]),
        "cellular": rng.choice([True, False]),
        "location": rng.choice([True, False]),
        "low_battery_mode": rng.choice([True, False]),
    }

    return {
        "contacts": contacts,
        "messages": messages,
        "reminders": reminders,
        "settings": settings,
        "current_timestamp": base_ts + 10 * 3600,
    }


# =====================================================================
# Tool executor
# =====================================================================

class TECToolExecutor:
    """Executes ToolSandbox-style tools against the simulated database."""

    def __init__(self, db: Dict[str, Any]):
        self.db = db

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool call and return the result."""
        method = getattr(self, f"_tool_{tool_name}", None)
        if method is None:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            return method(**arguments)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def _tool_search_contacts(self, name=None, phone_number=None, relationship=None):
        results = []
        for c in self.db["contacts"]:
            if name and name.lower() not in c["name"].lower():
                continue
            if phone_number and phone_number != c["phone_number"]:
                continue
            if relationship and relationship.lower() != c["relationship"].lower():
                continue
            results.append(c)
        return results

    def _tool_add_contact(self, name, phone_number, relationship=None):
        pid = _gen_person_id()
        contact = {"person_id": pid, "name": name, "phone_number": phone_number,
                    "relationship": relationship or "", "is_self": False}
        self.db["contacts"].append(contact)
        return pid

    def _tool_modify_contact(self, person_id, name=None, phone_number=None, relationship=None):
        for c in self.db["contacts"]:
            if c["person_id"] == person_id:
                if name: c["name"] = name
                if phone_number: c["phone_number"] = phone_number
                if relationship: c["relationship"] = relationship
                return None
        return "Error: Contact not found"

    def _tool_remove_contact(self, person_id):
        self.db["contacts"] = [c for c in self.db["contacts"] if c["person_id"] != person_id]
        return None

    def _tool_get_wifi_status(self):
        return self.db["settings"]["wifi"]

    def _tool_set_wifi_status(self, on):
        if on and self.db["settings"]["low_battery_mode"]:
            raise PermissionError("Wifi cannot be turned on in low battery mode")
        self.db["settings"]["wifi"] = on
        return None

    def _tool_get_cellular_service_status(self):
        return self.db["settings"]["cellular"]

    def _tool_set_cellular_service_status(self, on):
        if on and self.db["settings"]["low_battery_mode"]:
            raise PermissionError("Cellular service cannot be turned on in low battery mode")
        self.db["settings"]["cellular"] = on
        return None

    def _tool_add_reminder(self, content, reminder_timestamp=None):
        rid = _gen_person_id()
        self.db["reminders"].append({
            "reminder_id": rid,
            "content": content,
            "reminder_timestamp": reminder_timestamp or self.db["current_timestamp"] + 3600,
            "creation_timestamp": self.db["current_timestamp"],
        })
        return rid

    def _tool_search_reminder(self, content=None):
        results = []
        for r in self.db["reminders"]:
            if content and content.lower() not in r["content"].lower():
                continue
            results.append(r)
        return results

    def _tool_send_message_with_phone_number(self, phone_number, content):
        if not self.db["settings"]["cellular"]:
            raise ConnectionError("Cellular service is not enabled")
        mid = _gen_person_id()
        self.db["messages"].append({
            "message_id": mid,
            "sender_phone_number": self.db["contacts"][-1]["phone_number"],
            "sender_name": "Me",
            "content": content,
            "recipient_phone_number": phone_number,
            "creation_timestamp": self.db["current_timestamp"],
        })
        return mid

    def _tool_get_current_timestamp(self):
        return self.db["current_timestamp"]

    def _tool_timestamp_to_datetime_info(self, timestamp):
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        return {"year": dt.year, "month": dt.month, "day": dt.day,
                "hour": dt.hour, "minute": dt.minute, "second": dt.second,
                "weekday": dt.strftime("%A")}

    def _tool_datetime_info_to_timestamp(self, year, month, day, hour=0, minute=0, second=0):
        import datetime
        dt = datetime.datetime(year, month, day, hour, minute, second)
        return dt.timestamp()

    def _tool_shift_timestamp(self, timestamp, days=0, hours=0, minutes=0, seconds=0):
        return timestamp + days * 86400 + hours * 3600 + minutes * 60 + seconds

    def _tool_get_location_service_status(self):
        return self.db["settings"]["location"]

    def _tool_set_location_service_status(self, on):
        if on and self.db["settings"]["low_battery_mode"]:
            raise PermissionError("Location service cannot be turned on in low battery mode")
        self.db["settings"]["location"] = on
        return None

    def _tool_get_low_battery_mode_status(self):
        return self.db["settings"]["low_battery_mode"]

    def _tool_set_low_battery_mode_status(self, on):
        self.db["settings"]["low_battery_mode"] = on
        return None

    def _tool_get_current_location(self):
        if not self.db["settings"]["location"]:
            raise PermissionError("Location service is not enabled.")
        return {"latitude": 37.334606, "longitude": -122.009102}

    def _tool_search_messages(self, content=None, sender_phone_number=None):
        results = []
        for m in self.db["messages"]:
            if content and content.lower() not in m["content"].lower():
                continue
            if sender_phone_number and sender_phone_number != m.get("sender_phone_number"):
                continue
            results.append(m)
        return results


# =====================================================================
# Scenario generation
# =====================================================================

@dataclass
class TECScenario:
    """A Tool-Execute-Communicate scenario."""
    user_message: str
    expected_tool: str
    expected_args: Dict[str, Any]
    expected_answer_keywords: List[str]  # keywords that must appear in agent's response
    difficulty: str  # "easy", "medium", "hard"
    description: str = ""


def generate_scenario(seed: int) -> Tuple[TECScenario, Dict[str, Any]]:
    """Generate a random TEC scenario with database."""
    rng = random.Random(seed)
    db = generate_db(rng)

    scenario_types = [
        # Easy (10%) - single tool + communicate
        ("contact_lookup", 2),
        ("setting_query", 2),
        ("setting_change", 2),
        ("contact_add", 2),
        ("message_search", 2),
        # Medium (30%) - 2-tool chains
        ("send_message_to_contact", 6),
        ("find_sender_name", 6),
        ("contact_relationship", 4),
        ("reminder_add", 4),
        ("setting_change_blocked", 5),
        ("batch_modify_contacts", 5),         # search → modify each result
        # Hard (60%) - 3+ tool chains, canonicalization, error recovery
        ("error_recovery_full_chain", 15),    # error → diagnose → fix prereq → retry → communicate
        ("temporal_reminder", 12),            # get_timestamp → compute time → add_reminder
        ("recency_query", 10),               # get_timestamp → search with time bounds → communicate
        ("modify_contact_from_message", 8),
        ("multi_hop_location", 8),           # fix settings → get_location → communicate
        ("reminder_with_contact_check", 7),
    ]

    weights = [w for _, w in scenario_types]
    chosen = rng.choices([t for t, _ in scenario_types], weights=weights, k=1)[0]

    if chosen == "contact_lookup":
        contact = rng.choice(db["contacts"][:-1])  # exclude self
        scenario = TECScenario(
            user_message=f"What is {contact['name']}'s phone number?",
            expected_tool="search_contacts",
            expected_args={"name": contact["name"]},
            expected_answer_keywords=[contact["phone_number"]],
            difficulty="easy",
            description="Contact phone number lookup",
        )

    elif chosen == "setting_query":
        setting = rng.choice(["wifi", "cellular"])
        tool_name = "get_wifi_status" if setting == "wifi" else "get_cellular_service_status"
        value = db["settings"][setting]
        question = f"Is my {'WiFi' if setting == 'wifi' else 'cellular service'} on?"
        keywords = ["on" if value else "off", "yes" if value else "no"]
        scenario = TECScenario(
            user_message=question,
            expected_tool=tool_name,
            expected_args={},
            expected_answer_keywords=keywords,
            difficulty="easy",
            description=f"Check {setting} status",
        )

    elif chosen == "setting_change":
        setting = rng.choice(["wifi", "cellular"])
        action = rng.choice(["on", "off"])
        tool_name = "set_wifi_status" if setting == "wifi" else "set_cellular_service_status"
        # Make sure low battery mode is off so the change succeeds
        db["settings"]["low_battery_mode"] = False
        msg = f"Turn {'on' if action == 'on' else 'off'} my {'WiFi' if setting == 'wifi' else 'cellular service'}"
        keywords = ["done", "turned", setting, action]
        scenario = TECScenario(
            user_message=msg,
            expected_tool=tool_name,
            expected_args={"on": action == "on"},
            expected_answer_keywords=[],  # any confirmation is fine
            difficulty="easy",
            description=f"Toggle {setting} {action}",
        )

    elif chosen == "contact_add":
        new_name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        new_phone = _gen_phone()
        scenario = TECScenario(
            user_message=f"Add {new_name} to my contacts, phone number {new_phone}",
            expected_tool="add_contact",
            expected_args={"name": new_name, "phone_number": new_phone},
            expected_answer_keywords=[new_name],
            difficulty="easy",
            description="Add new contact",
        )

    elif chosen == "reminder_add":
        content = rng.choice(REMINDER_CONTENTS)
        scenario = TECScenario(
            user_message=f"Remind me to {content.lower()}",
            expected_tool="add_reminder",
            expected_args={"content": content},
            expected_answer_keywords=[content.lower().split()[0]],  # first word
            difficulty="easy",
            description="Add reminder",
        )

    elif chosen == "contact_relationship":
        contact = rng.choice(db["contacts"][:-1])
        scenario = TECScenario(
            user_message=f"What is my relationship with {contact['name']}?",
            expected_tool="search_contacts",
            expected_args={"name": contact["name"]},
            expected_answer_keywords=[contact["relationship"]],
            difficulty="medium",
            description="Look up contact relationship",
        )

    elif chosen == "message_search":
        if db["messages"]:
            msg = rng.choice(db["messages"])
            keyword = msg["content"].split()[0]
            scenario = TECScenario(
                user_message=f"Find messages containing '{keyword}'",
                expected_tool="search_messages",
                expected_args={"content": keyword},
                expected_answer_keywords=[keyword],
                difficulty="medium",
                description="Search messages by content",
            )
        else:
            # fallback
            scenario = TECScenario(
                user_message="Do I have any reminders?",
                expected_tool="search_reminder",
                expected_args={},
                expected_answer_keywords=[],
                difficulty="easy",
                description="Check reminders",
            )

    elif chosen == "setting_change_blocked":
        # Low battery mode blocks WiFi/cellular
        db["settings"]["low_battery_mode"] = True
        db["settings"]["wifi"] = False
        setting = rng.choice(["wifi", "cellular"])
        tool_name = "set_wifi_status" if setting == "wifi" else "set_cellular_service_status"
        msg = f"Turn on my {'WiFi' if setting == 'wifi' else 'cellular service'}"
        # The model should try, get PermissionError, then tell the user
        keywords = ["low battery", "cannot", "battery mode"]
        scenario = TECScenario(
            user_message=msg,
            expected_tool=tool_name,
            expected_args={"on": True},
            expected_answer_keywords=keywords,
            difficulty="medium",
            description=f"Toggle {setting} blocked by low battery",
        )

    elif chosen == "send_message_to_contact":
        # Must: search_contacts → get phone → send_message_with_phone_number
        contact = rng.choice(db["contacts"][:-1])
        msg_content = rng.choice(MESSAGE_CONTENTS)
        db["settings"]["cellular"] = True  # ensure sending works
        db["settings"]["low_battery_mode"] = False
        scenario = TECScenario(
            user_message=f"Send a message to {contact['name']} saying \"{msg_content}\"",
            expected_tool="search_contacts",  # first step
            expected_args={"name": contact["name"]},
            expected_answer_keywords=["sent", "message", contact["name"]],
            difficulty="hard",
            description="Send message to contact by name (multi-tool)",
        )

    elif chosen == "find_sender_name":
        # Must: search_messages → extract sender_phone → search_contacts
        if db["messages"]:
            msg = rng.choice(db["messages"])
            keyword = msg["content"].split()[0]
            # Ensure the sender is in contacts
            sender_phone = msg["sender_phone_number"]
            sender_contact = None
            for c in db["contacts"]:
                if c["phone_number"] == sender_phone:
                    sender_contact = c
                    break
            if sender_contact:
                scenario = TECScenario(
                    user_message=f"Who sent me the message about '{keyword}'?",
                    expected_tool="search_messages",
                    expected_args={"content": keyword},
                    expected_answer_keywords=[sender_contact["name"]],
                    difficulty="hard",
                    description="Find message sender name (multi-tool)",
                )
            else:
                # Fallback to simple contact lookup
                contact = rng.choice(db["contacts"][:-1])
                scenario = TECScenario(
                    user_message=f"What is {contact['name']}'s phone number?",
                    expected_tool="search_contacts",
                    expected_args={"name": contact["name"]},
                    expected_answer_keywords=[contact["phone_number"]],
                    difficulty="easy",
                    description="Contact phone number lookup",
                )
        else:
            contact = rng.choice(db["contacts"][:-1])
            scenario = TECScenario(
                user_message=f"What is {contact['name']}'s phone number?",
                expected_tool="search_contacts",
                expected_args={"name": contact["name"]},
                expected_answer_keywords=[contact["phone_number"]],
                difficulty="easy",
                description="Contact phone number lookup",
            )

    elif chosen == "error_recovery_chain":
        # Low battery → must disable low battery → enable WiFi/cellular → complete task
        db["settings"]["low_battery_mode"] = True
        db["settings"]["wifi"] = False
        db["settings"]["cellular"] = False
        target = rng.choice(["wifi", "cellular"])
        if target == "wifi":
            scenario = TECScenario(
                user_message="I need WiFi turned on. Can you do that?",
                expected_tool="set_wifi_status",
                expected_args={"on": True},
                expected_answer_keywords=["wifi", "on"],
                difficulty="hard",
                description="Enable WiFi (requires disabling low battery first)",
            )
        else:
            scenario = TECScenario(
                user_message="Turn on my cellular service please.",
                expected_tool="set_cellular_service_status",
                expected_args={"on": True},
                expected_answer_keywords=["cellular", "on"],
                difficulty="hard",
                description="Enable cellular (requires disabling low battery first)",
            )

    elif chosen == "modify_contact_from_message":
        # Search messages → find sender → modify their contact info
        if db["messages"]:
            msg = rng.choice(db["messages"])
            sender_phone = msg["sender_phone_number"]
            sender_contact = None
            for c in db["contacts"]:
                if c["phone_number"] == sender_phone:
                    sender_contact = c
                    break
            if sender_contact:
                new_phone = _gen_phone()
                keyword = msg["content"].split()[0]
                scenario = TECScenario(
                    user_message=f"The person who texted me about '{keyword}' has a new number: {new_phone}. Update their contact.",
                    expected_tool="search_messages",
                    expected_args={"content": keyword},
                    expected_answer_keywords=[sender_contact["name"], "updated"],
                    difficulty="hard",
                    description="Update contact from message sender (multi-tool)",
                )
            else:
                contact = rng.choice(db["contacts"][:-1])
                new_phone = _gen_phone()
                scenario = TECScenario(
                    user_message=f"Change {contact['name']}'s phone to {new_phone}",
                    expected_tool="search_contacts",
                    expected_args={"name": contact["name"]},
                    expected_answer_keywords=[contact["name"], "updated"],
                    difficulty="medium",
                    description="Update contact phone number",
                )
        else:
            contact = rng.choice(db["contacts"][:-1])
            new_phone = _gen_phone()
            scenario = TECScenario(
                user_message=f"Change {contact['name']}'s phone to {new_phone}",
                expected_tool="search_contacts",
                expected_args={"name": contact["name"]},
                expected_answer_keywords=[contact["name"], "updated"],
                difficulty="medium",
                description="Update contact phone number",
            )

    elif chosen == "reminder_with_contact_check":
        contact = rng.choice(db["contacts"][:-1])
        reminder_content = rng.choice(["Call", "Email", "Meet with", "Text"])
        scenario = TECScenario(
            user_message=f"Set a reminder to {reminder_content.lower()} {contact['name']}. But first check if they're in my contacts.",
            expected_tool="search_contacts",
            expected_args={"name": contact["name"]},
            expected_answer_keywords=[contact["name"], "reminder"],
            difficulty="hard",
            description="Verify contact then add reminder (multi-tool)",
        )

    elif chosen == "error_recovery_full_chain":
        # Must: try action → get error → diagnose → fix prereq → retry → succeed → COMMUNICATE
        db["settings"]["low_battery_mode"] = True
        db["settings"]["wifi"] = False
        db["settings"]["cellular"] = False
        db["settings"]["location"] = False
        target = rng.choice(["wifi", "cellular", "location"])
        tool_map = {"wifi": "set_wifi_status", "cellular": "set_cellular_service_status", "location": "set_location_service_status"}
        label_map = {"wifi": "WiFi", "cellular": "cellular service", "location": "location service"}
        scenario = TECScenario(
            user_message=f"I need {label_map[target]} turned on right now.",
            expected_tool=tool_map[target],
            expected_args={"on": True},
            expected_answer_keywords=[label_map[target].split()[0].lower(), "on", "enabled"],
            difficulty="hard",
            description=f"Full error recovery chain for {target}",
        )

    elif chosen == "temporal_reminder":
        # Must: get_current_timestamp → compute offset → add_reminder with correct timestamp
        days_ahead = rng.choice([1, 2, 3, 7])
        hour = rng.choice([9, 12, 15, 17])
        day_words = {1: "tomorrow", 2: "the day after tomorrow", 3: "in 3 days", 7: "next week"}
        content = rng.choice(REMINDER_CONTENTS)
        scenario = TECScenario(
            user_message=f"Remind me to {content.lower()} {day_words[days_ahead]} at {hour}:00.",
            expected_tool="add_reminder",
            expected_args={"content": content},
            expected_answer_keywords=[content.lower().split()[0], "reminder", "set"],
            difficulty="hard",
            description=f"Temporal reminder ({day_words[days_ahead]})",
        )

    elif chosen == "recency_query":
        # Must: get_current_timestamp → search with time bounds → communicate result
        query_type = rng.choice(["reminder", "message"])
        if query_type == "reminder" and db["reminders"]:
            target = rng.choice(db["reminders"])
            scenario = TECScenario(
                user_message="What's my most recent reminder about?",
                expected_tool="search_reminder",
                expected_args={},
                expected_answer_keywords=[target["content"].split()[0].lower()],
                difficulty="hard",
                description="Find most recent reminder (requires timestamp reasoning)",
            )
        elif db["messages"]:
            latest = max(db["messages"], key=lambda m: m["creation_timestamp"])
            scenario = TECScenario(
                user_message="What did my last text message say?",
                expected_tool="search_messages",
                expected_args={},
                expected_answer_keywords=[latest["content"].split()[0]],
                difficulty="hard",
                description="Find most recent message (requires timestamp reasoning)",
            )
        else:
            contact = rng.choice(db["contacts"][:-1])
            scenario = TECScenario(
                user_message=f"What is {contact['name']}'s phone number?",
                expected_tool="search_contacts",
                expected_args={"name": contact["name"]},
                expected_answer_keywords=[contact["phone_number"]],
                difficulty="easy",
                description="Contact phone number lookup",
            )

    elif chosen == "batch_modify_contacts":
        # Must: search_contacts(relationship=X) → get list → modify_contact for EACH
        rel = rng.choice(["friend", "colleague"])
        new_rel = rng.choice(["enemy", "acquaintance", "stranger"])
        # Ensure some contacts have this relationship
        count = 0
        for c in db["contacts"][:-1]:
            if rng.random() < 0.4:
                c["relationship"] = rel
                count += 1
        if count == 0:
            db["contacts"][0]["relationship"] = rel
            count = 1
        scenario = TECScenario(
            user_message=f"Change all my {rel}s to {new_rel}s.",
            expected_tool="search_contacts",
            expected_args={"relationship": rel},
            expected_answer_keywords=[new_rel, "updated"],
            difficulty="hard",
            description=f"Batch modify contacts ({rel} → {new_rel})",
        )

    elif chosen == "multi_hop_location":
        # Must: possibly fix low battery → enable location → get_current_location → communicate
        db["settings"]["low_battery_mode"] = rng.choice([True, False])
        db["settings"]["location"] = False
        scenario = TECScenario(
            user_message="What are my current GPS coordinates?",
            expected_tool="get_current_location",
            expected_args={},
            expected_answer_keywords=["latitude", "longitude"],
            difficulty="hard",
            description="Get location (may require fixing settings first)",
        )

    else:
        raise ValueError(f"Unknown scenario type: {chosen}")

    return scenario, db


# =====================================================================
# Game environment
# =====================================================================

class TECGame:
    """Tool-Execute-Communicate game for GRPO training.

    Single-turn: user asks → agent calls tool → tool result → agent responds.
    Multi-step within one "turn" from GRPO's perspective.

    Implements the tool-calling game interface (supports_structured_messages).
    """

    supports_structured_messages = True

    def __init__(self):
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        self._scenario: Optional[TECScenario] = None
        self._db: Optional[Dict[str, Any]] = None
        self._tools: Optional[TECToolExecutor] = None
        self._conversation: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._tool_called: bool = False
        self._tool_result: Optional[str] = None
        self._correct_tool: bool = False
        self._all_tools_called: List[str] = []
        self.max_steps: int = 10

    def reset(self, seed: int) -> None:
        # Use seed for scenario TYPE selection, but add randomness to DB details
        # so different rollouts in the same GRPO group see slightly different states.
        # This gives GRPO intra-group variance for advantage computation.
        self._scenario, self._db = generate_scenario(seed)

        # Randomize DB details that don't affect the scenario's core task
        # This makes each rollout unique even with the same seed
        import time
        noise_rng = random.Random(seed * 1000003 + int(time.time() * 1000) % 1000000 + id(self))

        # Shuffle contact order (affects which contact is "first" in search results)
        noise_rng.shuffle(self._db["contacts"][:-1])  # don't shuffle self-contact

        # Randomly toggle non-critical settings for non-settings scenarios
        if "setting" not in self._scenario.description.lower() and \
           "battery" not in self._scenario.description.lower() and \
           "location" not in self._scenario.description.lower() and \
           "wifi" not in self._scenario.description.lower() and \
           "cellular" not in self._scenario.description.lower():
            self._db["settings"]["wifi"] = noise_rng.choice([True, False])
            self._db["settings"]["cellular"] = noise_rng.choice([True, False])
            self._db["settings"]["low_battery_mode"] = noise_rng.choice([True, False])

        # Add 0-2 random extra contacts (noise)
        for _ in range(noise_rng.randint(0, 2)):
            self._db["contacts"].insert(-1, {
                "person_id": _gen_person_id(),
                "name": f"{noise_rng.choice(FIRST_NAMES)} {noise_rng.choice(LAST_NAMES)}",
                "phone_number": _gen_phone(),
                "relationship": noise_rng.choice(RELATIONSHIPS),
                "is_self": False,
            })

        self._tools = TECToolExecutor(copy.deepcopy(self._db))
        self._conversation = [{"role": "user", "content": self._scenario.user_message}]
        self._step_count = 0
        self._tool_called = False
        self._tool_result = None
        self._correct_tool = False
        self._all_tools_called = []

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def get_messages(self) -> List[Dict[str, Any]]:
        return list(self._conversation)

    def observe(self, player_id: int) -> str:
        if self.done:
            return ""
        if self._conversation:
            return self._conversation[-1].get("text", "")
        return ""

    def legal_actions(self) -> List[str]:
        return []

    def step(self, action: Optional[str]) -> None:
        """Process agent action (tool call JSON or text response)."""
        if self.done or action is None:
            return

        self._step_count += 1

        # Parse the action
        tool_call = self._parse_tool_call(action)

        if tool_call and not self._tool_called:
            # Agent is making a tool call
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})

            # Execute the tool
            result = self._tools.execute(tool_name, tool_args)
            result_str = json.dumps(result, default=str) if not isinstance(result, str) else result

            self._tool_called = True
            self._tool_result = result_str
            self._all_tools_called.append(tool_name)

            # Check if correct tool was called
            if tool_name == self._scenario.expected_tool:
                self._correct_tool = True

            # Add tool call and result to conversation
            self._conversation.append({
                "role": "assistant",
                "content": action,
                "tool_call": {"name": tool_name, "arguments": tool_args},
            })
            self._conversation.append({
                "role": "tool",
                "content": result_str,
            })

            # Don't end yet — wait for agent to communicate the result
            return

        elif not tool_call and self._tool_called:
            # Agent is responding to user after tool call — this is what we want!
            self._conversation.append({"role": "assistant", "content": action})
            self._evaluate(action)
            return

        elif not tool_call and not self._tool_called:
            # Agent responded without calling a tool
            # Check if the scenario required a tool call
            self._conversation.append({"role": "assistant", "content": action})
            # For setting changes, no-tool response is always wrong
            # For queries, check if the answer happens to be correct anyway
            self._evaluate_no_tool(action)
            return

        elif tool_call and self._tool_called:
            # Agent is calling additional tools — execute and track
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            result = self._tools.execute(tool_name, tool_args)
            result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
            self._tool_result = result_str  # update with latest result
            self._all_tools_called.append(tool_name)
            self._conversation.append({
                "role": "assistant", "content": action,
                "tool_call": {"name": tool_name, "arguments": tool_args},
            })
            self._conversation.append({"role": "tool", "content": result_str})
            return

        # Max steps — fail if we didn't complete
        if self._step_count >= self.max_steps:
            self.done = True
            self.rewards = {0: 0.0}

    def _evaluate(self, response_text: str):
        """Evaluate after agent responded to user post-tool-call."""
        tool_score = 1.0 if self._correct_tool else 0.0

        # Check communication: does the response contain expected keywords?
        comm_score = 0.0
        if self._scenario.expected_answer_keywords:
            response_lower = response_text.lower()
            matched = sum(1 for kw in self._scenario.expected_answer_keywords
                         if kw.lower() in response_lower)
            # Require ALL keywords to match for full credit
            if len(self._scenario.expected_answer_keywords) > 0:
                comm_score = matched / len(self._scenario.expected_answer_keywords)
        else:
            # For action scenarios (set_wifi, add_contact), any non-empty response is OK
            if len(response_text.strip()) > 5:
                comm_score = 1.0

        reward = 0.5 * tool_score + 0.5 * comm_score
        self.done = True
        self.rewards = {0: reward}

    def _evaluate_no_tool(self, response_text: str):
        """Evaluate when agent responded without calling any tool."""
        # Tool score is 0 since no tool was called
        # But check if the answer is somehow correct (unlikely)
        comm_score = 0.0
        if self._scenario.expected_answer_keywords:
            response_lower = response_text.lower()
            matched = sum(1 for kw in self._scenario.expected_answer_keywords
                         if kw.lower() in response_lower)
            if matched > 0:
                comm_score = 0.5  # partial credit for guessing

        reward = comm_score * 0.5  # max 0.25 without tool call
        self.done = True
        self.rewards = {0: reward}

    def _parse_tool_call(self, action: str) -> Optional[Dict[str, Any]]:
        """Parse a tool call from the agent's action string."""
        if action is None:
            return None

        # Try JSON parse
        try:
            parsed = json.loads(action)
            if isinstance(parsed, dict):
                if "name" in parsed and "arguments" in parsed:
                    return parsed
                if "function" in parsed:
                    fn = parsed["function"]
                    return {"name": fn.get("name", ""), "arguments": fn.get("arguments", {})}
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to extract from tool_call format
        try:
            match = re.search(r'"name"\s*:\s*"([^"]+)"', action)
            if match:
                name = match.group(1)
                args_match = re.search(r'"arguments"\s*:\s*(\{[^}]*\})', action)
                args = json.loads(args_match.group(1)) if args_match else {}
                return {"name": name, "arguments": args}
        except Exception:
            pass

        return None


# =====================================================================
# Extract action function (for game_registry)
# =====================================================================

def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action from model output."""
    if text:
        return text.strip()
    return None


# =====================================================================
# Test / validation
# =====================================================================

def test_environment(n_scenarios: int = 20):
    """Test the environment generates valid scenarios with varied rewards."""
    rewards = []
    for seed in range(n_scenarios):
        game = TECGame()
        game.reset(seed)

        scenario = game._scenario
        print(f"\nSeed {seed}: [{scenario.difficulty}] {scenario.description}")
        print(f"  User: {scenario.user_message}")
        print(f"  Expected tool: {scenario.expected_tool}")
        print(f"  Expected keywords: {scenario.expected_answer_keywords}")

        # Simulate: agent calls tool, then responds
        tool_call = json.dumps({"name": scenario.expected_tool, "arguments": scenario.expected_args})
        game.step(tool_call)

        if not game.done:
            # Simulate communication
            if scenario.expected_answer_keywords:
                response = f"The answer is: {', '.join(scenario.expected_answer_keywords)}"
            else:
                response = "Done! I've completed that for you."
            game.step(response)

        reward = game.rewards.get(0, 0.0)
        rewards.append(reward)
        print(f"  Reward: {reward}")

    print(f"\n{'='*50}")
    print(f"Rewards: mean={sum(rewards)/len(rewards):.2f}, "
          f"pass={sum(1 for r in rewards if r >= 0.99)}/{n_scenarios}, "
          f"partial={sum(1 for r in rewards if 0.01 < r < 0.99)}/{n_scenarios}, "
          f"fail={sum(1 for r in rewards if r <= 0.01)}/{n_scenarios}")


if __name__ == "__main__":
    test_environment()
