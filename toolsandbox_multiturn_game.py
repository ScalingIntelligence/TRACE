"""ToolSandbox Multi-Turn Game — trains clarification, datetime, and insufficient-info skills.

Targets three key ToolSandbox failure modes:
  1. MULTIPLE_USER_TURN (avg=0.211): Model doesn't ask clarifying questions
  2. CANONICALIZATION: Model can't resolve relative dates/times using tool chains
  3. INSUFFICIENT_INFORMATION: Model hallucinates instead of recognizing missing tools

All scenarios are hard. GRPO variance comes from **hint injection**:
  - Each rollout randomly gets a skill-specific hint in the system prompt
  - During training, hints are stripped so the model learns without them
  - Hinted rollouts succeed more → creates within-group reward variance

Hint stripping: call `strip_hints_from_samples(samples)` before training loss.
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

SYSTEM_PROMPT = (
    "Don't make assumptions about what values to plug into functions. "
    "Ask for clarification if a user request is ambiguous."
)

# Marker used to identify and strip hints before training
HINT_START = "\n\n###HINT_START###\n"
HINT_END = "\n###HINT_END###"

# =====================================================================
# Skill-specific hints
# =====================================================================

HINTS = {
    "multiturn": (
        "IMPORTANT execution rules for this conversation:\n"
        "1. The user's first message is INTENTIONALLY vague. You MUST respond with a text "
        "question asking for the missing information. Do NOT call any tool yet.\n"
        "2. After the user provides the missing info, use search_contacts to look up any "
        "person by name — do NOT ask the user for phone numbers, IDs, or other details you "
        "can look up yourself.\n"
        "3. After ALL tool calls are done, you MUST send a final text message to the user "
        "confirming what you did. Include the person's name and what action was completed.\n"
        "4. Common mistake to avoid: if the user says 'to Alice', call search_contacts(name='Alice') "
        "rather than asking 'what is Alice's phone number?'"
    ),
    "datetime": (
        "IMPORTANT: You are in a sandbox environment. The date/time is NOT what you think.\n"
        "You MUST follow these steps for ANY date/time task:\n"
        "1. FIRST call get_current_timestamp() — this tells you the ACTUAL current time\n"
        "2. Then call timestamp_to_datetime_info() on that timestamp to learn today's "
        "actual year, month, day, and weekday\n"
        "3. For 'next Tuesday': compute how many days from today's weekday to Tuesday, "
        "then use shift_timestamp(days=N) or datetime_info_to_timestamp()\n"
        "4. For 'tomorrow': use shift_timestamp(days=1) on the current timestamp\n"
        "5. For 'in 2 weeks': use shift_timestamp(days=14) on the current timestamp\n"
        "NEVER use year=2023 or year=2024. NEVER hardcode any timestamp.\n"
        "After completing the task, confirm to the user what you set up."
    ),
    "insufficient": (
        "CRITICAL: Before calling ANY tool, read your tool list carefully.\n"
        "If the user asks something that requires a tool you do NOT have, you must "
        "IMMEDIATELY respond with text saying you cannot help — do NOT call any tool.\n"
        "Specifically:\n"
        "- If you need the current time but get_current_timestamp is not in your tools → "
        "say 'I don't have access to the current time'\n"
        "- If you need weather data but search_weather is not in your tools → "
        "say 'I don't have access to weather information'\n"
        "- If you need to search contacts but search_contacts is not in your tools → "
        "say 'I cannot search your contacts'\n"
        "- If you need messages but search_messages is not in your tools → "
        "say 'I cannot access your messages'\n"
        "Do NOT fabricate timestamps, dates, or other values. Do NOT call tools with "
        "made-up arguments. Just tell the user you lack the capability."
    ),
}


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
    {"type": "function", "function": {"name": "get_current_timestamp", "description": "Get the current Unix timestamp.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "datetime_info_to_timestamp", "description": "Convert date/time components to Unix timestamp.", "parameters": {"type": "object", "properties": {"year": {"type": "integer"}, "month": {"type": "integer"}, "day": {"type": "integer"}, "hour": {"type": "integer"}, "minute": {"type": "integer"}, "second": {"type": "integer"}}, "required": ["year", "month", "day", "hour", "minute", "second"]}}},
    {"type": "function", "function": {"name": "timestamp_to_datetime_info", "description": "Convert Unix timestamp to date/time components (year, month, day, hour, minute, second, weekday).", "parameters": {"type": "object", "properties": {"timestamp": {"type": "number"}}, "required": ["timestamp"]}}},
    {"type": "function", "function": {"name": "add_reminder", "description": "Add a reminder with content and optional timestamp/location.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "reminder_timestamp": {"type": "integer"}, "latitude": {"type": "number"}, "longitude": {"type": "number"}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "search_reminder", "description": "Search reminders by content, creation recency, or reminder recency.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "creation_recency": {"type": "string"}, "reminder_recency": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "modify_reminder", "description": "Modify a reminder by reminder_id.", "parameters": {"type": "object", "properties": {"reminder_id": {"type": "string"}, "content": {"type": "string"}, "reminder_timestamp": {"type": "integer"}}, "required": ["reminder_id"]}}},
    {"type": "function", "function": {"name": "remove_reminder", "description": "Remove a reminder by ID.", "parameters": {"type": "object", "properties": {"reminder_id": {"type": "string"}}, "required": ["reminder_id"]}}},
    {"type": "function", "function": {"name": "send_message_with_phone_number", "description": "Send a text message to a phone number.", "parameters": {"type": "object", "properties": {"phone_number": {"type": "string"}, "content": {"type": "string"}}, "required": ["phone_number", "content"]}}},
    {"type": "function", "function": {"name": "search_messages", "description": "Search messages by content, sender, or recency.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "sender_phone_number": {"type": "string"}, "recency": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "search_holiday", "description": "Search for a holiday by name and optional year.", "parameters": {"type": "object", "properties": {"holiday_name": {"type": "string"}, "year": {"type": "integer"}}, "required": ["holiday_name"]}}},
    {"type": "function", "function": {"name": "search_lat_lon", "description": "Search for latitude/longitude of a location by name.", "parameters": {"type": "object", "properties": {"location_name": {"type": "string"}}, "required": ["location_name"]}}},
    {"type": "function", "function": {"name": "search_weather_around_lat_lon", "description": "Get weather information around a lat/lon coordinate.", "parameters": {"type": "object", "properties": {"latitude": {"type": "number"}, "longitude": {"type": "number"}}, "required": ["latitude", "longitude"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount from one currency to another.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "timestamp_diff", "description": "Compute the difference in seconds between two timestamps.", "parameters": {"type": "object", "properties": {"timestamp1": {"type": "number"}, "timestamp2": {"type": "number"}}, "required": ["timestamp1", "timestamp2"]}}},
    {"type": "function", "function": {"name": "shift_timestamp", "description": "Shift a timestamp by a given number of days/hours/minutes/seconds.", "parameters": {"type": "object", "properties": {"timestamp": {"type": "number"}, "days": {"type": "integer"}, "hours": {"type": "integer"}, "minutes": {"type": "integer"}, "seconds": {"type": "integer"}}, "required": ["timestamp"]}}},
    {"type": "function", "function": {"name": "unit_conversion", "description": "Convert a value from one unit to another.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["amount", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "seconds_to_hours_minutes_seconds", "description": "Convert seconds to hours, minutes, seconds.", "parameters": {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}}},
]


# =====================================================================
# Data pools
# =====================================================================

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Helen",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nancy", "Oscar", "Patricia",
    "Quinn", "Rachel", "Steven", "Tina", "Ursula", "Victor", "Wendy", "Xavier",
]
LAST_INITIALS = list("ABCDEFGHKLMNPQRSTVWX")
RELATIONSHIPS = ["friend", "boss", "colleague", "family", "neighbor", "classmate"]
MESSAGE_CONTENTS = [
    "Hey, are you free for dinner tonight?", "Don't forget about the project deadline",
    "Can you send me the report?", "Happy anniversary!", "The meeting moved to 4pm",
    "I'll pick you up at 7", "Check out this new restaurant", "Running 10 minutes late",
    "Thanks for your help yesterday", "Want to grab coffee tomorrow?",
]
REMINDER_CONTENTS = [
    "Buy groceries", "Call the dentist", "Pick up dry cleaning", "Submit expense report",
    "Pay electricity bill", "Water the plants", "Schedule car maintenance",
    "Renew gym membership", "Book flight tickets", "Send birthday card",
    "Update resume", "Clean the apartment", "Return library books", "Get haircut",
    "Buy birthday present", "Prepare presentation slides", "Call insurance company",
]
LOCATIONS = [
    ("Whole Foods", 37.3318, -122.0312), ("Trader Joe's", 37.3688, -122.0363),
    ("Target", 37.3229, -121.9273), ("Costco", 37.3492, -121.9524),
    ("Home Depot", 37.3801, -122.0014), ("Walmart", 37.3566, -121.9613),
]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _pid():
    return f"{random.randint(10000000,99999999):08x}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(100000000000,999999999999):012x}"

def _phone():
    return f"+1{random.randint(2000000000,9999999999)}"

def _make_name(rng):
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_INITIALS)}"


def _make_db(rng):
    contacts = []
    used_names = set()
    for _ in range(rng.randint(3, 6)):
        name = _make_name(rng)
        while name in used_names:
            name = _make_name(rng)
        used_names.add(name)
        contacts.append({"person_id": _pid(), "name": name, "phone_number": _phone(),
                         "relationship": rng.choice(RELATIONSHIPS), "is_self": False})
    contacts.append({"person_id": _pid(), "name": "Me", "phone_number": _phone(),
                      "relationship": "self", "is_self": True})

    messages = [{"message_id": _pid(), "sender_person_id": (c := rng.choice(contacts[:-1]))["person_id"],
                 "sender_phone_number": c["phone_number"], "sender_name": c["name"],
                 "content": rng.choice(MESSAGE_CONTENTS), "creation_timestamp": 1774500000 + i * 3600}
                for i in range(rng.randint(2, 5))]

    reminders = [{"reminder_id": _pid(), "content": rng.choice(REMINDER_CONTENTS),
                  "reminder_timestamp": 1774500000 + (i+1)*86400, "creation_timestamp": 1774500000 - 86400}
                 for i in range(rng.randint(1, 3))]

    # Current timestamp: March 27, 2026, 10:00 UTC (a Friday)
    current_ts = 1774537200.0
    return {
        "contacts": contacts, "messages": messages, "reminders": reminders,
        "settings": {"wifi": True, "cellular": True, "location": True, "low_battery_mode": False},
        "current_timestamp": current_ts,
        "holidays": {"Christmas Day": 1766649600, "New Year's Day": 1735689600,
                      "Thanksgiving": 1764460800, "Independence Day": 1751587200,
                      "Labor Day": 1757116800, "Easter": 1743984000},
        "locations": {n: (lat, lon) for n, lat, lon in LOCATIONS},
    }


# =====================================================================
# Tool executor
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

    def _t_get_current_timestamp(self):
        return self.db["current_timestamp"]

    def _t_datetime_info_to_timestamp(self, year, month, day, hour, minute, second):
        import calendar, datetime as dt
        return float(calendar.timegm(dt.datetime(year, month, day, hour, minute, second).timetuple()))

    def _t_timestamp_to_datetime_info(self, timestamp):
        import datetime as dt
        d = dt.datetime.utcfromtimestamp(timestamp)
        return {"year": d.year, "month": d.month, "day": d.day,
                "hour": d.hour, "minute": d.minute, "second": d.second,
                "weekday": d.isoweekday()}  # 1=Mon, 7=Sun

    def _t_timestamp_diff(self, timestamp1, timestamp2):
        return abs(timestamp1 - timestamp2)

    def _t_shift_timestamp(self, timestamp, days=0, hours=0, minutes=0, seconds=0):
        return timestamp + days*86400 + hours*3600 + minutes*60 + seconds

    def _t_seconds_to_hours_minutes_seconds(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return {"hours": h, "minutes": m, "seconds": s}

    def _t_add_reminder(self, content, reminder_timestamp=None, latitude=None, longitude=None):
        rid = _pid()
        self.db["reminders"].append({"reminder_id": rid, "content": content,
                                      "reminder_timestamp": reminder_timestamp or self.db["current_timestamp"]+3600,
                                      "creation_timestamp": self.db["current_timestamp"],
                                      "latitude": latitude, "longitude": longitude})
        return rid

    def _t_search_reminder(self, content=None, creation_recency=None, reminder_recency=None):
        r = list(self.db["reminders"])
        if content: r = [x for x in r if content.lower() in x["content"].lower()]
        if reminder_recency == "upcoming":
            now = self.db["current_timestamp"]
            r = sorted([x for x in r if x.get("reminder_timestamp", 0) > now],
                       key=lambda x: x.get("reminder_timestamp", 0))
        elif reminder_recency == "latest":
            r = sorted(r, key=lambda x: x.get("reminder_timestamp", 0), reverse=True)
        if creation_recency == "latest":
            r = sorted(r, key=lambda x: x.get("creation_timestamp", 0), reverse=True)
        return r

    def _t_modify_reminder(self, reminder_id, content=None, reminder_timestamp=None):
        for x in self.db["reminders"]:
            if x["reminder_id"] == reminder_id:
                if content: x["content"] = content
                if reminder_timestamp: x["reminder_timestamp"] = reminder_timestamp
                return None
        return "Error: Reminder not found"

    def _t_remove_reminder(self, reminder_id):
        self.db["reminders"] = [x for x in self.db["reminders"] if x["reminder_id"] != reminder_id]
        return None

    def _t_send_message_with_phone_number(self, phone_number, content):
        if not self.db["settings"]["cellular"]:
            raise ConnectionError("Cellular service is not enabled")
        return _pid()

    def _t_search_messages(self, content=None, sender_phone_number=None, recency=None):
        r = list(self.db["messages"])
        if content: r = [m for m in r if content.lower() in m["content"].lower()]
        if sender_phone_number: r = [m for m in r if sender_phone_number == m.get("sender_phone_number")]
        if recency == "latest":
            r = sorted(r, key=lambda m: m.get("creation_timestamp", 0), reverse=True)[:1]
        elif recency == "oldest":
            r = sorted(r, key=lambda m: m.get("creation_timestamp", 0))[:1]
        return r

    def _t_search_holiday(self, holiday_name, year=None):
        for n, ts in self.db["holidays"].items():
            if holiday_name.lower() in n.lower(): return ts
        return f"Error: Holiday '{holiday_name}' not found"

    def _t_search_lat_lon(self, location_name):
        for n, (lat, lon) in self.db["locations"].items():
            if location_name.lower() in n.lower(): return {"latitude": lat, "longitude": lon}
        return f"Error: Location '{location_name}' not found"

    def _t_search_weather_around_lat_lon(self, latitude, longitude):
        return {"temperature_celsius": 18.5, "temperature_fahrenheit": 65.3,
                "humidity": 62, "description": "Partly cloudy"}

    def _t_convert_currency(self, amount, from_currency, to_currency):
        rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "CNY": 7.24, "CAD": 1.36}
        fc, tc = from_currency.upper(), to_currency.upper()
        if fc not in rates or tc not in rates: return "Error: Unsupported currency"
        return round(amount / rates[fc] * rates[tc], 2)

    def _t_unit_conversion(self, amount, from_unit, to_unit):
        convs = {("celsius","fahrenheit"): lambda x: x*9/5+32, ("fahrenheit","celsius"): lambda x: (x-32)*5/9,
                 ("miles","kilometers"): lambda x: x*1.60934, ("kilometers","miles"): lambda x: x/1.60934}
        k = (from_unit.lower(), to_unit.lower())
        if k in convs: return round(convs[k](amount), 2)
        return f"Error: Cannot convert {from_unit} to {to_unit}"


# =====================================================================
# Scenario dataclass
# =====================================================================

@dataclass
class Scenario:
    skill: str          # "multiturn" | "datetime" | "insufficient"
    hint_key: str       # Key into HINTS dict
    initial_message: str
    user_responses: List[str]
    expected_tools: List[str]
    verify_keywords: List[str]
    tool_allow_list: Optional[List[str]] = None
    description: str = ""


# =====================================================================
# MULTI-TURN CLARIFICATION scenarios (40%)
# All require 2+ user interactions before tool execution
# =====================================================================

def _mt_send_message(rng, db):
    """Send a message: ask who → ask what → search_contacts → send."""
    contact = rng.choice(db["contacts"][:-1])
    msg = rng.choice(MESSAGE_CONTENTS)
    templates = ["Send a message", "I want to text someone", "I need to send a text",
                 "Help me send a message", "Text someone for me"]
    return Scenario(
        skill="multiturn", hint_key="multiturn", initial_message=rng.choice(templates),
        user_responses=[f"To {contact['name']}", f'Say "{msg}"'],
        expected_tools=["search_contacts", "send_message_with_phone_number"],
        verify_keywords=["sent", contact["name"].split()[0].lower()],
        description=f"MT: send msg to {contact['name']}")

def _mt_add_reminder(rng, db):
    """Add reminder: ask what → ask when → datetime_info_to_timestamp → add_reminder."""
    content = rng.choice(REMINDER_CONTENTS)
    day = rng.randint(1, 28)
    hour = rng.choice([9, 14, 17])
    templates = ["Set a reminder for me", "I need a reminder", "Can you add a reminder?",
                 "Add a reminder please"]
    return Scenario(
        skill="multiturn", hint_key="multiturn", initial_message=rng.choice(templates),
        user_responses=[f"It's about {content.lower()}", f"On April {day}, 2026 at {hour}:00"],
        expected_tools=["datetime_info_to_timestamp", "add_reminder"],
        verify_keywords=["reminder", content.split()[0].lower()],
        description=f"MT: add reminder '{content}'")

def _mt_remove_contact_by_phone(rng, db):
    """Remove contact by phone: ask who → search_contacts → remove."""
    contact = rng.choice(db["contacts"][:-1])
    templates = ["I want to delete someone from my contact", "Remove a contact for me",
                 "Delete a person from contacts", "I need to remove a contact"]
    return Scenario(
        skill="multiturn", hint_key="multiturn", initial_message=rng.choice(templates),
        user_responses=[f"The one with phone number {contact['phone_number']}"],
        expected_tools=["search_contacts", "remove_contact"],
        verify_keywords=[contact["name"].split()[0].lower(), "removed"],
        description=f"MT: remove contact by phone")

def _mt_modify_contact(rng, db):
    """Modify contact: ask who → ask what → search → modify."""
    contact = rng.choice(db["contacts"][:-1])
    new_phone = _phone()
    templates = ["I need to update a contact's information", "Change someone's phone number",
                 "Can you modify a contact for me?"]
    return Scenario(
        skill="multiturn", hint_key="multiturn",
        initial_message=rng.choice(templates),
        user_responses=[f"It's {contact['name'].split()[0]}", f"Change the phone number to {new_phone}"],
        expected_tools=["search_contacts", "modify_contact"],
        verify_keywords=[contact["name"].split()[0].lower(), "updated"],
        description=f"MT: modify {contact['name']}")

def _mt_search_message_recency(rng, db):
    """Search messages by recency: ask which → search."""
    recency = rng.choice(["latest", "oldest"])
    templates = ["I wanna find a message", "Can you look up a message for me?",
                 "I need to find a text", "Search my messages"]
    return Scenario(
        skill="multiturn", hint_key="multiturn", initial_message=rng.choice(templates),
        user_responses=[f"I want the {recency} one"],
        expected_tools=["search_messages"],
        verify_keywords=["message"],
        description=f"MT: search {recency} message")

def _mt_update_relationship(rng, db):
    """Update relationship: ask who → ask new rel → search → modify."""
    contact = rng.choice(db["contacts"][:-1])
    new_rel = rng.choice([r for r in RELATIONSHIPS if r != contact["relationship"]])
    return Scenario(
        skill="multiturn", hint_key="multiturn",
        initial_message="I need to update someone's relationship in my contacts",
        user_responses=[contact["name"].split()[0], f"Change the relationship to {new_rel}"],
        expected_tools=["search_contacts", "modify_contact"],
        verify_keywords=[contact["name"].split()[0].lower(), new_rel],
        description=f"MT: update {contact['name'].split()[0]}'s relationship")

def _mt_add_reminder_with_location(rng, db):
    """Add reminder at location: ask what → ask when → ask where → tools."""
    content = rng.choice(REMINDER_CONTENTS)
    loc_name, _, _ = rng.choice(LOCATIONS)
    day = rng.randint(1, 28)
    hour = rng.choice([9, 14, 17])
    return Scenario(
        skill="multiturn", hint_key="multiturn",
        initial_message=rng.choice(["Set a reminder for me", "I need a reminder", "Add a reminder please"]),
        user_responses=[f"It's about {content.lower()}", f"On April {day}, 2026 at {hour}:00",
                        f"At {loc_name}"],
        expected_tools=["datetime_info_to_timestamp", "search_lat_lon", "add_reminder"],
        verify_keywords=["reminder", content.split()[0].lower()],
        description=f"MT: reminder '{content}' at {loc_name}")


# =====================================================================
# DATETIME CANONICALIZATION scenarios (35%)
# Require multi-step timestamp operations
# =====================================================================

def _dt_reminder_next_weekday(rng, db):
    """'Remind me next Tuesday at 3pm' → get_current_timestamp → timestamp_to_datetime →
    compute days to target → datetime_info_to_timestamp → add_reminder."""
    content = rng.choice(REMINDER_CONTENTS)
    target_weekday = rng.choice(WEEKDAY_NAMES)
    hour = rng.choice([9, 10, 14, 15, 17])
    return Scenario(
        skill="datetime", hint_key="datetime",
        initial_message=f"Remind me to {content.lower()} next {target_weekday} at {hour}:00",
        user_responses=[],
        expected_tools=["get_current_timestamp", "timestamp_to_datetime_info",
                        "datetime_info_to_timestamp", "add_reminder"],
        verify_keywords=["reminder", content.split()[0].lower()],
        description=f"DT: reminder next {target_weekday}")

def _dt_reminder_week_delta(rng, db):
    """'Set a reminder for 2 weeks from now' → get_current_timestamp → shift_timestamp → add_reminder."""
    content = rng.choice(REMINDER_CONTENTS)
    weeks = rng.choice([1, 2, 3])
    return Scenario(
        skill="datetime", hint_key="datetime",
        initial_message=f"Remind me to {content.lower()} in {weeks} week{'s' if weeks > 1 else ''} from now",
        user_responses=[],
        expected_tools=["get_current_timestamp", "shift_timestamp", "add_reminder"],
        verify_keywords=["reminder", content.split()[0].lower()],
        description=f"DT: reminder in {weeks} weeks")

def _dt_days_until_holiday(rng, db):
    """'How many days until Christmas?' → get_current_timestamp → search_holiday → timestamp_diff."""
    holiday = rng.choice(list(db["holidays"].keys()))
    return Scenario(
        skill="datetime", hint_key="datetime",
        initial_message=f"How many days until {holiday}?",
        user_responses=[],
        expected_tools=["get_current_timestamp", "search_holiday", "timestamp_diff"],
        verify_keywords=["days", holiday.split()[0].lower()],
        description=f"DT: days until {holiday}")

def _dt_days_until_holiday_multiturn(rng, db):
    """Vague 'How long until a holiday?' → ask which → then compute."""
    holiday = rng.choice(list(db["holidays"].keys()))
    templates = ["How long until a holiday?", "When is the next holiday?",
                 "I want to know how far away a holiday is"]
    return Scenario(
        skill="datetime", hint_key="datetime",
        initial_message=rng.choice(templates),
        user_responses=[holiday],
        expected_tools=["get_current_timestamp", "search_holiday", "timestamp_diff"],
        verify_keywords=["days", holiday.split()[0].lower()],
        description=f"DT+MT: days until {holiday}")

def _dt_reminder_tomorrow(rng, db):
    """'Remind me tomorrow at 5pm' → get_current_timestamp → shift_timestamp(days=1) →
    datetime_info_to_timestamp → add_reminder."""
    content = rng.choice(REMINDER_CONTENTS)
    hour = rng.choice([9, 12, 15, 17, 20])
    return Scenario(
        skill="datetime", hint_key="datetime",
        initial_message=f"Remind me to {content.lower()} tomorrow at {hour}:00",
        user_responses=[],
        expected_tools=["get_current_timestamp", "add_reminder"],
        verify_keywords=["reminder", content.split()[0].lower()],
        description=f"DT: reminder tomorrow")

def _dt_find_weekday(rng, db):
    """'What day of the week is April 15?' → datetime_info_to_timestamp → timestamp_to_datetime_info."""
    month = rng.choice([4, 5, 6, 7])
    day = rng.randint(1, 28)
    return Scenario(
        skill="datetime", hint_key="datetime",
        initial_message=f"What day of the week is {['January','February','March','April','May','June','July'][month-1]} {day}, 2026?",
        user_responses=[],
        expected_tools=["datetime_info_to_timestamp", "timestamp_to_datetime_info"],
        verify_keywords=["day"],
        description=f"DT: find weekday for month {month} day {day}")

def _dt_temperature_at_time(rng, db):
    """'What's the temperature at Costco at 3pm?' → get_current_timestamp → search_lat_lon →
    search_weather → communicate in F."""
    loc_name, _, _ = rng.choice(LOCATIONS)
    return Scenario(
        skill="datetime", hint_key="datetime",
        initial_message=f"What's the temperature in Fahrenheit at {loc_name}?",
        user_responses=[],
        expected_tools=["search_lat_lon", "search_weather_around_lat_lon"],
        verify_keywords=["temperature", "fahrenheit"],
        description=f"DT: temperature at {loc_name}")


# =====================================================================
# INSUFFICIENT INFORMATION scenarios (25%)
# Model must recognize missing tools and decline
# =====================================================================

def _insuf_no_timestamp(rng, db):
    """Need current time but get_current_timestamp removed."""
    templates = ["How many days until Christmas?", "What day of the week is it today?",
                 "How long until the weekend?", "How many hours until midnight?"]
    return Scenario(
        skill="insufficient", hint_key="insufficient",
        initial_message=rng.choice(templates), user_responses=[], expected_tools=[],
        verify_keywords=[],
        tool_allow_list=["search_holiday", "timestamp_diff", "shift_timestamp",
                         "search_contacts", "add_reminder", "datetime_info_to_timestamp"],
        description="INSUF: no get_current_timestamp")

def _insuf_no_weather(rng, db):
    """Need weather but weather tool removed."""
    templates = ["What's the temperature outside right now?", "Is it going to rain today?",
                 "What's the weather like?", "Do I need a jacket today?"]
    return Scenario(
        skill="insufficient", hint_key="insufficient",
        initial_message=rng.choice(templates), user_responses=[], expected_tools=[],
        verify_keywords=[],
        tool_allow_list=["get_current_timestamp", "search_contacts", "add_reminder",
                         "search_lat_lon", "unit_conversion"],
        description="INSUF: no weather tool")

def _insuf_no_search_contacts(rng, db):
    """Need to search contacts but search_contacts removed."""
    templates = ["What is the phone number of my boss?", "Find my friend's contact info",
                 "Who is in my contact list?", "Look up my colleague's number"]
    return Scenario(
        skill="insufficient", hint_key="insufficient",
        initial_message=rng.choice(templates), user_responses=[], expected_tools=[],
        verify_keywords=[],
        tool_allow_list=["add_contact", "modify_contact", "remove_contact",
                         "get_current_timestamp", "add_reminder"],
        description="INSUF: no search_contacts")

def _insuf_no_location(rng, db):
    """Need location but location tools removed."""
    templates = ["What city am I in right now?", "What are my GPS coordinates?",
                 "Where am I located?"]
    return Scenario(
        skill="insufficient", hint_key="insufficient",
        initial_message=rng.choice(templates), user_responses=[], expected_tools=[],
        verify_keywords=[],
        tool_allow_list=["search_contacts", "add_reminder", "get_current_timestamp",
                         "search_weather_around_lat_lon"],
        description="INSUF: no location tools")

def _insuf_no_messages(rng, db):
    """Need to search messages but search_messages removed."""
    templates = ["What was the last message I received?", "Find my latest text",
                 "Who texted me most recently?"]
    return Scenario(
        skill="insufficient", hint_key="insufficient",
        initial_message=rng.choice(templates), user_responses=[], expected_tools=[],
        verify_keywords=[],
        tool_allow_list=["search_contacts", "send_message_with_phone_number",
                         "get_current_timestamp", "add_reminder"],
        description="INSUF: no search_messages")


# =====================================================================
# Scenario generation
# =====================================================================

_MULTITURN_GENERATORS = [
    _mt_send_message, _mt_add_reminder, _mt_remove_contact_by_phone,
    _mt_modify_contact, _mt_search_message_recency, _mt_update_relationship,
    _mt_add_reminder_with_location,
]
_DATETIME_GENERATORS = [
    _dt_reminder_next_weekday, _dt_reminder_week_delta, _dt_days_until_holiday,
    _dt_days_until_holiday_multiturn, _dt_reminder_tomorrow, _dt_find_weekday,
    _dt_temperature_at_time,
]
_INSUFFICIENT_GENERATORS = [
    _insuf_no_timestamp, _insuf_no_weather, _insuf_no_search_contacts,
    _insuf_no_location, _insuf_no_messages,
]


def generate_scenario(seed: int) -> Tuple[Scenario, dict]:
    rng = random.Random(seed)
    db = _make_db(rng)

    # Distribution: 40% multiturn, 35% datetime, 25% insufficient
    roll = rng.random()
    if roll < 0.40:
        scenario = rng.choice(_MULTITURN_GENERATORS)(rng, db)
    elif roll < 0.75:
        scenario = rng.choice(_DATETIME_GENERATORS)(rng, db)
    else:
        scenario = rng.choice(_INSUFFICIENT_GENERATORS)(rng, db)

    return scenario, db


# =====================================================================
# Hint stripping utility — call before training loss computation
# =====================================================================

def strip_hints_from_samples(samples):
    """Strip hint text from system prompts in all samples. Modifies in-place.

    Call this AFTER rollout collection, BEFORE computing training loss.
    This ensures the model trains on the base prompt (no hint conditioning).
    """
    for sample in samples:
        if not hasattr(sample, 'prompt_msgs') or not sample.prompt_msgs:
            continue
        for msg in sample.prompt_msgs:
            if msg.get("role") == "system" and HINT_START.strip() in msg.get("content", ""):
                # Strip everything between HINT_START and HINT_END
                content = msg["content"]
                start_idx = content.find(HINT_START.strip())
                end_marker = HINT_END.strip()
                end_idx = content.find(end_marker)
                if start_idx >= 0 and end_idx >= 0:
                    msg["content"] = content[:start_idx].rstrip()
                elif start_idx >= 0:
                    msg["content"] = content[:start_idx].rstrip()


# =====================================================================
# Game
# =====================================================================

class ToolSandboxMultiTurnGame:
    supports_structured_messages = True

    # Class-level counter for hint randomization across rollouts
    _rollout_counter = 0

    def __init__(self, hint_ratio=0.5):
        """
        Args:
            hint_ratio: Fraction of rollouts that receive hints (0.0-1.0).
                        Within a GRPO group of N rollouts, ~N*hint_ratio get hints.
        """
        self.hint_ratio = hint_ratio
        self.done = False
        self.current_player = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player = None
        self._scenario: Optional[Scenario] = None
        self._db = None
        self._tools: Optional[ToolExecutor] = None
        self._conversation: List[Dict] = []
        self._step_count = 0
        self._tool_called = False
        self._all_tools: List[str] = []
        self._user_response_idx = 0
        self._text_responses: List[str] = []
        self._got_clarification = False
        self._tool_schemas_active: List[Dict] = []
        self._use_hint = False
        self.max_steps = 16

    def reset(self, seed: int, use_hint: Optional[bool] = None) -> None:
        self._scenario, self._db = generate_scenario(seed)
        self._tools = ToolExecutor(copy.deepcopy(self._db))
        self._conversation = [{"role": "user", "content": self._scenario.initial_message}]
        self._step_count = 0
        self._tool_called = False
        self._all_tools = []
        self._user_response_idx = 0
        self._text_responses = []
        self._got_clarification = False
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

        # Hint injection: either explicit or random per rollout
        if use_hint is not None:
            self._use_hint = use_hint
        else:
            ToolSandboxMultiTurnGame._rollout_counter += 1
            hint_rng = random.Random(seed * 7919 + ToolSandboxMultiTurnGame._rollout_counter)
            self._use_hint = hint_rng.random() < self.hint_ratio

        # Set active tool schemas
        if self._scenario.tool_allow_list is not None:
            allowed = set(self._scenario.tool_allow_list)
            self._tool_schemas_active = [t for t in TOOL_SCHEMAS if t["function"]["name"] in allowed]
        else:
            self._tool_schemas_active = list(TOOL_SCHEMAS)

    def get_system_prompt(self) -> str:
        base = SYSTEM_PROMPT
        if self._use_hint and self._scenario:
            hint_text = HINTS.get(self._scenario.hint_key, "")
            if hint_text:
                return base + HINT_START + hint_text + HINT_END
        return base

    def get_tool_schemas(self) -> List[Dict]:
        return self._tool_schemas_active

    def get_messages(self) -> List[Dict]:
        return list(self._conversation)

    def observe(self, player_id: int) -> str:
        if self.done: return ""
        return self._conversation[-1].get("content", "") if self._conversation else ""

    def legal_actions(self) -> List[str]:
        return []

    def step(self, action: Optional[str]) -> None:
        if self.done:
            return
        if action is None:
            if self._step_count == 0:
                self.done = True
                self.rewards = {0: 0.0}
            return

        self._step_count += 1
        tc = self._parse_tool_call(action)

        if tc:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            result = self._tools.execute(name, args)
            result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
            self._tool_called = True
            self._all_tools.append(name)
            self._conversation.append({"role": "assistant", "content": action,
                                        "tool_call": {"name": name, "arguments": args}})
            self._conversation.append({"role": "tool", "content": result_str})
        else:
            clean_text = action.strip()
            self._text_responses.append(clean_text)
            self._conversation.append({"role": "assistant", "content": clean_text})

            skill = self._scenario.skill
            has_remaining_responses = self._user_response_idx < len(self._scenario.user_responses)

            if skill in ("multiturn", "datetime") and has_remaining_responses:
                # Still have scripted user responses to provide — this is a
                # clarifying question (whether before or between tool calls)
                self._got_clarification = True
                user_msg = self._scenario.user_responses[self._user_response_idx]
                self._user_response_idx += 1
                self._conversation.append({"role": "user", "content": user_msg})
            elif skill in ("multiturn", "datetime") and not self._tool_called:
                # No more responses and no tool called yet — agent is stuck, continue
                self._got_clarification = True
            elif skill in ("multiturn", "datetime") and self._tool_called:
                # No more responses + tool was called = final communication
                self._evaluate(clean_text)
            elif skill == "insufficient":
                self._evaluate_insufficient(clean_text)

        if self._step_count >= self.max_steps and not self.done:
            self.done = True
            self._compute_timeout_reward()

    def _evaluate(self, final_response: str):
        all_text = " ".join(t.lower() for t in self._text_responses)

        # Tool chain completeness
        if self._scenario.expected_tools:
            expected = set(self._scenario.expected_tools)
            called = set(self._all_tools)
            tool_score = len(expected & called) / len(expected)
        else:
            tool_score = 1.0 if self._tool_called else 0.0

        # Keyword communication
        if self._scenario.verify_keywords:
            matched = sum(1 for kw in self._scenario.verify_keywords if kw.lower() in all_text)
            comm_score = matched / len(self._scenario.verify_keywords)
        else:
            comm_score = 1.0 if len(final_response.strip()) > 10 else 0.0

        if self._scenario.skill == "multiturn":
            clarify = 1.0 if self._got_clarification else 0.0
            # Multiplicative: all three must be present + additive floor for gradient
            mult = clarify * tool_score * comm_score
            add = 0.15 * clarify + 0.15 * tool_score + 0.1 * comm_score
            self.rewards = {0: 0.6 * mult + 0.4 * add}
        elif self._scenario.skill == "datetime":
            # Datetime: tool chain is primary (70%), communication secondary (30%)
            mult = tool_score * comm_score
            add = 0.2 * tool_score + 0.1 * comm_score
            self.rewards = {0: 0.7 * mult + 0.3 * add}
        self.done = True

    def _evaluate_insufficient(self, response: str):
        resp = response.lower()
        decline = ["i don't have", "i cannot", "i can't", "i'm unable", "not available",
                   "don't have access", "unable to", "no way to", "not possible",
                   "i'm sorry", "unfortunately", "missing", "no tool", "don't currently have"]
        if self._tool_called:
            self.rewards = {0: 0.0}
        elif any(p in resp for p in decline):
            self.rewards = {0: 1.0}
        else:
            self.rewards = {0: 0.2}
        self.done = True

    def _compute_timeout_reward(self):
        if self._scenario.skill == "insufficient":
            self.rewards = {0: 0.0}; return
        r = 0.0
        if self._got_clarification: r += 0.1
        if self._tool_called: r += 0.1
        self.rewards = {0: r}

    def _parse_tool_call(self, action):
        if not action: return None
        try:
            p = json.loads(action)
            if isinstance(p, dict):
                if "name" in p and "arguments" in p: return p
                if "function" in p:
                    fn = p["function"]
                    return {"name": fn.get("name",""), "arguments": fn.get("arguments",{})}
        except (json.JSONDecodeError, ValueError): pass
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
    for seed in range(100):
        game = ToolSandboxMultiTurnGame()
        game.reset(seed)
        s = game._scenario
        skill_counts[s.skill] += 1
        hint = "H" if game._use_hint else " "
        print(f"Seed {seed:>3d} [{s.skill:>12s}] {hint} | {s.description:<45s} | {s.initial_message[:50]}")

    print(f"\nDistribution: {dict(skill_counts)}")
    print(f"Target: ~40 multiturn, ~35 datetime, ~25 insufficient")

    # Test hint stripping
    print("\n--- Hint stripping test ---")
    game = ToolSandboxMultiTurnGame()
    game.reset(42, use_hint=True)
    prompt_with = game.get_system_prompt()
    print(f"With hint ({len(prompt_with)} chars): ...{prompt_with[-80:]}")

    game.reset(42, use_hint=False)
    prompt_without = game.get_system_prompt()
    print(f"Without ({len(prompt_without)} chars): {prompt_without}")

    # Verify stripping works
    class FakeSample:
        def __init__(self, msgs): self.prompt_msgs = msgs
    sample = FakeSample([{"role": "system", "content": prompt_with}])
    strip_hints_from_samples([sample])
    print(f"Stripped ({len(sample.prompt_msgs[0]['content'])} chars): {sample.prompt_msgs[0]['content']}")
    assert sample.prompt_msgs[0]["content"] == SYSTEM_PROMPT, "Hint stripping failed!"
    print("Hint stripping: OK")
