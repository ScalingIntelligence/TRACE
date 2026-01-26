#!/usr/bin/env python3
"""
Tool definitions and tool-call parsing for Liar's Dice games.
"""
from __future__ import annotations

import inspect
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ToolFn = Callable[..., Optional[str]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    params: List[Tuple[str, str]]
    description: str


_SESSION_ID_RE = re.compile(r"#[A-Z0-9]{4}\Z")
_BID_ACTION_RE = re.compile(r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)(?:[,\s]+id:\s*([#A-Z0-9]+))?\]", re.IGNORECASE)
_CALL_ACTION_RE = re.compile(r"\[call(?:[,\s]+id:\s*([#A-Z0-9]+))?\]", re.IGNORECASE)


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_session_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if not _SESSION_ID_RE.fullmatch(s):
        return None
    return s


def _coerce_non_empty_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    return s


# =========================
# Game action tools
# =========================
def submit_bid(quantity: Any, face: Any, session_id: Any) -> Optional[str]:
    """Create a bid action with a session id."""
    qty = _coerce_int(quantity)
    face_val = _coerce_int(face)
    sid = _coerce_session_id(session_id)
    if qty is None or face_val is None or sid is None:
        return None
    if qty < 1 or face_val < 1 or face_val > 6:
        return None
    return f"[bid: {qty}, {face_val}, id: {sid}]"


def call_bluff(session_id: Any) -> Optional[str]:
    """Create a call action with a session id."""
    sid = _coerce_session_id(session_id)
    if sid is None:
        return None
    return f"[call, id: {sid}]"


def submit_bid_no_id(quantity: Any, face: Any) -> Optional[str]:
    """Create a bid action without a session id."""
    qty = _coerce_int(quantity)
    face_val = _coerce_int(face)
    if qty is None or face_val is None:
        return None
    if qty < 1 or face_val < 1 or face_val > 6:
        return None
    return f"[bid: {qty}, {face_val}]"


def call_bluff_no_id() -> Optional[str]:
    """Create a call action without a session id."""
    return "[call]"


# =========================
# Distractor tools (non-game)
# =========================
def get_weather(city: Any, day: Any) -> Optional[str]:
    city_val = _coerce_non_empty_str(city)
    day_val = _coerce_non_empty_str(day)
    if city_val is None or day_val is None:
        return None
    return f"WEATHER({city_val},{day_val})"


def convert_units(value: Any, from_unit: Any, to_unit: Any) -> Optional[str]:
    value_val = _coerce_float(value)
    from_val = _coerce_non_empty_str(from_unit)
    to_val = _coerce_non_empty_str(to_unit)
    if value_val is None or from_val is None or to_val is None:
        return None
    return f"CONVERT({value_val},{from_val},{to_val})"


def set_timer(label: Any, seconds: Any) -> Optional[str]:
    label_val = _coerce_non_empty_str(label)
    seconds_val = _coerce_int(seconds)
    if label_val is None or seconds_val is None or seconds_val < 0:
        return None
    return f"TIMER({label_val},{seconds_val})"


def lookup_contact(name: Any) -> Optional[str]:
    name_val = _coerce_non_empty_str(name)
    if name_val is None:
        return None
    return f"CONTACT({name_val})"


def get_news(topic: Any) -> Optional[str]:
    topic_val = _coerce_non_empty_str(topic)
    if topic_val is None:
        return None
    return f"NEWS({topic_val})"


def search_web(query: Any, top_k: Any) -> Optional[str]:
    query_val = _coerce_non_empty_str(query)
    top_k_val = _coerce_int(top_k)
    if query_val is None or top_k_val is None or top_k_val < 1:
        return None
    return f"SEARCH({query_val},{top_k_val})"


def send_email(to: Any, subject: Any, body: Any) -> Optional[str]:
    to_val = _coerce_non_empty_str(to)
    subject_val = _coerce_non_empty_str(subject)
    body_val = _coerce_non_empty_str(body)
    if to_val is None or subject_val is None or body_val is None:
        return None
    return f"EMAIL({to_val},{subject_val})"


def translate_text(text: Any, target_lang: Any) -> Optional[str]:
    text_val = _coerce_non_empty_str(text)
    target_val = _coerce_non_empty_str(target_lang)
    if text_val is None or target_val is None:
        return None
    return f"TRANSLATE({target_val})"


def calculate_tip(amount: Any, percent: Any) -> Optional[str]:
    amount_val = _coerce_float(amount)
    percent_val = _coerce_float(percent)
    if amount_val is None or percent_val is None or percent_val < 0:
        return None
    tip = amount_val * (percent_val / 100.0)
    return f"TIP({tip})"


def book_flight(origin: Any, destination: Any, date: Any) -> Optional[str]:
    origin_val = _coerce_non_empty_str(origin)
    dest_val = _coerce_non_empty_str(destination)
    date_val = _coerce_non_empty_str(date)
    if origin_val is None or dest_val is None or date_val is None:
        return None
    return f"FLIGHT({origin_val},{dest_val},{date_val})"


def play_music(artist: Any, title: Any) -> Optional[str]:
    artist_val = _coerce_non_empty_str(artist)
    title_val = _coerce_non_empty_str(title)
    if artist_val is None or title_val is None:
        return None
    return f"MUSIC({artist_val},{title_val})"


def set_alarm(time: Any, label: Any) -> Optional[str]:
    time_val = _coerce_non_empty_str(time)
    label_val = _coerce_non_empty_str(label)
    if time_val is None or label_val is None:
        return None
    return f"ALARM({time_val},{label_val})"


def add_calendar_event(title: Any, date: Any, time: Any, location: Any) -> Optional[str]:
    title_val = _coerce_non_empty_str(title)
    date_val = _coerce_non_empty_str(date)
    time_val = _coerce_non_empty_str(time)
    location_val = _coerce_non_empty_str(location)
    if title_val is None or date_val is None or time_val is None or location_val is None:
        return None
    return f"EVENT({title_val},{date_val},{time_val},{location_val})"


def get_stock_price(symbol: Any) -> Optional[str]:
    symbol_val = _coerce_non_empty_str(symbol)
    if symbol_val is None:
        return None
    return f"STOCK({symbol_val})"


def get_exchange_rate(base: Any, quote: Any) -> Optional[str]:
    base_val = _coerce_non_empty_str(base)
    quote_val = _coerce_non_empty_str(quote)
    if base_val is None or quote_val is None:
        return None
    return f"FX({base_val},{quote_val})"


def summarize_text(text: Any, max_words: Any) -> Optional[str]:
    text_val = _coerce_non_empty_str(text)
    max_words_val = _coerce_int(max_words)
    if text_val is None or max_words_val is None or max_words_val < 1:
        return None
    return f"SUMMARY({max_words_val})"


def generate_uuid() -> Optional[str]:
    return f"UUID({uuid.uuid4()})"


# =========================
# Tool specs and registries
# =========================
LIARS_DICE_ACTION_TOOL_SPECS_WITH_ID = [
    ToolSpec(
        name="submit_bid",
        params=[("quantity", "int >= 1"), ("face", "int 1-6"), ("session_id", "str from observation")],
        description="Place a bid of quantity dice showing face.",
    ),
    ToolSpec(
        name="call_bluff",
        params=[("session_id", "str from observation")],
        description="Challenge the last bid.",
    ),
]

LIARS_DICE_ACTION_TOOL_SPECS_NO_ID = [
    ToolSpec(
        name="submit_bid",
        params=[("quantity", "int >= 1"), ("face", "int 1-6")],
        description="Place a bid of quantity dice showing face.",
    ),
    ToolSpec(
        name="call_bluff",
        params=[],
        description="Challenge the last bid.",
    ),
]

DISTRACTOR_TOOL_SPECS = [
    ToolSpec(
        name="get_weather",
        params=[("city", "str"), ("day", "str")],
        description="Get a weather summary for a city and day.",
    ),
    ToolSpec(
        name="convert_units",
        params=[("value", "float"), ("from_unit", "str"), ("to_unit", "str")],
        description="Convert a value between units.",
    ),
    ToolSpec(
        name="set_timer",
        params=[("label", "str"), ("seconds", "int")],
        description="Set a timer for a duration.",
    ),
    ToolSpec(
        name="lookup_contact",
        params=[("name", "str")],
        description="Lookup a contact by name.",
    ),
    ToolSpec(
        name="get_news",
        params=[("topic", "str")],
        description="Get the latest headlines for a topic.",
    ),
    ToolSpec(
        name="search_web",
        params=[("query", "str"), ("top_k", "int")],
        description="Search the web and return top results.",
    ),
    ToolSpec(
        name="send_email",
        params=[("to", "str"), ("subject", "str"), ("body", "str")],
        description="Send an email message.",
    ),
    ToolSpec(
        name="translate_text",
        params=[("text", "str"), ("target_lang", "str")],
        description="Translate text into the target language.",
    ),
    ToolSpec(
        name="calculate_tip",
        params=[("amount", "float"), ("percent", "float")],
        description="Calculate a tip amount from a bill.",
    ),
    ToolSpec(
        name="book_flight",
        params=[("origin", "str"), ("destination", "str"), ("date", "str")],
        description="Book a one-way flight.",
    ),
    ToolSpec(
        name="play_music",
        params=[("artist", "str"), ("title", "str")],
        description="Play a song by artist and title.",
    ),
    ToolSpec(
        name="set_alarm",
        params=[("time", "str"), ("label", "str")],
        description="Set an alarm for a time.",
    ),
    ToolSpec(
        name="add_calendar_event",
        params=[("title", "str"), ("date", "str"), ("time", "str"), ("location", "str")],
        description="Add a calendar event.",
    ),
    ToolSpec(
        name="get_stock_price",
        params=[("symbol", "str")],
        description="Get the latest price for a stock symbol.",
    ),
    ToolSpec(
        name="get_exchange_rate",
        params=[("base", "str"), ("quote", "str")],
        description="Get an exchange rate between currencies.",
    ),
    ToolSpec(
        name="summarize_text",
        params=[("text", "str"), ("max_words", "int")],
        description="Summarize text to a word limit.",
    ),
    ToolSpec(
        name="generate_uuid",
        params=[],
        description="Generate a UUID.",
    ),
]

LIARS_DICE_TOOL_SPECS_WITH_ID = LIARS_DICE_ACTION_TOOL_SPECS_WITH_ID + DISTRACTOR_TOOL_SPECS
LIARS_DICE_TOOL_SPECS_NO_ID = LIARS_DICE_ACTION_TOOL_SPECS_NO_ID + DISTRACTOR_TOOL_SPECS

LIARS_DICE_TOOL_REGISTRY_WITH_ID: Dict[str, ToolFn] = {
    "submit_bid": submit_bid,
    "call_bluff": call_bluff,
    "get_weather": get_weather,
    "convert_units": convert_units,
    "set_timer": set_timer,
    "lookup_contact": lookup_contact,
    "get_news": get_news,
    "search_web": search_web,
    "send_email": send_email,
    "translate_text": translate_text,
    "calculate_tip": calculate_tip,
    "book_flight": book_flight,
    "play_music": play_music,
    "set_alarm": set_alarm,
    "add_calendar_event": add_calendar_event,
    "get_stock_price": get_stock_price,
    "get_exchange_rate": get_exchange_rate,
    "summarize_text": summarize_text,
    "generate_uuid": generate_uuid,
}

LIARS_DICE_TOOL_REGISTRY_NO_ID: Dict[str, ToolFn] = {
    "submit_bid": submit_bid_no_id,
    "call_bluff": call_bluff_no_id,
    "get_weather": get_weather,
    "convert_units": convert_units,
    "set_timer": set_timer,
    "lookup_contact": lookup_contact,
    "get_news": get_news,
    "search_web": search_web,
    "send_email": send_email,
    "translate_text": translate_text,
    "calculate_tip": calculate_tip,
    "book_flight": book_flight,
    "play_music": play_music,
    "set_alarm": set_alarm,
    "add_calendar_event": add_calendar_event,
    "get_stock_price": get_stock_price,
    "get_exchange_rate": get_exchange_rate,
    "summarize_text": summarize_text,
    "generate_uuid": generate_uuid,
}


# =========================
# Tool-call parsing + dispatch
# =========================
_TOOL_CALL_FORMAT = (
    "Choose exactly one tool from the list below.\n"
    "Think step by step carefully and include your reasoning before the JSON tool call.\n"
    "Use the exact parameter names listed for that tool.\n"
    "Example: {\"tool\": \"submit_bid\", \"quantity\": 2, \"face\": 5, \"session_id\": \"#AB12\"}\n"
)


def render_tool_list(tool_specs: List[ToolSpec]) -> str:
    lines = ["Available tools:"]
    for spec in tool_specs:
        if spec.params:
            params = ", ".join([f"{name}: {desc}" for name, desc in spec.params])
        else:
            params = "no params"
        lines.append(f"- {spec.name}({params}) - {spec.description}")
    return "\n".join(lines)


def build_tool_system_prompt(game_instructions: str, tool_specs: List[ToolSpec]) -> str:
    parts = [
        game_instructions.strip(),
        _TOOL_CALL_FORMAT.strip(),
        render_tool_list(tool_specs),
    ]
    return "\n".join([part for part in parts if part])


def _iter_json_objects(text: str) -> Iterable[str]:
    depth = 0
    start = None
    in_str = False
    escape = False
    for idx, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : idx + 1]
                    start = None


def extract_tool_call_with_text(text: str) -> Optional[Tuple[Dict[str, Any], str]]:
    if not text:
        return None
    for chunk in _iter_json_objects(text):
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "tool" in data:
            return data, chunk
    return None


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    result = extract_tool_call_with_text(text)
    if result is None:
        return None
    return result[0]


def _filter_tool_params(tool_fn: ToolFn, params: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(tool_fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in params.items() if k in allowed}


def tool_call_to_action(
    tool_call: Dict[str, Any],
    tool_registry: Dict[str, ToolFn],
) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    tool_name = tool_call.get("tool")
    if not isinstance(tool_name, str):
        return None
    tool_fn = tool_registry.get(tool_name)
    if tool_fn is None:
        return None
    params = {k: v for k, v in tool_call.items() if k != "tool"}
    params = _filter_tool_params(tool_fn, params)
    try:
        result = tool_fn(**params)
    except Exception:
        return None
    if not isinstance(result, str):
        return None
    return result


def extract_action_from_tool_call(
    text: str,
    legal_actions: List[str],
    tool_registry: Dict[str, ToolFn],
) -> Optional[str]:
    tool_call = extract_tool_call(text)
    if tool_call is None:
        return None
    action = tool_call_to_action(tool_call, tool_registry)
    if action in legal_actions:
        return action
    return None


def action_string_to_tool_call(action: str) -> Optional[Dict[str, Any]]:
    if not action:
        return None
    bid_match = _BID_ACTION_RE.fullmatch(action.strip())
    if bid_match:
        qty_str, face_str, session_id = bid_match.groups()
        tool_call: Dict[str, Any] = {
            "tool": "submit_bid",
            "quantity": int(qty_str),
            "face": int(face_str),
        }
        if session_id:
            tool_call["session_id"] = session_id
        return tool_call
    call_match = _CALL_ACTION_RE.fullmatch(action.strip())
    if call_match:
        tool_call = {"tool": "call_bluff"}
        session_id = call_match.group(1)
        if session_id:
            tool_call["session_id"] = session_id
        return tool_call
    return None


def tool_call_to_json(tool_call: Dict[str, Any]) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    tool_name = tool_call.get("tool")
    if not isinstance(tool_name, str):
        return None
    ordered: Dict[str, Any] = {"tool": tool_name}
    for key, value in tool_call.items():
        if key != "tool":
            ordered[key] = value
    return json.dumps(ordered, ensure_ascii=True)
