"""Constants for the Adversarial Policy Adherence Game.

Loads policies and tool schemas directly from tau2-bench source.
No fallbacks — tau2-bench must be available.
"""

import sys
import pathlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_GAME_DIR = pathlib.Path(__file__).resolve().parents[1]
_TAU2_DATA = _GAME_DIR / "tau2-bench" / "data" / "tau2" / "domains"
_TAU2_SRC = _GAME_DIR / "tau2-bench" / "src"

# ---------------------------------------------------------------------------
# Load policies directly from tau2-bench (no fallback)
# ---------------------------------------------------------------------------
AIRLINE_POLICY: str = (_TAU2_DATA / "airline" / "policy.md").read_text()
RETAIL_POLICY: str = (_TAU2_DATA / "retail" / "policy.md").read_text()

# ---------------------------------------------------------------------------
# Load tool schemas from tau2-bench source (exact match)
# ---------------------------------------------------------------------------
# Temporarily add tau2-bench src to path for import
_added_to_path = False
if str(_TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(_TAU2_SRC))
    _added_to_path = True

from tau2.domains.airline.environment import get_environment as _get_airline_env
from tau2.domains.retail.environment import get_environment as _get_retail_env

_airline_env = _get_airline_env()
_retail_env = _get_retail_env()

AIRLINE_TOOL_SCHEMAS = [t.openai_schema for t in _airline_env.get_tools()]
RETAIL_TOOL_SCHEMAS = [t.openai_schema for t in _retail_env.get_tools()]

# Clean up
del _airline_env, _retail_env

# Text tool defs (for observe() text-based protocol)
AIRLINE_TOOL_DEFS = "\n".join(
    f"- {t['function']['name']}: {t['function']['description']}"
    for t in AIRLINE_TOOL_SCHEMAS
)
RETAIL_TOOL_DEFS = "\n".join(
    f"- {t['function']['name']}: {t['function']['description']}"
    for t in RETAIL_TOOL_SCHEMAS
)

# ---------------------------------------------------------------------------
# Airport/city lookup (used by Template 3: destination_change)
# ---------------------------------------------------------------------------
AIRPORTS = [
    "SFO", "JFK", "LAX", "ORD", "DFW", "DEN", "SEA", "ATL", "MIA", "BOS",
    "PHX", "IAH", "LAS", "MCO", "EWR", "CLT", "MSP", "DTW", "PHL", "LGA",
]

CITIES = {
    "SFO": "San Francisco", "JFK": "New York", "LAX": "Los Angeles",
    "ORD": "Chicago", "DFW": "Dallas", "DEN": "Denver", "SEA": "Seattle",
    "ATL": "Atlanta", "MIA": "Miami", "BOS": "Boston", "PHX": "Phoenix",
    "IAH": "Houston", "LAS": "Las Vegas", "MCO": "Orlando", "EWR": "Newark",
    "CLT": "Charlotte", "MSP": "Minneapolis", "DTW": "Detroit",
    "PHL": "Philadelphia", "LGA": "New York",
}

