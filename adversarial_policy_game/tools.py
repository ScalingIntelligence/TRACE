"""Tool executor using tau2-bench Environment directly.

Creates the exact same Environment used during tau2-bench evaluation via
get_environment(), just initialized with synthetic Pydantic-validated
databases instead of the real db.json.

All tools, validation, side effects, return formats, and serialization
are identical to tau2-bench eval — zero reimplementation.
"""

import sys
import pathlib
import json
import copy
from typing import Dict, Any, List

# Ensure tau2-bench source is on path
_TAU2_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "tau2-bench" / "src")
if _TAU2_SRC not in sys.path:
    sys.path.insert(0, _TAU2_SRC)

# Suppress tau2-bench debug logging
import logging
logging.getLogger("tau2").setLevel(logging.WARNING)
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.disable("tau2")
except ImportError:
    pass

from tau2.environment.environment import Environment
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.retail.environment import get_environment as _get_retail_env
from tau2.domains.airline.environment import get_environment as _get_airline_env


class ToolExecutor:
    """Executes tools using tau2-bench Environment with synthetic data.

    Uses get_environment() from tau2-bench to create the exact same
    Environment object as evaluation. Only the data differs.
    """

    def __init__(self, domain: str, db: Dict[str, Any]):
        self.domain = domain
        self._tool_call_log: List[Dict[str, Any]] = []

        # Convert dict DB to Pydantic model, create tau2-bench Environment
        if domain == "retail":
            pydantic_db = RetailDB.model_validate(db)
            self._env = _get_retail_env(db=pydantic_db)
        elif domain == "airline":
            pydantic_db = FlightDB.model_validate(db)
            self._env = _get_airline_env(db=pydantic_db)
        else:
            raise ValueError(f"Unknown domain: {domain}")

    @property
    def db(self) -> Dict[str, Any]:
        """Return current DB state as a dict (for hash comparison)."""
        return self._env.tools.db.model_dump()

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """Return log of all tool calls made."""
        return list(self._tool_call_log)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return result string."""
        # Normalize retail order_id prefix
        if self.domain == "retail" and "order_id" in arguments:
            oid = str(arguments["order_id"])
            if oid and not oid.startswith("#"):
                arguments = {**arguments, "order_id": f"#{oid}"}

        # Log the call
        entry = {
            "name": tool_name,
            "arguments": copy.deepcopy(arguments),
        }
        self._tool_call_log.append(entry)

        try:
            # Parse JSON string arguments (model sometimes sends strings)
            parsed_args = {}
            for k, v in arguments.items():
                if isinstance(v, str):
                    try:
                        parsed = json.loads(v)
                        if isinstance(parsed, (list, dict)):
                            v = parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                parsed_args[k] = v

            # Execute via tau2-bench Environment (same as eval)
            result = self._env.use_tool(tool_name, **parsed_args)
            entry["error"] = False

            # Serialize using tau2-bench's Environment.to_json_str (same as eval)
            return Environment.to_json_str(result)

        except Exception as e:
            entry["error"] = True
            return f"Error: {e}"
