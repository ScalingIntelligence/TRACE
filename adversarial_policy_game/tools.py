"""Tool execution using tau2-bench's actual implementations.

Wraps tau2-bench AirlineTools/RetailTools directly so tool behavior
is identical to the tau2-bench evaluation. No re-implementations.
"""

import sys
import json
import copy
import pathlib
from datetime import datetime, date
from typing import Dict, Any, List

from pydantic import BaseModel

# Ensure tau2-bench source is in path
_TAU2_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "tau2-bench" / "src")
if _TAU2_SRC not in sys.path:
    sys.path.insert(0, _TAU2_SRC)

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.retail.tools import RetailTools


def to_json_str(resp: Any) -> str:
    """Convert tool response to JSON string.

    Matches tau2-bench Environment.to_json_str() exactly.
    """
    def _process(resp: Any) -> Any:
        if isinstance(resp, BaseModel):
            return resp.model_dump()
        elif isinstance(resp, str):
            return resp
        elif resp is None:
            return resp
        elif isinstance(resp, (int, float, bool)):
            return str(resp)
        elif isinstance(resp, list):
            return [_process(item) for item in resp]
        elif isinstance(resp, tuple):
            return tuple(_process(item) for item in resp)
        elif isinstance(resp, dict):
            return {k: _process(v) for k, v in resp.items()}
        elif isinstance(resp, (datetime, date)):
            return resp.isoformat()
        else:
            return str(resp)

    if not isinstance(resp, str):
        return json.dumps(_process(resp), default=str)
    return resp


class ToolExecutor:
    """Executes tools using tau2-bench's actual implementations."""

    def __init__(self, domain: str, db: Any):
        """Initialize with domain and Pydantic database model.

        Args:
            domain: "airline" or "retail"
            db: FlightDB or RetailDB Pydantic model
        """
        self.domain = domain
        self._db = db
        self._tool_call_log: List[Dict[str, Any]] = []

        if domain == "airline":
            self._tools = AirlineTools(db=db)
        else:
            self._tools = RetailTools(db=db)

    @property
    def db(self) -> Dict[str, Any]:
        """Return DB as plain dict (for verification)."""
        return self._db.model_dump()

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        return list(self._tool_call_log)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return result string."""
        # Normalize retail order_id: DB stores "#W..." but models often
        # strip the '#'. This is orthogonal to policy adherence (the skill
        # we're training), so we auto-correct to avoid blocking training.
        if self.domain == "retail" and "order_id" in arguments:
            oid = str(arguments["order_id"])
            if oid and not oid.startswith("#"):
                arguments = {**arguments, "order_id": f"#{oid}"}

        entry = {
            "name": tool_name,
            "arguments": copy.deepcopy(arguments),
        }
        self._tool_call_log.append(entry)

        try:
            result = self._tools.use_tool(tool_name, **arguments)
            entry["error"] = False
            return to_json_str(result)
        except Exception as e:
            entry["error"] = True
            return f"Error: {e}"
