"""AWM Game Environment - Agent World Model Integration

Integrates Snowflake's Agent World Model (AWM) pre-generated environments
with the existing GRPO training infrastructure.

Paper: "Agent World Model: Infinity Synthetic Environments for Agentic RL"
Repo:  https://github.com/Snowflake-Labs/agent-world-model

Each episode:
  1. Picks a random scenario + task from 1K pre-generated environments
  2. Copies the SQLite database for isolation
  3. Loads the FastAPI environment code via TestClient (no HTTP overhead)
  4. Model interacts with tools to complete the user's task
  5. Code-based verifier computes reward from DB state changes

Implements GameEnv protocol + structured messages interface for GRPO training.
"""

import copy
import inspect
import json
import os
import random
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Tuple

AWM_DATA_DIR = os.environ.get(
    "AWM_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "outputs"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z0-9_]', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def _create_db(db_path: str, db_schema: dict, sample_data: dict = None):
    """Create SQLite database from AWM schema and insert sample data."""
    conn = sqlite3.connect(db_path)
    try:
        for table in db_schema.get("tables", []):
            ddl = table.get("ddl", "").strip()
            if ddl:
                conn.execute(ddl)
            for idx in table.get("indexes", []):
                if isinstance(idx, str) and idx.strip():
                    try:
                        conn.execute(idx.strip())
                    except sqlite3.Error:
                        pass

        if sample_data and isinstance(sample_data, dict):
            for table_name, rows in sample_data.items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if isinstance(row, dict) and row:
                        cols = list(row.keys())
                        phs = ", ".join(["?"] * len(cols))
                        col_str = ", ".join(f'"{c}"' for c in cols)
                        try:
                            conn.execute(
                                f'INSERT INTO "{table_name}" ({col_str}) VALUES ({phs})',
                                [row[c] for c in cols],
                            )
                        except sqlite3.Error:
                            pass
            conn.commit()
    finally:
        conn.close()


def _patch_env_code(code: str, db_path: str) -> str:
    """Patch AWM environment code: fix DB path, remove uvicorn.run."""
    lines = []
    in_main = False
    for line in code.split("\n"):
        # Replace create_engine() with our DB path
        if "create_engine(" in line:
            indent = len(line) - len(line.lstrip())
            line = (
                f'{" " * indent}engine = create_engine('
                f'"sqlite:///{db_path}", '
                f'connect_args={{"check_same_thread": False}})'
            )

        # Skip if __name__ == "__main__" block
        stripped = line.strip()
        if stripped.startswith("if __name__") and "__main__" in stripped:
            in_main = True
            continue
        if in_main:
            if line and not line[0].isspace() and stripped:
                in_main = False
            else:
                continue

        # Skip uvicorn.run
        if "uvicorn.run" in line:
            continue

        lines.append(line)

    return "\n".join(lines)


def _load_app(code: str, db_path: str):
    """Load FastAPI app from AWM environment code via exec()."""
    patched = _patch_env_code(code, db_path)
    patched = (
        "import warnings\n"
        "warnings.filterwarnings('ignore', category=DeprecationWarning)\n"
        + patched
    )

    namespace = {"__name__": "__awm_env__", "__file__": "awm_env.py"}
    exec(patched, namespace)

    app = namespace.get("app")
    if app is None:
        for obj in namespace.values():
            if hasattr(obj, "openapi") and callable(getattr(obj, "openapi")):
                app = obj
                break
    if app is None:
        raise RuntimeError("No FastAPI app found in environment code")
    return app


def _resolve_schema_ref(ref: str, schemas: dict) -> dict:
    """Resolve a JSON Schema $ref to the actual schema."""
    name = ref.split("/")[-1]
    return schemas.get(name, {"type": "string"})


def _flatten_schema_properties(schema: dict, schemas: dict) -> Tuple[dict, list]:
    """Recursively resolve and flatten a schema's properties."""
    props = {}
    required = list(schema.get("required", []))

    for pname, pinfo in schema.get("properties", {}).items():
        if "$ref" in pinfo:
            pinfo = _resolve_schema_ref(pinfo["$ref"], schemas)
        resolved = {}
        for k in ("type", "description", "enum", "default", "items"):
            if k in pinfo:
                resolved[k] = pinfo[k]
        if not resolved.get("type"):
            resolved["type"] = "string"
        props[pname] = resolved

    return props, required


def _extract_tool_schemas(app) -> Tuple[List[Dict], Dict]:
    """Extract OpenAI-format tool schemas from a FastAPI app's OpenAPI spec."""
    try:
        openapi = app.openapi()
    except Exception:
        return [], {}

    all_schemas = openapi.get("components", {}).get("schemas", {})
    tools = []
    tool_map = {}  # op_id -> (method, path_template)

    for path, methods in openapi.get("paths", {}).items():
        for method, details in methods.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue

            op_id = details.get("operationId")
            if not op_id:
                clean = path.replace("/", "_").replace("{", "").replace("}", "").strip("_")
                op_id = f"{method}_{clean}"

            summary = details.get("summary", "")
            description = details.get("description", summary) or f"{method.upper()} {path}"

            properties = {}
            required_params = []

            # Path and query parameters
            for param in details.get("parameters", []):
                pname = param["name"]
                pschema = param.get("schema", {"type": "string"})
                if "$ref" in pschema:
                    pschema = _resolve_schema_ref(pschema["$ref"], all_schemas)
                properties[pname] = {
                    "type": pschema.get("type", "string"),
                    "description": param.get("description", ""),
                }
                if param.get("required", False):
                    required_params.append(pname)

            # Request body
            body = details.get("requestBody", {})
            if body:
                content = body.get("content", {})
                json_body = content.get("application/json", {})
                body_schema = json_body.get("schema", {})

                if "$ref" in body_schema:
                    body_schema = _resolve_schema_ref(body_schema["$ref"], all_schemas)

                if body_schema.get("properties"):
                    body_props, body_req = _flatten_schema_properties(body_schema, all_schemas)
                    properties.update(body_props)
                    required_params.extend(body_req)

            tool = {
                "type": "function",
                "function": {
                    "name": op_id,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(set(required_params)),
                    },
                },
            }
            tools.append(tool)
            tool_map[op_id] = (method, path)

    return tools, tool_map


def _execute_tool(client, tool_name: str, arguments: dict, tool_map: dict) -> str:
    """Execute a tool call against the FastAPI TestClient."""
    if tool_name not in tool_map:
        avail = list(tool_map.keys())
        return json.dumps({"error": f"Unknown tool: {tool_name}", "available": avail[:20]})

    method, path_template = tool_map[tool_name]

    # Fill path parameters
    path_params = re.findall(r'\{(\w+)\}', path_template)
    actual_path = path_template
    remaining = dict(arguments) if arguments else {}

    for pp in path_params:
        if pp in remaining:
            actual_path = actual_path.replace(f'{{{pp}}}', str(remaining.pop(pp)))
        else:
            return json.dumps({"error": f"Missing required path parameter: {pp}"})

    try:
        if method == "get":
            resp = client.get(actual_path, params=remaining or None)
        elif method == "post":
            resp = client.post(actual_path, json=remaining or None)
        elif method == "put":
            resp = client.put(actual_path, json=remaining or None)
        elif method == "delete":
            resp = client.delete(actual_path, params=remaining or None)
        elif method == "patch":
            resp = client.patch(actual_path, json=remaining or None)
        else:
            return json.dumps({"error": f"Unsupported HTTP method: {method}"})

        return resp.text
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Verifier execution
# ---------------------------------------------------------------------------

def _call_verifier(verify_func, initial_db_path: str, final_db_path: str,
                   final_answer: str = ""):
    """Call verifier with signature-aware argument passing."""
    sig = inspect.signature(verify_func)
    params = sig.parameters
    kwargs = {}
    if "initial_db_path" in params:
        kwargs["initial_db_path"] = initial_db_path
    if "final_db_path" in params:
        kwargs["final_db_path"] = final_db_path
    if "final_answer" in params:
        kwargs["final_answer"] = final_answer
    if kwargs:
        return verify_func(**kwargs)
    # Positional fallback
    args = [initial_db_path, final_db_path]
    if len(params) >= 3:
        args.append(final_answer)
    return verify_func(*args)


def _run_code_verifier(verifier_entry: dict, initial_db: str,
                       final_db: str) -> Tuple[float, str]:
    """Run code-based verifier. Returns (reward, reason)."""
    verification = verifier_entry.get("verification", {})
    code = verification.get("code", "")
    if not code or len(code.strip()) < 10:
        return 0.0, "No verifier code available"

    func_name = "verify_task_completion"
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def verify_") and "(" in line:
            func_name = line.split("(")[0].replace("def ", "").strip()
            break

    namespace = {
        "sqlite3": sqlite3, "json": json, "os": os, "re": re,
        "__builtins__": __builtins__,
    }
    try:
        exec(code, namespace)
        verify_func = namespace.get(func_name)
        if not verify_func:
            return 0.0, f"Verifier function '{func_name}' not found"

        result = _call_verifier(verify_func, initial_db, final_db)

        if isinstance(result, dict):
            status = result.get("result", "others")
            if status == "complete":
                return 1.0, "Task completed successfully"
            return 0.0, f"Task incomplete: {json.dumps(result, default=str)[:200]}"
        return 0.0, f"Unexpected verifier result type: {type(result).__name__}"
    except Exception as e:
        return 0.0, f"Verifier error: {e}"


def _run_sql_verifier(verifier_entry: dict, initial_db: str,
                      final_db: str) -> Tuple[float, str]:
    """Run SQL-based verifier (heuristic reward without LLM judge)."""
    verification = verifier_entry.get("verification", {})
    code = verification.get("code", "")
    if not code or len(code.strip()) < 10:
        return 0.0, "No verifier code available"

    func_name = "verify_task"
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def verify_") and "(" in line:
            func_name = line.split("(")[0].replace("def ", "").strip()
            break

    namespace = {
        "sqlite3": sqlite3, "json": json, "os": os,
        "__builtins__": __builtins__,
    }
    try:
        exec(code, namespace)
        verify_func = namespace.get(func_name)
        if not verify_func:
            return 0.0, f"Verifier function '{func_name}' not found"

        result = _call_verifier(verify_func, initial_db, final_db)

        if isinstance(result, dict):
            if result.get("execution_status") == "error":
                return 0.0, f"Verifier error: {result.get('error_message', '')}"
            result_str = json.dumps(result, default=str).lower()
            if "complete" in result_str or "success" in result_str:
                return 1.0, "Task appears complete"
            return 0.0, f"Task state: {json.dumps(result, default=str)[:200]}"
        return 0.0, f"Unexpected result: {result}"
    except Exception as e:
        return 0.0, f"Verifier error: {e}"


# ---------------------------------------------------------------------------
# Action extraction
# ---------------------------------------------------------------------------

def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract tool call JSON from model output.

    Supports:
    - Qwen3 native tool-calling (function calls after </think>)
    - <tool_call>...</tool_call> XML format
    - Raw JSON {"name": ..., "arguments": ...}
    """
    if not text:
        return None

    # Try <tool_call> XML
    tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if tc_match:
        try:
            call = json.loads(tc_match.group(1))
            if "name" in call:
                return json.dumps(call)
        except json.JSONDecodeError:
            pass

    # Try to find {"name": ..., "arguments": ...} pattern
    for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+"', text):
        start = match.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        call = json.loads(text[start:i + 1])
                        if "name" in call:
                            if "arguments" not in call:
                                call["arguments"] = {}
                            return json.dumps(call)
                    except json.JSONDecodeError:
                        pass
                    break

    # Try text itself as JSON
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            call = json.loads(stripped)
            if "name" in call:
                return json.dumps(call)
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "<instructions>\n"
    "You are a helpful assistant interacting with a database-backed application.\n"
    "You have access to tools that let you read and modify data.\n"
    "In each turn you can either:\n"
    "- Make a tool call to interact with the application.\n"
    "- Use respond_to_user to send a final answer to the user.\n"
    "You cannot do both at the same time.\n\n"
    "Strategy:\n"
    "1. First explore the available data using read/list tools.\n"
    "2. Then perform the requested operations using write tools.\n"
    "3. Verify your changes were applied correctly.\n"
    "4. Use respond_to_user to confirm completion.\n\n"
    "Always make sure you generate valid JSON only.\n"
    "</instructions>"
)


# ---------------------------------------------------------------------------
# AWM Game Environment
# ---------------------------------------------------------------------------

class AWMGame:
    """AWM tool-calling game environment for GRPO training.

    Implements the GameEnv protocol with structured messages interface,
    matching the pattern of existing games (multistep_task, tau_tool_calling).
    """

    supports_structured_messages = True

    def __init__(self, data_dir: str = None, max_steps: int = 20,
                 user_client=None):
        self.data_dir = data_dir or AWM_DATA_DIR
        self.max_steps = max_steps
        self._user_client = user_client  # unused but kept for API compat

        # Loaded data (populated by _load_data)
        self._envs: Dict[str, dict] = {}
        self._tasks: Dict[str, list] = {}
        self._verifiers: Dict[Tuple[str, int], dict] = {}
        self._db_schemas: Dict[str, dict] = {}
        self._samples: Dict[str, dict] = {}
        self._scenario_list: List[str] = []
        self._verifier_mode: str = "code"

        self._load_data()

        # GameEnv protocol
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        # Per-episode state
        self._app = None
        self._client = None
        self._tool_schemas: List[Dict] = []
        self._tool_map: Dict[str, Tuple[str, str]] = {}
        self._conversation: List[Dict[str, str]] = []
        self._tool_calls_history: List[Dict] = []
        self._step_count: int = 0
        self._scenario_name: Optional[str] = None
        self._task: Optional[str] = None
        self._task_id: Optional[int] = None
        self._temp_dir: Optional[str] = None
        self._initial_db: Optional[str] = None
        self._working_db: Optional[str] = None
        self._verifier_entry: Optional[dict] = None
        self._reason: str = ""
        self._env_loaded: bool = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        envs_path = os.path.join(self.data_dir, "gen_envs.jsonl")
        tasks_path = os.path.join(self.data_dir, "gen_tasks.jsonl")
        db_path = os.path.join(self.data_dir, "gen_db.jsonl")
        sample_path = os.path.join(self.data_dir, "gen_sample.jsonl")
        verifier_code_path = os.path.join(self.data_dir, "gen_verifier.pure_code.jsonl")
        verifier_sql_path = os.path.join(self.data_dir, "gen_verifier.jsonl")

        if not os.path.exists(envs_path):
            raise FileNotFoundError(
                f"AWM environments not found at {envs_path}. "
                "Run: bash setup_awm.sh"
            )

        for e in _load_jsonl(envs_path):
            self._envs[_norm(e["scenario"])] = e

        if os.path.exists(tasks_path):
            for t in _load_jsonl(tasks_path):
                self._tasks[_norm(t["scenario"])] = t.get("tasks", [])

        if os.path.exists(db_path):
            for d in _load_jsonl(db_path):
                self._db_schemas[_norm(d["scenario"])] = d.get("db_schema", {})

        if os.path.exists(sample_path):
            for s in _load_jsonl(sample_path):
                self._samples[_norm(s["scenario"])] = s.get("sample_data", {})

        # Load verifiers (prefer code-based)
        if os.path.exists(verifier_code_path):
            self._verifier_mode = "code"
            for v in _load_jsonl(verifier_code_path):
                key = (_norm(v["scenario"]), v.get("task_idx", 0))
                self._verifiers[key] = v
        elif os.path.exists(verifier_sql_path):
            self._verifier_mode = "sql"
            for v in _load_jsonl(verifier_sql_path):
                key = (_norm(v["scenario"]), v.get("task_idx", 0))
                self._verifiers[key] = v

        # Build valid scenario list
        db_dir = os.path.join(self.data_dir, "databases")
        for name in self._envs:
            if name not in self._tasks or not self._tasks[name]:
                continue
            # Need either DB schema or existing DB file
            has_db = (
                name in self._db_schemas
                or os.path.exists(os.path.join(db_dir, f"{name}.db"))
            )
            env_entry = self._envs[name]
            env_db = env_entry.get("db_path", "")
            if env_db and not os.path.isabs(env_db):
                env_db = os.path.normpath(
                    os.path.join(os.path.dirname(self.data_dir), env_db)
                )
            if has_db or (env_db and os.path.exists(env_db)):
                self._scenario_list.append(name)

        if not self._scenario_list:
            raise RuntimeError(
                f"No valid AWM scenarios found in {self.data_dir}. "
                "Ensure gen_envs.jsonl and gen_tasks.jsonl exist, "
                "and databases are available."
            )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def _cleanup(self):
        if getattr(self, "_client", None) is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._app = None

        temp = getattr(self, "_temp_dir", None)
        if temp and os.path.exists(temp):
            try:
                shutil.rmtree(temp)
            except Exception:
                pass
            self._temp_dir = None

    def __del__(self):
        self._cleanup()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _setup_database(self, scenario_name: str):
        """Create a fresh copy of the database for this episode."""
        self._temp_dir = tempfile.mkdtemp(prefix="awm_")
        self._working_db = os.path.join(self._temp_dir, "working.db")
        self._initial_db = os.path.join(self._temp_dir, "initial.db")

        # Try to find existing DB file
        db_dir = os.path.join(self.data_dir, "databases")
        candidates = [
            os.path.join(db_dir, f"{scenario_name}.db"),
        ]
        env_db = self._envs[scenario_name].get("db_path", "")
        if env_db:
            if os.path.isabs(env_db):
                candidates.insert(0, env_db)
            else:
                candidates.insert(
                    0,
                    os.path.normpath(
                        os.path.join(os.path.dirname(self.data_dir), env_db)
                    ),
                )

        copied = False
        for db_file in candidates:
            if os.path.exists(db_file):
                shutil.copy2(db_file, self._working_db)
                copied = True
                break

        if not copied:
            # Create from schema + sample data
            schema = self._db_schemas.get(scenario_name, {})
            sample = self._samples.get(scenario_name)
            if not schema:
                raise RuntimeError(f"No database found for scenario {scenario_name}")
            _create_db(self._working_db, schema, sample)

        shutil.copy2(self._working_db, self._initial_db)

    # ------------------------------------------------------------------
    # GameEnv protocol: reset
    # ------------------------------------------------------------------

    def reset(self, seed: int) -> None:
        self._cleanup()
        rng = random.Random(seed)

        self._scenario_name = rng.choice(self._scenario_list)
        tasks = self._tasks[self._scenario_name]
        self._task_id = rng.randint(0, len(tasks) - 1)
        self._task = tasks[self._task_id]

        self._verifier_entry = self._verifiers.get(
            (self._scenario_name, self._task_id)
        )

        self._setup_database(self._scenario_name)

        # Load environment
        self._env_loaded = False
        try:
            env_entry = self._envs[self._scenario_name]
            code = env_entry["full_code"]
            self._app = _load_app(code, self._working_db)
            from fastapi.testclient import TestClient
            self._client = TestClient(self._app)
            self._tool_schemas, self._tool_map = _extract_tool_schemas(self._app)
            self._env_loaded = True
        except Exception:
            self._tool_schemas = []
            self._tool_map = {}

        # Always include respond_to_user for final answers
        self._tool_schemas.append({
            "type": "function",
            "function": {
                "name": "respond_to_user",
                "description": (
                    "Send a message to the user. Use this to provide your "
                    "final answer or ask for clarification."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send",
                        }
                    },
                    "required": ["message"],
                },
            },
        })

        self._conversation = [{"role": "user", "content": self._task}]
        self._tool_calls_history = []
        self._step_count = 0

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None
        self._reason = ""

    # ------------------------------------------------------------------
    # Structured messages interface
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tool_schemas(self) -> List[Dict]:
        return self._tool_schemas

    def get_tool_schemas_compact(self) -> List[Dict]:
        """Compact schemas: strip descriptions to save prompt tokens."""
        compact = []
        for tool in self._tool_schemas:
            t = copy.deepcopy(tool)
            func = t.get("function", {})
            func.pop("description", None)
            params = func.get("parameters", {})
            for prop in params.get("properties", {}).values():
                prop.pop("description", None)
            compact.append(t)
        return compact

    def get_messages(self) -> List[Dict]:
        msgs = []
        for msg in self._conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                msgs.append({"role": "user", "content": content})
            elif role == "assistant":
                msgs.append({"role": "assistant", "content": content})
            elif role == "tool_call":
                tc = json.loads(content)
                msgs.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": f"call_{len(msgs)}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }],
                })
            elif role == "tool_result":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": f"call_{len(msgs) - 1}",
                    "content": content,
                })
        return msgs

    # ------------------------------------------------------------------
    # GameEnv protocol: observe / legal_actions / step
    # ------------------------------------------------------------------

    def observe(self, player_id: int) -> str:
        return "This game uses the tool-calling interface, not observe()."

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]) -> None:
        if self.done:
            return

        self._step_count += 1

        if action is None:
            self._finalize(0.0, "No action provided")
            return

        # Parse tool call
        try:
            tool_call = json.loads(action)
        except json.JSONDecodeError:
            self._finalize(0.0, "Invalid JSON action")
            self.invalid_player = 0
            return

        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        # respond_to_user -> end episode
        if tool_name == "respond_to_user":
            message = arguments.get("message", str(arguments))
            self._conversation.append({"role": "assistant", "content": message})
            reward, reason = self._compute_reward()
            self._finalize(reward, reason)
            return

        # Record and execute tool call
        self._conversation.append({
            "role": "tool_call",
            "content": json.dumps(tool_call),
        })
        self._tool_calls_history.append(tool_call)

        if not self._env_loaded:
            result = json.dumps({"error": "Environment failed to load"})
        elif tool_name in self._tool_map:
            result = _execute_tool(
                self._client, tool_name, arguments, self._tool_map
            )
        else:
            result = json.dumps({
                "error": f"Unknown tool: {tool_name}",
                "available": list(self._tool_map.keys())[:20],
            })

        # Truncate very long results to avoid context blowup
        if len(result) > 4000:
            result = result[:4000] + "\n... (truncated)"

        self._conversation.append({"role": "tool_result", "content": result})

        # Max steps
        if self._step_count >= self.max_steps:
            reward, reason = self._compute_reward()
            self._finalize(reward, f"Max steps reached. {reason}")

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> Tuple[float, str]:
        if not self._verifier_entry:
            if self._tool_calls_history:
                return 0.3, "No verifier; partial credit for tool usage"
            return 0.0, "No verifier and no tool calls"

        if self._verifier_mode == "code":
            return _run_code_verifier(
                self._verifier_entry, self._initial_db, self._working_db
            )
        return _run_sql_verifier(
            self._verifier_entry, self._initial_db, self._working_db
        )

    def _finalize(self, reward: float, reason: str):
        self.done = True
        self.rewards = {0: max(0.0, min(1.0, reward))}
        self._reason = reason

    def get_summary(self) -> Dict[str, Any]:
        return {
            "scenario": self._scenario_name,
            "task_id": self._task_id,
            "task": self._task,
            "steps": self._step_count,
            "tool_calls": self._tool_calls_history,
            "reward": self.rewards.get(0, 0.0),
            "reason": self._reason,
            "domain": "awm",
        }
