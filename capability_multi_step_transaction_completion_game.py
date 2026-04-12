"""Synthetic multi-step transaction-completion environment.

Targets the `multi_step_transaction_completion` capability identified in
`pipeline/selected_capabilities.json`: the agent must execute every required
state-changing sub-action of a multi-step user request rather than dropping
or short-circuiting steps.

Domain
------
A small synthetic project / todo "Ledger" application. The initial state
(one project, several todos with priority/assignee/due_date/tags/done) is
generated from the seed and dumped into the system prompt so the model has
full visibility. The user then asks for a list of 2-5 distinct CRUD
operations (priority change, reassign, due-date move, mark done, delete,
add tag). The agent has to call the matching tool for every requested
operation.

Reward
------
Dense and per-action:

    reward = n_required_actions_correctly_executed / n_required_actions

The agent terminates the episode by calling `respond_to_user`. If it
responds before completing every required action, it receives partial
credit. Read-only or off-script tool calls are tolerated (no penalty) but
do not earn credit. After `_max_steps` tool calls without a `respond_to_user`,
the episode auto-terminates with whatever credit has been accumulated.

This is intentionally NOT a tau2-bench airline/retail wrapper. It only
matches the OpenAI chat-completions tool-calling shape (so the same vLLM
client + `train/collect_rollouts.py` loop drives it) but uses an
independent domain so the model can't lean on prior knowledge of tau2.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from game_registry import GameSpec, register_game


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI chat-completions format — what vLLM serves)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_todo_priority",
            "description": (
                "Update the priority of an existing todo. priority must be one of "
                "low, medium, high, urgent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                    },
                },
                "required": ["todo_id", "priority"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_todo",
            "description": "Reassign an existing todo to a project member.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {"type": "string"},
                    "assignee": {"type": "string"},
                },
                "required": ["todo_id", "assignee"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_todo_due_date",
            "description": "Set the due date of an existing todo (ISO format YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {"type": "string"},
                    "due_date": {"type": "string"},
                },
                "required": ["todo_id", "due_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_todo_done",
            "description": "Mark an existing todo as completed.",
            "parameters": {
                "type": "object",
                "properties": {"todo_id": {"type": "string"}},
                "required": ["todo_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_todo",
            "description": "Permanently delete a todo.",
            "parameters": {
                "type": "object",
                "properties": {"todo_id": {"type": "string"}},
                "required": ["todo_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_tag",
            "description": "Attach a new tag to an existing todo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {"type": "string"},
                    "tag": {"type": "string"},
                },
                "required": ["todo_id", "tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "respond_to_user",
            "description": (
                "Send a final confirmation message to the user. Call this ONLY "
                "after every operation the user requested has been executed via "
                "the appropriate tool calls. Calling this prematurely ends the "
                "session with the user's request half-finished."
            ),
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Static content used by scenario generation
# ---------------------------------------------------------------------------
_TODO_TITLES = [
    "Draft Q2 launch plan",
    "Review pull request #482",
    "Schedule design sync",
    "Update onboarding docs",
    "Investigate prod latency spike",
    "Triage bug backlog",
    "Prepare board slides",
    "Migrate auth service",
    "Write postmortem",
    "Refactor billing module",
    "Audit S3 permissions",
    "Set up staging cluster",
    "Polish landing page hero",
    "Run user interviews",
    "Cut release v3.4",
    "Document incident response",
]

_USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]
_PRIORITIES = ["low", "medium", "high", "urgent"]
_TAGS = ["frontend", "backend", "infra", "design", "blocked", "research", "perf"]
_DUE_DATES = [
    "2026-05-01", "2026-05-08", "2026-05-15", "2026-05-22",
    "2026-05-29", "2026-06-05", "2026-06-12", "2026-06-19",
]
_PROJECT_NAMES = ["Phoenix", "Helix", "Northstar", "Atlas", "Glacier", "Lantern"]


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------
class MultiStepLedgerGame:
    """Synthetic multi-step ledger task targeting transaction completion."""

    def __init__(self, max_steps: int = 24, hint: bool = False):
        self._max_steps = int(max_steps)
        self._hint = bool(hint)

        # Per-episode state, populated by reset().
        self.done: bool = False
        self.current_player: int = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

        self._steps: int = 0
        self._reason: str = ""
        self._tool_call_counter: int = 0

        self._project_name: str = ""
        self._members: List[str] = []
        self._todos: Dict[str, Dict[str, Any]] = {}
        self._todo_order: List[str] = []
        self._user_request: str = ""
        self._required_actions: List[Dict[str, Any]] = []
        self._completed_required_idx: Set[int] = set()

        self._oai_messages: List[Dict[str, Any]] = []
        self._conversation: List[Dict[str, str]] = []

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------
    def reset(self, seed: int) -> None:
        rng = random.Random(int(seed))

        self._project_name = rng.choice(_PROJECT_NAMES)
        self._members = rng.sample(_USERS, k=rng.randint(3, 5))

        n_todos = rng.randint(10, 14)
        titles = rng.sample(_TODO_TITLES, k=n_todos)
        self._todos = {}
        self._todo_order = []
        for i, title in enumerate(titles, start=1):
            tid = f"t{i}"
            self._todos[tid] = {
                "id": tid,
                "title": title,
                "priority": rng.choice(_PRIORITIES),
                "assignee": rng.choice(self._members),
                "due_date": rng.choice(_DUE_DATES),
                "tags": rng.sample(_TAGS, k=rng.randint(1, 3)),
                "done": False,
            }
            self._todo_order.append(tid)

        self._user_request, self._required_actions = self._generate_request(rng)

        self.done = False
        self.rewards = {0: 0.0}
        self.invalid_player = None
        self._steps = 0
        self._completed_required_idx = set()
        self._tool_call_counter = 0
        self._reason = ""

        self._oai_messages = [{"role": "user", "content": self._user_request}]
        self._conversation = [{"role": "user", "text": self._user_request}]

    # -----------------------------------------------------------------
    # Scenario generation
    # -----------------------------------------------------------------
    def _generate_request(
        self, rng: random.Random
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a multi-step request whose required actions are a MIX of:
        - per-todo ops (referenced by id)
        - selector/bulk ops ("for every todo with X, do Y") that expand into
          multiple required tool calls the agent has to enumerate from state.

        The mix is what stresses transaction completion: the agent must (a)
        scan the state for matching todos, (b) issue the right number of
        tool calls, and (c) not stop early.
        """
        lines: List[str] = []
        required: List[Dict[str, Any]] = []

        # ---- bulk / selector ops (1 or 2 of these) -----------------------
        n_bulk = rng.randint(1, 2)
        used_bulk: Set[str] = set()
        for _ in range(n_bulk):
            kind = rng.choice([
                "mark_priority_done",
                "reassign_user",
                "tag_priority",
                "due_date_priority",
            ])
            if kind in used_bulk:
                continue
            used_bulk.add(kind)

            if kind == "mark_priority_done":
                target_p = rng.choice(_PRIORITIES)
                matched = [
                    tid for tid in self._todo_order
                    if self._todos[tid]["priority"] == target_p
                    and not self._todos[tid]["done"]
                ]
                if len(matched) >= 2:
                    lines.append(
                        f'- Mark every {target_p}-priority todo as done.'
                    )
                    for tid in matched:
                        required.append({
                            "name": "mark_todo_done",
                            "args": {"todo_id": tid},
                            "match_keys": ["todo_id"],
                        })

            elif kind == "reassign_user":
                from_users = [
                    u for u in self._members
                    if sum(1 for tid in self._todo_order
                           if self._todos[tid]["assignee"] == u) >= 2
                ]
                if from_users:
                    src = rng.choice(from_users)
                    dst_choices = [m for m in self._members if m != src]
                    if dst_choices:
                        dst = rng.choice(dst_choices)
                        matched = [
                            tid for tid in self._todo_order
                            if self._todos[tid]["assignee"] == src
                        ]
                        lines.append(
                            f'- Reassign every todo currently assigned to '
                            f'{src} to {dst}.'
                        )
                        for tid in matched:
                            required.append({
                                "name": "assign_todo",
                                "args": {"todo_id": tid, "assignee": dst},
                                "match_keys": ["todo_id", "assignee"],
                            })

            elif kind == "tag_priority":
                target_p = rng.choice(_PRIORITIES)
                new_tag = rng.choice(_TAGS)
                matched = [
                    tid for tid in self._todo_order
                    if self._todos[tid]["priority"] == target_p
                    and new_tag not in self._todos[tid]["tags"]
                ]
                if len(matched) >= 2:
                    lines.append(
                        f'- Add the tag "{new_tag}" to every {target_p}-priority todo.'
                    )
                    for tid in matched:
                        required.append({
                            "name": "add_tag",
                            "args": {"todo_id": tid, "tag": new_tag},
                            "match_keys": ["todo_id", "tag"],
                        })

            elif kind == "due_date_priority":
                target_p = rng.choice(_PRIORITIES)
                new_d = rng.choice(_DUE_DATES)
                matched = [
                    tid for tid in self._todo_order
                    if self._todos[tid]["priority"] == target_p
                    and self._todos[tid]["due_date"] != new_d
                ]
                if len(matched) >= 2:
                    lines.append(
                        f'- Move the due date of every {target_p}-priority '
                        f'todo to {new_d}.'
                    )
                    for tid in matched:
                        required.append({
                            "name": "set_todo_due_date",
                            "args": {"todo_id": tid, "due_date": new_d},
                            "match_keys": ["todo_id", "due_date"],
                        })

        # ---- per-todo ops (2-4 of these) ---------------------------------
        n_per = rng.randint(2, 4)
        # Avoid colliding with bulk targets where the selector already
        # owns the todo for a given tool — use a different tool when
        # possible.
        already_targeted = {
            (req["name"], req["args"].get("todo_id"))
            for req in required
        }

        per_pool = ["set_priority", "assign", "set_due", "mark_done", "delete", "tag"]
        rng.shuffle(per_pool)

        candidate_ids = [
            tid for tid in self._todo_order
            if all((opname, tid) not in already_targeted for opname in [
                "set_todo_priority", "assign_todo", "set_todo_due_date",
                "mark_todo_done", "delete_todo", "add_tag",
            ])
        ]
        rng.shuffle(candidate_ids)

        for op in per_pool[:n_per]:
            if not candidate_ids:
                break
            tid = candidate_ids.pop()
            todo = self._todos[tid]

            if op == "set_priority":
                new_p = rng.choice([p for p in _PRIORITIES if p != todo["priority"]])
                lines.append(
                    f'- Set the priority of todo {tid} ("{todo["title"]}") to {new_p}.'
                )
                required.append({
                    "name": "set_todo_priority",
                    "args": {"todo_id": tid, "priority": new_p},
                    "match_keys": ["todo_id", "priority"],
                })

            elif op == "assign":
                others = [m for m in self._members if m != todo["assignee"]]
                new_a = rng.choice(others) if others else todo["assignee"]
                lines.append(
                    f'- Reassign todo {tid} ("{todo["title"]}") to {new_a}.'
                )
                required.append({
                    "name": "assign_todo",
                    "args": {"todo_id": tid, "assignee": new_a},
                    "match_keys": ["todo_id", "assignee"],
                })

            elif op == "set_due":
                new_d = rng.choice([d for d in _DUE_DATES if d != todo["due_date"]])
                lines.append(
                    f'- Move the due date of todo {tid} ("{todo["title"]}") to {new_d}.'
                )
                required.append({
                    "name": "set_todo_due_date",
                    "args": {"todo_id": tid, "due_date": new_d},
                    "match_keys": ["todo_id", "due_date"],
                })

            elif op == "mark_done":
                lines.append(
                    f'- Mark todo {tid} ("{todo["title"]}") as done.'
                )
                required.append({
                    "name": "mark_todo_done",
                    "args": {"todo_id": tid},
                    "match_keys": ["todo_id"],
                })

            elif op == "delete":
                lines.append(
                    f'- Delete todo {tid} ("{todo["title"]}").'
                )
                required.append({
                    "name": "delete_todo",
                    "args": {"todo_id": tid},
                    "match_keys": ["todo_id"],
                })

            elif op == "tag":
                candidate_tags = [t for t in _TAGS if t not in todo["tags"]]
                new_tag = rng.choice(candidate_tags) if candidate_tags else "blocked"
                lines.append(
                    f'- Add the tag "{new_tag}" to todo {tid} ("{todo["title"]}").'
                )
                required.append({
                    "name": "add_tag",
                    "args": {"todo_id": tid, "tag": new_tag},
                    "match_keys": ["todo_id", "tag"],
                })

        # Fallback: if for some reason no required actions got generated
        # (very small project, all bulk filters empty), fall back to a
        # simple per-todo op so the episode is still scorable.
        if not required:
            tid = rng.choice(self._todo_order)
            new_p = rng.choice([p for p in _PRIORITIES
                                if p != self._todos[tid]["priority"]])
            lines.append(
                f'- Set the priority of todo {tid} '
                f'("{self._todos[tid]["title"]}") to {new_p}.'
            )
            required.append({
                "name": "set_todo_priority",
                "args": {"todo_id": tid, "priority": new_p},
                "match_keys": ["todo_id", "priority"],
            })

        rng.shuffle(lines)
        request = (
            f"I need several updates to the {self._project_name} project. "
            f"Please perform ALL of the following operations and then confirm "
            f"once every one is finished:\n\n"
            + "\n".join(lines)
            + "\n\nDo not stop until every operation above has been executed."
        )
        return request, required

    # -----------------------------------------------------------------
    # System prompt + state dump (per-episode)
    # -----------------------------------------------------------------
    def get_system_prompt(self) -> str:
        prompt = (
            "You are a project-management assistant for the Ledger app. You "
            "help users manage their projects and todos by calling the tools "
            "provided.\n\n"
            "RULES:\n"
            "1. When the user asks you to perform a sequence of operations, you "
            "MUST execute EVERY operation via the appropriate tool call.\n"
            "2. Do NOT stop after the first operation. Do NOT skip any.\n"
            "3. Only after every operation is complete, call `respond_to_user` "
            "to confirm. Calling `respond_to_user` early ends the session with "
            "the request half-finished.\n"
            "4. Make one tool call at a time. Wait for the tool result, then "
            "decide the next call.\n\n"
            f"## Current state of project '{self._project_name}'\n\n"
            f"Members: {', '.join(self._members)}\n\n"
            "Todos:\n"
        )
        for tid in self._todo_order:
            t = self._todos[tid]
            tags = ",".join(t["tags"]) if t["tags"] else "—"
            prompt += (
                f"- {tid} | title: \"{t['title']}\" | priority: {t['priority']} "
                f"| assignee: {t['assignee']} | due: {t['due_date']} "
                f"| tags: {tags} | done: {str(t['done']).lower()}\n"
            )

        if self._hint:
            prompt += (
                "\n<expert_guidance>\n"
                "CRITICAL OVERRIDE — execute exactly these tool calls in order, "
                "then call respond_to_user.\n"
            )
            for i, req in enumerate(self._required_actions, start=1):
                prompt += (
                    f"STEP {i}: call `{req['name']}` with arguments "
                    f"{json.dumps(req['args'])}\n"
                )
            prompt += (
                "FINAL: call `respond_to_user` with a brief confirmation message.\n"
                "</expert_guidance>\n"
            )

        return prompt

    # -----------------------------------------------------------------
    # Accessors used by collect_rollouts
    # -----------------------------------------------------------------
    def get_messages(self) -> List[Dict[str, Any]]:
        return self._oai_messages

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    # -----------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------
    def step(self, action: Optional[str]) -> None:
        if self.done:
            return

        if action is None:
            self._steps += 1
            self._maybe_finalize()
            return

        try:
            parsed = json.loads(action)
            name = parsed.get("name", "") or ""
            args = parsed.get("arguments", {}) or {}
            if not isinstance(args, dict):
                args = {}
        except Exception:
            self._steps += 1
            self._maybe_finalize()
            return

        if name == "respond_to_user":
            msg = args.get("message", "") if isinstance(args, dict) else ""
            self._oai_messages.append({
                "role": "assistant",
                "content": msg,
                "tool_calls": None,
            })
            self._conversation.append({"role": "assistant", "text": msg})
            self._finalize(reason="agent called respond_to_user")
            return

        # Any other tool call: execute it, record it, possibly credit it.
        result_str = self._execute_tool(name, args)
        self._record_tool_call(name, args, result_str)
        self._steps += 1
        self._maybe_finalize()

    def _record_tool_call(
        self, name: str, args: Dict[str, Any], result: str
    ) -> None:
        self._tool_call_counter += 1
        tc_id = f"call_{self._tool_call_counter:04d}"
        self._oai_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }],
        })
        self._oai_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": result,
        })
        self._conversation.append({
            "role": "tool_call",
            "text": json.dumps({"name": name, "arguments": args}),
        })
        self._conversation.append({"role": "tool_result", "text": result})

    # -----------------------------------------------------------------
    # Tool implementations (in-memory state mutations)
    # -----------------------------------------------------------------
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "set_todo_priority":
            tid = args.get("todo_id")
            priority = args.get("priority")
            if tid not in self._todos:
                return json.dumps({"error": f"unknown todo_id '{tid}'"})
            if priority not in _PRIORITIES:
                return json.dumps({
                    "error": f"invalid priority '{priority}'; must be one of {_PRIORITIES}"
                })
            self._todos[tid]["priority"] = priority
            self._maybe_credit(name, args)
            return json.dumps({"ok": True, "todo": self._todos[tid]})

        if name == "assign_todo":
            tid = args.get("todo_id")
            assignee = args.get("assignee")
            if tid not in self._todos:
                return json.dumps({"error": f"unknown todo_id '{tid}'"})
            if not assignee:
                return json.dumps({"error": "assignee is required"})
            self._todos[tid]["assignee"] = assignee
            self._maybe_credit(name, args)
            return json.dumps({"ok": True, "todo": self._todos[tid]})

        if name == "set_todo_due_date":
            tid = args.get("todo_id")
            due = args.get("due_date")
            if tid not in self._todos:
                return json.dumps({"error": f"unknown todo_id '{tid}'"})
            if not due:
                return json.dumps({"error": "due_date is required"})
            self._todos[tid]["due_date"] = due
            self._maybe_credit(name, args)
            return json.dumps({"ok": True, "todo": self._todos[tid]})

        if name == "mark_todo_done":
            tid = args.get("todo_id")
            if tid not in self._todos:
                return json.dumps({"error": f"unknown todo_id '{tid}'"})
            self._todos[tid]["done"] = True
            self._maybe_credit(name, args)
            return json.dumps({"ok": True, "todo": self._todos[tid]})

        if name == "delete_todo":
            tid = args.get("todo_id")
            if tid not in self._todos:
                return json.dumps({"error": f"unknown todo_id '{tid}'"})
            removed = self._todos.pop(tid)
            if tid in self._todo_order:
                self._todo_order.remove(tid)
            self._maybe_credit(name, args)
            return json.dumps({"ok": True, "deleted": removed})

        if name == "add_tag":
            tid = args.get("todo_id")
            tag = args.get("tag")
            if tid not in self._todos:
                return json.dumps({"error": f"unknown todo_id '{tid}'"})
            if not tag:
                return json.dumps({"error": "tag is required"})
            if tag not in self._todos[tid]["tags"]:
                self._todos[tid]["tags"].append(tag)
            self._maybe_credit(name, args)
            return json.dumps({"ok": True, "todo": self._todos[tid]})

        return json.dumps({"error": f"unknown tool '{name}'"})

    def _maybe_credit(self, name: str, args: Dict[str, Any]) -> None:
        for i, req in enumerate(self._required_actions):
            if i in self._completed_required_idx:
                continue
            if req["name"] != name:
                continue
            if all(args.get(k) == req["args"].get(k) for k in req["match_keys"]):
                self._completed_required_idx.add(i)
                return

    # -----------------------------------------------------------------
    # Termination
    # -----------------------------------------------------------------
    def _maybe_finalize(self) -> None:
        if self._steps >= self._max_steps:
            self._finalize(reason=f"max_steps ({self._max_steps}) reached")

    def _finalize(self, reason: str) -> None:
        n_done = len(self._completed_required_idx)
        n_total = len(self._required_actions)
        reward = (n_done / n_total) if n_total > 0 else 0.0
        self.rewards = {0: float(reward)}
        self.done = True
        self._reason = (
            f"{reason}; completed {n_done}/{n_total} required actions"
        )

    # -----------------------------------------------------------------
    # Summary for collect_rollouts
    # -----------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        return {
            "reward": self.rewards.get(0, 0.0),
            "reason": self._reason,
            "steps": self._steps,
            "domain": "multi_step_ledger",
            "n_required": len(self._required_actions),
            "n_completed": len(self._completed_required_idx),
        }

    # -----------------------------------------------------------------
    # GameEnv protocol stubs (collect_rollouts uses the rich interface
    # above; these are here so the class still satisfies game_registry's
    # Protocol typing).
    # -----------------------------------------------------------------
    def observe(self, player_id: int) -> str:
        return self.get_system_prompt() + "\n\n" + json.dumps(self._oai_messages)

    def legal_actions(self) -> List[str]:
        return []


def _no_op_extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
register_game(GameSpec(
    name="capability_multi_step_transaction_completion",
    make_env=lambda **kw: MultiStepLedgerGame(**kw),
    extract_action=_no_op_extract_action,
    system_prompt="",  # per-episode prompt comes from get_system_prompt()
    max_gen_tokens=1024,
))
