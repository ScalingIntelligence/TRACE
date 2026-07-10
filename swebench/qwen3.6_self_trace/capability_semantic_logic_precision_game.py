"""Synthetic env for capability `semantic-logic-precision`.

Targets the largest failure mode of Qwen3.6-27B on SWE-bench Verified
(discovered open-ended in phase 2 v2: 61.0% of unresolved instances):
model implements a logically incorrect algorithm, misunderstanding
domain protocols, mathematical rules, or language semantics.

Reuses parser + apply + reward machinery from
``capability_regression_safe_edit_game`` (same single-turn agentless-mini
SEARCH/REPLACE format). Scenarios come from
``trace_v4_scenarios.ALL_SEMANTIC_LOGIC_SCENARIOS``.

Format: single-turn repair (SEARCH/REPLACE inside ```python fences).
Reward = 0.4*target_passes + 0.4*no_regression + 0.1*minimal_edit +
0.1*format_fidelity (same composition as the other capability envs).
"""
from __future__ import annotations

import os
import random
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from game_registry import GameSpec, register_game
from trace_v4_scenarios import (
    SemanticLogicScenario,
    ALL_SEMANTIC_LOGIC_SCENARIOS,
    get_semantic_logic_scenario,
)
from capability_regression_safe_edit_game import (
    SYSTEM_PROMPT_BASE, USER_TEMPLATE, EXPERT_GUIDANCE_TEMPLATE,
    parse_search_replace, apply_edits, _run_pytest, _Edit,
)


@dataclass
class _SLPState:
    seed: int = 0
    workdir: Optional[Path] = None
    scenario: Optional[SemanticLogicScenario] = None
    hinted: bool = False
    submitted: bool = False
    pre_results: dict = field(default_factory=dict)
    post_results: dict = field(default_factory=dict)
    raw_output: Optional[str] = None
    edits: list = field(default_factory=list)
    apply_ok: bool = False
    apply_error: str = ""
    apply_info: dict = field(default_factory=dict)
    reward_breakdown: dict = field(default_factory=dict)


DEFAULT_HINT_RATIO = float(os.environ.get("TRACE_SLP_HINT_RATIO", "0.40"))


class SemanticLogicPrecisionGame:
    supports_structured_messages = True
    max_steps = 1

    def __init__(self, *, hint_ratio: float = DEFAULT_HINT_RATIO) -> None:
        self.hint_ratio = float(hint_ratio)
        self._state = _SLPState()
        self._messages: list[dict] = []
        self.done = False
        self.rewards: dict[int, float] = {0: 0.0}
        self.current_player: int = 0
        self.invalid_player = None

    def observe(self, player_id: int) -> str:  # noqa: ARG002
        return "(structured-messages env)"

    def legal_actions(self) -> list[str]:
        return [] if self.done else ["<SEARCH/REPLACE in ```python fence>"]

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_BASE

    def get_tool_schemas(self) -> list[dict]:
        return []

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    @property
    def _conversation(self) -> list[dict]:
        return self._messages

    def reset(self, seed: int) -> None:
        self._state = _SLPState(seed=seed)
        self._messages = []
        self.done = False
        self.rewards = {0: 0.0}
        scenario = get_semantic_logic_scenario(seed)
        self._state.scenario = scenario
        rng = random.Random((seed * 2654435761) & 0xFFFFFFFF)
        self._state.hinted = (rng.random() < self.hint_ratio)
        wd = Path(tempfile.mkdtemp(
            prefix=f"trace_slp_{seed}_{uuid.uuid4().hex[:8]}_"
        ))
        self._state.workdir = wd
        # Write module + tests
        (wd / scenario.module_filename).write_text(scenario.module_code)
        (wd / "tests").mkdir(exist_ok=True)
        (wd / "tests" / "__init__.py").write_text("")
        tp = wd / scenario.test_filename
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(scenario.test_code)
        self._state.pre_results = _run_pytest(wd)

        # Build the user prompt
        user_body = USER_TEMPLATE.format(
            pr_description=scenario.pr_description,
            module_filename=scenario.module_filename,
            module_code=scenario.module_code,
        )
        hint_body = scenario.hint_body or scenario.naive_fix_description
        if self._state.hinted and hint_body:
            user_body += EXPERT_GUIDANCE_TEMPLATE.format(hint_body=hint_body)

        # NOTE: system message is added by inference.messages_for_game via
        # env.get_system_prompt(); do NOT include it in self._messages.
        self._messages.append({"role": "user", "content": user_body})

    def step(self, action) -> None:
        # action is the raw model output (text containing SEARCH/REPLACE)
        if isinstance(action, dict):
            text = action.get("content") or action.get("output") or ""
        else:
            text = str(action or "")
        self._state.raw_output = text
        self._messages.append({"role": "assistant", "content": text})

        edits = parse_search_replace(text)
        self._state.edits = edits
        if not edits:
            # no parseable edit → mark submit with 0 reward
            self._state.apply_error = "no_search_replace_blocks"
            self._finalize_reward()
            return

        apply_ok, info = apply_edits(
            self._state.workdir, edits,
            module_filename=self._state.scenario.module_filename,
        )
        self._state.apply_ok = apply_ok
        self._state.apply_info = info

        if not apply_ok:
            self._state.apply_error = info.get("error", "apply_failed")
            self._finalize_reward()
            return

        self._state.post_results = _run_pytest(self._state.workdir)
        self._finalize_reward()

    def _finalize_reward(self) -> None:
        sc = self._state.scenario
        pre = self._state.pre_results
        post = self._state.post_results

        target_passes = 1.0 if post.get(sc.target_test) == "PASSED" else 0.0
        pre_passing = {t for t, s in pre.items() if s == "PASSED"}
        if pre_passing:
            preserved = sum(
                1 for t in pre_passing if post.get(t) == "PASSED"
            )
            no_regression = preserved / len(pre_passing)
        else:
            no_regression = 1.0

        # minimal edit reward: fewer/simpler S/R blocks better
        n_edits = max(1, len(self._state.edits))
        minimal_edit = 1.0 / n_edits  # 1.0 for single edit, 0.5 for two, etc.

        # format fidelity: did we parse?
        format_fidelity = 1.0 if self._state.edits else 0.0

        if not self._state.apply_ok:
            reward = 0.0
        else:
            reward = (
                0.4 * target_passes
                + 0.4 * no_regression
                + 0.1 * minimal_edit
                + 0.1 * format_fidelity
            )

        self._state.reward_breakdown = {
            "target_passes": target_passes,
            "no_regression": no_regression,
            "minimal_edit": minimal_edit,
            "format_fidelity": format_fidelity,
            "apply_ok": self._state.apply_ok,
            "apply_error": self._state.apply_error,
        }
        self.rewards = {0: float(reward)}
        self.done = True
        self._state.submitted = True
        # cleanup workdir
        try:
            shutil.rmtree(self._state.workdir, ignore_errors=True)
        except Exception:
            pass


def _make_env(**kwargs):
    return SemanticLogicPrecisionGame(**kwargs)


def _extract_action(raw_output: str, _legal):
    return raw_output  # the game's step() does parsing internally


register_game(GameSpec(
    name="capability_semantic_logic_precision",
    make_env=_make_env,
    extract_action=_extract_action,
    action_space=[],  # free-form generation
    stop_sequences=None,
))
