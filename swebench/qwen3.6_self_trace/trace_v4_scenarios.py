"""Scenario library for capability `semantic-logic-precision`.

Loaded from scenarios_v4_all.json produced by phase3_env_synthesis_v4.py
(adversarial in-loop with mutation strategies). Each scenario has been
self-tested (target test FAILS pre-fix, oracle PASSES, no regression breaks)
and smoke-tested with N=10 base-model rollouts (rate<=0.5 verified).

Domains used are clean-room (no SWE-bench library leakage):
  - PID autotuner (Ziegler-Nichols formula)
  - smelter draft regulator (state reset)
  - particle tracker (kinetic energy formula)
  - chemical reactor (energy balance)
  - voice allocator (state mutation in priority dispatch)
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class SemanticLogicScenario:
    name: str
    domain: str
    module_filename: str
    module_code: str
    test_filename: str
    test_code: str
    target_test: str
    pr_description: str
    oracle_search: str
    oracle_replace: str
    naive_fix_description: str
    hint_body: str = ""  # optional curriculum hint


def _load_from_disk() -> List[SemanticLogicScenario]:
    candidates = [
        Path(os.environ.get("TRACE_V4_SCENARIOS_PATH", "")),
        Path("/workspace/trace_pipeline/env/scenarios_v4_all.json"),
        Path(__file__).parent / "scenarios_v4_all.json",
    ]
    for p in candidates:
        if p and p.is_file():
            data = json.loads(p.read_text())
            out = []
            for d in data:
                out.append(SemanticLogicScenario(
                    name=d["name"],
                    domain=d.get("__domain", d.get("domain", "unknown")),
                    module_filename=d["module_filename"],
                    module_code=d["module_code"],
                    test_filename=d["test_filename"],
                    test_code=d["test_code"],
                    target_test=d["target_test"],
                    pr_description=d["pr_description"],
                    oracle_search=d["oracle_search"],
                    oracle_replace=d["oracle_replace"],
                    naive_fix_description=d.get("naive_fix_description", ""),
                    hint_body=d.get("hint_body", ""),
                ))
            return out
    raise FileNotFoundError(
        "No scenarios_v4_all.json found. Set TRACE_V4_SCENARIOS_PATH or "
        "place at /workspace/trace_pipeline/env/scenarios_v4_all.json"
    )


ALL_SEMANTIC_LOGIC_SCENARIOS: List[SemanticLogicScenario] = _load_from_disk()


def get_semantic_logic_scenario(seed: int) -> SemanticLogicScenario:
    """Deterministic scenario pick from seed (uniform over the library)."""
    rng = random.Random((seed * 6364136223846793005 + 1442695040888963407)
                        & 0xFFFFFFFF)
    return ALL_SEMANTIC_LOGIC_SCENARIOS[rng.randrange(
        len(ALL_SEMANTIC_LOGIC_SCENARIOS)
    )]
