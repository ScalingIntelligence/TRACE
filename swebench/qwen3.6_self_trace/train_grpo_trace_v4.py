"""Wrapper that pre-imports the semantic-logic capability game.

Usage (mirrors `python -m train` / `train.train_grpo` CLI):
    torchrun --nproc_per_node=4 train_grpo_trace_v4.py \\
      --games "capability_semantic_logic_precision:1.0" \\
      --model Qwen/Qwen3.6-27B \\
      ... (other GRPO args)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TRAINER_DIR = "/workspace/games"
if os.path.isdir(_TRAINER_DIR) and _TRAINER_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_DIR)

# Self-register our game at import time.
try:
    import capability_semantic_logic_precision_game       # noqa: F401, E402
except ModuleNotFoundError as e:
    missing = getattr(e, "name", "")
    if missing == "capability_regression_safe_edit_game":
        raise RuntimeError(
            "Missing capability_regression_safe_edit_game.py. The SWE-bench "
            "semantic-logic env depends on the shared SEARCH/REPLACE parser, "
            "patch applier, pytest runner, and prompt templates from that file. "
            "Restore that helper before running Phase 5 training."
        ) from e
    raise
except FileNotFoundError as e:
    if "scenarios_v4_all.json" in str(e):
        raise RuntimeError(
            "Missing generated scenario artifact scenarios_v4_all.json. Run "
            "Phase 3/4 first and set TRACE_V4_SCENARIOS_PATH to the accepted "
            "scenario JSON, or place it at "
            "/workspace/trace_pipeline/env/scenarios_v4_all.json before "
            "starting Phase 5 training."
        ) from e
    raise
from game_registry import GAMES, list_game_names           # noqa: E402

_REQUIRED = {"capability_semantic_logic_precision"}
_missing = _REQUIRED - set(GAMES.keys())
if _missing:
    raise RuntimeError(
        f"Game failed to register: {sorted(_missing)}. "
        f"Registered: {sorted(list_game_names())}"
    )

print(f"[trainer] {len(GAMES)} games registered, target: "
      f"capability_semantic_logic_precision", flush=True)

# Hand off to the repo trainer.
try:
    from train.train_grpo import main  # noqa: E402
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Could not import train.train_grpo. Run this wrapper with the repository "
        "root on PYTHONPATH, for example: PYTHONPATH=/path/to/TRACE torchrun ..."
    ) from e

if __name__ == "__main__":
    sys.exit(main())
