#!/usr/bin/env python3
"""
Generate SkyRL-compatible training datasets for all skill games.

Usage:
    python skill_skyrl_dataset.py --output_dir ~/data/skill_games
    python skill_skyrl_dataset.py --games adversarial_policy multistep_task --num_train_seeds 500
"""

import argparse
import os
import sys

# Ensure games directory is importable
_GAMES_DIR = os.path.dirname(os.path.abspath(__file__))
if _GAMES_DIR not in sys.path:
    sys.path.insert(0, _GAMES_DIR)

import datasets

from game_registry import GAMES
from skill_game_env import SkillGameEnvConfig, build_initial_prompt, _USER_LLM_GAMES


def _make_cfg() -> SkillGameEnvConfig:
    """Build config from env vars (for user-LLM games)."""
    return SkillGameEnvConfig(
        user_llm_base_url=os.environ.get("USER_LLM_BASE_URL"),
        user_llm_model=os.environ.get("USER_LLM_MODEL"),
    )


def generate_split(
    game_names: list,
    seed_start: int,
    seed_end: int,
    output_path: str,
):
    """Generate one dataset split (train or val)."""
    cfg = _make_cfg()
    rows = []
    for game_name in game_names:
        # Check that user-LLM games have the required config
        if game_name in _USER_LLM_GAMES and not cfg.user_llm_base_url:
            print(f"  SKIP {game_name}: USER_LLM_BASE_URL not set")
            continue
        n_ok, n_fail = 0, 0
        for seed in range(seed_start, seed_end):
            try:
                prompt = build_initial_prompt(game_name, seed, cfg)
                rows.append(
                    {
                        "prompt": prompt,
                        "env_class": "skill_game",
                        "reward_spec": {"game_name": game_name},
                        "game_name": game_name,
                        "seed": seed,
                    }
                )
                n_ok += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  Warning: {game_name} seed={seed}: {e}")
                continue
        print(f"  {game_name}: {n_ok} ok, {n_fail} failed")

    ds = datasets.Dataset.from_list(rows)
    ds.to_parquet(output_path)
    print(f"Saved {len(rows)} episodes to {output_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate SkyRL datasets for skill games")
    parser.add_argument(
        "--games",
        nargs="+",
        default=None,
        help="Game names to include (default: all registered games)",
    )
    parser.add_argument("--num_train_seeds", type=int, default=1000)
    parser.add_argument("--num_val_seeds", type=int, default=100)
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/data/skill_games"),
    )
    args = parser.parse_args()

    available = list(GAMES.keys())
    game_names = args.games or available
    # Validate
    for g in game_names:
        if g not in GAMES:
            print(f"Error: unknown game '{g}'. Available: {available}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating datasets for: {game_names}")
    print(f"  Train seeds: 0..{args.num_train_seeds - 1}")
    print(f"  Val seeds:   {args.num_train_seeds}..{args.num_train_seeds + args.num_val_seeds - 1}")
    print()

    # Train split
    print("=== Train split ===")
    generate_split(
        game_names,
        seed_start=0,
        seed_end=args.num_train_seeds,
        output_path=os.path.join(args.output_dir, "train.parquet"),
    )
    print()

    # Validation split (non-overlapping seeds)
    print("=== Validation split ===")
    generate_split(
        game_names,
        seed_start=args.num_train_seeds,
        seed_end=args.num_train_seeds + args.num_val_seeds,
        output_path=os.path.join(args.output_dir, "validation.parquet"),
    )


if __name__ == "__main__":
    main()
