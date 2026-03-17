#!/usr/bin/env python3
"""
SkyRL training entrypoint for skill games.

Registers the SkillGameEnv with SkyRL-Gym, then launches standard
GRPO/PPO training.

Usage (local):
    python skill_skyrl_train.py \\
        data.train_data=[~/data/skill_games/train.parquet] \\
        data.val_data=[~/data/skill_games/validation.parquet] \\
        environment.env_class=skill_game \\
        trainer.policy.model.path=Qwen/Qwen2.5-7B-Instruct \\
        generator.max_turns=30 \\
        generator.n_samples_per_prompt=5 \\
        generator.sampling_params.max_generate_length=1024 \\
        trainer.train_batch_size=64 \\
        trainer.policy_mini_batch_size=16
"""

import os
import sys

# Ensure games directory is on the path so that game_registry, game modules,
# and skill_game_env are all importable by the Ray workers.
_GAMES_DIR = os.path.dirname(os.path.abspath(__file__))
if _GAMES_DIR not in sys.path:
    sys.path.insert(0, _GAMES_DIR)

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    """Ray task that registers skill game envs and runs training."""
    # Ensure games directory is importable inside the Ray worker
    games_dir = os.path.dirname(os.path.abspath(__file__))
    if games_dir not in sys.path:
        sys.path.insert(0, games_dir)

    # Register the unified SkillGameEnv
    register(
        id="skill_game",
        entry_point="skill_game_env:SkillGameEnv",
    )

    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
