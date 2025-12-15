"""Converts Unsloth Python notebook to script."""

from __future__ import annotations

import argparse
import atexit
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _setup_matx_storage(matx_root: str) -> dict[str, Path]:
    """Force HF + datasets + general caches to go to /matx mount."""
    root = Path(matx_root).expanduser().resolve()
    paths = {
        "root": root,
        "hf_home": root / ".cache" / "huggingface",
        "hf_datasets": root / ".cache" / "huggingface" / "datasets",
        "hf_hub": root / ".cache" / "huggingface" / "hub",
        "torch_home": root / ".cache" / "torch",
        "xdg_cache": root / ".cache",
        "runs": root / "workplace" / "games",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # Hugging Face caches
    os.environ.setdefault("HF_HOME", str(paths["hf_home"]))
    os.environ.setdefault("HF_DATASETS_CACHE", str(paths["hf_datasets"]))
    os.environ.setdefault("HF_HUB_CACHE", str(paths["hf_hub"]))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(paths["hf_hub"]))

    # Misc caches
    os.environ.setdefault("TORCH_HOME", str(paths["torch_home"]))
    os.environ.setdefault("XDG_CACHE_HOME", str(paths["xdg_cache"]))

    # Logging/reporting
    wandb_dir = paths["runs"] / "wandb"
    wandb_cache = paths["root"] / ".cache" / "wandb"
    wandb_data = paths["root"] / ".cache" / "wandb_data"
    wandb_artifacts = paths["root"] / ".cache" / "wandb_artifacts"
    for p in (wandb_dir, wandb_cache, wandb_data, wandb_artifacts):
        p.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(wandb_artifacts))

    os.environ.setdefault("TRACKIO_DIR", str(paths["runs"] / "trackio"))

    (paths["runs"] / "outputs").mkdir(parents=True, exist_ok=True)
    return paths


def _ensure_openenv(openenv_dir: Path) -> None:
    """Clone OpenEnv onto /matx if it's not present."""
    openenv_dir = openenv_dir.resolve()
    if (openenv_dir / "src").exists():
        return
    openenv_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train.py] Cloning OpenEnv into: {openenv_dir}")
    subprocess.run(
        ["git", "clone", "https://github.com/meta-pytorch/OpenEnv.git", str(openenv_dir)],
        check=True,
    )


def _maybe_kill_process(proc) -> None:
    try:
        if proc is None:
            return
        proc.terminate()
    except Exception:
        pass


class _EMA:
    """EMA used as a role baseline (reward shaping)."""

    def __init__(self, gamma: float = 0.95):
        self.gamma = float(gamma)
        self._value = 0.0
        self._initialized = False

    def get(self) -> float:
        return float(self._value) if self._initialized else 0.0

    def update(self, x: float) -> None:
        x = float(x)
        if not self._initialized:
            self._value = x
            self._initialized = True
            return
        self._value = self.gamma * self._value + (1.0 - self.gamma) * x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matx-root",
        default="/matx/u/acshi",
        help="Root of the 3T /matx mount where caches + outputs should be stored.",
    )
    parser.add_argument(
        "--openenv-dir",
        default=None,
        help="Where to clone OpenEnv (default: <matx-root>/workplace/games/OpenEnv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save checkpoints (default: <matx-root>/workplace/games/outputs).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Max GRPO training steps (default: 600).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint save frequency (default: 100).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint in output_dir if available.",
    )

    # W&B logging
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default).",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    parser.set_defaults(use_wandb=True)
    parser.add_argument(
        "--wandb-entity",
        default="acshi-stanford-university",
        help="W&B entity/team to log to.",
    )
    parser.add_argument(
        "--wandb-project",
        default="games",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="Optional W&B run name (defaults to an informative timestamped name).",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode. Use offline/disabled if needed.",
    )
    args = parser.parse_args()

    # Must be set BEFORE importing HF / datasets to ensure caches go to /matx.
    paths = _setup_matx_storage(args.matx_root)
    runs_dir = paths["runs"]

    # Configure W&B BEFORE initializing the Trainer (HF/TRL callback reads env vars).
    if args.use_wandb:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_mode is not None:
            os.environ["WANDB_MODE"] = args.wandb_mode

        if args.wandb_name is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            args.wandb_name = f"qwen3-4b-kuhn-grpo-{ts}"

    openenv_dir = Path(args.openenv_dir) if args.openenv_dir else (runs_dir / "OpenEnv")
    output_dir = Path(args.output_dir) if args.output_dir else (runs_dir / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone OpenEnv
    _ensure_openenv(openenv_dir)

    # Add OpenEnv source to PYTHONPATH
    sys.path.insert(0, str(openenv_dir / "src"))
    working_directory = str(openenv_dir)

    # Notebook code starts here
    # IMPORTANT: Unsloth should be imported before transformers for full patching.
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from unsloth import execute_with_time_limit, launch_openenv

    import numpy as np
    from datasets import Dataset

    from envs.openspiel_env import OpenSpielEnv
    from envs.openspiel_env.models import OpenSpielAction

    # Load model & tokenizer (changed model only).
    max_seq_length = 768  # Can increase for longer RL output
    lora_rank = 4  # Larger rank = smarter, but slower
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        max_seq_length=max_seq_length,
        offload_embedding=True,  # Offload embeddings to save more VRAM
        cache_dir=str(paths["hf_hub"]),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # OpenSpiel Kuhn Poker environment
    global port
    global openenv_process
    port = 9000
    openenv_process = None

    server = "envs.openspiel_env.server.app:app"
    environment = {
        **os.environ,
        "PYTHONPATH": f"{working_directory}/src",
        "OPENSPIEL_GAME": "kuhn_poker",
        "OPENSPIEL_AGENT_PLAYER": "0",
        "OPENSPIEL_OPPONENT_POLICY": "random",
    }

    import functools

    launch_openenv = functools.partial(
        launch_openenv,
        working_directory=working_directory,
        server=server,
        environment=environment,
        openenv_class=OpenSpielEnv,
    )

    atexit.register(lambda: _maybe_kill_process(globals().get("openenv_process", None)))

    # Prompt (produce agent outputs actions)
    prompt = (
        "\n".join(
            [
                "You are playing 2-player Kuhn Poker (OpenSpiel: kuhn_poker) against a random opponent.",
                "You are the learning agent (Player 0).",
                "Kuhn Poker basics:",
                "- Each player antes 1 chip and receives one private card (J/Q/K).",
                "- There is a single betting round.",
                "- Actions (by stage):",
                "  * First decision (no bet yet): [Check] or [Bet]",
                "  * If you checked and the opponent then bets: you must respond with [Fold] or [Call]",
                "Your job: choose a valid action plan.",
                "\nOutput format (IMPORTANT):",
                "- You MAY write reasoning traces before the final two lines.",
                "- Then output exactly two lines:",
                "  FIRST: [Check] OR FIRST: [Bet]",
                "  RESPONSE: [Fold] OR RESPONSE: [Call]",
                "(The RESPONSE will be used only if a second decision is needed.)",
            ]
        )
    ).strip()

    # SPIRAL-style reward shaping
    # Invalid action => terminate & penalize agent
    INVALID_ACTION_PENALTY = -1.5
    ROLE_BASELINE = _EMA(gamma=0.95)

    ACTION_RE = re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE)

    def extract_kuhn_plan(text: str) -> tuple[str | None, str | None]:
        """Extract (first_action, response_action) from model text."""
        if not text:
            return None, None
        matches = ACTION_RE.findall(text)
        first = None
        response = None
        for m in matches:
            a = m.strip().lower()
            if first is None and a in ("check", "bet"):
                first = a
                continue
            if response is None and a in ("fold", "call"):
                response = a
        return first, response

    def _extract_agent_reward(reward_obj) -> float:
        """Best-effort extraction of the agent (player 0) reward from OpenEnv's reward payload."""
        if reward_obj is None:
            return 0.0
        try:
            if isinstance(reward_obj, dict):
                return float(reward_obj.get(0, 0.0))
            if isinstance(reward_obj, (list, tuple)):
                return float(reward_obj[0]) if len(reward_obj) > 0 else 0.0
            return float(reward_obj)
        except Exception:
            return 0.0

    def _kuhn_action_to_id(action_name: str) -> int:
        """Map semantic action name to OpenSpiel Kuhn Poker action id."""
        if action_name in ("check", "fold"):
            return 0
        if action_name in ("bet", "call"):
            return 1
        raise ValueError(f"Unknown action: {action_name}")

    @execute_with_time_limit(5)
    def run_kuhn_episode(first_action: str | None, response_action: str | None):
        """Play one Kuhn Poker episode using the model's parsed plan."""
        # If FIRST is missing or invalid, terminate immediately
        if first_action not in ("check", "bet"):
            return float(INVALID_ACTION_PENALTY), True, 0, "invalid_first"

        global port, openenv_process
        port, openenv_process = launch_openenv(port, openenv_process)
        result = openenv_process.reset()
        current_state = result.observation

        total_reward = 0.0
        agent_decisions = 0

        while not current_state.done:
            # Kuhn Poker agent (player0) can act at most twice
            if agent_decisions == 0:
                action_name = first_action
                allowed = ("check", "bet")
            else:
                # Second decision is only possible if the opponent bet after our check
                action_name = response_action
                allowed = ("fold", "call")

            if action_name not in allowed:
                return float(INVALID_ACTION_PENALTY), True, agent_decisions, "invalid_response"

            action_id = _kuhn_action_to_id(action_name)

            # Extra sanity check vs environment's legal_actions
            if hasattr(current_state, "legal_actions") and current_state.legal_actions is not None:
                if action_id not in current_state.legal_actions:
                    return float(INVALID_ACTION_PENALTY), True, agent_decisions, "illegal_action_id"

            action = OpenSpielAction(action_id=action_id, game_name="kuhn_poker")
            result = openenv_process.step(action)
            current_state = result.observation

            if result.reward is not None:
                total_reward += _extract_agent_reward(result.reward)

            agent_decisions += 1

            # The agent should not have more than 2 decisions in Kuhn
            if agent_decisions > 2 and not current_state.done:
                return float(INVALID_ACTION_PENALTY), True, agent_decisions, "too_many_turns"

        return float(total_reward), False, agent_decisions, "terminal"

    # Extra per-step RL stats
    global LATEST_RL_EXTRA_LOGS
    LATEST_RL_EXTRA_LOGS = {}

    def spiral_kuhn_reward(completions, **kwargs):
        """Reward function aligned with SPIRAL's Kuhn Poker setup."""
        global LATEST_RL_EXTRA_LOGS

        scores: list[float] = []
        raw_rewards: list[float] = []
        shaped_rewards: list[float] = []
        agent_turns: list[float] = []

        invalid_count = 0
        timeout_count = 0
        exception_count = 0
        win_count = 0
        loss_count = 0
        draw_count = 0

        baseline_start = ROLE_BASELINE.get()

        for completion in completions:
            response_text = completion[0]["content"]
            first_action, response_action = extract_kuhn_plan(response_text)

            try:
                raw_reward, invalid, n_turns, reason = run_kuhn_episode(first_action, response_action)
            except TimeoutError:
                raw_reward, invalid, n_turns, reason = float(INVALID_ACTION_PENALTY), True, 0, "timeout"
                timeout_count += 1
            except Exception:
                raw_reward, invalid, n_turns, reason = float(INVALID_ACTION_PENALTY), True, 0, "exception"
                exception_count += 1

            # SPIRAL-style role baseline shaping
            baseline_before = ROLE_BASELINE.get()
            ROLE_BASELINE.update(raw_reward)
            shaped = float(raw_reward - baseline_before)

            scores.append(shaped)
            raw_rewards.append(float(raw_reward))
            shaped_rewards.append(float(shaped))
            agent_turns.append(float(n_turns))

            if invalid:
                invalid_count += 1
            else:
                if raw_reward > 0:
                    win_count += 1
                elif raw_reward < 0:
                    loss_count += 1
                else:
                    draw_count += 1

        n = max(1, len(completions))
        baseline_end = ROLE_BASELINE.get()

        def _mean(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        def _std(xs: list[float]) -> float:
            return float(np.std(xs)) if xs else 0.0

        # Diagnostics
        LATEST_RL_EXTRA_LOGS = {
            "env/invalid_action_rate": invalid_count / n,
            "env/timeout_rate": timeout_count / n,
            "env/exception_rate": exception_count / n,
            "env/raw_reward_mean": _mean(raw_rewards),
            "env/raw_reward_std": _std(raw_rewards),
            "env/raw_reward_min": float(min(raw_rewards)) if raw_rewards else 0.0,
            "env/raw_reward_max": float(max(raw_rewards)) if raw_rewards else 0.0,
            "env/shaped_reward_mean": _mean(shaped_rewards),
            "env/agent_turns_mean": _mean(agent_turns),
            "env/agent_turns_min": float(min(agent_turns)) if agent_turns else 0.0,
            "env/agent_turns_max": float(max(agent_turns)) if agent_turns else 0.0,
            "env/win_rate": win_count / n,
            "env/loss_rate": loss_count / n,
            "env/draw_rate": draw_count / n,
            "env/role_baseline_start": float(baseline_start),
            "env/role_baseline_end": float(baseline_end),
        }

        return scores

    # Dataset
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": prompt.strip()}],
                "answer": 0,
                "reasoning_effort": "low",
            }
        ]
        * 1000
    )

    maximum_length = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.strip()}],
            add_generation_prompt=True,
        )
    )
    print(f"[train.py] maximum_length = {maximum_length}")

    # GRPO training setup
    max_prompt_length = maximum_length + 1  # + 1 just in case!
    max_completion_length = max_seq_length - max_prompt_length

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=2e-4,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        report_to=(
            ["wandb", "trackio"] if args.use_wandb else "trackio"
        ),
        run_name=(args.wandb_name if args.use_wandb else None),
        output_dir=str(output_dir),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            spiral_kuhn_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    from transformers import TrainerCallback

    class _ExtraRLMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            global LATEST_RL_EXTRA_LOGS
            if isinstance(LATEST_RL_EXTRA_LOGS, dict) and LATEST_RL_EXTRA_LOGS:
                logs.update(LATEST_RL_EXTRA_LOGS)

    trainer.add_callback(_ExtraRLMetricsCallback())

    # Train
    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
