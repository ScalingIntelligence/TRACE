"""Converts Unsloth Python notebook to script."""

from __future__ import annotations

import argparse
import atexit
import os
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
        help="Max GRPO training steps (default matches notebook: 600).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint save frequency (default matches notebook: 100).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint in output_dir if available.",
    )

    # W&B logging (requested)
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

    paths = _setup_matx_storage(args.matx_root)
    runs_dir = paths["runs"]

    # Configure wandb
    if args.use_wandb:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_mode is not None:
            os.environ["WANDB_MODE"] = args.wandb_mode

        # Set run name
        if args.wandb_name is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            args.wandb_name = f"qwen3-4b-2048-grpo-{ts}"

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
    from unsloth import check_python_modules, create_locked_down_function
    from unsloth import execute_with_time_limit, is_port_open, launch_openenv

    import torch
    import numpy as np
    from datasets import Dataset
    from transformers import TextStreamer

    from envs.openspiel_env import OpenSpielEnv
    from envs.openspiel_env.models import OpenSpielAction, OpenSpielObservation

    # Load model & tokenizer (changed model only).
    max_seq_length = 768  # Can increase for longer RL output
    lora_rank = 4  # Larger rank = smarter, but slower
    model_name = "unsloth/gpt-oss-20b"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        max_seq_length=max_seq_length,
        offload_embedding=True,  # Offload embeddings to save more VRAM
        # Ensure weights download into /matx
        cache_dir=str(paths["hf_hub"]),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )

    # 2048 environment with OpenEnv
    global port
    global openenv_process
    port = 9000
    openenv_process = None
    server = "envs.openspiel_env.server.app:app"
    environment = {
        **os.environ,
        "PYTHONPATH": f"{working_directory}/src",
        "OPENSPIEL_GAME": "2048",
        "OPENSPIEL_AGENT_PLAYER": "0",
        "OPENSPIEL_OPPONENT_POLICY": "random",
    }

    # Augment Unsloth's OpenEnv creation function
    import functools

    launch_openenv = functools.partial(
        launch_openenv,
        working_directory=working_directory,
        server=server,
        environment=environment,
        openenv_class=OpenSpielEnv,
    )

    # Ensure OpenEnv process is cleaned up
    atexit.register(lambda: _maybe_kill_process(globals().get("openenv_process", None)))

    def convert_to_board(current_state: OpenSpielObservation):
        n = len(current_state.info_state)
        size = int(np.sqrt(n))
        board = np.array_split(np.array(current_state.info_state, dtype=int), size)
        board = [x.tolist() for x in board]
        return board, size

    # 2048 Game Renderer
    def render_board(obs, colors: bool = True, border: bool = True, dot_for_zero: bool = True) -> str:
        """
        Pretty-print the board with colors that scale from 0 up to self.target.
        Uses ANSI 256-color codes (works in most terminals). Set colors=False to disable.
        """
        import math

        b, size = convert_to_board(obs)
        mx = max((max(row) for row in b), default=0)
        cell_w = max(3, len(str(mx)))

        RESET = "\x1b[0m"

        # A smooth-ish gradient from cool → warm
        # (blue/cyan/green → yellow/orange/red). Tweak or expand as you like.
        GRAD = [
            33,
            39,
            45,
            51,
            50,
            49,
            48,
            47,
            46,
            82,
            118,
            154,
            190,
            226,
            220,
            214,
            208,
            202,
            196,
        ]
        ZERO_FG = 239  # dim gray

        def color_code(v: int) -> str:
            if not colors:
                return ""
            if v == 0:
                return f"\x1b[38;5;{ZERO_FG}m"
            # Normalize by exponent relative to target: r in [0,1]
            t = max(2, 2048)  # safety; avoid log2(1)
            # Guard: if v is not a power of two or is <1, handle gracefully
            try:
                r = max(0.0, min(1.0, math.log2(v) / math.log2(t)))
            except ValueError:
                r = 0.0
            idx = int(round(r * (len(GRAD) - 1)))
            return f"\x1b[38;5;{GRAD[idx]}m"

        def fmt(v: int) -> str:
            s = "." if (v == 0 and dot_for_zero) else str(v)
            s = s.rjust(cell_w)
            return color_code(v) + s + (RESET if colors else "")

        def hline(left: str, mid: str, right: str) -> str:
            return left + mid.join("─" * cell_w for _ in range(size)) + right

        rows = []
        if border:
            rows.append(hline("┌", "┬", "┐"))
        for r in range(size):
            content = "│".join(fmt(v) for v in b[r])
            rows.append(("│" + content + "│") if border else content)
            if border:
                rows.append(
                    hline(
                        "└" if r == size - 1 else "├",
                        "┴" if r == size - 1 else "┼",
                        "┘" if r == size - 1 else "┤",
                    )
                )
        return "\n".join(rows)

    # RL Environment Setup
    from typing import Callable
    import itertools

    def _execute_strategy(strategy, current_state: OpenSpielObservation):
        assert callable(strategy)

        steps = 0
        total_reward = 0
        while not current_state.done:
            board, size = convert_to_board(current_state)
            action = strategy(board)
            try:
                action = int(action)
            except Exception:
                return steps, False
            steps += 1
            if type(action) is not int or action not in current_state.legal_actions:
                return steps, max(itertools.chain.from_iterable(board)) == 2048

            global port, openenv_process
            port, openenv_process = launch_openenv(port, openenv_process)
            action = OpenSpielAction(action_id=action, game_name="2048")
            result = openenv_process.step(action)
            current_state = result.observation
            if result.reward is not None:
                total_reward += result.reward
        return steps, max(itertools.chain.from_iterable(board)) == 2048

    @execute_with_time_limit(5)
    def execute_strategy(strategy: Callable, current_state: OpenSpielObservation):
        return _execute_strategy(strategy, current_state)

    # Data & RL task setup
    prompt = (
        "\n".join(
            [
                "Create a new short 2048 strategy using only native Python code.",
                "You are given a list of list of numbers for the current board state.",
                'Output one action for "0", "1", "2", "3" on what is the optimal next step.',
                "Output your new short function in backticks using the format below:",
                "```python",
                "def strategy(board):",
                '    return "0" # Example',
                "```",
                "All helper functions should be inside def strategy. Only output the short function `strategy`.",
            ]
        )
    ).strip()

    # Reward functions
    def extract_function(text: str):
        if text.count("```") >= 2:
            first = text.find("```") + 3
            second = text.find("```", first)
            fx = text[first:second].strip()
            fx = fx.removeprefix("python\n")
            fx = fx[fx.find("def") :]
            if fx.startswith("def strategy(board):"):
                return fx
        return None

    def function_works(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                ok, info = check_python_modules(function)
            if function is None or "error" in info:
                score = -2.0
            else:
                try:
                    _new_strategy = create_locked_down_function(function)
                    score = 1.0
                except Exception:
                    score = -0.5
            scores.append(score)
        return scores

    def no_cheating(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                ok, info = check_python_modules(function)
                scores.append(1.0 if ok else -20.0)  # Penalize heavily!
            else:
                scores.append(-1.0)  # Failed creating function
        return scores

    global PRINTER
    PRINTER = 0

    # Extra per-step RL stats (pure logging; does NOT change rewards or training).
    # Populated inside reward functions and then surfaced via a Trainer callback.
    global LATEST_RL_EXTRA_LOGS
    LATEST_RL_EXTRA_LOGS = {}

    def strategy_succeeds(completions, **kwargs):
        global PRINTER
        global LATEST_RL_EXTRA_LOGS
        scores = []

        # Collect a few environment-side stats for logging.
        n = max(1, len(completions))
        steps_list = []
        success_count = 0
        timeout_count = 0
        exception_count = 0
        valid_function_count = 0

        for completion in completions:
            printed = False
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
            if PRINTER % 5 == 0:
                printed = True
                print(function)
            PRINTER += 1
            if function is not None:
                ok, info = check_python_modules(function)
            if function is None or "error" in info:
                scores.append(0)
                continue
            try:
                new_strategy = create_locked_down_function(function)
                valid_function_count += 1
            except Exception:
                scores.append(0)
                continue
            try:
                # Reset OpenEnv to an initial state!
                global port, openenv_process
                port, openenv_process = launch_openenv(port, openenv_process)
                result = openenv_process.reset()
                current_state = result.observation
                steps, if_done = execute_strategy(new_strategy, current_state)
                steps_list.append(float(steps))
                success_count += int(bool(if_done))
                print(f"Steps = {steps} If Done = {if_done}")
                if printed is False:
                    print(function)
                print(render_board(current_state))
                if if_done:
                    scores.append(20.0)  # Success - massively reward!
                else:
                    scores.append(2.0)  # Failed but function works!
            except TimeoutError:
                print("Timeout")
                timeout_count += 1
                scores.append(-1.0)  # Failed with timeout
            except Exception as e:
                print(f"Exception = {str(e)}")
                exception_count += 1
                scores.append(-3.0)  # Failed

        # These are diagnostics
        # They help interpret reward movements and environment reliability.
        LATEST_RL_EXTRA_LOGS = {
            "env/strategy_valid_function_rate": valid_function_count / n,
            "env/strategy_success_rate": success_count / n,
            "env/strategy_timeout_rate": timeout_count / n,
            "env/strategy_exception_rate": exception_count / n,
            "env/strategy_steps_mean": float(np.mean(steps_list)) if steps_list else 0.0,
            "env/strategy_steps_min": float(np.min(steps_list)) if steps_list else 0.0,
            "env/strategy_steps_max": float(np.max(steps_list)) if steps_list else 0.0,
        }
        return scores

    # Dataset (same as notebook)
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
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=2,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        report_to=(
            ["wandb", "trackio"]
            if args.use_wandb
            else "trackio"
        ),  # TrackIO (notebook default) + W&B (requested)
        run_name=(args.wandb_name if args.use_wandb else None),
        output_dir=str(output_dir),
        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            function_works,
            no_cheating,
            strategy_succeeds,
        ],
        args=training_args,
        train_dataset=dataset,
        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )

    # Set extra RL diagnostics
    from transformers import TrainerCallback

    class _ExtraRLMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            global LATEST_RL_EXTRA_LOGS
            if isinstance(LATEST_RL_EXTRA_LOGS, dict) and LATEST_RL_EXTRA_LOGS:
                logs.update(LATEST_RL_EXTRA_LOGS)

    trainer.add_callback(_ExtraRLMetricsCallback())

    # Train (same call as notebook; resume is optional via flag)
    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # (Optional) Merge and save / push (kept as notebook; disabled)
    # Merge and push to hub in mxfp4 4bit format
    if False:
        model.save_pretrained_merged(str(output_dir / "finetuned_model"), tokenizer, save_method="mxfp4")
    if False:
        model.push_to_hub_merged(
            "repo_id/repo_name",
            tokenizer,
            token="hf...",
            save_method="mxfp4",
        )

    # Merge and push to hub in 16bit
    if False:
        model.save_pretrained_merged(
            str(output_dir / "finetuned_model"), tokenizer, save_method="merged_16bit"
        )
    if False:  # Pushing to HF Hub
        model.push_to_hub_merged(
            "hf/gpt-oss-finetune",
            tokenizer,
            save_method="merged_16bit",
            token="",
        )


if __name__ == "__main__":
    main()
