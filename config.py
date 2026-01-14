#!/usr/bin/env python3
"""
Configuration, constants, and environment setup for PPO training.
"""
import argparse
import os
import torch
from pathlib import Path


# =========================
# Parse command-line arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="PPO self-play for Kuhn Poker")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for cache, wandb, and outputs. Defaults to /matx/u/{USER} if not specified."
    )
    parser.add_argument(
        "--use_constrained_decoding",
        type=bool,
        default=False,
        help="If True, use constrained decoding to force action-only outputs. If False, use normal generation. Default: True"
    )

    parser.add_argument(
        "--game",
        type=str,
        default="kuhn_poker",
        choices=["kuhn_poker", "liars_dice"],
        help="Game to train on: kuhn_poker or liars_dice"
    )
    return parser.parse_args()


# =========================
# Setup paths and environment
# =========================
def setup_environment(args):
    """Setup cache directories, environment variables, and device settings."""
    
    # Determine root directory
    if args.root is not None:
        root = Path(args.root).resolve()
    else:
        root = Path(f"/matx/u/{os.getenv('USER')}").resolve()
    
    # Setup HuggingFace cache directories
    hf_home = root / ".cache" / "huggingface"
    hf_hub = hf_home / "hub"
    hf_datasets = hf_home / "datasets"
    
    for p in (hf_home, hf_hub, hf_datasets):
        p.mkdir(parents=True, exist_ok=True)
    
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets))
    
    # Setup wandb directory
    wandb_dir = root / "workplace" / "games" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_PROJECT", "games")
    
    # Setup output directory
    output_dir_path = root / "workplace" / "games" / "outputs"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Rollout log path
    rollout_log_path = Path(__file__).resolve().parent / "selfplay_rollouts_ppo.jsonl"
    
    # Optional: allocator fragmentation guard
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    
    return {
        "root": root,
        "hf_home": hf_home,
        "hf_hub": hf_hub,
        "hf_datasets": hf_datasets,
        "wandb_dir": wandb_dir,
        "output_dir_path": output_dir_path,
        "rollout_log_path": rollout_log_path,
        "device": device,
    }


# =========================
# Constants
# =========================
class Config:
    """Training and game configuration constants."""
    
    # Model settings
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    MAX_SEQ_LENGTH = 768
    LORA_RANK = 4
    LORA_ALPHA = 8
    
    # Game settings
    NUM_ROUNDS = 5
    NUM_DICE = 5
    GAMES_PER_ITER = 64
    
    # PPO hyperparameters
    PPO_EPOCHS = 1
    MINI_BATCH_SIZE = 32
    STATS_CHUNK_SIZE = 2
    LR = 1e-7
    CLIP_EPS = 0.2
    VF_COEF = 0.5
    
    # Generation settings
    MAX_GEN_TOKENS = 8
    MAX_GEN_TOKENS_LIARS_DICE = 16
    TEMPERATURE = 0.7
    MAX_TOKENS_MATH_EVAL = 7000
    
    # Checkpointing and evaluation
    SAVE_EVERY_ITERS = 5
    EVAL_EVERY_ITERS = 5
    EVAL_GAMES = 25
    MATH_EVAL_SAMPLES = 50
    MATH_EVAL_EVERY_ITERS = 10000

    USE_ROLE_BASELINE = True
    ROLE_BASELINE_EMA_GAMMA = 0.95
    
    # Math evaluation datasets
    MATH_EVAL_DATASETS = ["math", "amc", "aime"]
    
    # Prompt templates
    SYSTEM_PROMPT_KUHN = (
        "You are playing Kuhn Poker.\n"
        "Respond with EXACTLY ONE action token and NOTHING ELSE.\n"
        "Valid outputs: [check] or [bet] or [call] or [fold].\n"
        "Do not add any whitespace, punctuation, explanation, or extra text.\n"
    )

    SYSTEM_PROMPT_LIARS_DICE = (
        "You are playing Liar's Dice.\n"
        "Respond with EXACTLY ONE action and NOTHING ELSE.\n"
        "Valid outputs: [bid: quantity, face] or [call]\n"
        "Examples: [bid: 3, 4] or [call]\n"
        "Do not add any whitespace, punctuation, explanation, or extra text.\n"
    )
    
    MATH_SYSTEM_PROMPT = (
        "You are a helpful math assistant. Solve the following problem step by step. "
        "Put your final answer in \\boxed{}."
    )


# =========================
# Action constants
# =========================
ACTION_STRS_KUHN = ["[check]", "[bet]", "[call]", "[fold]"]

CURRENT_GAME = "kuhn_poker"
ACTION_STRS = ACTION_STRS_KUHN

def get_system_prompt(game: str) -> str:
    """Get the system prompt for a game."""
    if game == "liars_dice":
        return Config.SYSTEM_PROMPT_LIARS_DICE
    return Config.SYSTEM_PROMPT_KUHN


def get_max_gen_tokens(game: str) -> int:
    """Get max generation tokens for a game."""
    if game == "liars_dice":
        return Config.MAX_GEN_TOKENS_LIARS_DICE
    return Config.MAX_GEN_TOKENS

    
def autocast_ctx(device):
    """Return appropriate autocast context for the device."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.no_grad()

