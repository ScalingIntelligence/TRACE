#!/usr/bin/env python3
import argparse
import os
import torch
from pathlib import Path


# =========================
# Parse command-line arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="PPO/GRPO training for game environments")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for cache, wandb, and outputs. Defaults to /matx/u/{USER} if not specified."
    )
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="Game to train on (must be registered in game_registry).",
    )

    parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint directory to resume from (e.g., /path/to/ppo_ckpt_iter_210)"
    )

    parser.add_argument(
        "--rollout_log",
        type=str,
        default="selfplay_rollouts_ppo.jsonl",
        help="Filename for rollout logs"
    )

    parser.add_argument(
        "--model",
        type=str,
        # default="Qwen/Qwen3-4B-Instruct-2507",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="HuggingFace model name to use for training and inference"
    )

    return parser.parse_args()




# =========================
# Setup paths and environment
# =========================
def setup_environment(args):
    
    # Determine root directory
    if getattr(args, 'root', None) is not None:
        root = Path(args.root).resolve()
    else:
        root = Path.home()
    
    # Setup HuggingFace cache directories — respect HF_HOME env var if set,
    # otherwise use local storage (not NFS) to avoid slow model loading
    local_home = Path.home()
    hf_home = Path(os.environ.get("HF_HOME", str(local_home / ".cache" / "huggingface")))
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
    
    
    # Setup output directory — use HF_HOME for checkpoints so they land on
    # the path the user configured (e.g. /workspace/.cache/huggingface)
    game_name = getattr(args, 'game', None) or 'default'
    output_dir_path = hf_home / game_name
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Rollout log path
    if hasattr(args, 'rollout_log'):
        gameplay_dir = Path(__file__).resolve().parent.parent / "gameplay_rollouts"
        gameplay_dir.mkdir(exist_ok=True)
        rollout_log_path = gameplay_dir / args.rollout_log
    else:
        rollout_log_path = None
    
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
    
    # Model settings
    # MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    MAX_SEQ_LENGTH = 16000
    LORA_RANK = 16
    LORA_ALPHA = 8
    
    # Game settings
    GAMES_PER_ITER = 256
    # PPO hyperparameters
    PPO_EPOCHS = 1
    MINI_BATCH_SIZE = 4
    STATS_CHUNK_SIZE = 1
    LR = 1e-6
    CLIP_EPS = 0.2
    VF_COEF = 0.5

    # Generation settings
    ENABLE_THINKING = True
    MAX_GEN_TOKENS = 2048 if ENABLE_THINKING else 8
    TEMPERATURE = 0.7

    # Checkpointing
    SAVE_EVERY_ITERS = 5

    USE_ROLE_BASELINE = True
    ROLE_BASELINE_EMA_GAMMA = 0.95


def autocast_ctx(device):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.no_grad()
