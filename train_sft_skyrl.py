#!/usr/bin/env python3
"""
SFT trainer using SkyRL's distributed training backend on trajectories
collected by collect_rollouts.py.

Uses SkyRL's WorkerDispatch + FSDP policy workers with loss_fn="cross_entropy"
for supervised fine-tuning. Loads tau2-bench format JSON files via SFTBuffer.

Supports:
  - Multi-GPU FSDP (via SkyRL Ray workers)
  - LoRA (via SkyRL's PEFT integration)
  - Full-parameter fine-tuning
  - Gradient checkpointing
  - HuggingFace model export
  - Checkpoint save/resume

Prerequisites:
    # Install SkyRL with FSDP dependencies (from repo root):
    pip install -e SkyRL[fsdp]
    # Or run via uv from the SkyRL directory:
    cd SkyRL && uv run --isolated --extra fsdp python ../train_sft_skyrl.py ...

Usage:
    # Full-parameter SFT on 1 GPU:
    python train_sft_skyrl.py \
        --sft-data game_rollouts/tau_tool_calling.json \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507

    # LoRA SFT on 4 GPUs:
    python train_sft_skyrl.py \
        --sft-data game_rollouts/file1.json,game_rollouts/file2.json \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --num-gpus 4 \
        --lora-rank 16 --lora-alpha 16

    # With custom hyperparameters:
    python train_sft_skyrl.py \
        --sft-data game_rollouts/tau_tool_calling.json \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --lr 1e-5 --num-epochs 3 --micro-batch-size 2
"""

import argparse
import json
import math
import os
import random
import sys
import time

# Ensure SkyRL is importable even if not pip-installed
_SKYRL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SkyRL")
if os.path.isdir(_SKYRL_DIR) and _SKYRL_DIR not in sys.path:
    sys.path.insert(0, _SKYRL_DIR)

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

import ray
import torch
from datetime import datetime
from pathlib import Path
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer

from ray.util.placement_group import placement_group

from skyrl.train.config import SkyRLTrainConfig
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl.train.utils.utils import initialize_ray, validate_cfg, ResolvedPlacementGroup
from skyrl.train.utils import get_ray_pg_ready_with_timeout

from config import Config
from ppo import build_prompt_plus_action
from sft_buffer import SFTBuffer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT trainer using SkyRL backend on collect_rollouts.py data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data --
    parser.add_argument("--sft-data", type=str, required=True,
        help="Comma-separated paths to tau2-bench simulation JSON files "
             "(output of collect_rollouts.py)")

    # -- Model --
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME,
        help="HuggingFace model name or path")

    # -- LoRA --
    parser.add_argument("--lora-rank", type=int, default=0,
        help="LoRA rank (0 = full-parameter fine-tuning)")
    parser.add_argument("--lora-alpha", type=int, default=16,
        help="LoRA alpha scaling factor")

    # -- Hardware --
    parser.add_argument("--num-gpus", type=int, default=1,
        help="Number of GPUs for FSDP training")
    parser.add_argument("--num-nodes", type=int, default=1,
        help="Number of nodes for multi-node training")

    # -- Optimization --
    parser.add_argument("--lr", type=float, default=1e-5,
        help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3,
        help="Number of full passes over the SFT dataset")
    parser.add_argument("--micro-batch-size", type=int, default=2,
        help="Micro-batch size per GPU (for gradient accumulation)")
    parser.add_argument("--batch-size", type=int, default=16,
        help="Global batch size (controls gradient accumulation)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")
    parser.add_argument("--weight-decay", type=float, default=0.01,
        help="Weight decay for AdamW")

    # -- Data processing --
    parser.add_argument("--compact-tools", action="store_true", default=False,
        help="Use compressed tool schemas (strips descriptions)")
    parser.add_argument("--max-seq-length", type=int, default=None,
        help="Max sequence length (default: Config.MAX_SEQ_LENGTH)")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Limit to first N samples for debugging")

    # -- Checkpointing --
    parser.add_argument("--save-every", type=int, default=1,
        help="Save checkpoint every N epochs")
    parser.add_argument("--output-dir", type=str, default=None,
        help="Output directory for checkpoints")

    # -- Misc --
    parser.add_argument("--shuffle-seed", type=int, default=42,
        help="Random seed for data shuffling")
    parser.add_argument("--strategy", type=str, default="fsdp2",
        choices=["fsdp", "fsdp2"],
        help="Distributed training strategy")
    parser.add_argument("--logger", type=str, default="console",
        choices=["console", "wandb"],
        help="Logging backend")
    parser.add_argument("--wandb-project", type=str, default="games",
        help="wandb project name (only used with --logger wandb)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Batch construction from SFTBuffer data
# ---------------------------------------------------------------------------

def collate_sft_batch(
    seqs: List[torch.Tensor],
    prompt_lens: List[int],
    action_lens: List[int],
    tokenizer,
) -> TrainingInputBatch:
    """Collate pre-tokenized sequences into a TrainingInputBatch.

    Follows the same format as SkyRL's SFT example:
    - sequences: [B, max_len] left-padded token IDs
    - attention_mask: [B, max_len] 1 for real tokens, 0 for padding
    - loss_mask: [B, max_num_actions] 1 for response tokens to compute loss on

    The loss_mask and metadata["response_length"] tell the SkyRL worker
    which tokens to compute cross-entropy loss over.
    """
    max_len = max(s.shape[0] for s in seqs)
    max_num_actions = max(action_lens)
    pad_id = tokenizer.pad_token_id

    batch_seqs = []
    batch_attn = []
    batch_loss_mask = []

    for seq, pl, al in zip(seqs, prompt_lens, action_lens):
        seq_len = seq.shape[0]
        pad_len = max_len - seq_len

        # Left-pad sequences (SkyRL convention)
        padded_seq = torch.cat([
            torch.full((pad_len,), pad_id, dtype=torch.long),
            seq,
        ])
        batch_seqs.append(padded_seq)

        # Attention mask: 0 for pad, 1 for real tokens
        attn = torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(seq_len, dtype=torch.long),
        ])
        batch_attn.append(attn)

        # Loss mask: only 1 for the response/action tokens
        # Dimension: [max_num_actions] — right-aligned per-example
        action_pad = max_num_actions - al
        loss_mask = torch.cat([
            torch.zeros(action_pad, dtype=torch.long),
            torch.ones(al, dtype=torch.long),
        ])
        batch_loss_mask.append(loss_mask)

    batch = TrainingInputBatch({
        "sequences": torch.stack(batch_seqs),
        "attention_mask": torch.stack(batch_attn),
        "loss_mask": torch.stack(batch_loss_mask),
    })
    batch.metadata = {"response_length": max_num_actions}
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    max_seq_len = args.max_seq_length or Config.MAX_SEQ_LENGTH

    # ---- Output directory ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(os.path.expanduser("~/exports/sft_skyrl"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build SkyRL config ----
    cfg = SkyRLTrainConfig()

    # Model
    cfg.trainer.policy.model.path = args.model

    # LoRA
    if args.lora_rank > 0:
        cfg.trainer.policy.model.lora.rank = args.lora_rank
        cfg.trainer.policy.model.lora.alpha = args.lora_alpha

    # Hardware placement
    cfg.trainer.placement.policy_num_gpus_per_node = args.num_gpus
    cfg.trainer.placement.policy_num_nodes = args.num_nodes
    cfg.generator.inference_engine.tensor_parallel_size = args.num_gpus

    # Training hyperparameters
    cfg.trainer.policy.optimizer_config.lr = args.lr
    cfg.trainer.policy.optimizer_config.weight_decay = args.weight_decay
    cfg.trainer.policy.optimizer_config.max_grad_norm = args.max_grad_norm
    cfg.trainer.micro_train_batch_size_per_gpu = args.micro_batch_size
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.strategy = args.strategy
    cfg.trainer.logger = args.logger
    cfg.trainer.flash_attn = True

    validate_cfg(cfg)

    # ---- Initialize Ray ----
    initialize_ray(cfg)

    # ---- Load tokenizer ----
    print(f"[SFT] Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load SFT data via SFTBuffer ----
    sft_paths = [p.strip() for p in args.sft_data.split(",") if p.strip()]
    print(f"[SFT] Loading data from {len(sft_paths)} files...")
    sft_buffer = SFTBuffer(
        sft_paths, tokenizer,
        compact_tools=args.compact_tools,
        max_seq_len=max_seq_len,
    )
    total_samples = len(sft_buffer)
    print(f"[SFT] Loaded {total_samples} samples")

    if total_samples == 0:
        print("[SFT] ERROR: No samples loaded. Check that JSON files contain "
              "reward=1.0 simulations.")
        ray.shutdown()
        return

    # Truncate if requested
    if args.max_samples is not None and args.max_samples < total_samples:
        sft_buffer._seqs_cpu = sft_buffer._seqs_cpu[:args.max_samples]
        sft_buffer._prompt_lens = sft_buffer._prompt_lens[:args.max_samples]
        sft_buffer._action_lens = sft_buffer._action_lens[:args.max_samples]
        total_samples = len(sft_buffer)
        print(f"[SFT] Truncated to {total_samples} samples (--max-samples)")

    seqs = sft_buffer._seqs_cpu
    prompt_lens = sft_buffer._prompt_lens
    action_lens = sft_buffer._action_lens

    # ---- Compute training schedule ----
    global_batch_size = args.batch_size
    steps_per_epoch = math.ceil(total_samples / global_batch_size)
    total_steps = steps_per_epoch * args.num_epochs

    # ---- Initialize policy worker ----
    print(f"[SFT] Initializing policy worker ({args.num_gpus} GPUs, "
          f"strategy={args.strategy})...")
    num_gpus = args.num_gpus
    raw_pg = placement_group(
        [{"GPU": num_gpus, "CPU": num_gpus}],
        strategy="PACK",
    )
    get_ray_pg_ready_with_timeout(raw_pg, timeout=120)
    pg = ResolvedPlacementGroup(raw_pg)

    actor_group = PPORayActorGroup(
        cfg.trainer,
        num_nodes=args.num_nodes,
        num_gpus_per_node=num_gpus,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=1.0,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
    )
    ray.get(actor_group.async_init_model(
        cfg.trainer.policy.model.path,
        num_training_steps=total_steps,
    ))

    dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)

    # ---- Print config ----
    seq_lens = [s.shape[0] for s in seqs]
    print(f"\n{'=' * 60}")
    print(f"  SFT Training (SkyRL backend)")
    print(f"{'=' * 60}")
    print(f"  Model:              {args.model}")
    print(f"  LoRA rank:          {args.lora_rank}")
    print(f"  LoRA alpha:         {args.lora_alpha}")
    print(f"  Strategy:           {args.strategy}")
    print(f"  Num GPUs:           {args.num_gpus}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Weight decay:       {args.weight_decay}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Total samples:      {total_samples}")
    print(f"  Global batch size:  {global_batch_size}")
    print(f"  Micro batch/GPU:    {args.micro_batch_size}")
    print(f"  Steps/epoch:        {steps_per_epoch}")
    print(f"  Total steps:        {total_steps}")
    print(f"  Max seq length:     {max_seq_len}")
    print(f"  Max grad norm:      {args.max_grad_norm}")
    print(f"  Compact tools:      {args.compact_tools}")
    print(f"  Seq length stats:   min={min(seq_lens)} max={max(seq_lens)} "
          f"mean={sum(seq_lens)/len(seq_lens):.0f}")
    print(f"  Output dir:         {output_dir}")
    print(f"{'=' * 60}\n")

    # ====================================================================
    # Training loop
    # ====================================================================

    for epoch in range(args.num_epochs):
        t_epoch_start = time.time()

        # Shuffle sample order
        rng = random.Random(args.shuffle_seed + epoch)
        sample_order = list(range(total_samples))
        rng.shuffle(sample_order)

        epoch_seqs = [seqs[i] for i in sample_order]
        epoch_pls = [prompt_lens[i] for i in sample_order]
        epoch_als = [action_lens[i] for i in sample_order]

        epoch_loss_acc = 0.0
        epoch_tokens_acc = 0
        num_batches_done = 0

        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}"):
            # Slice out a batch
            start = step * global_batch_size
            end = min(start + global_batch_size, total_samples)

            batch_seqs = epoch_seqs[start:end]
            batch_pls = epoch_pls[start:end]
            batch_als = epoch_als[start:end]

            batch = collate_sft_batch(batch_seqs, batch_pls, batch_als, tokenizer)

            # Forward-backward with cross-entropy loss
            metrics = dispatch.forward_backward(
                "policy", batch, loss_fn="cross_entropy",
            )

            # Optimizer step
            grad_norm = dispatch.optim_step("policy")

            loss_val = metrics.get("final_loss", metrics.get("loss", 0.0))
            epoch_loss_acc += float(loss_val)
            epoch_tokens_acc += sum(batch_als)
            num_batches_done += 1

            if step % 10 == 0:
                avg_loss = epoch_loss_acc / max(1, num_batches_done)
                print(
                    f"  [epoch {epoch}] step={step}/{steps_per_epoch} "
                    f"loss={loss_val:.4f} avg_loss={avg_loss:.4f} "
                    f"grad_norm={grad_norm}"
                )

        # ---- Epoch summary ----
        epoch_time = time.time() - t_epoch_start
        avg_epoch_loss = epoch_loss_acc / max(1, num_batches_done)
        print(
            f"\n[Epoch {epoch}/{args.num_epochs}] "
            f"avg_loss={avg_epoch_loss:.4f} "
            f"tokens={epoch_tokens_acc:,} "
            f"steps={num_batches_done} "
            f"time={epoch_time:.1f}s"
        )

        # ---- Save checkpoint ----
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = str(output_dir / f"sft_epoch_{epoch}_{timestamp}")
            print(f"[Checkpoint] Saving HF model to {export_dir}...")
            dispatch.save_hf_model("policy", export_dir, tokenizer)

            # Save metadata
            meta = {
                "epoch": epoch,
                "avg_loss": avg_epoch_loss,
                "total_samples": total_samples,
                "model_name": args.model,
                "lora_rank": args.lora_rank,
                "lr": args.lr,
                "global_batch_size": global_batch_size,
            }
            meta_path = Path(export_dir) / "sft_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"[Checkpoint] Saved to {export_dir}")

    # ---- Cleanup ----
    print(f"\n[SFT] Training complete! {args.num_epochs} epochs, "
          f"{steps_per_epoch * args.num_epochs} steps")
    ray.shutdown()


if __name__ == "__main__":
    main()
