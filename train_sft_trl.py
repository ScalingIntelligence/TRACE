#!/usr/bin/env python3
"""
SFT trainer using unsloth + TRL SFTTrainer.

Drop-in alternative to train_sft.py that leverages TRL's SFTTrainer
instead of a manual training loop.  Uses the same data loading,
tokenization (build_prompt_plus_action), and LoRA setup via unsloth.

Loss is computed only on action (completion) tokens — prompt tokens
are masked with -100 in labels, identical to train_sft.py.  Uses the
same per-sequence normalized loss: -mean(sum(logp_action) / action_len),
so each sequence contributes equally regardless of length.

Usage:
    # Single GPU:
    python train_sft_trl.py --sft-data file1.json,file2.json

    # Multi-GPU (via accelerate):
    accelerate launch train_sft_trl.py --sft-data file1.json,file2.json

Note: --pack-sequences uses group_by_length (length-sorted batching)
      rather than TRL's packing, because TRL packing does not support
      pre-tokenized label masking.  The original train_sft.py is
      recommended if true sequence packing is needed.
"""
import argparse
import os
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")

from unsloth import FastLanguageModel  # must be imported before trl/transformers

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

import torch
import torch.nn.functional as F
from datasets import Dataset
from pathlib import Path
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig

from config import Config, setup_environment
from ppo import build_prompt_plus_action
from sft_buffer import load_sft_samples_multi


# ---------------------------------------------------------------------------
# Custom SFTTrainer with per-sequence normalized loss
# ---------------------------------------------------------------------------

class PerSeqNormSFTTrainer(SFTTrainer):
    """SFTTrainer with per-sequence normalized loss matching train_sft.py.

    Default TRL loss is per-token cross-entropy, where longer sequences
    dominate the gradient.  This trainer instead computes:

        loss = -mean_over_batch( sum(logp_action_tokens) / action_len )

    so each sequence contributes equally regardless of length.  This is
    identical to train_sft.py's logprob_action_tokens(normalize_by_len=True).
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Action mask: where labels are not -100
        action_mask = (shift_labels != -100).float()

        # Per-token log-probs via cross-entropy
        B, T, V = shift_logits.shape
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1).clamp(0, V - 1)
        flat_nll = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        token_logp = -flat_nll.view(B, T)

        # Per-sequence: sum of action log-probs / action_len
        per_seq_logp = (token_logp * action_mask).sum(dim=1)
        action_lens = action_mask.sum(dim=1).clamp(min=1)
        per_seq_normalized = per_seq_logp / action_lens

        loss = -per_seq_normalized.mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT trainer (TRL) for tau2-bench trajectory data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data --
    parser.add_argument("--sft-data", type=str, required=True,
        help="Comma-separated paths to tau2-bench simulation JSON files")

    # -- Model --
    parser.add_argument("--model", type=str, default=None,
        help="HuggingFace model name (default: Config.MODEL_NAME)")

    # -- LoRA --
    parser.add_argument("--lora-rank", type=int, default=16,
        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
        help="LoRA alpha scaling factor")

    # -- Optimization --
    parser.add_argument("--lr", type=float, default=1e-5,
        help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3,
        help="Number of full passes over the SFT dataset")
    parser.add_argument("--mini-batch-size", type=int, default=4,
        help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
        help="Number of mini-batches to accumulate before each optimizer step")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
        help="Fraction of total steps for linear warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01,
        help="Weight decay for optimizer")
    parser.add_argument("--optim", type=str, default="adamw_8bit",
        help="Optimizer name (adamw_torch, adamw_8bit, paged_adamw_8bit)")

    # -- Performance --
    parser.add_argument("--pack-sequences", action="store_true", default=False,
        help="Enable group_by_length for reduced padding waste "
             "(true packing not supported with pre-tokenized label masking)")

    # -- Data processing --
    parser.add_argument("--compact-tools", action="store_true", default=False,
        help="Use compressed tool schemas (strips descriptions)")
    parser.add_argument("--max-seq-length", type=int, default=None,
        help="Max sequence length (default: Config.MAX_SEQ_LENGTH)")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Train on only the first N samples (after seq-length filtering)")
    parser.add_argument("--max-samples-per-file", type=str, default=None,
        help="Limit each input file to at most N samples before merging. "
             "A single int applies to all files; a comma-separated list "
             "sets per-file limits matching --sft-data order.")
    parser.add_argument("--shuffle-seed", type=int, default=42,
        help="Random seed for data shuffling")

    # -- Checkpointing --
    parser.add_argument("--save-every", type=int, default=1,
        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--output-dir", type=str, default=None,
        help="Output directory for checkpoints (default: auto)")

    # -- Logging --
    parser.add_argument("--log-every", type=int, default=10,
        help="Log metrics every N optimizer steps")
    parser.add_argument("--wandb-project", type=str, default="games",
        help="wandb project name")

    # -- Root directory --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def build_pretokenized_dataset(
    samples,
    tokenizer,
    max_seq_len: int,
    min_prompt_len: int = 128,
):
    """Pre-tokenize SFT samples into a HuggingFace Dataset with labels.

    Each sample is tokenized via build_prompt_plus_action (same as
    train_sft.py).  Labels have -100 on prompt tokens so loss is
    computed only on the action (completion) tokens.

    Returns:
        (Dataset, stats_dict)
    """
    data_rows = []
    truncated = 0
    dropped = 0

    for sample in samples:
        try:
            seq, pl, _al = build_prompt_plus_action(
                tokenizer,
                sample.prompt_msgs,
                sample.completion_text,
                tools=sample.tools,
            )
        except Exception as e:
            print(f"[SFT-TRL] Warning: tokenization failed for task {sample.task_id}: {e}")
            continue

        total_len = seq.shape[0]

        # Front-truncate prompt if sequence exceeds max_seq_len
        if max_seq_len and total_len > max_seq_len:
            excess = total_len - max_seq_len
            new_pl = pl - excess
            if new_pl < min_prompt_len:
                dropped += 1
                continue
            seq = seq[excess:]
            pl = new_pl
            truncated += 1

        input_ids = seq.tolist()
        # Mask prompt tokens with -100; keep action token ids as labels
        labels = [-100] * pl + input_ids[pl:]

        data_rows.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        })

    stats = {"total": len(data_rows), "truncated": truncated, "dropped": dropped}
    return Dataset.from_list(data_rows), stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Environment setup ----
    env = setup_environment(args)
    max_seq_len = args.max_seq_length or Config.MAX_SEQ_LENGTH

    # ---- Output directory ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = env["output_dir_path"] / "sft_trl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + LoRA via unsloth ----
    model_name = args.model or Config.MODEL_NAME
    print(f"[Model] Loading {model_name} with LoRA rank={args.lora_rank}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        use_exact_model_name=True,
        cache_dir=str(env["hf_hub"]),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Model] Loaded with LoRA — SFT-TRL mode")

    # ---- Load SFT data ----
    sft_paths = [p.strip() for p in args.sft_data.split(",") if p.strip()]

    # Parse --max-samples-per-file
    max_per_file = None
    if args.max_samples_per_file is not None:
        parts = [x.strip() for x in args.max_samples_per_file.split(",")]
        if len(parts) == 1:
            max_per_file = [int(parts[0])] * len(sft_paths)
        else:
            if len(parts) != len(sft_paths):
                raise ValueError(
                    f"--max-samples-per-file has {len(parts)} values but "
                    f"--sft-data has {len(sft_paths)} files"
                )
            max_per_file = [int(x) for x in parts]

    print(f"[SFT-TRL] Loading data from {len(sft_paths)} files...")
    samples = load_sft_samples_multi(
        sft_paths,
        compact_tools=args.compact_tools,
        max_samples_per_file=max_per_file,
    )

    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        print(f"[SFT-TRL] Truncated to first {args.max_samples} samples (--max-samples)")

    print(f"[SFT-TRL] Loaded {len(samples)} samples from {len(sft_paths)} files")

    if not samples:
        print("[SFT-TRL] ERROR: No samples loaded. "
              "Check that JSON files contain reward=1.0 simulations.")
        return

    # ---- Pre-tokenize into HF Dataset ----
    dataset, stats = build_pretokenized_dataset(
        samples, tokenizer, max_seq_len,
    )

    if stats["truncated"] or stats["dropped"]:
        print(f"[SFT-TRL] Front-truncation: {stats['truncated']} truncated, "
              f"{stats['dropped']} dropped (action too long for max_seq_len={max_seq_len})")

    total_samples = stats["total"]
    print(f"[SFT-TRL] Pre-tokenized {total_samples} samples")

    if total_samples == 0:
        print("[SFT-TRL] ERROR: No samples survived tokenization + length filtering.")
        return

    # ---- Sequence length stats ----
    seq_lens = [len(row["input_ids"]) for row in dataset]
    print(f"[SFT-TRL] Sequence length stats: "
          f"min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={sum(seq_lens)/len(seq_lens):.0f}, "
          f"median={sorted(seq_lens)[len(seq_lens)//2]}")

    # ---- wandb setup ----
    report_to = "none"
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        try:
            import wandb
            wandb.login(key=wandb_api_key, relogin=True)
            os.environ.setdefault("WANDB_ENTITY", "forge_scaling_intelligence_lab")
            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            report_to = "wandb"
        except Exception:
            pass

    run_name = os.getenv("WANDB_NAME", f"sft-trl-{int(time.time())}")

    # ---- Compute schedule info for save_steps ----
    effective_batch = args.mini_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = max(1, total_samples // effective_batch)
    save_steps = max(1, steps_per_epoch * args.save_every)

    # ---- SFTConfig ----
    sft_config = SFTConfig(
        output_dir=str(output_dir),

        # Batch / accumulation
        per_device_train_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Schedule
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,

        # Optimizer
        optim=args.optim,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        # Precision
        bf16=True,
        fp16=False,

        # DDP — LoRA leaves some params unused; static graph avoids the
        # extra autograd traversal from find_unused_parameters=True.
        ddp_find_unused_parameters=False,

        # Gradient checkpointing (unsloth handles the actual implementation
        # via use_gradient_checkpointing="unsloth" above)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=args.log_every,
        report_to=report_to,
        run_name=run_name,

        # Saving — save at end of each epoch
        save_strategy="epoch",
        save_total_limit=args.num_epochs + 1,

        # Data handling — dataset is already pre-tokenized with labels,
        # so TRL skips its own tokenization (checks "input_ids" in columns).
        # We disable packing since it requires text-format data.
        max_length=max_seq_len,
        packing=False,
        group_by_length=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,

        # Reproducibility
        seed=args.shuffle_seed,
    )

    # ---- Data collator: pads input_ids/attention_mask, labels padded with -100 ----
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ---- Trainer (per-sequence normalized loss) ----
    trainer = PerSeqNormSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        args=sft_config,
    )

    # ---- Print configuration ----
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 60}")
    print(f"  SFT-TRL Training Configuration")
    print(f"{'=' * 60}")
    print(f"  Model:              {model_name}")
    print(f"  LoRA rank:          {args.lora_rank}")
    print(f"  LoRA alpha:         {args.lora_alpha}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Optimizer:          {args.optim}")
    print(f"  Weight decay:       {args.weight_decay}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Total samples:      {total_samples}")
    print(f"  Mini-batch size:    {args.mini_batch_size}")
    print(f"  Grad accum steps:   {args.gradient_accumulation_steps}")
    print(f"  Effective batch:    {effective_batch}")
    print(f"  Steps/epoch:        ~{steps_per_epoch}")
    print(f"  Save every:         {save_steps} steps ({args.save_every} epoch(s))")
    print(f"  Group by length:    True")
    print(f"  Max seq length:     {max_seq_len}")
    print(f"  Max grad norm:      {args.max_grad_norm}")
    print(f"  Compact tools:      {args.compact_tools}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"  Output dir:         {output_dir}")
    print(f"{'=' * 60}\n")

    # ---- Train ----
    trainer.train(resume_from_checkpoint=args.resume)

    # ---- Save final model ----
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[SFT-TRL] Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
