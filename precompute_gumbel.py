#!/usr/bin/env python3
"""
Precompute Gumbel top-k teacher outputs for offline distillation.

Runs the base model (no LoRA) over all SFT training data, extracts
log-probs at action-token positions, applies Gumbel top-k sampling,
and saves compressed teacher outputs alongside the training data.

Usage:
    # Single GPU:
    python precompute_gumbel.py --sft-data file1.json,file2.json --gumbel-k 100

    # Multi-GPU (8 GPUs):
    torchrun --nproc_per_node=8 precompute_gumbel.py --sft-data file1.json --gumbel-k 100
"""
import argparse
import os
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
from sft_buffer import SFTBuffer
from gumbel_topk import sample_gumbel_topk
from dist_utils import (
    dist_pre_init, dist_nccl_init,
    dist_cleanup, barrier,
    suppress_print,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute Gumbel top-k teacher outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sft-data", type=str, required=True,
                   help="Comma-separated paths to tau2-bench simulation JSON files")
    p.add_argument("--model", type=str, default=None,
                   help="HuggingFace model name (default: Config.MODEL_NAME)")
    p.add_argument("--gumbel-k", type=int, default=100,
                   help="Number of top-k tokens to sample per position")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for teacher forward passes")
    p.add_argument("--output", type=str, default=None,
                   help="Output .pt file path (default: auto)")
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="Max sequence length (default: Config.MAX_SEQ_LENGTH)")
    p.add_argument("--compact-tools", action="store_true", default=False)
    p.add_argument("--max-samples-per-file", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--root", type=str, default=None,
                   help="Root directory for cache and outputs")
    return p.parse_args()


def _resolve_output_path(args, env, k, total_samples):
    """Determine output .pt file path (deterministic across ranks)."""
    if args.output:
        return Path(args.output)
    out_dir = env["output_dir_path"] / "gumbel"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"gumbel_k{k}_n{total_samples}.pt"


def _process_batch(model, seqs, prompt_lens, action_lens, batch_idx, k, pad_id, device):
    """Run teacher forward pass on a batch and extract Gumbel top-k samples.

    Returns list of (orig_idx, gumbel_dict) tuples.
    """
    batch_seqs = [seqs[i] for i in batch_idx]
    batch_pls = [prompt_lens[i] for i in batch_idx]
    batch_als = [action_lens[i] for i in batch_idx]

    max_len = max(s.shape[0] for s in batch_seqs)
    B = len(batch_seqs)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    for i, s in enumerate(batch_seqs):
        L = s.shape[0]
        ids[i, :L] = s
        attn[i, :L] = 1

    ids = ids.to(device)
    attn = attn.to(device)

    with torch.no_grad():
        with autocast_ctx(device):
            outputs = model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    results = []
    empty = {
        "tokens": torch.zeros(0, k, dtype=torch.int32),
        "log_probs": torch.zeros(0, k, dtype=torch.float16),
        "thresholds": torch.zeros(0, k, dtype=torch.float16),
    }

    for i in range(B):
        pl = batch_pls[i]
        al = batch_als[i]
        orig_idx = batch_idx[i]

        if al <= 0:
            results.append((orig_idx, {key: val.clone() for key, val in empty.items()}))
            continue

        # Action tokens at positions [pl, pl+al).
        # Predicting token at position t requires logits at position t-1,
        # so we need logits at positions [pl-1, pl+al-1).
        assert pl >= 1, f"prompt_len must be >= 1, got {pl}"
        l_start = pl - 1
        l_end = pl + al - 1

        if l_end <= l_start:
            results.append((orig_idx, {key: val.clone() for key, val in empty.items()}))
            continue

        # [num_action_positions, V]
        action_logits = logits[i, l_start:l_end, :].float()
        log_probs = F.log_softmax(action_logits, dim=-1)

        tok, lp, thr = sample_gumbel_topk(log_probs, k)

        results.append((orig_idx, {
            "tokens": tok.cpu().to(torch.int32),
            "log_probs": lp.cpu().to(torch.float16),
            "thresholds": thr.cpu().to(torch.float16),
        }))

    del ids, attn, logits, outputs
    torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()

    # ---- Distributed init (Phase 1: env vars + CUDA device, NO NCCL) ----
    rank, world_size, local_rank = dist_pre_init()
    if world_size > 1 and rank != 0:
        suppress_print()

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    env = setup_environment(args)
    hf_hub = env["hf_hub"]
    max_seq_len = args.max_seq_length or Config.MAX_SEQ_LENGTH

    # ---- Load model (no LoRA — this IS the base/reference model) ----
    from unsloth import FastLanguageModel

    model_name = args.model or Config.MODEL_NAME
    print(f"[Model] Rank {rank}: Loading {model_name} (base, no LoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        use_exact_model_name=True,
        cache_dir=str(hf_hub),
        device_map={"": local_rank},
    )
    print(f"[Model] Rank {rank}: Model loaded successfully", flush=True)

    # ---- Distributed init (Phase 2: NCCL after model loading) ----
    if world_size > 1:
        print(f"[NCCL] Rank {rank}: Initializing NCCL process group...", flush=True)
        dist_nccl_init()
        print(f"[NCCL] Rank {rank}: NCCL initialized", flush=True)
        barrier()

    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ---- Load SFT data (all ranks load independently — deterministic) ----
    sft_paths = [p.strip() for p in args.sft_data.split(",") if p.strip()]

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

    print(f"[Data] Loading from {len(sft_paths)} files...")
    sft_buffer = SFTBuffer(sft_paths, tokenizer, compact_tools=args.compact_tools,
                           max_samples_per_file=max_per_file,
                           max_seq_len=max_seq_len)
    total_samples = len(sft_buffer)
    print(f"[Data] Loaded {total_samples} samples")

    if args.max_samples is not None and args.max_samples < total_samples:
        sft_buffer._seqs_cpu = sft_buffer._seqs_cpu[:args.max_samples]
        sft_buffer._prompt_lens = sft_buffer._prompt_lens[:args.max_samples]
        sft_buffer._action_lens = sft_buffer._action_lens[:args.max_samples]
        total_samples = len(sft_buffer)
        print(f"[Data] Truncated to {total_samples} samples")

    if total_samples == 0:
        print("[Data] ERROR: No samples loaded.")
        dist_cleanup()
        return

    seqs = sft_buffer._seqs_cpu
    prompt_lens = sft_buffer._prompt_lens
    action_lens = sft_buffer._action_lens

    k = args.gumbel_k

    # ---- Sort by length for efficient batching ----
    indices = sorted(range(total_samples), key=lambda i: seqs[i].shape[0])

    # ---- Partition indices across ranks (interleaved for balanced lengths) ----
    my_indices = indices[rank::world_size]
    my_total = len(my_indices)

    print(f"[Gumbel] Rank {rank}: k={k}, processing {my_total}/{total_samples} samples")

    # ---- Process this rank's samples ----
    # Output: list of dicts, one per sample, each containing:
    #   tokens:     [num_action_positions, k]  int32
    #   log_probs:  [num_action_positions, k]  float16
    #   thresholds: [num_action_positions, k]  float16
    my_gumbel_data = []

    t_start = time.time()

    for batch_start in tqdm(
        range(0, my_total, args.batch_size),
        desc=f"Rank {rank}" if world_size > 1 else "Precomputing",
        disable=(rank != 0),
    ):
        batch_end = min(batch_start + args.batch_size, my_total)
        batch_idx = my_indices[batch_start:batch_end]

        results = _process_batch(
            model, seqs, prompt_lens, action_lens,
            batch_idx, k, pad_id, device,
        )
        my_gumbel_data.extend(results)

    elapsed = time.time() - t_start
    print(f"[Gumbel] Rank {rank}: Processed {len(my_gumbel_data)} samples in {elapsed:.1f}s")

    # ---- Gather results across ranks ----
    if world_size > 1:
        # Each rank saves its shard to a deterministic path, rank 0 merges.
        # This avoids sending large variable-size data through NCCL.
        out_path = _resolve_output_path(args, env, k, total_samples)
        shard_dir = out_path.parent / ".gumbel_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        shard_path = shard_dir / f"shard_{rank}.pt"
        torch.save(my_gumbel_data, str(shard_path))
        print(f"[Gumbel] Rank {rank}: Saved shard ({len(my_gumbel_data)} samples) to {shard_path}", flush=True)

        barrier()

        if rank == 0:
            all_gumbel_data = list(my_gumbel_data)
            for r in range(1, world_size):
                other_path = shard_dir / f"shard_{r}.pt"
                other_data = torch.load(str(other_path), map_location="cpu", weights_only=False)
                all_gumbel_data.extend(other_data)
                os.remove(str(other_path))
            os.remove(str(shard_path))
            try:
                shard_dir.rmdir()
            except OSError:
                pass

            assert len(all_gumbel_data) == total_samples, (
                f"Gathered {len(all_gumbel_data)} samples but expected {total_samples}"
            )
        else:
            all_gumbel_data = None
    else:
        all_gumbel_data = my_gumbel_data

    # ---- Save (rank 0 only, or single-GPU) ----
    if rank == 0:
        # Sort back to original sample order
        all_gumbel_data.sort(key=lambda x: x[0])
        gumbel_list = [d for _, d in all_gumbel_data]

        out_path = _resolve_output_path(args, env, k, total_samples)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "gumbel_k": k,
            "model_name": model_name,
            "num_samples": total_samples,
            "data": gumbel_list,
        }
        torch.save(save_data, str(out_path))

        total_positions = sum(d["tokens"].shape[0] for d in gumbel_list)
        bytes_per_pos = k * (4 + 2 + 2)  # int32 + fp16 + fp16
        total_bytes = total_positions * bytes_per_pos
        print(f"[Gumbel] Saved to {out_path}")
        print(f"[Gumbel] {total_positions} action positions, "
              f"{total_bytes / 1024 / 1024:.1f} MB uncompressed")

    # ---- Cleanup ----
    if world_size > 1:
        barrier()
    dist_cleanup()


if __name__ == "__main__":
    main()
