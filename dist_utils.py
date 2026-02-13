#!/usr/bin/env python3
"""
Distributed training utilities for GRPO trainer.

Provides helper functions for multi-GPU training with torch.distributed.
Falls back to single-GPU no-ops when not running under torchrun/distributed.
"""
import builtins
import os
import pickle
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist


def dist_init() -> Tuple[int, int, int]:
    """Initialize distributed training if running under torchrun.

    Returns:
        (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def isolate_gpu_per_rank() -> int:
    """Restrict each rank to a single GPU via CUDA_VISIBLE_DEVICES.

    Must be called BEFORE any CUDA operations (including torch.cuda.set_device).
    Rewrites CUDA_VISIBLE_DEVICES so each rank sees only its assigned GPU as
    cuda:0.  This prevents from_pretrained / accelerate / bitsandbytes from
    creating CUDA contexts on all visible GPUs, which causes NCCL hangs.

    Returns:
        The original LOCAL_RANK (before overriding to 0).
    """
    if "LOCAL_RANK" not in os.environ:
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if visible:
        gpus = [g.strip() for g in visible.split(",")]
        if local_rank < len(gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[local_rank]
    else:
        # No CUDA_VISIBLE_DEVICES set; isolate to the local_rank-th GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    # Override LOCAL_RANK to 0 since each process now sees exactly 1 GPU
    os.environ["LOCAL_RANK"] = "0"
    return local_rank


def dist_pre_init() -> Tuple[int, int, int]:
    """Phase 1 of distributed init: read env vars and set CUDA device.

    Does NOT initialize NCCL. Use this before model loading to avoid
    from_pretrained / accelerate triggering unexpected collective operations
    when torch.distributed.is_initialized() is True.

    Call dist_nccl_init() after model loading to complete initialization.

    Returns:
        (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def dist_nccl_init() -> None:
    """Phase 2 of distributed init: initialize NCCL process group.

    Call after model loading is complete, paired with dist_pre_init().
    Passes device_id to avoid the 'Guessing device ID' NCCL warning/hang.
    No-op if already initialized or not running under torchrun.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(minutes=30),
        )


def dist_cleanup() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank() -> bool:
    """True if this is rank 0 or not running distributed."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier() -> None:
    """Synchronize all ranks. No-op if not distributed."""
    if dist.is_initialized():
        dist.barrier()


def suppress_print() -> None:
    """Replace builtins.print with a no-op (for non-main ranks)."""
    builtins.print = lambda *args, **kwargs: None


def broadcast_objects(obj: Any) -> Any:
    """Broadcast a Python object from rank 0 to all other ranks.

    Uses pickle serialization. Returns the broadcasted object on all ranks.
    No-op if not distributed (returns obj unchanged).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return obj

    if dist.get_rank() == 0:
        data = pickle.dumps(obj)
        size = torch.tensor([len(data)], dtype=torch.long, device="cuda")
    else:
        size = torch.tensor([0], dtype=torch.long, device="cuda")

    dist.broadcast(size, src=0)

    if dist.get_rank() == 0:
        data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).to("cuda")
    else:
        data_tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda")

    dist.broadcast(data_tensor, src=0)

    if dist.get_rank() != 0:
        obj = pickle.loads(data_tensor.cpu().numpy().tobytes())

    return obj


def shard_batches(
    batch_indices: List[int],
    rank: int,
    world_size: int,
) -> Tuple[List[int], int]:
    """Shard a list of batch indices across ranks.

    Distributes indices round-robin so each rank gets a roughly equal share.

    Returns:
        (my_indices, total_count): indices for this rank, total number of batches
    """
    total = len(batch_indices)
    if world_size <= 1:
        return batch_indices, total

    my_indices = [batch_indices[i] for i in range(rank, total, world_size)]
    return my_indices, total


def allreduce_coalesced_grads(params: List[torch.nn.Parameter]) -> None:
    """All-reduce gradients across ranks using coalesced communication.

    Combines small gradient tensors into larger buffers for better
    communication efficiency. No-op if not distributed.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    world_size = dist.get_world_size()

    # Coalesce into a single flat buffer for fewer all-reduce calls
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)

    # Scatter back
    offset = 0
    for g in grads:
        numel = g.numel()
        g.copy_(flat[offset:offset + numel].reshape(g.shape))
        offset += numel


def allreduce_scalars(local_dict: Dict[str, float]) -> Dict[str, float]:
    """All-reduce scalar metrics across ranks and return summed results.

    Returns:
        Dictionary with summed values across all ranks.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return local_dict

    keys = sorted(local_dict.keys())
    vals = torch.tensor([local_dict[k] for k in keys], dtype=torch.float64, device="cuda")
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)

    return {k: float(vals[i].item()) for i, k in enumerate(keys)}
