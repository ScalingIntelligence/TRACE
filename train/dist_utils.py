#!/usr/bin/env python3
import builtins
import os
import pickle
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist


def dist_init() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        timeout_mins = int(os.environ.get("NCCL_TIMEOUT_MINS", "180"))
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=timeout_mins),
        )
        return rank, world_size, local_rank
    return 0, 1, 0


def isolate_gpu_per_rank() -> int:
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
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def dist_nccl_init() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        timeout_mins = int(os.environ.get("NCCL_TIMEOUT_MINS", "180"))
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(minutes=timeout_mins),
        )


def dist_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def suppress_print() -> None:
    builtins.print = lambda *args, **kwargs: None


def broadcast_objects(obj: Any) -> Any:
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

    # Move to CPU and free CUDA tensor immediately to avoid GPU memory
    # fragmentation.  Without this, the serialized payload stays in the
    # CUDA caching allocator and inflates peak GPU memory vs single-GPU.
    cpu_bytes = data_tensor.cpu().numpy().tobytes()
    del data_tensor, size

    if dist.get_rank() != 0:
        obj = pickle.loads(cpu_bytes)

    del cpu_bytes
    return obj


def shard_batches(
    batch_indices: List[int],
    rank: int,
    world_size: int,
) -> Tuple[List[int], int]:
    total = len(batch_indices)
    if world_size <= 1:
        return batch_indices, total

    my_indices = [batch_indices[i] for i in range(rank, total, world_size)]
    return my_indices, total


def allreduce_coalesced_grads(params: List[torch.nn.Parameter]) -> None:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    world_size = dist.get_world_size()

    # Collect existing gradients; zero-initialize missing ones so every
    # rank contributes the same number of elements to the all-reduce.
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad)
        else:
            grads.append(torch.zeros_like(p))

    # Coalesce into a single flat buffer for fewer all-reduce calls
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)

    # Scatter back into the actual .grad tensors
    offset = 0
    for p, g in zip(params, grads):
        numel = g.numel()
        reduced = flat[offset:offset + numel].reshape(g.shape)
        if p.grad is not None:
            p.grad.copy_(reduced)
        else:
            p.grad = reduced.clone()
        offset += numel


def allreduce_scalars(local_dict: Dict[str, float]) -> Dict[str, float]:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return local_dict

    keys = sorted(local_dict.keys())
    vals = torch.tensor([local_dict[k] for k in keys], dtype=torch.float64, device="cuda")
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)

    return {k: float(vals[i].item()) for i, k in enumerate(keys)}
