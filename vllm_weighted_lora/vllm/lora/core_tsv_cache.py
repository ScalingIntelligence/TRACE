# SPDX-License-Identifier: Apache-2.0
"""Cache layer for core/TSV merged LoRA adapters.

Handles cache key computation, file-based locking for multi-process safety,
and orchestration of merge-on-miss with ThreadPoolExecutor.
"""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

from vllm.logger import init_logger
from vllm.lora.core_tsv_merge import core_tsv_merge

logger = init_logger(__name__)

CACHE_DIR = "/tmp/vllm_core_tsv_cache"

# Single-thread executor so SVD merge doesn't block the async event loop
# but also doesn't compete with itself for GPU resources.
_merge_executor = ThreadPoolExecutor(max_workers=1)


def compute_cache_key(adapter_paths: list[str], weights: list[float]) -> str:
    """Deterministic hash of adapter paths + rounded weights."""
    key_data = sorted(
        (os.path.realpath(p), round(w, 4))
        for p, w in zip(adapter_paths, weights)
    )
    raw = json.dumps(key_data, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_entry_dir(cache_key: str) -> str:
    return os.path.join(CACHE_DIR, cache_key)


def _is_cache_hit(cache_key: str) -> bool:
    entry = _cache_entry_dir(cache_key)
    return os.path.isfile(os.path.join(entry, "adapter_model.safetensors"))


def _run_merge_locked(
    adapter_paths: list[str],
    weights: list[float],
    cache_key: str,
    target_rank: int | None,
) -> str:
    """Acquire flock, merge if still missing, return cache entry path.

    Called from a thread — safe to block.
    """
    entry_dir = _cache_entry_dir(cache_key)
    os.makedirs(entry_dir, exist_ok=True)

    lock_path = os.path.join(entry_dir, ".lock")
    with open(lock_path, "w") as lock_fd:
        logger.info("Acquiring lock for core/TSV merge: %s", cache_key)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # Double-check after acquiring lock
            if _is_cache_hit(cache_key):
                logger.info("Cache hit after lock (another process merged): %s", cache_key)
                return entry_dir

            # Merge to a temp dir first, then move atomically
            tmp_dir = tempfile.mkdtemp(
                prefix=f"core_tsv_{cache_key}_",
                dir=CACHE_DIR,
            )
            try:
                logger.info(
                    "Cache miss — running core/TSV merge for %d adapters (key=%s)",
                    len(adapter_paths), cache_key,
                )
                core_tsv_merge(
                    adapter_paths=adapter_paths,
                    weights=weights,
                    output_path=tmp_dir,
                    target_rank=target_rank,
                    force_cpu=True,  # GPU is occupied by vLLM model
                )
                # Move merged files into the cache entry dir atomically
                for fname in ("adapter_model.safetensors", "adapter_config.json"):
                    src = os.path.join(tmp_dir, fname)
                    dst = os.path.join(entry_dir, fname)
                    shutil.move(src, dst)
                logger.info("Core/TSV merge cached: %s", entry_dir)
            finally:
                # Clean up temp dir (may still have leftover files on error)
                shutil.rmtree(tmp_dir, ignore_errors=True)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return entry_dir


async def get_or_merge(
    adapter_paths: list[str],
    weights: list[float],
    target_rank: int | None = None,
) -> tuple[str, str]:
    """Return (cache_entry_path, cache_key) for the given adapter combination.

    If cache miss, performs core/TSV merge in a thread pool (blocks caller
    via await, but doesn't block the event loop for other requests).
    """
    cache_key = compute_cache_key(adapter_paths, weights)
    entry_dir = _cache_entry_dir(cache_key)

    # Fast path: cache hit without locking
    if _is_cache_hit(cache_key):
        logger.info("Core/TSV cache hit: %s", cache_key)
        return entry_dir, cache_key

    # Slow path: merge in thread pool
    import asyncio
    loop = asyncio.get_running_loop()
    result_path = await loop.run_in_executor(
        _merge_executor,
        _run_merge_locked,
        adapter_paths,
        weights,
        cache_key,
        target_rank,
    )
    return result_path, cache_key
