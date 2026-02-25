#!/usr/bin/env python3
"""
Evaluation functions for tau2-bench.
"""
import time
import random
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from inference import InferenceBackend
from config import Config


# =========================
# tau2-bench evaluation
# =========================

def _tau2_harness_dir() -> Path:
    """Return the path to the vendored tau2-bench eval harness."""
    return Path(__file__).resolve().parent / "evals" / "benchmarks" / "tau2_bench_eval"


def _load_tau2_task_ids(tasks_path: Path) -> List[Any]:
    """Load tau2 task ids from a tasks.json file.

    The upstream tau2-bench JSON schema may evolve; we try a few common layouts.
    Supports both integer IDs (airline, retail) and string IDs (telecom).
    """
    with open(tasks_path, "r") as f:
        data = json.load(f)

    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError(f"Unexpected tasks.json format: {type(data)}")

    ids: List[Any] = []
    for idx, t in enumerate(tasks):
        if isinstance(t, dict):
            for k in ("task_id", "taskId", "id", "task"):  # best-effort
                if k in t:
                    v = t.get(k)
                    if v is not None:
                        # Accept both int and string IDs
                        if isinstance(v, int):
                            ids.append(v)
                        elif isinstance(v, str):
                            # Try to convert to int if numeric, otherwise keep as string
                            if v.strip().isdigit():
                                ids.append(int(v.strip()))
                            else:
                                ids.append(v.strip())
                        break
            else:
                # Fallback to index only if no ID field found
                ids.append(idx)
        else:
            ids.append(idx)

    # Keep deterministic order (works for both int and string)
    ids = sorted(set(ids), key=lambda x: (isinstance(x, str), str(x)))
    return ids


def _split_round_robin(items: List[Any], n_shards: int) -> List[List[Any]]:
    if n_shards <= 1:
        return [list(items)]
    shards: List[List[Any]] = [[] for _ in range(n_shards)]
    for i, x in enumerate(items):
        shards[i % n_shards].append(x)
    return shards


def _parse_tau2_success_from_file(path: Path) -> Optional[Dict[str, float]]:
    """Parse a tau2 simulation artifact and return success stats if possible."""
    try:
        if path.suffix.lower() in {".jsonl", ".jl"}:
            rows = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        elif path.suffix.lower() == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = data.get("rows") or data.get("simulations") or data.get("results") or []
            else:
                rows = []
        else:
            return None

        if not isinstance(rows, list) or not rows:
            return None

        total = 0
        success = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            total += 1

            reward_info = r.get('reward_info')
            if reward_info and isinstance(reward_info, dict):
                reward = reward_info.get('reward', 0)
                success += 1 if reward > 0 else 0
                continue
            
            
            for k in ("success", "is_success", "task_success", "passed", "solved"):
                if k in r:
                    v = r.get(k)
                    success += 1 if bool(v) else 0
                    break
            else:
                total -= 1

        if total <= 0:
            return None

        return {
            "total": float(total),
            "success": float(success),
            "success_rate": float(success) / float(total),
        }
    except Exception:
        return None


def evaluate_tau2_bench(
    *,
    backend: InferenceBackend,
    output_dir: Path,
    iteration: int,
    domains: List[str],
    num_tasks: Optional[int] = 8,
    num_trials: int = 1,
    max_concurrency_per_shard: int = 1,
    seed: int = 42,
    agent_adapter_name: Optional[str] = None,
    user_base_model_name: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Run tau2-bench evaluation and return summary metrics.

    This function assumes you already have one or more vLLM OpenAI-compatible servers
    running (as you do for PPO rollouts). It shards task IDs across available vLLM
    base URLs so evaluation load is distributed across GPUs.

    Agent model:
      - Uses the *currently-trained* LoRA adapter loaded into vLLM (default: VLLM_LORA_NAME / ppo_policy).

    User model:
      - Uses the *base* model served by vLLM (default: VLLM_MODEL / Config.MODEL_NAME).

    Requirements:
      - tau2 CLI must be installed and on PATH.
      - tau2 data must be downloaded (run setup_data.py in evals/benchmarks/tau2_bench_eval/).
    """
    harness_dir = _tau2_harness_dir()
    if not harness_dir.exists():
        print(f"[tau2_eval] Harness directory not found: {harness_dir} (skipping)")
        return {}

    if shutil.which("tau2") is None:
        print(
            "[tau2_eval] 'tau2' CLI not found on PATH (skipping). "
            "Install tau2-bench (see evals/benchmarks/tau2_bench_eval/README.md)."
        )
        return {}

    # Extract vLLM base URLs from backend
    vllm_urls: List[str] = []
    if hasattr(backend, "backends"):
        vllm_urls = [getattr(b, "base_url") for b in getattr(backend, "backends")]
    elif hasattr(backend, "base_url"):
        vllm_urls = [getattr(backend, "base_url")]

    vllm_urls = [u for u in vllm_urls if isinstance(u, str) and u.strip()]

    if not vllm_urls:
        print("[tau2_eval] No vLLM servers detected from backend (skipping).")
        return {}

    # Agent adapter name defaults to the LoRA name used by the vLLM backend
    if agent_adapter_name is None:
        if hasattr(backend, "lora_name"):
            agent_adapter_name = getattr(backend, "lora_name")
        elif hasattr(backend, "backends") and getattr(backend, "backends"):
            agent_adapter_name = getattr(getattr(backend, "backends")[0], "lora_name", None)
        agent_adapter_name = agent_adapter_name or os.getenv("VLLM_LORA_NAME", "ppo_policy")

    # User base model defaults to the model served by vLLM
    if user_base_model_name is None:
        if hasattr(backend, "model_name"):
            user_base_model_name = getattr(backend, "model_name")
        elif hasattr(backend, "backends") and getattr(backend, "backends"):
            user_base_model_name = getattr(getattr(backend, "backends")[0], "model_name", None)
        user_base_model_name = user_base_model_name or os.getenv("VLLM_MODEL", Config.MODEL_NAME)

    # Data dir: make sure TAU2_DATA_DIR points to <...>/data (which contains tau2/)
    tau2_data_dir = (harness_dir / "data").resolve()

    # Output/logging
    out_root = (output_dir / "tau2_bench").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    logs: Dict[str, float] = {}
    saved_eval_files: List[Path] = []  # paths of copied simulation artifacts

    # ---- Run domains ----
    for domain in domains:
        tasks_path = harness_dir / "data" / "tau2" / "domains" / str(domain) / "tasks.json"
        if not tasks_path.exists() and str(domain).endswith("-workflow"):
            alt_domain = str(domain).replace("-workflow", "")
            alt_path = harness_dir / "data" / "tau2" / "domains" / alt_domain / "tasks.json"
            if alt_path.exists():
                tasks_path = alt_path

        if not tasks_path.exists():
            print(
                f"[tau2_eval] Missing tau2 tasks file: {tasks_path}. "
                "Run: python evals/benchmarks/tau2_bench_eval/setup_data.py"
            )
            continue

        all_ids = _load_tau2_task_ids(tasks_path)
        if not all_ids:
            print(f"[tau2_eval] No tasks found for domain={domain} (skipping).")
            continue

        selected_ids = list(all_ids)
        if num_tasks is not None:
            # Deterministic sample for stable comparisons across training.
            rng = random.Random(int(seed))
            rng.shuffle(selected_ids)
            selected_ids = selected_ids[: int(num_tasks)]

        # Shard across vLLM servers
        n_shards = min(len(vllm_urls), max(1, len(selected_ids)))
        shards = _split_round_robin(selected_ids, n_shards)

        t0 = time.time()

        procs = []
        save_prefixes: List[str] = []
        for shard_idx, shard_ids in enumerate(shards):
            if not shard_ids:
                continue

            base_url = vllm_urls[shard_idx % len(vllm_urls)]
            # Include timestamp to ensure unique names across runs
            timestamp = int(time.time())
            save_to = f"tau2_{domain}_iter{int(iteration)}_shard{int(shard_idx)}_{timestamp}"
            save_prefixes.append(save_to)

            cmd = [
                sys.executable,
                str((harness_dir / "main.py").resolve()),
                "--config",
                str((harness_dir / "config.yml").resolve()),
                "--domain",
                str(domain),
                "--agent-llm",
                f"vllm://{agent_adapter_name}",
                "--user-llm",
                f"vllm://{user_base_model_name}",
                "--num-trials",
                str(int(num_trials)),
                "--max-concurrency",
                str(int(max_concurrency_per_shard)),
                "--seed",
                str(int(seed) + int(shard_idx)),
                "--save-to",
                save_to,
            ]
            if verbose:
                cmd.append("--verbose")
            # Task ids
            cmd.append("--task-ids")
            cmd.extend([str(x) for x in shard_ids])

            env = os.environ.copy()
            env["OPENAI_API_BASE"] = base_url
            env["OPENAI_BASE_URL"] = base_url
            env.setdefault("OPENAI_API_KEY", "dummy-key")
            env["TAU2_DATA_DIR"] = str(tau2_data_dir)

            log_path = out_root / f"{save_to}.log"
            f = open(log_path, "w")
            print(f"[tau2_eval] Launch: {save_to} base_url={base_url} tasks={len(shard_ids)} -> {log_path}")
            p = subprocess.Popen(cmd, cwd=str(harness_dir), env=env, stdout=f, stderr=subprocess.STDOUT)
            procs.append((p, f, save_to))

        # Wait for shards
        failed = []
        running = {save_to: (p, f) for p, f, save_to in procs}

        while running:
            still_running = []
            for save_to, (p, f) in list(running.items()):
                rc = p.poll()
                if rc is not None:
                    f.close()
                    if rc != 0:
                        failed.append((save_to, rc))
                    print(f"[tau2_eval] {save_to} completed with status {rc}")
                    del running[save_to]
                else:
                    still_running.append(save_to)

            if running:
                elapsed = time.time() - t0
                time.sleep(5)

        t1 = time.time()
        logs[f"eval_tau2/{domain}_time_sec"] = float(t1 - t0)

        if failed:
            print(f"[tau2_eval] One or more shards failed for domain={domain}: {failed}")
            

        # Collect and parse results
        simulations_dir = tau2_data_dir / "simulations"
        if not simulations_dir.exists():
            simulations_dir = harness_dir / "tau2" / "simulations" 
        total = 0.0
        success = 0.0
        copied = 0

        for save_to in save_prefixes:
            matches = sorted(simulations_dir.glob(f"{save_to}*"))
            for src in matches:
                if src.is_dir():
                    continue

                stats = _parse_tau2_success_from_file(src)
                if stats is not None:
                    total += stats["total"]
                    success += stats["success"]
                    
                dst = out_root / src.name
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                    saved_eval_files.append(dst)
                except Exception:
                    pass

        if total > 0:
            logs[f"eval_tau2/{domain}_success_rate"] = float(success / total)
            logs[f"eval_tau2/{domain}_success"] = float(success)
            logs[f"eval_tau2/{domain}_total"] = float(total)
        logs[f"eval_tau2/{domain}_artifacts_copied"] = float(copied)

        print(
            f"[tau2_eval] domain={domain} success_rate={logs.get(f'eval_tau2/{domain}_success_rate', float('nan')):.3f} "
            f"(total={int(total)}), time={t1-t0:.1f}s, copied={copied}"
        )

    return logs, saved_eval_files
