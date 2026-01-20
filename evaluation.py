#!/usr/bin/env python3
"""
Evaluation functions for poker agents and math benchmarks.
"""
import gc
import time
import random
import os
import sys
import json
import shutil
import subprocess
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_from_disk


from game_registry import GameSpec
from inference import InferenceBackend, messages_for_game, messages_for_math, build_prompt_text, generate_completion
from config import Config


# =========================
# Math evaluation helpers
# =========================
def extract_boxed_answer(text: str) -> str:
    """Extract the model's answer from \\boxed{...}."""
    if "boxed" not in text:
        return ""

    ans = text.split("boxed")[-1]

    if not ans:
        return ""

    if ans[0] == "{":
        stack = 1
        result = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                result += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                result += c
            else:
                result += c

        return result.strip()
    else:
        return ans.split("$")[0].strip()


@torch.no_grad()
def evaluate_math(
    model,
    tokenizer,
    data_path: Path,
    dataset_name: str,
    num_samples: int = 50,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    backend = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate model on the math benchmark (grading using harness)"""
    import sys
    _HARNESS_PATH = Path(__file__).resolve().parent / "evals" / "benchmarks" / "math-evaluation-harness"
    sys.path.insert(0, str(_HARNESS_PATH))

    from grader import math_equal
    from parser import extract_answer, strip_string, parse_ground_truth

    # Loading the dataset.
    dataset_path = data_path / dataset_name

    if not dataset_path.exists():
        print(f"[math_eval] Dataset not found: {dataset_path}")
        return {f"eval_math/{dataset_name}": 0.0}

    dataset = load_from_disk(dataset_path)

    total = len(dataset)
    if num_samples < total:
        indices = random.sample(range(total), num_samples)
        samples = [dataset[i] for i in indices]
    else:
        samples = [dataset[i] for i in range(total)]
    

    prompts = []
    ground_truths = []
    for sample in samples:
        question = sample.get("problem", sample.get('question', ""))
        if not question:
            continue

        try:
            _, ground_truth = parse_ground_truth(sample, dataset_name)
        except Exception:
            ground_truth = sample.get("answer", "")
            if ground_truth:
                ground_truth = strip_string(str(ground_truth))

        if not ground_truth:
            continue

        msgs = messages_for_math(question)
        prompt_text = build_prompt_text(tokenizer, msgs)
        prompts.append(prompt_text)
        ground_truths.append(ground_truth)

    if not prompts:
        return {f"eval_math/{dataset_name}_accuracy": 0.0}


    if backend is not None and backend.is_enabled() and backend.supports_batch():
        completions = backend.generate(prompts, temperature=temperature, max_new_tokens=max_new_tokens, mode="math")
    else:
        print("falling back")
        was_training = model.training
        model.eval()
        completions = []
        for prompt in tqdm(prompts, desc=f"Math eval ({dataset_name})"):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 0.01),
                top_p=1.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            completion = tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
            completions.append(completion)

            del input_ids, out_ids
            if device == "cuda":
                torch.cuda.empty_cache()

        if was_training:
            model.train()


    correct = 0
    for completion, ground_truth in zip(completions, ground_truths):
        predicted = extract_boxed_answer(completion)
        if not predicted:
            predicted = extract_answer(completion, dataset_name)
    
        predicted = strip_string(predicted)

        if math_equal(predicted, ground_truth):
            correct += 1


    total_evaluated = len(ground_truths)
    
    accuracy = correct / max(1, total_evaluated)

    return {
        f"eval_math/{dataset_name}_accuracy": accuracy,
        f"eval_math/{dataset_name}_correct": correct,
        f"eval_math/{dataset_name}_total": total_evaluated,
    }


# =========================
# Poker evaluation: vs base model
# =========================
@torch.no_grad()
def evaluate_vs_base(
    current_model,
    base_model_adapter_dir: Path,
    tokenizer,
    backend: InferenceBackend,
    num_games: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    hf_hub: Path,
    use_constrained_decoding: bool,
    device: str,
    game_spec: GameSpec,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Evaluate current trained model against the base (untrained) model.
    Loads base model + base adapter each eval (simple, correct; costs time).
    """
    rng = random.Random(int(seed))

    use_vllm_for_base = (
        backend.is_enabled() and backend.supports_batch()
        and hasattr(backend, "has_base_adapter")
        and backend.has_base_adapter()
    )
    if not use_vllm_for_base:
        print("Loading base model via HF", time.time())
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name=Config.MODEL_NAME,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            offload_embedding=False,
            cache_dir=str(hf_hub),
        )
        base_model = FastLanguageModel.get_peft_model(
            base_model,
            r=Config.LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=Config.LORA_ALPHA,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        base_model.load_adapter(str(base_model_adapter_dir), adapter_name="base")
        base_model.set_adapter("base")
        base_model = base_model.to(device)
        base_model.eval()
    else:
        base_model = None
        print("[eval] Using vLLM for base model", time.time())
    

    half = max(1, num_games // 2)
    envs = []
    current_is_p0: List[bool] = []
    env_kwargs = env_kwargs or {}
    for i in range(num_games):
        env = game_spec.make_env(**env_kwargs)
        env.reset(rng.randint(0, 2**31 - 1))
        envs.append(env)
        current_is_p0.append(i < half)

    turn_counts = [0 for _ in range(num_games)]
    current_extraction_failures = 0
    current_total_moves = 0
    while True:
        active = [i for i, e in enumerate(envs) if not e.done]
        if not active:
            break


        current_prompts: List[str] = []
        current_meta: List[Tuple[int, int, List[str], str]] = []

        for i in active:
            env = envs[i]
            pid = env.current_player
            is_current_turn = (pid == 0 and current_is_p0[i]) or (pid == 1 and (not current_is_p0[i]))
            if not is_current_turn:
                continue
            obs = env.observe(pid)
            legal = env.legal_actions()
            msgs = messages_for_game(pid, obs, game_spec)
            current_prompts.append(build_prompt_text(tokenizer, msgs))
            current_meta.append((i, pid, legal, obs))
            if len(current_prompts) >= Config.EVAL_BATCH_SIZE:
                break
        if current_prompts:
            if backend.supports_batch():
                completions = backend.generate(
                    current_prompts,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    game_spec=game_spec,
                    use_guided_choice=use_constrained_decoding,
                )
            else:
                completions = [
                    generate_completion(
                        current_model, 
                        tokenizer, 
                        pid, 
                        obs, 
                        temperature=temperature, 
                        max_new_tokens=max_new_tokens,
                        use_constrained_decoding=use_constrained_decoding,
                        device=device,
                        game_spec=game_spec,
                    )
                    for (_, pid, _legal, obs) in current_meta
                ]
            for j, (i, pid, legal, _obs) in enumerate(current_meta):
                completion = completions[j]
                act = game_spec.extract_action(completion, legal)
                current_total_moves += 1
                if act is None:
                    current_extraction_failures += 1
                    act = rng.choice(legal)
                envs[i].step(act)
                turn_counts[i] += 1

        base_meta: List[Tuple[int, int, List[str], str]] = []
        for i in active:
            env = envs[i]
            if env.done:
                continue
            pid = env.current_player
            is_base_turn = (pid == 0 and (not current_is_p0[i])) or (pid == 1 and current_is_p0[i])
            if not is_base_turn:
                continue
            obs = env.observe(pid)
            legal = env.legal_actions()
            base_meta.append((i, pid, legal, obs))

        if base_meta:
            if use_vllm_for_base:
                base_prompts = [
                     build_prompt_text(tokenizer, messages_for_game(pid, obs, game_spec))
                    for (_, pid, _, obs) in base_meta
                ]
                completions = backend.generate(
                    base_prompts,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    game_spec=game_spec,
                    adapter_name=backend.base_lora_name,
                )
            else:
                completions = [
                    generate_completion(
                        base_model, 
                        tokenizer, 
                        pid, 
                        obs, 
                        temperature=temperature, 
                        max_new_tokens=max_new_tokens,
                        use_constrained_decoding=use_constrained_decoding,
                        device=device,
                        game_spec=game_spec,
                    )
                    for (_, pid, _legal, obs) in base_meta
                ]
            for j, (i, pid, legal, _obs) in enumerate(base_meta):
                completion = completions[j]
                act = game_spec.extract_action(completion, legal)
                if act is None:
                    act = rng.choice(legal)
                envs[i].step(act)
                turn_counts[i] += 1

    wins_current = 0
    invalids = 0
    total_turns = sum(turn_counts)
    for i, env in enumerate(envs):
        invalids += 1 if env.invalid_player is not None else 0
        current_won = (env.rewards[0] > 0) if current_is_p0[i] else (env.rewards[1] > 0)
        wins_current += 1 if current_won else 0

    if base_model is not None:
        del base_model
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "eval/win_rate_vs_base": wins_current / max(1, num_games),
        "eval/invalid_game_rate_vs_base":current_extraction_failures / max(1, current_total_moves),  # ADD THIS
        "eval/turns_per_game_mean_vs_base": total_turns / max(1, num_games),
    }

# =========================
# tau2-bench evaluation
# =========================

def _tau2_harness_dir() -> Path:
    """Return the path to the vendored tau2-bench eval harness."""
    return Path(__file__).resolve().parent / "evals" / "benchmarks" / "tau2_bench_eval"


def _load_tau2_task_ids(tasks_path: Path) -> List[int]:
    """Load tau2 task ids from a tasks.json file.

    The upstream tau2-bench JSON schema may evolve; we try a few common layouts.
    """
    with open(tasks_path, "r") as f:
        data = json.load(f)

    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError(f"Unexpected tasks.json format: {type(data)}")

    ids: List[int] = []
    for idx, t in enumerate(tasks, start=1):
        if isinstance(t, dict):
            for k in ("task_id", "taskId", "id", "task"):  # best-effort
                if k in t:
                    v = t.get(k)
                    if isinstance(v, int):
                        ids.append(int(v))
                        break
                    if isinstance(v, str) and v.strip().isdigit():
                        ids.append(int(v.strip()))
                        break
            else:
                ids.append(idx)
        else:
            ids.append(idx)

    # Keep deterministic order
    ids = sorted(set(ids))
    return ids


def _split_round_robin(items: List[int], n_shards: int) -> List[List[int]]:
    if n_shards <= 1:
        return [list(items)]
    shards: List[List[int]] = [[] for _ in range(n_shards)]
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
        for p, f, save_to in procs:
            rc = p.wait()
            f.close()
            if rc != 0:
                failed.append((save_to, rc))

        t1 = time.time()
        logs[f"eval_tau2/{domain}_time_sec"] = float(t1 - t0)

        if failed:
            print(f"[tau2_eval] One or more shards failed for domain={domain}: {failed}")
            continue

        # Collect and parse results
        simulations_dir = tau2_data_dir / "tau2" / "simulations"
        if not simulations_dir.exists():
            simulations_dir = harness_dir / "data" / "simulations"
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

    return logs
