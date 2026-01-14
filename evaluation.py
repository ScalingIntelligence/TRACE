#!/usr/bin/env python3
"""
Evaluation functions for poker agents and math benchmarks.
"""
import gc
import random
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

    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=Config.MODEL_NAME,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        offload_embedding=True,
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
                if act is None:
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

    del base_model
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "eval/win_rate_vs_base": wins_current / max(1, num_games),
        "eval/invalid_game_rate_vs_base": invalids / max(1, num_games),
        "eval/turns_per_game_mean_vs_base": total_turns / max(1, num_games),
    }
