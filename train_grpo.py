#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) trainer for game environments.

Implements the rl4rl-style GRPO training loop adapted for the games repo:

  Inner loop (per seed):  Sample `group_size` rollouts from the SAME game seed.
                          All rollouts start from an identical initial state;
                          diversity comes purely from temperature sampling.

  Outer loop (per batch): Iterate over `groups_per_batch` different seeds.

Key differences from the PPO trainer (train_ppo.py):
  - No value head:       Advantages come from group-relative reward centering,
                         not a learned value baseline. This keeps backbone
                         representations general instead of encoding game-state
                         evaluation.
  - Group-relative advantages: "Which of my N attempts at THIS problem was best?"
                         Scale-invariant, works across different reward ranges.
  - Full text training:  No constrained decoding. The model generates freely
                         (including chain-of-thought reasoning). Gradient flows
                         through ALL completion tokens, not just action tokens.
  - Importance sampling: Default loss is -E[ratio * advantage] (pure importance
                         sampling). Optional PPO-style clipping is available.
  - KL penalty:          Optional penalty to prevent catastrophic forgetting.
  - Higher capacity:     Default LoRA rank 32 (vs 4 in PPO), higher LR.

Works with all GameEnv-protocol games in the registry. Supports vLLM inference
backend exactly like the PPO trainer (hot LoRA reload each iteration).

Usage:
    # With vLLM server (recommended):
    VLLM_BASE_URL=http://localhost:8000 python train_grpo.py \\
        --game adversarial_policy --group-size 8 --groups-per-batch 16

    # With HF local generation:
    python train_grpo.py --game kuhn_poker --group-size 4 --groups-per-batch 8
"""
from unsloth import FastLanguageModel

import argparse
import json
import os
import random
import time
import torch
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
from game_registry import get_game_spec, list_game_names, GameSpec
from inference import (
    InferenceBackend,
    HFLocalBackend,
    init_inference_backend,
    messages_for_game,
    build_prompt_text,
    generate_completion,
)
from ppo import (
    JSONLLogger,
    pad_to_device,
    build_prompt_plus_action,
    logprob_action_tokens,
)
from evaluation import evaluate_vs_base, evaluate_math, evaluate_tau2_bench

try:
    import wandb
except Exception:
    wandb = None

try:
    import trackio
except Exception:
    trackio = None


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class GRPOSample:
    """A single turn from a game rollout, tagged with group membership."""
    prompt_msgs: list          # Chat-template messages for this turn
    completion_text: str       # Full model output (thinking + action)
    player_id: int             # Which player produced this action
    reward: float              # Terminal reward for this player
    group_id: int              # Group index (same seed → same group)
    game_id: int               # Unique game identifier


# ============================================================================
# Group-based rollout collection
# ============================================================================

def collect_grpo_rollouts(
    *,
    backend: InferenceBackend,
    tokenizer,
    game_spec: GameSpec,
    groups_per_batch: int,
    group_size: int,
    temperature: float,
    max_new_tokens: int,
    base_seed: int,
    logger: JSONLLogger,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[GRPOSample], Dict[str, float]]:
    """
    Collect rollouts with the GRPO inner/outer loop structure.

    Creates `groups_per_batch` groups, each containing `group_size` games
    played from the SAME random seed.  Within a group, games differ only
    because of temperature sampling — the model explores different strategies
    from an identical starting position.

    Returns:
        samples:  List of GRPOSample with group membership tagged.
        metrics:  Dict of collection statistics.
    """
    rng = random.Random(base_seed)
    env_kwargs = env_kwargs or {}

    # ---- Outer loop: one seed per group ----
    group_seeds = [rng.randint(0, 2**31 - 1) for _ in range(groups_per_batch)]
    total_games = groups_per_batch * group_size

    samples: List[GRPOSample] = []
    invalid_games = 0
    total_turns = 0
    extraction_failures = 0
    p0_wins = 0
    response_lengths: List[int] = []

    if backend.supports_batch():
        # --- Batch mode (vLLM) ---
        envs = []
        g_ids: List[int] = []        # group_id per env
        gm_ids: List[int] = []       # game_id per env
        episode_steps: List[List[Tuple[list, Optional[str], int, str]]] = []

        for g_idx, g_seed in enumerate(group_seeds):
            for s_idx in range(group_size):
                env = game_spec.make_env(**env_kwargs)
                env.reset(g_seed)  # Same seed within group!
                envs.append(env)
                g_ids.append(g_idx)
                gm_ids.append(base_seed * 1_000_000 + g_idx * 1000 + s_idx)
                episode_steps.append([])

        # Play all games in parallel
        while True:
            active = [i for i, e in enumerate(envs) if not e.done]
            if not active:
                break

            prompts: List[str] = []
            meta: List[Tuple[int, int, str, List[str], list]] = []

            for i in active:
                env = envs[i]
                pid = env.current_player
                obs = env.observe(pid)
                legal = env.legal_actions()
                msgs = messages_for_game(pid, obs, game_spec)
                prompts.append(build_prompt_text(tokenizer, msgs))
                meta.append((i, pid, obs, legal, msgs))

            t0 = time.time()
            completions = backend.generate(
                prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                game_spec=game_spec,
            )
            t1 = time.time()

            for j, (i, pid, obs, legal, msgs) in enumerate(meta):
                completion = completions[j]
                act = game_spec.extract_action(completion, legal)
                if act is None:
                    extraction_failures += 1

                response_lengths.append(len(completion))
                episode_steps[i].append((msgs, act, pid, completion))
                envs[i].step(act)
                total_turns += 1

                logger.log({
                    "type": "step",
                    "game_id": gm_ids[i],
                    "group_id": g_ids[i],
                    "player_id": pid,
                    "legal_actions": legal,
                    "action": act,
                    "completion": completion[:500],
                    "illegal_move": act is None,
                    "timestamp": time.time(),
                })

        # Convert finished games to samples
        for i, env in enumerate(envs):
            invalid_games += 1 if env.invalid_player is not None else 0
            p0_wins += 1 if env.rewards.get(0, 0.0) > 0 else 0

            logger.log({
                "type": "game_end",
                "game": game_spec.name,
                "game_id": gm_ids[i],
                "group_id": g_ids[i],
                "rewards": env.rewards,
                "invalid_player": env.invalid_player,
                "timestamp": time.time(),
            })

            for msgs, act, pid, completion in episode_steps[i]:
                samples.append(GRPOSample(
                    prompt_msgs=msgs,
                    completion_text=completion,
                    player_id=pid,
                    reward=float(env.rewards.get(pid, 0.0)),
                    group_id=g_ids[i],
                    game_id=gm_ids[i],
                ))

    else:
        # --- Sequential mode (HF local) ---
        for g_idx, g_seed in enumerate(group_seeds):
            for s_idx in range(group_size):
                game_id = base_seed * 1_000_000 + g_idx * 1000 + s_idx
                env = game_spec.make_env(**env_kwargs)
                env.reset(g_seed)

                ep_steps: List[Tuple[list, Optional[str], int, str]] = []
                while not env.done:
                    pid = env.current_player
                    obs = env.observe(pid)
                    legal = env.legal_actions()

                    completion = generate_completion(
                        backend.model,
                        backend.tokenizer,
                        pid,
                        obs,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        device=backend.device,
                        game_spec=game_spec,
                    )

                    act = game_spec.extract_action(completion, legal)
                    if act is None:
                        extraction_failures += 1

                    response_lengths.append(len(completion))
                    ep_steps.append((messages_for_game(pid, obs, game_spec), act, pid, completion))
                    env.step(act)
                    total_turns += 1

                invalid_games += 1 if env.invalid_player is not None else 0
                p0_wins += 1 if env.rewards.get(0, 0.0) > 0 else 0

                for msgs, act, pid, completion in ep_steps:
                    samples.append(GRPOSample(
                        prompt_msgs=msgs,
                        completion_text=completion,
                        player_id=pid,
                        reward=float(env.rewards.get(pid, 0.0)),
                        group_id=g_idx,
                        game_id=game_id,
                    ))

    # Metrics
    resp_len_mean = (sum(response_lengths) / len(response_lengths)) if response_lengths else 0.0

    metrics = {
        "env/total_games": total_games,
        "env/total_samples": len(samples),
        "env/invalid_move_rate": extraction_failures / max(1, total_turns),
        "env/turns_per_game_mean": total_turns / max(1, total_games),
        "env/win_rate_p0": p0_wins / max(1, total_games),
        "env/response_len_chars_mean": resp_len_mean,
    }
    return samples, metrics


# ============================================================================
# GRPO advantage computation
# ============================================================================

def compute_group_advantages(samples: List[GRPOSample]) -> torch.Tensor:
    """
    Compute group-relative advantages — the core GRPO mechanism.

    For each (group_id, player_id) pair, center rewards around the group mean:

        advantage_i = reward_i - mean(rewards in my group)

    This is a scale-invariant, comparative signal: "was this attempt better or
    worse than the average attempt at the same problem?"  It works identically
    regardless of whether rewards are {-1, +1} or continuous.

    All turns from the same game get the same advantage (since they all
    contribute to the same terminal reward).

    For 2-player games, advantages are computed separately per player within
    each group, so player 0 and player 1 have independent baselines.
    """
    advantages = torch.zeros(len(samples), dtype=torch.float32)

    # Collect per-game rewards, grouped by (group_id, player_id)
    group_game_rewards: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
    for s in samples:
        key = (s.group_id, s.player_id)
        group_game_rewards[key][s.game_id] = s.reward

    # Compute mean reward per group
    group_means: Dict[Tuple[int, int], float] = {}
    for key, game_rewards in group_game_rewards.items():
        rewards = list(game_rewards.values())
        group_means[key] = sum(rewards) / len(rewards)

    # Assign centered advantages
    for i, s in enumerate(samples):
        key = (s.group_id, s.player_id)
        advantages[i] = s.reward - group_means[key]

    return advantages


# ============================================================================
# Argument parsing
# ============================================================================

def parse_grpo_args():
    parser = argparse.ArgumentParser(
        description="GRPO trainer for game environments (rl4rl-style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Game and model --
    parser.add_argument("--game", type=str, default="kuhn_poker",
        help="Game to train on (from game_registry)")
    parser.add_argument("--model", type=str, default=None,
        help="HuggingFace model name (default: Config.MODEL_NAME)")

    # -- GRPO structure (inner/outer loop) --
    parser.add_argument("--group-size", type=int, default=8,
        help="Inner loop: rollouts per seed. More = cleaner advantage signal.")
    parser.add_argument("--groups-per-batch", type=int, default=16,
        help="Outer loop: different seeds per iteration. More = more diversity.")

    # -- LoRA --
    parser.add_argument("--lora-rank", type=int, default=32,
        help="LoRA rank (higher = more capacity for reasoning changes)")
    parser.add_argument("--lora-alpha", type=int, default=16,
        help="LoRA alpha scaling factor")

    # -- Optimization --
    parser.add_argument("--lr", type=float, default=1e-5,
        help="Learning rate (higher than PPO for meaningful updates)")
    parser.add_argument("--epochs", type=int, default=1,
        help="Training epochs per iteration over collected samples")
    parser.add_argument("--mini-batch-size", type=int, default=4,
        help="Mini-batch size for gradient updates")
    parser.add_argument("--stats-chunk-size", type=int, default=2,
        help="Chunk size for computing old logprobs (lower = less VRAM)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")

    # -- GRPO loss function --
    parser.add_argument("--use-clipping", action="store_true",
        help="Use clipped surrogate (like PPO). Default: pure importance sampling.")
    parser.add_argument("--clip-eps", type=float, default=0.2,
        help="Clipping epsilon (only used with --use-clipping)")
    parser.add_argument("--kl-coef", type=float, default=0.01,
        help="KL penalty coefficient against rollout policy (0 to disable)")

    # -- Generation --
    parser.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature for rollouts")
    parser.add_argument("--normalize-by-len", action="store_true",
        help="Normalize logprobs by action length (NOT recommended per rl4rl analysis)")

    # -- Checkpointing and evaluation --
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--rollout-log", type=str, default="selfplay_rollouts_grpo.jsonl",
        help="Filename for rollout logs")
    parser.add_argument("--save-every", type=int, default=10,
        help="Save checkpoint every N iterations")
    parser.add_argument("--eval-every", type=int, default=50,
        help="Eval vs base model every N iterations")
    parser.add_argument("--eval-games", type=int, default=50,
        help="Number of games for eval vs base")
    parser.add_argument("--math-eval-every", type=int, default=100,
        help="Math benchmark eval every N iterations (0 to disable)")
    parser.add_argument("--tau2-eval-every", type=int, default=0,
        help="Tau2-bench eval every N iterations (0 to disable)")

    # -- Root directory (passed through to setup_environment) --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ============================================================================
# Main training loop
# ============================================================================

def main():
    args = parse_grpo_args()

    # ---- Game setup ----
    game = args.game
    try:
        game_spec = get_game_spec(game)
    except KeyError as e:
        available = ", ".join(list_game_names())
        raise RuntimeError(f"{e}. Available games: {available}") from e

    max_gen_tokens = game_spec.max_gen_tokens
    env_kwargs: Dict[str, Any] = {}
    if game_spec.name == "kuhn_poker":
        env_kwargs["num_rounds"] = Config.NUM_ROUNDS
    elif game_spec.name == "liars_dice":
        env_kwargs["num_dice"] = Config.NUM_DICE

    # ---- Environment setup (reuse existing config infrastructure) ----
    env_config = setup_environment(args)
    device = env_config["device"]
    hf_hub = env_config["hf_hub"]
    output_dir_path = env_config["output_dir_path"]
    rollout_log_path = env_config["rollout_log_path"]

    total_games_per_iter = args.group_size * args.groups_per_batch

    print("=" * 60)
    print("GRPO TRAINER")
    print("=" * 60)
    print(f"  Game:                {game_spec.name}")
    print(f"  Group size (inner):  {args.group_size}")
    print(f"  Groups/batch (outer):{args.groups_per_batch}")
    print(f"  Total games/iter:    {total_games_per_iter}")
    print(f"  LoRA rank:           {args.lora_rank}")
    print(f"  LoRA alpha:          {args.lora_alpha}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  KL coefficient:      {args.kl_coef}")
    print(f"  Use clipping:        {args.use_clipping}")
    print(f"  Normalize by len:    {args.normalize_by_len}")
    print(f"  Temperature:         {args.temperature}")
    print(f"  Max gen tokens:      {max_gen_tokens}")
    print(f"  Device:              {device}")
    print(f"  Output dir:          {output_dir_path}")
    print("=" * 60)

    # ---- Load model + LoRA (higher rank than PPO) ----
    model_name = args.model or Config.MODEL_NAME
    print(f"[Model] Loading {model_name} with LoRA rank={args.lora_rank}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        offload_embedding=True,
        cache_dir=str(hf_hub),
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
    FastLanguageModel.for_training(model)

    # ---- Resume from checkpoint ----
    start_iter = 0
    if args.resume:
        import safetensors.torch
        from peft import set_peft_model_state_dict

        resume_path = Path(args.resume)
        print(f"[Resume] Loading from {resume_path}")

        adapter_path = resume_path / "adapter_model.safetensors"
        if adapter_path.exists():
            state = safetensors.torch.load_file(str(adapter_path))
            set_peft_model_state_dict(model, state)
            print(f"[Resume] Loaded adapter from {adapter_path}")
        else:
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        # Extract iteration number from directory name
        try:
            start_iter = int(resume_path.name.split("_")[-2]) + 1
        except (ValueError, IndexError):
            start_iter = 0
        print(f"[Resume] Starting from iteration {start_iter}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    print(f"[Model] Loaded — NO value head (GRPO style)")

    # ---- Save initial adapter for eval baseline ----
    base_model_adapter_dir = output_dir_path / "base_model_adapter_grpo"
    if not args.resume:
        base_model_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(base_model_adapter_dir))
        print(f"[Base] Saved initial adapter to {base_model_adapter_dir}")

    # ---- Initialize inference backend (vLLM or HF local) ----
    inference_backend = init_inference_backend(model, tokenizer, device)
    vllm_adapter_dir = output_dir_path / "vllm_adapter_latest_grpo"
    if hasattr(inference_backend, "sync_base_adapter"):
        inference_backend.sync_base_adapter(base_model_adapter_dir)

    # ---- wandb ----
    if wandb:
        if not os.getenv("WANDB_NAME"):
            os.environ["WANDB_NAME"] = f"grpo-{game}-{int(time.time())}"
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        wandb.init(entity="forge_scaling_intelligence_lab", project="games")
        wandb.config.update({
            "trainer": "grpo",
            "game": game,
            "group_size": args.group_size,
            "groups_per_batch": args.groups_per_batch,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "kl_coef": args.kl_coef,
            "use_clipping": args.use_clipping,
            "temperature": args.temperature,
        })
        print("[wandb] Initialized")

    # ---- Optimizer (LoRA params only, NO value head) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    rollout_logger = JSONLLogger(rollout_log_path)

    print(f"[GRPO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"[GRPO] Starting training loop...")

    global_step = start_iter * 25 if args.resume else 0
    math_eval_data_path = Path(__file__).resolve().parent / "data"

    # ======================================================================
    # Main GRPO loop
    # ======================================================================
    for it in tqdm(range(start_iter, 10_000), desc="GRPO iters", initial=start_iter, total=10_000):
        t_iter_start = time.time()

        # ---- 1. Sync policy to vLLM ----
        inference_backend.sync_policy(model, vllm_adapter_dir)
        if not inference_backend.is_enabled():
            inference_backend = HFLocalBackend(model, tokenizer, device)

        # ---- 2. Periodic evaluation ----
        if args.eval_every and it % args.eval_every == 0 and it > 0:
            model.eval()
            inference_backend.sync_policy(model, vllm_adapter_dir)
            if not inference_backend.is_enabled():
                inference_backend = HFLocalBackend(model, tokenizer, device)

            print(f"\n[eval {it}] Starting eval vs base ({args.eval_games} games)")
            t_eval0 = time.time()
            eval_logs = evaluate_vs_base(
                current_model=model,
                base_model_adapter_dir=base_model_adapter_dir,
                tokenizer=tokenizer,
                backend=inference_backend,
                num_games=args.eval_games,
                temperature=0.0,
                max_new_tokens=max_gen_tokens,
                seed=20_000 + it,
                hf_hub=hf_hub,
                device=device,
                game_spec=game_spec,
                env_kwargs=env_kwargs,
            )
            t_eval1 = time.time()
            wr = eval_logs.get("eval/win_rate_vs_base", 0.0)
            print(f"[eval {it}] win_rate={wr:.3f} ({t_eval1 - t_eval0:.1f}s)")
            if wr > 0.5:
                print(f"[eval {it}] Model is better than base!")

            if wandb:
                wandb.log(eval_logs, step=global_step)

        # tau2-bench evaluation
        if args.tau2_eval_every and it % args.tau2_eval_every == 0 and it > 0:
            model.eval()
            inference_backend.sync_policy(model, vllm_adapter_dir)
            if not inference_backend.is_enabled():
                inference_backend = HFLocalBackend(model, tokenizer, device)

            print(f"\n[eval {it}] Starting tau2-bench evaluation...")
            t_tau2_0 = time.time()
            tau2_logs = evaluate_tau2_bench(
                backend=inference_backend,
                output_dir=output_dir_path,
                iteration=it,
                domains=list(Config.TAU2_EVAL_DOMAINS),
                num_tasks=Config.TAU2_EVAL_NUM_TASKS,
                num_trials=Config.TAU2_EVAL_NUM_TRIALS,
                max_concurrency_per_shard=Config.TAU2_EVAL_MAX_CONCURRENCY_PER_SHARD,
                seed=Config.TAU2_EVAL_SEED,
            )
            t_tau2_1 = time.time()
            if tau2_logs:
                tau2_logs["eval_tau2/time_total_sec"] = t_tau2_1 - t_tau2_0
                print(f"[eval {it}] tau2-bench done ({t_tau2_1 - t_tau2_0:.1f}s)")
                if wandb:
                    wandb.log(tau2_logs, step=global_step)

        # ---- 3. Collect group-based rollouts ----
        t_collect0 = time.time()
        samples, env_metrics = collect_grpo_rollouts(
            backend=inference_backend,
            tokenizer=tokenizer,
            game_spec=game_spec,
            groups_per_batch=args.groups_per_batch,
            group_size=args.group_size,
            temperature=args.temperature,
            max_new_tokens=max_gen_tokens,
            base_seed=it,
            logger=rollout_logger,
            env_kwargs=env_kwargs,
        )
        t_collect1 = time.time()

        if not samples:
            print(f"[iter {it}] No samples collected, skipping")
            continue

        # ---- 4. Compute group-relative advantages ----
        advantages = compute_group_advantages(samples)

        # ---- 5. Pre-tokenize all samples ----
        seqs_cpu: List[torch.Tensor] = []
        prompt_lens: List[int] = []
        action_lens: List[int] = []

        for s in samples:
            ids, pL, aL = build_prompt_plus_action(tokenizer, s.prompt_msgs, s.completion_text)
            seqs_cpu.append(ids.cpu())
            prompt_lens.append(pL)
            action_lens.append(aL)

        N = len(seqs_cpu)

        # ---- 6. Compute old_logp (no value head needed!) ----
        model.eval()
        old_logp_cpu = torch.empty(N, dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, N, args.stats_chunk_size):
                end = min(N, start + args.stats_chunk_size)
                mb_idx = list(range(start, end))
                mb_seqs = [seqs_cpu[j] for j in mb_idx]
                mb_pl = [prompt_lens[j] for j in mb_idx]
                mb_al = [action_lens[j] for j in mb_idx]

                mb_ids, mb_attn = pad_to_device(mb_seqs, tokenizer.pad_token_id, device)

                with autocast_ctx(device):
                    outputs = model(
                        input_ids=mb_ids,
                        attention_mask=mb_attn,
                        use_cache=False,
                    )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                mb_old_logp = logprob_action_tokens(
                    logits, mb_ids, mb_pl, mb_al,
                    normalize_by_len=args.normalize_by_len,
                )
                old_logp_cpu[mb_idx] = mb_old_logp.detach().cpu()

                del mb_ids, mb_attn, logits, outputs, mb_old_logp

        # ---- 7. GRPO training ----
        model.train()
        idxs = list(range(N))

        policy_loss_acc = 0.0
        kl_loss_acc = 0.0
        approx_kl_acc = 0.0
        clip_frac_acc = 0.0
        ratio_mean_acc = 0.0
        updates = 0

        for _epoch in range(args.epochs):
            random.shuffle(idxs)
            for start in range(0, N, args.mini_batch_size):
                mb = idxs[start : start + args.mini_batch_size]
                if not mb:
                    continue

                mb_seqs = [seqs_cpu[i] for i in mb]
                mb_pl = [prompt_lens[i] for i in mb]
                mb_al = [action_lens[i] for i in mb]

                mb_ids, mb_attn = pad_to_device(mb_seqs, tokenizer.pad_token_id, device)
                mb_old_logp = old_logp_cpu[mb].to(device)
                mb_adv = advantages[mb].to(device)

                # Forward pass
                outputs = model(
                    input_ids=mb_ids,
                    attention_mask=mb_attn,
                    use_cache=False,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                new_logp = logprob_action_tokens(
                    logits, mb_ids, mb_pl, mb_al,
                    normalize_by_len=args.normalize_by_len,
                )

                # Importance sampling ratio
                ratio = torch.exp(new_logp - mb_old_logp)

                # Policy loss
                if args.use_clipping:
                    # PPO-style clipped surrogate
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps
                    ) * mb_adv
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                else:
                    # Pure importance sampling (rl4rl default)
                    policy_loss = -torch.mean(ratio * mb_adv)

                # KL penalty (against rollout policy)
                kl_loss = torch.tensor(0.0, device=device)
                if args.kl_coef > 0:
                    # KL ≈ E[log(π_new / π_old)] = E[new_logp - old_logp]
                    kl_loss = args.kl_coef * torch.mean(new_logp - mb_old_logp)

                loss = policy_loss + kl_loss

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optim.step()

                # Tracking (no grad)
                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - new_logp)
                    clip_frac = torch.mean(
                        (torch.abs(ratio - 1.0) > args.clip_eps).float()
                    )
                    ratio_mean_val = torch.mean(ratio)

                global_step += 1
                updates += 1
                policy_loss_acc += float(policy_loss.item())
                kl_loss_acc += float(kl_loss.item())
                approx_kl_acc += float(approx_kl.item())
                clip_frac_acc += float(clip_frac.item())
                ratio_mean_acc += float(ratio_mean_val.item())

                del mb_ids, mb_attn, logits, outputs, new_logp, ratio, loss

        t_train1 = time.time()

        # ---- 8. Logging ----
        avg_advantage = float(advantages.mean().item())
        avg_abs_advantage = float(advantages.abs().mean().item())
        all_rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
        avg_reward = float(all_rewards.mean().item())

        logs = {
            "iter": it,
            "global_step": global_step,
            "grpo/policy_loss": policy_loss_acc / max(1, updates),
            "grpo/kl_loss": kl_loss_acc / max(1, updates),
            "grpo/approx_kl": approx_kl_acc / max(1, updates),
            "grpo/clip_frac": clip_frac_acc / max(1, updates),
            "grpo/ratio_mean": ratio_mean_acc / max(1, updates),
            "grpo/avg_reward": avg_reward,
            "grpo/avg_advantage": avg_advantage,
            "grpo/avg_abs_advantage": avg_abs_advantage,
            "grpo/num_samples": N,
            "time/collect_sec": t_collect1 - t_collect0,
            "time/train_sec": t_train1 - t_collect1,
            "time/iter_sec": t_train1 - t_iter_start,
            **env_metrics,
        }

        print(
            f"[iter {it}] step={global_step} "
            f"reward={avg_reward:.3f} abs_adv={avg_abs_advantage:.3f} "
            f"invalid={env_metrics['env/invalid_move_rate']:.3f} "
            f"KL={logs['grpo/approx_kl']:.4f} "
            f"collect={logs['time/collect_sec']:.1f}s train={logs['time/train_sec']:.1f}s"
        )

        if wandb:
            wandb.log(logs, step=global_step)

        # ---- Math evaluation ----
        if args.math_eval_every and it % args.math_eval_every == 0 and it > 0:
            model.eval()
            all_math_logs: Dict[str, float] = {}
            for ds_name in Config.MATH_EVAL_DATASETS:
                math_start = time.time()
                math_logs = evaluate_math(
                    model=model,
                    tokenizer=tokenizer,
                    data_path=math_eval_data_path,
                    dataset_name=ds_name,
                    num_samples=Config.MATH_EVAL_SAMPLES,
                    temperature=0.0,
                    max_new_tokens=Config.MAX_TOKENS_MATH_EVAL,
                    backend=inference_backend,
                    device=device,
                )
                math_end = time.time()
                acc = math_logs.get(f"eval_math/{ds_name}_accuracy", 0.0)
                print(f"[eval {it}] {ds_name}: accuracy={acc:.3f} ({math_end - math_start:.1f}s)")
                all_math_logs.update(math_logs)
            if wandb:
                wandb.log(all_math_logs, step=global_step)

        # ---- 9. Save checkpoint ----
        if it % args.save_every == 0:
            ckpt_dir = output_dir_path / f"grpo_ckpt_iter_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"[checkpoint] Saved to {ckpt_dir}")

    # ---- Final save ----
    final_dir = output_dir_path / f"grpo_{game}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[GRPO] Training complete! Saved to {final_dir}")


if __name__ == "__main__":
    main()
