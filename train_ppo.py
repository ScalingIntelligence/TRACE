#!/usr/bin/env python3
# =========================
# PPO self-play for Kuhn Poker - Main Entry Point
# =========================
"""
Main training script that orchestrates PPO training for Kuhn Poker.
Refactored for clarity with separated concerns:
- config.py: Configuration and environment setup
- kuhn_poker.py: Game environment
- model.py: Neural network architecture
- inference.py: Generation backends
- training.py: PPO training logic
- evaluation.py: Evaluation functions
"""
from unsloth import FastLanguageModel
import os
import random
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import our refactored modules
from config import parse_args, setup_environment, Config
from game_registry import get_game_spec, list_game_names
from model import PolicyWithValueHead
from inference import init_inference_backend, HFLocalBackend
from ppo import (
    JSONLLogger,
    EMA,
    collect_games,
    pad_to_device,
    build_prompt_plus_action,
    logprob_action_tokens,
    values_from_hidden,
)
from evaluation import evaluate_vs_base, evaluate_math, evaluate_tau2_bench

# Optional tracking libs
try:
    import wandb
except Exception:
    wandb = None

try:
    import trackio
except Exception:
    trackio = None


# =========================
# Main execution
# =========================
def main():
    # Parse arguments
    args = parse_args()
    game = args.game
    num_dice = Config.NUM_DICE
    try:
        game_spec = get_game_spec(game)
    except KeyError as e:
        available = ", ".join(list_game_names())
        raise RuntimeError(f"{e}. Available games: {available}") from e

    max_gen_tokens = game_spec.max_gen_tokens
    env_kwargs = {}
    if game_spec.name == "kuhn_poker":
        env_kwargs["num_rounds"] = Config.NUM_ROUNDS
    elif game_spec.name == "liars_dice":
        env_kwargs["num_dice"] = num_dice
    
    # Set dynamic rollout log name
    args.rollout_log = f"rollouts_ppo_{game_spec.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # Setup environment
    env_config = setup_environment(args)
    device = env_config["device"]
    hf_hub = env_config["hf_hub"]
    output_dir_path = env_config["output_dir_path"]
    rollout_log_path = env_config["rollout_log_path"]
    
    print(f"[Setup] Device: {device}")
    print(f"[Setup] Output directory: {output_dir_path}")
    print(f"[Setup] Game: {game_spec.name}")
    print(f"[Setup] Max generation tokens: {max_gen_tokens}")
    if game_spec.action_space:
        print(f"[Setup] Action space size: {len(game_spec.action_space)}")
    if env_kwargs:
        print(f"[Setup] Env kwargs: {env_kwargs}")
    
    # =========================
    # Load model + LoRA
    # =========================
    print(f"[Model] Loading {Config.MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.MODEL_NAME,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        offload_embedding=True,
        cache_dir=str(hf_hub),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=Config.LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=Config.LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastLanguageModel.for_training(model)

    start_iter = 0
    if args.resume:
        import safetensors.torch
        from peft import set_peft_model_state_dict

        resume_path = Path(args.resume)
        print(f"[Resume] Loading checkpoint from {resume_path}")

        adapter_path = resume_path / "policy" / "adapter_model.safetensors"

        if adapter_path.exists():
            adapter_state = safetensors.torch.load_file(str(adapter_path))
            set_peft_model_state_dict(model, adapter_state)
            print(f"[Resume] Loaded LoRA adapter from {adapter_path}")
        else:
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")
        
        start_iter = int(resume_path.name.split("_")[-1]) + 1
        print(f"[Resume] Will start from iteration {start_iter}")

    

    


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    ac = PolicyWithValueHead(model, device=device).to(device)

    if args.resume:
        value_head_path = Path(args.resume) / "value_head.pt"
        if value_head_path.exists():
            ac.value_head.load_state_dict(torch.load(value_head_path, map_location=device))
            print(f"[Resume] Loaded value head from {value_head_path}")
        else:
            print(f"[Resume] Warning: value_head.pt not found, starting with fresh value head")

    
    
    print(f"[Model] Model loaded and wrapped with value head")

        # Save initial base model adapter state for evaluation (only if starting fresh)
    base_model_adapter_dir = output_dir_path / "base_model_adapter"
    if not args.resume:
        base_model_adapter_dir.mkdir(parents=True, exist_ok=True)
        ac.lm.save_pretrained(str(base_model_adapter_dir))
        print(f"[Base Model] Saved initial adapter state to {base_model_adapter_dir}")
    else:
        print(f"[Resume] Using existing base model adapter from {base_model_adapter_dir}")

    # Initialize inference backend
    inference_backend = init_inference_backend(ac.lm, tokenizer, device)
    vllm_adapter_dir = output_dir_path / "vllm_adapter_latest"
    if hasattr(inference_backend, "sync_base_adapter"):
        inference_backend.sync_base_adapter(base_model_adapter_dir)

    

    # =========================
    # wandb init
    # =========================
    if wandb:
        if not os.getenv("WANDB_NAME"):
            os.environ["WANDB_NAME"] = f"qwen3-4b-{game}-spiral-ppo-{int(time.time())}"
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        wandb.init(
            entity="forge_scaling_intelligence_lab",
            project="games",
        )
        print("[wandb] Initialized")
    
    # =========================
    # Setup optimizer and logger
    # =========================
    trainable_params = [p for p in ac.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=Config.LR)
    rollout_logger = JSONLLogger(rollout_log_path)
    
    print(f"[Training] Starting PPO training loop...")
    print(f"[Training] Hyperparameters:")
    print(f"  - Games per iter: {Config.GAMES_PER_ITER}")
    print(f"  - PPO epochs: {Config.PPO_EPOCHS}")
    print(f"  - Mini batch size: {Config.MINI_BATCH_SIZE}")
    print(f"  - Learning rate: {Config.LR}")
    print(f"  - Clip epsilon: {Config.CLIP_EPS}")
    print(f"  - Value function coef: {Config.VF_COEF}")
    
    # =========================
    # Main PPO loop
    # =========================
    global_step = start_iter * 25 if args.resume else 0

    math_eval_data_path = Path(__file__).resolve().parent / "data"

    role_baseline_ema = None
    if Config.USE_ROLE_BASELINE:
        role_baseline_ema = {0: EMA(Config.ROLE_BASELINE_EMA_GAMMA), 1: EMA(Config.ROLE_BASELINE_EMA_GAMMA)}

    for it in tqdm(range(start_iter, 10_000), desc="PPO iters", initial=start_iter, total=10_000):
        t_collect0 = time.time()

        # Sync policy to inference backend
        inference_backend.sync_policy(ac.lm, vllm_adapter_dir)
        if not inference_backend.is_enabled():
            inference_backend = HFLocalBackend(ac.lm, tokenizer, device)

                # Periodic eval (deterministic)
        if it % Config.EVAL_EVERY_ITERS == 0 and it > 0:
            ac.eval()

            inference_backend.sync_policy(ac.lm, vllm_adapter_dir)
            if not inference_backend.is_enabled():
                inference_backend = HFLocalBackend(ac.lm, tokenizer, device)

            
            print(f"[eval {it}] Starting eval vs base model ({Config.EVAL_GAMES} games)")
            t_eval_base0 = time.time()
            eval_logs_base = evaluate_vs_base(
                current_model=ac.lm,
                base_model_adapter_dir=base_model_adapter_dir,
                tokenizer=tokenizer,
                backend=inference_backend,
                num_games=Config.EVAL_GAMES,
                temperature=0.0,  # deterministic
                max_new_tokens=max_gen_tokens,
                seed=20_000 + it,
                hf_hub=hf_hub,
                device=device,
                game_spec=game_spec,
                env_kwargs=env_kwargs,
            )
            t_eval_base1 = time.time()
            win_rate_base = eval_logs_base.get("eval/win_rate_vs_base", 0.0)
            print(f"[eval {it}] vs base: win_rate={win_rate_base:.3f} ({t_eval_base1-t_eval_base0:.1f}s)")

            all_eval_logs = {
                **eval_logs_base,
                "eval/time_vs_base_sec": t_eval_base1 - t_eval_base0,
                "eval/improvement_vs_base": win_rate_base - 0.5,
            }

            print(f"[eval {it}] Summary: vs_base={win_rate_base:.3f}")
            if win_rate_base > 0.5:
                print(f"[eval {it}] ✓ Model is better than base model!")


            if wandb:
                wandb.log(all_eval_logs, step=global_step)


        # Periodic tau2-bench evaluation (sharded across vLLM servers)
        if Config.TAU2_EVAL_EVERY_ITERS and it % Config.TAU2_EVAL_EVERY_ITERS == 0 and it > 0:
            ac.eval()

            inference_backend.sync_policy(ac.lm, vllm_adapter_dir)
            if not inference_backend.is_enabled():
                inference_backend = HFLocalBackend(ac.lm, tokenizer, device)

            print(f"[eval {it}] Starting tau2-bench evaluation...")
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
                verbose=False,
            )
            t_tau2_1 = time.time()

            if tau2_logs:
                tau2_logs["eval_tau2/time_total_sec"] = t_tau2_1 - t_tau2_0
                print(f"[eval {it}] tau2-bench done ({t_tau2_1 - t_tau2_0:.1f}s)")
                if wandb:
                    wandb.log(tau2_logs, step=global_step)
            else:
                print(f"[eval {it}] tau2-bench skipped ({t_tau2_1 - t_tau2_0:.1f}s)")


        # Collect rollouts
        batch, env_metrics = collect_games(
            model=ac.lm,
            tokenizer=tokenizer,
            backend=inference_backend,
            num_games=Config.GAMES_PER_ITER,
            temperature=Config.TEMPERATURE,
            max_new_tokens=max_gen_tokens,
            seed=it,
            logger=rollout_logger,
            device=device,
            role_baseline_ema=role_baseline_ema,
            game_spec=game_spec,
            env_kwargs=env_kwargs,
        )
        t_collect1 = time.time()

        random.shuffle(batch)

        # Pre-tokenize ON CPU
        seqs_cpu = []
        prompt_lens = []
        action_lens = []
        returns_cpu = torch.empty((len(batch),), dtype=torch.float32)

        for i, s in enumerate(batch):
            ids, pL, aL = build_prompt_plus_action(tokenizer, s.prompt_msgs, s.action_str)
            seqs_cpu.append(ids.cpu())
            prompt_lens.append(pL)
            action_lens.append(aL)
            returns_cpu[i] = float(s.ret)

        N = len(seqs_cpu)

        # Compute old_logp + old_v in chunks
        ac.eval()
        old_logp_cpu = torch.empty((N,), dtype=torch.float32)
        old_v_cpu = torch.empty((N,), dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, N, Config.STATS_CHUNK_SIZE):
                mb_idx = list(range(start, min(N, start + Config.STATS_CHUNK_SIZE)))
                mb_seqs = [seqs_cpu[j] for j in mb_idx]
                mb_prompt_lens = [prompt_lens[j] for j in mb_idx]
                mb_action_lens = [action_lens[j] for j in mb_idx]

                mb_ids, mb_attn = pad_to_device(mb_seqs, tokenizer.pad_token_id, device)

                logits, last_h = ac(input_ids=mb_ids, attention_mask=mb_attn)
                mb_old_logp = logprob_action_tokens(logits, mb_ids, mb_prompt_lens, mb_action_lens, normalize_by_len=True)
                mb_old_v = values_from_hidden(last_h, ac.value_head, mb_prompt_lens)

                old_logp_cpu[mb_idx] = mb_old_logp.detach().cpu()
                old_v_cpu[mb_idx] = mb_old_v.detach().cpu()

                del mb_ids, mb_attn, logits, last_h, mb_old_logp, mb_old_v


        adv_cpu = (returns_cpu - old_v_cpu)
        adv_cpu = (adv_cpu - adv_cpu.mean()) / (adv_cpu.std(unbiased=False) + 1e-8)

        # PPO updates
        ac.train()
        idxs = list(range(N))

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        approx_kl_acc = 0.0
        clip_frac_acc = 0.0
        ratio_mean_acc = 0.0
        updates = 0

        for _epoch in range(Config.PPO_EPOCHS):
            random.shuffle(idxs)
            for start in range(0, N, Config.MINI_BATCH_SIZE):
                mb = idxs[start:start + Config.MINI_BATCH_SIZE]
                mb_seqs = [seqs_cpu[i] for i in mb]
                mb_prompt_lens = [prompt_lens[i] for i in mb]
                mb_action_lens = [action_lens[i] for i in mb]

                mb_ids, mb_attn = pad_to_device(mb_seqs, tokenizer.pad_token_id, device)

                mb_returns = returns_cpu[mb].to(device)
                mb_old_logp = old_logp_cpu[mb].to(device)
                mb_adv = adv_cpu[mb].to(device)

                logits, last_h = ac(input_ids=mb_ids, attention_mask=mb_attn)

                new_logp = logprob_action_tokens(logits, mb_ids, mb_prompt_lens, mb_action_lens, normalize_by_len=True)
                new_v = values_from_hidden(last_h, ac.value_head, mb_prompt_lens)

                ratio = torch.exp(new_logp - mb_old_logp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - Config.CLIP_EPS, 1.0 + Config.CLIP_EPS) * mb_adv
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                value_loss = 0.5 * F.mse_loss(new_v, mb_returns)
                loss = policy_loss + Config.VF_COEF * value_loss

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optim.step()

                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - new_logp)
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > Config.CLIP_EPS).float())
                    ratio_mean = torch.mean(ratio)

                global_step += 1
                updates += 1
                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                approx_kl_acc += float(approx_kl.item())
                clip_frac_acc += float(clip_frac.item())
                ratio_mean_acc += float(ratio_mean.item())

                del mb_ids, mb_attn, logits, last_h, new_logp, new_v, ratio, unclipped, clipped, loss


        t_train1 = time.time()

        policy_loss_mean = policy_loss_acc / max(1, updates)
        value_loss_mean = value_loss_acc / max(1, updates)
        avg_return = float(returns_cpu.mean().item())
        abs_return = float(returns_cpu.abs().mean().item())

        logs = {
            "iter": it,
            "global_step": global_step,
            "ppo/policy_loss": policy_loss_mean,
            "ppo/value_loss": value_loss_mean,
            "ppo/avg_return": avg_return,
            "ppo/avg_abs_return": abs_return,
            "ppo/approx_kl": approx_kl_acc / max(1, updates),
            "ppo/clip_frac": clip_frac_acc / max(1, updates),
            "ppo/ratio_mean": ratio_mean_acc / max(1, updates),
            "time/collect_sec": t_collect1 - t_collect0,
            "time/train_sec": t_train1 - t_collect1,
            "time/iter_sec": t_train1 - t_collect0,
            **env_metrics,
        }

        print(
            f"[iter {it}] step={global_step} "
            f"avg_return={avg_return:.3f} win_p0={env_metrics['env/win_rate_p0']:.3f} "
            f"invalid={env_metrics['env/invalid_move_rate']:.3f} "
            f"KL={logs['ppo/approx_kl']:.4f} clip={logs['ppo/clip_frac']:.3f} "
            f"collect={logs['time/collect_sec']:.1f}s train={logs['time/train_sec']:.1f}s"
        )

        if wandb:
            wandb.log(logs, step=global_step)

        # Math evaluation
        if it % Config.MATH_EVAL_EVERY_ITERS == 0 and it > 0:
            ac.eval()
            print(f"[eval {it}] Starting math benchmark evaluation...")

            all_math_logs = {}
            for ds_name in Config.MATH_EVAL_DATASETS:
                math_start = time.time()
                math_logs = evaluate_math(
                    model=ac.lm,
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

                print(f"[eval {it}] {ds_name}: accuracy={acc:.3f} ({math_end-math_start:.1f}s)")
                
                all_math_logs.update(math_logs)
                all_math_logs[f"eval_math/{ds_name}_time_sec"] = math_end - math_start
            
            if wandb:
                wandb.log(all_math_logs, step=global_step)

        # Save checkpoints
        if it % Config.SAVE_EVERY_ITERS == 0:
            ckpt_dir = output_dir_path / f"ppo_ckpt_iter_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ac.lm.save_pretrained(str(ckpt_dir / "policy"))
            tokenizer.save_pretrained(str(ckpt_dir / "policy"))
            torch.save(ac.value_head.state_dict(), str(ckpt_dir / "value_head.pt"))
            print(f"[checkpoint] Saved checkpoint to {ckpt_dir}")

    # Final save
    final_dir = output_dir_path / f"spiral_{game}_ppo_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    ac.lm.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    torch.save(ac.value_head.state_dict(), str(final_dir / "value_head.pt"))
    print(f"[Training] Complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
