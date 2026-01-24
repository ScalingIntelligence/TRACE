#!/usr/bin/env python3
"""
Evaluate a PPO LoRA checkpoint vs an untrained base model using
two vLLM OpenAI-compatible servers.
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple

import requests
from transformers import AutoTokenizer

from config import Config
from game_registry import GameSpec, get_game_spec
from inference import VLLMServerBackend, build_prompt_text, messages_for_game, normalize_vllm_openai_base_url


def _build_liars_dice_system_prompt(num_dice: int) -> str:
    total_dice = num_dice * 2
    return (
        f"You are playing Liar's Dice with {num_dice} dice per player ({total_dice} total in play).\n"
        + ("" if Config.ENABLE_THINKING else "Respond with EXACTLY ONE action and NOTHING ELSE.\n")
        + "Valid actions: [bid: quantity, face] or [call]\n"
        + "Examples: [bid: 3, 4] or [call]\n"
        + ("" if Config.ENABLE_THINKING else "Do not add any whitespace, punctuation, explanation, or extra text.\n")
    )


def _make_liars_dice_spec(num_dice: int, max_new_tokens: int) -> GameSpec:
    base_spec = get_game_spec("liars_dice")
    return GameSpec(
        name=base_spec.name,
        make_env=base_spec.make_env,
        extract_action=base_spec.extract_action,
        action_space=base_spec.action_space,
        stop_sequences=base_spec.stop_sequences,
        system_prompt=_build_liars_dice_system_prompt(num_dice),
        max_gen_tokens=max_new_tokens,
    )


def _load_lora_adapter(
    *,
    base_url: str,
    lora_name: str,
    lora_path: Path,
    api_key: Optional[str],
    timeout_s: float,
) -> None:
    url = normalize_vllm_openai_base_url(base_url)
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"lora_name": lora_name, "lora_path": str(lora_path)}

    # Best-effort unload to avoid stale adapters.
    try:
        requests.post(
            url + "/unload_lora_adapter",
            headers=headers,
            json={"lora_name": lora_name},
            timeout=timeout_s,
        )
    except Exception:
        pass

    resp = requests.post(url + "/load_lora_adapter", headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to load LoRA '{lora_name}' from '{lora_path}': "
            f"{resp.status_code} {resp.text[:300]}"
        )


def _evaluate_game(
    *,
    policy_backend: VLLMServerBackend,
    base_backend: VLLMServerBackend,
    tokenizer,
    policy_lora_name: str,
    game_spec: GameSpec,
    num_games: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    batch_size: int,
    env_kwargs: Optional[Dict[str, Any]],
    print_interval: int,
    turn_log_fh: Optional[IO[str]],
) -> Dict[str, Any]:
    rng = random.Random(int(seed))

    half = max(1, num_games // 2)
    env_kwargs = env_kwargs or {}
    envs = []
    policy_is_p0: List[bool] = []
    game_seeds: List[int] = []
    for i in range(num_games):
        if env_kwargs:
            env = game_spec.make_env(**env_kwargs)
        else:
            env = game_spec.make_env()
        game_seed = rng.randint(0, 2**31 - 1)
        env.reset(game_seed)
        envs.append(env)
        policy_is_p0.append(i < half)
        game_seeds.append(game_seed)

    turn_counts = [0 for _ in range(num_games)]
    policy_extraction_failures = 0
    policy_total_moves = 0
    base_extraction_failures = 0
    base_total_moves = 0

    start = time.time()
    next_report = print_interval if print_interval > 0 else None

    while True:
        active = [i for i, e in enumerate(envs) if not e.done]
        if not active:
            break

        policy_prompts: List[str] = []
        policy_meta: List[Tuple[int, int, List[str], str, str]] = []
        for i in active:
            env = envs[i]
            pid = env.current_player
            is_policy_turn = (pid == 0 and policy_is_p0[i]) or (pid == 1 and (not policy_is_p0[i]))
            if not is_policy_turn:
                continue
            obs = env.observe(pid)
            legal = env.legal_actions()
            msgs = messages_for_game(pid, obs, game_spec)
            prompt_text = build_prompt_text(tokenizer, msgs)
            policy_prompts.append(prompt_text)
            policy_meta.append((i, pid, legal, obs, prompt_text))
            if len(policy_prompts) >= batch_size:
                break

        if policy_prompts:
            completions = policy_backend.generate(
                policy_prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                game_spec=game_spec,
                adapter_name=policy_lora_name,
            )
            for j, (i, pid, legal, obs, prompt_text) in enumerate(policy_meta):
                completion = completions[j]
                extracted = game_spec.extract_action(completion, legal)
                policy_total_moves += 1
                if extracted is None:
                    policy_extraction_failures += 1
                    act = rng.choice(legal)
                else:
                    act = extracted
                if turn_log_fh is not None:
                    record = {
                        "game": game_spec.name,
                        "game_id": i,
                        "game_seed": game_seeds[i],
                        "turn_index": turn_counts[i],
                        "player_id": pid,
                        "model_role": "policy",
                        "policy_is_p0": policy_is_p0[i],
                        "observation": obs,
                        "prompt": prompt_text,
                        "legal_actions": legal,
                        "completion": completion,
                        "extracted_action": extracted,
                        "action": act,
                        "used_fallback": extracted is None,
                    }
                    turn_log_fh.write(json.dumps(record) + "\n")
                envs[i].step(act)
                turn_counts[i] += 1

        base_prompts: List[str] = []
        base_meta: List[Tuple[int, int, List[str], str, str]] = []
        for i in active:
            env = envs[i]
            if env.done:
                continue
            pid = env.current_player
            is_base_turn = (pid == 0 and (not policy_is_p0[i])) or (pid == 1 and policy_is_p0[i])
            if not is_base_turn:
                continue
            obs = env.observe(pid)
            legal = env.legal_actions()
            msgs = messages_for_game(pid, obs, game_spec)
            prompt_text = build_prompt_text(tokenizer, msgs)
            base_prompts.append(prompt_text)
            base_meta.append((i, pid, legal, obs, prompt_text))
            if len(base_prompts) >= batch_size:
                break

        if base_prompts:
            completions = base_backend.generate(
                base_prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                game_spec=game_spec,
            )
            for j, (i, pid, legal, obs, prompt_text) in enumerate(base_meta):
                completion = completions[j]
                extracted = game_spec.extract_action(completion, legal)
                base_total_moves += 1
                if extracted is None:
                    base_extraction_failures += 1
                    act = rng.choice(legal)
                else:
                    act = extracted
                if turn_log_fh is not None:
                    record = {
                        "game": game_spec.name,
                        "game_id": i,
                        "game_seed": game_seeds[i],
                        "turn_index": turn_counts[i],
                        "player_id": pid,
                        "model_role": "base",
                        "policy_is_p0": policy_is_p0[i],
                        "observation": obs,
                        "prompt": prompt_text,
                        "legal_actions": legal,
                        "completion": completion,
                        "extracted_action": extracted,
                        "action": act,
                        "used_fallback": extracted is None,
                    }
                    turn_log_fh.write(json.dumps(record) + "\n")
                envs[i].step(act)
                turn_counts[i] += 1

        if next_report is not None:
            done_count = sum(1 for e in envs if e.done)
            if done_count >= next_report:
                elapsed = time.time() - start
                print(f"[eval] {done_count}/{num_games} games finished ({elapsed:.1f}s elapsed)")
                next_report += print_interval

    wins_policy = 0
    wins_policy_as_p0 = 0
    wins_policy_as_p1 = 0
    for i, env in enumerate(envs):
        policy_won = (env.rewards[0] > 0) if policy_is_p0[i] else (env.rewards[1] > 0)
        if policy_won:
            wins_policy += 1
            if policy_is_p0[i]:
                wins_policy_as_p0 += 1
            else:
                wins_policy_as_p1 += 1

    total_turns = sum(turn_counts)
    elapsed_s = time.time() - start

    return {
        "game": game_spec.name,
        "env_kwargs": env_kwargs,
        "num_games": num_games,
        "policy_wins": wins_policy,
        "base_wins": num_games - wins_policy,
        "policy_win_rate": wins_policy / max(1, num_games),
        "policy_wins_as_p0": wins_policy_as_p0,
        "policy_wins_as_p1": wins_policy_as_p1,
        "policy_extraction_fail_rate": policy_extraction_failures / max(1, policy_total_moves),
        "base_extraction_fail_rate": base_extraction_failures / max(1, base_total_moves),
        "turns_per_game_mean": total_turns / max(1, num_games),
        "elapsed_s": elapsed_s,
    }


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_adapter = repo_root / "outputs" / "ppo_ckpt_iter_210" / "policy"
    parser = argparse.ArgumentParser(description="Evaluate PPO vs base using vLLM servers.")
    parser.add_argument(
        "--game",
        type=str,
        default="liars_dice",
        help="Game spec name (e.g., liars_dice or openspiel_liars_poker).",
    )
    parser.add_argument("--policy-url", default="http://localhost:8041", help="Policy vLLM base URL.")
    parser.add_argument("--base-url", default="http://localhost:8042", help="Base vLLM base URL.")
    parser.add_argument(
        "--policy-adapter",
        type=str,
        default=str(default_adapter),
        help="Path to PPO LoRA adapter directory (used if loading adapter into vLLM).",
    )
    parser.add_argument(
        "--policy-lora-name",
        type=str,
        default="ppo_policy",
        help="LoRA name used by vLLM for the PPO adapter.",
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default=Config.MODEL_NAME,
        help="Base model name served by vLLM.",
    )
    parser.add_argument("--num-games", type=int, default=200, help="Number of games to play.")
    parser.add_argument(
        "--num-dice",
        type=int,
        default=Config.NUM_DICE,
        help="Dice per player (liars_dice only).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max tokens per move (defaults to the game config).",
    )
    parser.add_argument("--batch-size", type=int, default=Config.EVAL_BATCH_SIZE, help="Batch size per side.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="vLLM request timeout seconds.")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key for vLLM server.")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path (defaults to policy adapter dir).",
    )
    parser.add_argument(
        "--no-load-policy-lora",
        action="store_true",
        help="Skip loading the policy LoRA via /load_lora_adapter.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--turn-log",
        type=str,
        default=None,
        help="Optional JSONL path to log every turn's model output.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=20,
        help="Progress print interval in completed games (0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    adapter_path = Path(args.policy_adapter).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else adapter_path

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    env_kwargs: Dict[str, Any] = {}
    if args.game == "liars_dice":
        if args.num_dice <= 0:
            raise ValueError("--num-dice must be positive.")
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else Config.MAX_GEN_TOKENS_LIARS_DICE
        game_spec = _make_liars_dice_spec(args.num_dice, max_new_tokens)
        env_kwargs = {"num_dice": args.num_dice}
    else:
        game_spec = get_game_spec(args.game)
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else game_spec.max_gen_tokens

    policy_backend = VLLMServerBackend(
        base_url=args.policy_url,
        model_name=args.base_model_name,
        api_key=args.api_key,
        timeout_s=args.timeout_s,
        allow_lora_reload=False,
    )
    base_backend = VLLMServerBackend(
        base_url=args.base_url,
        model_name=args.base_model_name,
        api_key=args.api_key,
        timeout_s=args.timeout_s,
        allow_lora_reload=False,
    )

    if not policy_backend.is_enabled():
        raise RuntimeError(f"Policy vLLM server not reachable: {args.policy_url}")
    if not base_backend.is_enabled():
        raise RuntimeError(f"Base vLLM server not reachable: {args.base_url}")

    if not args.no_load_policy_lora:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Policy adapter not found: {adapter_path}")
        print(f"[eval] Loading LoRA '{args.policy_lora_name}' from {adapter_path}")
        _load_lora_adapter(
            base_url=args.policy_url,
            lora_name=args.policy_lora_name,
            lora_path=adapter_path,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
        )

    turn_log_fh: Optional[IO[str]] = None
    if args.turn_log:
        turn_log_path = Path(args.turn_log).expanduser().resolve()
        turn_log_path.parent.mkdir(parents=True, exist_ok=True)
        turn_log_fh = open(turn_log_path, "w")
        print(f"[eval] turn_log={turn_log_path}")

    print("[eval] Starting evaluation")
    print(f"[eval] policy_url={args.policy_url} (lora={args.policy_lora_name})")
    print(f"[eval] base_url={args.base_url}")
    print(f"[eval] game={game_spec.name} env_kwargs={env_kwargs}")
    print(f"[eval] num_games={args.num_games} temperature={args.temperature} max_new_tokens={max_new_tokens}")

    try:
        results = _evaluate_game(
            policy_backend=policy_backend,
            base_backend=base_backend,
            tokenizer=tokenizer,
            policy_lora_name=args.policy_lora_name,
            game_spec=game_spec,
            num_games=args.num_games,
            temperature=args.temperature,
            max_new_tokens=max_new_tokens,
            seed=args.seed,
            batch_size=args.batch_size,
            env_kwargs=env_kwargs,
            print_interval=args.print_interval,
            turn_log_fh=turn_log_fh,
        )
    finally:
        if turn_log_fh is not None:
            turn_log_fh.close()

    print(
        f"[eval] policy_win_rate={results['policy_win_rate']:.1%} "
        f"(wins={results['policy_wins']}/{results['num_games']})"
    )
    print(
        f"[eval] policy_extraction_fail_rate={results['policy_extraction_fail_rate']:.2%} "
        f"base_extraction_fail_rate={results['base_extraction_fail_rate']:.2%}"
    )
    print(f"[eval] turns_per_game_mean={results['turns_per_game_mean']:.2f} elapsed_s={results['elapsed_s']:.1f}")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()