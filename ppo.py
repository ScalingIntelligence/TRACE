#!/usr/bin/env python3
"""
PPO training logic and rollout collection.
"""
import json
import random
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from config import Config
from liars_dice_tools import (
    action_string_to_tool_call,
    extract_tool_call_with_text,
    tool_call_to_json,
)

from game_registry import GameSpec
from inference import InferenceBackend, HFLocalBackend, messages_for_game, build_prompt_text, generate_completion
# =========================
# Logging helper
# =========================
class JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, payload: dict):
        self._fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._fh.flush()

class EMA:
    """Exponential Moving Average for role basline (ie extra reward for 
    going second if going second is disadvantage)"""
    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
        self.value = 0.0
        self.initialized = False

    def update(self, new_value: float):
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.gamma * self.value + (1 - self.gamma) * new_value

    def get(self) -> float:
        return self.value if self.initialized else 0.0


# =========================
# Trajectory storage
# =========================
@dataclass
class StepSample:
    prompt_msgs: list
    action_str: str
    player_id: int
    ret: float
    completion_text: str
    game_id: int


# =========================
# Batching helpers (pad per minibatch)
# =========================
def pad_to_device(seqs: List[torch.Tensor], pad_id: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to max length and move to device."""
    max_len = max(s.shape[0] for s in seqs)
    bsz = len(seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        ids[i, :L] = s
        attn[i, :L] = 1
    return ids.to(device, non_blocking=True), attn.to(device, non_blocking=True)


def build_prompt_plus_action(tokenizer, prompt_msgs: list, action_str: str) -> Tuple[torch.Tensor, int, int]:
    """Build concatenated prompt + action tokens."""
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, 
        add_generation_prompt=True, 
        return_tensors="pt",
        enable_thinking=Config.ENABLE_THINKING
    )[0]
    action_ids = tokenizer(action_str, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return torch.cat([prompt_ids, action_ids], dim=0), int(prompt_ids.shape[0]), int(action_ids.shape[0])


def logprob_action_tokens(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: List[int],
    action_lens: List[int],
    normalize_by_len: bool = True,
) -> torch.Tensor:
    """Compute log probability of action tokens."""
    logp = F.log_softmax(logits.float(), dim=-1)  # [B,T,V] fp32
    B, T, _V = logp.shape
    out = torch.zeros((B,), device=logp.device, dtype=torch.float32)
    for i in range(B):
        pL = prompt_lens[i]
        aL = action_lens[i]
        s = 0.0
        for k in range(aL):
            tok_pos = pL + k
            if tok_pos <= 0 or tok_pos >= T:
                continue
            tok_id = int(input_ids[i, tok_pos].item())
            s = s + logp[i, tok_pos - 1, tok_id]
        if normalize_by_len:
            s = s / max(1, aL)
        out[i] = s
    return out


def values_from_hidden(last_hidden: torch.Tensor, value_head, prompt_lens: List[int]) -> torch.Tensor:
    """Extract value predictions from hidden states."""
    B, T, H = last_hidden.shape
    hs = []
    for i in range(B):
        idx = max(0, min(T - 1, prompt_lens[i] - 1))
        hs.append(last_hidden[i, idx, :])
    hs = torch.stack(hs, dim=0)
    v = value_head(hs.float()).squeeze(-1)
    return v


# =========================
# Self-play collector
# =========================
_TOOL_CALL_GAMES = {
    "liars_dice_tool",
    "liars_dice_memory_tool",
    "liars_dice_memory_updated_tool",
}


def _action_text_for_training(
    game_spec: GameSpec,
    completion: str,
    env_action: str,
    valid_tool_call: bool,
) -> str:
    # Always return the full completion for PPO
    return completion

def collect_games(
    *,
    model,
    tokenizer,
    backend: InferenceBackend,
    num_games: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    logger: JSONLLogger,
    use_constrained_decoding: bool,
    device: str,
    game_spec: GameSpec,
    env_kwargs: Optional[Dict[str, Any]] = None,
    role_baseline_ema: dict = None,
) -> Tuple[List[StepSample], Dict[str, float]]:
    """Collect self-play game trajectories."""
    rng = random.Random(int(seed))
    samples: List[StepSample] = []
    env_kwargs = env_kwargs or {}

    invalid_games = 0
    total_turns = 0
    p0_wins = 0
    extraction_failures = 0

    if backend.supports_batch():
        envs = []
        game_ids: List[int] = []
        episode_steps: List[List[Tuple[list, str, int, str, str]]] = []
        turn_idxs = [0 for _ in range(num_games)]

        for g in range(num_games):
            game_id = seed * 1_000_000 + g
            env = game_spec.make_env(**env_kwargs)
            env.reset(rng.randint(0, 2**31 - 1))
            envs.append(env)
            game_ids.append(game_id)
            episode_steps.append([])

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
                use_guided_choice=use_constrained_decoding,
            )
            t1 = time.time()
            per_item_dt = (t1 - t0) / max(1, len(active))

            for j, (i, pid, obs, legal, msgs) in enumerate(meta):
                completion = completions[j]
                act = game_spec.extract_action(completion, legal)
                illegal_move = (act is None)
                if act is None:
                    extraction_failures += 1

                action_text = _action_text_for_training(
                    game_spec,
                    completion,
                    act,
                    valid_tool_call=not illegal_move,
                )
                episode_steps[i].append((msgs, act, pid, completion, action_text))
                envs[i].step(act)

                turn_idxs[i] += 1
                total_turns += 1

                logger.log({
                    "type": "step",
                    "game_id": game_ids[i],
                    "turn_idx": turn_idxs[i],
                    "player_id": pid,
                    "legal_actions": legal,
                    "action": act,
                    "completion": completion,
                    "illegal_move": illegal_move,
                    "duration_generate_sec": per_item_dt,
                    "timestamp": time.time(),
                })

        for i, env in enumerate(envs):
            invalid_games += 1 if env.invalid_player is not None else 0
            p0_wins += 1 if env.rewards.get(0, 0.0) > 0 else 0

            logger.log({
                "type": "game_end",
                "game": game_spec.name,
                "game_id": game_ids[i],
                "turns": turn_idxs[i],
                "rewards": env.rewards,
                "invalid_player": env.invalid_player,
                "timestamp": time.time(),
            })

            for pm, act, pid, completion, action_text in episode_steps[i]:
                player_reward = float(env.rewards[pid])
                if role_baseline_ema is not None:
                    baseline = role_baseline_ema[pid].get()
                    role_baseline_ema[pid].update(player_reward)
                    player_reward -= baseline

                samples.append(StepSample(
                    prompt_msgs=pm,
                    action_str=(
                        action_text
                        if action_text is not None
                        else (act if act is not None else completion[:350] + "\n...\n" + completion[-50:])
                    ),
                    player_id=pid,
                    ret=player_reward,
                    completion_text=completion,
                    game_id=game_ids[i],
                ))
    else:
        for g in range(num_games):
            game_id = seed * 1_000_000 + g
            env = game_spec.make_env(**env_kwargs)
            env.reset(rng.randint(0, 2**31 - 1))

            episode_steps: List[Tuple[list, str, int, str, str]] = []
            turn_idx = 0

            while not env.done:
                pid = env.current_player
                obs = env.observe(pid)
                legal = env.legal_actions()

                t0 = time.time()
                completion = generate_completion(
                    model, 
                    tokenizer, 
                    pid, 
                    obs, 
                    temperature=temperature, 
                    max_new_tokens=max_new_tokens,
                    use_constrained_decoding=use_constrained_decoding,
                    device=device,
                    game_spec=game_spec,
                )
                t1 = time.time()

                act = game_spec.extract_action(completion, legal)
                illegal_move = (act is None)
                if act is None:
                    extraction_failures += 1
                    act = rng.choice(legal)

                action_text = _action_text_for_training(
                    game_spec,
                    completion,
                    act,
                    valid_tool_call=not illegal_move,
                )
                episode_steps.append((messages_for_game(pid, obs, game_spec), act, pid, completion, action_text))
                env.step(act)

                turn_idx += 1
                total_turns += 1

                logger.log({
                    "type": "step",
                    "game_id": game_id,
                    "turn_idx": turn_idx,
                    "player_id": pid,
                    "legal_actions": legal,
                    "action": act,
                    "completion": completion,
                    "duration_generate_sec": t1 - t0,
                    "timestamp": time.time(),
                })

            invalid_games += 1 if env.invalid_player is not None else 0
            p0_wins += 1 if env.rewards.get(0, 0.0) > 0 else 0

            

            logger.log({
                "type": "game_end",
                "game_id": game_id,
                "turns": turn_idx,
                "rewards": env.rewards,
                "invalid_player": env.invalid_player,
                "timestamp": time.time(),
            })

           
            for pm, act, pid, completion, action_text in episode_steps:
                player_reward = float(env.rewards[pid])
                if role_baseline_ema is not None:
                    baseline = role_baseline_ema[pid].get()
                    role_baseline_ema[pid].update(player_reward)
                    player_reward -= baseline

                samples.append(StepSample(
                    prompt_msgs=pm,
                    action_str=action_text,
                    player_id=pid,
                    ret=float(player_reward),
                    completion_text=completion,
                    game_id=game_id,
                ))

    metrics = {
        "env/invalid_move_rate": extraction_failures / max(1, total_turns),
        "env/turns_per_game_mean": total_turns / max(1, num_games),
        "env/win_rate_p0": p0_wins / max(1, num_games),
    }
    return samples, metrics
