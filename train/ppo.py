#!/usr/bin/env python3
import json
import random
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from .config import Config


from game_registry import GameSpec
from .inference import InferenceBackend, HFLocalBackend, messages_for_game, build_prompt_text, generate_completion
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
    max_len = max(s.shape[0] for s in seqs)
    bsz = len(seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        ids[i, :L] = s
        attn[i, :L] = 1
    return ids.to(device, non_blocking=True), attn.to(device, non_blocking=True)


def build_prompt_plus_action(tokenizer, prompt_msgs: list, action_str: str, tools=None) -> Tuple[torch.Tensor, int, int]:
    kwargs = dict(
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=Config.ENABLE_THINKING,
    )
    if tools:
        kwargs["tools"] = tools
    prompt_ids = tokenizer.apply_chat_template(prompt_msgs, **kwargs)[0]
    action_ids = tokenizer(action_str, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return torch.cat([prompt_ids, action_ids], dim=0), int(prompt_ids.shape[0]), int(action_ids.shape[0])


def logprob_action_tokens(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: List[int],
    action_lens: List[int],
    normalize_by_len: bool = True,
    chunk_size: int = 512,
) -> torch.Tensor:
    B, T, V = logits.shape
    device = logits.device
    
    # Handle edge case
    if T < 2:
        return torch.zeros(B, device=device, dtype=torch.float32)
    
    # Convert to tensors once
    prompt_lens_t = torch.tensor(prompt_lens, device=device, dtype=torch.long)
    action_lens_t = torch.tensor(action_lens, device=device, dtype=torch.long)
    
    # Compute action boundaries (in the shifted indexing)
    # To predict token at input position P, we use logits at position P-1
    action_start = (prompt_lens_t - 1).clamp(min=0)  # [B]
    action_end = (action_start + action_lens_t).clamp(max=T - 1)  # [B]
    
    # Result accumulator
    summed = torch.zeros(B, device=device, dtype=torch.float32)
    
    # Process in chunks along the sequence dimension
    # Only process positions that might contain action tokens
    global_start = int(action_start.min().item())
    global_end = int(action_end.max().item())
    
    for chunk_start in range(global_start, global_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, global_end, T - 1)
        
        if chunk_end <= chunk_start:
            continue
        
        # Get logits and labels for this chunk only
        # logits[:, chunk_start:chunk_end, :] predicts tokens at positions chunk_start+1 to chunk_end+1
        chunk_logits = logits[:, chunk_start:chunk_end, :]  # [B, chunk_len, V]
        chunk_labels = input_ids[:, chunk_start + 1:chunk_end + 1]  # [B, chunk_len]
        
        chunk_len = chunk_end - chunk_start
        
        # Compute cross-entropy for this chunk
        flat_logits = chunk_logits.reshape(-1, V)  # [B * chunk_len, V]
        flat_labels = chunk_labels.reshape(-1).clamp(0, V - 1)  # [B * chunk_len]
        
        chunk_nll = F.cross_entropy(flat_logits, flat_labels, reduction='none')  # [B * chunk_len]
        chunk_logp = -chunk_nll.view(B, chunk_len)  # [B, chunk_len]
        
        # Free memory
        del chunk_logits, chunk_labels, flat_logits, flat_labels, chunk_nll
        
        # Create mask for this chunk: which positions are in the action region?
        chunk_positions = torch.arange(chunk_start, chunk_end, device=device).unsqueeze(0)  # [1, chunk_len]
        chunk_mask = (chunk_positions >= action_start.unsqueeze(1)) & \
                     (chunk_positions < action_end.unsqueeze(1))  # [B, chunk_len]
        
        # Accumulate masked log-probs
        summed += (chunk_logp * chunk_mask.float()).sum(dim=1)
        
        del chunk_logp, chunk_mask
    
    # Normalize by action length if requested
    if normalize_by_len:
        summed = summed / action_lens_t.float().clamp(min=1)
    
    return summed


def per_token_action_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: List[int],
    action_lens: List[int],
    chunk_size: int = 512,
) -> List[torch.Tensor]:
    B, T, V = logits.shape
    device = logits.device

    if T < 2:
        return [torch.zeros(al, device=device, dtype=torch.float32) for al in action_lens]

    prompt_lens_t = torch.tensor(prompt_lens, device=device, dtype=torch.long)
    action_lens_t = torch.tensor(action_lens, device=device, dtype=torch.long)

    # Action boundaries in shifted indexing (logits at pos P-1 predict token at pos P)
    action_start = (prompt_lens_t - 1).clamp(min=0)  # [B]
    action_end = (action_start + action_lens_t).clamp(max=T - 1)  # [B]

    # Pre-allocate per-token result tensors
    results = [torch.zeros(al, device=device, dtype=torch.float32) for al in action_lens]

    global_start = int(action_start.min().item())
    global_end = int(action_end.max().item())

    for chunk_start in range(global_start, global_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, global_end, T - 1)
        if chunk_end <= chunk_start:
            continue

        chunk_logits = logits[:, chunk_start:chunk_end, :]  # [B, chunk_len, V]
        chunk_labels = input_ids[:, chunk_start + 1:chunk_end + 1]  # [B, chunk_len]
        chunk_len = chunk_end - chunk_start

        flat_logits = chunk_logits.reshape(-1, V)
        flat_labels = chunk_labels.reshape(-1).clamp(0, V - 1)

        chunk_nll = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        chunk_logp = -chunk_nll.view(B, chunk_len)

        del chunk_logits, chunk_labels, flat_logits, flat_labels, chunk_nll

        # Scatter per-token logprobs into result tensors
        chunk_positions = torch.arange(chunk_start, chunk_end, device=device)  # [chunk_len]
        for i in range(B):
            a_s = int(action_start[i].item())
            a_e = int(action_end[i].item())
            # Positions in this chunk that fall within sample i's action region
            mask = (chunk_positions >= a_s) & (chunk_positions < a_e)
            if not mask.any():
                continue
            # Offset into the result tensor
            local_pos = chunk_positions[mask] - a_s
            results[i][local_pos] = chunk_logp[i][mask].float()

        del chunk_logp

    return results


def values_from_hidden(last_hidden: torch.Tensor, value_head, prompt_lens: List[int]) -> torch.Tensor:
    B, T, H = last_hidden.shape
    device = last_hidden.device
    
    # Convert prompt lengths to tensor and compute indices
    # We want the hidden state at position (prompt_len - 1) for each sample
    prompt_lens_t = torch.tensor(prompt_lens, device=device, dtype=torch.long)
    indices = (prompt_lens_t - 1).clamp(min=0, max=T - 1)  # [B]
    
    # Use advanced indexing to gather hidden states
    # indices: [B] -> need to index last_hidden[i, indices[i], :]
    batch_indices = torch.arange(B, device=device)  # [B]
    hs = last_hidden[batch_indices, indices, :]  # [B, H]
    
    # Pass through value head
    v = value_head(hs.float()).squeeze(-1)
    return v


# =========================
# Self-play collector
# =========================
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
    device: str,
    game_spec: GameSpec,
    env_kwargs: Optional[Dict[str, Any]] = None,
    role_baseline_ema: dict = None,
) -> Tuple[List[StepSample], Dict[str, float]]:
    rng = random.Random(int(seed))
    samples: List[StepSample] = []
    env_kwargs = env_kwargs or {}

    invalid_games = 0
    total_turns = 0
    p0_wins = 0
    extraction_failures = 0
    response_lengths: List[int] = []  # Track response lengths in characters
    response_token_counts: List[int] = []  # Track response lengths in tokens (approx)

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
            )
            t1 = time.time()
            per_item_dt = (t1 - t0) / max(1, len(active))

            for j, (i, pid, obs, legal, msgs) in enumerate(meta):
                completion = completions[j]
                act = game_spec.extract_action(completion, legal)
                illegal_move = (act is None)
                if act is None:
                    extraction_failures += 1

                # Track response length
                response_lengths.append(len(completion))
                # Approximate token count (rough heuristic: ~4 chars per token)
                response_token_counts.append(len(completion) // 4)

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
                    device=device,
                    game_spec=game_spec,
                )
                t1 = time.time()

                act = game_spec.extract_action(completion, legal)
                illegal_move = (act is None)
                if act is None:
                    extraction_failures += 1
                    act = rng.choice(legal)

                # Track response length
                response_lengths.append(len(completion))
                response_token_counts.append(len(completion) // 4)

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

    # Compute response length statistics
    if response_lengths:
        resp_len_mean = sum(response_lengths) / len(response_lengths)
        resp_len_max = max(response_lengths)
        resp_len_min = min(response_lengths)
        resp_tok_mean = sum(response_token_counts) / len(response_token_counts)
    else:
        resp_len_mean = resp_len_max = resp_len_min = resp_tok_mean = 0

    metrics = {
        "env/invalid_move_rate": extraction_failures / max(1, total_turns),
        "env/turns_per_game_mean": total_turns / max(1, num_games),
        "env/win_rate_p0": p0_wins / max(1, num_games),
        "env/response_len_chars_mean": resp_len_mean,
        "env/response_len_chars_max": resp_len_max,
        "env/response_len_chars_min": resp_len_min,
        "env/response_tokens_mean": resp_tok_mean,
    }
    return samples, metrics
