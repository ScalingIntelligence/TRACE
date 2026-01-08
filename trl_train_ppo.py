"""
Kuhn Poker PPO (TRL-style) with terminal env rewards + eval vs random/base.

This file builds on TRL's experimental PPO trainer for correctness.
Key differences from standard TRL PPO:
  - Rollout comes from self-play in KuhnPoker env (not from dataset + reward model)
  - Reward is terminal env outcome (zero-sum), applied at end of response
  - Still uses TRL PPO update mechanics (masking/GAE/minibatches/accelerate)
"""

# Unsloth
from unsloth import FastLanguageModel

import gc
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import GenerationConfig

# === TRL imports - use experimental.ppo ===
from trl.experimental.ppo import PPOConfig, PPOTrainer
from trl.experimental.ppo.ppo_trainer import (
    INVALID_LOGPROB,
    masked_mean,
    masked_whiten,
    batch_generation,
    forward,
    PolicyAndValueWrapper,
)
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    empty_cache,
    selective_log_softmax,
    pad,
)


def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving the position of the
    first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.

    Args:
        bools (`torch.Tensor`):
            An N-dimensional boolean tensor.
        dtype (`torch.dtype`, optional):
            The desired data type of the output tensor. Defaults to `torch.long`.

    Returns:
        `torch.Tensor`:
            An (N-1)-dimensional tensor of integers indicating the position of the first True in each row. If no True
            value is found in a row, returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )

# -----------------------------------------------------------------------------
# Kuhn Poker Environment (from train_ppo.py)
# -----------------------------------------------------------------------------

_ACTION_RE = re.compile(r"\[(check|bet|call|fold)\]", re.IGNORECASE)
_CARD_RANK = {"J": 0, "Q": 1, "K": 2}
ACTION_STRS = ["[check]", "[bet]", "[call]", "[fold]"]

SYSTEM_PROMPT = (
    "You are playing Kuhn Poker.\n"
    "Respond with EXACTLY ONE action token and NOTHING ELSE.\n"
    "Valid outputs: [check] or [bet] or [call] or [fold].\n"
)


def _extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract the first legal action from text."""
    matches = _ACTION_RE.findall(text or "")
    for m in matches:
        a = f"[{m.lower()}]"
        if a in legal_actions:
            return a
    return None


def _messages(player_id: int, observation: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation},
    ]


class KuhnPoker:
    """Kuhn Poker environment for multi-round games."""
    
    def __init__(self, num_rounds: int = 5):
        self.num_rounds = num_rounds
        self.reset(0)

    def reset(self, seed: int):
        rng = random.Random(int(seed))
        self.start_player0 = rng.randint(0, 1)
        self.round_cards = [rng.sample(["J", "Q", "K"], 2) for _ in range(self.num_rounds)]
        self.round_idx = 1
        self.chips = [0, 0]
        self.history = []
        self.actions_in_round = []
        self.bet_by = None
        self.current_player = self._round_start_player()
        self.done = False
        self.invalid_player = None
        self.rewards = {0: 0.0, 1: 0.0}

    def _round_start_player(self) -> int:
        return (self.start_player0 + (self.round_idx - 1)) % 2

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ["[check]", "[bet]"] if self.bet_by is None else ["[call]", "[fold]"]

    def _card(self, player_id: int) -> str:
        return self.round_cards[self.round_idx - 1][player_id]

    def _round_history_str(self) -> str:
        if not self.actions_in_round:
            return "None"
        return " ".join([f"P{p}:{a}" for p, a in self.actions_in_round])

    def _full_history_str(self) -> str:
        if not self.history:
            return "None"
        return " ".join([f"R{r}P{p}:{a}" for r, p, a in self.history])

    def observe(self, player_id: int) -> str:
        legal = ", ".join(self.legal_actions())
        return (
            f"[GAME] You are Player {player_id} in a {self.num_rounds} round game of Kuhn Poker.\n"
            "Game Rules:\n"
            "- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)\n"
            "- Each player antes 1 chip and receives 1 card each round\n"
            f"- Game continues for {self.num_rounds} rounds\n"
            "- The player with the most chips after all rounds wins\n"
            "Action Rules:\n"
            "- '[check]': Pass without betting (only if no bet is on the table)\n"
            "- '[bet]': Add 1 chip to the pot (only if no bet is on the table)\n"
            "- '[call]': Match an opponent's bet by adding 1 chip to the pot\n"
            "- '[fold]': Surrender your hand and let your opponent win the pot\n"
            f"[GAME] Scores (chips won so far): Player 0: {self.chips[0]}, Player 1: {self.chips[1]}\n"
            f"[GAME] Starting round {self.round_idx} out of {self.num_rounds} rounds.\n"
            f"Your card is: {self._card(player_id)}\n"
            f"[GAME] Betting history this round: {self._round_history_str()}\n"
            f"[GAME] Full game history: {self._full_history_str()}\n"
            f"Your available actions are: {legal}\n"
        )

    def _end_round_showdown(self, pot_win: int):
        c0, c1 = self._card(0), self._card(1)
        winner = 0 if _CARD_RANK[c0] > _CARD_RANK[c1] else 1
        payoff = pot_win if winner == 0 else -pot_win
        self.chips[0] += payoff
        self.chips[1] -= payoff
        self.round_idx += 1
        self.actions_in_round = []
        self.bet_by = None
        if self.round_idx > self.num_rounds:
            self.done = True
            outcome = 0
            if self.chips[0] > self.chips[1]:
                outcome = 1
            elif self.chips[0] < self.chips[1]:
                outcome = -1
            self.rewards = {0: float(outcome), 1: float(-outcome)}
        else:
            self.current_player = self._round_start_player()

    def _terminate_invalid(self, player_id: int):
        self.done = True
        self.invalid_player = player_id
        other = 1 - player_id
        self.rewards = {0: 0.5, 1: 0.5}
        self.rewards[player_id] = -1.5
        self.rewards[other] = 0.5

    def step(self, action: Optional[str]):
        if self.done:
            return
        if action not in self.legal_actions():
            self._terminate_invalid(self.current_player)
            return

        p = self.current_player
        self.actions_in_round.append((p, action))
        self.history.append((self.round_idx, p, action))

        if self.bet_by is None:
            if action == "[bet]":
                self.bet_by = p
                self.current_player = 1 - p
                return
            if len(self.actions_in_round) == 2:
                self._end_round_showdown(1)
                return
            self.current_player = 1 - p
            return

        if action == "[fold]":
            bettor = self.bet_by
            payoff = 1 if bettor == 0 else -1
            self.chips[0] += payoff
            self.chips[1] -= payoff
            self.round_idx += 1
            self.actions_in_round = []
            self.bet_by = None
            if self.round_idx > self.num_rounds:
                self.done = True
                outcome = 0
                if self.chips[0] > self.chips[1]:
                    outcome = 1
                elif self.chips[0] < self.chips[1]:
                    outcome = -1
                self.rewards = {0: float(outcome), 1: float(-outcome)}
            else:
                self.current_player = self._round_start_player()
            return

        self._end_round_showdown(2)


# -----------------------------------------------------------------------------
# Value model wrapper compatible with TRL's PolicyAndValueWrapper
# -----------------------------------------------------------------------------

class ValueModelWrapper(nn.Module):
    """
    TRL-compatible value model that shares backbone with policy.
    
    TRL's PolicyAndValueWrapper expects:
      - value_model.base_model_prefix -> attribute name of backbone
      - getattr(value_model, base_model_prefix) -> the backbone module
      - value_model.score(hidden_states) -> value predictions
    """
    def __init__(self, policy_model: nn.Module, hidden_size: int):
        super().__init__()
        # Store base_model_prefix from policy
        self.base_model_prefix = getattr(policy_model, "base_model_prefix", "model")
        
        # Share the exact same backbone module object
        backbone = getattr(policy_model, self.base_model_prefix, None)
        if backbone is None:
            # Try common alternatives
            for attr in ["model", "transformer", "base_model"]:
                if hasattr(policy_model, attr):
                    backbone = getattr(policy_model, attr)
                    self.base_model_prefix = attr
                    break
        if backbone is None:
            raise ValueError(f"Could not find backbone in policy_model")
        
        setattr(self, self.base_model_prefix, backbone)
        
        # Value head: Linear(hidden_size, 1) in fp32 for numerical stability
        self.score = nn.Linear(hidden_size, 1, dtype=torch.float32)
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass returning hidden states (required by TRL's get_reward)."""
        backbone = getattr(self, self.base_model_prefix)
        kwargs = dict(kwargs)
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict", True)
        return backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class NoOpRewardModel(nn.Module):
    """Placeholder reward model - env provides terminal rewards."""
    def __init__(self):
        super().__init__()
        # Need a dummy parameter to satisfy accelerator.prepare
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        
    def forward(self, *args, **kwargs):
        raise RuntimeError("NoOpRewardModel should never be called (env provides rewards).")


# -----------------------------------------------------------------------------
# Kuhn Poker PPO Trainer
# -----------------------------------------------------------------------------

class KuhnPokerPPOTrainer(PPOTrainer):
    """
    TRL-style PPO trainer adapted to Kuhn Poker self-play.
    
    Key differences from standard PPO:
      - Rollout from self-play (not dataset + reward model)  
      - Terminal env rewards (not per-token from reward model)
      - Same PPO update mechanics (GAE, clipping, minibatches)
    """

    def __init__(
        self,
        *args,
        kuhn_env_cls,
        num_rounds: int,
        base_adapter_dir: Path,
        eval_games: int = 50,
        eval_every_steps: int = 20,
        max_new_tokens: int = 8,
        **kwargs,
    ):
        self.kuhn_env_cls = kuhn_env_cls
        self.num_rounds = int(num_rounds)
        self.turns_per_game = 2 * self.num_rounds  # Max turns per game
        self.base_adapter_dir = Path(base_adapter_dir)
        self.eval_games = int(eval_games)
        self.eval_every_steps = int(eval_every_steps)
        self.max_new_tokens = int(max_new_tokens)

        super().__init__(*args, **kwargs)

        # TRL PPO uses left padding for generation
        if hasattr(self.processing_class, "padding_side"):
            self.processing_class.padding_side = "left"
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token = self.processing_class.eos_token

    def _prompt_text(self, pid: int, obs: str) -> str:
        """Build prompt string using chat template."""
        msgs = _messages(pid, obs)
        return self.processing_class.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def _rollout_selfplay(self, seed: int, generation_config: GenerationConfig):
        """
        Collect one PPO rollout batch via Kuhn Poker self-play.
        
        Returns tensors matching TRL PPO format:
          queries, query_responses, responses,
          logprobs, ref_logprobs, values,
          sequence_lengths, scores,
          game_ids,  # NEW: track which game each sample belongs to
          env_metrics
        """
        args = self.args
        device = self.accelerator.device
        tok = self.processing_class
        pad_id = tok.pad_token_id
        eos_id = tok.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must have eos_token_id")

        # Collect complete games (variable batch size, like train_ppo.py)
        # Each game produces variable number of turns
        games_per_rank = max(1, args.local_batch_size // self.turns_per_game + 1)

        rng = random.Random(int(seed) + self.local_seed)

        # Run self-play games
        envs = []
        game_ids = []  # Track game IDs
        for g in range(games_per_rank):
            game_id = seed * 1_000_000 + g
            env = self.kuhn_env_cls(num_rounds=self.num_rounds)
            env.reset(rng.randint(0, 2**31 - 1))
            envs.append(env)
            game_ids.append(game_id)

        prompt_ids_list: List[torch.Tensor] = []
        response_ids_list: List[torch.Tensor] = []
        player_ids: List[int] = []
        env_idxs: List[int] = []  # Index into envs array (same as game_id index)
        invalid_fallbacks = 0

        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            
            # Play all games step by step
            while any(not env.done for env in envs):
                # Collect active games
                active_indices = [i for i, env in enumerate(envs) if not env.done]
                if not active_indices:
                    break
                    
                prompts, legals, pids = [], [], []
                for i in active_indices:
                    env = envs[i]
                    pid = env.current_player
                    obs = env.observe(pid)
                    legal = env.legal_actions()
                    prompts.append(self._prompt_text(pid, obs))
                    legals.append(legal)
                    pids.append(pid)

                # Tokenize prompts
                enc = tok(
                    prompts,
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                q = enc["input_ids"].to(device)
                attn = enc.get("attention_mask", None)
                if attn is not None:
                    attn = attn.to(device)

                # Generate actions
                query_responses, _ = batch_generation(
                    unwrapped_model.policy,
                    q,
                    args.local_rollout_forward_batch_size,
                    pad_id,
                    generation_config,
                )

                context_len = q.shape[1]
                gen = query_responses[:, context_len:]
                texts = tok.batch_decode(gen, skip_special_tokens=True)

                # Process generated actions
                for j, i in enumerate(active_indices):
                    txt = texts[j]
                    legal = legals[j]
                    pid = pids[j]
                    
                    act = _extract_action(txt, legal)
                    if act is None:
                        act = rng.choice(legal)
                        invalid_fallbacks += 1
                    
                    envs[i].step(act)

                    # Store prompt (unpadded) and action
                    if attn is not None:
                        unp = q[j][attn[j].bool()].detach().cpu()
                    else:
                        row = q[j].detach().cpu()
                        nz = (row != pad_id).nonzero(as_tuple=False)
                        unp = row[nz[0].item():] if len(nz) else row

                    act_ids = tok(act, add_special_tokens=False)["input_ids"]
                    act_ids = torch.tensor(act_ids + [eos_id], dtype=torch.long)

                    prompt_ids_list.append(unp)
                    response_ids_list.append(act_ids)
                    player_ids.append(pid)
                    env_idxs.append(i)

                del q, attn, query_responses, gen
                empty_cache()

        # Keep all collected samples (variable batch size, like train_ppo.py)
        # Don't truncate/pad - accept variable batch size from complete games
        total_steps = len(prompt_ids_list)

        # Compute terminal rewards for each sample
        scores = torch.empty((total_steps,), dtype=torch.float32)
        sample_game_ids = []  # Track game_id for each sample
        p0_wins = 0
        for gi, env in enumerate(envs):
            if env.rewards.get(0, 0.0) > 0:
                p0_wins += 1
        for k in range(total_steps):
            env = envs[env_idxs[k]]
            pid = player_ids[k]
            scores[k] = float(env.rewards[pid])
            sample_game_ids.append(game_ids[env_idxs[k]])  # Map env_idx to game_id

        # Pad prompts LEFT, responses RIGHT (TRL convention)
        max_q = max(x.numel() for x in prompt_ids_list)
        queries = torch.full((total_steps, max_q), pad_id, dtype=torch.long)
        for i, ids in enumerate(prompt_ids_list):
            L = ids.numel()
            queries[i, -L:] = ids

        max_r = max(x.numel() for x in response_ids_list)
        responses = torch.full((total_steps, max_r), pad_id, dtype=torch.long)
        for i, ids in enumerate(response_ids_list):
            L = ids.numel()
            responses[i, :L] = ids

        queries = queries.to(device)
        responses = responses.to(device)
        scores = scores.to(device)

        query_responses = torch.cat([queries, responses], dim=1)
        context_len = queries.shape[1]

        # Compute logprobs and values
        logprobs_chunks = []
        ref_logprobs_chunks = []
        values_chunks = []
        seq_len_chunks = []

        do_kl = float(args.kl_coef) > 0.0
        ref_policy = self.ref_model

        for start in range(0, total_steps, args.local_rollout_forward_batch_size):
            end = min(start + args.local_rollout_forward_batch_size, total_steps)
            mb_qr = query_responses[start:end]
            mb_resp = responses[start:end]

            # Policy logprobs
            out = forward(self.model.policy, mb_qr, pad_id)
            logits = out.logits[:, context_len - 1 : -1]
            logits /= args.temperature + 1e-7
            mb_logprob = selective_log_softmax(logits, mb_resp)
            del out, logits
            empty_cache()

            # Reference logprobs for KL
            if do_kl:
                if ref_policy is None:
                    with self.null_ref_context():
                        ref_out = forward(self.model.policy, mb_qr, pad_id)
                else:
                    ref_out = forward(ref_policy, mb_qr, pad_id)
                ref_logits = ref_out.logits[:, context_len - 1 : -1]
                ref_logits /= args.temperature + 1e-7
                mb_ref_logprob = selective_log_softmax(ref_logits, mb_resp)
                del ref_out, ref_logits
                empty_cache()
            else:
                mb_ref_logprob = torch.zeros_like(mb_logprob)

            # Value predictions
            unwrapped_value_model = self.accelerator.unwrap_model(self.model).value_model
            full_value, _, _ = get_reward(unwrapped_value_model, mb_qr, pad_id, context_len)
            mb_value = full_value[:, context_len - 1 : -1].squeeze(-1)
            del full_value
            empty_cache()

            # Sequence lengths (position of first pad token - 1, or length-1 if no pad)
            mb_seq_len = first_true_indices(mb_resp == pad_id) - 1

            logprobs_chunks.append(mb_logprob)
            ref_logprobs_chunks.append(mb_ref_logprob)
            values_chunks.append(mb_value)
            seq_len_chunks.append(mb_seq_len)

        logprobs = torch.cat(logprobs_chunks, 0)
        ref_logprobs = torch.cat(ref_logprobs_chunks, 0)
        values = torch.cat(values_chunks, 0)
        sequence_lengths = torch.cat(seq_len_chunks, 0)

        env_metrics = {
            "env/games_per_rank": len(envs),
            "env/samples_collected": total_steps,
            "env/win_rate_p0": p0_wins / max(1, len(envs)),
            "env/parse_fail_rate": invalid_fallbacks / max(1, total_steps),
        }

        return (
            queries, query_responses, responses,
            logprobs, ref_logprobs, values,
            sequence_lengths, scores,
            sample_game_ids,  # NEW: return game IDs for GAE grouping
            env_metrics,
        )

    @torch.no_grad()
    def _policy_act(self, policy_model: nn.Module, pid: int, obs: str, temperature: float) -> str:
        """Generate a single action from policy."""
        tok = self.processing_class
        msgs = _messages(pid, obs)
        input_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.accelerator.device)
        prompt_len = input_ids.shape[-1]

        do_sample = float(temperature) > 0.0
        out = policy_model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=(temperature if do_sample else None),
            top_p=1.0,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        txt = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        return txt

    @torch.no_grad()
    def evaluate_vs_random(self, num_games: int, temperature: float = 0.0) -> Dict[str, float]:
        """Evaluate policy vs random opponent."""
        rng = random.Random(10_000_000 + self.state.global_step + self.accelerator.process_index)
        half = max(1, num_games // 2)

        # Split games across ranks
        per_rank = math.ceil(num_games / self.accelerator.num_processes)
        start = self.accelerator.process_index * per_rank
        end = min(num_games, start + per_rank)
        local_games = max(0, end - start)
        
        if local_games == 0:
            local = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.accelerator.device)
            gathered = self.accelerator.gather_for_metrics(local.unsqueeze(0))
            return {"eval/win_rate_vs_random": 0.0}

        policy = self.accelerator.unwrap_model(self.model).policy
        wins = 0
        invalids = 0
        total_turns = 0

        for gi in range(start, end):
            env = self.kuhn_env_cls(num_rounds=self.num_rounds)
            env.reset(rng.randint(0, 2**31 - 1))
            policy_is_p0 = gi < half

            while not env.done:
                pid = env.current_player
                is_policy_turn = (pid == 0 and policy_is_p0) or (pid == 1 and (not policy_is_p0))
                legal = env.legal_actions()

                if is_policy_turn:
                    obs = env.observe(pid)
                    completion = self._policy_act(policy, pid, obs, temperature=temperature)
                    act = _extract_action(completion, legal) or rng.choice(legal)
                else:
                    act = rng.choice(legal)

                env.step(act)
                total_turns += 1

            invalids += 1 if env.invalid_player is not None else 0
            policy_won = (env.rewards[0] > 0) if policy_is_p0 else (env.rewards[1] > 0)
            wins += 1 if policy_won else 0

        local = torch.tensor([wins, local_games, invalids, total_turns], dtype=torch.float32, device=self.accelerator.device)
        gathered = self.accelerator.gather_for_metrics(local.unsqueeze(0))

        wins_sum = gathered[:, 0].sum()
        games_sum = gathered[:, 1].sum()
        inv_sum = gathered[:, 2].sum()
        turns_sum = gathered[:, 3].sum()

        return {
            "eval/win_rate_vs_random": (wins_sum / games_sum).item() if games_sum > 0 else 0.0,
            "eval/invalid_game_rate_vs_random": (inv_sum / games_sum).item() if games_sum > 0 else 0.0,
            "eval/turns_per_game_mean_vs_random": (turns_sum / games_sum).item() if games_sum > 0 else 0.0,
        }

    @torch.no_grad()
    def evaluate_vs_base(self, num_games: int, temperature: float = 0.0) -> Dict[str, float]:
        """Evaluate policy vs base (untrained) model."""
        rng = random.Random(20_000_000 + self.state.global_step + self.accelerator.process_index)
        half = max(1, num_games // 2)

        per_rank = math.ceil(num_games / self.accelerator.num_processes)
        start = self.accelerator.process_index * per_rank
        end = min(num_games, start + per_rank)
        local_games = max(0, end - start)
        
        if local_games == 0:
            local = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.accelerator.device)
            gathered = self.accelerator.gather_for_metrics(local.unsqueeze(0))
            return {"eval/win_rate_vs_base": 0.0}

        # Load base model with base adapter
        base_model, _tok2 = FastLanguageModel.from_pretrained(
            model_name=self.model.policy.config._name_or_path,
            max_seq_length=getattr(self.model.policy.config, "max_position_embeddings", 1024),
            dtype=None,
            load_in_4bit=True,
        )
        base_model = FastLanguageModel.get_peft_model(
            base_model,
            r=getattr(self.args, "lora_rank", 4),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=getattr(self.args, "lora_alpha", 8),
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        base_model.load_adapter(str(self.base_adapter_dir), adapter_name="base")
        base_model.set_adapter("base")
        base_model = base_model.to(self.accelerator.device)
        base_model.eval()

        policy = self.accelerator.unwrap_model(self.model).policy

        wins = 0
        invalids = 0
        total_turns = 0

        for gi in range(start, end):
            env = self.kuhn_env_cls(num_rounds=self.num_rounds)
            env.reset(rng.randint(0, 2**31 - 1))
            policy_is_p0 = gi < half

            while not env.done:
                pid = env.current_player
                legal = env.legal_actions()
                policy_turn = (pid == 0 and policy_is_p0) or (pid == 1 and (not policy_is_p0))
                obs = env.observe(pid)

                if policy_turn:
                    completion = self._policy_act(policy, pid, obs, temperature=temperature)
                else:
                    completion = self._policy_act(base_model, pid, obs, temperature=temperature)

                act = _extract_action(completion, legal) or rng.choice(legal)
                env.step(act)
                total_turns += 1

            invalids += 1 if env.invalid_player is not None else 0
            policy_won = (env.rewards[0] > 0) if policy_is_p0 else (env.rewards[1] > 0)
            wins += 1 if policy_won else 0

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        local = torch.tensor([wins, local_games, invalids, total_turns], dtype=torch.float32, device=self.accelerator.device)
        gathered = self.accelerator.gather_for_metrics(local.unsqueeze(0))

        wins_sum = gathered[:, 0].sum()
        games_sum = gathered[:, 1].sum()
        inv_sum = gathered[:, 2].sum()
        turns_sum = gathered[:, 3].sum()

        return {
            "eval/win_rate_vs_base": (wins_sum / games_sum).item() if games_sum > 0 else 0.0,
            "eval/invalid_game_rate_vs_base": (inv_sum / games_sum).item() if games_sum > 0 else 0.0,
            "eval/turns_per_game_mean_vs_base": (turns_sum / games_sum).item() if games_sum > 0 else 0.0,
        }

    def train(self):
        """
        Main PPO training loop - follows TRL PPOTrainer structure but uses self-play rollouts.
        """
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        device = accelerator.device
        tok = self.processing_class
        pad_id = tok.pad_token_id

        # Generation config for self-play
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=(args.temperature + 1e-7),
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("=== Kuhn Poker self-play PPO training (TRL-style) ===")
        start_time = time.time()

        # Stats accumulators (same shape as TRL)
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)

        model.train()

        # Initialize trainer state
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += args.batch_size

            with torch.no_grad():
                # Collect self-play rollout
                (
                    queries, query_responses, responses,
                    logprobs, ref_logprobs, values,
                    sequence_lengths, scores,
                    sample_game_ids,
                    env_metrics,
                ) = self._rollout_selfplay(seed=update, generation_config=generation_config)

                context_length = queries.shape[1]
                local_batch_size = queries.shape[0]  # Variable batch size from complete games

                # Check for EOS tokens
                contain_eos_token = torch.any(responses == tok.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

                # Build masks (same as TRL PPOTrainer)
                response_idxs = torch.arange(responses.shape[1], device=device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > sequence_lengths_p1.unsqueeze(1)
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # Compute KL divergence and rewards
                logr = ref_logprobs - logprobs
                if args.kl_estimator == "k1":
                    kl = -logr
                else:  # k3
                    kl = (logr.exp() - 1) - logr

                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()

                # Add terminal env score at end of response
                actual_start = torch.arange(rewards.size(0), device=device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[actual_start, actual_end] += scores

                # Optionally whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # GAE computation across game trajectories (not per-action)
                # Group samples by game_id and compute GAE backward through each game
                # For single-token actions, we use the value at the action position
                advantages = torch.zeros_like(rewards)
                returns = torch.zeros_like(values)
                
                # Group samples by game_id
                from collections import defaultdict
                game_groups = defaultdict(list)
                for idx, game_id in enumerate(sample_game_ids):
                    game_groups[game_id].append(idx)
                
                # Compute GAE for each game trajectory
                for game_id, sample_indices in game_groups.items():
                    # Samples should be in game order (they're collected sequentially)
                    game_samples = sorted(sample_indices)  # Keep original order
                    
                    if len(game_samples) == 0:
                        continue
                    
                    # Get terminal reward for this game (all samples from same game have same reward)
                    terminal_reward = scores[game_samples[0]].item()
                    
                    # Extract per-action values for this game
                    # For single-token actions, use value at position 0 (the action token)
                    game_values = []
                    for idx in game_samples:
                        # Value at the action position (first token of response, position 0)
                        # values shape: [batch, seq_len] where seq_len is response length
                        if values.shape[1] > 0:
                            # Use value at first position (action token)
                            game_values.append(values[idx, 0].item())
                        else:
                            game_values.append(0.0)
                    
                    # Compute GAE backward through game trajectory
                    # Terminal reward is at the end, propagate backward
                    game_advantages = []
                    game_returns = []
                    lastgaelam = 0.0
                    
                    # Reverse through game (from last action to first)
                    for i in reversed(range(len(game_samples))):
                        idx = game_samples[i]
                        value = game_values[i]
                        
                        # Next value is value of next action in game (or terminal reward if last)
                        if i == len(game_samples) - 1:
                            # Last action in game: next value is terminal reward (bootstrap)
                            nextvalue = terminal_reward
                        else:
                            # Next value is value of next action in game
                            nextvalue = game_values[i + 1]
                        
                        # GAE: delta = reward + gamma * next_value - value
                        # For terminal reward, it's only at the last action
                        if i == len(game_samples) - 1:
                            reward = terminal_reward
                        else:
                            reward = 0.0  # Intermediate actions have zero reward (only terminal matters)
                        
                        delta = reward + args.gamma * nextvalue - value
                        lastgaelam = delta + args.gamma * args.lam * lastgaelam
                        
                        game_advantages.append(lastgaelam)
                        game_returns.append(lastgaelam + value)
                    
                    # Reverse back to forward order
                    game_advantages = game_advantages[::-1]
                    game_returns = game_returns[::-1]
                    
                    # Store advantages and returns at the action position (position 0 for single-token actions)
                    for i, idx in enumerate(game_samples):
                        if advantages.shape[1] > 0:
                            advantages[idx, 0] = game_advantages[i]
                            returns[idx, 0] = game_returns[i]
                
                # Whitening across all samples (not per-game)
                advantages_flat = advantages[~padding_mask]
                if advantages_flat.numel() > 0:
                    advantages_mean = advantages_flat.mean()
                    advantages_std = advantages_flat.std(unbiased=False) + 1e-8
                    advantages = (advantages - advantages_mean) / advantages_std
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                empty_cache()

            # PPO update epochs (same structure as TRL, but with variable batch size)
            # Calculate actual minibatch size based on variable local_batch_size
            actual_mini_batch_size = max(1, local_batch_size // args.num_mini_batches)
            
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(local_batch_size)
                minibatch_idx = 0

                for mini_batch_start in range(0, local_batch_size, actual_mini_batch_size):
                    mini_batch_end = min(mini_batch_start + actual_mini_batch_size, local_batch_size)
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0

                    for micro_batch_start in range(0, len(mini_batch_inds), args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = min(micro_batch_start + args.per_device_train_batch_size, len(mini_batch_inds))
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # Extract per-action advantages and returns (position 0 for single-token actions)
                            mb_advantage = advantages[micro_batch_inds, 0]  # Per-action advantage
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds, 0]  # Per-action return
                            mb_values = values[micro_batch_inds, 0]  # Per-action value

                            # Forward pass through policy+value model
                            output, vpred_temp = forward(model, mb_query_responses, pad_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7

                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)

                            # Extract per-action value prediction (position 0 for single-token actions)
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpred_action = vpred[:, 0]  # Per-action value at position 0

                            # Value loss with clipping (per-action)
                            vpredclipped = torch.clamp(
                                vpred_action,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred_action - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * vf_loss_max.mean()
                            vf_clipfrac = (vf_losses2 > vf_losses1).float().mean()

                            # Policy loss with clipping (per-action)
                            # Use logprob at action position (position 0 for single-token actions)
                            mb_logprob_action = mb_logprobs[:, 0]  # Per-action logprob
                            new_logprob_action = new_logprobs[:, 0]  # Per-action new logprob
                            
                            logprobs_diff = new_logprob_action - mb_logprob_action
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()

                            # Total loss
                            loss = pg_loss + args.vf_coef * vf_loss

                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            # Track stats (use safe indexing for variable batch sizes)
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                entropy_action = entropy[:, 0].mean()  # Per-action entropy
                                approxkl = 0.5 * (logprobs_diff**2).mean()

                                # Safe indexing for variable batch sizes
                                if minibatch_idx < args.num_mini_batches and gradient_accumulation_idx < args.gradient_accumulation_steps:
                                    approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                    pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                                    pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                    vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                    vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                                    entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy_action
                                    ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()

                        gradient_accumulation_idx += 1

                    minibatch_idx += 1
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac,
                        logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max, pg_loss,
                        loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_return, mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    empty_cache()

            # Logging (same metrics as TRL)
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                ppo_reward = mean_non_score_reward + scores.mean()

                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {
                    "eps": eps,
                    "episode": self.state.episode,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "objective/kl": accelerator.gather_for_metrics(mean_kl).mean().item(),
                    "objective/entropy": accelerator.gather_for_metrics(mean_entropy).mean().item(),
                    "objective/non_score_reward": accelerator.gather_for_metrics(mean_non_score_reward).mean().item(),
                    "objective/ppo_reward": accelerator.gather_for_metrics(ppo_reward).mean().item(),
                    "objective/terminal_score_mean": accelerator.gather_for_metrics(scores.mean()).mean().item(),
                    "policy/approxkl_avg": accelerator.gather_for_metrics(approxkl_stats).mean().item(),
                    "policy/clipfrac_avg": accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item(),
                    "loss/policy_avg": accelerator.gather_for_metrics(pg_loss_stats).mean().item(),
                    "loss/value_avg": accelerator.gather_for_metrics(vf_loss_stats).mean().item(),
                    "val/clipfrac_avg": accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item(),
                    "policy/entropy_avg": accelerator.gather_for_metrics(entropy_stats).mean().item(),
                    "val/ratio": accelerator.gather_for_metrics(ratio_stats).mean().item(),
                    **env_metrics,
                }
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            # Periodic evaluation
            if self.eval_every_steps > 0 and (self.state.global_step % self.eval_every_steps == 0):
                model.eval()
                t0 = time.time()
                eval_r = self.evaluate_vs_random(num_games=self.eval_games, temperature=0.0)
                eval_b = self.evaluate_vs_base(num_games=self.eval_games, temperature=0.0)
                eval_logs = {**eval_r, **eval_b, "eval/time_sec": time.time() - t0}
                self.log(eval_logs)
                accelerator.print(f"[eval] vs_random: {eval_r.get('eval/win_rate_vs_random', 0):.3f}, vs_base: {eval_b.get('eval/win_rate_vs_base', 0):.3f}")
                model.train()

            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(args, self.state, self.control)

            # Cleanup
            del (
                kl, mean_kl, mean_entropy, mean_non_score_reward, non_score_reward,
                queries, query_responses, responses, logprobs, ref_logprobs,
                values, sequence_lengths, rewards, advantages, returns, scores,
                sequence_lengths_p1, padding_mask, padding_mask_p1, response_idxs, contain_eos_token,
            )
            empty_cache()
            gc.collect()

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(args, self.state, self.control)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    """Main entry point for Kuhn Poker TRL-PPO training."""
    
    # ---------- Kuhn settings ----------
    NUM_ROUNDS = 5
    GAMES_PER_UPDATE = 8  # per rank
    TURNS_PER_GAME = 2 * NUM_ROUNDS  # Max turns
    LOCAL_BATCH_SIZE = GAMES_PER_UPDATE * TURNS_PER_GAME

    # ---------- PPO settings ----------
    PPO_EPOCHS = 2
    NUM_MINI_BATCHES = 4
    PER_DEVICE_TRAIN_BSZ = 16
    GRAD_ACCUM = max(1, LOCAL_BATCH_SIZE // (PER_DEVICE_TRAIN_BSZ * NUM_MINI_BATCHES))

    MAX_NEW_TOKENS = 8
    TEMPERATURE = 0.7
    CLIP_EPS = 0.2
    VF_COEF = 0.5

    TOTAL_UPDATES = 1000
    EVAL_EVERY = 20
    EVAL_GAMES = 50

    OUTPUT_DIR = Path("./outputs/kuhn_trl_ppo").resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_ADAPTER_DIR = OUTPUT_DIR / "base_adapter"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Load policy model (Unsloth) ----------
    max_seq_length = 768
    lora_rank = 4

    policy, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    policy = FastLanguageModel.get_peft_model(
        policy,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastLanguageModel.for_training(policy)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = policy.to(device)

    # ---------- Create value model ----------
    cfg = policy.config
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden_size from policy.config")

    value_model = ValueModelWrapper(policy, int(hidden_size)).to(device)

    # ---------- Save base adapter ----------
    BASE_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(BASE_ADAPTER_DIR))
    print(f"[base] saved initial adapter to: {BASE_ADAPTER_DIR}")

    # ---------- Dummy dataset (TRL requires one for init) ----------
    dummy = Dataset.from_dict({"input_ids": [[tokenizer.eos_token_id]] * 1024})

    # ---------- PPOConfig ----------
    ppo_args = PPOConfig(
        exp_name="kuhn-trl-ppo",
        seed=3407,
        output_dir=str(OUTPUT_DIR),
        # Batch sizing
        per_device_train_batch_size=PER_DEVICE_TRAIN_BSZ,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_mini_batches=NUM_MINI_BATCHES,
        num_ppo_epochs=PPO_EPOCHS,
        local_rollout_forward_batch_size=PER_DEVICE_TRAIN_BSZ,
        per_device_eval_batch_size=8,
        # Generation
        response_length=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        # PPO hyperparams
        cliprange=CLIP_EPS,
        cliprange_value=CLIP_EPS,
        vf_coef=VF_COEF,
        gamma=1.0,
        lam=1.0,
        whiten_rewards=False,
        kl_coef=0.0,  # No KL for self-play
        kl_estimator="k1",
        # Training length
        total_episodes=TOTAL_UPDATES * LOCAL_BATCH_SIZE,
        num_train_epochs=1.0,
        # Logging/saving
        report_to=[],
        logging_steps=1,
        save_steps=0.0,
        eval_steps=0.0,
        push_to_hub=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        disable_tqdm=False,
        num_sample_generations=0,
        # Adapter names
        model_adapter_name=None,
        ref_adapter_name=None,
    )
    # Store LoRA config for evaluate_vs_base
    setattr(ppo_args, "lora_rank", lora_rank)
    setattr(ppo_args, "lora_alpha", lora_rank * 2)

    # ---------- Create trainer ----------
    trainer = KuhnPokerPPOTrainer(
        args=ppo_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,  # kl_coef=0 so not needed
        reward_model=NoOpRewardModel().to(device),  # Unused
        train_dataset=dummy,
        value_model=value_model,
        eval_dataset=dummy.select(range(32)),
        # Kuhn Poker specific
        kuhn_env_cls=KuhnPoker,
        num_rounds=NUM_ROUNDS,
        base_adapter_dir=BASE_ADAPTER_DIR,
        eval_games=EVAL_GAMES,
        eval_every_steps=EVAL_EVERY,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # ---------- Train ----------
    trainer.train()

    # ---------- Final eval ----------
    final_r = trainer.evaluate_vs_random(num_games=200, temperature=0.0)
    final_b = trainer.evaluate_vs_base(num_games=200, temperature=0.0)
    if trainer.accelerator.is_main_process:
        print("[final eval] vs random:", final_r)
        print("[final eval] vs base:", final_b)

    # ---------- Save final model ----------
    if trainer.accelerator.is_main_process:
        final_dir = OUTPUT_DIR / "final_policy"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.accelerator.unwrap_model(trainer.model).policy.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        torch.save(value_model.score.state_dict(), str(final_dir / "value_head.pt"))
        print(f"[save] wrote: {final_dir}")


if __name__ == "__main__":
    main()
