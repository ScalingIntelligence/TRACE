import copy
import os
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList

from unsloth import FastLanguageModel

# =========================
# Your exact cache + wandb layout
# =========================
ROOT = Path(f"/matx/u/{os.getenv('USER')}").resolve()
HF_HOME = ROOT / ".cache" / "huggingface"
HF_HUB = HF_HOME / "hub"
HF_DATASETS = HF_HOME / "datasets"

for p in (HF_HOME, HF_HUB, HF_DATASETS):
    p.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS))

WANDB_DIR = ROOT / "workplace" / "games" / "wandb"
WANDB_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_DIR", str(WANDB_DIR))
os.environ.setdefault("WANDB_PROJECT", "games")

# output_dir layout (same as yours)
output_dir_path = ROOT / "workplace" / "games" / "outputs"
output_dir_path.mkdir(parents=True, exist_ok=True)

# rollouts log next to this file OR under outputs (choose your preference)
ROLLOUT_LOG_PATH = Path(__file__).resolve().parent / "selfplay_rollouts_ppo.jsonl"
# ROLLOUT_LOG_PATH = output_dir_path / "selfplay_rollouts.jsonl"

# =========================
# Optional tracking libs (same style)
# =========================
try:
    import wandb
except Exception:
    wandb = None

try:
    import trackio
except Exception:
    trackio = None

# =========================
# Torch perf knobs
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# =========================
# Logging helper
# =========================
class _JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, payload: dict):
        self._fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._fh.flush()

# =========================
# Kuhn Poker env (your logic)
# =========================
_ACTION_RE = re.compile(r"\[(check|bet|call|fold)\]", re.IGNORECASE)
_CARD_RANK = {"J": 0, "Q": 1, "K": 2}

def _extract_action(text: str, legal_actions: List[str]) -> str | None:
    matches = _ACTION_RE.findall(text or "")
    if matches:
        for m in reversed(matches):
            a = f"[{m.lower()}]"
            if a in legal_actions:
                return a
    return None

class KuhnPoker:
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
        # Keep your invalid penalty scheme
        self.rewards = {0: 0.5, 1: 0.5}
        self.rewards[player_id] = -1.5
        self.rewards[other] = 0.5

    def step(self, action: str | None):
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

# =========================
# Reasoning prompt (allowed) + early stop on FINAL action line
# =========================
SYSTEM_PROMPT = (
    "You are playing Kuhn Poker.\n"
    "Think step by step and you MAY include reasoning.\n"
    "Important:\n"
    "- Do NOT include any bracketed action tokens in your reasoning.\n"
    "- After reasoning, output exactly one FINAL action token on its own line: "
    "[check] or [bet] or [call] or [fold]."
)

def _messages(player_id: int, observation: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation},
    ]

_FINAL_ACTION_LINE_RE = re.compile(r"\n\[(check|bet|call|fold)\]\s*$", re.IGNORECASE)

class StopOnFinalActionLine(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, tail_tokens: int = 96):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.tail_tokens = int(tail_tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen = input_ids[0, self.prompt_len:]
        if gen.numel() == 0:
            return False
        tail = gen[-self.tail_tokens:].tolist()
        txt = self.tokenizer.decode(tail, skip_special_tokens=True)
        return _FINAL_ACTION_LINE_RE.search(txt) is not None

@torch.no_grad()
def _generate_completion(player_id: int, observation: str, temperature: float, max_new_tokens: int) -> str:
    msgs = _messages(player_id, observation)
    input_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    stopper = StopOnFinalActionLine(tokenizer, prompt_len=input_ids.shape[-1], tail_tokens=96)

    was_training = model.training
    model.eval()
    out_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stopper]),
    )
    if was_training:
        model.train()
    return tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

# =========================
# PPO actor-critic wrapper (value head)
# =========================
class PolicyWithValueHead(nn.Module):
    """
    Adds V(s) head. V(s) is computed from last hidden state at the last prompt token.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        cfg = getattr(base_model, "config", None)

        hidden_size = None
        for attr in ("hidden_size", "n_embd", "d_model"):
            if cfg is not None and hasattr(cfg, attr):
                hidden_size = int(getattr(cfg, attr))
                break
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model.config")

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )

def _pad_batch(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(s.shape[0] for s in seqs)
    bsz = len(seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        ids[i, :L] = s
        attn[i, :L] = 1
    return ids.to(DEVICE), attn.to(DEVICE)

def _build_prompt_plus_action(prompt_msgs: list, action_str: str) -> Tuple[torch.Tensor, int, int]:
    """
    We train only on the action token(s), NOT on the reasoning text.
    """
    prompt_ids = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, return_tensors="pt")[0]
    action_ids = tokenizer(action_str, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return torch.cat([prompt_ids, action_ids], dim=0), int(prompt_ids.shape[0]), int(action_ids.shape[0])

def _logprob_action_tokens(logits: torch.Tensor,
                           input_ids: torch.Tensor,
                           prompt_lens: List[int],
                           action_lens: List[int]) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    B, T, V = logp.shape
    out = torch.zeros((B,), device=logp.device)
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
        out[i] = s
    return out

def _values_from_hidden(last_hidden: torch.Tensor, value_head: nn.Module, prompt_lens: List[int]) -> torch.Tensor:
    B, T, H = last_hidden.shape
    hs = []
    for i in range(B):
        idx = max(0, min(T - 1, prompt_lens[i] - 1))
        hs.append(last_hidden[i, idx, :])
    hs = torch.stack(hs, dim=0)
    return value_head(hs).squeeze(-1)

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

# =========================
# Self-play collector (ONE rollout per episode)
# =========================
def collect_games(
    num_games: int,
    num_rounds: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    logger: _JSONLLogger,
) -> Tuple[List[StepSample], Dict[str, float]]:
    rng = random.Random(int(seed))
    samples: List[StepSample] = []

    invalid_games = 0
    total_turns = 0
    p0_wins = 0

    for g in range(num_games):
        game_id = seed * 1_000_000 + g
        env = KuhnPoker(num_rounds=num_rounds)
        env.reset(rng.randint(0, 2**31 - 1))

        episode_steps: List[Tuple[list, str, int, str]] = []
        turn_idx = 0

        while not env.done:
            pid = env.current_player
            obs = env.observe(pid)
            legal = env.legal_actions()

            t0 = time.time()
            completion = _generate_completion(pid, obs, temperature=temperature, max_new_tokens=max_new_tokens)
            t1 = time.time()

            act = _extract_action(completion, legal)

            # If parsing fails, choose random legal to avoid constant invalid terminations early.
            # If you prefer to "teach" invalid via penalty, set this to `pass` and let env terminate invalid.
            if act is None:
                act = rng.choice(legal)

            episode_steps.append((_messages(pid, obs), act, pid, completion))
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

        # terminal
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

        for pm, act, pid, completion in episode_steps:
            samples.append(StepSample(
                prompt_msgs=pm,
                action_str=act,
                player_id=pid,
                ret=float(env.rewards[pid]),
                completion_text=completion,
            ))

    metrics = {
        "env/invalid_game_rate": invalid_games / max(1, num_games),
        "env/turns_per_game_mean": total_turns / max(1, num_games),
        "env/win_rate_p0": p0_wins / max(1, num_games),
    }
    return samples, metrics

# =========================
# Load model + LoRA (same base model as yours)
# =========================
max_seq_length = 768
lora_rank = 4

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
    offload_embedding=True,
    cache_dir=str(HF_HUB),
)
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
FastLanguageModel.for_training(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = model.to(DEVICE)

# Wrap with value head (critic)
ac = PolicyWithValueHead(model).to(DEVICE)

# =========================
# Exact wandb init style you asked to keep
# =========================
if wandb:
    if not os.getenv("WANDB_NAME"):
        os.environ["WANDB_NAME"] = f"gpt-oss-20b-kuhn-spiral-ppo-{int(time.time())}"
    wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "games"),
        name=os.getenv("WANDB_NAME"),
    )

# =========================
# PPO hyperparams (tune these)
# =========================
NUM_ROUNDS = 5
GAMES_PER_ITER = 32          # games collected per iteration
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64

LR = 2e-4
CLIP_EPS = 0.2
VF_COEF = 0.5

# Reasoning + action budget per move; early-stopper usually ends much earlier.
MAX_GEN_TOKENS = 256
TEMPERATURE = 1.0

# Save cadence
SAVE_EVERY_ITERS = 20

# Optimizer over trainable params (LoRA + value_head)
trainable_params = [p for p in ac.parameters() if p.requires_grad]
optim = torch.optim.AdamW(trainable_params, lr=LR)

# Logger
rollout_logger = _JSONLLogger(ROLLOUT_LOG_PATH)

# =========================
# Main PPO loop
# =========================
global_step = 0

for it in range(10_000):
    # 1) Collect self-play episodes (ONE rollout per game)
    t_collect0 = time.time()
    batch, env_metrics = collect_games(
        num_games=GAMES_PER_ITER,
        num_rounds=NUM_ROUNDS,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_GEN_TOKENS,
        seed=it,
        logger=rollout_logger,
    )
    t_collect1 = time.time()

    random.shuffle(batch)

    # 2) Build tensors: prompt + action only (train on action token(s))
    seqs: List[torch.Tensor] = []
    prompt_lens: List[int] = []
    action_lens: List[int] = []
    returns: List[float] = []

    for s in batch:
        ids, pL, aL = _build_prompt_plus_action(s.prompt_msgs, s.action_str)
        seqs.append(ids)
        prompt_lens.append(pL)
        action_lens.append(aL)
        returns.append(s.ret)

    input_ids, attn = _pad_batch(seqs, tokenizer.pad_token_id)
    returns_t = torch.tensor(returns, device=DEVICE, dtype=torch.float32)

    # 3) Old policy stats (behavior = current policy before update)
    ac.eval()
    with torch.no_grad():
        out = ac(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        logits = out.logits
        last_h = out.hidden_states[-1]

        old_logp = _logprob_action_tokens(logits, input_ids, prompt_lens, action_lens)
        old_v = _values_from_hidden(last_h, ac.value_head, prompt_lens)

        adv = (returns_t - old_v).detach()
        # advantage normalization helps a lot
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    # 4) PPO update
    ac.train()
    N = input_ids.shape[0]
    idxs = list(range(N))

    policy_loss_acc = 0.0
    value_loss_acc = 0.0
    updates = 0

    for epoch in range(PPO_EPOCHS):
        random.shuffle(idxs)
        for start in range(0, N, MINI_BATCH_SIZE):
            mb = idxs[start:start + MINI_BATCH_SIZE]
            mb_ids = input_ids[mb]
            mb_attn = attn[mb]
            mb_returns = returns_t[mb]
            mb_old_logp = old_logp[mb]
            mb_adv = adv[mb]

            out = ac(input_ids=mb_ids, attention_mask=mb_attn, output_hidden_states=True)
            logits = out.logits
            last_h = out.hidden_states[-1]

            mb_prompt_lens = [prompt_lens[i] for i in mb]
            mb_action_lens = [action_lens[i] for i in mb]

            new_logp = _logprob_action_tokens(logits, mb_ids, mb_prompt_lens, mb_action_lens)
            new_v = _values_from_hidden(last_h, ac.value_head, mb_prompt_lens)

            ratio = torch.exp(new_logp - mb_old_logp)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            value_loss = 0.5 * F.mse_loss(new_v, mb_returns)

            loss = policy_loss + VF_COEF * value_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optim.step()

            global_step += 1
            updates += 1
            policy_loss_acc += float(policy_loss.item())
            value_loss_acc += float(value_loss.item())

    t_train1 = time.time()

    # 5) Metrics + logging
    policy_loss_mean = policy_loss_acc / max(1, updates)
    value_loss_mean = value_loss_acc / max(1, updates)
    avg_return = float(returns_t.mean().item())
    abs_return = float(returns_t.abs().mean().item())

    logs = {
        "iter": it,
        "global_step": global_step,
        "ppo/policy_loss": policy_loss_mean,
        "ppo/value_loss": value_loss_mean,
        "ppo/avg_return": avg_return,
        "ppo/avg_abs_return": abs_return,
        "time/collect_sec": t_collect1 - t_collect0,
        "time/train_sec": t_train1 - t_collect1,
        "time/iter_sec": t_train1 - t_collect0,
        **env_metrics,
    }

    print(
        f"[iter {it}] step={global_step} "
        f"avg_return={avg_return:.3f} invalid_game_rate={env_metrics['env/invalid_game_rate']:.3f} "
        f"collect={logs['time/collect_sec']:.1f}s train={logs['time/train_sec']:.1f}s"
    )

    if wandb:
        wandb.log(logs, step=global_step)

    # 6) Save checkpoints into your output_dir layout
    if it % SAVE_EVERY_ITERS == 0:
        ckpt_dir = output_dir_path / f"ppo_ckpt_iter_{it}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # save policy (LoRA adapters included) + tokenizer
        ac.model.save_pretrained(str(ckpt_dir / "policy"))
        tokenizer.save_pretrained(str(ckpt_dir / "policy"))
        # save value head weights
        torch.save(ac.value_head.state_dict(), str(ckpt_dir / "value_head.pt"))

# Final save (same “save_model name” semantics as your old script)
final_dir = output_dir_path / "spiral_kuhn_ppo_model"
final_dir.mkdir(parents=True, exist_ok=True)
ac.model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
torch.save(ac.value_head.state_dict(), str(final_dir / "value_head.pt"))
