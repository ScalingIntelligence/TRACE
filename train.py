import copy
import os
import json
import random
import re
import time
from pathlib import Path
from typing import Dict

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
# os.environ.setdefault(f"WANDB_ENTITY", f"hangoo94-stanford-university")
os.environ.setdefault("WANDB_PROJECT", "games")

import torch
from torch.utils.data import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

try:
    import wandb
except Exception:
    wandb = None

try:
    import trackio
except Exception:
    trackio = None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


class _EMA:
    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self.value = 0.0
        self.initialized = False

    def update(self, x: float):
        if not self.initialized:
            self.value = float(x)
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * float(x)


class _JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, payload: dict):
        self._fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._fh.flush()


_ACTION_RE = re.compile(r"\[(check|bet|call|fold)\]", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_action(text: str, legal_actions: list[str]) -> str | None:
    matches = _ACTION_RE.findall(text or "")
    if matches:
        for m in reversed(matches):
            a = f"[{m.lower()}]"
            if a in legal_actions:
                return a
    boxed = _BOXED_RE.findall(text or "")
    if boxed:
        for m in reversed(boxed):
            a = f"[{m.strip().lower()}]"
            if a in legal_actions:
                return a
    return None


_CARD_RANK = {"J": 0, "Q": 1, "K": 2}


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

    def legal_actions(self) -> list[str]:
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

    def to_json(self) -> str:
        d = {
            "num_rounds": self.num_rounds,
            "start_player0": self.start_player0,
            "round_cards": self.round_cards,
            "round_idx": self.round_idx,
            "chips": self.chips,
            "history": self.history,
            "actions_in_round": self.actions_in_round,
            "bet_by": self.bet_by,
            "current_player": self.current_player,
            "done": self.done,
            "invalid_player": self.invalid_player,
            "rewards": self.rewards,
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "KuhnPoker":
        d = json.loads(s)
        env = cls(num_rounds=int(d["num_rounds"]))
        env.start_player0 = int(d["start_player0"])
        env.round_cards = d["round_cards"]
        env.round_idx = int(d["round_idx"])
        env.chips = list(d["chips"])
        env.history = [tuple(x) for x in d["history"]]
        env.actions_in_round = [tuple(x) for x in d["actions_in_round"]]
        env.bet_by = d["bet_by"]
        env.current_player = int(d["current_player"])
        env.done = bool(d["done"])
        env.invalid_player = d["invalid_player"]
        env.rewards = {int(k): float(v) for k, v in d["rewards"].items()}
        return env


def _messages(player_id: int, observation: str):
    return [
        {
            "role": "system",
            "content": (
                "You are playing Kuhn Poker. Think step by step, then respond with exactly one legal action token "
                "in brackets: [check], [bet], [call], or [fold]."
            ),
        },
        {"role": "user", "content": observation},
    ]


# Cap generation length because valid actions are single tokens; large values
# were causing very long generations and GPU OOM.
MAX_ACTION_TOKENS = 512
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if wandb:
    if not os.getenv("WANDB_NAME"):
        os.environ["WANDB_NAME"] = f"gpt-oss-20b-kuhn-spiral-grpo-{int(time.time())}"
    wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
    wandb.init(
        # entity=os.getenv("WANDB_ENTITY", "hangoo94-stanford-university"),
        project=os.getenv("WANDB_PROJECT", "games"),
        name=os.getenv("WANDB_NAME"),
    )

ROLE_BASELINE_EMA_GAMMA = 0.95
_BASELINES = {0: _EMA(ROLE_BASELINE_EMA_GAMMA), 1: _EMA(ROLE_BASELINE_EMA_GAMMA)}
_LATEST_RL_EXTRA_LOGS: dict[str, float] = {}

USE_ROLE_BASELINE = True
FILTER_ZERO_ADV = True
REWARD_SCALING = 1.0
USE_INTERMEDIATE_REWARDS = True
REWARD_GAMMA = 1.0

ROLLOUT_LOG_PATH = Path(__file__).resolve().parent / "selfplay_rollouts.jsonl"


class _SelfPlayCollector:
    def __init__(
        self,
        num_rounds: int,
        temperature: float,
        max_new_tokens: int,
        log_path: Path,
        seed: int = 0,
    ):
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._rng = random.Random(seed)
        self._logger = _JSONLLogger(log_path)
        self._game_id = 0

    def _next_seed(self) -> int:
        return self._rng.randint(0, 2**31 - 1)

    def collect_games(self, num_games: int) -> list[dict]:
        samples: list[dict] = []
        for _ in range(num_games):
            game_id = self._game_id
            self._game_id += 1

            env = KuhnPoker(num_rounds=self.num_rounds)
            env.reset(self._next_seed())

            turn_records: list[dict] = []
            per_player_indices = {0: [], 1: []}
            turn_idx = 0

            while not env.done:
                pid = env.current_player
                obs = env.observe(pid)
                state_json = env.to_json()
                legal = env.legal_actions()

                turn_t_start = time.time()
                raw_response = _generate_action(pid, obs, self.temperature, self.max_new_tokens)
                turn_t_after_generate = time.time()
                action = _extract_action(raw_response, legal)
                action_valid = action in legal

                player_turn_idx = len(per_player_indices[pid])
                per_player_indices[pid].append(len(turn_records))
                turn_records.append(
                    {
                        "prompt": _messages(pid, obs),
                        "player_id": pid,
                        "state_json": state_json,
                        "game_id": game_id,
                        "turn_idx": turn_idx,
                        "player_turn_idx": player_turn_idx,
                    }
                )

                env.step(action)
                turn_idx += 1
                turn_t_end = time.time()

                self._logger.log(
                    {
                        "type": "step",
                        "game_id": game_id,
                        "turn_idx": turn_idx,
                        "player_id": pid,
                        "player_turn_idx": player_turn_idx,
                        "observation": obs,
                        "raw_response": raw_response,
                        "action": action,
                        "action_valid": action_valid,
                        "legal_actions": legal,
                        "duration_generate_sec": turn_t_after_generate - turn_t_start,
                        "duration_turn_sec": turn_t_end - turn_t_start,
                        "timestamp": time.time(),
                    }
                )

            rewards = env.rewards
            for pid in (0, 1):
                player_indices = per_player_indices[pid]
                total = len(player_indices)
                for idx_pos, record_idx in enumerate(player_indices):
                    turn_records[record_idx]["player_turns_total"] = total
                    turn_records[record_idx]["steps_from_end"] = total - idx_pos - 1
                    turn_records[record_idx]["final_reward"] = rewards[pid]

            self._logger.log(
                {
                    "type": "game_end",
                    "game_id": game_id,
                    "turns": turn_idx,
                    "rewards": rewards,
                    "invalid_player": env.invalid_player,
                    "timestamp": time.time(),
                }
            )

            samples.extend(turn_records)
        return samples


class _SelfPlayDataset(Dataset):
    def __init__(
        self,
        collector: _SelfPlayCollector,
        games_per_batch: int,
        size: int = 1024,
        reuse_count: int = 2,
    ):
        super().__init__()
        if size <= 0:
            raise ValueError("size must be positive")
        if reuse_count <= 0:
            raise ValueError("reuse_count must be positive")
        self.collector = collector
        self.games_per_batch = games_per_batch
        self.size = size
        self.reuse_count = reuse_count
        self._buffer: list[dict] = []
        self._cache: dict[int, dict] = {}
        self._cache_uses: dict[int, int] = {}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        if idx in self._cache:
            sample = self._cache[idx]
            self._cache_uses[idx] += 1
        else:
            if not self._buffer:
                self._buffer = self.collector.collect_games(self.games_per_batch)
            sample = self._buffer.pop(0)
            self._cache[idx] = sample
            self._cache_uses[idx] = 1

        if self._cache_uses[idx] >= self.reuse_count:
            self._cache.pop(idx, None)
            self._cache_uses.pop(idx, None)

        return copy.deepcopy(sample)


def _generate_action(player_id: int, observation: str, temperature: float, max_new_tokens: int) -> str:
    msgs = _messages(player_id, observation)
    device = next(model.parameters()).device
    input_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    if was_training:
        model.train()
    return tokenizer.decode(out_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)


def _rollout(env: KuhnPoker, temperature: float, max_new_tokens: int) -> tuple[dict, Dict[int, int], int]:
    turn_counts = {0: 0, 1: 0}
    turns = 0
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        while not env.done:
            pid = env.current_player
            obs = env.observe(pid)
            txt = _generate_action(pid, obs, temperature, max_new_tokens)
            act = _extract_action(txt, env.legal_actions())
            env.step(act)
            turns += 1
            turn_counts[pid] += 1
    if was_training:
        model.train()
    return env.rewards, turn_counts, turns


def spiral_kuhn_reward(
    prompts=None,
    completions=None,
    player_id=None,
    state_json=None,
    trainer_state=None,
    **kwargs,
):
    n = len(completions) if completions is not None else 0
    if n == 0 or state_json is None or player_id is None:
        return [0.0] * n

    def _tolist(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x] * n

    pids = _tolist(player_id)
    states = _tolist(state_json)
    if len(pids) != n and len(pids) > 0 and n % len(pids) == 0:
        k = n // len(pids)
        pids = [pid for pid in pids for _ in range(k)]
    if len(states) != n and len(states) > 0 and n % len(states) == 0:
        k = n // len(states)
        states = [st for st in states for _ in range(k)]

    rewards = []
    game_lens = 0
    invalids = 0
    wins_p0 = 0
    raw_sum = 0.0
    adv_sum = 0.0

    for comp, pid, st in zip(completions, pids, states):
        response = comp[0]["content"] if isinstance(comp, list) else str(comp)
        pid = int(pid)
        env = KuhnPoker.from_json(st)
        act = _extract_action(response, env.legal_actions())
        env.step(act)

        # Terminal reward only: give reward only if game ended, otherwise 0
        if env.done:
            raw_reward = float(env.rewards[pid]) * REWARD_SCALING
            total_turns = 1  # This action ended the game
        else:
            # Game continues - no reward until terminal state
            raw_reward = 0.0
            total_turns = 1

        base = _BASELINES[pid].value if USE_ROLE_BASELINE else 0.0
        if USE_ROLE_BASELINE and env.done:
            # Only update baseline on terminal rewards
            _BASELINES[pid].update(raw_reward)
            raw_reward -= base
        elif USE_ROLE_BASELINE:
            # For non-terminal actions, subtract baseline but reward is 0
            raw_reward = 0.0 - base

        # No intermediate reward discounting needed since we only give terminal rewards
        adv = raw_reward
        if FILTER_ZERO_ADV and adv == 0.0:
            adv = 0.0
        rewards.append(adv)

        game_lens += total_turns
        invalids += 1 if env.invalid_player is not None else 0
        if env.done:
            wins_p0 += 1 if env.rewards.get(0, 0.0) > 0 else 0
            raw_sum += float(env.rewards.get(pid, 0.0))
        adv_sum += adv

    global _LATEST_RL_EXTRA_LOGS
    _LATEST_RL_EXTRA_LOGS = {
        "env/game_len_mean": game_lens / max(1, n),
        "env/invalid_rate": invalids / max(1, n),
        "env/win_rate_p0": wins_p0 / max(1, n),
        "env/raw_reward_mean": raw_sum / max(1, n),
        "env/adv_mean": adv_sum / max(1, n),
        "baseline/p0": _BASELINES[0].value,
        "baseline/p1": _BASELINES[1].value,
    }

    return rewards


sample_env = KuhnPoker(num_rounds=5)
sample_env.reset(0)
sample_pid = sample_env.current_player
sample_prompt = _messages(sample_pid, sample_env.observe(sample_pid))
maximum_length = len(
    tokenizer.apply_chat_template(sample_prompt, add_generation_prompt=True, tokenize=True)
)
max_prompt_length = maximum_length + 1
max_completion_length = MAX_ACTION_TOKENS
if max_completion_length <= 0:
    raise ValueError(
        f"max_completion_length={max_completion_length} is not positive; reduce prompt size or increase max_seq_length"
    )

report_to = []
if wandb:
    report_to.append("wandb")
if trackio:
    report_to.append("trackio")
if not report_to:
    report_to = "none"

output_dir_path = ROOT / "workplace" / "games" / "outputs"
output_dir_path.mkdir(parents=True, exist_ok=True)

training_args = GRPOConfig(
    temperature=1.0,
    learning_rate=2e-4,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=600,
    save_steps=100,
    report_to=report_to,
    run_name=os.environ.get("WANDB_NAME", None),
    output_dir=str(output_dir_path),
    remove_unused_columns=False,
)

collector = _SelfPlayCollector(
    num_rounds=5,
    temperature=training_args.temperature,
    max_new_tokens=MAX_ACTION_TOKENS,
    log_path=ROLLOUT_LOG_PATH,
    seed=0,
)
prompt_batch = max(1, training_args.generation_batch_size // training_args.num_generations)
dataset_size = max(prompt_batch, prompt_batch * 16, 1024)
dataset_size -= dataset_size % prompt_batch
reuse_count = max(
    1,
    training_args.num_generations
    * training_args.num_iterations
    * (training_args.steps_per_generation or 1),
)
train_dataset = _SelfPlayDataset(
    collector,
    games_per_batch=2,
    size=dataset_size,
    reuse_count=reuse_count,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=spiral_kuhn_reward,
    args=training_args,
    train_dataset=train_dataset,
)

from transformers import TrainerCallback


class _ExtraRLMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        global _LATEST_RL_EXTRA_LOGS
        if isinstance(_LATEST_RL_EXTRA_LOGS, dict) and _LATEST_RL_EXTRA_LOGS:
            logs.update(_LATEST_RL_EXTRA_LOGS)


trainer.add_callback(_ExtraRLMetricsCallback())

trainer.train()
trainer.save_model("spiral_kuhn_grpo_model")
