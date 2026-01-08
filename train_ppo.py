
#!/usr/bin/env python3
# =========================
# PPO self-play for Kuhn Poker with STRICT action-only generation (no reasoning)
# - Keeps your cache + wandb + output_dir layout exactly
# - Fixes:
#   (1) gpt-oss MoE output handling (no last_hidden_state / hidden_states None)
#   (2) dtype mismatches (bf16 vs fp32) in value head + PPO math
#   (3) big memory spikes: pad PER-MINIBATCH + compute old_logp/old_v in chunks
# - NEW (your request):
#   (4) HARD "no reasoning": constrained decoding so model can ONLY output
#       exactly one of: [check] [bet] [call] [fold]
#       - HF local: prefix_allowed_tokens_fn + stop on exact action tokens
#       - vLLM: guided decoding (guided_choice) + stop strings
# - Tweaks:
#   - Lower MAX_GEN_TOKENS (action-only)
#   - Deterministic eval (temperature=0.0)
#   - Optionally normalize action logprob by action length (reduces variance)
# =========================
from unsloth import FastLanguageModel
import argparse
import copy
import gc
import os
import json
import random
import re
import time
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from datasets import load_from_disk

import sys
_HARNESS_PATH = Path(__file__).resolve().parent / "evals" / "benchmarks" / "math-evaluation-harness"
sys.path.insert(0, str(_HARNESS_PATH))

from grader import math_equal
from parser import extract_answer, strip_string, parse_ground_truth



# ------------------------------------------
# Optional: allocator fragmentation guard
# (safe even if already set outside)
# ------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# =========================
# Parse command-line arguments
# =========================
parser = argparse.ArgumentParser(description="PPO self-play for Kuhn Poker")
parser.add_argument(
    "--root",
    type=str,
    default=None,
    help="Root directory for cache, wandb, and outputs. Defaults to /matx/u/{USER} if not specified."
)
parser.add_argument(
    "--use_constrained_decoding",
    type=bool,
    default=False,
    help="If True, use constrained decoding to force action-only outputs. If False, use normal generation. Default: True"
)
args = parser.parse_args()
USE_CONSTRAINED_DECODING = bool(args.use_constrained_decoding)

# =========================
# Your exact cache + wandb layout
# =========================
if args.root is not None:
    ROOT = Path(args.root).resolve()
else:
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

output_dir_path = ROOT / "workplace" / "games" / "outputs"
output_dir_path.mkdir(parents=True, exist_ok=True)

ROLLOUT_LOG_PATH = Path(__file__).resolve().parent / "selfplay_rollouts_ppo.jsonl"

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


def _autocast_ctx():
    # Keep model forward bf16 when on GPU; PPO math will explicitly use fp32 where needed.
    if DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # CPU autocast isn’t helpful here
    return torch.no_grad()  # dummy context, overridden by callers


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

ACTION_STRS = ["[check]", "[bet]", "[call]", "[fold]"]


def _extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """
    Take the FIRST legal action found.
    With strict decoding, the completion should be exactly one action anyway.
    """
    matches = _ACTION_RE.findall(text or "")
    for m in matches:
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


# =========================
# STRICT action-only prompt + constrained decoding (HF local)
# =========================
SYSTEM_PROMPT = (
    "You are playing Kuhn Poker.\n"
    "Respond with EXACTLY ONE action token and NOTHING ELSE.\n"
    "Valid outputs: [check] or [bet] or [call] or [fold].\n"
    "Do not add any whitespace, punctuation, explanation, or extra text.\n"
)


def _messages(player_id: int, observation: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation},
    ]


def _encode_action_candidates(tokenizer) -> List[List[int]]:
    cands = []
    for s in ACTION_STRS:
        ids = tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        cands.append(ids)
    return cands


class StopOnAnyAction(StoppingCriteria):
    """
    Stop when the generated suffix exactly ends with any action candidate token sequence.
    (No newline reliance.)
    """
    def __init__(self, prompt_len: int, action_token_ids: List[List[int]]):
        self.prompt_len = int(prompt_len)
        self.action_token_ids = [list(x) for x in action_token_ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen = input_ids[0, self.prompt_len:].tolist()
        if not gen:
            return False
        for cand in self.action_token_ids:
            L = len(cand)
            if L > 0 and len(gen) >= L and gen[-L:] == cand:
                return True
        return False


def _make_prefix_allowed_fn(tokenizer, prompt_len: int, action_token_ids: List[List[int]]):
    """
    Constrained decoding:
    only allow tokens that can lead to one of the candidate action strings.
    This makes it impossible to emit reasoning.
    """
    cands = [tuple(c) for c in action_token_ids]

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.LongTensor):
        gen = input_ids.tolist()[prompt_len:]
        matching = []
        for cand in cands:
            if len(gen) <= len(cand) and tuple(gen) == cand[: len(gen)]:
                matching.append(cand)

        if not matching:
            # Fallback: allow EOS (should basically never happen)
            return [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        for cand in matching:
            if len(gen) == len(cand):
                return [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        allowed = set()
        for cand in matching:
            allowed.add(cand[len(gen)])
        return list(allowed)

    return prefix_allowed_tokens_fn


@torch.no_grad()
def _generate_completion(model, tokenizer, player_id: int, observation: str, temperature: float, max_new_tokens: int) -> str:
    """
    HF local generation. If use_constrained_decoding is True, constrained to output exactly one action string.
    Returns the action token string (e.g. "[bet]") when possible.
    """
    msgs = _messages(player_id, observation)
    input_ids = tokenizer.apply_chat_template(
        msgs, 
        add_generation_prompt=True, 
        return_tensors="pt",
        enable_thinking=False
    ).to(DEVICE)

    prompt_len = int(input_ids.shape[-1])
    
    was_training = model.training
    model.eval()

    do_sample = float(temperature) > 0.0

    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "temperature": float(temperature) if do_sample else None,
        "top_p": 1.0,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if USE_CONSTRAINED_DECODING:
        action_token_ids = _encode_action_candidates(tokenizer)
        stopper = StopOnAnyAction(prompt_len=prompt_len, action_token_ids=action_token_ids)
        prefix_allowed = _make_prefix_allowed_fn(tokenizer, prompt_len=prompt_len, action_token_ids=action_token_ids)
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed

    out_ids = model.generate(**generate_kwargs)

    if was_training:
        model.train()

    txt = tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)
    act = _extract_action(txt, ACTION_STRS)
    return act if act is not None else txt


# =========================
# vLLM backend: guided decoding (hard action-only)
# =========================
_VLLM_STOP_STRINGS = ["[check]", "[bet]", "[call]", "[fold]"]


def _normalize_vllm_openai_base_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return url + "/v1"


def _build_prompt_text(tokenizer, msgs: list) -> str:
    """
    Build a string prompt using the model's chat template.
    Uses string prompts for vLLM/OpenAI-server compatibility.
    """
    try:
        return tokenizer.apply_chat_template(
            msgs, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            msgs, 
            add_generation_prompt=True, 
            return_tensors="pt",
            enable_thinking=False
        )[0]
        return tokenizer.decode(ids, skip_special_tokens=False)


class _InferenceBackend:
    name: str = "base"

    def is_enabled(self) -> bool:
        return False

    def supports_batch(self) -> bool:
        return False

    def sync_policy(self, policy_model: nn.Module, adapter_dir: Path) -> None:
        return None

    def generate(self, prompts: List[str], temperature: float, max_new_tokens: int) -> List[str]:
        raise NotImplementedError


class _HFLocalBackend(_InferenceBackend):
    name = "hf_local"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def is_enabled(self) -> bool:
        return True

    def supports_batch(self) -> bool:
        # HF constrained decoding is easiest per-sample (fine for your current small GAMES_PER_ITER)
        return False

    def generate(self, prompts: List[str], temperature: float, max_new_tokens: int) -> List[str]:
        raise RuntimeError("_HFLocalBackend.generate should not be called with string prompts.")

    def generate_one(self, player_id: int, observation: str, temperature: float, max_new_tokens: int) -> str:
        return _generate_completion(self.model, self.tokenizer, player_id, observation, temperature, max_new_tokens)


class _VLLMServerBackend(_InferenceBackend):
    """Calls a running vllm serve OpenAI-compatible server."""

    name = "vllm_server"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
        lora_name: str = "ppo_policy",
        allow_lora_reload: bool = True,
    ):
        self.base_url = _normalize_vllm_openai_base_url(base_url)
        self.model_name = model_name
        self.lora_name = lora_name
        self.allow_lora_reload = bool(allow_lora_reload)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()
        self.headers: Dict[str, str] = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self._active_model_for_generation = self.lora_name if self.allow_lora_reload else self.model_name
        self._ok = self._probe_server()

    def _probe_server(self) -> bool:
        try:
            r = self.session.get(self.base_url + "/models", headers=self.headers, timeout=min(10.0, self.timeout_s))
            return r.status_code == 200
        except Exception:
            return False

    def is_enabled(self) -> bool:
        return self._ok

    def supports_batch(self) -> bool:
        return True

    def _post_json(self, path: str, payload: dict) -> requests.Response:
        return self.session.post(
            self.base_url + path,
            headers=self.headers,
            json=payload,
            timeout=self.timeout_s,
        )

    def sync_policy(self, policy_model: nn.Module, adapter_dir: Path) -> None:
        if not self._ok or not self.allow_lora_reload:
            return

        adapter_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(adapter_dir))

        try:
            # Unload if present (best-effort)
            self._post_json("/unload_lora_adapter", {"lora_name": self.lora_name})
            r = self._post_json(
                "/load_lora_adapter",
                {"lora_name": self.lora_name, "lora_path": str(adapter_dir)},
            )
            r.raise_for_status()
        except Exception as e:
            print(
                f"[vLLM] LoRA reload failed ({type(e).__name__}: {e}). "
                "Disabling vLLM backend and falling back to local HF generation for correctness."
            )
            self._ok = False

    def generate(self, prompts: List[str], temperature: float, max_new_tokens: int, mode: str = "poker") -> List[str]:
        if not self._ok:
            raise RuntimeError("vLLM server backend is not available")

        # HARD action-only via guided decoding (guided_choice) if constrained decoding is enabled
        payload = {
            "model": self._active_model_for_generation,
            "prompt": prompts,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": 1.0,
            "n": 1,
            "stream": False,
        }
        
        if mode == "poker":
            payload["stop"] = _VLLM_STOP_STRINGS
            payload["include_stop_str_in_output"] = True
            if USE_CONSTRAINED_DECODING:
                payload["extra_body"] = {
                    "guided_choice": ACTION_STRS,
                }

        r = self._post_json("/completions", payload)
        if r.status_code != 200:
            # If server doesn't support guided decoding, disable backend to avoid broken training logic.
            print(f"[vLLM] Generation failed (status={r.status_code}). Response: {r.text[:300]}")
            self._ok = False
            raise RuntimeError("vLLM guided decoding not available / request failed")

        data = r.json()
        out = [""] * len(prompts)
        for ch in data.get("choices", []):
            idx = int(ch.get("index", 0))
            if 0 <= idx < len(out):
                out[idx] = ch.get("text", "")
        return out


def _init_inference_backend(model, tokenizer) -> _InferenceBackend:
    """Pick the fastest available inference backend."""
    vllm_base_url = os.getenv("VLLM_BASE_URL", "").strip()
    if vllm_base_url:
        backend = _VLLMServerBackend(
            base_url=vllm_base_url,
            model_name=os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
            api_key=os.getenv("VLLM_API_KEY", "") or None,
            timeout_s=float(os.getenv("VLLM_TIMEOUT_S", "120")),
            lora_name=os.getenv("VLLM_LORA_NAME", "ppo_policy"),
            allow_lora_reload=os.getenv("VLLM_ALLOW_LORA_RELOAD", "1") == "1",
        )
        if backend.is_enabled():
            print(f"[vLLM] Using OpenAI-compatible server backend at {backend.base_url} (model={backend.model_name}).")
            return backend
        norm_url = _normalize_vllm_openai_base_url(vllm_base_url)
        print(f"[vLLM] Server at {norm_url} not reachable; falling back to local HF generation.")
    return _HFLocalBackend(model, tokenizer)


# =========================
# PPO actor-critic wrapper (value head)
# =========================
def _unwrap_backbone(causal_lm: nn.Module) -> nn.Module:
    m = causal_lm
    for _ in range(3):
        if hasattr(m, "model") and isinstance(getattr(m, "model"), nn.Module):
            m = m.model
        else:
            break
    return m


def _get_lm_head(causal_lm: nn.Module) -> nn.Module:
    if hasattr(causal_lm, "lm_head") and isinstance(causal_lm.lm_head, nn.Module):
        return causal_lm.lm_head
    head = causal_lm.get_output_embeddings()
    if head is None:
        raise ValueError("Could not find lm_head / output embeddings.")
    return head


class PolicyWithValueHead(nn.Module):
    """
    Memory-friendly:
    - backbone forward returns last hidden
    - logits = lm_head(last_hidden)
    - value head runs in fp32 on pooled hidden state
    """
    def __init__(self, causal_lm: nn.Module):
        super().__init__()
        self.lm = causal_lm
        self.backbone = _unwrap_backbone(causal_lm)
        self.lm_head = _get_lm_head(causal_lm)

        cfg = getattr(self.lm, "config", None)
        hidden_size = None
        for attr in ("hidden_size", "n_embd", "d_model"):
            if cfg is not None and hasattr(cfg, attr):
                hidden_size = int(getattr(cfg, attr))
                break
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model.config")

        self.value_head = nn.Linear(hidden_size, 1).to(dtype=torch.float32)

    def forward(self, input_ids, attention_mask=None):
        with (_autocast_ctx() if DEVICE == "cuda" else torch.no_grad()):
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            last_hidden = out.last_hidden_state
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            last_hidden = out[0]
        else:
            raise RuntimeError(f"Backbone output has no last hidden state. Type={type(out)}")

        logits = self.lm_head(last_hidden)
        return logits, last_hidden


# =========================
# Batching helpers (pad per minibatch)
# =========================
def _pad_to_device(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(s.shape[0] for s in seqs)
    bsz = len(seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        ids[i, :L] = s
        attn[i, :L] = 1
    return ids.to(DEVICE, non_blocking=True), attn.to(DEVICE, non_blocking=True)


def _build_prompt_plus_action(tokenizer, prompt_msgs: list, action_str: str) -> Tuple[torch.Tensor, int, int]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, 
        add_generation_prompt=True, 
        return_tensors="pt",
        enable_thinking=False
    )[0]
    action_ids = tokenizer(action_str, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return torch.cat([prompt_ids, action_ids], dim=0), int(prompt_ids.shape[0]), int(action_ids.shape[0])


def _logprob_action_tokens(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: List[int],
    action_lens: List[int],
    normalize_by_len: bool = True,   # NEW: reduces variance / clip spikes
) -> torch.Tensor:
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


def _values_from_hidden(last_hidden: torch.Tensor, value_head: nn.Module, prompt_lens: List[int]) -> torch.Tensor:
    B, T, H = last_hidden.shape
    hs = []
    for i in range(B):
        idx = max(0, min(T - 1, prompt_lens[i] - 1))
        hs.append(last_hidden[i, idx, :])
    hs = torch.stack(hs, dim=0)
    v = value_head(hs.float()).squeeze(-1)
    return v


MATH_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)

def _math_messages(question: str): 
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

def extract_boxed_answer(text: str) -> str:
    """ Extract the model's answer from \\boxed{...}. """

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
) -> Dict[str, float]:
    """Evaluate model on the math benchmark (grading using harness)"""

    #Loading the dataset.
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

        msgs = _math_messages(question)
        prompt_text = _build_prompt_text(tokenizer, msgs)
        prompts.append(prompt_text)
        ground_truths.append(ground_truth)

    if not prompts:
        return {f"eval_math/{dataset_name}_accuracy": 0.0}


    if backend is not None and backend.is_enabled() and backend.supports_batch():
        completions = backend.generate(prompts, temperature=temperature, max_new_tokens=max_new_tokens, mode = "math")
    else:
        print("falling back")
        was_training = model.training
        model.eval()
        completions = []
        for prompt in tqdm(prompts, desc=f"Math eval ({dataset_name})"):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            out_ids = model.generate(
                input_ids = input_ids,
                max_new_tokens = max_new_tokens,
                do_sample=(temperature > 0),
                temperature = max(temperature, 0.01),
                top_p = 1.0,
                use_cache = True,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            completion = tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
            completions.append(completion)

            del input_ids, out_ids
            if DEVICE == "cuda":
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
    
    accuracy = correct/ max(1, total_evaluated)

    return {
        f"eval_math/{dataset_name}_accuracy": accuracy,
        f"eval_math/{dataset_name}_correct": correct,
        f"eval_math/{dataset_name}_total": total_evaluated,
    }






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
# Self-play collector
# =========================
def collect_games(
    model,
    tokenizer,
    backend: _InferenceBackend,
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

    if backend.supports_batch():
        envs: List[KuhnPoker] = []
        game_ids: List[int] = []
        episode_steps: List[List[Tuple[list, str, int, str]]] = []
        turn_idxs = [0 for _ in range(num_games)]

        for g in range(num_games):
            game_id = seed * 1_000_000 + g
            env = KuhnPoker(num_rounds=num_rounds)
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
                msgs = _messages(pid, obs)
                prompts.append(_build_prompt_text(tokenizer, msgs))
                meta.append((i, pid, obs, legal, msgs))

            t0 = time.time()
            completions = backend.generate(prompts, temperature=temperature, max_new_tokens=max_new_tokens)
            t1 = time.time()
            per_item_dt = (t1 - t0) / max(1, len(active))

            for j, (i, pid, obs, legal, msgs) in enumerate(meta):
                completion = completions[j]
                act = _extract_action(completion, legal)
                if act is None:
                    act = rng.choice(legal)

                episode_steps[i].append((msgs, act, pid, completion))
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
                    "duration_generate_sec": per_item_dt,
                    "timestamp": time.time(),
                })

        for i, env in enumerate(envs):
            invalid_games += 1 if env.invalid_player is not None else 0
            p0_wins += 1 if env.rewards.get(0, 0.0) > 0 else 0

            logger.log({
                "type": "game_end",
                "game_id": game_ids[i],
                "turns": turn_idxs[i],
                "rewards": env.rewards,
                "invalid_player": env.invalid_player,
                "timestamp": time.time(),
            })

            for pm, act, pid, completion in episode_steps[i]:
                samples.append(StepSample(
                    prompt_msgs=pm,
                    action_str=act,
                    player_id=pid,
                    ret=float(env.rewards[pid]),
                    completion_text=completion,
                    game_id=game_ids[i],
                ))
    else:
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
                completion = _generate_completion(model, tokenizer, pid, obs, temperature=temperature, max_new_tokens=max_new_tokens)
                t1 = time.time()

                act = _extract_action(completion, legal)
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
                    game_id=game_id,
                ))

    metrics = {
        "env/invalid_game_rate": invalid_games / max(1, num_games),
        "env/turns_per_game_mean": total_turns / max(1, num_games),
        "env/win_rate_p0": p0_wins / max(1, num_games),
    }
    return samples, metrics


# =========================
# Lightweight evaluation vs random opponent
# =========================
@torch.no_grad()
def evaluate_vs_random(
    current_model,
    tokenizer,
    backend: _InferenceBackend,
    num_games: int,
    num_rounds: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(int(seed))
    half = max(1, num_games // 2)

    envs: List[KuhnPoker] = []
    current_is_p0: List[bool] = []
    for i in range(num_games):
        env = KuhnPoker(num_rounds=num_rounds)
        env.reset(rng.randint(0, 2**31 - 1))
        envs.append(env)
        current_is_p0.append(i < half)

    turn_counts = [0 for _ in range(num_games)]

    while True:
        active = [i for i, e in enumerate(envs) if not e.done]
        if not active:
            break

        policy_idxs: List[int] = []
        prompts: List[str] = []
        meta: List[Tuple[int, int, List[str], str]] = []

        for i in active:
            env = envs[i]
            pid = env.current_player
            is_policy_turn = (pid == 0 and current_is_p0[i]) or (pid == 1 and (not current_is_p0[i]))
            if not is_policy_turn:
                continue
            obs = env.observe(pid)
            legal = env.legal_actions()
            msgs = _messages(pid, obs)
            policy_idxs.append(i)
            prompts.append(_build_prompt_text(tokenizer, msgs))
            meta.append((i, pid, legal, obs))

        if prompts:
            if backend.supports_batch():
                completions = backend.generate(prompts, temperature=temperature, max_new_tokens=max_new_tokens)
            else:
                completions = [
                    _generate_completion(current_model, tokenizer, pid, obs, temperature=temperature, max_new_tokens=max_new_tokens)
                    for (_, pid, _legal, obs) in meta
                ]

            for j, (i, pid, legal, _obs) in enumerate(meta):
                completion = completions[j]
                act = _extract_action(completion, legal)
                if act is None:
                    act = rng.choice(legal)
                envs[i].step(act)
                turn_counts[i] += 1

        for i in active:
            env = envs[i]
            if env.done:
                continue
            pid = env.current_player
            is_policy_turn = (pid == 0 and current_is_p0[i]) or (pid == 1 and (not current_is_p0[i]))
            if is_policy_turn:
                continue
            legal = env.legal_actions()
            env.step(rng.choice(legal))
            turn_counts[i] += 1

    wins_current = 0
    invalids = 0
    total_turns = sum(turn_counts)
    for i, env in enumerate(envs):
        invalids += 1 if env.invalid_player is not None else 0
        current_won = (env.rewards[0] > 0) if current_is_p0[i] else (env.rewards[1] > 0)
        wins_current += 1 if current_won else 0

    return {
        "eval/win_rate_vs_random": wins_current / max(1, num_games),
        "eval/invalid_game_rate": invalids / max(1, num_games),
        "eval/turns_per_game_mean": total_turns / max(1, num_games),
    }


@torch.no_grad()
def evaluate_vs_base(
    current_model,
    base_model_adapter_dir: Path,
    tokenizer,
    backend: _InferenceBackend,
    num_games: int,
    num_rounds: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> Dict[str, float]:
    """
    Evaluate current trained model against the base (untrained) model.
    Loads base model + base adapter each eval (simple, correct; costs time).
    """
    rng = random.Random(int(seed))

    base_model, _ = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        max_seq_length=768,
        dtype=None,
        load_in_4bit=True,
        offload_embedding=True,
        cache_dir=str(HF_HUB),
    )
    base_model = FastLanguageModel.get_peft_model(
        base_model,
        r=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    base_model.load_adapter(str(base_model_adapter_dir), adapter_name="base")
    base_model.set_adapter("base")
    base_model = base_model.to(DEVICE)
    base_model.eval()

    half = max(1, num_games // 2)
    envs: List[KuhnPoker] = []
    current_is_p0: List[bool] = []
    for i in range(num_games):
        env = KuhnPoker(num_rounds=num_rounds)
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
            msgs = _messages(pid, obs)
            current_prompts.append(_build_prompt_text(tokenizer, msgs))
            current_meta.append((i, pid, legal, obs))

        if current_prompts:
            if backend.supports_batch():
                completions = backend.generate(current_prompts, temperature=temperature, max_new_tokens=max_new_tokens)
            else:
                completions = [
                    _generate_completion(current_model, tokenizer, pid, obs, temperature=temperature, max_new_tokens=max_new_tokens)
                    for (_, pid, _legal, obs) in current_meta
                ]
            for j, (i, pid, legal, _obs) in enumerate(current_meta):
                completion = completions[j]
                act = _extract_action(completion, legal)
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
                _generate_completion(base_model, tokenizer, pid, obs, temperature=temperature, max_new_tokens=max_new_tokens)
                for (_, pid, _legal, obs) in base_meta
            ]
            for j, (i, pid, legal, _obs) in enumerate(base_meta):
                completion = completions[j]
                act = _extract_action(completion, legal)
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
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "eval/win_rate_vs_base": wins_current / max(1, num_games),
        "eval/invalid_game_rate_vs_base": invalids / max(1, num_games),
        "eval/turns_per_game_mean_vs_base": total_turns / max(1, num_games),
    }


# =========================
# Load model + LoRA
# =========================
max_seq_length = 768
lora_rank = 4

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
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

ac = PolicyWithValueHead(model).to(DEVICE)

# Save initial base model adapter state for evaluation
BASE_MODEL_ADAPTER_DIR = output_dir_path / "base_model_adapter"
BASE_MODEL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
ac.lm.save_pretrained(str(BASE_MODEL_ADAPTER_DIR))
print(f"[Base Model] Saved initial adapter state to {BASE_MODEL_ADAPTER_DIR}")

inference_backend = _init_inference_backend(ac.lm, tokenizer)
VLLM_ADAPTER_DIR = output_dir_path / "vllm_adapter_latest"

# =========================
# wandb init (kept)
# =========================
if wandb:
    if not os.getenv("WANDB_NAME"):
        os.environ["WANDB_NAME"] = f"qwen3-4b-kuhn-spiral-ppo-{int(time.time())}"
    wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
    wandb.init(
        entity="forge_scaling_intelligence_lab",
        project="games",
    )

# =========================
# PPO hyperparams
# =========================
NUM_ROUNDS = 5
GAMES_PER_ITER = 8
PPO_EPOCHS = 2
MINI_BATCH_SIZE = 16
STATS_CHUNK_SIZE = 2

LR = 1e-6
CLIP_EPS = 0.2
VF_COEF = 0.5

# NEW: action-only decoding => keep this small
MAX_GEN_TOKENS = 8
TEMPERATURE = 0.7
MAX_TOKENS_MATH_EVAL = 3072

SAVE_EVERY_ITERS = 20
EVAL_EVERY_ITERS = 20
EVAL_GAMES = 25

MATH_EVAL_DATA_PATH = Path(__file__).resolve().parent / "data"
MATH_EVAL_DATASETS = ["math", "amc", "aime"]
MATH_EVAL_SAMPLES = 50
MATH_EVAL_EVERY_ITERS = EVAL_EVERY_ITERS #can change if math slow



trainable_params = [p for p in ac.parameters() if p.requires_grad]
optim = torch.optim.AdamW(trainable_params, lr=LR)

rollout_logger = _JSONLLogger(ROLLOUT_LOG_PATH)

# =========================
# Main PPO loop
# =========================
global_step = 0

for it in tqdm(range(10_000), desc="PPO iters"):
    t_collect0 = time.time()

    inference_backend.sync_policy(ac.lm, VLLM_ADAPTER_DIR)
    if not inference_backend.is_enabled():
        inference_backend = _HFLocalBackend(ac.lm, tokenizer)

    

    batch, env_metrics = collect_games(
        model=ac.lm,
        tokenizer=tokenizer,
        backend=inference_backend,
        num_games=GAMES_PER_ITER,
        num_rounds=NUM_ROUNDS,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_GEN_TOKENS,
        seed=it,
        logger=rollout_logger,
    )
    t_collect1 = time.time()

    random.shuffle(batch)

    # Pre-tokenize ON CPU
    seqs_cpu: List[torch.Tensor] = []
    prompt_lens: List[int] = []
    action_lens: List[int] = []
    returns_cpu = torch.empty((len(batch),), dtype=torch.float32)

    for i, s in enumerate(batch):
        ids, pL, aL = _build_prompt_plus_action(tokenizer, s.prompt_msgs, s.action_str)
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
        for start in range(0, N, STATS_CHUNK_SIZE):
            mb_idx = list(range(start, min(N, start + STATS_CHUNK_SIZE)))
            mb_seqs = [seqs_cpu[j] for j in mb_idx]
            mb_prompt_lens = [prompt_lens[j] for j in mb_idx]
            mb_action_lens = [action_lens[j] for j in mb_idx]

            mb_ids, mb_attn = _pad_to_device(mb_seqs, tokenizer.pad_token_id)

            logits, last_h = ac(input_ids=mb_ids, attention_mask=mb_attn)
            mb_old_logp = _logprob_action_tokens(logits, mb_ids, mb_prompt_lens, mb_action_lens, normalize_by_len=True)
            mb_old_v = _values_from_hidden(last_h, ac.value_head, mb_prompt_lens)

            old_logp_cpu[mb_idx] = mb_old_logp.detach().cpu()
            old_v_cpu[mb_idx] = mb_old_v.detach().cpu()

            del mb_ids, mb_attn, logits, last_h, mb_old_logp, mb_old_v
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

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

    for _epoch in range(PPO_EPOCHS):
        random.shuffle(idxs)
        for start in range(0, N, MINI_BATCH_SIZE):
            mb = idxs[start:start + MINI_BATCH_SIZE]
            mb_seqs = [seqs_cpu[i] for i in mb]
            mb_prompt_lens = [prompt_lens[i] for i in mb]
            mb_action_lens = [action_lens[i] for i in mb]

            mb_ids, mb_attn = _pad_to_device(mb_seqs, tokenizer.pad_token_id)

            mb_returns = returns_cpu[mb].to(DEVICE)
            mb_old_logp = old_logp_cpu[mb].to(DEVICE)
            mb_adv = adv_cpu[mb].to(DEVICE)

            logits, last_h = ac(input_ids=mb_ids, attention_mask=mb_attn)

            new_logp = _logprob_action_tokens(logits, mb_ids, mb_prompt_lens, mb_action_lens, normalize_by_len=True)
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

            with torch.no_grad():
                approx_kl = torch.mean(mb_old_logp - new_logp)
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > CLIP_EPS).float())
                ratio_mean = torch.mean(ratio)

            global_step += 1
            updates += 1
            policy_loss_acc += float(policy_loss.item())
            value_loss_acc += float(value_loss.item())
            approx_kl_acc += float(approx_kl.item())
            clip_frac_acc += float(clip_frac.item())
            ratio_mean_acc += float(ratio_mean.item())

            del mb_ids, mb_attn, logits, last_h, new_logp, new_v, ratio, unclipped, clipped, loss
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

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
        f"invalid={env_metrics['env/invalid_game_rate']:.3f} "
        f"KL={logs['ppo/approx_kl']:.4f} clip={logs['ppo/clip_frac']:.3f} "
        f"collect={logs['time/collect_sec']:.1f}s train={logs['time/train_sec']:.1f}s"
    )

    if wandb:
        wandb.log(logs, step=global_step)

    

    # Periodic eval (deterministic; no reasoning possible anyway)
    if it % EVAL_EVERY_ITERS == 0:
        ac.eval()

        inference_backend.sync_policy(ac.lm, VLLM_ADAPTER_DIR)
        if not inference_backend.is_enabled():
            inference_backend = _HFLocalBackend(ac.lm, tokenizer)

        print(f"[eval {it}] Starting eval vs random ({EVAL_GAMES} games)")
        t_eval_random0 = time.time()
        eval_logs_random = evaluate_vs_random(
            current_model=ac.lm,
            tokenizer=tokenizer,
            backend=inference_backend,
            num_games=EVAL_GAMES,
            num_rounds=NUM_ROUNDS,
            temperature=0.0,          # deterministic
            max_new_tokens=MAX_GEN_TOKENS,
            seed=10_000 + it,
        )
        t_eval_random1 = time.time()
        win_rate_random = eval_logs_random.get("eval/win_rate_vs_random", 0.0)
        print(f"[eval {it}] vs random: win_rate={win_rate_random:.3f} ({t_eval_random1-t_eval_random0:.1f}s)")

        print(f"[eval {it}] Starting eval vs base model ({EVAL_GAMES} games)")
        t_eval_base0 = time.time()
        eval_logs_base = evaluate_vs_base(
            current_model=ac.lm,
            base_model_adapter_dir=BASE_MODEL_ADAPTER_DIR,
            tokenizer=tokenizer,
            backend=inference_backend,
            num_games=EVAL_GAMES,
            num_rounds=NUM_ROUNDS,
            temperature=0.0,          # deterministic
            max_new_tokens=MAX_GEN_TOKENS,
            seed=20_000 + it,
        )
        t_eval_base1 = time.time()
        win_rate_base = eval_logs_base.get("eval/win_rate_vs_base", 0.0)
        print(f"[eval {it}] vs base: win_rate={win_rate_base:.3f} ({t_eval_base1-t_eval_base0:.1f}s)")

        all_eval_logs = {
            **eval_logs_random,
            **eval_logs_base,
            "eval/time_vs_random_sec": t_eval_random1 - t_eval_random0,
            "eval/time_vs_base_sec": t_eval_base1 - t_eval_base0,
            "eval/improvement_vs_base": win_rate_base - 0.5,
            "eval/improvement_vs_random": win_rate_random - 0.5,
        }

        print(f"[eval {it}] Summary: vs_random={win_rate_random:.3f}, vs_base={win_rate_base:.3f}")
        if win_rate_base > 0.5:
            print(f"[eval {it}] ✓ Model is better than base model!")
        if win_rate_random > 0.5:
            print(f"[eval {it}] ✓ Model is better than random!")

        if wandb:
            wandb.log(all_eval_logs, step=global_step)

    
     # 6a) Math evaluation (can be added to block if we decide to use same frequency)

    if it % MATH_EVAL_EVERY_ITERS == 0 and it > 0:
        ac.eval()
        print(f"[eval {it}] Starting math benchmark evaluation...")

        all_math_logs = {}
        for ds_name in MATH_EVAL_DATASETS:
            math_start = time.time()
            math_logs = evaluate_math(
                model = ac.lm,
                tokenizer = tokenizer,
                data_path = MATH_EVAL_DATA_PATH,
                dataset_name = ds_name,
                num_samples = MATH_EVAL_SAMPLES,
                temperature = 0.0,
                max_new_tokens = MAX_TOKENS_MATH_EVAL,
                backend = inference_backend,
            )

            math_end = time.time()

            acc = math_logs.get(f"eval_math/{ds_name}_accuracy", 0.0)

            print(f"[eval {it}] {ds_name}: accuracy={acc:.3f} ({math_end-math_start:.1f}s)")
            
            all_math_logs.update(math_logs)
            all_math_logs[f"eval_math/{ds_name}_time_sec"] = math_end - math_start
        
        if wandb:
            wandb.log(all_math_logs, step=global_step)


    
    # Save checkpoints
    if it % SAVE_EVERY_ITERS == 0:
        ckpt_dir = output_dir_path / f"ppo_ckpt_iter_{it}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ac.lm.save_pretrained(str(ckpt_dir / "policy"))
        tokenizer.save_pretrained(str(ckpt_dir / "policy"))
        torch.save(ac.value_head.state_dict(), str(ckpt_dir / "value_head.pt"))

# Final save
final_dir = output_dir_path / "spiral_kuhn_ppo_model"
final_dir.mkdir(parents=True, exist_ok=True)
ac.lm.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
torch.save(ac.value_head.state_dict(), str(final_dir / "value_head.pt"))
