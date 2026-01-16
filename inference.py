#!/usr/bin/env python3
"""
Inference backends and generation utilities.
"""
import os
import requests
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
from transformers import StoppingCriteria, StoppingCriteriaList

from config import Config
from game_registry import GameSpec



# =========================
# Prompt building
# =========================
def messages_for_game(player_id: int, observation: str, game_spec: GameSpec) -> list:
    """Build messages for a game."""
    system_prompt = game_spec.system_prompt or "You are playing a 2-player game. Respond with exactly one action."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]


def messages_for_math(question: str) -> list:
    """Build messages for math problem."""
    return [
        {"role": "system", "content": Config.MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_prompt_text(tokenizer, msgs: list) -> str:
    """
    Build a string prompt using the model's chat template.
    Uses string prompts for vLLM/OpenAI-server compatibility.
    """
    try:
        return tokenizer.apply_chat_template(
            msgs, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=Config.ENABLE_THINKING
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            msgs, 
            add_generation_prompt=True, 
            return_tensors="pt",
            enable_thinking=Config.ENABLE_THINKING
        )[0]
        return tokenizer.decode(ids, skip_special_tokens=False)


# =========================
# Constrained decoding helpers
# =========================
def encode_action_candidates(tokenizer, action_space: List[str]) -> List[List[int]]:
    """Encode action strings to token IDs."""
    cands = []
    for s in action_space:
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


def make_prefix_allowed_fn(tokenizer, prompt_len: int, action_token_ids: List[List[int]]):
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


# =========================
# HF local generation
# =========================
@torch.no_grad()
def generate_completion(
    model, 
    tokenizer, 
    player_id: int, 
    observation: str, 
    temperature: float, 
    max_new_tokens: int,
    use_constrained_decoding: bool,
    device: str,
    game_spec: Optional[GameSpec] = None,
) -> str:
    """
    HF local generation. If use_constrained_decoding is True, constrained to output exactly one action string.
    Returns the action token string (e.g. "[bet]") when possible.
    """
    game_spec = game_spec or GameSpec(
        name="default",
        make_env=lambda: None,  # type: ignore
        extract_action=lambda _t, _l: None,
        action_space=[],
        system_prompt="You are playing a 2-player game. Respond with one action.",
        max_gen_tokens=max_new_tokens,
    )
    msgs = messages_for_game(player_id, observation, game_spec)
    input_ids = tokenizer.apply_chat_template(
        msgs, 
        add_generation_prompt=True, 
        return_tensors="pt",
        enable_thinking=Config.ENABLE_THINKING
    ).to(device)

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

    should_constrain = bool(use_constrained_decoding and game_spec.action_space)
    if should_constrain:
        action_token_ids = encode_action_candidates(tokenizer, game_spec.action_space)
        stopper = StopOnAnyAction(prompt_len=prompt_len, action_token_ids=action_token_ids)
        prefix_allowed = make_prefix_allowed_fn(tokenizer, prompt_len=prompt_len, action_token_ids=action_token_ids)
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed


    out_ids = model.generate(**generate_kwargs)

    if was_training:
        model.train()

    txt = tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)
    return txt


# =========================
# Inference backend abstraction
# =========================
def normalize_vllm_openai_base_url(base_url: str) -> str:
    """Normalize vLLM OpenAI-compatible base URL."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return url + "/v1"


class InferenceBackend:
    """Base class for inference backends."""
    name: str = "base"

    def is_enabled(self) -> bool:
        return False

    def supports_batch(self) -> bool:
        return False

    def sync_policy(self, policy_model: nn.Module, adapter_dir: Path) -> None:
        return None

    def generate(
        self,
        prompts: List[str],
        temperature: float,
        max_new_tokens: int,
        game_spec: Optional[GameSpec] = None,
        use_guided_choice: bool = False,
        mode: str = "game",
    ) -> List[str]:
        raise NotImplementedError


class HFLocalBackend(InferenceBackend):
    """Local HuggingFace generation backend."""
    name = "hf_local"

    def __init__(self, model, tokenizer, use_constrained_decoding: bool, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.use_constrained_decoding = use_constrained_decoding
        self.device = device

    def is_enabled(self) -> bool:
        return True

    def supports_batch(self) -> bool:
        # HF constrained decoding is easiest per-sample (fine for your current small GAMES_PER_ITER)
        return False

    def generate(
        self,
        prompts: List[str],
        temperature: float,
        max_new_tokens: int,
        game_spec: Optional[GameSpec] = None,
        use_guided_choice: bool = False,
        mode: str = "game",
    ) -> List[str]:
        raise RuntimeError("HFLocalBackend.generate should not be called with string prompts.")

    def generate_one(
        self,
        player_id: int,
        observation: str,
        temperature: float,
        max_new_tokens: int,
        game_spec: Optional[GameSpec] = None,
    ) -> str:
        return generate_completion(
            self.model, 
            self.tokenizer, 
            player_id, 
            observation, 
            temperature, 
            max_new_tokens,
            self.use_constrained_decoding,
            self.device,
            game_spec=game_spec,
        )


class VLLMServerBackend(InferenceBackend):
    """Calls a running vllm serve OpenAI-compatible server."""

    name = "vllm_server"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
        lora_name: str = "ppo_policy",
        base_lora_name: str = "base_policy",
        allow_lora_reload: bool = True,
        use_constrained_decoding: bool = False,
    ):
        self.base_url = normalize_vllm_openai_base_url(base_url)
        self.model_name = model_name
        self.lora_name = lora_name
        self.allow_lora_reload = bool(allow_lora_reload)
        self.use_constrained_decoding = use_constrained_decoding
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()
        self.headers: Dict[str, str] = {}
        self.base_lora_name = base_lora_name
        self._base_adapter_loaded = False

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self._active_model_for_generation = self.lora_name if self.allow_lora_reload else self.model_name
        self._ok = self._probe_server()
        self._server_reachable_but_no_lora = False
        # Fail fast if the server is reachable but doesn't expose the LoRA runtime
        # endpoints we rely on for PPO (hot-reloading the adapter each iteration).
        if self._ok and self.allow_lora_reload and not self._probe_lora_runtime_api():
            print(
                "[vLLM] Server is reachable, but LoRA runtime endpoints are unavailable. "
                "PPO training requires /v1/load_lora_adapter to hot-reload the policy adapter.\n"
                "Fix: start vLLM with --enable-lora and set VLLM_ALLOW_RUNTIME_LORA_UPDATING=True "
                "in the *server* environment before running `vllm serve`.\n"
                "Falling back to local HF generation."
            )
            self._server_reachable_but_no_lora = True
            self._ok = False

    def _probe_server(self) -> bool:
        try:
            r = self.session.get(self.base_url + "/models", headers=self.headers, timeout=min(10.0, self.timeout_s))
            return r.status_code == 200
        except Exception:
            return False

    def _probe_lora_runtime_api(self) -> bool:
        """Return True if the server exposes the runtime LoRA management endpoints.

        vLLM only enables these endpoints when both:
          - the server is started with --enable-lora, and
          - the server env sets VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
        """
        try:
            # If the route exists, a GET will typically return 405 (method not allowed)
            # while a missing route returns 404.
            r = self.session.get(
                self.base_url + "/load_lora_adapter",
                headers=self.headers,
                timeout=min(10.0, self.timeout_s),
            )
            return r.status_code != 404
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
            # Give a more actionable message for the common misconfiguration.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                hint = (
                    " Server returned 404 for /v1/load_lora_adapter. "
                    "This almost always means the vLLM server was started without runtime LoRA updating enabled. "
                    "Start the server with --enable-lora and set VLLM_ALLOW_RUNTIME_LORA_UPDATING=True "
                    "in the server environment."
                )
            else:
                hint = ""
            print(
                f"[vLLM] LoRA reload failed ({type(e).__name__}: {e}). "
                f"Disabling vLLM backend and falling back to local HF generation for correctness.{hint}"
            )
            self._ok = False

    def sync_base_adapter(self, adapter_dir: Path) -> bool:
        """ Load the base adapter to vLLM a single time at the start """
        if not self._ok or not self.allow_lora_reload:
            return False
        if self._base_adapter_loaded:
            return True
        try:
            self._post_json("/unload_lora_adapter", {"lora_name": self.base_lora_name})
            r = self._post_json(
                "/load_lora_adapter",
                {"lora_name": self.base_lora_name, "lora_path": str(adapter_dir)},
            )
            r.raise_for_status()
            self._base_adapter_loaded = True
            print(f"[vLLM] Loaded base adapter '{self.base_lora_name}'")
            return True
        except Exception as e:
            print(f"[vLLM] Failed to load base adapter: {e}")
            return False

    def has_base_adapter(self) -> bool:
        return self._base_adapter_loaded

    def generate(
        self,
        prompts: List[str],
        temperature: float,
        max_new_tokens: int,
        game_spec: Optional[GameSpec] = None,
        use_guided_choice: bool = False,
        mode: str = "game",
        adapter_name: Optional[str] = None,
    ) -> List[str]:
        if not self._ok:
            raise RuntimeError("vLLM server backend is not available")

        model_to_use = adapter_name if adapter_name else self._active_model_for_generation
        # HARD action-only via guided decoding (guided_choice) if constrained decoding is enabled
        payload = {
            "model": model_to_use,
            "prompt": prompts,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": 1.0,
            "n": 1,
            "stream": False,
        }
        
        if mode == "game" and game_spec is not None:
            stop = game_spec.stop_sequences if game_spec.stop_sequences else None
            guided_choice = None
            if game_spec.action_space and (use_guided_choice or self.use_constrained_decoding):
                guided_choice = game_spec.action_space

            if stop:
                payload["stop"] = stop
                payload["include_stop_str_in_output"] = True
            if guided_choice:
                payload.setdefault("extra_body", {})
                payload["extra_body"]["guided_choice"] = guided_choice

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


# =========================
# Backend initialization
# =========================
def init_inference_backend(model, tokenizer, use_constrained_decoding: bool, device: str) -> InferenceBackend:
    """Pick the fastest available inference backend."""
    vllm_base_url = os.getenv("VLLM_BASE_URL", "").strip()
    if vllm_base_url:
        backend = VLLMServerBackend(
            base_url=vllm_base_url,
            model_name=os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
            api_key=os.getenv("VLLM_API_KEY", "") or None,
            timeout_s=float(os.getenv("VLLM_TIMEOUT_S", "120")),
            lora_name=os.getenv("VLLM_LORA_NAME", "ppo_policy"),
            allow_lora_reload=os.getenv("VLLM_ALLOW_LORA_RELOAD", "1") == "1",
            use_constrained_decoding=use_constrained_decoding,
        )
        if backend.is_enabled():
            print(f"[vLLM] Using OpenAI-compatible server backend at {backend.base_url} (model={backend.model_name}).")
            return backend
        # Check if server was reachable but missing LoRA endpoints
        if hasattr(backend, '_server_reachable_but_no_lora') and backend._server_reachable_but_no_lora:
            # Error message already printed in VLLMServerBackend.__init__
            pass
        else:
            norm_url = normalize_vllm_openai_base_url(vllm_base_url)
            print(f"[vLLM] Server at {norm_url} not reachable; falling back to local HF generation.")
    return HFLocalBackend(model, tokenizer, use_constrained_decoding, device)
