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

from config import Config
from game_registry import GameSpec



# =========================
# Prompt building
# =========================
def messages_for_game(player_id: int, observation: str, game_spec: GameSpec, env=None) -> list:
    """Build messages for a game.

    If env supports structured messages (e.g. adversarial_policy), uses
    the game's native system prompt + conversation format for better
    training→eval alignment.
    """
    if env is not None and getattr(env, 'supports_structured_messages', False):
        system = env.get_system_prompt()
        conv = env.get_messages()
        return [{"role": "system", "content": system}] + conv

    system_prompt = game_spec.system_prompt or "You are playing a 2-player game. Respond with exactly one action."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]


def tools_for_game(env=None, compact: bool = False) -> Optional[list]:
    """Get OpenAI-format tool schemas from env if it supports structured messages.

    When tools are returned, they should be passed to apply_chat_template(tools=...)
    so the tokenizer formats them identically to how vLLM's /chat/completions
    endpoint formats them during evaluation.

    Args:
        compact: If True, return compressed schemas (descriptions stripped) for
            training. Reduces prompt length by ~60% with no loss of structural info.
    """
    if env is not None and getattr(env, 'supports_structured_messages', False):
        if compact and hasattr(env, 'get_tool_schemas_compact'):
            schemas = env.get_tool_schemas_compact()
        else:
            schemas = env.get_tool_schemas()
        return schemas if schemas else None
    return None


def build_prompt_text(tokenizer, msgs: list, tools: Optional[list] = None) -> str:
    """
    Build a string prompt using the model's chat template.
    Uses string prompts for vLLM/OpenAI-server compatibility.

    When tools is provided, passes it to apply_chat_template(tools=...)
    so Qwen3's template formats them as <tools>...</tools> XML — matching
    exactly what vLLM's /chat/completions does during evaluation.
    """
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=Config.ENABLE_THINKING,
    )
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(msgs, **kwargs)
    except TypeError:
        kwargs.pop("tokenize", None)
        kwargs["return_tensors"] = "pt"
        ids = tokenizer.apply_chat_template(msgs, **kwargs)[0]
        return tokenizer.decode(ids, skip_special_tokens=False)


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
    device: str,
    game_spec: Optional[GameSpec] = None,
    env=None,
    compact_tools: bool = False,
) -> str:
    """
    HF local generation. Returns the full completion text.
    """
    game_spec = game_spec or GameSpec(
        name="default",
        make_env=lambda: None,  # type: ignore
        extract_action=lambda _t, _l: None,
        action_space=[],
        system_prompt="You are playing a 2-player game. Respond with one action.",
        max_gen_tokens=max_new_tokens,
    )
    msgs = messages_for_game(player_id, observation, game_spec, env=env)
    tools = tools_for_game(env, compact=compact_tools)
    kwargs = dict(
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=Config.ENABLE_THINKING,
    )
    if tools:
        kwargs["tools"] = tools
    input_ids = tokenizer.apply_chat_template(msgs, **kwargs).to(device)

    prompt_len = int(input_ids.shape[-1])

    was_training = model.training
    model.eval()

    do_sample = float(temperature) > 0.0

    out_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        top_p=1.0,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

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
        mode: str = "game",
    ) -> List[str]:
        raise NotImplementedError


class HFLocalBackend(InferenceBackend):
    """Local HuggingFace generation backend."""
    name = "hf_local"

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
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
    ):
        self.base_url = normalize_vllm_openai_base_url(base_url)
        self.model_name = model_name
        self.lora_name = lora_name
        self.allow_lora_reload = bool(allow_lora_reload)
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
        mode: str = "game",
        adapter_name: Optional[str] = None,
    ) -> List[str]:
        if not self._ok:
            raise RuntimeError("vLLM server backend is not available")

        model_to_use = adapter_name if adapter_name else self._active_model_for_generation
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
            if stop:
                payload["stop"] = stop
                payload["include_stop_str_in_output"] = True

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

class MultiVLLMBackend(InferenceBackend):
    """ Balances inference across more than 1 vLLM server. """
    name = "multi_vllm"

    def __init__(self, backends: List[VLLMServerBackend]):
        self.backends = [b for b in backends if b.is_enabled()]
        if not self.backends:
            raise RuntimeError("No vLLM servers are enabled")
        
    def is_enabled(self) -> bool:
        return any(b.is_enabled() for b in self.backends)

    def supports_batch(self) -> bool:
        return self.is_enabled()

    def has_base_adapter(self) -> bool:
        return all(b.has_base_adapter() for b in self.backends)
    
    @property
    def base_lora_name(self) -> str:
        """Return base_lora_name from first backend."""
        return self.backends[0].base_lora_name

        
    def sync_policy(self, policy_model: nn.Module, adapter_dir: Path) -> None:
        """ Sync policy to each backend """
        for backend in self.backends:
            backend.sync_policy(policy_model, adapter_dir)
    
    def sync_base_adapter(self, adapter_dir: Path) -> bool:
        """ Sync base adapter to each backend """

        results = [backend.sync_base_adapter(adapter_dir) for backend in self.backends]
        return all(results)


    def generate(
        self,
        prompts: List[str],
        temperature: float,
        max_new_tokens: int,
        game_spec = None,
        mode: str = "game",
        adapter_name: Optional[str] = None,
    ) -> List[str]:
        """ Split prompts across backends and generate in parallel. """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_backends = len(self.backends)
        n_prompts = len(prompts)

        chunk_size = (n_prompts + n_backends - 1) // n_backends
        chunks = []
        for i in range(n_backends):
            start = i * chunk_size
            end = min(start + chunk_size, n_prompts)
            if start < end:
                chunks.append((i, prompts[start:end]))

        results = {}

        def generate_chunk(idx, backend, chunk):
            return idx, backend.generate(
                    chunk,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    game_spec=game_spec,
                    mode=mode,
                    adapter_name=adapter_name,
                )
        
        with ThreadPoolExecutor(max_workers=n_backends) as executor:
            futures= [
                executor.submit(generate_chunk, idx, self.backends[idx % len(self.backends)], chunk)
                for idx, chunk in chunks
            ]
            for future in as_completed(futures):
                idx, completions = future.result()
                results[idx] = completions
        #have to flatten back results
        output = []
        for i in range(len(chunks)):
            output.extend(results[i])
        
        return output

# =========================
# Backend initialization
# =========================
def init_inference_backend(model, tokenizer, device: str) -> InferenceBackend:
    """Pick the fastest available inference backend."""

    vllm_base_urls = os.getenv("VLLM_BASE_URLS", "").strip()
    vllm_base_url = os.getenv("VLLM_BASE_URL", "").strip()

    if vllm_base_urls:
        urls = [u.strip() for u in vllm_base_urls.split(",")]
    elif vllm_base_url:
        urls = [vllm_base_url]
    else:
        urls = []

    if urls:
        model_name = os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        api_key = os.getenv("VLLM_API_KEY", "") or None
        timeout_s = float(os.getenv("VLLM_TIMEOUT_S", "1200"))
        lora_name = os.getenv("VLLM_LORA_NAME", "ppo_policy")
        allow_lora_reload = os.getenv("VLLM_ALLOW_LORA_RELOAD", "1") == "1"

        backends = []
        for url in urls:
            backend = VLLMServerBackend(
                base_url=url,
                model_name=model_name,
                api_key=api_key,
                timeout_s=timeout_s,
                lora_name=lora_name,
                allow_lora_reload=allow_lora_reload,
            )

            if backend.is_enabled():
                print(f"[vLLM] Connected to server at {backend.base_url}")
                backends.append(backend)
            else:
                print(f"[vLLM] Server at {url} not reachable, skipping")

        if len(backends) > 1:
            print(f"[vLLM] Using {len(backends)} servers for parallel inference")
            return MultiVLLMBackend(backends) 
        elif len(backends) == 1:
            print(f"[vLLM] Using single server at {backends[0].base_url}")
            return backends[0]
        else:
            print("[vLLM] No servers reachable, falling back to local HF generation")
    
    return HFLocalBackend(model, tokenizer, device)
