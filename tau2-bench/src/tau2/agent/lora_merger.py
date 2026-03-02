"""
LoRA merging utilities for the orchestrator + skill LLMs system.

Supports:
  - Linear weighted merge: base + Σ αᵢ * LoRAᵢ via PEFT
  - Caching merged adapters by combination hash (avoids redundant merging)
  - Loading/unloading adapters to running vLLM servers via HTTP API
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from loguru import logger


class LoRAMerger:
    """
    Merges multiple LoRA adapters with weights using PEFT and caches results.

    Cache key = SHA256(base_model + sorted[(adapter_path, weight), ...])
    """

    def __init__(self, base_model: str, cache_dir: str = "/tmp/lora_merge_cache"):
        self.base_model = base_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, adapters: List[Tuple[str, float]]) -> str:
        """Deterministic cache key from base model + adapter combination."""
        sorted_adapters = sorted(adapters, key=lambda x: x[0])
        key_data = json.dumps(
            {"base": self.base_model, "adapters": sorted_adapters},
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def adapter_name_for(self, adapters: List[Tuple[str, float]]) -> str:
        """Return a unique adapter name for a given combination."""
        key = self._cache_key(adapters)
        return f"merged_{key}"

    def get_or_merge(self, adapters: List[Tuple[str, float]]) -> Path:
        """
        Get cached merged adapter or merge and cache.

        Args:
            adapters: List of (adapter_path, weight) tuples.

        Returns:
            Path to the merged adapter directory.
        """
        key = self._cache_key(adapters)
        cached_path = self.cache_dir / key

        # Check cache
        if cached_path.exists() and (cached_path / "adapter_config.json").exists():
            logger.info(f"LoRA merge cache hit: {key}")
            return cached_path

        logger.info(
            f"Merging {len(adapters)} LoRA adapters "
            f"(key={key}, base={self.base_model})"
        )

        return self._merge(adapters, cached_path)

    def _merge(
        self, adapters: List[Tuple[str, float]], output_dir: Path
    ) -> Path:
        """
        Merge multiple LoRA adapters using PEFT.

        Strategy: Linear weighted merge
          merged = base + Σ αᵢ * LoRAᵢ

        The merge is done on CPU to avoid consuming GPU memory.
        """
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                f"LoRA merging requires peft and transformers: {e}. "
                "Install with: pip install peft transformers"
            )

        # Load base model on CPU
        logger.info(f"Loading base model: {self.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        if len(adapters) == 1:
            # Single adapter: just load and optionally scale
            adapter_path, weight = adapters[0]
            logger.info(f"Loading single adapter: {adapter_path} (weight={weight})")
            model = PeftModel.from_pretrained(
                base_model, adapter_path, adapter_name="default"
            )
            if weight != 1.0:
                self._scale_adapter(model, "default", weight)
            # Merge adapter into base weights and save as a LoRA adapter
            merged = model.merge_and_unload()
        else:
            # Multiple adapters: load each, scale, then merge
            adapter_names = []
            weights = []

            for i, (adapter_path, weight) in enumerate(adapters):
                adapter_name = f"skill_{i}"
                adapter_names.append(adapter_name)
                weights.append(weight)

                if i == 0:
                    logger.info(
                        f"Loading adapter 0: {adapter_path} (weight={weight})"
                    )
                    model = PeftModel.from_pretrained(
                        base_model, adapter_path, adapter_name=adapter_name
                    )
                else:
                    logger.info(
                        f"Loading adapter {i}: {adapter_path} (weight={weight})"
                    )
                    model.load_adapter(adapter_path, adapter_name=adapter_name)

            # Use PEFT's add_weighted_adapter for proper merging
            logger.info(
                f"Merging adapters {adapter_names} with weights {weights}"
            )
            model.add_weighted_adapter(
                adapters=adapter_names,
                weights=weights,
                adapter_name="merged",
                combination_type="linear",
            )
            model.set_adapter("merged")
            merged = model.merge_and_unload()

        # Save the merged model's adapter weights
        # We save as a full model then extract what's needed
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(output_dir))
        logger.info(f"Merged adapter saved to {output_dir}")

        # Clean up to free memory
        del merged, base_model
        if "model" in dir():
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return output_dir

    @staticmethod
    def _scale_adapter(model, adapter_name: str, weight: float) -> None:
        """Scale all LoRA parameters of a named adapter by weight."""
        model.set_adapter(adapter_name)
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data *= weight


# ---------------------------------------------------------------------------
# vLLM adapter management
# ---------------------------------------------------------------------------


def load_adapter_to_vllm(
    base_url: str, adapter_name: str, adapter_path: str
) -> bool:
    """Load a LoRA adapter to a running vLLM server."""
    server_url = base_url.rstrip("/").replace("/v1", "")
    load_url = f"{server_url}/v1/load_lora_adapter"

    payload = {"lora_name": adapter_name, "lora_path": adapter_path}

    try:
        response = requests.post(load_url, json=payload, timeout=60)
        if response.status_code == 200:
            logger.info(f"Loaded adapter '{adapter_name}' to vLLM at {server_url}")
            return True
        elif "already exists" in response.text.lower():
            logger.info(f"Adapter '{adapter_name}' already loaded on {server_url}")
            return True
        else:
            logger.error(
                f"Failed to load adapter '{adapter_name}': "
                f"{response.status_code} - {response.text}"
            )
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading adapter '{adapter_name}': {e}")
        return False


def unload_adapter_from_vllm(base_url: str, adapter_name: str) -> bool:
    """Unload a LoRA adapter from a running vLLM server."""
    server_url = base_url.rstrip("/").replace("/v1", "")
    unload_url = f"{server_url}/v1/unload_lora_adapter"

    payload = {"lora_name": adapter_name}

    try:
        response = requests.post(unload_url, json=payload, timeout=60)
        if response.status_code == 200:
            logger.info(f"Unloaded adapter '{adapter_name}' from {server_url}")
            return True
        else:
            logger.warning(
                f"Failed to unload adapter '{adapter_name}': "
                f"{response.status_code} - {response.text}"
            )
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error unloading adapter '{adapter_name}': {e}")
        return False


def ensure_skill_adapters_loaded(
    skills: list, api_base: Optional[str] = None
) -> None:
    """
    Pre-load all individual skill LoRA adapters to the vLLM server.

    Args:
        skills: List of skill dicts or SkillConfig objects with adapter_path/adapter_name.
        api_base: Default vLLM api_base URL if not specified per-skill.
    """
    for skill in skills:
        if hasattr(skill, "adapter_path"):
            adapter_path = skill.adapter_path
            adapter_name = skill.adapter_name
            skill_api_base = (skill.llm_args or {}).get("api_base", api_base)
        else:
            adapter_path = skill.get("adapter_path")
            adapter_name = skill.get("adapter_name")
            skill_api_base = (skill.get("llm_args") or {}).get("api_base", api_base)

        if adapter_path and adapter_name and skill_api_base:
            load_adapter_to_vllm(skill_api_base, adapter_name, adapter_path)
