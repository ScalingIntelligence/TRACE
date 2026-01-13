#!/usr/bin/env python3
"""
Inference backends for model comparison.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import httpx
from typing import Dict, Tuple, Optional
from openai import OpenAI
from transformers import AutoTokenizer
import os

def normalize_vllm_base_url(base_url: str) -> str:
    """Normalize vLLM OpenAI-compatible base URL."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return url + "/v1"


class InferenceBackend:
    """Base class for inference backends."""
    
    def generate_action(
        self,
        observation: str,
        env_id: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate an action from the model.
        
        Args:
            observation: The observation string to send to the model
            env_id: Environment ID
            model: Model name
            temperature: Temperature for sampling
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (action, error_message). If error_message is not None, action is None.
        """
        raise NotImplementedError


class OpenAIBackend(InferenceBackend):
    """OpenAI API backend."""
    
    def __init__(self, api_key: str, pool_connections: int = os.cpu_count(), pool_maxsize: int = os.cpu_count()):
        self.api_key = api_key
        # Create custom httpx client with larger connection pool
        limits = httpx.Limits(max_keepalive_connections=pool_connections, max_connections=pool_maxsize)
        http_client = httpx.Client(limits=limits, timeout=httpx.Timeout(60.0, connect=10.0))
        self.client = OpenAI(
            api_key=api_key,
            http_client=http_client
        )
    
    def generate_action(
        self,
        observation: str,
        env_id: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate action using OpenAI API."""
        if max_tokens is None:
            if any(model.startswith(prefix) for prefix in ["gpt-5", "o1", "o3"]):
                max_tokens = 5000
            else:
                max_tokens = 500
        
        system_prompt = (
            f"You are playing a game called {env_id}. "
            "Read the observation carefully and respond with ONLY your action in the required format. "
            "Do not explain your reasoning, just provide the action."
        )
        
        no_system_message = any([
            model.startswith("o1"),
            model.startswith("o3"),
        ])
        
        uses_completion_tokens = any([
            model.startswith("gpt-4o"),
            model.startswith("gpt-5"),
            model.startswith("o1"),
            model.startswith("o3"),
        ])
        
        no_custom_temperature = any([
            model.startswith("o1"),
            model.startswith("o3"),
            model.startswith("gpt-5"),
        ])
        
        if no_system_message:
            combined_message = f"{system_prompt}\n\n{observation}"
            messages = [{"role": "user", "content": combined_message}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": observation}
            ]
        
        request_params = {
            "model": model,
            "messages": messages,
        }
        
        if not no_custom_temperature:
            request_params["temperature"] = temperature
        
        if uses_completion_tokens:
            request_params["max_completion_tokens"] = max_tokens
        else:
            request_params["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**request_params)
        action = response.choices[0].message.content
        
        if action is None or (isinstance(action, str) and action.strip() == ""):
            return "", None
        
        return action.strip(), None


class VLLMBackend(InferenceBackend):
    """vLLM server backend."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout_s: float = 120.0, model_name: str = None, pool_connections: int = os.cpu_count(), pool_maxsize: int = os.cpu_count(), enable_thinking: bool = False):
        self.base_url = normalize_vllm_base_url(base_url)
        self.timeout_s = float(timeout_s)
        self.enable_thinking = enable_thinking
        self.session = requests.Session()
        
        # Configure HTTP adapter with larger connection pool
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.headers: Dict[str, str] = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _post_json(self, path: str, payload: dict) -> requests.Response:
        """Make a POST request to the vLLM server."""
        return self.session.post(
            self.base_url + path,
            headers=self.headers,
            json=payload,
            timeout=self.timeout_s,
        )
    
    def generate_action(
        self,
        observation: str,
        env_id: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate action using vLLM server."""
        if max_tokens is None:
            max_tokens = 500
        
        system_prompt = (
            f"You are playing a game called {env_id}. "
            "Read the observation carefully and respond with ONLY your action in the required format. "
            "Do not explain your reasoning, just provide the action."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation}
        ]
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True, 
            enable_thinking=self.enable_thinking
        )
        payload = {
            "model": model,
            "prompt": token_ids,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": 1.0,
            "n": 1,
            "stream": False,
        }
        
        r = self._post_json("/completions", payload)
        
        if r.status_code != 200:
            error_msg = f"vLLM server error (status={r.status_code}): {r.text[:300]}"
            return None, error_msg
        
        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            return None, "vLLM server returned no choices"
        
        action = choices[0].get("text", "").strip()
        
        if not action:
            return "", None
        
        return action, None
