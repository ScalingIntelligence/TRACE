"""
Core game logic for running model-vs-model evaluations.
"""

import sys
import random
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from config import GAMES_PATH, AVAILABLE_MODELS, SYSTEM_PROMPT, DEFAULT_MAX_SEQ_LENGTH
sys.path.insert(0, str(GAMES_PATH))

from kuhn_poker import KuhnPoker, extract_action

from unsloth import FastLanguageModel

def load_model(
    model_name: str,
    adapter_path: Optional[str] = None,
    device: str = "cude",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ) -> Tuple:
    """ Load model, finding its path if it's in available models, 
    otherwise assuming name is full path """

    if model_name in AVAILABLE_MODELS::
        model_path = AVAILABLE_MODELS[model_name]
    
    else:
        model_path = model_name
    
    print(f"[load_model] Loading {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    if adapter_path:
            model = FastLanguageModel.get_peft_model(
        model, r=4, target_modules=[...], lora_alpha=8, ...
        )
        # Then load the saved weights
        model.load_adapter(adapter_path, adapter_name="trained")
        model.set_adapter("trained")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
    return model, tokenizer

def _build_messages(player_id: int, observation: str) -> list:
    """ Build chat messages for the model. """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation},
    ]

@torch.no_grad()
def generate_action(
    model,
    tokenizer,
    player_id: int,
    observation: str,
    legal_actions: list,
    temperature: float,
    max_new_tokens: int,
    device: str,
) -> Tuple[str, str]:
    
