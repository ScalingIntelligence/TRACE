"""
Core game logic for running model-vs-model evaluations.
"""

import sys
from pathlib import Path

# Add root games folder to path FIRST so kuhn_poker.py finds the right config
GAMES_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GAMES_PATH))

import os
import random
import torch
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

# Import from local eval_config (experiments/model_vs_model/eval_config.py)
from eval_config import AVAILABLE_MODELS, SYSTEM_PROMPT, DEFAULT_MAX_SEQ_LENGTH

# This will now correctly import from games/kuhn_poker.py which uses games/config.py
from kuhn_poker import KuhnPoker
import re

# More flexible action extraction that handles:
# - [check], [bet], etc. (with brackets)
# - check, bet, etc. (without brackets)
# - <think>...</think> check (after thinking tags)
_FLEXIBLE_ACTION_RE = re.compile(r"\[?(check|bet|call|fold)\]?", re.IGNORECASE)

def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """
    Extract action from model output. More flexible than the training version.
    Handles actions with or without brackets.
    """
    if not text:
        return None
    
    # Remove thinking tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    matches = _FLEXIBLE_ACTION_RE.findall(text)
    for m in matches:
        a = f"[{m.lower()}]"
        if a in legal_actions:
            return a
    return None

from unsloth import FastLanguageModel


def load_model(
    model_name: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> Tuple:
    """Load model, finding its path if it's in available models, 
    otherwise assuming name is full path."""

    if model_name in AVAILABLE_MODELS:
        model_path = AVAILABLE_MODELS[model_name]
    else:
        model_path = model_name
    
    print(f"[load_model] Loading {model_path}...")

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        # Don't offload embeddings - causes device mismatch during generation
        # offload_embedding=True,
        cache_dir=cache_dir,
    )

    if adapter_path:
        model.load_adapter(adapter_path, adapter_name="trained")
        model.set_adapter("trained")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)
    print(f"[load_model] {model_name} loaded successfully")
    return model, tokenizer


def _build_messages(player_id: int, observation: str) -> list:
    """Build chat messages for the model."""
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
    legal_actions: List[str],
    temperature: float,
    max_new_tokens: int,
) -> Tuple[Optional[str], str]:
    """
    Generate an action from the model given the game observation.
    
    Returns:
        Tuple of (action, raw_output) where action is None if invalid.
    """
    messages = _build_messages(player_id, observation)
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Don't force device - let model handle it (needed for offloaded embeddings)
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Move to the device where the model's non-offloaded layers are
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode only the new tokens
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Extract action from output
    action = extract_action(raw_output, legal_actions)
    
    return action, raw_output


def play_game(
    model_p0,
    tokenizer_p0,
    model_p1,
    tokenizer_p1,
    num_rounds: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    verbose: bool = False,
    model_p0_name: str = "P0",
    model_p1_name: str = "P1",
) -> Dict:
    """
    Play a single game of Kuhn Poker between two models.
    
    Returns:
        Dict with game results including winner, rewards, and game log.
    """
    game = KuhnPoker(num_rounds=num_rounds)
    game.reset(seed)
    
    models = {0: model_p0, 1: model_p1}
    tokenizers = {0: tokenizer_p0, 1: tokenizer_p1}
    model_names = {0: model_p0_name, 1: model_p1_name}
    
    game_log = []
    
    if verbose:
        print(f"\n    --- Game (seed={seed}) ---")
        print(f"    {model_p0_name} (P0) vs {model_p1_name} (P1)")
    
    while not game.done:
        player = game.current_player
        observation = game.observe(player)
        legal_actions = game.legal_actions()
        
        action, raw_output = generate_action(
            model=models[player],
            tokenizer=tokenizers[player],
            player_id=player,
            observation=observation,
            legal_actions=legal_actions,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        
        if verbose:
            # Truncate long outputs for display
            display_output = raw_output[:80] + "..." if len(raw_output) > 80 else raw_output
            display_output = display_output.replace('\n', ' ')
            action_str = action if action else "INVALID"
            print(f"    R{game.round_idx} {model_names[player]:20} | Legal: {legal_actions} | Action: {action_str:8} | Output: '{display_output}'")
        
        game_log.append({
            "round": game.round_idx,
            "player": player,
            "model": model_names[player],
            "legal_actions": legal_actions,
            "raw_output": raw_output,
            "action": action,
        })
        
        game.step(action)
    
    # Determine winner: reward > 0 means that player won
    winner = None
    if game.rewards[0] > 0:
        winner = 0
    elif game.rewards[1] > 0:
        winner = 1
    # else it's a draw (winner=None)
    
    if verbose:
        if game.invalid_player is not None:
            print(f"    Result: {model_names[game.invalid_player]} made INVALID move -> {model_names[1 - game.invalid_player]} WINS")
        elif winner is not None:
            print(f"    Result: {model_names[winner]} WINS (chips: P0={game.chips[0]}, P1={game.chips[1]})")
        else:
            print(f"    Result: DRAW (chips: P0={game.chips[0]}, P1={game.chips[1]})")
    
    return {
        "winner": winner,
        "rewards": game.rewards,
        "chips": game.chips,
        "invalid_player": game.invalid_player,
        "game_log": game_log,
    }


def run_matchup(
    model_p0,
    tokenizer_p0,
    model_p1,
    tokenizer_p1,
    num_games: int,
    num_rounds: int,
    temperature: float,
    max_new_tokens: int,
    start_seed: int = 0,
    verbose: bool = False,
    model_p0_name: str = "P0",
    model_p1_name: str = "P1",
) -> Dict:
    """
    Run multiple games between two models.
    
    Returns:
        Dict with aggregated statistics.
    """
    wins = {0: 0, 1: 0}
    draws = 0
    invalid_counts = {0: 0, 1: 0}
    all_games = []
    
    for i in tqdm(range(num_games), desc="Playing games"):
        seed = start_seed + i
        result = play_game(
            model_p0=model_p0,
            tokenizer_p0=tokenizer_p0,
            model_p1=model_p1,
            tokenizer_p1=tokenizer_p1,
            num_rounds=num_rounds,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
            verbose=verbose,
            model_p0_name=model_p0_name,
            model_p1_name=model_p1_name,
        )
        
        if result["winner"] is not None:
            wins[result["winner"]] += 1
        else:
            draws += 1
        
        if result["invalid_player"] is not None:
            invalid_counts[result["invalid_player"]] += 1
        
        all_games.append(result)
    
    return {
        "wins": wins,
        "draws": draws,
        "invalid_counts": invalid_counts,
        "p0_win_rate": wins[0] / num_games,
        "p1_win_rate": wins[1] / num_games,
        "draw_rate": draws / num_games,
        "all_games": all_games,
    }
