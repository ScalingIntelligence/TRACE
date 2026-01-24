#!/usr/bin/env python3
"""
Run model-vs-model evaluation for ANY registered game.

Usage:
    python run_any_game.py --game liars_dice_memory --num_games 5 --verbose
    python run_any_game.py --game liars_dice --vllm_url http://localhost:8076 --num_games 10
"""
import sys
from pathlib import Path

# Add games directory to path
GAMES_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GAMES_PATH))

import argparse
import random
import requests
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer

from config import Config
from game_registry import get_game_spec, list_game_names


class VLLMBackend:
    """Backend for calling vLLM server using /completions endpoint (like training code)."""
    
    def __init__(self, base_url: str, model_name: str = "default", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()
        
        # Auto-detect model name from server
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=10)
            if r.status_code == 200:
                models = r.json().get("data", [])
                if models:
                    self.model_name = models[0].get("id", model_name)
        except Exception:
            pass
        
        # Load tokenizer for prompt formatting (same as training code)
        print(f"Loading tokenizer from {Config.MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    def _format_prompt(self, messages: List[Dict]) -> str:
        """Format messages using tokenizer's chat template with thinking enabled."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=Config.ENABLE_THINKING,
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
            return self.tokenizer.decode(ids, skip_special_tokens=False)
    
    def generate(self, messages: List[Dict], temperature: float, max_new_tokens: int) -> str:
        # Format prompt locally (with thinking enabled), then use /completions
        prompt = self._format_prompt(messages)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        }
        
        r = self.session.post(
            f"{self.base_url}/completions",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["text"]


def play_game(
    game_name: str,
    backend: VLLMBackend,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    verbose: bool = False,
) -> Dict:
    """Play a single game with the model against itself."""
    
    game_spec = get_game_spec(game_name)
    env = game_spec.make_env()
    env.reset(seed)
    
    game_log = []
    
    if verbose:
        print(f"\n    --- Game (seed={seed}) ---")
    
    while not env.done:
        player = env.current_player
        observation = env.observe(player)
        legal_actions = env.legal_actions()
        
        messages = [
            {"role": "system", "content": game_spec.system_prompt},
            {"role": "user", "content": observation},
        ]
        
        raw_output = backend.generate(
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        
        action = game_spec.extract_action(raw_output, legal_actions)
        
        if action is None:
            action_status = "INVALID"
        else:
            action_status = "OK"
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Player {player}'s turn")
            print(f"{'='*70}")
            print(f"\n--- OBSERVATION (what player sees) ---")
            print(observation)
            print(f"\n--- LEGAL ACTIONS ({len(legal_actions)} total) ---")
            print(legal_actions[:8])
            if len(legal_actions) > 8:
                print(f"... and {len(legal_actions) - 8} more")
            print(f"\n--- MODEL OUTPUT ---")
            print(raw_output)
            print(f"\n--- EXTRACTED ACTION: {action} ({action_status}) ---")
        
        game_log.append({
            "player": player,
            "legal_actions": legal_actions,
            "raw_output": raw_output,
            "action": action,
            "action_status": action_status,
        })
        
        env.step(action)
    
    # Determine winner
    winner = None
    if env.rewards.get(0, 0) > env.rewards.get(1, 0):
        winner = 0
    elif env.rewards.get(1, 0) > env.rewards.get(0, 0):
        winner = 1
    
    if verbose:
        if env.invalid_player is not None:
            print(f"    Result: Player {env.invalid_player} made INVALID move -> Player {1 - env.invalid_player} WINS")
        elif winner is not None:
            print(f"    Result: Player {winner} WINS")
        else:
            print(f"    Result: DRAW")
    
    return {
        "winner": winner,
        "rewards": env.rewards,
        "invalid_player": env.invalid_player,
        "game_log": game_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Run model-vs-model for any game")
    parser.add_argument("--game", type=str, required=True,
                        help=f"Game to play. Available: {list_game_names()}")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8076",
                        help="vLLM server URL")
    parser.add_argument("--num_games", type=int, default=10,
                        help="Number of games to play")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max tokens for generation")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed game logs")
    
    args = parser.parse_args()
    
    # Validate game
    try:
        game_spec = get_game_spec(args.game)
    except KeyError as e:
        print(f"Error: {e}")
        print(f"Available games: {list_game_names()}")
        return
    
    print("=" * 60)
    print(f"Game: {args.game}")
    print(f"vLLM URL: {args.vllm_url}")
    print(f"Num games: {args.num_games}")
    print(f"Temperature: {args.temperature}")
    print("=" * 60)
    
    # Connect to vLLM
    backend = VLLMBackend(args.vllm_url)
    print(f"Connected to vLLM, model: {backend.model_name}")
    
    # Run games
    wins = {0: 0, 1: 0}
    draws = 0
    invalid_count = 0
    
    for i in range(args.num_games):
        seed = args.seed + i
        result = play_game(
            game_name=args.game,
            backend=backend,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            seed=seed,
            verbose=args.verbose,
        )
        
        if result["winner"] is not None:
            wins[result["winner"]] += 1
        else:
            draws += 1
        
        if result["invalid_player"] is not None:
            invalid_count += 1
        
        # Progress
        if (i + 1) % 5 == 0 or i == args.num_games - 1:
            print(f"Game {i+1}/{args.num_games}: P0={wins[0]} ({wins[0]/(i+1):.1%}), P1={wins[1]} ({wins[1]/(i+1):.1%}), Draws={draws}, Invalid={invalid_count}")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Player 0 wins: {wins[0]} ({wins[0]/args.num_games:.1%})")
    print(f"Player 1 wins: {wins[1]} ({wins[1]/args.num_games:.1%})")
    print(f"Draws: {draws} ({draws/args.num_games:.1%})")
    print(f"Invalid moves: {invalid_count}")


if __name__ == "__main__":
    main()
