#!/usr/bin/env python3
"""
Interactive game viewer - see exactly what the LLM sees and does.

Usage:
    python view_any_game.py --game liars_dice_memory --seed 42
    python view_any_game.py --game liars_dice --seed 42 --play  # Play yourself
"""
import sys
from pathlib import Path

# Add the games directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import random
import torch
from unsloth import FastLanguageModel

from config import Config
from game_registry import get_game_spec, list_game_names


def parse_args():
    parser = argparse.ArgumentParser(description="View game step by step")
    parser.add_argument("--game", type=str, default="liars_dice_memory",
                        help=f"Game to play. Available: {list_game_names()}")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--play", action="store_true", 
                        help="Play as Player 0 yourself (instead of watching LLM)")
    parser.add_argument("--no-model", action="store_true",
                        help="Just show observations without running model")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint to load")
    return parser.parse_args()


def load_model(checkpoint_path=None):
    """Load the model."""
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.MODEL_NAME,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    if checkpoint_path:
        print(f"Loading adapter from {checkpoint_path}...")
        model.load_adapter(checkpoint_path, adapter_name="trained")
        model.set_adapter("trained")
    
    model.eval()
    FastLanguageModel.for_inference(model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded.\n")
    return model, tokenizer


def generate_response(model, tokenizer, messages, temperature, max_tokens):
    """Generate model response."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=Config.ENABLE_THINKING,
    ).to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response


def get_human_action(legal_actions):
    """Get action from human player."""
    print("\n" + "-" * 40)
    print("YOUR TURN - Legal actions:")
    for i, action in enumerate(legal_actions):
        print(f"  [{i}] {action}")
    
    while True:
        try:
            choice = input("\nEnter action number or type action directly: ").strip()
            
            # Try as index
            if choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(legal_actions):
                    return legal_actions[idx]
            
            # Try as direct action
            for legal in legal_actions:
                if choice.lower() in legal.lower() or legal.lower() in choice.lower():
                    return legal
            
            print(f"Invalid. Choose from: {legal_actions}")
        except KeyboardInterrupt:
            print("\nGame aborted.")
            sys.exit(0)


def main():
    args = parse_args()
    
    # Load game spec
    try:
        game_spec = get_game_spec(args.game)
    except KeyError as e:
        print(f"Error: {e}")
        print(f"Available games: {list_game_names()}")
        sys.exit(1)
    
    print("=" * 70)
    print(f"GAME: {game_spec.name}")
    print("=" * 70)
    print(f"\nSYSTEM PROMPT:\n{game_spec.system_prompt}")
    print("=" * 70)
    
    # Load model if needed
    model, tokenizer = None, None
    if not args.no_model and not args.play:
        model, tokenizer = load_model(args.checkpoint)
    
    # Create environment
    env = game_spec.make_env()
    env.reset(args.seed)
    
    turn = 0
    while not env.done:
        turn += 1
        player = env.current_player
        obs = env.observe(player)
        legal = env.legal_actions()
        
        print(f"\n{'=' * 70}")
        print(f"TURN {turn} | Player {player}")
        print(f"{'=' * 70}")
        
        # Show full observation
        print(f"\n{'─' * 40}")
        print("OBSERVATION (what the LLM sees):")
        print(f"{'─' * 40}")
        print(obs)
        
        print(f"\n{'─' * 40}")
        print(f"LEGAL ACTIONS ({len(legal)} options):")
        print(f"{'─' * 40}")
        # Show first 10 and last few if there are many
        if len(legal) <= 15:
            for a in legal:
                print(f"  {a}")
        else:
            for a in legal[:8]:
                print(f"  {a}")
            print(f"  ... ({len(legal) - 12} more) ...")
            for a in legal[-4:]:
                print(f"  {a}")
        
        # Get action
        if args.no_model:
            # Just show observation, pick random action
            action = random.choice(legal)
            print(f"\n[Random action selected: {action}]")
            input("\nPress Enter to continue...")
        
        elif args.play and player == 0:
            # Human plays as P0
            action = get_human_action(legal)
            print(f"\nYou played: {action}")
        
        else:
            # Model plays
            messages = [
                {"role": "system", "content": game_spec.system_prompt},
                {"role": "user", "content": obs},
            ]
            
            print(f"\n{'─' * 40}")
            print("MODEL OUTPUT:")
            print(f"{'─' * 40}")
            
            response = generate_response(
                model, tokenizer, messages, 
                args.temperature, args.max_tokens
            )
            print(response)
            
            # Extract action
            action = game_spec.extract_action(response, legal)
            
            print(f"\n{'─' * 40}")
            if action is None:
                action = random.choice(legal)
                print(f"EXTRACTED ACTION: None (fallback random: {action})")
            else:
                print(f"EXTRACTED ACTION: {action}")
            print(f"{'─' * 40}")
            
            if not args.play:
                input("\nPress Enter for next turn...")
        
        env.step(action)
    
    # Game over
    print(f"\n{'=' * 70}")
    print("GAME OVER")
    print(f"{'=' * 70}")
    print(f"Rewards: {env.rewards}")
    if env.invalid_player is not None:
        print(f"Invalid player: {env.invalid_player}")
    
    winner = 0 if env.rewards.get(0, 0) > env.rewards.get(1, 0) else 1
    if env.rewards.get(0, 0) == env.rewards.get(1, 0):
        print("Result: DRAW")
    else:
        print(f"Result: Player {winner} WINS")


if __name__ == "__main__":
    main()
