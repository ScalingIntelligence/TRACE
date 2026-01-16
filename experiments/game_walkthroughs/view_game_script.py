#!/usr/bin/env python3
"""
Watch Qwen-4B play against itself, printing all prompts and responses.
"""
import sys
from pathlib import Path

# Add the games directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    max_seq_length=10000,
    dtype=None,
    load_in_4bit=True,
)
model.eval()

# Import your game
from openspiel_wrapper import OpenSpielEnv, OPENSPIEL_GAME_CONFIGS

# Pick a game
cfg = OPENSPIEL_GAME_CONFIGS["openspiel_breakthrough"]  # or "openspiel_dots_and_boxes"
env = OpenSpielEnv(cfg)
env.reset(seed=42)

print("="*60)
print(f"GAME: {cfg.name}")
print(f"SYSTEM PROMPT:\n{cfg.system_prompt}")
print("="*60)

turn = 0
while not env.done:
    turn += 1
    player = env.current_player
    obs = env.observe(player)
    legal = env.legal_actions()
    
    # Build messages
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": obs},
    ]
    
    # Tokenize
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=True,  # or False
    ).to("cuda")
    
    print(f"\n{'='*60}")
    print(f"TURN {turn} | Player {player}")
    print(f"{'='*60}")
    print(f"\n--- OBSERVATION ---\n{obs}")
    print(f"\n--- LEGAL ACTIONS ---\n{legal}")
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,  # increase if using thinking
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\n--- MODEL OUTPUT ---\n{response}")
    
    # Extract action
    from openspiel_wrapper import _extract_openspiel_action
    action = _extract_openspiel_action(response, legal)
    
    if action is None:
        import random
        action = random.choice(legal)
        print(f"\n--- ACTION (FALLBACK RANDOM) ---\n{action}")
    else:
        print(f"\n--- ACTION (EXTRACTED) ---\n{action}")
    
    env.step(action)

print(f"\n{'='*60}")
print("GAME OVER")
print(f"Rewards: {env.rewards}")
print(f"Invalid player: {env.invalid_player}")
print(f"{'='*60}")