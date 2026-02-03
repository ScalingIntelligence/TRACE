"""
Test base model performance on various games.
This helps determine if a game is too easy or appropriately challenging.
"""

import json
import random
from vllm import LLM, SamplingParams

# Import all games
from conditional_action_game import ConditionalActionGame, SYSTEM_PROMPT_CONDITIONAL_ACTION, extract_action as extract_conditional
from tool_chain_game import ToolChainGame, SYSTEM_PROMPT_TOOL_CHAIN, extract_action as extract_tool_chain
from policy_action_game import PolicyActionGame, SYSTEM_PROMPT_POLICY_ACTION, extract_action as extract_policy_action


def test_game(game_class, system_prompt, extract_fn, game_name, num_games=30, model_path="Qwen/Qwen3-4B-Instruct-2507"):
    """Test base model performance on a game."""

    print(f"\n{'='*60}")
    print(f"Testing {game_name}")
    print(f"{'='*60}")

    # Initialize model
    llm = LLM(model=model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["}"],  # Stop at JSON object end
    )

    game = game_class() if game_class == PolicyActionGame else game_class(max_steps=10 if 'conditional' in game_name else 5)

    wins = 0
    losses = 0
    invalid = 0
    reasons = {}

    for seed in range(num_games):
        game.reset(seed)

        step_count = 0
        max_steps = 10 if 'conditional' in game_name else 5

        while not game.done and step_count < max_steps:
            obs = game.observe(0)

            # Format prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obs}
            ]

            # Apply chat template
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Generate
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()

            # Add closing brace if needed
            if response and not response.endswith("}"):
                response += "}"

            # Extract action
            action = extract_fn(response, game.legal_actions())

            # Take step
            game.step(action)
            step_count += 1

        # Record result
        reward = game.rewards.get(0, 0)
        if reward > 0:
            wins += 1
        else:
            losses += 1

        if game.invalid_player is not None:
            invalid += 1

        # Track reason
        reason = getattr(game._state, 'reason', 'unknown')
        reasons[reason] = reasons.get(reason, 0) + 1

    print(f"\nBASE MODEL RESULTS ON {game_name} ({num_games} games)")
    print(f"Win rate: {wins}/{num_games} = {100*wins/num_games:.1f}%")
    print(f"Invalid parses: {invalid}")
    print(f"Breakdown:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    return wins / num_games


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="conditional_action",
                        choices=["conditional_action", "tool_chain", "policy_action", "all"])
    parser.add_argument("--num_games", type=int, default=30)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    games_to_test = {
        "conditional_action": (ConditionalActionGame, SYSTEM_PROMPT_CONDITIONAL_ACTION, extract_conditional),
        "tool_chain": (ToolChainGame, SYSTEM_PROMPT_TOOL_CHAIN, extract_tool_chain),
        "policy_action": (PolicyActionGame, SYSTEM_PROMPT_POLICY_ACTION, extract_policy_action),
    }

    if args.game == "all":
        for name, (game_class, system_prompt, extract_fn) in games_to_test.items():
            test_game(game_class, system_prompt, extract_fn, name, args.num_games, args.model)
    else:
        game_class, system_prompt, extract_fn = games_to_test[args.game]
        test_game(game_class, system_prompt, extract_fn, args.game, args.num_games, args.model)


if __name__ == "__main__":
    main()
