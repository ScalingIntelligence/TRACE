#!/usr/bin/env python3
"""Test TEC game environment with actual vLLM model to check reward variance."""
import json
import os
import sys
from openai import OpenAI

from tec_game import TECGame, TOOL_SCHEMAS

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:5051/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507")

client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")


def run_episode(seed: int, temperature: float = 1.0) -> dict:
    """Run one episode and return results."""
    game = TECGame()
    game.reset(seed)

    system_prompt = game.get_system_prompt()
    tools = game.get_tool_schemas()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": game._scenario.user_message},
    ]

    max_steps = 5
    for step in range(max_steps):
        if game.done:
            break

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=512,
            )
        except Exception as e:
            print(f"  API error: {e}")
            break

        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            # Agent made a tool call
            tc = msg.tool_calls[0]
            tool_call_json = json.dumps({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
            })
            game.step(tool_call_json)

            # Add to messages for next turn
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }],
            })
            # Add tool result
            if game._tool_result:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": game._tool_result,
                })
        else:
            # Agent sent a text response
            text = msg.content or ""
            game.step(text)
            messages.append({"role": "assistant", "content": text})

    if not game.done:
        game.done = True
        game.rewards = {0: 0.5 if game._correct_tool else 0.0}

    return {
        "seed": seed,
        "reward": game.rewards.get(0, 0.0),
        "scenario": game._scenario.description,
        "user_msg": game._scenario.user_message,
        "difficulty": game._scenario.difficulty,
        "tool_called": game._tool_called,
        "correct_tool": game._correct_tool,
        "steps": game._step_count,
    }


def main():
    n_seeds = 20
    n_trials = 3  # run each seed multiple times to check variance
    temperature = 1.0

    print(f"Testing TEC environment with vLLM at {VLLM_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seeds: {n_seeds}, Trials per seed: {n_trials}, Temperature: {temperature}")
    print()

    all_results = []
    for seed in range(n_seeds):
        seed_rewards = []
        for trial in range(n_trials):
            result = run_episode(seed, temperature=temperature)
            seed_rewards.append(result["reward"])
            all_results.append(result)

        # Show per-seed summary
        avg = sum(seed_rewards) / len(seed_rewards)
        variance = max(seed_rewards) - min(seed_rewards)
        scenario = all_results[-1]["scenario"]
        user_msg = all_results[-1]["user_msg"][:60]
        print(f"Seed {seed:>2d} [{all_results[-1]['difficulty']:>6s}] "
              f"rewards={seed_rewards} avg={avg:.2f} var={variance:.2f} "
              f"| {scenario}")

    # Summary
    rewards = [r["reward"] for r in all_results]
    perfect = sum(1 for r in rewards if r >= 0.99)
    partial = sum(1 for r in rewards if 0.01 < r < 0.99)
    failed = sum(1 for r in rewards if r <= 0.01)

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({len(rewards)} total episodes)")
    print(f"{'='*60}")
    print(f"  Mean reward: {sum(rewards)/len(rewards):.3f}")
    print(f"  Perfect: {perfect}/{len(rewards)} ({perfect/len(rewards)*100:.1f}%)")
    print(f"  Partial (0.5): {partial}/{len(rewards)} ({partial/len(rewards)*100:.1f}%)")
    print(f"  Failed: {failed}/{len(rewards)} ({failed/len(rewards)*100:.1f}%)")

    # Check GRPO signal
    seeds_with_variance = 0
    for seed in range(n_seeds):
        seed_rewards = [r["reward"] for r in all_results if r["seed"] == seed]
        if max(seed_rewards) != min(seed_rewards):
            seeds_with_variance += 1

    print(f"\n  Seeds with reward variance: {seeds_with_variance}/{n_seeds}")
    print(f"  (Need variance for GRPO advantage signal)")


if __name__ == "__main__":
    main()
