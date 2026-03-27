#!/usr/bin/env python3
"""Test TEC v2 game with vLLM model — verify reward distribution and failure reasons."""
import json, os
from openai import OpenAI
from tec_game_v2 import TECGameV2

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:9001/v1")
MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507")
client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")


def run_episode(seed, temp=1.0):
    game = TECGameV2()
    game.reset(seed)
    messages = [{"role": "system", "content": game.get_system_prompt()},
                {"role": "user", "content": game._scenario.user_message}]
    tools = game.get_tool_schemas()

    all_actions = []
    for step in range(10):
        if game.done: break
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages,
                                                tools=tools, temperature=temp, max_tokens=512)
        except Exception as e:
            print(f"  API error: {e}")
            break
        msg = r.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            action_json = json.dumps({"name": tc.function.name,
                                       "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}})
            game.step(action_json)
            all_actions.append(f"TOOL:{tc.function.name}")
            messages.append({"role": "assistant", "content": None,
                            "tool_calls": [{"id": tc.id, "type": "function",
                                           "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]})
            if game._tool_result:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": game._tool_result})
        else:
            text = msg.content or ""
            game.step(text)
            all_actions.append(f"TEXT:{text[:60]}")
            messages.append({"role": "assistant", "content": text})

    if not game.done:
        game.done = True
        game.rewards = {0: 0.0}

    return game.rewards.get(0, 0.0), all_actions, game._scenario


# Run 30 seeds, 3 trials each
n_seeds = 30
n_trials = 3
variance_count = 0

for seed in range(n_seeds):
    results = []
    for trial in range(n_trials):
        reward, actions, scenario = run_episode(seed, temp=1.0)
        results.append((reward, actions))

    rewards = [r[0] for r in results]
    has_var = max(rewards) != min(rewards)
    if has_var: variance_count += 1

    # Show details for first trial
    r0, a0 = results[0]
    action_str = " → ".join(a0[:6])
    if len(a0) > 6: action_str += f" ... ({len(a0)} total)"

    status = "VAR" if has_var else "   "
    print(f"Seed {seed:>2d} [{scenario.skill:>11s}] r={[round(r,2) for r in rewards]} {status} | {scenario.description}")
    print(f"         Actions: {action_str}")

    # Show WHY it failed/partially succeeded
    if r0 < 0.99:
        if r0 == 0.5 and scenario.skill == "communicate":
            print(f"         → Tool called but no communication to user")
        elif r0 == 0.0 and scenario.skill == "recovery":
            print(f"         → Didn't complete error recovery chain")
        elif r0 == 0.25:
            print(f"         → Partial: tried something but incomplete")
        elif 0 < r0 < 0.5:
            print(f"         → Partial: some steps done, keywords missing")
    print()

# Summary
all_rewards = []
for seed in range(n_seeds):
    for trial in range(n_trials):
        r, _, _ = run_episode(seed, temp=1.0)
        all_rewards.append(r)

perfect = sum(1 for r in all_rewards if r >= 0.99)
partial = sum(1 for r in all_rewards if 0.01 < r < 0.99)
failed = sum(1 for r in all_rewards if r <= 0.01)
print(f"{'='*60}")
print(f"  SUMMARY ({len(all_rewards)} episodes)")
print(f"  Mean: {sum(all_rewards)/len(all_rewards):.3f}")
print(f"  Perfect: {perfect}/{len(all_rewards)} ({perfect/len(all_rewards)*100:.1f}%)")
print(f"  Partial: {partial}/{len(all_rewards)} ({partial/len(all_rewards)*100:.1f}%)")
print(f"  Failed: {failed}/{len(all_rewards)} ({failed/len(all_rewards)*100:.1f}%)")
print(f"  Seeds with variance: {variance_count}/{n_seeds}")
