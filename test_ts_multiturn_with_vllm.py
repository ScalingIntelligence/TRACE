#!/usr/bin/env python3
"""Test ToolSandbox Multi-Turn game with vLLM — verify reward distribution and GRPO viability."""
import json, os, sys
from openai import OpenAI
from toolsandbox_multiturn_game import ToolSandboxMultiTurnGame

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:2020/v1")
MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507")
client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")


def run_episode(seed, temp=1.0, verbose=False, use_hint=None):
    game = ToolSandboxMultiTurnGame()
    game.reset(seed, use_hint=use_hint)
    scenario = game._scenario

    messages = [{"role": "system", "content": game.get_system_prompt()}]
    messages.extend(game.get_messages())  # Initial user message
    tools = game.get_tool_schemas()

    all_actions = []
    for step in range(game.max_steps):
        if game.done:
            break
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools,
                temperature=temp, max_tokens=512,
            )
        except Exception as e:
            if verbose:
                print(f"  API error: {e}")
            break

        msg = r.choices[0].message

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            action_json = json.dumps({"name": tc.function.name, "arguments": args})
            game.step(action_json)
            all_actions.append(f"TOOL:{tc.function.name}({json.dumps(args)[:40]})")

            # Build messages for next turn
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tc.id, "type": "function",
                               "function": {"name": tc.function.name,
                                           "arguments": tc.function.arguments}}],
            })
            # Get tool result from conversation
            if len(game._conversation) >= 2 and game._conversation[-1]["role"] == "tool":
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": game._conversation[-1]["content"],
                })
        else:
            text = msg.content or ""
            # Strip thinking tags if present
            import re
            text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            if not text_clean:
                text_clean = text.strip()

            game.step(text_clean)
            all_actions.append(f"TEXT:{text_clean[:60]}")
            messages.append({"role": "assistant", "content": text_clean})

            # If game added a user response (multi-turn), add it to messages
            if not game.done and game._conversation and game._conversation[-1]["role"] == "user":
                messages.append({"role": "user", "content": game._conversation[-1]["content"]})

    if not game.done:
        game.done = True
        game._compute_timeout_reward()

    return game.rewards.get(0, 0.0), all_actions, scenario, game._use_hint


def main():
    n_seeds = 30
    n_trials = 3
    verbose = "--verbose" in sys.argv

    all_results = {}
    variance_count = 0
    from collections import defaultdict
    skill_rewards = defaultdict(list)
    hint_rewards = {"hint": [], "no_hint": []}

    for seed in range(n_seeds):
        results = []
        # Run 2 trials without hint, 2 with hint (simulates GRPO group with hint_ratio=0.5)
        for trial in range(n_trials):
            use_hint = trial >= (n_trials // 2)  # first half no hint, second half hint
            reward, actions, scenario, had_hint = run_episode(seed, temp=1.0, verbose=verbose, use_hint=use_hint)
            results.append((reward, actions, scenario, had_hint))
            skill_rewards[scenario.skill].append(reward)
            hint_rewards["hint" if had_hint else "no_hint"].append(reward)

        rewards = [r[0] for r in results]
        has_var = max(rewards) != min(rewards)
        if has_var:
            variance_count += 1

        r0, a0, s0, h0 = results[0]
        action_str = " -> ".join(a0[:8])
        if len(a0) > 8:
            action_str += f" ... ({len(a0)} total)"

        r_no = [round(r[0],2) for r in results if not r[3]]
        r_yes = [round(r[0],2) for r in results if r[3]]
        status = "VAR" if has_var else "   "
        print(f"Seed {seed:>2d} [{s0.skill:>12s}] no_hint={r_no} hint={r_yes} {status} | {s0.description}")
        print(f"         Init: \"{s0.initial_message[:60]}\"")
        print(f"         Actions: {action_str}")
        print()

    # Summary
    print("=" * 70)
    print(f"  SUMMARY ({n_seeds} seeds x {n_trials} trials = {n_seeds * n_trials} episodes)")
    print("=" * 70)

    for skill, rewards in skill_rewards.items():
        if not rewards:
            continue
        mean_r = sum(rewards) / len(rewards)
        perfect = sum(1 for r in rewards if r >= 0.99)
        partial = sum(1 for r in rewards if 0.01 < r < 0.99)
        failed = sum(1 for r in rewards if r <= 0.01)
        print(f"\n  {skill.upper()} ({len(rewards)} episodes):")
        print(f"    Mean reward:  {mean_r:.3f}")
        print(f"    Perfect:      {perfect}/{len(rewards)} ({perfect/len(rewards)*100:.1f}%)")
        print(f"    Partial:      {partial}/{len(rewards)} ({partial/len(rewards)*100:.1f}%)")
        print(f"    Failed:       {failed}/{len(rewards)} ({failed/len(rewards)*100:.1f}%)")

    total = sum(len(v) for v in skill_rewards.values())
    all_rewards = [r for v in skill_rewards.values() for r in v]
    print(f"\n  OVERALL ({total} episodes):")
    print(f"    Mean reward:  {sum(all_rewards)/len(all_rewards):.3f}")
    print(f"    Seeds with reward variance: {variance_count}/{n_seeds}")

    # Hint impact analysis
    print(f"\n  HINT IMPACT:")
    for label in ["no_hint", "hint"]:
        rr = hint_rewards[label]
        if rr:
            print(f"    {label:>7s}: mean={sum(rr)/len(rr):.3f} (n={len(rr)})")
    if hint_rewards["hint"] and hint_rewards["no_hint"]:
        diff = sum(hint_rewards["hint"])/len(hint_rewards["hint"]) - sum(hint_rewards["no_hint"])/len(hint_rewards["no_hint"])
        print(f"    Hint lift: {diff:+.3f}")
        print(f"    -> GRPO variance from hints: {'GOOD' if abs(diff) > 0.05 else 'LOW - hints may need strengthening'}")


if __name__ == "__main__":
    main()
