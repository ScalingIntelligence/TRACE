#!/usr/bin/env python3
"""Analyze specific failed retail tasks from tau-bench evaluation."""

import json
import sys
import textwrap

FILE = "/home/hangook/games/evals/benchmarks/tau2_bench_eval/data/simulations/qwen-3-30b-retail-qwen3-30b.json"

with open(FILE) as f:
    data = json.load(f)

tasks = data["tasks"]
sims = data["simulations"]

TARGET_TASKS = [0, 10, 14, 27, 28, 29, 30, 31, 33, 34, 36, 37, 41, 42, 43, 45, 57, 59, 62, 67, 76]

def extract_tool_calls(messages):
    """Extract all tool calls made by the assistant."""
    calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "unknown")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = fn.get("arguments", {})
                calls.append({"name": name, "args": args})
    return calls

def extract_conversation_text(messages):
    """Extract readable conversation flow."""
    lines = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "assistant":
            if content:
                lines.append(f"ASSISTANT: {content[:300]}")
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except:
                        args = fn.get("arguments", {})
                    lines.append(f"  TOOL_CALL: {name}({json.dumps(args, separators=(',',':'))[:200]})")
        elif role == "user":
            if content:
                lines.append(f"USER: {content[:300]}")
        elif role == "tool":
            if content:
                lines.append(f"  TOOL_RESULT: {str(content)[:200]}")
    return "\n".join(lines)

def analyze_task(idx):
    """Analyze a single task and its simulation."""
    task = tasks[idx]
    sim = sims[idx]
    ri = sim["reward_info"]

    print(f"\n{'='*80}")
    print(f"TASK {idx}")
    print(f"{'='*80}")

    # Task description
    scenario = task.get("user_scenario", {})
    instructions = scenario.get("instructions", {}) if isinstance(scenario, dict) else {}
    if isinstance(instructions, dict):
        print(f"Reason for call: {instructions.get('reason_for_call', 'N/A')}")
        print(f"Known info: {instructions.get('known_info', 'N/A')}")

    # Termination
    print(f"Termination: {sim.get('termination_reason', 'N/A')}")

    # Reward breakdown
    print(f"Reward: {ri.get('reward', 'N/A')}")
    rb = ri.get("reward_breakdown", {})
    print(f"Reward breakdown: {json.dumps(rb)}")

    # DB check
    db = ri.get("db_check", {})
    print(f"DB check: {json.dumps(db)}")

    # Action checks
    ac = ri.get("action_checks", [])
    if ac:
        print(f"\nAction checks ({len(ac)} actions):")
        for a in ac:
            action = a.get("action", {})
            name = action.get("name", "?")
            match = a.get("action_match", "?")
            reward = a.get("action_reward", "?")
            args_summary = json.dumps(action.get("arguments", {}), separators=(',',':'))[:150]
            status = "OK" if match else "FAIL"
            print(f"  [{status}] {name} -> {args_summary}")

    # Expected actions from eval criteria
    ec = task.get("evaluation_criteria", {})
    expected_actions = ec.get("actions", [])
    print(f"\nExpected actions ({len(expected_actions)}):")
    for ea in expected_actions:
        name = ea.get("name", "?")
        args_summary = json.dumps(ea.get("arguments", {}), separators=(',',':'))[:150]
        print(f"  EXPECTED: {name} -> {args_summary}")

    # Communicate checks
    cc = ri.get("communicate_checks", None)
    if cc:
        print(f"\nCommunicate checks ({len(cc)}):")
        for c in cc:
            print(f"  [{c.get('communicate_match','?')}] {str(c.get('communicate_info',''))[:200]}")

    # NL assertions
    nl = ri.get("nl_assertions", None)
    if nl:
        print(f"\nNL assertions: {json.dumps(nl, indent=2)[:500]}")

    # Communicate info from eval criteria
    comm_info = ec.get("communicate_info", [])
    if comm_info:
        print(f"\nExpected communicate_info ({len(comm_info)}):")
        for ci in comm_info:
            print(f"  {json.dumps(ci, separators=(',',':'))[:200]}")

    # Full conversation
    print(f"\n--- Conversation ({len(sim['messages'])} messages) ---")
    conv = extract_conversation_text(sim["messages"])
    print(conv)

    # Tool calls summary
    tool_calls = extract_tool_calls(sim["messages"])
    print(f"\n--- Tool calls summary ({len(tool_calls)} calls) ---")
    for tc in tool_calls:
        print(f"  {tc['name']}({json.dumps(tc['args'], separators=(',',':'))[:200]})")


for idx in TARGET_TASKS:
    analyze_task(idx)
