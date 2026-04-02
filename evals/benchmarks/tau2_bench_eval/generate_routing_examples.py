#!/usr/bin/env python3
"""Generate example decision-point trajectories from each game for few-shot routing."""

import sys, json
sys.path.insert(0, "/home/ubuntu/hangook/games")

# TC game
from tau_tool_calling_env_revised.scenarios import generate_scenario as tc_gen

print("=== TOOL CALLING GAME EXAMPLES ===\n")
for seed in range(42, 80):
    sc = tc_gen(seed, 'airline')
    if sc.is_refusal or sc.scenario_type == 'policy_gated':
        continue
    print(f"Seed {seed} ({sc.scenario_type}): {sc.description}")
    print(f"  User: {sc.initial_message[:250]}")
    actions = []
    for a in sc.expected_actions[:2]:
        key_args = {k:v for k,v in a.arguments.items()
                   if k in ['reservation_id','cabin','flight_number','user_id','total_baggages']}
        actions.append(f"{a.name}({key_args})")
    print(f"  Expected: {'; '.join(actions)}")
    print(f"  Facts: {dict(list(sc.key_facts.items())[:3])}")
    print()
    if seed > 55:
        break

# PC game
from precondition_game_revised.scenarios import generate_scenario as pc_gen

print("\n=== PRECONDITION GAME EXAMPLES ===\n")
count = 0
for seed in range(42, 100):
    sc = pc_gen(seed, 'airline')
    print(f"Seed {seed} ({sc.scenario_type}, refusal={sc.is_refusal}): {sc.description}")
    print(f"  User: {sc.initial_message[:250]}")
    actions = []
    for a in sc.expected_actions[:2]:
        key_args = {k:v for k,v in a.arguments.items() if k in ['reservation_id','cabin']}
        actions.append(f"{a.name}({key_args})")
    if not actions:
        actions = ["respond_to_user (refuse)"]
    print(f"  Expected: {'; '.join(actions)}")
    print(f"  Facts: {dict(list(sc.key_facts.items())[:3])}")
    print()
    count += 1
    if count >= 6:
        break

# STD game
from structured_data_game_revised import generate_scenario as std_gen

print("\n=== STRUCTURED DATA REASONING GAME EXAMPLES ===\n")
count = 0
for seed in range(42, 100):
    sc = std_gen(seed, 'airline')
    print(f"Seed {seed} ({sc.scenario_type}): {sc.description}")
    print(f"  User: {sc.initial_message[:250]}")
    print(f"  Communicate: {sc.communicate_info}")
    actions = []
    for a in sc.expected_actions[:2]:
        key_args = {k:v for k,v in a.arguments.items() if k in ['reservation_id','cabin','flight_number']}
        actions.append(f"{a.name}({key_args})")
    print(f"  Expected: {'; '.join(actions)}")
    print()
    count += 1
    if count >= 6:
        break

# MT game
from multistep_task_game_revised import generate_scenario as mt_gen

print("\n=== MULTISTEP TASK GAME EXAMPLES ===\n")
count = 0
for seed in range(42, 100):
    sc = mt_gen(seed, 'airline')
    print(f"Seed {seed} ({sc.scenario_type}): {sc.description}")
    print(f"  User: {sc.initial_message[:250]}")
    actions = []
    for a in sc.expected_actions[:4]:
        actions.append(a.name)
    print(f"  Expected sequence: {' → '.join(actions)}")
    print()
    count += 1
    if count >= 4:
        break
