#!/usr/bin/env python3
"""
Show uniquely solved problems across multiple result files.

Usage:
    python unique_solved.py result1.json result2.json [result3.json ...]
    python unique_solved.py result1.json result2.json --names "Base" "LoRA" --threshold 0.5
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_task_rewards(results, best_of=False):
    task_rewards = {}
    for sim in results.get("simulations", []):
        task_id = str(sim.get("task_id", sim.get("task", {}).get("id", "unknown")))
        reward_info = sim.get("reward_info", {})
        reward = reward_info.get("reward") if reward_info else sim.get("reward", 0.0)
        if reward is None:
            reward = 0.0
        if task_id in task_rewards:
            existing = task_rewards[task_id]
            if isinstance(existing, list):
                existing.append(reward)
            else:
                task_rewards[task_id] = [existing, reward]
        else:
            task_rewards[task_id] = reward

    for task_id, rewards in task_rewards.items():
        if isinstance(rewards, list):
            task_rewards[task_id] = max(rewards) if best_of else sum(rewards) / len(rewards)

    return task_rewards


def main():
    parser = argparse.ArgumentParser(description="Show uniquely solved problems across multiple result files")
    parser.add_argument("files", nargs='+', help="Result JSON files")
    parser.add_argument("--names", "-n", nargs='+', default=None, help="Names for each file")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Win threshold (default: 0.5)")
    parser.add_argument("--best-of", action="store_true", help="Use max reward across trials")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.files):
        print("Error: number of --names must match number of files", file=sys.stderr)
        sys.exit(1)

    names = args.names or [Path(f).stem for f in args.files]

    # Load all results and get winning task sets
    win_sets = []
    all_rewards = []
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        rewards = get_task_rewards(load_results(filepath), best_of=args.best_of)
        all_rewards.append(rewards)
        win_sets.append({tid for tid, r in rewards.items() if r >= args.threshold})

    all_tasks = set()
    for r in all_rewards:
        all_tasks |= r.keys()

    def sort_key(tid):
        return (int(tid) if tid.isdigit() else float('inf'), tid)

    # Print per-file summary
    print("=" * 70)
    print(f"{'SUMMARY':^70}")
    print("=" * 70)
    for i, name in enumerate(names):
        print(f"  {name}: {len(win_sets[i])}/{len(all_tasks)} solved")
    union = set().union(*win_sets)
    print(f"  Union: {len(union)}/{len(all_tasks)} solved")
    intersection = set.intersection(*win_sets) if win_sets else set()
    print(f"  Intersection (all solve): {len(intersection)}/{len(all_tasks)}")

    # Uniquely solved by each file
    print()
    print("=" * 70)
    print(f"{'UNIQUELY SOLVED (solved by exactly one file)':^70}")
    print("=" * 70)

    for i, name in enumerate(names):
        unique = win_sets[i] - set().union(*(win_sets[j] for j in range(len(win_sets)) if j != i))
        print(f"\n  Only {name} solves ({len(unique)}):")
        if unique:
            for tid in sorted(unique, key=sort_key):
                rewards_str = ", ".join(
                    f"{names[j]}={all_rewards[j].get(tid, 0.0):.2f}" for j in range(len(names))
                )
                print(f"    T{tid}: {rewards_str}")
        else:
            print(f"    (none)")

    # Solved by none
    unsolved = all_tasks - union
    print(f"\n  Solved by none ({len(unsolved)}):")
    if unsolved:
        for tid in sorted(unsolved, key=sort_key):
            print(f"    T{tid}")
    else:
        print(f"    (none)")

    print()


if __name__ == "__main__":
    main()
