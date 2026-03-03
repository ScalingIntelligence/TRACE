#!/usr/bin/env python3
"""
Compare two τ²-bench evaluation results.

Usage:
    python compare_results.py <result1.json> <result2.json>
    python compare_results.py results/base_model.json results/ppo_model.json --names "Base Model" "PPO Model"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_task_rewards(results: Dict[str, Any], best_of: bool = False) -> Dict[str, float]:
    """
    Extract task rewards from results.
    Returns a dict mapping task_id to reward.

    If best_of is True, takes the max reward across trials; otherwise averages.
    """
    task_rewards = {}
    simulations = results.get("simulations", [])

    for sim in simulations:
        task_id = str(sim.get("task_id", sim.get("task", {}).get("id", "unknown")))
        # Reward can be in reward_info.reward or directly as reward
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

    # Aggregate rewards for tasks with multiple trials
    for task_id, rewards in task_rewards.items():
        if isinstance(rewards, list):
            task_rewards[task_id] = max(rewards) if best_of else sum(rewards) / len(rewards)

    return task_rewards


def compare_results(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    name1: str = "Model 1",
    name2: str = "Model 2",
    threshold: float = 0.5,
    best_of: bool = False
) -> Dict[str, Any]:
    """
    Compare two sets of results.

    Args:
        results1: First result set
        results2: Second result set
        name1: Name for first model
        name2: Name for second model
        threshold: Reward threshold to consider a task "won" (default 0.5)
        best_of: If True, take max reward across trials instead of average

    Returns:
        Comparison statistics
    """
    rewards1 = get_task_rewards(results1, best_of=best_of)
    rewards2 = get_task_rewards(results2, best_of=best_of)

    # Get all task IDs
    all_tasks = set(rewards1.keys()) | set(rewards2.keys())

    # Categorize tasks
    model1_only_wins = []  # Model 1 wins, Model 2 loses
    model2_only_wins = []  # Model 2 wins, Model 1 loses
    both_win = []          # Both models win
    neither_wins = []      # Both models lose
    model1_better = []     # Model 1 has higher reward (but both may win/lose)
    model2_better = []     # Model 2 has higher reward
    tied = []              # Same reward

    for task_id in sorted(all_tasks, key=lambda x: int(x) if x.isdigit() else x):
        r1 = rewards1.get(task_id, 0.0)
        r2 = rewards2.get(task_id, 0.0)

        win1 = r1 >= threshold
        win2 = r2 >= threshold

        if win1 and win2:
            both_win.append((task_id, r1, r2))
        elif win1 and not win2:
            model1_only_wins.append((task_id, r1, r2))
        elif win2 and not win1:
            model2_only_wins.append((task_id, r1, r2))
        else:
            neither_wins.append((task_id, r1, r2))

        # Track which model is better
        if r1 > r2:
            model1_better.append((task_id, r1, r2))
        elif r2 > r1:
            model2_better.append((task_id, r1, r2))
        else:
            tied.append((task_id, r1, r2))

    # Calculate statistics
    total_tasks = len(all_tasks)
    avg_reward1 = sum(rewards1.values()) / len(rewards1) if rewards1 else 0
    avg_reward2 = sum(rewards2.values()) / len(rewards2) if rewards2 else 0
    win_rate1 = sum(1 for r in rewards1.values() if r >= threshold) / len(rewards1) if rewards1 else 0
    win_rate2 = sum(1 for r in rewards2.values() if r >= threshold) / len(rewards2) if rewards2 else 0

    return {
        "name1": name1,
        "name2": name2,
        "total_tasks": total_tasks,
        "avg_reward1": avg_reward1,
        "avg_reward2": avg_reward2,
        "win_rate1": win_rate1,
        "win_rate2": win_rate2,
        "model1_only_wins": model1_only_wins,
        "model2_only_wins": model2_only_wins,
        "both_win": both_win,
        "neither_wins": neither_wins,
        "model1_better": model1_better,
        "model2_better": model2_better,
        "tied": tied,
        "rewards1": rewards1,
        "rewards2": rewards2,
    }


def print_comparison(comparison: Dict[str, Any], verbose: bool = False):
    """Print comparison results in a readable format."""
    name1 = comparison["name1"]
    name2 = comparison["name2"]

    print("=" * 80)
    print(f"COMPARISON: {name1} vs {name2}")
    print("=" * 80)

    # Overall statistics
    print(f"\n{'OVERALL STATISTICS':^80}")
    print("-" * 80)
    print(f"{'Metric':<30} {name1:>20} {name2:>20}")
    print("-" * 80)
    print(f"{'Total Tasks':<30} {comparison['total_tasks']:>20}")
    print(f"{'Average Reward':<30} {comparison['avg_reward1']:>20.4f} {comparison['avg_reward2']:>20.4f}")
    print(f"{'Win Rate (reward >= 0.5)':<30} {comparison['win_rate1']:>19.1%} {comparison['win_rate2']:>19.1%}")

    # Win comparison
    print(f"\n{'WIN COMPARISON':^80}")
    print("-" * 80)
    print(f"Both models win:              {len(comparison['both_win']):>5} tasks")
    print(f"Only {name1} wins:      {len(comparison['model1_only_wins']):>5} tasks")
    print(f"Only {name2} wins:      {len(comparison['model2_only_wins']):>5} tasks")
    print(f"Neither model wins:           {len(comparison['neither_wins']):>5} tasks")

    # Head-to-head
    print(f"\n{'HEAD-TO-HEAD (by reward value)':^80}")
    print("-" * 80)
    print(f"{name1} better:          {len(comparison['model1_better']):>5} tasks")
    print(f"{name2} better:          {len(comparison['model2_better']):>5} tasks")
    print(f"Tied:                         {len(comparison['tied']):>5} tasks")

    # Detailed task lists
    if comparison['model1_only_wins']:
        print(f"\n{'TASKS WON ONLY BY ' + name1.upper():^80}")
        print("-" * 80)
        print(f"{'Task ID':<15} {name1 + ' Reward':>20} {name2 + ' Reward':>20}")
        print("-" * 80)
        for task_id, r1, r2 in comparison['model1_only_wins']:
            print(f"{task_id:<15} {r1:>20.4f} {r2:>20.4f}")

    if comparison['model2_only_wins']:
        print(f"\n{'TASKS WON ONLY BY ' + name2.upper():^80}")
        print("-" * 80)
        print(f"{'Task ID':<15} {name1 + ' Reward':>20} {name2 + ' Reward':>20}")
        print("-" * 80)
        for task_id, r1, r2 in comparison['model2_only_wins']:
            print(f"{task_id:<15} {r1:>20.4f} {r2:>20.4f}")

    if verbose:
        if comparison['both_win']:
            print(f"\n{'TASKS WON BY BOTH MODELS':^80}")
            print("-" * 80)
            print(f"{'Task ID':<15} {name1 + ' Reward':>20} {name2 + ' Reward':>20}")
            print("-" * 80)
            for task_id, r1, r2 in comparison['both_win']:
                print(f"{task_id:<15} {r1:>20.4f} {r2:>20.4f}")

        if comparison['neither_wins']:
            print(f"\n{'TASKS LOST BY BOTH MODELS':^80}")
            print("-" * 80)
            print(f"{'Task ID':<15} {name1 + ' Reward':>20} {name2 + ' Reward':>20}")
            print("-" * 80)
            for task_id, r1, r2 in comparison['neither_wins']:
                print(f"{task_id:<15} {r1:>20.4f} {r2:>20.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two τ²-bench evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_results.py result1.json result2.json
  python compare_results.py base.json ppo.json --names "Base Model" "PPO Model"
  python compare_results.py result1.json result2.json --verbose
  python compare_results.py result1.json result2.json --threshold 0.8
        """
    )

    parser.add_argument("result1", help="Path to first results JSON file")
    parser.add_argument("result2", help="Path to second results JSON file")
    parser.add_argument(
        "--names", "-n",
        nargs=2,
        default=None,
        help="Names for the two models (default: derived from filenames)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Reward threshold to consider a task 'won' (default: 0.5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed task lists for all categories"
    )
    parser.add_argument(
        "--best-of",
        action="store_true",
        help="Use max reward across trials instead of average"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Check files exist
    if not Path(args.result1).exists():
        print(f"Error: File not found: {args.result1}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.result2).exists():
        print(f"Error: File not found: {args.result2}", file=sys.stderr)
        sys.exit(1)

    # Load results
    results1 = load_results(args.result1)
    results2 = load_results(args.result2)

    # Determine names
    if args.names:
        name1, name2 = args.names
    else:
        name1 = Path(args.result1).stem
        name2 = Path(args.result2).stem

    # Compare
    comparison = compare_results(results1, results2, name1, name2, args.threshold, best_of=args.best_of)

    # Output
    if args.json:
        # Convert tuples to lists for JSON serialization
        output = {
            k: v if not isinstance(v, list) or not v or not isinstance(v[0], tuple)
            else [list(item) for item in v]
            for k, v in comparison.items()
        }
        print(json.dumps(output, indent=2))
    else:
        print_comparison(comparison, verbose=args.verbose)


if __name__ == "__main__":
    main()
