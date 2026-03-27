#!/usr/bin/env python3
"""
Contrastive Skill Selection Analyzer

Analyzes judge model outputs across multiple attempts to determine
which skills are consistently selected and meet a minimum success threshold.

Input JSON format:
[
  {
    "attempt": 1,
    "result": {
      "skill_name": {
        "assigned_model": "tool_calling",
        "success_rate": 0.85,
        "failed_cases": ["T1", "T26"],
        "total_cases": 50
      },
      ...
    }
  },
  ...
]

Usage:
  python analyze_contrastive_skills.py results.json --threshold 10
  python analyze_contrastive_skills.py results.json --threshold 10 --plot --output skills_analysis.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def aggregate_skills(attempts: list[dict]) -> dict:
    """Aggregate skill stats across all attempts.

    Returns dict of skill_name -> {
        appearances: int,
        success_rates: list[float],
        assigned_models: list[str],
        failed_cases_union: set,
        total_cases: list[int],
    }
    """
    skills = defaultdict(lambda: {
        "appearances": 0,
        "success_rates": [],
        "assigned_models": [],
        "failed_cases_union": set(),
        "total_cases": [],
    })

    for attempt in attempts:
        result = attempt.get("result", {})
        for skill_name, skill_data in result.items():
            s = skills[skill_name]
            s["appearances"] += 1

            # Handle success_rate: can be float (0.85) or string ("85%")
            sr = skill_data.get("success_rate", 0)
            if isinstance(sr, str):
                sr = float(sr.strip("%")) / 100.0
            s["success_rates"].append(float(sr))

            model = skill_data.get("assigned_model", "unknown")
            s["assigned_models"].append(model)

            failed = skill_data.get("failed_cases", [])
            if isinstance(failed, list):
                s["failed_cases_union"].update(failed)
            elif isinstance(failed, (int, float, str)):
                # If failed_cases is a percentage or count, store as-is
                pass

            tc = skill_data.get("total_cases", 0)
            if tc:
                s["total_cases"].append(tc)

    return dict(skills)


def compute_summary(skills: dict, num_attempts: int) -> list[dict]:
    """Compute summary stats for each skill."""
    rows = []
    for name, data in sorted(skills.items()):
        rates = data["success_rates"]
        models = data["assigned_models"]
        # Most common model assignment
        model_counts = defaultdict(int)
        for m in models:
            model_counts[m] += 1
        dominant_model = max(model_counts, key=model_counts.get)
        model_consistency = model_counts[dominant_model] / len(models) * 100

        rows.append({
            "skill": name,
            "appearances": data["appearances"],
            "appearance_rate": data["appearances"] / num_attempts * 100,
            "mean_success_rate": np.mean(rates) * 100,
            "std_success_rate": np.std(rates) * 100 if len(rates) > 1 else 0.0,
            "min_success_rate": np.min(rates) * 100,
            "max_success_rate": np.max(rates) * 100,
            "dominant_model": dominant_model,
            "model_consistency": model_consistency,
            "unique_failed_cases": len(data["failed_cases_union"]),
        })
    return rows


def filter_by_threshold(rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    """Split rows into kept and discarded based on mean_success_rate >= threshold."""
    kept = [r for r in rows if r["mean_success_rate"] >= threshold]
    discarded = [r for r in rows if r["mean_success_rate"] < threshold]
    return kept, discarded


def print_table(rows: list[dict], title: str):
    """Print a formatted table to stdout."""
    if not rows:
        print(f"\n{title}: (none)")
        return

    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    header = (
        f"{'Skill':<25} {'Appear':<8} {'Mean%':<8} {'Std%':<7} "
        f"{'Min%':<7} {'Max%':<7} {'Model':<15} {'MdlCon%':<8} {'FailUnion':<9}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(rows, key=lambda x: x["mean_success_rate"], reverse=True):
        print(
            f"{r['skill']:<25} "
            f"{r['appearances']:<8} "
            f"{r['mean_success_rate']:<8.1f} "
            f"{r['std_success_rate']:<7.1f} "
            f"{r['min_success_rate']:<7.1f} "
            f"{r['max_success_rate']:<7.1f} "
            f"{r['dominant_model']:<15} "
            f"{r['model_consistency']:<8.1f} "
            f"{r['unique_failed_cases']:<9}"
        )


def print_consistency_matrix(attempts: list[dict]):
    """Show which skills appear in which attempts."""
    all_skills = set()
    for a in attempts:
        all_skills.update(a.get("result", {}).keys())
    all_skills = sorted(all_skills)

    if not all_skills:
        return

    print(f"\n{'=' * 80}")
    print(" Skill Presence Across Attempts")
    print(f"{'=' * 80}")

    n_attempts = len(attempts)
    # Header
    header = f"{'Skill':<25} " + " ".join(f"A{a['attempt']:>2}" for a in attempts) + "  Count"
    print(header)
    print("-" * len(header))

    for skill in all_skills:
        presence = []
        count = 0
        for a in attempts:
            if skill in a.get("result", {}):
                presence.append(" ✓ ")
                count += 1
            else:
                presence.append(" · ")
        print(f"{skill:<25} {''.join(presence)}  {count}/{n_attempts}")


def plot_results(kept: list[dict], discarded: list[dict], threshold: float, output_path: str):
    """Generate a bar plot of skill success rates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[WARNING] matplotlib not installed, skipping plot. Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(kept + discarded) * 0.8), 6))

    all_rows = sorted(kept + discarded, key=lambda x: x["mean_success_rate"], reverse=True)
    names = [r["skill"] for r in all_rows]
    means = [r["mean_success_rate"] for r in all_rows]
    stds = [r["std_success_rate"] for r in all_rows]
    colors = ["#2196F3" if r["mean_success_rate"] >= threshold else "#EF5350" for r in all_rows]

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold}%)")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Success Rate (%)")
    ax.set_title("Contrastive Skill Selection Analysis")
    ax.legend()
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=8)

    # Add appearance rate as secondary info
    for i, r in enumerate(all_rows):
        ax.text(i, 2, f"{r['appearances']}x", ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n[INFO] Plot saved to: {output_path}")


def export_selected_skills(kept: list[dict], output_path: str):
    """Export the selected skills as a JSON for downstream use."""
    selected = []
    for r in sorted(kept, key=lambda x: x["mean_success_rate"], reverse=True):
        selected.append({
            "skill": r["skill"],
            "assigned_model": r["dominant_model"],
            "mean_success_rate": round(r["mean_success_rate"], 2),
            "consistency": round(r["model_consistency"], 2),
            "appearances": r["appearances"],
        })

    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"[INFO] Selected skills exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze contrastive skill selection results")
    parser.add_argument("input", help="Path to JSON results file")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Minimum mean success rate (%%) to keep a skill (default: 10)")
    parser.add_argument("--plot", action="store_true", help="Generate a bar plot")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the plot (default: <input>_analysis.png)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export selected skills to JSON file")
    args = parser.parse_args()

    # Load
    attempts = load_results(args.input)
    num_attempts = len(attempts)
    print(f"Loaded {num_attempts} attempt(s) from {args.input}")

    # Aggregate
    skills = aggregate_skills(attempts)
    summary = compute_summary(skills, num_attempts)

    # Filter
    kept, discarded = filter_by_threshold(summary, args.threshold)

    # Print tables
    print_table(kept, f"SELECTED SKILLS (mean success >= {args.threshold}%)")
    print_table(discarded, f"DISCARDED SKILLS (mean success < {args.threshold}%)")

    # Consistency matrix
    print_consistency_matrix(attempts)

    # Summary stats
    print(f"\n{'=' * 80}")
    print(" Summary")
    print(f"{'=' * 80}")
    print(f"  Total unique skills across all attempts: {len(summary)}")
    print(f"  Skills kept (>= {args.threshold}%): {len(kept)}")
    print(f"  Skills discarded (< {args.threshold}%): {len(discarded)}")
    if kept:
        best = max(kept, key=lambda x: x["mean_success_rate"])
        print(f"  Best skill: {best['skill']} ({best['mean_success_rate']:.1f}% mean success)")
        avg_consistency = np.mean([r["model_consistency"] for r in kept])
        print(f"  Avg model consistency (kept): {avg_consistency:.1f}%")

    # Plot
    if args.plot:
        output_path = args.output or str(Path(args.input).stem + "_analysis.png")
        plot_results(kept, discarded, args.threshold, output_path)

    # Export
    if args.export:
        export_selected_skills(kept, args.export)


if __name__ == "__main__":
    main()
