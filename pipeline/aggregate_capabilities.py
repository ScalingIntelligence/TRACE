#!/usr/bin/env python3
"""
TRACE Capability Aggregation (paper-faithful)

Aggregates per-run labeling outputs from Phase 2 of the capability selection pipeline
and applies the paper's dual-threshold filter (rho, delta) plus cross-run consistency.

For each capability c and each run r, given the per-trajectory labels (NA / PRESENT /
LACKING) on both failed and successful trajectories:

  Cov_r(c)  = lacking_failed / total_failed
              (fraction of failed trajectories that LACK this capability)

  ER_minus  = lacking_failed / (lacking_failed + present_failed)
              (rate of LACKING among failed trajectories where the capability is applicable)

  ER_plus   = lacking_passed / (lacking_passed + present_passed)
              (rate of LACKING among successful trajectories where the capability is applicable)

  Delta_r   = ER_minus - ER_plus
              (the contrastive gap — how much more lacking on failed than passed)

A capability passes a single run iff Cov_r(c) >= rho AND Delta_r(c) >= delta.
A capability is selected iff it passes in at least K out of N runs (cross-run consistency).

Usage:
  python pipeline/aggregate_capabilities.py \\
    --runs pipeline/run_*.json \\
    --rho 0.10 \\
    --delta 0.20 \\
    --k 8 \\
    --candidates pipeline/candidate_capabilities.json \\
    --export pipeline/selected_capabilities.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_metrics(labels: dict, totals: dict) -> dict:
    """For each capability, compute Cov and Delta from the label counts in one run."""
    metrics = {}
    total_failed = totals.get("total_failed", 0)
    total_passed = totals.get("total_passed", 0)

    for cap, lbls in labels.items():
        lf = lbls.get("lacking_failed", [])
        pf = lbls.get("present_failed", [])
        lp = lbls.get("lacking_passed", [])
        pp = lbls.get("present_passed", [])
        n_lf, n_pf, n_lp, n_pp = len(lf), len(pf), len(lp), len(pp)

        cov = n_lf / total_failed if total_failed > 0 else 0.0
        er_minus = n_lf / (n_lf + n_pf) if (n_lf + n_pf) > 0 else 0.0
        er_plus = n_lp / (n_lp + n_pp) if (n_lp + n_pp) > 0 else 0.0
        delta = er_minus - er_plus

        metrics[cap] = {
            "cov": cov,
            "delta": delta,
            "er_minus": er_minus,
            "er_plus": er_plus,
            "n_lacking_failed": n_lf,
            "n_present_failed": n_pf,
            "n_lacking_passed": n_lp,
            "n_present_passed": n_pp,
            "lacking_failed_ids": lf,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 2 labeling runs and apply (rho, delta, K) filter"
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Per-run JSON files from Phase 2 (glob patterns ok)")
    parser.add_argument("--rho", type=float, default=0.10,
                        help="Coverage threshold (paper default: 0.10)")
    parser.add_argument("--delta", type=float, default=0.20,
                        help="Contrastive gap threshold (paper default: 0.20)")
    parser.add_argument("--k", type=int, default=8,
                        help="Min runs a capability must pass to be kept (default: 8)")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Phase 1 candidate capabilities JSON (for descriptions)")
    parser.add_argument("--export", type=str, required=True,
                        help="Path to write the final selected_capabilities.json")
    args = parser.parse_args()

    # Expand globs
    run_paths = []
    for p in args.runs:
        matched = sorted(glob.glob(p))
        if matched:
            run_paths.extend(matched)
        elif Path(p).exists():
            run_paths.append(p)
    print(f"Loaded {len(run_paths)} run files")
    if not run_paths:
        print("ERROR: no run files found")
        return

    runs = [load_run(p) for p in run_paths]
    n_runs = len(runs)

    # Load candidate descriptions for the final export
    descriptions = {}
    if args.candidates and Path(args.candidates).exists():
        with open(args.candidates) as f:
            cands = json.load(f)
        for c in cands.get("candidates", []):
            descriptions[c["name"]] = c.get("description", "")

    # Compute per-run metrics
    all_caps = set()
    per_run_metrics = []
    for run in runs:
        m = compute_metrics(run.get("labels", {}), run.get("totals", {}))
        per_run_metrics.append(m)
        all_caps.update(m.keys())

    # Aggregate per capability
    print(f"\n{'Capability':<45s} {'mean Cov':>10s} {'mean Δ':>10s} {'passes':>10s}")
    print("-" * 80)

    summary = []
    for cap in sorted(all_caps):
        covs = [m[cap]["cov"] for m in per_run_metrics if cap in m]
        deltas = [m[cap]["delta"] for m in per_run_metrics if cap in m]
        passes_per_run = [
            (cap in m
             and m[cap]["cov"] >= args.rho
             and m[cap]["delta"] >= args.delta)
            for m in per_run_metrics
        ]
        n_passes = sum(passes_per_run)

        # Union of failed task IDs labeled LACKING across all runs
        failed_union = set()
        for m in per_run_metrics:
            if cap in m:
                failed_union.update(m[cap]["lacking_failed_ids"])

        summary.append({
            "skill": cap,
            "description": descriptions.get(cap, ""),
            "mean_cov": float(np.mean(covs)) if covs else 0.0,
            "std_cov": float(np.std(covs)) if covs else 0.0,
            "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
            "std_delta": float(np.std(deltas)) if deltas else 0.0,
            "n_runs_passed": n_passes,
            "n_runs_total": n_runs,
            "passes_consistency": n_passes >= args.k,
            "example_failed_cases": sorted(failed_union)[:8],
            "status": "PENDING",
        })

        flag = "PASS" if n_passes >= args.k else "    "
        print(f"{cap:<45s} {np.mean(covs):>10.3f} {np.mean(deltas):>10.3f} "
              f"{n_passes:>4d}/{n_runs:<4d} {flag}")

    # Filter and sort by mean_delta descending (largest gap first)
    selected = [s for s in summary if s["passes_consistency"]]
    selected.sort(key=lambda x: -x["mean_delta"])

    print(f"\nSelected {len(selected)}/{len(summary)} capabilities "
          f"(rho={args.rho}, delta={args.delta}, k={args.k}/{n_runs})")
    print(f"Filter: capabilities must satisfy Cov >= {args.rho} AND Delta >= {args.delta} "
          f"in at least {args.k}/{n_runs} runs")

    with open(args.export, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nExported to {args.export}")

    if selected:
        print(f"\nTop selected capabilities (sorted by mean Δ):")
        for s in selected[:10]:
            print(f"  - {s['skill']:<45s} Cov={s['mean_cov']:.3f}  "
                  f"Δ={s['mean_delta']:.3f}  ({s['n_runs_passed']}/{n_runs})")


if __name__ == "__main__":
    main()
