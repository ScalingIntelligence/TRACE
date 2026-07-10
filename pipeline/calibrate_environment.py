#!/usr/bin/env python3
"""Check whether collected TRACE rollouts are trainable.

Input is one or more JSON files produced by `train/collect_rollouts.py`.
The script reports reward distribution, per-seed pass@k, and GRPO group
variance diagnostics, then exits nonzero if the configured thresholds fail.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _as_float(value, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _load_rollouts(paths: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        data = json.loads(path.read_text())
        simulations = data.get("simulations")
        if not isinstance(simulations, list):
            raise ValueError(f"{path} does not look like collect_rollouts JSON: missing simulations[]")
        for pos, sim in enumerate(simulations):
            reward_info = sim.get("reward_info") or {}
            rows.append({
                "source": str(path),
                "pos": pos,
                "task_id": str(sim.get("task_id", pos)),
                "sample_idx": sim.get("sample_idx"),
                "reward": _as_float(reward_info.get("reward")),
            })
    if not rows:
        raise ValueError("No simulations found in rollout file(s).")
    return rows


def _group_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)

    for samples in grouped.values():
        if all(s.get("sample_idx") is not None for s in samples):
            samples.sort(key=lambda s: int(s["sample_idx"]))
        else:
            samples.sort(key=lambda s: s["pos"])
    return dict(grouped)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _rate(n: int, d: int) -> float:
    return n / d if d else 0.0


def _parse_k_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(k for k in out if k > 0))


def _parse_constraints(raw: str) -> dict[int, float]:
    if not raw:
        return {}
    out: dict[int, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, value = part.split("=", 1)
        elif ":" in part:
            k, value = part.split(":", 1)
        else:
            raise ValueError(f"Bad pass@k constraint {part!r}; expected K=VALUE")
        out[int(k)] = float(value)
    return out


def _check_between(name: str, value: float, lo: float | None, hi: float | None, errors: list[str]) -> None:
    if lo is not None and value < lo:
        errors.append(f"{name}={value:.3f} below minimum {lo:.3f}")
    if hi is not None and value > hi:
        errors.append(f"{name}={value:.3f} above maximum {hi:.3f}")


def calibrate(args) -> tuple[dict, list[str], list[str]]:
    rows = _load_rollouts(args.rollouts)
    grouped = _group_by_task(rows)
    rewards = [r["reward"] for r in rows]
    successes = [r >= args.success_threshold for r in rewards]

    group_rewards = [[s["reward"] for s in samples] for samples in grouped.values()]
    sample_counts = [len(g) for g in group_rewards]
    expected = args.group_size

    full_groups = [g for g in group_rewards if len(g) >= expected]
    diagnostic_groups = full_groups if full_groups else group_rewards

    all_success = sum(1 for g in diagnostic_groups if g and all(r >= args.success_threshold for r in g))
    all_zero = sum(1 for g in diagnostic_groups if g and all(r <= args.zero_threshold for r in g))
    constant = sum(1 for g in diagnostic_groups if len({round(r, 8) for r in g}) <= 1)
    informative = len(diagnostic_groups) - constant

    pass_at_k = {}
    for k in args.pass_at_k:
        n = 0
        for samples in grouped.values():
            first_k = samples[:k]
            if any(s["reward"] >= args.success_threshold for s in first_k):
                n += 1
        pass_at_k[k] = _rate(n, len(grouped))

    metrics = {
        "num_rollouts": len(rows),
        "num_tasks": len(grouped),
        "expected_group_size": expected,
        "min_samples_per_task": min(sample_counts),
        "median_samples_per_task": statistics.median(sample_counts),
        "max_samples_per_task": max(sample_counts),
        "mean_reward": _mean(rewards),
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "success_rate": _mean([1.0 if x else 0.0 for x in successes]),
        "zero_rate": _mean([1.0 if r <= args.zero_threshold else 0.0 for r in rewards]),
        "num_groups_diagnostic": len(diagnostic_groups),
        "informative_group_rate": _rate(informative, len(diagnostic_groups)),
        "constant_group_rate": _rate(constant, len(diagnostic_groups)),
        "all_success_group_rate": _rate(all_success, len(diagnostic_groups)),
        "all_zero_group_rate": _rate(all_zero, len(diagnostic_groups)),
        "pass_at_k": pass_at_k,
    }

    errors: list[str] = []
    warnings: list[str] = []

    if not args.allow_partial_groups and metrics["median_samples_per_task"] < expected:
        errors.append(
            "median samples/task is below --group-size. You likely forgot "
            "`--select-topk <GROUP_SIZE>` when running train/collect_rollouts.py."
        )

    _check_between("mean_reward", metrics["mean_reward"], args.mean_min, args.mean_max, errors)
    _check_between("success_rate", metrics["success_rate"], args.success_min, args.success_max, errors)
    _check_between("std_reward", metrics["std_reward"], args.std_min, None, errors)
    _check_between("all_success_group_rate", metrics["all_success_group_rate"], None, args.max_all_success_groups, errors)
    _check_between("all_zero_group_rate", metrics["all_zero_group_rate"], None, args.max_all_zero_groups, errors)
    _check_between("constant_group_rate", metrics["constant_group_rate"], None, args.max_constant_groups, errors)
    _check_between("informative_group_rate", metrics["informative_group_rate"], args.min_informative_groups, None, errors)

    for k, lo in args.min_pass_at_k.items():
        value = pass_at_k.get(k)
        if value is None:
            warnings.append(f"pass@{k} was not requested in --pass-at-k; skipping minimum.")
        elif value < lo:
            errors.append(f"pass@{k}={value:.3f} below minimum {lo:.3f}")

    for k, hi in args.max_pass_at_k.items():
        value = pass_at_k.get(k)
        if value is None:
            warnings.append(f"pass@{k} was not requested in --pass-at-k; skipping maximum.")
        elif value > hi:
            errors.append(f"pass@{k}={value:.3f} above maximum {hi:.3f}")

    return metrics, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate TRACE rollout reward distribution and pass@k calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("rollouts", nargs="+", help="collect_rollouts.py JSON output file(s)")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--zero-threshold", type=float, default=0.0)
    parser.add_argument("--mean-min", type=float, default=0.2)
    parser.add_argument("--mean-max", type=float, default=0.6)
    parser.add_argument("--success-min", type=float, default=0.2)
    parser.add_argument("--success-max", type=float, default=0.6)
    parser.add_argument("--std-min", type=float, default=0.05)
    parser.add_argument("--max-all-success-groups", type=float, default=0.2)
    parser.add_argument("--max-all-zero-groups", type=float, default=0.2)
    parser.add_argument("--max-constant-groups", type=float, default=0.5)
    parser.add_argument("--min-informative-groups", type=float, default=0.5)
    parser.add_argument("--pass-at-k", default="1,4,8,16")
    parser.add_argument("--min-pass-at-k", default="", help="Comma list like 1=0.10,4=0.30")
    parser.add_argument("--max-pass-at-k", default="", help="Comma list like 1=0.70,16=0.95")
    parser.add_argument("--allow-partial-groups", action="store_true")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    args.pass_at_k = _parse_k_list(args.pass_at_k)
    args.min_pass_at_k = _parse_constraints(args.min_pass_at_k)
    args.max_pass_at_k = _parse_constraints(args.max_pass_at_k)

    try:
        metrics, errors, warnings = calibrate(args)
    except Exception as e:
        print(f"calibration failed: {e}", file=sys.stderr)
        return 2

    report = {"metrics": metrics, "errors": errors, "warnings": warnings}
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2))

    print("TRACE environment calibration")
    print("=" * 32)
    for key, value in metrics.items():
        if key == "pass_at_k":
            for k, v in value.items():
                print(f"pass@{k}: {v:.3f}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nFAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nPASSED: rollout distribution is within configured thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
