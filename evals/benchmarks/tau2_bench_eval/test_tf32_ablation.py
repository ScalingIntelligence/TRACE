#!/usr/bin/env python3
"""
A/B test: Does the TF32 disable fix actually affect determinism for bf16 models?

Plan:
  1. Run 3 tasks with solo adapter (WITH fix) twice on port 9000 → compare
  2. Run 3 tasks with 5-skill orchestrator (WITH fix) twice on port 9000 → compare
  3. Remove TF32 fix from runtime, restart port 9003
  4. Run same 3 tasks solo adapter (WITHOUT fix) twice on port 9003 → compare
  5. Run same 3 tasks orchestrator (WITHOUT fix) twice on port 9003 → compare

This script handles steps 1-2 and 4-5 (restart is manual).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
TASK_IDS = [2, 15, 37]  # Mix of difficulty levels


def run_eval(config_path: str, save_tag: str, port: int, task_ids=None):
    """Run tau2 eval with given config, overriding port and save path."""
    if task_ids is None:
        task_ids = TASK_IDS

    save_to = f"single_lora/results/tf32-ablation-{save_tag}"

    cmd = [
        sys.executable, str(EVAL_DIR / "main.py"),
        "--config", config_path,
        "--task-ids", *[str(t) for t in task_ids],
        "--save-to", save_to,
    ]

    # Override port in the config via env
    env = os.environ.copy()

    print(f"\n{'='*60}")
    print(f"Running: {save_tag}")
    print(f"  Config: {Path(config_path).name}")
    print(f"  Port: {port}")
    print(f"  Tasks: {task_ids}")
    print(f"  Save: {save_to}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(EVAL_DIR), env=env, timeout=600)
    if result.returncode != 0:
        print(f"WARNING: eval returned {result.returncode}")

    return save_to


def find_result_file(save_to: str) -> Path | None:
    """Find the simulation result file."""
    base = EVAL_DIR / save_to
    # tau2 saves results under data/simulations/ relative to the package
    patterns = [
        base / "**/*.json",
        Path(os.path.expanduser("~")) / "hangook/games/tau2-bench/data/simulations" / Path(save_to).name / "*.json",
    ]

    # Direct check
    if base.exists():
        for f in sorted(base.rglob("*.json")):
            return f

    # Check tau2 data dir
    tau2_data = os.environ.get("TAU2_DATA_DIR", "")
    if tau2_data:
        sim_dir = Path(tau2_data) / "simulations"
        if sim_dir.exists():
            # Find most recent matching directory
            for d in sorted(sim_dir.iterdir(), reverse=True):
                if save_to.split("/")[-1] in d.name:
                    for f in sorted(d.rglob("*.json")):
                        return f
    return None


def extract_actions(result_path: Path) -> dict:
    """Extract task-level actions/messages from result file for comparison."""
    with open(result_path) as f:
        data = json.load(f)

    # Extract per-task results
    summaries = {}
    if isinstance(data, list):
        for item in data:
            tid = item.get("task_id", "unknown")
            # Get agent messages only (exclude user sim messages)
            messages = []
            for msg in item.get("messages", item.get("trajectory", [])):
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])
                    if role == "assistant":
                        messages.append({
                            "content": content,
                            "tool_calls": tool_calls,
                        })
            summaries[str(tid)] = messages
    elif isinstance(data, dict):
        # Could be a single result or wrapped format
        for key in ["results", "simulations", "data"]:
            if key in data:
                return extract_actions_from_list(data[key])
        # Single result
        tid = data.get("task_id", "unknown")
        summaries[str(tid)] = data

    return summaries


def extract_actions_from_list(results_list):
    summaries = {}
    for item in results_list:
        tid = item.get("task_id", "unknown")
        messages = []
        for msg in item.get("messages", item.get("trajectory", [])):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                messages.append({
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls", []),
                })
        summaries[str(tid)] = messages
    return summaries


def hash_results(result_path: Path) -> str:
    """Hash the entire result file."""
    with open(result_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def compare_runs(tag_a: str, tag_b: str, result_dir: Path) -> bool:
    """Compare two runs by finding their result files and diffing."""
    # Find result files
    files_a = sorted(result_dir.rglob(f"*{tag_a}*/**/*.json")) if result_dir.exists() else []
    files_b = sorted(result_dir.rglob(f"*{tag_b}*/**/*.json")) if result_dir.exists() else []

    if not files_a or not files_b:
        print(f"Could not find result files for {tag_a} or {tag_b}")
        return False

    hash_a = hash_results(files_a[0])
    hash_b = hash_results(files_b[0])

    match = hash_a == hash_b
    print(f"\n{'✓' if match else '✗'} {tag_a} vs {tag_b}: {'MATCH' if match else 'DIFFER'}")
    print(f"  {files_a[0].name}: {hash_a}")
    print(f"  {files_b[0].name}: {hash_b}")

    return match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["with-fix", "without-fix", "compare"], required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--solo-config", default="single_lora/config-airline-solo-multistep.yml")
    parser.add_argument("--orch-config", default="single_lora/config-airline-5skill-orchestrator.yml")
    args = parser.parse_args()

    if args.phase == "with-fix":
        # Run solo adapter twice
        run_eval(args.solo_config, f"solo-fix-run1-p{args.port}", args.port)
        run_eval(args.solo_config, f"solo-fix-run2-p{args.port}", args.port)

        # Run orchestrator twice
        run_eval(args.orch_config, f"orch-fix-run1-p{args.port}", args.port)
        run_eval(args.orch_config, f"orch-fix-run2-p{args.port}", args.port)

    elif args.phase == "without-fix":
        # Run solo adapter twice
        run_eval(args.solo_config, f"solo-nofix-run1-p{args.port}", args.port)
        run_eval(args.solo_config, f"solo-nofix-run2-p{args.port}", args.port)

        # Run orchestrator twice
        run_eval(args.orch_config, f"orch-nofix-run1-p{args.port}", args.port)
        run_eval(args.orch_config, f"orch-nofix-run2-p{args.port}", args.port)

    print("\n" + "="*60)
    print("Phase complete. Run with --phase compare to see results.")


if __name__ == "__main__":
    main()
