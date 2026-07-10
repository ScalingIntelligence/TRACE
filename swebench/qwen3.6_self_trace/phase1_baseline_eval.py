"""Phase 1 — baseline SWE-bench Verified eval on Qwen3.6-27B via mini-swe-agent.

vLLM serves the model locally on Pod-E (`http://localhost:8000/v1`).
Per-instance SWE-bench sandboxes run on Modal via `swerex_modal`.
Grading also runs on Modal via `swebench.harness.run_evaluation --modal true`.

Produces the same output contract as before:
  /tmp/trace_pipeline/runs/baseline/preds.json        (mini-extra default)
  /tmp/trace_pipeline/runs/baseline/preds.jsonl       (swebench-grader format)
  /tmp/trace_pipeline/runs/baseline/<iid>/<iid>.traj.json
  /tmp/trace_pipeline/runs/baseline/modal-reports/<grader>.<runid>.json
  /tmp/trace_pipeline/runs/baseline/summary.json      (Pass@1 + counts)

The agent harness, sandboxes, and grader are the canonical SWE-bench eval
stack — matching the prior Qwen3-30B-A3B run in docs/swebench_results.md.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PIPE_ROOT = Path(os.environ.get("TRACE_PIPE_ROOT", "/tmp/trace_pipeline"))
RUN_DIR = PIPE_ROOT / "runs" / "baseline"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("TRACE_MODEL", "Qwen/Qwen3.6-27B")
CONFIG_YAML = os.environ.get(
    "TRACE_PHASE1_CONFIG",
    "/tmp/trace_pipeline_code/configs/swebench_qwen36_vllm.yaml",
)
SUBSET = os.environ.get("TRACE_PHASE1_SUBSET", "verified")
SPLIT = os.environ.get("TRACE_PHASE1_SPLIT", "test")
WORKERS = os.environ.get("TRACE_PHASE1_WORKERS", "60")
RUN_ID = os.environ.get("TRACE_PHASE1_RUN_ID", "trace_baseline_qwen36")


def _run(cmd, **kwargs):
    print(f"[phase1] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kwargs)


def run_agent():
    """mini-extra swebench → writes preds.json + per-instance .traj.json."""
    cmd = [
        "mini-extra", "swebench",
        "-c", "swebench",
        "-c", CONFIG_YAML,
        "--subset", SUBSET,
        "--split", SPLIT,
        "-o", str(RUN_DIR),
        "-w", str(WORKERS),
    ]
    env = os.environ.copy()
    # Ensure /workspace/.local/bin is on PATH (mini-extra + modal live there)
    env["PATH"] = f"/workspace/.local/bin:{env.get('PATH', '')}"
    r = _run(cmd, env=env)
    if r.returncode != 0:
        print(f"[phase1] mini-extra swebench exited {r.returncode}", flush=True)
        return r.returncode
    return 0


def to_jsonl():
    """preds.json → preds.jsonl (one row per dict value) for swebench grader."""
    pj = RUN_DIR / "preds.json"
    if not pj.is_file():
        print(f"[phase1] preds.json missing — agent run did not complete", flush=True)
        return False
    data = json.loads(pj.read_text())
    out = RUN_DIR / "preds.jsonl"
    with open(out, "w") as f:
        for v in data.values():
            f.write(json.dumps(v) + "\n")
    print(f"[phase1] wrote {out} ({len(data)} preds)", flush=True)
    return True


def grade():
    """swebench.harness.run_evaluation --modal true → grading_report.json."""
    out_dir = RUN_DIR / "modal-reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--split", SPLIT,
        "--predictions_path", str(RUN_DIR / "preds.jsonl"),
        "--max_workers", str(WORKERS),
        "--run_id", RUN_ID,
        "--modal", "true",
        "--report_dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["PATH"] = f"/workspace/.local/bin:{env.get('PATH', '')}"
    r = _run(cmd, env=env, cwd=str(RUN_DIR))
    return r.returncode


def collect_summary():
    """Find the grader's *.<run_id>.json report and compute Pass@1."""
    # mini-extra reports as `hosted_vllm__<Model>.<run_id>.json` at cwd
    candidates = list(RUN_DIR.glob(f"*.{RUN_ID}.json"))
    candidates += list((RUN_DIR / "modal-reports").glob(f"*.{RUN_ID}.json"))
    if not candidates:
        print(f"[phase1] no grading report found for run_id={RUN_ID}", flush=True)
        return 1
    report_path = candidates[0]
    print(f"[phase1] grading report: {report_path}", flush=True)
    report = json.loads(report_path.read_text())
    resolved = report.get("resolved_ids", [])
    unresolved = report.get("unresolved_ids", [])
    empty = report.get("empty_patch_ids", [])
    errors = report.get("error_ids", [])
    total = len(resolved) + len(unresolved) + len(empty) + len(errors)
    pass_at_1 = len(resolved) / total if total else 0.0
    summary = {
        "harness": "mini-swe-agent + swerex_modal + swebench-grader",
        "model": MODEL_ID,
        "run_id": RUN_ID,
        "n_total": total,
        "n_resolved": len(resolved),
        "n_unresolved": len(unresolved),
        "n_empty_patch": len(empty),
        "n_error": len(errors),
        "pass_at_1": pass_at_1,
        "pass_at_1_pct": round(100 * pass_at_1, 2),
        "report_path": str(report_path),
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[phase1] DONE — Pass@1 = {summary['n_resolved']}/{summary['n_total']} "
        f"= {summary['pass_at_1_pct']:.2f}%",
        flush=True,
    )
    return 0


def hf_push_baseline():
    """Push run_dir to HF dataset repo (one-shot, idempotent)."""
    hf_repo = os.environ.get("TRACE_HF_BASELINE_REPO",
                              "scalingintelligence/qwen36-trace-baseline")
    if not os.environ.get("HF_TOKEN"):
        print(f"[phase1] HF_TOKEN not set — skipping HF upload to {hf_repo}",
              flush=True)
        return 0
    try:
        from hf_push import push
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from hf_push import push
    return push("dataset", hf_repo, str(RUN_DIR),
                commit_message=f"baseline eval — {MODEL_ID}, run_id={RUN_ID}")


def main():
    t0 = time.time()
    print(f"[phase1] model={MODEL_ID} config={CONFIG_YAML} run_id={RUN_ID}", flush=True)
    rc = run_agent()
    if rc != 0:
        return rc
    if not to_jsonl():
        return 2
    rc = grade()
    if rc != 0:
        print(f"[phase1] grader exited {rc} — attempting summary anyway", flush=True)
    rc = collect_summary()
    # HF push (skips silently if no token)
    hf_push_baseline()
    print(f"[phase1] elapsed={time.time() - t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
