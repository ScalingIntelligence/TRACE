"""Phase 5 — GRPO training handoff sketch for the internal Pod-T setup.

This script is not a complete public trainer launcher. It records the internal
handoff shape used on Pod-E/Pod-T, but the actual `torchrun` command is still a
stub until the release includes the missing SWE runtime helper and cluster
packaging.

It:

  1. Bundles scenarios_final.json + a runtime wrapper into a tarball.
  2. Copies the tarball to Pod-T via `oc cp`.
  3. Triggers Pod-T to:
       - install minimum dependencies (datasets, requests, etc. — they're
         already there in tarun-* pods; openjarvis pods may need install)
       - start a co-located vLLM on Pod-T (2 GPUs)
       - launch the GRPO trainer (2 GPUs trainer w/ data parallel,
         since only 4 GPUs total)
       - push iter5/iter10/iter15 LoRA checkpoints to HF
  4. Poll for iter15 to land on HF.
  5. Tear down the trainer.

Because Pod-T's openjarvis-132 has only 4 GPUs (vs the 8-GPU pods we used
before), we use a smaller trainer footprint: 2 trainer workers + 2 vLLM.

Outputs when completed:
  /tmp/trace_pipeline/training/state.json   (poll status)
  HF repo  $TRACE_TRAINER_HF_REPO  with grpo_ckpt_iter_5/10/15
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PIPE_ROOT = Path(os.environ.get("TRACE_PIPE_ROOT", "/tmp/trace_pipeline"))
TRAIN_DIR = PIPE_ROOT / "training"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

POD_T = os.environ.get("TRACE_POD_T", "openjarvis-distributedruns-132-master-0")
NAMESPACE = os.environ.get("TRACE_NAMESPACE", "hazy-labs")
HF_REPO = os.environ.get("TRACE_TRAINER_HF_REPO", "tarsur385/qwen36-trace-v1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_REPO = os.environ.get("TRACE_HF_REPO", "Qwen/Qwen3.6-27B")
N_EPOCHS = int(os.environ.get("TRACE_TRAIN_EPOCHS", "15"))
SAVE_EVERY = int(os.environ.get("TRACE_TRAIN_SAVE_EVERY", "5"))
POLL_INTERVAL = int(os.environ.get("TRACE_TRAIN_POLL_S", "300"))
MAX_TRAIN_HOURS = int(os.environ.get("TRACE_TRAIN_MAX_HOURS", "12"))


def oc_exec(pod: str, cmd: str, timeout: int = 60):
    full = ["oc", "exec", pod, "-n", NAMESPACE, "--", "/bin/bash", "-c", cmd]
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def oc_cp(src: str, pod: str, dst: str):
    return subprocess.run(
        ["oc", "cp", src, f"{NAMESPACE}/{pod}:{dst}"],
        capture_output=True, text=True, timeout=300,
    )


def hf_has_iter15():
    """Check if the iter15 checkpoint is on HF."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN or None)
    try:
        for f in api.list_repo_files(HF_REPO):
            if "grpo_ckpt_iter_15" in f:
                return True
    except Exception:
        pass
    return False


def main():
    if os.environ.get("TRACE_ALLOW_PHASE5_STUB", "0") != "1":
        print(
            "[phase5] This script is an internal handoff sketch and still "
            "contains TRAINER_LAUNCH_STUB. Set TRACE_ALLOW_PHASE5_STUB=1 only "
            "if you intentionally want to exercise the pod-copy skeleton.",
            flush=True,
        )
        return 2

    scenarios = json.loads((PIPE_ROOT / "env" / "scenarios_final.json").read_text())
    print(f"[phase5] training on {len(scenarios)} trainable scenarios "
          f"pod_T={POD_T}", flush=True)
    print(f"[phase5] hf_repo={HF_REPO} model={MODEL_REPO}", flush=True)

    if hf_has_iter15():
        print(f"[phase5] iter15 already on HF; skipping training", flush=True)
        return 0

    # Step 1: copy scenarios + trainer launcher to Pod-T
    scenario_path = TRAIN_DIR / "scenarios.json"
    scenario_path.write_text(json.dumps(scenarios, indent=2))
    print(f"[phase5] uploading scenarios to {POD_T}", flush=True)
    oc_exec(POD_T, "mkdir -p /tmp/trace_train", timeout=30)
    r = oc_cp(str(scenario_path), POD_T, "/tmp/trace_train/scenarios.json")
    if r.returncode != 0:
        print(f"[phase5] oc cp failed: {r.stderr}", flush=True)
        return 2

    # Step 2: install dependencies + launch trainer skeleton
    # NOTE: full trainer wiring is left as a TODO that requires our train_grpo
    # infrastructure on Pod-T. For openjarvis pods we install minimum env.
    launcher = TRAIN_DIR / "launch_train.sh"
    launcher.write_text(f"""#!/bin/bash
set -euo pipefail
export HF_TOKEN={HF_TOKEN}
export TRACE_HF_REPO={HF_REPO}
export MODEL_REPO={MODEL_REPO}
export PYTHONUNBUFFERED=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Start co-located vLLM on GPUs 0,1
mkdir -p /tmp/trace_train/logs
CUDA_VISIBLE_DEVICES=0,1 \\
  vllm serve $MODEL_REPO \\
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \\
  --tensor-parallel-size 2 --max-model-len 16384 \\
  --gpu-memory-utilization 0.85 \\
  --enable-lora --max-loras 1 --max-lora-rank 16 \\
  --served-model-name {MODEL_REPO} \\
  --enforce-eager --trust-remote-code \\
  > /tmp/trace_train/logs/vllm.log 2>&1 &
echo "VLLM_PID=$!"

# TRAINER STUB: a real GRPO loop would go here. We emit checkpoints to
# /tmp/trace_train/ckpt_iter_{{5,10,15}}/ and rely on a watcher to push them.
# Because the openjarvis pod doesn't have our trainer infra, this is a
# placeholder; in practice we'd `oc cp` the trainer code from rl-master if
# alive, or rebuild from /workspace/games when the trainer image is set up.
echo "TRAINER_LAUNCH_STUB"
""")
    r = oc_cp(str(launcher), POD_T, "/tmp/trace_train/launch_train.sh")
    if r.returncode != 0:
        print(f"[phase5] launcher upload failed: {r.stderr}", flush=True)
        return 2
    oc_exec(POD_T, "chmod +x /tmp/trace_train/launch_train.sh")

    # Step 3: launch trainer (background; survives this script's exit)
    cmd = (
        "setsid /tmp/trace_train/launch_train.sh "
        "> /tmp/trace_train/launch.log 2>&1 < /dev/null & "
        "echo TRAINER_PID=$!"
    )
    rc, so, se = oc_exec(POD_T, cmd, timeout=30)
    print(f"[phase5] launched: rc={rc} stdout={so} stderr={se}", flush=True)

    # Step 4: poll HF for iter15
    t0 = time.time()
    while time.time() - t0 < MAX_TRAIN_HOURS * 3600:
        if hf_has_iter15():
            print(f"[phase5] iter15 found on HF after "
                  f"{(time.time()-t0)/3600:.1f}h", flush=True)
            return 0
        elapsed_h = (time.time() - t0) / 3600
        print(f"[phase5] waiting … elapsed={elapsed_h:.1f}h", flush=True)
        time.sleep(POLL_INTERVAL)

    print(f"[phase5] training timed out after {MAX_TRAIN_HOURS}h", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
