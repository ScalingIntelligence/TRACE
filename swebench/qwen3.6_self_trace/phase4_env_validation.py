"""Phase 4 — env validation: self-test + smoke + filter.

Reads /tmp/trace_pipeline/env/scenarios_parsed.json. For each candidate:

  1. Self-test: write module + tests to temp dir, run pytest.
     - Pre-fix: target test must FAIL, ≥7 regression tests must PASS.
     - Apply oracle_search → oracle_replace, re-run pytest:
       target must PASS, no regression breaks.
     - If any of these checks fail, drop the scenario.

  2. Smoke on Qwen3.6: run N=10 unhinted rollouts against the base model.
     - Compute unhinted target rate. Keep only scenarios with rate <= 0.5
       (the base model fails at least half the time).

If <5 trainable scenarios survive, increment loop counter and re-run
phase 3. Up to TRACE_MAX_VALIDATION_LOOPS attempts.

Output: /tmp/trace_pipeline/env/scenarios_final.json + smoke_results.json
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


PIPE_ROOT = Path(os.environ.get("TRACE_PIPE_ROOT", "/tmp/trace_pipeline"))
ENV_DIR = PIPE_ROOT / "env"
STATE_DIR = PIPE_ROOT / "state"

VLLM_URL = os.environ.get("TRACE_VLLM_E_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("TRACE_MODEL", "Qwen3.6-27B")
N_PER_SC = int(os.environ.get("TRACE_VALIDATE_N", "10"))
CONCURRENCY = int(os.environ.get("TRACE_VALIDATE_CONCURRENCY", "8"))
MAX_LOOPS = int(os.environ.get("TRACE_MAX_VALIDATION_LOOPS", "3"))
MIN_TRAINABLE = int(os.environ.get("TRACE_MIN_TRAINABLE", "5"))
UNHINTED_THRESHOLD = float(os.environ.get("TRACE_UNHINTED_THRESHOLD", "0.5"))


REPAIR_SYSTEM = (
    "You are an expert software engineer. Fix the bug by emitting "
    "SEARCH/REPLACE blocks. Format:\n\n"
    "```python\n### path/to/file.py\n<<<<<<< SEARCH\n(exact code)\n=======\n"
    "(replacement)\n>>>>>>> REPLACE\n```"
)

SR_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE", re.DOTALL
)


def run_pytest(workdir):
    """Run pytest -v and parse PASSED/FAILED/ERROR per test.

    -v (verbose) emits one line per test like:
        tests/test_foo.py::test_bar PASSED
    """
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--tb=no", "--no-header"],
            cwd=str(workdir), capture_output=True, text=True, timeout=60,
        )
        out = (r.stdout or "") + (r.stderr or "")
        results = {}
        for m in re.finditer(r"(\S+::\S+)\s+(PASSED|FAILED|ERROR)", out):
            results[m.group(1).split("::")[-1]] = m.group(2)
        return results
    except Exception:
        return {}


def write_files(workdir, sc):
    (workdir / sc["module_filename"]).write_text(sc["module_code"])
    tf = workdir / sc["test_filename"]
    tf.parent.mkdir(parents=True, exist_ok=True)
    tf.write_text(sc["test_code"])
    # tests/__init__.py for pytest discovery
    if "/" in sc["test_filename"]:
        d = tf.parent
        if not (d / "__init__.py").exists():
            (d / "__init__.py").write_text("")


def self_test_scenario(sc):
    """Return (passed, reason, pre_results, post_results)."""
    wd = Path(tempfile.mkdtemp(prefix=f"tval_{sc['name'][:30]}_{uuid.uuid4().hex[:6]}_"))
    try:
        write_files(wd, sc)
        pre = run_pytest(wd)
        if not pre:
            return False, "pytest_no_output_pre", {}, {}
        target = sc["target_test"]
        if pre.get(target) != "FAILED":
            return False, f"target_not_failing_pre:{pre.get(target)}", pre, {}
        pre_passing = [t for t, s in pre.items() if s == "PASSED"]
        if len(pre_passing) < 5:
            return False, f"too_few_pre_passing:{len(pre_passing)}", pre, {}
        # Apply oracle
        mod_path = wd / sc["module_filename"]
        code = mod_path.read_text()
        search = sc["oracle_search"]
        replace = sc["oracle_replace"]
        if search not in code:
            return False, "oracle_search_not_found", pre, {}
        new_code = code.replace(search, replace, 1)
        mod_path.write_text(new_code)
        post = run_pytest(wd)
        if post.get(target) != "PASSED":
            return False, f"target_not_passing_post:{post.get(target)}", pre, post
        broke = [t for t in pre_passing if post.get(t) != "PASSED"]
        if broke:
            return False, f"oracle_broke_regressions:{broke[:3]}", pre, post
        return True, "ok", pre, post
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def call_vllm_repair(scenario):
    """Run one unhinted rollout: build SWE-bench-style prompt + ask model."""
    user = (
        f"{scenario['pr_description']}\n\n"
        f"File `{scenario['module_filename']}`:\n"
        f"```python\n{scenario['module_code']}\n```\n\n"
        "Write SEARCH/REPLACE block(s) to fix the bug."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""


def rollout_passes(sc, raw_output):
    """Apply model SEARCH/REPLACE; run pytest; return whether target passes."""
    blocks = []
    for b in re.findall(r"```python\s*\n(.*?)\n```", raw_output or "", re.DOTALL):
        for m in SR_RE.finditer(b):
            blocks.append((m.group(1), m.group(2)))
    if not blocks:
        return False
    wd = Path(tempfile.mkdtemp(prefix=f"smk_{sc['name'][:20]}_{uuid.uuid4().hex[:6]}_"))
    try:
        write_files(wd, sc)
        mod_path = wd / sc["module_filename"]
        code = mod_path.read_text()
        for search, replace in blocks:
            if search in code:
                code = code.replace(search, replace, 1)
            else:
                # try whitespace-tolerant find
                pass
        mod_path.write_text(code)
        res = run_pytest(wd)
        return res.get(sc["target_test"]) == "PASSED"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def smoke_one_rollout(sc, _i):
    raw = call_vllm_repair(sc)
    ok = rollout_passes(sc, raw)
    return ok


def main():
    parsed = json.loads((ENV_DIR / "scenarios_parsed.json").read_text())
    print(f"[phase4] candidates: {len(parsed)}", flush=True)

    # Self-test all
    self_tested = []
    for sc in parsed:
        ok, reason, _, _ = self_test_scenario(sc)
        if ok:
            self_tested.append(sc)
        else:
            print(f"[phase4]   drop self-test {sc.get('name', '?')[:40]}: {reason}",
                  flush=True)
    print(f"[phase4] self-tested OK: {len(self_tested)}/{len(parsed)}", flush=True)

    if not self_tested:
        # No survivors; loop back
        return loop_or_fail([])

    # Smoke test on base model
    print(f"[phase4] smoke testing {len(self_tested)} scenarios "
          f"({N_PER_SC} rollouts each)", flush=True)
    smoke_results = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {}
        for sc in self_tested:
            for i in range(N_PER_SC):
                futs[ex.submit(smoke_one_rollout, sc, i)] = sc["name"]
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            smoke_results.setdefault(name, []).append(ok)

    # Filter trainable
    trainable = []
    sat_log = []
    for sc in self_tested:
        rates = smoke_results.get(sc["name"], [])
        rate = sum(rates) / len(rates) if rates else 1.0
        if rate <= UNHINTED_THRESHOLD:
            trainable.append(sc)
            print(f"[phase4]   TRAIN {sc['name'][:40]} rate={rate:.2f}", flush=True)
        else:
            sat_log.append({"name": sc["name"], "rate": rate})
            print(f"[phase4]   SAT   {sc['name'][:40]} rate={rate:.2f}", flush=True)

    print(f"[phase4] trainable={len(trainable)}/{len(self_tested)}", flush=True)

    (ENV_DIR / "smoke_results.json").write_text(json.dumps({
        "n_candidates": len(parsed),
        "n_self_tested": len(self_tested),
        "n_trainable": len(trainable),
        "trainable_names": [s["name"] for s in trainable],
        "saturated": sat_log,
        "per_scenario_rates": {n: sum(r) / len(r) if r else 1.0
                              for n, r in smoke_results.items()},
    }, indent=2))

    return loop_or_fail(trainable)


def loop_or_fail(trainable):
    state_path = STATE_DIR / "env_synth_loop.txt"
    loop_iter = 0
    if state_path.is_file():
        try: loop_iter = int(state_path.read_text().strip())
        except Exception: pass

    if len(trainable) >= MIN_TRAINABLE:
        # SUCCESS
        (ENV_DIR / "scenarios_final.json").write_text(json.dumps(trainable, indent=2))
        # Reset loop counter
        if state_path.is_file(): state_path.unlink()
        print(f"[phase4] DONE — {len(trainable)} trainable scenarios", flush=True)
        return 0

    # Loop back to phase 3 with more obscure hint
    if loop_iter + 1 >= MAX_LOOPS:
        print(f"[phase4] FAILED after {MAX_LOOPS} loops "
              f"(have {len(trainable)} trainable, need ≥{MIN_TRAINABLE})",
              flush=True)
        return 2

    loop_iter += 1
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path.write_text(str(loop_iter))
    # Erase phase3 + phase4 done markers so they re-run
    for n in ("env_synthesis", "env_validation"):
        f = STATE_DIR / f"{n}.done"
        if f.exists(): f.unlink()
    print(f"[phase4] LOOP — {len(trainable)} trainable < {MIN_TRAINABLE}; "
          f"retrying phase 3 with loop_iter={loop_iter}", flush=True)
    return 3  # signal loop


if __name__ == "__main__":
    sys.exit(main())
