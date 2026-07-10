"""Phase 3 v4 — stripped-back synthesis with high-N smoke variance.

Insight from v3: the complex prompt overwhelmed XML production (70% parse
failures), AND all 10 scenarios that did parse were perfectly solved at
rate=1.0 across N=3 rollouts.

v4 changes:
  1. MINIMAL prompt (cut by ~70%) — focus on getting parse success
  2. N_SMOKE=10 (3× more rollouts to surface any solver variance)
  3. ACCEPT_THRESHOLD=0.5 (accept only scenarios the base model fails on at
     least half of N=10 rollouts)
  4. MUTATION-based revision (when saturated): ask model to RESTRUCTURE
     surface code (rename vars, inline/expand helpers, add red-herring
     paths) while KEEPING the same bug + oracle. Mutation breaks the
     "I wrote it so I recognize it" pattern.

Output:
  /workspace/trace_pipeline/env/scenarios_parsed_v4.json
  /workspace/trace_pipeline/env/scenarios_raw_v4.jsonl
  /workspace/trace_pipeline/env/scenarios_revision_log_v4.jsonl
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


PIPE_ROOT = Path(os.environ.get("TRACE_PIPE_ROOT", "/workspace/trace_pipeline"))
ENV_DIR = PIPE_ROOT / "env"
ENV_DIR.mkdir(parents=True, exist_ok=True)

VLLM_URL = os.environ.get("TRACE_VLLM_E_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("TRACE_MODEL", "Qwen/Qwen3.6-27B")
N_CANDIDATES = int(os.environ.get("TRACE_SYNTH_N", "20"))
CONCURRENCY = int(os.environ.get("TRACE_SYNTH_CONCURRENCY", "4"))
MAX_REVISE_SELFTEST = int(os.environ.get("TRACE_MAX_REVISE_SELFTEST", "2"))
MAX_REVISE_MUTATION = int(os.environ.get("TRACE_MAX_REVISE_MUTATION", "3"))
N_SMOKE = int(os.environ.get("TRACE_N_SMOKE", "10"))
ACCEPT_THRESHOLD = float(os.environ.get("TRACE_ACCEPT_THRESHOLD", "0.5"))


DOMAINS = [
    "rocket telemetry", "chemical reactor", "LiDAR dispatcher",
    "PID autotuner", "marine rudder", "HVAC damper", "seismometer gain",
    "B-tree split", "tokenizer state machine", "convex hull",
    "viewport clip", "PRNG seed schedule", "wavelet denoiser",
    "GPS NMEA parser", "MIDI voice allocator", "radar beacon decoder",
    "particle tracker", "smelter draft regulator", "yaw controller",
    "freight loadplan", "thermal solver", "packet classifier",
    "mesh smoother", "gradient clipper",
]


SR_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE", re.DOTALL
)


# ------------------------------------------------------------------
# MINIMAL synthesis prompt (small enough to keep XML clean)
# ------------------------------------------------------------------
def make_synth_system():
    return (
        "You are designing synthetic Python bug-fix scenarios for RL training. "
        "The scenarios must RESEMBLE REAL OPEN-SOURCE LIBRARY CODE in structure, "
        "not toy scripts.\n\n"
        "REQUIRED PYTHON PATTERNS (use 2-3 per scenario):\n"
        "  - Class hierarchies (base class + 2-3 subclasses with method overrides)\n"
        "  - Decorators (@classmethod, @property, @cached_property, custom decorators)\n"
        "  - Type hints throughout\n"
        "  - dataclasses or namedtuples\n"
        "  - Module-level constants + private helpers (_underscore funcs)\n"
        "  - Iterator/generator protocols\n"
        "  - Context managers (__enter__/__exit__)\n"
        "  - Method chaining / fluent APIs\n\n"
        "SCENARIO REQUIREMENTS:\n"
        "  - Clean-room: no django/sympy/sphinx/matplotlib/sklearn/astropy/pytest "
        "API references\n"
        "  - Module length: 150-300 LOC (NOT a toy script)\n"
        "  - One module file + pytest tests (10+ tests)\n"
        "  - target test FAILS pre-fix, regression tests (8+) PASS pre-fix\n"
        "  - oracle SEARCH/REPLACE fixes target without breaking regressions\n"
        "  - Bug location should be PLAUSIBLE in a real library: inside a method "
        "of one of the subclasses, in a property setter, in a helper function "
        "called by multiple methods, etc.\n\n"
        "Output exactly this XML (no fences, no preamble):\n\n"
        "<scenario>\n"
        "<name>kebab-id</name>\n"
        "<module_filename>mod.py</module_filename>\n"
        "<module_code>\n# python (150-300 LOC, library-style)\n</module_code>\n"
        "<test_filename>tests/test_mod.py</test_filename>\n"
        "<test_code>\n# pytest (10+ tests)\n</test_code>\n"
        "<target_test>test_name</target_test>\n"
        "<pr_description>1 para symptom only, no fix hints</pr_description>\n"
        "<oracle_search>\n# verbatim from module_code\n</oracle_search>\n"
        "<oracle_replace>\n# replacement\n</oracle_replace>\n"
        "<naive_fix_description>what naive solver would try</naive_fix_description>\n"
        "<hint_body>one paragraph hint that NAMES the buggy method/class but not the fix</hint_body>\n"
        "</scenario>"
    )


def call_synth(sys_prompt, user_msg, max_tokens=6000, temperature=0.8):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ------------------------------------------------------------------
# XML parsing
# ------------------------------------------------------------------
TAG_FIELDS = ["name", "module_filename", "module_code",
              "test_filename", "test_code", "target_test",
              "pr_description", "oracle_search", "oracle_replace",
              "naive_fix_description"]
# hint_body is OPTIONAL (older scenarios don't have it)
OPTIONAL_TAGS = ["hint_body"]
INLINE = {"name", "module_filename", "test_filename", "target_test",
          "pr_description", "naive_fix_description", "hint_body"}


def extract_tag(raw, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", raw or "", re.DOTALL)
    if not m: return None
    v = m.group(1)
    if tag in INLINE: return v.strip()
    v = v.strip("\n")
    v = re.sub(r"^```(?:python)?\s*\n", "", v)
    v = re.sub(r"\n```\s*$", "", v)
    return v


def parse_scenario(raw):
    scn = re.search(r"<scenario>(.*?)</scenario>", raw or "", re.DOTALL)
    body = scn.group(1) if scn else raw
    obj = {}
    for f in TAG_FIELDS:
        v = extract_tag(body, f)
        if v is None or (isinstance(v, str) and not v.strip()):
            return None, f"missing_tag:{f}"
        obj[f] = v
    for f in OPTIONAL_TAGS:
        v = extract_tag(body, f)
        if v is not None and (not isinstance(v, str) or v.strip()):
            obj[f] = v
    return obj, "ok"


# ------------------------------------------------------------------
# Pytest helpers
# ------------------------------------------------------------------
def run_pytest(workdir):
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
    if "/" in sc["test_filename"]:
        if not (tf.parent / "__init__.py").exists():
            (tf.parent / "__init__.py").write_text("")


def self_test(sc):
    wd = Path(tempfile.mkdtemp(prefix=f"st_{uuid.uuid4().hex[:6]}_"))
    try:
        write_files(wd, sc)
        pre = run_pytest(wd)
        if not pre:
            return False, "pytest_no_output_pre"
        target = sc["target_test"]
        if pre.get(target) != "FAILED":
            return False, f"target_not_failing_pre:{pre.get(target)}"
        passing = [t for t, s in pre.items() if s == "PASSED"]
        if len(passing) < 5:
            return False, f"too_few_pre_passing:{len(passing)}"
        mp = wd / sc["module_filename"]
        code = mp.read_text()
        if sc["oracle_search"] not in code:
            return False, "oracle_search_not_found"
        new_code = code.replace(sc["oracle_search"], sc["oracle_replace"], 1)
        mp.write_text(new_code)
        post = run_pytest(wd)
        if post.get(target) != "PASSED":
            return False, f"target_not_passing_post:{post.get(target)}"
        broke = [t for t in passing if post.get(t) != "PASSED"]
        if broke:
            return False, f"oracle_broke_regressions:{broke[:3]}"
        return True, "ok"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ------------------------------------------------------------------
# Smoke (rollouts)
# ------------------------------------------------------------------
REPAIR_SYS = (
    "You are an expert software engineer. Fix the bug by emitting SEARCH/REPLACE "
    "blocks. Format:\n\n"
    "```python\n### path/to/file.py\n<<<<<<< SEARCH\n(exact code)\n=======\n"
    "(replacement)\n>>>>>>> REPLACE\n```"
)


def rollout(sc):
    user = (
        f"{sc['pr_description']}\n\n"
        f"File `{sc['module_filename']}`:\n"
        f"```python\n{sc['module_code']}\n```\n\n"
        "Write SEARCH/REPLACE block(s) to fix the bug."
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": REPAIR_SYS},
                     {"role": "user", "content": user}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""


def rollout_passes(sc, raw):
    blocks = []
    for b in re.findall(r"```python\s*\n(.*?)\n```", raw or "", re.DOTALL):
        for m in SR_RE.finditer(b):
            blocks.append((m.group(1), m.group(2)))
    if not blocks: return False
    wd = Path(tempfile.mkdtemp(prefix=f"sm_{uuid.uuid4().hex[:6]}_"))
    try:
        write_files(wd, sc)
        mp = wd / sc["module_filename"]
        code = mp.read_text()
        for search, replace in blocks:
            if search in code:
                code = code.replace(search, replace, 1)
        mp.write_text(code)
        res = run_pytest(wd)
        return res.get(sc["target_test"]) == "PASSED"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def smoke(sc, n=N_SMOKE):
    success_text = None
    successes = 0
    rollouts = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(rollout, sc) for _ in range(n)]
        for f in as_completed(futs):
            try: raw = f.result()
            except Exception: raw = ""
            rollouts.append(raw)
    for raw in rollouts:
        if rollout_passes(sc, raw):
            successes += 1
            if success_text is None: success_text = raw
    return successes / n, success_text, rollouts


# ------------------------------------------------------------------
# Self-test revision (small, focused)
# ------------------------------------------------------------------
def revise_for_selftest(sc, reason):
    sys_prompt = make_synth_system()
    user = (
        f"Previous scenario FAILED self-test: {reason}\n\n"
        f"<scenario>\n"
        f"<name>{sc['name']}</name>\n"
        f"<module_filename>{sc['module_filename']}</module_filename>\n"
        f"<module_code>\n{sc['module_code']}\n</module_code>\n"
        f"<test_filename>{sc['test_filename']}</test_filename>\n"
        f"<test_code>\n{sc['test_code']}\n</test_code>\n"
        f"<target_test>{sc['target_test']}</target_test>\n"
        f"<pr_description>{sc['pr_description']}</pr_description>\n"
        f"<oracle_search>\n{sc['oracle_search']}\n</oracle_search>\n"
        f"<oracle_replace>\n{sc['oracle_replace']}\n</oracle_replace>\n"
        f"<naive_fix_description>{sc['naive_fix_description']}</naive_fix_description>\n"
        f"</scenario>\n\n"
        "Fix the issue. Output only a corrected <scenario>...</scenario>."
    )
    return call_synth(sys_prompt, user)


# ------------------------------------------------------------------
# MUTATION-based difficulty revision
# ------------------------------------------------------------------
MUTATION_STRATEGIES = [
    "Rename ALL identifiers in module_code to non-domain names (e.g., temperature → t_var_3, "
    "process_signal → fn_a). KEEP test_code identical. UPDATE oracle_search/oracle_replace "
    "to match the new names.",

    "Inline-expand any short helper functions in module_code into their callers (so the bug "
    "lives in a 20-30 line function instead of a 5-line one). KEEP tests untouched. UPDATE "
    "oracle_search/oracle_replace to match new inlined location.",

    "Add 3-4 DECOY edit sites elsewhere in module_code: nearby lines that LOOK like they might "
    "need the same fix (similar pattern) but actually don't. Variable names should be confusable. "
    "KEEP tests + target untouched. UPDATE oracle_search/oracle_replace if module_code shifted.",

    "Insert a long, detailed but MISLEADING docstring at the top of module_code suggesting "
    "the fix is in a DIFFERENT (unrelated) function. KEEP tests + bug location. UPDATE oracle "
    "if module_code positions changed.",
]


def revise_for_mutation(sc, success_rollout, attempt):
    """Apply a mutation strategy. KEEP the bug, change the surface code."""
    strategy = MUTATION_STRATEGIES[attempt % len(MUTATION_STRATEGIES)]
    sys_prompt = make_synth_system()
    user = (
        f"The previous scenario was SOLVED by the model. Here's its successful "
        f"patch:\n\n```\n{success_rollout[:1200]}\n```\n\n"
        f"Previous scenario:\n<scenario>\n"
        f"<name>{sc['name']}</name>\n"
        f"<module_filename>{sc['module_filename']}</module_filename>\n"
        f"<module_code>\n{sc['module_code']}\n</module_code>\n"
        f"<test_filename>{sc['test_filename']}</test_filename>\n"
        f"<test_code>\n{sc['test_code']}\n</test_code>\n"
        f"<target_test>{sc['target_test']}</target_test>\n"
        f"<pr_description>{sc['pr_description']}</pr_description>\n"
        f"<oracle_search>\n{sc['oracle_search']}\n</oracle_search>\n"
        f"<oracle_replace>\n{sc['oracle_replace']}\n</oracle_replace>\n"
        f"</scenario>\n\n"
        f"MUTATION: {strategy}\n\n"
        "IMPORTANT: the bug + test behavior must be PRESERVED. Only the surface "
        "of module_code changes. Update oracle to match the mutated code.\n\n"
        "Output only a corrected <scenario>...</scenario>."
    )
    return call_synth(sys_prompt, user, temperature=0.9)


# ------------------------------------------------------------------
# Per-slot loop
# ------------------------------------------------------------------
def process_one(domain, raw_log_f, rev_log_f):
    log = {"domain": domain, "events": []}
    sys_prompt = make_synth_system()
    user = (
        f"Domain: {domain}\n\n"
        "Write ONE scenario with a subtle bug. Begin with <scenario> and end "
        "with </scenario>."
    )
    try:
        raw = call_synth(sys_prompt, user)
    except Exception as e:
        log["events"].append({"stage": "generate", "ok": False, "err": str(e)[:200]})
        rev_log_f.write(json.dumps(log) + "\n"); rev_log_f.flush()
        return None
    raw_log_f.write(json.dumps({"domain": domain, "raw": raw}) + "\n")
    raw_log_f.flush()
    sc, why = parse_scenario(raw)
    if sc is None:
        log["events"].append({"stage": "parse", "ok": False, "reason": why})
        rev_log_f.write(json.dumps(log) + "\n"); rev_log_f.flush()
        return None
    log["events"].append({"stage": "parse", "ok": True, "name": sc["name"]})

    # self-test loop
    for a in range(MAX_REVISE_SELFTEST + 1):
        ok, reason = self_test(sc)
        log["events"].append({"stage": "self_test", "attempt": a, "ok": ok,
                              "reason": reason})
        if ok: break
        if a >= MAX_REVISE_SELFTEST:
            rev_log_f.write(json.dumps(log) + "\n"); rev_log_f.flush()
            return None
        try:
            new_raw = revise_for_selftest(sc, reason)
        except Exception:
            rev_log_f.write(json.dumps(log) + "\n"); rev_log_f.flush()
            return None
        new_sc, why = parse_scenario(new_raw)
        if new_sc is None:
            log["events"].append({"stage": "revise_selftest_parse", "ok": False,
                                  "reason": why})
            rev_log_f.write(json.dumps(log) + "\n"); rev_log_f.flush()
            return None
        sc = new_sc

    # smoke + mutation loop
    for a in range(MAX_REVISE_MUTATION + 1):
        rate, success_text, _ = smoke(sc)
        log["events"].append({"stage": "smoke", "attempt": a, "rate": rate})
        if rate <= ACCEPT_THRESHOLD:
            log["accepted"] = True
            log["final_rate"] = rate
            rev_log_f.write(json.dumps(log) + "\n"); rev_log_f.flush()
            return sc
        if a >= MAX_REVISE_MUTATION or success_text is None:
            log["accepted"] = False
            log["final_rate"] = rate
            rev_log_f.write(json.dumps(log) + "\n"); rev_log_f.flush()
            return None
        try:
            new_raw = revise_for_mutation(sc, success_text, a)
        except Exception:
            rev_log_f.write(json.dumps(log) + "\n"); rev_log_f.flush()
            return None
        new_sc, why = parse_scenario(new_raw)
        if new_sc is None:
            log["events"].append({"stage": "revise_mutation_parse", "ok": False,
                                  "reason": why})
            continue
        # mutated may need self-test re-run
        ok, reason = self_test(new_sc)
        if not ok:
            log["events"].append({"stage": "revise_mutation_selftest",
                                  "ok": False, "reason": reason})
            continue
        sc = new_sc
    return None


def main():
    t0 = time.time()
    print(f"[phase3v4] N={N_CANDIDATES} concurrency={CONCURRENCY} "
          f"smoke={N_SMOKE} threshold<={ACCEPT_THRESHOLD}", flush=True)

    raw_log = open(ENV_DIR / "scenarios_raw_v4.jsonl", "w")
    rev_log = open(ENV_DIR / "scenarios_revision_log_v4.jsonl", "w")

    accepted = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(process_one, DOMAINS[i % len(DOMAINS)],
                          raw_log, rev_log)
                for i in range(N_CANDIDATES)]
        n_done = 0
        for fut in as_completed(futs):
            try: r = fut.result()
            except Exception: r = None
            if r is not None: accepted.append(r)
            n_done += 1
            elapsed = time.time() - t0
            print(f"[phase3v4] {n_done}/{N_CANDIDATES} done, "
                  f"{len(accepted)} accepted, elapsed={elapsed:.0f}s", flush=True)

    raw_log.close()
    rev_log.close()
    (ENV_DIR / "scenarios_parsed_v4.json").write_text(
        json.dumps(accepted, indent=2)
    )
    print(f"[phase3v4] DONE in {time.time() - t0:.0f}s — "
          f"{len(accepted)}/{N_CANDIDATES} accepted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
