"""Phase 3 v6 — synthetic environments for `deep-call-graph-traversal`.

Targets the SECOND-largest discovered weakness from phase 2 (16.3% of Qwen3.6
SWE-bench Verified unresolved failures):

    > Model patches the SYMPTOM where the test fails, instead of tracing
    > through 3-5 call hops to find the actual root cause. Multiple callers
    > of the buggy helper → fixing surface only doesn't unblock the test.

Key design from prior iters:
  - Per Qwen-team practice: clean-room SYNTHETIC domains, no library leakage
  - module length 300-600 LOC (matches typical SWE-bench file)
  - rate <= 0.5 (model fails at least half of N=10 rollouts at temp=1.0)
  - hybrid anchor parser (tolerant to malformed XML at long output)

What makes this CAPABILITY different from semantic-logic-precision:
  - The bug is NOT the line that the test exercises directly.
  - The bug lives 3-5 call hops DOWN from the test-touched function.
  - At the symptom site, the surface code LOOKS correct.
  - The naive "patch the symptom" fix produces something that LOOKS
    plausible but doesn't make the test pass (because the bug propagates
    upward through multiple callers).
  - Tests catch the real bug only when the root cause is fixed.

Scenario shape:
  - Module exposes a high-level API: `compute_summary(...)` (the "symptom" site).
  - compute_summary calls _aggregate() which calls _accumulate() which calls
    _normalize() — the bug is in _normalize().
  - 4-6 OTHER public methods also indirectly use _normalize() — so the bug
    has visible downstream effects (e.g., test_aggregate_consistency also
    fails because _aggregate now disagrees with _accumulate).
  - target test is one of these downstream checks; fixing only
    compute_summary leaves the cross-method invariant broken.

Output:
  scenarios_parsed_v6.json
  scenarios_raw_v6.jsonl
  scenarios_revision_log_v6.jsonl
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
MAX_REVISE_MUTATION = int(os.environ.get("TRACE_MAX_REVISE_MUTATION", "5"))
N_SMOKE = int(os.environ.get("TRACE_N_SMOKE", "10"))
ACCEPT_THRESHOLD = float(os.environ.get("TRACE_ACCEPT_THRESHOLD", "0.5"))
TARGET_ACCEPT = int(os.environ.get("TRACE_TARGET_ACCEPT", "10"))


# Synthetic domains, no library leakage
DOMAINS = [
    "telemetry budget allocator", "geofence violation tracker",
    "battery thermal envelope", "vibration spectrum classifier",
    "magnetic-field declination corrector", "centrifuge balance scheduler",
    "elevator group dispatcher", "spaceframe load redistributor",
    "tide-gauge anomaly window", "fuel cell purge controller",
    "drone swarm collision avoider", "weather station calibration",
    "gantry crane sway damper", "hydroponics nutrient ladder",
    "satellite pass scheduler", "wind shear gust predictor",
    "RFID inventory drift", "warehouse robot path budget",
    "subway berth alignment", "tunnel ventilation regulator",
]


SR_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE", re.DOTALL
)


# ------------------------------------------------------------------
# Synthesis prompt (targets the deep-call-graph capability)
# ------------------------------------------------------------------
def make_synth_system():
    return (
        "You are designing synthetic Python bug-fix scenarios for RL training. "
        "Target capability: DEEP CALL GRAPH TRAVERSAL.\n\n"
        "CAPABILITY DEFINITION:\n"
        "  The model under training patches the SYMPTOM at the surface where "
        "the test fails, instead of tracing 3-5 call hops to find the actual "
        "root cause. Multiple callers of the buggy helper → fixing the surface "
        "only doesn't unblock the test.\n\n"
        "REQUIRED SCENARIO STRUCTURE:\n"
        "  - Module exposes a HIGH-LEVEL public API (e.g., `compute_xxx`, "
        "`run_xxx`, `process_xxx`).\n"
        "  - The high-level API calls a MIDDLE-LAYER helper which calls a "
        "LOW-LEVEL helper (3+ levels of indirection).\n"
        "  - The actual BUG is in the LOW-LEVEL helper.\n"
        "  - 3-5 OTHER public methods also call the same low-level helper "
        "(directly or transitively).\n"
        "  - The target test exercises a DIFFERENT public method than the "
        "naive-fix would touch. Patching the top-level method does NOT "
        "make the target test pass (the bug still propagates).\n"
        "  - oracle_search/oracle_replace must target the LOW-LEVEL function.\n"
        "  - naive_fix_description: describes how a naive solver would patch "
        "the surface (the wrong fix).\n\n"
        "STRICT RULES:\n"
        "  - HARD NO-LEAKAGE: NO references to django, sympy, sphinx, "
        "matplotlib, sklearn, astropy, scipy, pandas, numpy public APIs, "
        "requests, flask, fastapi, twisted, pytest fixtures, pylint, mypy. "
        "Synthetic engineering domain only. Stdlib-only imports.\n"
        "  - Module length: 300-600 LOC (with at least 4-6 public methods and "
        "3 layers of helper indirection).\n"
        "  - One module file + pytest tests (10+ tests).\n"
        "  - Target test FAILS pre-fix, 8+ regression tests PASS pre-fix.\n"
        "  - Oracle SEARCH/REPLACE targets the LOW-LEVEL bug — NOT the "
        "high-level symptom.\n"
        "  - EMIT EVERY OPEN AND CLOSE TAG. Never write </tag> without <tag>.\n\n"
        "Output exactly this XML (no fences, no preamble):\n\n"
        "<scenario>\n"
        "<name>kebab-id</name>\n"
        "<module_filename>mod.py</module_filename>\n"
        "<module_code>\n"
        "# python (300-600 LOC, synthetic-domain, 3-layer call graph,\n"
        "# bug in deepest helper, multiple public callers)\n"
        "</module_code>\n"
        "<test_filename>tests/test_mod.py</test_filename>\n"
        "<test_code>\n# pytest (10+ tests; target test exercises a DIFFERENT\n"
        "# public method than where the naive solver would patch)\n</test_code>\n"
        "<target_test>test_name</target_test>\n"
        "<pr_description>1 para symptom only; describes the symptom at the\n"
        "TOP level so the naive solver looks at the wrong place; no fix hint\n"
        "</pr_description>\n"
        "<oracle_search>\n# verbatim from module_code — the LOW-LEVEL helper\n"
        "</oracle_search>\n"
        "<oracle_replace>\n# replacement for low-level helper\n</oracle_replace>\n"
        "<naive_fix_description>describes the WRONG fix a shallow solver\n"
        "would attempt at the symptom site (top-level public method)\n"
        "</naive_fix_description>\n"
        "<hint_body>one paragraph hint that NAMES the low-level helper but\n"
        "not the fix\n</hint_body>\n"
        "</scenario>"
    )


def call_synth(sys_prompt, user_msg, max_tokens=12000, temperature=0.85):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=1200)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ------------------------------------------------------------------
# Hybrid anchor parser (handles missing open/close tags in long outputs)
# ------------------------------------------------------------------
TAG_FIELDS = ["name", "module_filename", "module_code",
              "test_filename", "test_code", "target_test",
              "pr_description", "oracle_search", "oracle_replace",
              "naive_fix_description"]
INLINE = {"name", "module_filename", "test_filename", "target_test",
          "pr_description", "naive_fix_description", "hint_body"}


def parse_scenario(raw):
    raw = raw or ""
    scn = re.search(r"<scenario>(.*?)(?:</scenario>|$)", raw, re.DOTALL)
    body = scn.group(1) if scn else raw
    ordered = TAG_FIELDS + ["hint_body"]
    obj = {}
    cursor = 0
    for i, f in enumerate(ordered):
        close_m = re.search(rf"</{f}>", body[cursor:])
        close_pos = (cursor + close_m.start()) if close_m else None
        next_open_pos = None
        for g in ordered[i+1:]:
            mm = re.search(rf"<{g}>", body[cursor:])
            if mm:
                next_open_pos = cursor + mm.start()
                break
        if close_pos is None and next_open_pos is None:
            continue
        end = close_pos if close_pos is not None else next_open_pos
        if next_open_pos is not None and next_open_pos < end:
            end = next_open_pos
        open_m = re.search(rf"<{f}>", body[cursor:end])
        start = cursor + open_m.end() if open_m else cursor
        if start >= end:
            continue
        value = body[start:end]
        if f in INLINE:
            value = value.strip()
        else:
            value = value.strip("\n")
            value = re.sub(r"^```(?:python)?\s*\n", "", value)
            value = re.sub(r"\n```\s*$", "", value)
        obj[f] = value
        cursor = end + (len(f) + 3 if close_pos == end else 0)
    for f in TAG_FIELDS:
        if not obj.get(f) or not str(obj[f]).strip():
            return None, f"missing_tag:{f}"
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
    LEAK_TERMS = ['django', 'sympy', 'sphinx', 'matplotlib', 'sklearn', 'astropy',
                  'scipy', 'pandas', 'flask', 'fastapi', 'pylint', 'mypy',
                  'twisted', 'Django', 'SymPy', 'NumPy', 'PyTorch', 'TensorFlow']
    full_text = sc.get('module_code', '') + sc.get('test_code', '') + sc.get('pr_description', '')
    for t in LEAK_TERMS:
        if t in full_text:
            return False, f'leakage:contains {t!r}'

    wd = Path(tempfile.mkdtemp(prefix=f"st_{uuid.uuid4().hex[:6]}_"))
    try:
        write_files(wd, sc)
        pre = run_pytest(wd)
        if not pre:
            return False, "no_test_results_pre"
        target = sc["target_test"]
        if pre.get(target) != "FAILED":
            return False, f"target_not_failing_pre:{pre.get(target)}"
        passing = [t for t, s in pre.items() if s == "PASSED"]
        # Apply oracle
        mp = wd / sc["module_filename"]
        code = mp.read_text()
        if sc["oracle_search"] not in code:
            return False, "oracle_search_not_in_module"
        code = code.replace(sc["oracle_search"], sc["oracle_replace"], 1)
        mp.write_text(code)
        post = run_pytest(wd)
        if post.get(target) != "PASSED":
            return False, f"target_not_passing_post:{post.get(target)}"
        broke = [t for t in passing if post.get(t) != "PASSED"]
        if broke:
            return False, f"broken_tests:{broke[:3]}"
        return True, "ok"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


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
    if not blocks:
        return False
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
            try:
                raw = f.result()
            except Exception:
                raw = ""
            rollouts.append(raw)
    for raw in rollouts:
        if rollout_passes(sc, raw):
            successes += 1
            if success_text is None:
                success_text = raw
    return successes / n, success_text, rollouts


# ------------------------------------------------------------------
# Mutation strategies (tuned for deep-call-graph capability)
# ------------------------------------------------------------------
MUTATION_STRATEGIES = [
    ("Apply ALL of these mutations: "
     "(a) ADD 2 more public methods that use the buggy low-level helper. "
     "(b) ADD a misleading comment on the buggy line saying 'INVARIANT: this is correct'. "
     "(c) Inline-expand the MIDDLE-LAYER helper into its callers (extends the call chain "
     "by 1 hop and makes the bug harder to localize). "
     "(d) Rename the low-level helper to something generic (`_fn_a`). "
     "Keep tests + oracle semantics. Update oracle_search/oracle_replace."),

    ("Apply ALL of these mutations: "
     "(a) Add a DECOY low-level helper with a similar name (e.g., `_normalize` vs "
     "`_normalize_value`) where the decoy is correct and called from a different path. "
     "(b) Insert a long misleading docstring on the high-level public method "
     "pointing to a 'known issue with input parsing' as the cause. "
     "(c) Add 4 IDENTICAL-LOOKING lines elsewhere that LOOK like the buggy line "
     "but are correct. "
     "(d) Rename the buggy variable to a name shared with an unrelated parameter. "
     "PRESERVE bug + tests. Update oracle."),

    ("Apply ALL of these mutations: "
     "(a) WRAP the call chain in a class hierarchy: BaseClass (high-level), "
     "Mixin (middle), and a subclass that overrides the low-level helper "
     "(where the bug lives). Use multiple inheritance / MRO to make it harder. "
     "(b) Add `@cached_property` on the high-level method (changes apparent control flow). "
     "(c) Add 2 sibling subclasses with their own (correct) low-level helpers — they SHOULDN'T be touched. "
     "(d) Update pr_description to suggest the bug is in the CACHING layer (misdirection). "
     "PRESERVE bug + tests. Update oracle."),
]


# ------------------------------------------------------------------
# Loops
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


def revise_for_mutation(sc, success_rollout, attempt):
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
        "Output only a corrected <scenario>...</scenario>."
    )
    return call_synth(sys_prompt, user, temperature=0.9)


def process_one(domain, raw_log_f, rev_log_f):
    log = {"domain": domain, "events": []}
    sys_prompt = make_synth_system()
    user = (
        f"Domain: {domain}\n\n"
        "Write ONE deep-call-graph-traversal scenario per spec. Begin with "
        "<scenario> and end with </scenario>."
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
        if ok:
            break
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
        ok, reason = self_test(new_sc)
        if not ok:
            log["events"].append({"stage": "revise_mutation_selftest",
                                  "ok": False, "reason": reason})
            continue
        sc = new_sc
    return None


def main():
    t0 = time.time()
    print(f"[phase3v6] N={N_CANDIDATES} concurrency={CONCURRENCY} "
          f"smoke={N_SMOKE} threshold<={ACCEPT_THRESHOLD} target_accept={TARGET_ACCEPT}",
          flush=True)

    raw_log = open(ENV_DIR / "scenarios_raw_v6.jsonl", "w")
    rev_log = open(ENV_DIR / "scenarios_revision_log_v6.jsonl", "w")

    accepted = []
    domain_iter = iter(DOMAINS * 5)  # cycle if N > len(DOMAINS)
    futures_in_flight = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        # Prime the pool
        for _ in range(min(CONCURRENCY, N_CANDIDATES)):
            domain = next(domain_iter)
            futures_in_flight.append(ex.submit(process_one, domain, raw_log, rev_log))

        n_submitted = len(futures_in_flight)
        n_done = 0

        while futures_in_flight and len(accepted) < TARGET_ACCEPT:
            # Wait for at least one to finish
            done_now = []
            for fut in list(futures_in_flight):
                if fut.done():
                    done_now.append(fut)
            if not done_now:
                # short sleep
                time.sleep(2)
                continue
            for fut in done_now:
                futures_in_flight.remove(fut)
                n_done += 1
                try:
                    r = fut.result()
                except Exception:
                    r = None
                if r is not None:
                    accepted.append(r)
                elapsed = time.time() - t0
                print(f"[phase3v6] {n_done} done, {len(accepted)} accepted/{TARGET_ACCEPT} target, "
                      f"in-flight={len(futures_in_flight)}, elapsed={elapsed:.0f}s",
                      flush=True)
                if len(accepted) >= TARGET_ACCEPT:
                    break
                if n_submitted < N_CANDIDATES * 3:  # cap total attempts
                    try:
                        domain = next(domain_iter)
                    except StopIteration:
                        continue
                    futures_in_flight.append(ex.submit(process_one, domain, raw_log, rev_log))
                    n_submitted += 1

        # Cancel any remaining once we've hit target
        for fut in futures_in_flight:
            fut.cancel()

    raw_log.close()
    rev_log.close()
    (ENV_DIR / "scenarios_parsed_v6.json").write_text(
        json.dumps(accepted, indent=2)
    )
    print(f"[phase3v6] DONE in {time.time() - t0:.0f}s — "
          f"{len(accepted)}/{TARGET_ACCEPT} accepted, "
          f"{n_submitted} total attempted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
