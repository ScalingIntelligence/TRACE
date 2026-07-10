"""Phase 2 v2 — open-ended capability mining (no pre-defined buckets).

Three-stage pipeline:

  Stage 1 — Root-cause analysis. For each unresolved failure, ask
  Qwen3.6 to write 1-2 sentences describing the SPECIFIC capability /
  skill the model was missing. No constraints on label format.

  Stage 2 — Cluster discovery. Show Qwen3.6 ALL the root-cause analyses
  at once. Ask it to identify 4-6 distinct capability categories. Output
  a JSON manifest with category names + descriptions + signature traits.

  Stage 3 — Re-classification. For each failure, ask Qwen3.6 to bucket
  it into one of the DISCOVERED categories (using `guided_choice` over
  the discovered names). Pick the dominant category as the training target.

Inputs:
  /workspace/trace_pipeline/runs/baseline/preds.json
  /workspace/trace_pipeline/runs/baseline/*.<run_id>.json (grading)
Outputs:
  /workspace/trace_pipeline/analysis/capability_target_v2.json
  /workspace/trace_pipeline/analysis/root_causes.jsonl  (stage 1 raw)
  /workspace/trace_pipeline/analysis/discovered_categories.json  (stage 2)
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset


PIPE_ROOT = Path(os.environ.get("TRACE_PIPE_ROOT", "/workspace/trace_pipeline"))
ANALYSIS_DIR = PIPE_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

VLLM_URL = os.environ.get("TRACE_VLLM_E_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("TRACE_MODEL", "Qwen/Qwen3.6-27B")
RUN_ID = os.environ.get("TRACE_PHASE1_RUN_ID", "trace_baseline_qwen36_v2")
CONCURRENCY = int(os.environ.get("TRACE_MINE_CONCURRENCY", "20"))


# ------------------------------------------------------------------
# Stage 1 — Free-form root-cause analysis
# ------------------------------------------------------------------
ROOTCAUSE_SYSTEM = (
    "You are an expert software engineering analyst. You analyze why a "
    "code-fix model failed by comparing its attempt to the gold (correct) patch.\n\n"
    "Write EXACTLY 1-2 short sentences identifying the SPECIFIC capability "
    "or skill the model was missing. Avoid generic phrases like 'made a mistake' "
    "or 'misunderstood the problem'. Be CONCRETE: name the operation, the "
    "structural detail, or the reasoning step that was missed.\n\n"
    "Examples of good answers:\n"
    "- 'Failed to walk into the helper function `_compute_offset` to apply the fix; "
    "edited the public API instead.'\n"
    "- 'Right branch identified but used `+1` where the gold patch uses `-1` "
    "(sign-direction reasoning).'\n"
    "- 'Did not realize fix needs to be at class-attribute scope, not inside "
    "the method body.'\n\n"
    "Output ONLY the 1-2 sentences. No JSON, no headers, no preamble."
)


def call_rootcause(problem, model_patch, gold_patch):
    user = (
        f"## Bug report\n{problem.strip()[:2000]}\n\n"
        f"## Model's attempted patch\n```diff\n{(model_patch or '').strip()[:2500]}\n```\n\n"
        f"## Gold patch (correct fix)\n```diff\n{(gold_patch or '').strip()[:2500]}\n```\n\n"
        "What specific capability did the model lack? (1-2 sentences)"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": ROOTCAUSE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"<ERROR: {type(e).__name__}: {str(e)[:100]}>"


# ------------------------------------------------------------------
# Stage 2 — Cluster discovery from all root causes
# ------------------------------------------------------------------
CLUSTER_SYSTEM = (
    "You are an expert at clustering software engineering failure patterns. "
    "You will receive a list of root-cause analyses describing why a code-fix "
    "model failed on different bugs. Your job is to identify 4-6 DISTINCT "
    "capability categories that emerge from the data.\n\n"
    "Each category must:\n"
    "  - Have a short name (2-5 words, kebab-case)\n"
    "  - Have a one-sentence description of the missing capability\n"
    "  - Have 2-3 signature phrases that would distinguish it from other categories\n"
    "  - Cover at least 5% of the failures\n\n"
    "Aim for MUTUALLY EXCLUSIVE categories at the level of underlying mechanism, "
    "not surface symptoms. Prefer 'deep-call-graph-traversal' over 'wrong function'.\n\n"
    "Output STRICT JSON only (no markdown, no commentary):\n"
    '{"categories": [{"name": "...", "description": "...", "signatures": ["...", "...", "..."]}, ...]}'
)


def call_cluster(root_causes):
    listed = "\n".join(f"{i+1}. {rc}" for i, rc in enumerate(root_causes))
    user = (
        f"Below are {len(root_causes)} root-cause analyses of failed bug-fix "
        f"attempts. Identify 4-6 distinct capability categories.\n\n"
        f"{listed}\n\n"
        "Output JSON with `categories` field as instructed. Start your output "
        "with `{` immediately — no preamble."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CLUSTER_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.5,
        "max_tokens": 16384,
        # NOTE: thinking off — we want JSON directly, not <think>...</think>
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    if not raw:
        raise ValueError("model returned empty content")
    raw = raw.strip()
    # Strip fences if any
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    # Extract first {...} block in case there's prose
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    return json.loads(raw)


def main_from_stage2():
    """Resume from stage 2 — assumes root_causes.jsonl is already there."""
    rc_path = ANALYSIS_DIR / "root_causes.jsonl"
    if not rc_path.is_file():
        print(f"[phase2v2] no root_causes.jsonl — run from scratch", flush=True)
        return None
    root_causes = {}
    for line in open(rc_path):
        d = json.loads(line)
        root_causes[d["instance_id"]] = d["root_cause"]
    return root_causes


# ------------------------------------------------------------------
# Stage 3 — Re-classify each failure into discovered categories
# ------------------------------------------------------------------
def make_reclassify_system(categories):
    cats_md = "\n".join(
        f"- **{c['name']}** — {c['description']}" for c in categories
    )
    return (
        "Classify the failure into ONE of these capability categories:\n\n"
        f"{cats_md}\n\n"
        "Output ONLY the category name (exactly as written above)."
    )


def call_reclassify(rootcause, categories):
    names = [c["name"] for c in categories]
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": make_reclassify_system(categories)},
            {"role": "user", "content": f"Root cause: {rootcause}\n\nCategory name:"},
        ],
        "temperature": 0.0, "max_tokens": 50,
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"guided_choice": names},
    }
    try:
        r = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return names[-1]  # fallback


def main():
    t0 = time.time()
    print(f"[phase2v2] model={MODEL}", flush=True)

    # Resume from stage 2 if root_causes.jsonl already exists
    cached = main_from_stage2()
    if cached:
        print(f"[phase2v2] resuming from stage 2 — {len(cached)} root causes "
              f"loaded from disk", flush=True)
        root_causes = cached
        goto_stage_2(root_causes, t0)
        return 0

    # ----- load grading + preds + dataset ----------------------
    run_dir = PIPE_ROOT / "runs" / "baseline"
    grading_path = next(iter(
        list(run_dir.glob(f"*.{RUN_ID}.json"))
        + list((run_dir / "modal-reports-v2").glob(f"*.{RUN_ID}.json"))
    ), None)
    if grading_path is None:
        print(f"[phase2v2] no grading report for run_id={RUN_ID}", flush=True)
        return 1
    grading = json.loads(grading_path.read_text())
    unresolved = grading.get("unresolved_ids", [])
    print(f"[phase2v2] {len(unresolved)} unresolved failures", flush=True)

    preds = json.loads((run_dir / "preds.json").read_text())
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    gold_patch = {d["instance_id"]: d["patch"] for d in ds}
    problem = {d["instance_id"]: d["problem_statement"] for d in ds}

    failures = [
        {"instance_id": iid,
         "model_patch": preds.get(iid, {}).get("model_patch", ""),
         "problem": problem.get(iid, ""),
         "gold_patch": gold_patch.get(iid, "")}
        for iid in unresolved
        if iid in preds and preds[iid].get("model_patch", "").strip()
    ]
    print(f"[phase2v2] {len(failures)} failures with nonempty model patches",
          flush=True)
    if not failures:
        return 1

    # ----- Stage 1: root-cause analysis ---------------------------
    print(f"[phase2v2] Stage 1: root-cause analysis (concurrency={CONCURRENCY})…",
          flush=True)
    root_causes = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {
            ex.submit(call_rootcause, f["problem"], f["model_patch"], f["gold_patch"]):
                f["instance_id"]
            for f in failures
        }
        n_done = 0
        for fut in as_completed(futs):
            iid = futs[fut]
            try: rc = fut.result()
            except Exception as e: rc = f"<ERROR: {e}>"
            root_causes[iid] = rc
            n_done += 1
            if n_done % 10 == 0:
                print(f"[phase2v2]   {n_done}/{len(failures)}", flush=True)

    # Save raw root causes
    rc_path = ANALYSIS_DIR / "root_causes.jsonl"
    with open(rc_path, "w") as f:
        for iid, rc in root_causes.items():
            f.write(json.dumps({"instance_id": iid, "root_cause": rc}) + "\n")
    print(f"[phase2v2]   wrote {rc_path}", flush=True)

    goto_stage_2(root_causes, t0)
    return 0


def goto_stage_2(root_causes, t0):
    # ----- Stage 2: cluster discovery -----------------------------
    print(f"[phase2v2] Stage 2: cluster discovery from "
          f"{len(root_causes)} analyses…", flush=True)
    rc_list = list(root_causes.values())
    discovery = call_cluster(rc_list)
    categories = discovery.get("categories", [])
    print(f"[phase2v2]   discovered {len(categories)} categories:", flush=True)
    for c in categories:
        print(f"     - {c.get('name','?')}: "
              f"{c.get('description','')[:80]}", flush=True)
    disc_path = ANALYSIS_DIR / "discovered_categories.json"
    disc_path.write_text(json.dumps(discovery, indent=2))

    # ----- Stage 3: re-classify all failures ----------------------
    print(f"[phase2v2] Stage 3: re-classifying {len(root_causes)} into "
          f"discovered categories…", flush=True)
    assignments = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {
            ex.submit(call_reclassify, rc, categories): iid
            for iid, rc in root_causes.items()
        }
        for fut in as_completed(futs):
            iid = futs[fut]
            try: cat = fut.result()
            except Exception: cat = categories[-1]["name"]
            assignments[iid] = cat

    # Tally
    from collections import Counter
    counts = Counter(assignments.values())
    target = counts.most_common(1)[0][0]
    target_pct = round(100 * counts[target] / len(assignments), 1)
    print(f"[phase2v2] target = {target} ({target_pct}% of {len(assignments)})",
          flush=True)
    print(f"[phase2v2] distribution:", flush=True)
    for c, n in counts.most_common():
        pct = round(100 * n / len(assignments), 1)
        print(f"     {c}: {n} ({pct}%)", flush=True)

    out = {
        "model": MODEL,
        "n_failures_analyzed": len(assignments),
        "discovered_categories": categories,
        "bucket_counts": dict(counts),
        "bucket_pct": {c: round(100 * n / len(assignments), 1)
                      for c, n in counts.items()},
        "target_label": target,
        "target_pct": target_pct,
        "per_task": [{"instance_id": iid, "category": c, "root_cause": root_causes[iid]}
                     for iid, c in assignments.items()],
    }
    (ANALYSIS_DIR / "capability_target_v2.json").write_text(json.dumps(out, indent=2))
    print(f"[phase2v2] DONE in {time.time() - t0:.0f}s — "
          f"target=`{target}` ({target_pct}%)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
