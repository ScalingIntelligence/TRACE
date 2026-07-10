"""Phase 6 — final SWE-bench eval with the trained LoRA.

Downloads iter15 LoRA from HF, loads it into Pod-E's vLLM, and re-runs
the same eval as phase 1 — but selecting `model = <lora_name>` per
request. Compares Pass@1 to baseline.

Output:
  /tmp/trace_pipeline/runs/lora_iter15/output.jsonl
  /tmp/trace_pipeline/final/report.md
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


PIPE_ROOT = Path(os.environ.get("TRACE_PIPE_ROOT", "/tmp/trace_pipeline"))
RUN_DIR = PIPE_ROOT / "runs" / "lora_iter15"
RUN_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR = PIPE_ROOT / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

VLLM_URL = os.environ.get("TRACE_VLLM_E_URL", "http://localhost:8000/v1")
LORA_NAME = os.environ.get("TRACE_LORA_NAME", "qwen36_trace_iter15")
HF_REPO = os.environ.get("TRACE_TRAINER_HF_REPO", "tarsur385/qwen36-trace-v1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_STRUCT = os.environ.get(
    "TRACE_REPO_STRUCTURES",
    "/tmp/agentless_work/repo_structures/repo_structure/repo_structures",
)
GOLD_LOC = os.environ.get(
    "TRACE_GOLD_LOC_FILE",
    "/workspace/games/swe-rl/resources/sweb_verified_gt_loc.jsonl",
)
CONCURRENCY = int(os.environ.get("TRACE_FINAL_CONCURRENCY", "8"))


SR_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE", re.DOTALL
)

SYSTEM_PROMPT = (
    "You are an expert software engineer. Fix the bug below by emitting "
    "SEARCH/REPLACE blocks. Format:\n\n"
    "```python\n### path/to/file.py\n<<<<<<< SEARCH\n(exact code)\n=======\n"
    "(replacement)\n>>>>>>> REPLACE\n```"
)


def load_file_content(iid, file_path):
    p = Path(REPO_STRUCT) / f"{iid}.json"
    if not p.is_file():
        return ""
    with open(p) as f:
        j = json.load(f)
    for top_key, sub in j.get("structure", {}).items():
        path = file_path
        if path.startswith(top_key + "/"):
            path = path[len(top_key) + 1:]
        node = sub
        for part in path.split("/"):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if isinstance(node, dict):
            text = node.get("text")
            if isinstance(text, list):
                return "\n".join(text)
            if isinstance(text, str):
                return text
    return ""


def patch_context_for_file(patch_text, file_path):
    """Fallback: derive pre-fix code from gold-patch context lines."""
    if not patch_text:
        return ""
    parts = re.split(r"^diff --git ", patch_text, flags=re.MULTILINE)
    for blk in parts:
        if not blk.strip(): continue
        first = blk.split("\n", 1)[0]
        if file_path in first:
            out = []
            in_hunk = False
            for ln in blk.split("\n"):
                if ln.startswith("@@"):
                    if out: out.append("# ... (skipped between hunks) ...")
                    in_hunk = True; continue
                if not in_hunk or ln.startswith("+++") or ln.startswith("---"): continue
                if ln.startswith("-"): out.append(ln[1:])
                elif ln.startswith("+"): continue
                else: out.append(ln[1:] if ln.startswith(" ") else ln)
            return "\n".join(out)
    return ""


def parse_added(text):
    out = set()
    for b in re.findall(r"```python\s*\n(.*?)\n```", text or "", re.DOTALL):
        for srm in SR_RE.finditer(b):
            sl = set(srm.group(1).split("\n"))
            rl = set(srm.group(2).split("\n"))
            for ln in (rl - sl):
                if ln.strip():
                    out.add(ln)
    return out


def parse_gold(patch):
    added = set()
    cur = None
    for line in (patch or "").split("\n"):
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                cur = parts[-1].removeprefix("b/")
        elif line.startswith("+++"):
            p = line[4:].strip()
            cur = p.removeprefix("b/") if p != "/dev/null" else cur
        elif line.startswith("+") and not line.startswith("+++"):
            if cur and line[1:].strip():
                added.add(line[1:])
    return added


def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def load_lora_on_vllm():
    """Download iter15 LoRA from HF and POST /v1/load_lora_adapter."""
    from huggingface_hub import snapshot_download
    print(f"[phase6] downloading {HF_REPO} grpo_ckpt_iter_15_* …", flush=True)
    local_dir = "/tmp/trace_pipeline/loras"
    snapshot_download(repo_id=HF_REPO, allow_patterns=["grpo_ckpt_iter_15_*/*"],
                      local_dir=local_dir, token=HF_TOKEN or None)
    paths = list(Path(local_dir).glob("grpo_ckpt_iter_15_*"))
    if not paths:
        print(f"[phase6] no iter_15 dir found in {HF_REPO}", flush=True)
        return False
    adapter_path = str(paths[0])
    print(f"[phase6] loading LoRA {LORA_NAME} from {adapter_path}", flush=True)
    r = requests.post(
        f"{VLLM_URL.replace('/v1', '')}/v1/load_lora_adapter",
        json={"lora_name": LORA_NAME, "lora_path": adapter_path}, timeout=120,
    )
    print(f"[phase6] load lora status={r.status_code} body={r.text[:200]}",
          flush=True)
    return r.status_code == 200


def call_vllm(messages, max_tokens=8192, timeout=900):
    payload = {"model": LORA_NAME, "messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    r = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def process_one(d, gold_loc, gold_added, gold_patch_map=None):
    iid = d["instance_id"]
    fp = (gold_loc.get(iid) or [None])[0] or ""
    if not fp:
        return {"instance_id": iid, "raw_output": "", "jaccard": 0.0, "err": "no_loc"}
    fc = load_file_content(iid, fp)
    if not fc and gold_patch_map:
        fc = patch_context_for_file(gold_patch_map.get(iid, ""), fp)
    if not fc:
        return {"instance_id": iid, "raw_output": "", "jaccard": 0.0, "err": "no_file"}
    user = (
        f"{d['problem_statement']}\n\n"
        f"File `{fp}`:\n```python\n{fc}\n```\n\n"
        "Write SEARCH/REPLACE block(s) to fix the bug."
    )
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        raw = call_vllm(msgs)
    except Exception as e:
        return {"instance_id": iid, "raw_output": "", "jaccard": 0.0,
                "err": f"vllm:{type(e).__name__}"}
    ma = parse_added(raw)
    gad = gold_added.get(iid, set())
    return {"instance_id": iid, "raw_output": raw, "jaccard": jaccard(ma, gad),
            "model_added_count": len(ma), "gold_added_count": len(gad),
            "file_path": fp}


def main():
    if not load_lora_on_vllm():
        return 2

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    gold_loc = {}
    gold_patch_map = {}
    if Path(GOLD_LOC).is_file():
        with open(GOLD_LOC) as f:
            for line in f:
                dd = json.loads(line)
                gold_loc[dd["instance_id"]] = dd.get("found_files", [])
    else:
        for d in ds:
            files = []
            for ln in d["patch"].split("\n"):
                if ln.startswith("diff --git"):
                    parts = ln.split()
                    if len(parts) >= 4:
                        f = parts[-1].removeprefix("b/")
                        if f not in files: files.append(f)
            gold_loc[d["instance_id"]] = files
    gold_patch_map = {d["instance_id"]: d["patch"] for d in ds}
    gold_added = {d["instance_id"]: parse_gold(d["patch"]) for d in ds}
    instances = list(ds)

    out_path = RUN_DIR / "output.jsonl"
    done = set()
    if out_path.is_file():
        with open(out_path) as f:
            for line in f:
                try: done.add(json.loads(line)["instance_id"])
                except Exception: pass
    todo = [d for d in instances if d["instance_id"] not in done]
    print(f"[phase6] {len(todo)} to run", flush=True)

    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = {ex.submit(process_one, d, gold_loc, gold_added, gold_patch_map): d["instance_id"]
                    for d in todo}
            n = 0
            for fut in as_completed(futs):
                try: rec = fut.result()
                except Exception as e:
                    rec = {"instance_id": futs[fut], "raw_output": "",
                           "jaccard": 0.0, "err": f"future:{type(e).__name__}"}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += 1
                if n % 25 == 0:
                    print(f"[phase6] {n}/{len(todo)}", flush=True)

    rows = [json.loads(l) for l in open(out_path)]
    passed = sum(1 for r in rows if r.get("jaccard", 0) >= 0.95)
    n_total = len(rows)

    # Compare with baseline
    base_summary_path = PIPE_ROOT / "runs" / "baseline" / "summary.json"
    base_passed = 0
    if base_summary_path.is_file():
        base_passed = json.loads(base_summary_path.read_text()).get("n_passed", 0)

    summary = {
        "model": os.environ.get("TRACE_MODEL", "Qwen3.6-27B"),
        "lora": LORA_NAME,
        "hf_repo": HF_REPO,
        "baseline_passed": base_passed,
        "lora_passed": passed,
        "n_total": n_total,
        "delta_tasks": passed - base_passed,
        "delta_pp": round(100 * (passed - base_passed) / n_total, 2)
            if n_total else 0,
    }
    (FINAL_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    report = (
        f"# TRACE pipeline final report\n\n"
        f"Model: `{summary['model']}` + LoRA `{LORA_NAME}` "
        f"(from {HF_REPO})\n\n"
        f"| Run | Pass@1 |\n|---|---:|\n"
        f"| baseline | {base_passed}/{n_total} = "
        f"{100*base_passed/n_total:.2f}% |\n"
        f"| LoRA iter15 | {passed}/{n_total} = "
        f"{100*passed/n_total:.2f}% |\n"
        f"| **Δ** | **{summary['delta_tasks']:+d} tasks "
        f"({summary['delta_pp']:+.2f} pp)** |\n"
    )
    (FINAL_DIR / "report.md").write_text(report)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
