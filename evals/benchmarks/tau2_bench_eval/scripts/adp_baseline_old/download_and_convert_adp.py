#!/usr/bin/env python3
"""
Step 1: Download ADP SFT data directly from HuggingFace and convert to
LLaMA-Factory format.

We download individual JSONL files directly via huggingface_hub (avoiding
load_dataset which fails due to mismatched columns across splits).
"""

import json
import os
import random
import math
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "neulab/agent-data-collection"
OUTPUT_DIR = Path("/home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval/scripts/adp_baseline/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (dataset_name, sft_filename, sampling_weight)
# Prefer openhands SFT format; fall back to sweagent or agentlab
DATASETS = [
    # agenttuning (upsampled 2x)
    ("agenttuning_alfworld", "full_sft/full_sft_openhands.jsonl", 2.0),
    ("agenttuning_db",       "full_sft/full_sft_openhands.jsonl", 2.0),
    ("agenttuning_kg",       "full_sft/full_sft_openhands.jsonl", 2.0),
    ("agenttuning_mind2web", "full_sft/full_sft_openhands.jsonl", 2.0),
    ("agenttuning_os",       "full_sft/full_sft_openhands.jsonl", 2.0),
    ("agenttuning_webshop",  "full_sft/full_sft_openhands.jsonl", 2.0),
    # coding
    ("code_feedback",    "full_sft/full_sft_openhands.jsonl", 1.0),
    ("codeactinstruct",  "full_sft/full_sft_openhands.jsonl", 1.0),
    # web browsing
    ("go-browse-wa",  "full_sft/full_sft_openhands.jsonl", 1.0),
    ("mind2web",      "full_sft/full_sft_openhands.jsonl", 1.0),
    ("nnetnav-live",  "full_sft/full_sft_openhands.jsonl", 1.0),
    ("nnetnav-wa",    "full_sft/full_sft_openhands.jsonl", 1.0),
    ("synatra",       "full_sft/full_sft_openhands.jsonl", 0.01),
    # SWE
    ("nebius_SWE-agent-trajectories",          "full_sft/full_sft_openhands.jsonl", 1.0),
    ("swe-gym_openhands_sampled_trajectories", "full_sft/full_sft_openhands.jsonl", 3.0),
    ("swe-smith",                              "full_sft/full_sft_openhands.jsonl", 1.0),
    # tool use / misc
    ("openhands",          "full_sft/full_sft_openhands.jsonl", 1.0),
    ("orca_agentinstruct", "full_sft/full_sft_openhands.jsonl", 0.001),
    # optional / newer datasets
    ("coderforge_preview",       "full_sft/full_sft_openhands.jsonl", 1.0),
    ("mini-coder",               "full_sft/full_sft_openhands.jsonl", 1.0),
    ("nemotron_terminal_corpus", "full_sft/full_sft_openhands.jsonl", 1.0),
    ("swe-play-trajectories",    "full_sft/full_sft_openhands.jsonl", 1.0),
    ("toucan_1_5m",              "full_sft/full_sft_openhands.jsonl", 0.001),
]


def download_sft_file(dataset_name: str, filename: str) -> Path:
    """Download a single SFT JSONL file from the ADP dataset repo."""
    filepath = f"{dataset_name}/{filename}"
    print(f"  Downloading {filepath}...")
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filepath,
            repo_type="dataset",
        )
        return Path(local_path)
    except Exception as e:
        # Try sweagent fallback
        alt = filepath.replace("openhands", "sweagent")
        print(f"    openhands not found, trying {alt}...")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=alt,
                repo_type="dataset",
            )
            return Path(local_path)
        except Exception as e2:
            # Try agentlab fallback
            alt2 = filepath.replace("openhands", "agentlab")
            print(f"    sweagent not found, trying {alt2}...")
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=alt2,
                    repo_type="dataset",
                )
                return Path(local_path)
            except Exception as e3:
                print(f"    SKIPPED {dataset_name}: no SFT file found ({e3})")
                return None


def load_jsonl(path: Path) -> list:
    """Load a JSONL file, returning list of dicts. Skips nulls/malformed lines."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line == "null":
                continue
            try:
                item = json.loads(line)
                if item is not None and isinstance(item, dict):
                    data.append(item)
            except json.JSONDecodeError:
                continue
    return data


def sample_data(data: list, weight: float) -> list:
    """Apply ADP paper's sampling weight."""
    n = len(data)
    m = math.ceil(weight * n)

    if weight < 1.0:
        return random.sample(data, min(m, n))
    elif weight > 1.0:
        return random.choices(data, k=m)
    else:
        return data


def convert_item(item: dict) -> dict | None:
    """
    Convert a single ADP SFT item to LLaMA-Factory sharegpt format.

    Expected input: {"id": ..., "conversations": [...], "system": "..."}
    Each conversation turn: {"from": "human"/"gpt", "value": "..."}
    """
    convs = item.get("conversations", [])
    if not convs or len(convs) < 2:
        return None

    clean_convs = []
    for msg in convs:
        value = (msg.get("value") or "").strip()
        role = msg.get("from", "")
        if value and role:
            clean_convs.append({"from": role, "value": value})

    if len(clean_convs) < 2:
        return None

    entry = {"conversations": clean_convs}

    system = (item.get("system") or "").strip()
    if system:
        entry["system"] = system

    return entry


def main():
    random.seed(42)

    all_data = []
    stats = []

    print("Downloading and converting ADP datasets...\n")

    for dataset_name, filename, weight in DATASETS:
        print(f"[{dataset_name}] weight={weight}")

        local_path = download_sft_file(dataset_name, filename)
        if local_path is None:
            stats.append((dataset_name, 0, 0, "SKIPPED"))
            continue

        raw = load_jsonl(local_path)
        sampled = sample_data(raw, weight)

        converted = []
        for item in sampled:
            entry = convert_item(item)
            if entry is not None:
                converted.append(entry)

        all_data.extend(converted)
        stats.append((dataset_name, len(raw), len(converted), "OK"))
        print(f"    {len(raw)} raw -> {len(converted)} converted (total: {len(all_data)})\n")

    # Shuffle
    random.shuffle(all_data)

    # Save as JSONL (one JSON object per line) to avoid PyArrow int32 overflow
    output_path = OUTPUT_DIR / "adp_sft_train.jsonl"
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<45} {'Raw':>8} {'Used':>8} {'Status'}")
    print(f"{'-'*75}")
    for name, raw, used, status in stats:
        print(f"  {name:<43} {raw:>8} {used:>8}  {status}")
    print(f"{'-'*75}")
    print(f"  {'TOTAL':<43} {'':>8} {len(all_data):>8}")
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
