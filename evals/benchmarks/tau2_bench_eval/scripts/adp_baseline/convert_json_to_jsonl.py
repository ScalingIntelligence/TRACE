#!/usr/bin/env python3
"""Convert the large adp_sft_train.json (single JSON array) to JSONL format.

The datasets library / pyarrow can't handle a ~47GB single JSON array because
pyarrow's block_size overflows int32. JSONL is read line-by-line and works fine.
"""
import json
import sys
from pathlib import Path

INPUT = Path(__file__).parent / "data" / "adp_sft_train.json"
OUTPUT = Path(__file__).parent / "data" / "adp_sft_train.jsonl"

print(f"Loading {INPUT} ...")
print(f"  File size: {INPUT.stat().st_size / 1e9:.2f} GB")

with open(INPUT, "r") as f:
    data = json.load(f)

print(f"  Loaded {len(data)} examples")
print(f"Writing {OUTPUT} ...")

with open(OUTPUT, "w") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"  Done. {OUTPUT.stat().st_size / 1e9:.2f} GB")
print(f"  {len(data)} lines written")
