#!/usr/bin/env python3
"""Upload the trained ADP LoRA adapter to HuggingFace."""

from huggingface_hub import HfApi

ADAPTER_PATH = "/home/ubuntu/.cache/huggingface/adp_baseline/sft_lora"
REPO_ID = "tarsur909/adp-baseline-lora-r16"
HF_TOKEN = "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"

api = HfApi()
api.create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True)
api.upload_folder(
    folder_path=ADAPTER_PATH,
    repo_id=REPO_ID,
    token=HF_TOKEN,
)
print(f"Uploaded to: https://huggingface.co/{REPO_ID}")
