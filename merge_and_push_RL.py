import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.makedirs("/workspace/tmp", exist_ok=True)

from unsloth import FastLanguageModel
from huggingface_hub import HfApi
import torch
import gc

ADAPTER_PATHs = [
    "/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_5_20260308_042720",
    "/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_10_20260308_050043",
    "/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_15_20260308_053131",
    "/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_20_20260308_055159",
    "/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_25_20260308_062754",
    "/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_30_20260308_070158",
]
TARGET_REPOs = [
    "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-5",
    "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-10",
    "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-15",
    "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-20",
    "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-25",
    "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-30",
]
HF_TOKEN = "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"
MAX_SEQ_LENGTH = 16000


for ADAPTER_PATH, TARGET_REPO in zip(ADAPTER_PATHs, TARGET_REPOs):
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(ADAPTER_PATH)}")
    print(f"Target:     {TARGET_REPO}")
    print(f"{'='*60}")

    print("Loading model + adapter via unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        use_exact_model_name=True,
    )

    save_dir = f"/dev/sda2/merged_upload/{TARGET_REPO.split('/')[-1]}"
    print(f"Merging and saving to {save_dir}...")
    model.save_pretrained_merged(save_dir, tokenizer, save_method="merged_16bit")

    print(f"Uploading to {TARGET_REPO}...")
    api = HfApi()
    api.create_repo(TARGET_REPO, token=HF_TOKEN, exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=TARGET_REPO, token=HF_TOKEN)

    print(f"Done! {TARGET_REPO} pushed to HuggingFace.")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
