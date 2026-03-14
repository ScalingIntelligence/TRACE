import os
os.environ["HF_HOME"] = "/home/ubuntu/.cache/huggingface"
os.environ["TMPDIR"] = "/tmp"

from unsloth import FastLanguageModel
from huggingface_hub import HfApi
import torch
import gc
import shutil

ADAPTER_PATHs = [
    "/home/ubuntu/.cache/huggingface/adversarial_policy/grpo_ckpt_iter_20_20260313_181607",
    "/home/ubuntu/.cache/huggingface/adversarial_policy/grpo_ckpt_iter_25_20260313_194228",
    "/home/ubuntu/.cache/huggingface/adversarial_policy/grpo_ckpt_iter_30_20260313_210428",
    "/home/ubuntu/.cache/huggingface/adversarial_policy/grpo_ckpt_iter_35_20260313_222857",
]
TARGET_REPOs = [
    "tarsur909/mix-multistep-structured-20",
    "tarsur909/mix-multistep-structured-25",
    "tarsur909/mix-multistep-structured-30",
    "tarsur909/mix-multistep-structured-35",
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

    save_dir = f"/home/ubuntu/merged_upload/{TARGET_REPO.split('/')[-1]}"
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

    print(f"Deleting merged folder {save_dir}...")
    shutil.rmtree(save_dir)
    print(f"Deleted {save_dir}.")
