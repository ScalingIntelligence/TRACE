import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.makedirs("/workspace/tmp", exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi
import torch

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# ADAPTER_PATHs =   [ "/workspace/.cache/huggingface/default/sft/sft_ckpt_epoch_0_20260302_110400", "/workspace/.cache/huggingface/default/sft/sft_ckpt_epoch_1_20260302_121237", "/workspace/.cache/huggingface/default/sft/sft_ckpt_epoch_2_20260302_132119"]
# TARGET_REPOs = [ "tarsur909/Qwen3-30B-A3B-Instruct-2507-expert-sft-1", "tarsur909/Qwen3-30B-A3B-Instruct-2507-expert-sft-2", "tarsur909/Qwen3-30B-A3B-Instruct-2507-expert-sft-3"]
ADAPTER_PATHs =   [ "/workspace/.cache/huggingface/structured_data_v2/grpo_ckpt_iter_10_20260302_120731", "/workspace/.cache/huggingface/structured_data_v2/grpo_ckpt_iter_15_20260302_120731", "/workspace/.cache/huggingface/structured_data_v2/grpo_ckpt_iter_20_20260302_130336", "/workspace/.cache/huggingface/structured_data_v2/grpo_ckpt_iter_30_20260302_140008"]
TARGET_REPOs = [ "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v2-grpo-10", "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v2-grpo-15", "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v2-grpo-20",  "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-v2-grpo-30"]
HF_TOKEN = "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"


for ADAPTER_PATH, TARGET_REPO in zip(ADAPTER_PATHs, TARGET_REPOs):
    print("Loading base model in fp32 for high-precision merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging LoRA weights into base model (fp32)...")
    model = model.merge_and_unload()

    print("Casting merged model to bf16...")
    model = model.to(torch.bfloat16)

    # Save to /workspace (overlay / has limited space), then upload
    save_dir = f"/workspace/merged_upload/{TARGET_REPO.split('/')[-1]}"
    print(f"Saving merged model to {save_dir}...")
    model.save_pretrained(save_dir, max_shard_size="5GB", safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    print(f"Uploading to {TARGET_REPO}...")
    api = HfApi()
    api.create_repo(TARGET_REPO, token=HF_TOKEN, exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=TARGET_REPO, token=HF_TOKEN)

    print("Done! Model pushed to HuggingFace.")
    del base_model, model, tokenizer
