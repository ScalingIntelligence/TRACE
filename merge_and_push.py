import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi
import torch

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATHs =   [ "/workspace/.cache/huggingface/multistep_task/grpo_ckpt_iter_10_20260227_184006"]
TARGET_REPOs = [ "tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-sft-multistep-task-10"]
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
