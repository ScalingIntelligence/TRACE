import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATHs =   ["/workspace/.cache/huggingface/tau_tool_calling/grpo_ckpt_iter_5_20260223_230030" ]
TARGET_REPOs = ["tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-tool-5"]
HF_TOKEN = "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"


print("Loading base model in fp32 for high-precision merge...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

for ADAPTER_PATH, TARGET_REPO in zip(ADAPTER_PATHs, TARGET_REPOs):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging LoRA weights into base model (fp32)...")
    model = model.merge_and_unload()

    print("Casting merged model to bf16...")
    model = model.to(torch.bfloat16)

    print(f"Pushing merged model to {TARGET_REPO}...")
    model.push_to_hub(TARGET_REPO, token=HF_TOKEN, max_shard_size="5GB")

    print("Pushing tokenizer...")
    tokenizer.push_to_hub(TARGET_REPO, token=HF_TOKEN)

    print("Done! Model pushed to HuggingFace.")
    del model, tokenizer
