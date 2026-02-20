import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH = "/workspace/.cache/huggingface/adversarial_policy/grpo_ckpt_iter_5_20260217_161724"
TARGET_REPO = "tarsur909/Qwen3-30B-A3B-Instruct-2507-adv-sft-v1-5"
HF_TOKEN = ""

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("Loading base model in fp32 for high-precision merge...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

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
