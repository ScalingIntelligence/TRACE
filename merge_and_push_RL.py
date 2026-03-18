import os
os.environ["HF_HOME"] = "/home/ubuntu/.cache/huggingface"
os.environ["TMPDIR"] = "/home/ubuntu/tmp"
os.makedirs("/home/ubuntu/tmp", exist_ok=True)

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import torch
import gc
import shutil

# --- Configuration ---

HF_TOKEN = "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"

# Merge method: "cat" (concatenate, default), "linear", "ties", "dare_ties", "dare_linear"
COMBINATION_TYPE = "linear"

# Each entry: (target_repo, list of (adapter_path, weight) tuples)
# Single-adapter entries just merge that one adapter into the base model.
# Multi-adapter entries combine all adapters before merging.
MERGE_JOBS = [
    (
        "tarsur909/merged-tc-sd",
        [
            ("/home/ubuntu/.cache/huggingface/tau_tool_calling/grpo_ckpt_iter_10_20260311_001228", 1.0),
            ("/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_20_20260308_055159", 1.0),
        ],
    ),
    # Add more jobs here, e.g.:
    # (
    #     "tarsur909/another-merged-model",
    #     [
    #         ("/path/to/adapter_a", 1.0),
    #         ("/path/to/adapter_b", 0.5),
    #         ("/path/to/adapter_c", 0.5),
    #     ],
    # ),
]

# --- Run merge jobs ---

for job_idx, (target_repo, adapters) in enumerate(MERGE_JOBS):
    print(f"\n{'='*60}")
    print(f"Job {job_idx + 1}/{len(MERGE_JOBS)}: {target_repo}")
    print(f"{'='*60}")

    adapter_paths = [a[0] for a in adapters]
    adapter_weights = [a[1] for a in adapters]

    # Auto-detect base model from first adapter
    cfg = PeftConfig.from_pretrained(adapter_paths[0])
    base_model = cfg.base_model_name_or_path
    print(f"Base model: {base_model}")

    for path, weight in adapters:
        print(f"  - {os.path.basename(path)} (weight={weight})")

    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if len(adapter_paths) == 1:
        # Single adapter: just load and merge
        print(f"Loading adapter: {os.path.basename(adapter_paths[0])}")
        model = PeftModel.from_pretrained(model, adapter_paths[0])
        print("Merging adapter into base model weights...")
        model = model.merge_and_unload()
    else:
        # Multiple adapters with target_parameters: merge each separately
        # and combine weight deltas manually to avoid PEFT limitation
        print("Saving base model state_dict to CPU for delta computation...")
        base_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Accumulate weighted deltas
        combined_delta = {}
        total_weight = sum(adapter_weights)

        for i, (path, weight) in enumerate(adapters):
            print(f"\nMerging adapter {i}: {os.path.basename(path)} (weight={weight})")
            # Reload base model for each adapter
            if i > 0:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            peft_model = PeftModel.from_pretrained(model, path)
            merged = peft_model.merge_and_unload()
            merged_state = merged.state_dict()

            for k in merged_state:
                delta = merged_state[k].cpu() - base_state[k].cpu()
                if delta.any():
                    if k not in combined_delta:
                        combined_delta[k] = delta * (weight / total_weight)
                    else:
                        combined_delta[k] = combined_delta[k] + delta * (weight / total_weight)

            del peft_model, merged, merged_state
            gc.collect()
            torch.cuda.empty_cache()

            # Reload clean base for next iteration or final application
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Apply combined deltas to base model
        print(f"\nApplying combined deltas ({len(combined_delta)} parameters changed)...")
        model_state = model.state_dict()
        for k, delta in combined_delta.items():
            model_state[k] = model_state[k] + delta.to(model_state[k].device)
        model.load_state_dict(model_state)
        del base_state, combined_delta, model_state
        gc.collect()
        torch.cuda.empty_cache()

    # Save merged model
    save_dir = f"/home/ubuntu/merged_upload/{target_repo.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving merged model to {save_dir}...")
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    # Upload to HuggingFace
    print(f"Uploading to {target_repo}...")
    api = HfApi()
    api.create_repo(target_repo, token=HF_TOKEN, exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=target_repo, token=HF_TOKEN)
    print(f"Done! {target_repo} pushed to HuggingFace.")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Deleting merged folder {save_dir}...")
    shutil.rmtree(save_dir)
    print(f"Deleted {save_dir}.")

print(f"\nAll {len(MERGE_JOBS)} jobs complete.")
