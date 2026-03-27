"""
Verify that the patched vllm_weighted_lora produces IDENTICAL outputs
to stock vLLM 0.15.0 for normal (non-weighted) LoRA inference.

Strategy:
  1. Load model + LoRA adapter using stock vLLM offline engine
  2. Run inference with fixed seed → save output logits
  3. Swap in patched punica_gpu/base_linear/fused_moe modules
  4. Run identical inference → compare logits

If the normal code path is truly unchanged, outputs must be bit-identical.
"""
import importlib
import json
import os
import shutil
import sys
import tempfile

import torch


def get_vllm_site_packages():
    """Get the stock vLLM install path."""
    import vllm
    return os.path.dirname(os.path.dirname(vllm.__file__))


PATCHED_DIR = "/home/ubuntu/hangook/games/vllm_weighted_lora"
STOCK_DIR = "/home/ubuntu/miniconda3/envs/games/lib/python3.13/site-packages"

# Files that differ between stock and patched
PATCHED_FILES = [
    "vllm/lora/layers/base_linear.py",
    "vllm/lora/layers/fused_moe.py",
    "vllm/lora/punica_wrapper/punica_base.py",
    "vllm/lora/punica_wrapper/punica_gpu.py",
]

ADAPTER_PATH = "/home/ubuntu/.cache/huggingface/tau_tool_calling/grpo_ckpt_iter_40_20260311_030533"
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def swap_files(src_dir, dst_dir, files):
    """Copy files from src to dst, return originals for restore."""
    backups = {}
    for f in files:
        dst = os.path.join(dst_dir, f)
        src = os.path.join(src_dir, f)
        backup = dst + ".bak"
        if os.path.exists(dst):
            shutil.copy2(dst, backup)
            backups[f] = backup
        shutil.copy2(src, dst)
        # Clear __pycache__
        pyc_dir = os.path.join(os.path.dirname(dst), "__pycache__")
        base = os.path.splitext(os.path.basename(f))[0]
        if os.path.isdir(pyc_dir):
            for pyc in os.listdir(pyc_dir):
                if pyc.startswith(base):
                    os.remove(os.path.join(pyc_dir, pyc))
    return backups


def restore_files(dst_dir, backups):
    """Restore backed-up files."""
    for f, backup in backups.items():
        dst = os.path.join(dst_dir, f)
        shutil.move(backup, dst)
        pyc_dir = os.path.join(os.path.dirname(dst), "__pycache__")
        base = os.path.splitext(os.path.basename(f))[0]
        if os.path.isdir(pyc_dir):
            for pyc in os.listdir(pyc_dir):
                if pyc.startswith(base):
                    os.remove(os.path.join(pyc_dir, pyc))


def run_inference(label: str):
    """Run inference and return output token IDs + logprobs."""
    # Force reimport of patched modules
    mods_to_reload = [m for m in list(sys.modules.keys())
                      if m.startswith("vllm.lora")]
    for m in mods_to_reload:
        del sys.modules[m]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Remove any weighted config files (all ports)
    import glob as _glob
    for cfg_path in _glob.glob("/tmp/vllm_weighted_lora_config_*.json"):
        os.remove(cfg_path)

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=16,
        max_model_len=256,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=42,
        enforce_eager=True,  # disable torch.compile for determinism
    )

    lora_request = LoRARequest("test_adapter", 1, ADAPTER_PATH)

    sampling = SamplingParams(
        temperature=0.0,  # greedy for determinism
        max_tokens=32,
        seed=42,
        logprobs=5,
    )

    prompts = [
        "Hello, how can I help you today?",
        "I need to cancel my flight reservation.",
        "What is the weather like in Seoul?",
    ]

    outputs = llm.generate(
        prompts,
        sampling,
        lora_request=lora_request,
    )

    results = []
    for out in outputs:
        tokens = out.outputs[0].token_ids
        text = out.outputs[0].text
        # Collect logprobs for comparison
        lps = []
        if out.outputs[0].logprobs:
            for lp_dict in out.outputs[0].logprobs:
                # Get top logprob value for determinism check
                if lp_dict:
                    top = max(lp_dict.values(), key=lambda x: x.logprob)
                    lps.append((top.decoded_token, top.logprob))
        results.append({
            "tokens": list(tokens),
            "text": text,
            "logprobs": lps,
        })

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    import gc; gc.collect()

    print(f"\n[{label}] Results:")
    for i, r in enumerate(results):
        print(f"  Prompt {i}: {r['text'][:80]}...")
        print(f"    Tokens: {r['tokens'][:10]}...")

    return results


def compare_results(stock_results, patched_results):
    """Compare outputs from stock and patched runs."""
    print("\n" + "=" * 60)
    print("COMPARISON: Stock vs Patched")
    print("=" * 60)

    all_match = True
    for i, (stock, patched) in enumerate(zip(stock_results, patched_results)):
        tokens_match = stock["tokens"] == patched["tokens"]
        text_match = stock["text"] == patched["text"]

        if tokens_match and text_match:
            print(f"  Prompt {i}: IDENTICAL ✓")
        else:
            all_match = False
            print(f"  Prompt {i}: MISMATCH ✗")
            if not tokens_match:
                # Find first divergence
                for j, (st, pt) in enumerate(zip(stock["tokens"], patched["tokens"])):
                    if st != pt:
                        print(f"    First token diff at position {j}: stock={st} patched={pt}")
                        break
            print(f"    Stock:   {stock['text'][:100]}")
            print(f"    Patched: {patched['text'][:100]}")

        # Compare logprobs
        if stock["logprobs"] and patched["logprobs"]:
            max_lp_diff = 0.0
            for (s_tok, s_lp), (p_tok, p_lp) in zip(stock["logprobs"], patched["logprobs"]):
                max_lp_diff = max(max_lp_diff, abs(s_lp - p_lp))
            print(f"    Max logprob diff: {max_lp_diff:.2e}")

    return all_match


def main():
    print("=" * 60)
    print("Test: Normal vLLM path identical between stock and patched")
    print("=" * 60)

    # Step 1: Run with stock vLLM (current install)
    print("\n--- Running with STOCK vLLM 0.15.0 ---")
    stock_results = run_inference("STOCK")

    # Step 2: Swap in patched files
    print("\n--- Swapping in PATCHED files ---")
    backups = swap_files(PATCHED_DIR, STOCK_DIR, PATCHED_FILES)

    try:
        # Step 3: Run with patched vLLM
        print("\n--- Running with PATCHED vLLM ---")
        patched_results = run_inference("PATCHED")

        # Step 4: Compare
        all_match = compare_results(stock_results, patched_results)

        if all_match:
            print("\n" + "=" * 60)
            print("ALL OUTPUTS IDENTICAL — normal path is unchanged")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("OUTPUTS DIFFER — patch may affect normal path!")
            print("=" * 60)
            sys.exit(1)

    finally:
        # Step 5: Restore stock files
        print("\n--- Restoring STOCK files ---")
        restore_files(STOCK_DIR, backups)
        print("Stock files restored.")


if __name__ == "__main__":
    main()
