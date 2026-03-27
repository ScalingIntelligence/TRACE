"""
End-to-end determinism test for weighted LoRA via vLLM server.

Usage:
  1. Start vLLM server with LoRA support (see launch command below)
  2. Run this script: python tests/test_weighted_lora_server_determinism.py

Expects server at http://localhost:8080/v1
"""
import json
import sys
import requests

BASE_URL = "http://localhost:8080/v1"

LORA_ADAPTERS = [
    {
        "name": "structured-grpo-iter40",
        "path": "/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_40_20260308_080312",
    },
    {
        "name": "tau-tool-calling-grpo-iter40",
        "path": "/home/ubuntu/.cache/huggingface/tau_tool_calling/grpo_ckpt_iter_40_20260311_030533",
    },
]

TEST_PROMPT = "What is the capital of France? Answer in one word."


def load_lora_adapters():
    """Load LoRA adapters via the vLLM API."""
    for adapter in LORA_ADAPTERS:
        resp = requests.post(
            f"{BASE_URL}/load_lora_adapter",
            json={
                "lora_name": adapter["name"],
                "lora_path": adapter["path"],
            },
        )
        if resp.status_code == 200:
            print(f"  Loaded: {adapter['name']}")
        else:
            print(f"  Failed to load {adapter['name']}: {resp.text}")
            return False
    return True


def set_weighted_lora(adapters_with_weights):
    """Set weighted LoRA config via the API."""
    resp = requests.post(
        f"{BASE_URL}/create_weighted_lora",
        json={
            "adapters": [
                {"name": name, "weight": weight}
                for name, weight in adapters_with_weights
            ]
        },
    )
    if resp.status_code != 200:
        print(f"  Failed to set weighted LoRA: {resp.text}")
        return False
    return True


def clear_weighted_lora():
    """Clear weighted LoRA config."""
    import os, tempfile
    from urllib.parse import urlparse
    port_tag = str(urlparse(BASE_URL).port or "default")
    config_path = os.path.join(
        tempfile.gettempdir(), f"vllm_weighted_lora_config_{port_tag}.json"
    )
    with open(config_path, "w") as f:
        json.dump([], f)


def generate(model, prompt, max_tokens=50):
    """Generate text via the chat completions API."""
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 42,
        },
    )
    if resp.status_code != 200:
        print(f"  Generation failed: {resp.text}")
        return None
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def test_base_model_determinism():
    """Base model (no LoRA) should be deterministic."""
    print("\n--- Test: Base model determinism ---")
    results = []
    for i in range(3):
        text = generate("Qwen/Qwen3-30B-A3B-Instruct-2507", TEST_PROMPT)
        if text is None:
            return False
        results.append(text)
        print(f"  Run {i+1}: {text[:80]!r}")

    if len(set(results)) == 1:
        print("  PASS: Base model is deterministic")
        return True
    else:
        print("  FAIL: Base model is NOT deterministic")
        return False


def test_single_adapter_determinism():
    """Single LoRA adapter should be deterministic."""
    print("\n--- Test: Single adapter determinism ---")
    adapter_name = LORA_ADAPTERS[0]["name"]
    results = []
    for i in range(3):
        text = generate(adapter_name, TEST_PROMPT)
        if text is None:
            return False
        results.append(text)
        print(f"  Run {i+1}: {text[:80]!r}")

    if len(set(results)) == 1:
        print("  PASS: Single adapter is deterministic")
        return True
    else:
        print("  FAIL: Single adapter is NOT deterministic")
        return False


def test_weighted_lora_determinism():
    """Weighted LoRA (multiple adapters) should be deterministic."""
    print("\n--- Test: Weighted LoRA determinism ---")

    weights = [
        (LORA_ADAPTERS[0]["name"], 0.7),
        (LORA_ADAPTERS[1]["name"], 0.3),
    ]
    if not set_weighted_lora(weights):
        return False

    # Use first adapter name as the model (weighted config overrides)
    adapter_name = LORA_ADAPTERS[0]["name"]
    results = []
    for i in range(5):
        text = generate(adapter_name, TEST_PROMPT)
        if text is None:
            return False
        results.append(text)
        print(f"  Run {i+1}: {text[:80]!r}")

    clear_weighted_lora()

    if len(set(results)) == 1:
        print("  PASS: Weighted LoRA is deterministic")
        return True
    else:
        print("  FAIL: Weighted LoRA is NOT deterministic")
        for i, r in enumerate(results):
            if r != results[0]:
                print(f"    First diff at run {i+1}")
        return False


def test_weighted_single_adapter_determinism():
    """Weighted LoRA with single adapter (w=1.0) should be deterministic."""
    print("\n--- Test: Weighted single adapter determinism ---")

    weights = [(LORA_ADAPTERS[0]["name"], 1.0)]
    if not set_weighted_lora(weights):
        return False

    adapter_name = LORA_ADAPTERS[0]["name"]
    results = []
    for i in range(3):
        text = generate(adapter_name, TEST_PROMPT)
        if text is None:
            return False
        results.append(text)
        print(f"  Run {i+1}: {text[:80]!r}")

    clear_weighted_lora()

    if len(set(results)) == 1:
        print("  PASS: Weighted single adapter is deterministic")
        return True
    else:
        print("  FAIL: Weighted single adapter is NOT deterministic")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Weighted LoRA Server Determinism Tests")
    print("=" * 60)

    # Check server is running
    try:
        resp = requests.get(f"{BASE_URL}/models")
        print(f"Server is running. Models: {[m['id'] for m in resp.json()['data']]}")
    except requests.ConnectionError:
        print("ERROR: Server not running at", BASE_URL)
        print("\nStart server with:")
        print("  conda activate games && \\")
        print("  export HF_HOME=/workspace/.cache/huggingface && \\")
        print("  export VLLM_RPC_TIMEOUT=2000 && \\")
        print("  export CUDA_VISIBLE_DEVICES=0 && \\")
        print("  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True && \\")
        print("  PYTHONDONTWRITEBYTECODE=1 vllm serve \\")
        print("    Qwen/Qwen3-30B-A3B-Instruct-2507 \\")
        print("    --host 0.0.0.0 --port 8080 --dtype bfloat16 \\")
        print("    --max-model-len 32000 --enable-lora --max-loras 2 \\")
        print("    --gpu-memory-utilization 0.9 \\")
        print("    --enable-auto-tool-choice --tool-call-parser hermes \\")
        print("    --no-enable-prefix-caching --max-num-seqs 1")
        sys.exit(1)

    # Load adapters
    print("\nLoading LoRA adapters...")
    if not load_lora_adapters():
        print("Failed to load adapters")
        sys.exit(1)

    # Run tests
    all_passed = True
    all_passed &= test_base_model_determinism()
    all_passed &= test_single_adapter_determinism()
    all_passed &= test_weighted_single_adapter_determinism()
    all_passed &= test_weighted_lora_determinism()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SERVER DETERMINISM TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)
