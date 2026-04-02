#!/usr/bin/env python3
"""
τ²-bench Evaluation Harness

A wrapper around τ²-bench that provides:
- YAML-based configuration with CLI overrides
- Support for local models via vLLM
- Automatic output directory creation
- Environment variable management

Usage:
    python main.py --config config.yml
    python main.py --config config.yml --domain retail --num-trials 3
    python main.py --config config.yml --agent-llm vllm://my-model --user-llm gpt-4o
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def should_use_vllm(llm_name: str) -> bool:
    """Check if LLM should use vLLM based on the name."""
    if not llm_name:
        return False
    return llm_name.startswith("vllm://") or llm_name.startswith("hf://") or llm_name.startswith("huggingface://")


def get_vllm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get vLLM configuration (base_url, base_urls, and api_key)."""
    vllm_config = config.get("vllm", {})
    return {
        "base_url": vllm_config.get("base_url"),
        "base_urls": vllm_config.get("base_urls"),  # List of URLs for deterministic multi-server (shared)
        "agent_base_urls": vllm_config.get("agent_base_urls"),  # Separate URLs for agent
        "user_base_urls": vllm_config.get("user_base_urls"),  # Separate URLs for user
        "api_key": vllm_config.get("api_key"),
    }


def load_lora_adapter(base_url: str, adapter_name: str, adapter_path: str) -> bool:
    """
    Load a LoRA adapter into the vLLM server dynamically.

    Args:
        base_url: vLLM server base URL (e.g., http://localhost:8080/v1)
        adapter_name: Name to register the adapter as
        adapter_path: Path to the LoRA adapter directory

    Returns:
        True if successful, False otherwise
    """
    # Remove /v1 suffix if present for the load endpoint
    server_url = base_url.rstrip('/').replace('/v1', '')
    load_url = f"{server_url}/v1/load_lora_adapter"

    payload = {
        "lora_name": adapter_name,
        "lora_path": adapter_path
    }

    print(f"Loading LoRA adapter '{adapter_name}' from: {adapter_path}")

    try:
        response = requests.post(load_url, json=payload, timeout=60)
        if response.status_code == 200:
            print(f"✓ Successfully loaded LoRA adapter '{adapter_name}'")
            return True
        else:
            # Check if adapter already exists
            if "already exists" in response.text.lower():
                print(f"✓ LoRA adapter '{adapter_name}' already loaded")
                return True
            print(f"✗ Failed to load LoRA adapter: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Error loading LoRA adapter: {e}")
        return False


def check_model_available(base_url: str, model_name: str) -> bool:
    """Check if a model is available on the vLLM server."""
    server_url = base_url.rstrip('/').replace('/v1', '')
    models_url = f"{server_url}/v1/models"

    try:
        response = requests.get(models_url, timeout=10)
        if response.status_code == 200:
            models = response.json().get("data", [])
            model_ids = [m["id"] for m in models]
            return model_name in model_ids
    except requests.exceptions.RequestException:
        pass
    return False


def convert_vllm_model_name(llm_name: str) -> str:
    """
    Convert vLLM-style model names to OpenAI-compatible format for LiteLLM.

    Examples:
        vllm://my-model -> openai/my-model
        hf://model-name -> openai/model-name
    """
    if not llm_name:
        return llm_name

    if llm_name.startswith("vllm://"):
        return "openai/" + llm_name[7:]  # Remove "vllm://" prefix
    elif llm_name.startswith("hf://"):
        return "openai/" + llm_name[5:]  # Remove "hf://" prefix
    elif llm_name.startswith("huggingface://"):
        return "openai/" + llm_name[14:]  # Remove "huggingface://" prefix

    return llm_name


def build_tau2_command(config: Dict[str, Any], cli_overrides: Dict[str, Any]) -> list[str]:
    """Build the tau2 run command from config and CLI overrides."""
    # Merge config with CLI overrides
    merged = {**config, **cli_overrides}

    # Get vLLM config
    vllm_config = get_vllm_config(merged)

    # Check which models use vLLM
    agent_llm = merged.get("agent_llm")
    user_llm = merged.get("user_llm")
    agent_uses_vllm = should_use_vllm(agent_llm)
    user_uses_vllm = should_use_vllm(user_llm)

    # Check for LoRA adapter configuration
    lora_adapter_path = merged.get("lora_adapter_path")
    lora_adapter_name = merged.get("lora_adapter_name", "custom_adapter")

    # If LoRA adapter is specified and agent uses vLLM, load the adapter
    # Resolve base_url for LoRA loading: prefer base_url, fall back to first agent_base_urls entry
    _lora_base_url = vllm_config.get("base_url")
    if not _lora_base_url and vllm_config.get("agent_base_urls"):
        _lora_base_url = vllm_config["agent_base_urls"][0]
    if lora_adapter_path and agent_uses_vllm and _lora_base_url:
        base_url = _lora_base_url

        # Check if adapter is already loaded
        if not check_model_available(base_url, lora_adapter_name):
            # Load the adapter
            if not load_lora_adapter(base_url, lora_adapter_name, lora_adapter_path):
                print(f"Warning: Failed to load LoRA adapter, falling back to base model")
            else:
                # Use the adapter name as the model
                agent_llm = f"vllm://{lora_adapter_name}"
                merged["agent_llm"] = f"openai/{lora_adapter_name}"
                print(f"Using LoRA adapter as agent: {lora_adapter_name}")
        else:
            # Adapter already loaded, use it
            agent_llm = f"vllm://{lora_adapter_name}"
            merged["agent_llm"] = f"openai/{lora_adapter_name}"
            print(f"Using existing LoRA adapter as agent: {lora_adapter_name}")

    # Load and concatenate extra agent prompt files
    agent_prompt_files = merged.get("agent_prompt_files", [])
    if agent_prompt_files:
        parts = []
        for fpath in agent_prompt_files:
            p = Path(fpath)
            if not p.exists():
                print(f"Warning: agent_prompt_file not found, skipping: {fpath}")
                continue
            parts.append(p.read_text().strip())
            print(f"Loaded agent prompt file: {fpath}")
        extra_system_prompt = "\n\n".join(parts) if parts else None
    else:
        extra_system_prompt = None

    # Prepare llm_args for agent and user separately
    # This allows mixed model evaluation (e.g., vLLM agent + OpenAI user)
    agent_llm_args = {"temperature": 0.0, **merged.get("agent_llm_args", {})}
    user_llm_args = {"temperature": 0.0, **merged.get("user_llm_args", {})}

    # Inject extra system prompt into agent_llm_args for passthrough to LLMAgent
    if extra_system_prompt:
        agent_llm_args["extra_system_prompt"] = extra_system_prompt

    # Orchestrator + skill LLMs configuration
    orchestrator_config = merged.get("orchestrator")
    if orchestrator_config:
        print(f"Orchestrator mode enabled with {len(orchestrator_config.get('skills', []))} skills")
        print("  Weight-merge mode: adapters merged into base weights (deterministic)")
        print("  Server should run WITHOUT --enable-lora")
        # Inject orchestrator config into agent_llm_args for passthrough to tau2
        agent_llm_args["orchestrator_config"] = orchestrator_config

    # Check if using multi-server deterministic mode
    # Support both shared (base_urls) and separate (agent_base_urls/user_base_urls) configurations
    use_shared_multi_server = vllm_config.get("base_urls") is not None and len(vllm_config.get("base_urls", [])) > 0
    use_separate_multi_server = (
        vllm_config.get("agent_base_urls") is not None and len(vllm_config.get("agent_base_urls", [])) > 0 and
        vllm_config.get("user_base_urls") is not None and len(vllm_config.get("user_base_urls", [])) > 0
    )
    use_multi_server = use_shared_multi_server or use_separate_multi_server

    # Convert vLLM model names and inject api_base into llm_args
    if agent_uses_vllm:
        # Only convert if not already converted (for LoRA case)
        if not merged["agent_llm"].startswith("openai/"):
            merged["agent_llm"] = convert_vllm_model_name(agent_llm)
            print(f"Converted agent LLM: {agent_llm} -> {merged['agent_llm']}")
        # Only set api_base if NOT using multi-server mode (multi-server routing handled by tau2)
        if not use_multi_server and vllm_config.get("base_url"):
            agent_llm_args["api_base"] = vllm_config["base_url"]
            print(f"Agent will use vLLM at: {vllm_config['base_url']}")
        # Set API key (use EMPTY for vLLM servers that don't require auth)
        agent_llm_args["api_key"] = vllm_config.get("api_key") or "EMPTY"

    if user_uses_vllm:
        merged["user_llm"] = convert_vllm_model_name(user_llm)
        print(f"Converted user LLM: {user_llm} -> {merged['user_llm']}")
        # Only set api_base if NOT using multi-server mode
        if not use_multi_server and vllm_config.get("base_url"):
            user_llm_args["api_base"] = vllm_config["base_url"]
            print(f"User will use vLLM at: {vllm_config['base_url']}")
        # Set API key (use EMPTY for vLLM servers that don't require auth)
        user_llm_args["api_key"] = vllm_config.get("api_key") or "EMPTY"

    if use_separate_multi_server:
        print(f"Using separate multi-server mode:")
        print(f"  Agent servers ({len(vllm_config['agent_base_urls'])}):")
        for i, url in enumerate(vllm_config['agent_base_urls']):
            print(f"    {i}: {url}")
        print(f"  User servers ({len(vllm_config['user_base_urls'])}):")
        for i, url in enumerate(vllm_config['user_base_urls']):
            print(f"    {i}: {url}")
    elif use_shared_multi_server:
        print(f"Using shared multi-server mode with {len(vllm_config['base_urls'])} servers:")
        for i, url in enumerate(vllm_config['base_urls']):
            print(f"  Server {i}: {url}")

    # Build command
    cmd = ["tau2", "run"]

    # Required parameters
    if merged.get("domain"):
        cmd.extend(["--domain", str(merged["domain"])])

    if merged.get("agent_llm"):
        cmd.extend(["--agent-llm", str(merged["agent_llm"])])

    if merged.get("user_llm"):
        cmd.extend(["--user-llm", str(merged["user_llm"])])

    # Add llm_args for agent and user
    cmd.extend(["--agent-llm-args", json.dumps(agent_llm_args)])
    cmd.extend(["--user-llm-args", json.dumps(user_llm_args)])

    # Optional parameters
    if merged.get("agent"):
        cmd.extend(["--agent", str(merged["agent"])])

    if merged.get("user"):
        cmd.extend(["--user", str(merged["user"])])

    if merged.get("num_trials") is not None:
        cmd.extend(["--num-trials", str(merged["num_trials"])])

    if merged.get("num_tasks") is not None:
        cmd.extend(["--num-tasks", str(merged["num_tasks"])])

    if merged.get("task_ids") is not None:
        task_ids = merged["task_ids"]
        if isinstance(task_ids, list):
            cmd.extend(["--task-ids"] + [str(tid) for tid in task_ids])
        else:
            cmd.extend(["--task-ids", str(task_ids)])

    if merged.get("max_steps") is not None:
        cmd.extend(["--max-steps", str(merged["max_steps"])])

    if merged.get("max_concurrency") is not None:
        cmd.extend(["--max-concurrency", str(merged["max_concurrency"])])

    if merged.get("seed") is not None:
        cmd.extend(["--seed", str(merged["seed"])])

    if merged.get("save_to"):
        cmd.extend(["--save-to", str(merged["save_to"])])

    if merged.get("verbose"):
        cmd.extend(["--log-level", "DEBUG"])

    # Add multi-server URLs for deterministic evaluation
    if use_separate_multi_server:
        cmd.extend(["--agent-api-base-urls"] + vllm_config["agent_base_urls"])
        cmd.extend(["--user-api-base-urls"] + vllm_config["user_base_urls"])
    elif use_shared_multi_server:
        cmd.extend(["--api-base-urls"] + vllm_config["base_urls"])

    return cmd


def ensure_tau2_data_dir() -> None:
    """
    Ensure TAU2_DATA_DIR is set and points to valid data directory.

    If not set, tries to find tau2-bench repository in common locations.
    """
    if 'TAU2_DATA_DIR' in os.environ:
        data_dir = Path(os.environ['TAU2_DATA_DIR'])
        if data_dir.exists():
            return
        print(f"Warning: TAU2_DATA_DIR is set but directory doesn't exist: {data_dir}")

    # Try to find tau2-bench data in common locations
    current_dir = Path(__file__).parent
    possible_locations = [
        current_dir / "data",  # Local data directory (from setup_data.py)
        current_dir / "tau2-bench" / "data",  # In same directory as eval harness
        current_dir.parent.parent / "tau2-bench" / "data",  # In weaver_for_RL root
        Path.home() / "tau2-bench" / "data",  # In home directory
    ]

    for location in possible_locations:
        if location.exists() and (location / "tau2").exists():
            os.environ['TAU2_DATA_DIR'] = str(location)
            print(f"Found tau2-bench data directory: {location}")
            return

    # Couldn't find data directory
    print("\n" + "!" * 80)
    print("ERROR: tau2-bench data files not found!")
    print("!" * 80)
    print("\nτ²-bench requires data files (tasks, policies, etc.).")
    print("\nQuick setup (recommended):")
    print("  python setup_data.py")
    print("\nThis downloads only the necessary data files (~2MB).")
    print("\nAlternatively, set TAU2_DATA_DIR manually:")
    print("  export TAU2_DATA_DIR=/path/to/tau2-bench/data")
    print("!" * 80)
    sys.exit(1)


def run_evaluation(config: Dict[str, Any], cli_overrides: Dict[str, Any]) -> None:
    """Run τ²-bench evaluation with given configuration."""
    # Ensure TAU2_DATA_DIR is set
    ensure_tau2_data_dir()

    # Auto-create output directory if specified
    output_dir = config.get("output_dir", "outputs")
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}")

    # Build tau2 command
    cmd = build_tau2_command(config, cli_overrides)

    # Print configuration
    print("=" * 80)
    print("τ²-bench Evaluation Configuration")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    print()

    # Run tau2 command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            cwd=Path(__file__).parent  # Run from tau2_bench_eval directory
        )

        print("\n" + "=" * 80)
        print("✓ Evaluation completed successfully!")
        print("=" * 80)

        # tau2 saves results in data/tau2/simulations/ by default
        print("\nResults location:")
        print("  τ²-bench saves results to: data/tau2/simulations/")
        print("  Use 'tau2 view' to browse results")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running τ²-bench: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("\n✗ Error: 'tau2' command not found.", file=sys.stderr)
        print("Please install τ²-bench:", file=sys.stderr)
        print("  uv sync", file=sys.stderr)
        print("  or", file=sys.stderr)
        print("  pip install -e .", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="τ²-bench Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python main.py --config config.yml

  # Override domain and trials
  python main.py --config config.yml --domain retail --num-trials 3

  # Use vLLM for agent, OpenAI for user
  python main.py --config config.yml --agent-llm vllm://my-model --user-llm gpt-4o

  # Run specific tasks
  python main.py --config config.yml --task-ids 1 2 3
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to YAML configuration file (default: config.yml)"
    )

    # Domain configuration
    parser.add_argument("--domain", type=str, help="Domain (airline, retail, telecom, telecom-workflow, mock)")

    # Model configuration
    parser.add_argument("--agent-llm", type=str, help="LLM for the agent")
    parser.add_argument("--user-llm", type=str, help="LLM for user simulator")
    parser.add_argument("--agent", type=str, help="Custom agent class")
    parser.add_argument("--user", type=str, help="Custom user class")

    # Task configuration
    parser.add_argument("--num-trials", type=int, help="Number of trials per task")
    parser.add_argument("--num-tasks", type=int, help="Number of tasks to run")
    parser.add_argument("--task-ids", type=int, nargs="+", help="Specific task IDs to run")

    # Execution configuration
    parser.add_argument("--max-steps", type=int, help="Maximum number of steps per simulation (default: 200)")
    parser.add_argument("--max-concurrency", type=int, help="Number of concurrent simulations")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--save-to", type=str, help="Custom filename prefix for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    # Extract CLI overrides (only non-None values)
    cli_overrides = {
        k.replace("-", "_"): v for k, v in vars(args).items()
        if v is not None and k != "config"
    }

    # Run evaluation
    run_evaluation(config, cli_overrides)


if __name__ == "__main__":
    main()
