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
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


def setup_vllm_env(config: Dict[str, Any]) -> None:
    """Setup environment variables for vLLM if needed."""
    vllm_config = config.get("vllm", {})
    if not vllm_config:
        return

    base_url = vllm_config.get("base_url")
    api_key = vllm_config.get("api_key")

    if base_url:
        os.environ['OPENAI_API_BASE'] = base_url
        print(f"Setting OPENAI_API_BASE={base_url}")

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    elif 'OPENAI_API_KEY' not in os.environ:
        # LiteLLM requires an API key even for local servers
        os.environ['OPENAI_API_KEY'] = 'dummy-key'
        print("Setting OPENAI_API_KEY=dummy-key (required by LiteLLM)")


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

    # Check if we need vLLM
    agent_llm = merged.get("agent_llm")
    user_llm = merged.get("user_llm")

    uses_vllm = should_use_vllm(agent_llm) or should_use_vllm(user_llm)
    if uses_vllm:
        setup_vllm_env(merged)

        # Convert vLLM model names to OpenAI-compatible format
        if should_use_vllm(agent_llm):
            merged["agent_llm"] = convert_vllm_model_name(agent_llm)
            print(f"Converted agent LLM: {agent_llm} -> {merged['agent_llm']}")

        if should_use_vllm(user_llm):
            merged["user_llm"] = convert_vllm_model_name(user_llm)
            print(f"Converted user LLM: {user_llm} -> {merged['user_llm']}")

    # Build command
    cmd = ["tau2", "run"]

    # Required parameters
    if merged.get("domain"):
        cmd.extend(["--domain", str(merged["domain"])])

    if merged.get("agent_llm"):
        cmd.extend(["--agent-llm", str(merged["agent_llm"])])

    if merged.get("user_llm"):
        cmd.extend(["--user-llm", str(merged["user_llm"])])

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

    if merged.get("max_concurrency") is not None:
        cmd.extend(["--max-concurrency", str(merged["max_concurrency"])])

    if merged.get("seed") is not None:
        cmd.extend(["--seed", str(merged["seed"])])

    if merged.get("save_to"):
        cmd.extend(["--save-to", str(merged["save_to"])])

    if merged.get("verbose"):
        cmd.append("--verbose")

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
