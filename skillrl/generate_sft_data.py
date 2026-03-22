#!/usr/bin/env python3
"""
Generate cold-start SFT data for SkillRL.

Runs tau2-bench simulations with skills injected into the agent's system prompt,
collects successful trajectories, and converts them to sharegpt format for SFT training.

The teacher model (via vLLM) acts as the agent with skill-augmented prompts.
Successful trajectories demonstrate skill-grounded behavior.

Usage:
    # Requires a running vLLM server for both agent and user simulator.
    # Agent server (with or without LoRA):
    #   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8080
    # User server:
    #   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 9000

    python skillrl/generate_sft_data.py \
        --agent-url http://localhost:8080/v1 \
        --user-url http://localhost:9000/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --domains airline,retail \
        --target-samples 1000 \
        --num-trials 5 \
        --output skillrl/data/skillrl_sft_train.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GAMES_DIR = SCRIPT_DIR.parent
EVAL_DIR = GAMES_DIR / "evals" / "benchmarks" / "tau2_bench_eval"


def load_skills_for_system_prompt(domain: str, skill_bank_dir: str) -> str:
    """Load and format skills for injection into system prompt."""
    sys.path.insert(0, str(GAMES_DIR))
    from skillrl.skillbank import SkillBank

    path = Path(skill_bank_dir) / f"{domain}_skills.json"
    sb = SkillBank(str(path))
    # Include ALL skills (no category filtering) for SFT data
    return sb.get_skills_for_prompt(top_k=100)


def run_tau2_simulations(
    domain: str,
    agent_url: str,
    user_url: str,
    model: str,
    num_trials: int,
    temperature: float,
    save_name: str,
    max_tasks: int = None,
) -> Path:
    """Run tau2-bench simulations and return path to results file."""
    config = {
        "domain": domain,
        "agent_llm": f"vllm://{model}",
        "agent_llm_args": {
            "temperature": temperature,
            "max_context_length": 16000,
        },
        "user_llm": f"vllm://{model}",
        "user_llm_args": {
            "temperature": 0.0,
        },
        "vllm": {
            "base_url": agent_url.rstrip("/v1").rstrip("/"),
        },
        "num_trials": num_trials,
    }

    # Write temp config
    config_path = Path(tempfile.mktemp(suffix=".yml"))
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Build command
    cmd = [
        sys.executable, str(EVAL_DIR / "main.py"),
        "--config", str(config_path),
        "--save-to", save_name,
    ]

    if max_tasks:
        # Use task-ids to limit
        task_ids = ",".join(str(i) for i in range(max_tasks))
        cmd.extend(["--task-ids", task_ids])

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"
    env["OPENAI_API_BASE"] = agent_url
    env["OPENAI_BASE_URL"] = agent_url

    # Set user URL
    user_base = user_url.rstrip("/v1").rstrip("/")
    cmd.extend(["--user-api-base-urls", user_base])

    print(f"[Generate] Running tau2 simulation: domain={domain}, trials={num_trials}")
    print(f"[Generate] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(EVAL_DIR))

    if result.returncode != 0:
        print(f"[Generate] STDERR: {result.stderr[-2000:]}")
        print(f"[Generate] WARNING: tau2 exited with code {result.returncode}")

    config_path.unlink(missing_ok=True)

    # Find output file
    sim_dir = EVAL_DIR / "data" / "simulations"
    results = sorted(sim_dir.glob(f"{save_name}*.json*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if results:
        return results[0]
    return None


def extract_successful_trajectories(sim_path: Path) -> list:
    """Extract successful simulations from a tau2-bench results file."""
    with open(sim_path) as f:
        data = json.load(f)

    simulations = data.get("simulations", [])
    successful = []
    for sim in simulations:
        reward = sim.get("reward_info", {}).get("reward", 0.0)
        if reward > 0:
            successful.append(sim)

    return successful


def convert_to_sharegpt(sim: dict, domain: str, skills_section: str, policy: str) -> dict:
    """Convert a tau2-bench simulation to sharegpt format with skills."""
    # Build system prompt with skills
    system = (
        "<instructions>\n"
        "You are a customer service agent that helps the user according to the <policy> provided below.\n"
        "In each turn you can either:\n"
        "- Send a message to the user.\n"
        "- Make a tool call.\n"
        "You cannot do both at the same time.\n"
        "\n"
        "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\n"
        "</instructions>\n"
        "<policy>\n"
        f"{policy}\n"
        "</policy>\n"
        f"{skills_section}"
    )

    conversations = []
    messages = sim.get("messages", [])

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if role == "user":
            # Skip terminal signals
            if content and content.strip() in ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"):
                continue
            conversations.append({"from": "human", "value": content or ""})

        elif role == "assistant":
            if tool_calls:
                # Format tool calls as JSON
                tc_formatted = []
                for tc in tool_calls:
                    tc_entry = {
                        "name": tc.get("function", {}).get("name", tc.get("name", "")),
                        "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", {})),
                    }
                    # Parse string arguments
                    if isinstance(tc_entry["arguments"], str):
                        try:
                            tc_entry["arguments"] = json.loads(tc_entry["arguments"])
                        except json.JSONDecodeError:
                            pass
                    tc_formatted.append(tc_entry)

                value = json.dumps({"tool_calls": tc_formatted}, ensure_ascii=False)
                conversations.append({"from": "gpt", "value": value})
            elif content:
                conversations.append({"from": "gpt", "value": content})

        elif role == "tool":
            # Tool responses become human turns
            tool_name = msg.get("name", "tool")
            conversations.append({
                "from": "human",
                "value": f"[Tool Response] {tool_name}: {content or ''}",
            })

    # Merge consecutive same-role turns
    merged = []
    for turn in conversations:
        if merged and merged[-1]["from"] == turn["from"]:
            merged[-1]["value"] += "\n" + turn["value"]
        else:
            merged.append(turn)

    # Ensure starts with human and has at least 2 turns
    if not merged or merged[0]["from"] != "human" or len(merged) < 2:
        return None

    # Ensure ends with gpt
    while merged and merged[-1]["from"] != "gpt":
        merged.pop()
    if len(merged) < 2:
        return None

    return {
        "conversations": merged,
        "system": system,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate SkillRL cold-start SFT data")
    parser.add_argument("--agent-url", type=str, default="http://localhost:8080/v1",
                        help="vLLM URL for agent (teacher) model")
    parser.add_argument("--user-url", type=str, default="http://localhost:9000/v1",
                        help="vLLM URL for user simulator model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="Model name")
    parser.add_argument("--domains", type=str, default="airline,retail",
                        help="Comma-separated domains")
    parser.add_argument("--target-samples", type=int, default=1000,
                        help="Target number of SFT samples")
    parser.add_argument("--num-trials", type=int, default=5,
                        help="Number of trials per task (more trials = more chances for success)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Agent temperature (>0 for diversity)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks per domain (default: all)")
    parser.add_argument("--skill-bank-dir", type=str,
                        default=str(SCRIPT_DIR / "data"),
                        help="Directory with *_skills.json files")
    parser.add_argument("--output", type=str,
                        default=str(SCRIPT_DIR / "data" / "skillrl_sft_train.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--existing-sims", type=str, default=None,
                        help="Comma-separated paths to existing simulation JSONs to also include")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]

    # Load policies
    data_dir = EVAL_DIR / "data" / "tau2" / "domains"
    policies = {}
    for domain in domains:
        policy_path = data_dir / domain / "policy.md"
        if policy_path.exists():
            policies[domain] = policy_path.read_text()
        else:
            print(f"[Generate] WARNING: No policy found at {policy_path}")
            policies[domain] = ""

    # Load skills sections
    skills_sections = {}
    for domain in domains:
        skills_sections[domain] = load_skills_for_system_prompt(domain, args.skill_bank_dir)

    all_examples = []

    # Option 1: Convert existing simulation files if provided
    if args.existing_sims:
        for sim_path_str in args.existing_sims.split(","):
            sim_path = Path(sim_path_str.strip())
            if not sim_path.exists():
                print(f"[Generate] WARNING: {sim_path} not found, skipping")
                continue

            print(f"[Generate] Loading existing simulations from {sim_path}")
            successes = extract_successful_trajectories(sim_path)
            print(f"[Generate] Found {len(successes)} successful trajectories")

            # Detect domain from filename
            domain = "airline" if "airline" in sim_path.name else "retail"

            for sim in successes:
                example = convert_to_sharegpt(
                    sim, domain, skills_sections.get(domain, ""), policies.get(domain, "")
                )
                if example:
                    all_examples.append(example)

        print(f"[Generate] Converted {len(all_examples)} examples from existing simulations")

    # Option 2: Run new simulations to reach target
    remaining = args.target_samples - len(all_examples)
    if remaining > 0:
        samples_per_domain = remaining // len(domains) + 1

        for domain in domains:
            print(f"\n[Generate] === {domain.upper()} ===")
            print(f"[Generate] Need ~{samples_per_domain} more samples")

            # Estimate: with temperature=0.7, ~40-50% success rate, 5 trials per task
            # airline has 50 tasks, retail has 115 tasks
            task_count = args.max_tasks or (50 if domain == "airline" else 115)
            expected_successes = int(task_count * args.num_trials * 0.4)
            print(f"[Generate] Running {task_count} tasks x {args.num_trials} trials "
                  f"(expecting ~{expected_successes} successes)")

            save_name = f"skillrl-sft-{domain}"
            result_path = run_tau2_simulations(
                domain=domain,
                agent_url=args.agent_url,
                user_url=args.user_url,
                model=args.model,
                num_trials=args.num_trials,
                temperature=args.temperature,
                save_name=save_name,
                max_tasks=args.max_tasks,
            )

            if result_path and result_path.exists():
                successes = extract_successful_trajectories(result_path)
                print(f"[Generate] Got {len(successes)} successful trajectories from {domain}")

                for sim in successes:
                    example = convert_to_sharegpt(
                        sim, domain, skills_sections[domain], policies[domain]
                    )
                    if example:
                        all_examples.append(example)
            else:
                print(f"[Generate] WARNING: No results file found for {domain}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\n[Generate] === DONE ===")
    print(f"[Generate] Total examples: {len(all_examples)}")
    print(f"[Generate] Saved to: {output_path}")

    # Stats
    airline_count = sum(1 for e in all_examples if "airline" in e["system"][:500].lower())
    retail_count = len(all_examples) - airline_count
    print(f"[Generate] Airline: {airline_count}, Retail: {retail_count}")


if __name__ == "__main__":
    main()
