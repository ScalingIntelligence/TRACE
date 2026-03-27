#!/usr/bin/env python3
"""
Run tau2-bench evaluation with SkillRL skills injected into the agent's system prompt.

Temporarily patches tau2's SYSTEM_PROMPT in the source file, runs evaluation
via the normal tau2 CLI, then restores the original file.

Usage:
    # Single run with all skills:
    python eval_with_skills.py \
        --domain airline \
        --agent-url http://localhost:5050/v1 \
        --user-url http://localhost:5051/v1 \
        --skill-bank-file skillrl/data/skills_new.json

    # Single run with specific number of skills:
    python eval_with_skills.py \
        --domain airline \
        --num-skills 4 \
        --skill-bank-file skillrl/data/skills_new.json ...

    # Ablation study (runs 0, 1, 2, 4, 8, all automatically):
    python eval_with_skills.py \
        --domain airline \
        --ablation \
        --skill-bank-file skillrl/data/skills_new.json ...
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR / "evals" / "benchmarks" / "tau2_bench_eval"

# The tau2 source file that contains SYSTEM_PROMPT
TAU2_AGENT_FILE = SCRIPT_DIR / "tau2-bench" / "src" / "tau2" / "agent" / "llm_agent.py"


def run_single_eval(args, skills_section, save_name, num_skills_label):
    """Run a single evaluation with the given skills injected. Returns result dict or None."""

    if not TAU2_AGENT_FILE.exists():
        print(f"ERROR: tau2 agent file not found at {TAU2_AGENT_FILE}")
        return None

    original_content = TAU2_AGENT_FILE.read_text()

    # Find the SYSTEM_PROMPT definition and inject skills
    old_prompt = '''SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()'''

    if skills_section:
        escaped_skills = skills_section.replace('{', '{{').replace('}', '}}')
        new_prompt = '''SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
''' + escaped_skills + '''
""".strip()'''
    else:
        # No skills — use original prompt (baseline)
        new_prompt = old_prompt

    if old_prompt not in original_content:
        print("ERROR: Could not find SYSTEM_PROMPT in tau2 source to patch.")
        return None

    patched_content = original_content.replace(old_prompt, new_prompt)
    TAU2_AGENT_FILE.write_text(patched_content)
    print(f"[SkillRL Eval] Patched {TAU2_AGENT_FILE.name} ({num_skills_label})")

    # Also patch llm_utils.py to fix Python 3.13 threading + lazy import bug
    llm_utils_file = TAU2_AGENT_FILE.parent.parent / "utils" / "llm_utils.py"
    llm_utils_original = llm_utils_file.read_text()
    if "from transformers import AutoTokenizer" not in llm_utils_original.split("def _get_tokenizer")[0]:
        llm_utils_patched = llm_utils_original.replace(
            "def _get_tokenizer(model: str):",
            "try:\n    from transformers import AutoTokenizer as _AutoTokenizer\nexcept ImportError:\n    _AutoTokenizer = None\n\ndef _get_tokenizer(model: str):",
        ).replace(
            "    from transformers import AutoTokenizer\n    tokenizer = AutoTokenizer.from_pretrained",
            "    AutoTokenizer = _AutoTokenizer\n    tokenizer = AutoTokenizer.from_pretrained",
        )
        llm_utils_file.write_text(llm_utils_patched)

    try:
        agent_base = args.agent_url.removesuffix("/v1").removesuffix("/")
        user_base = args.user_url.removesuffix("/v1").removesuffix("/")

        import yaml
        config = {
            "domain": args.domain,
            "agent_llm": f"openai/{args.model}",
            "agent_llm_args": {
                "temperature": 0.0,
                "max_context_length": 32000,
                "tokenizer_model": args.model,
                "api_base": agent_base + "/v1",
                "api_key": "EMPTY",
            },
            "user_llm": f"openai/{args.model}",
            "user_llm_args": {
                "temperature": 0.0,
                "api_base": user_base + "/v1",
                "api_key": "EMPTY",
            },
            "num_trials": args.num_trials,
            "max_steps": 50,
            "seed": 42,
            "save_to": save_name,
        }

        config_path = Path(tempfile.mktemp(suffix=".yml"))
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        cmd = [
            sys.executable, str(EVAL_DIR / "main.py"),
            "--config", str(config_path),
        ]
        if args.task_ids:
            cmd.extend(["--task-ids"] + args.task_ids.split(","))
        if args.max_concurrency:
            cmd.extend(["--max-concurrency", str(args.max_concurrency)])

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "EMPTY")

        print(f"[SkillRL Eval] Running: domain={args.domain}, save_to={save_name}")
        print()

        t0 = time.time()
        result = subprocess.run(cmd, env=env, cwd=str(EVAL_DIR))
        elapsed = time.time() - t0

        config_path.unlink(missing_ok=True)

        # Read the results file and inject the skills prompt used
        results_path = EVAL_DIR / "data" / "simulations" / f"{save_name}.json"
        if results_path.exists():
            with open(results_path) as f:
                eval_data = json.load(f)

            # Save the injected skills prompt into the result file
            eval_data.setdefault("info", {})
            eval_data["info"]["skills_prompt"] = skills_section or "(none — baseline)"
            eval_data["info"]["num_skills"] = num_skills_label
            with open(results_path, "w") as f:
                json.dump(eval_data, f, ensure_ascii=False)

            sims = eval_data["simulations"]
            total = len(sims)
            rewards = [s["reward_info"]["reward"] for s in sims]
            passed = sum(1 for r in rewards if r == 1.0)
            avg_reward = sum(rewards) / total if total > 0 else 0.0
            return {
                "num_skills": num_skills_label,
                "save_to": save_name,
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0.0,
                "avg_reward": avg_reward,
                "elapsed_seconds": round(elapsed, 1),
            }
        else:
            print(f"[SkillRL Eval] WARNING: Results file not found at {results_path}")
            return None

    finally:
        TAU2_AGENT_FILE.write_text(original_content)
        if 'llm_utils_original' in locals():
            llm_utils_file.write_text(llm_utils_original)
        print(f"[SkillRL Eval] Restored original source files")


def main():
    parser = argparse.ArgumentParser(description="Tau2-bench eval with SkillRL skill injection")
    parser.add_argument("--domain", type=str, required=True, choices=["airline", "retail"])
    parser.add_argument("--agent-url", type=str, default="http://localhost:8080/v1")
    parser.add_argument("--user-url", type=str, default="http://localhost:9000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--skill-bank-dir", type=str,
                        default=str(SCRIPT_DIR / "skillrl" / "data"))
    parser.add_argument("--skill-bank-file", type=str, default=None,
                        help="Path to a specific skill bank JSON file (overrides --skill-bank-dir)")
    parser.add_argument("--num-skills", type=int, default=None,
                        help="Number of skills to include. Default: all skills.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study: 0, 1, 2, 4, 8, all skills. "
                             "Results saved to a summary JSON.")
    parser.add_argument("--ablation-steps", type=str, default=None,
                        help="Custom ablation steps as comma-separated ints (e.g., '0,1,2,4,8,12'). "
                             "Default: 0,1,2,4,8,<all>")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--save-to", type=str, default=None)
    parser.add_argument("--task-ids", type=str, default=None,
                        help="Comma-separated task IDs to evaluate (default: all)")
    parser.add_argument("--agent-llm", type=str, default=None,
                        help="Agent LLM override (e.g., vllm://adapter_name)")
    parser.add_argument("--max-concurrency", type=int, default=None)
    args = parser.parse_args()

    # Load skill bank
    from skillrl.skillbank import SkillBank
    if args.skill_bank_file:
        skill_path = Path(args.skill_bank_file)
    else:
        skill_path = Path(args.skill_bank_dir) / f"{args.domain}_skills.json"
    sb = SkillBank(str(skill_path))
    print(f"[SkillRL Eval] Loaded {sb}")

    if args.ablation:
        # --- Ablation mode ---
        total_skills = sb.total_skills
        if args.ablation_steps:
            steps = [int(x.strip()) for x in args.ablation_steps.split(",")]
        else:
            steps = sorted(set([1, 2, 4, 8, total_skills]))

        skill_bank_name = skill_path.stem
        summary = {
            "skill_bank": str(skill_path),
            "domain": args.domain,
            "model": args.model,
            "total_skills_available": total_skills,
            "ablation_steps": steps,
            "results": [],
        }

        print(f"\n{'='*60}")
        print(f"  ABLATION STUDY: {args.domain}")
        print(f"  Skill bank: {skill_path.name} ({total_skills} skills)")
        print(f"  Steps: {steps}")
        print(f"{'='*60}\n")

        for k in steps:
            print(f"\n{'─'*60}")
            print(f"  Running with {k} skills...")
            print(f"{'─'*60}")

            if k == 0:
                skills_section = None
            else:
                skills_section = sb.get_skills_for_prompt(top_k=k)

            save_name = args.save_to or f"{skill_bank_name}-{args.domain}"
            save_name_k = f"{save_name}-k{k}"

            result = run_single_eval(args, skills_section, save_name_k, k)
            if result:
                summary["results"].append(result)
                print(f"\n  >> k={k}: {result['passed']}/{result['total']} "
                      f"({result['pass_rate']*100:.1f}%) "
                      f"[{result['elapsed_seconds']}s]")

        # Save summary
        summary_path = EVAL_DIR / "data" / "simulations" / f"{save_name}-ablation.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print final table
        print(f"\n\n{'='*60}")
        print(f"  ABLATION RESULTS: {args.domain}")
        print(f"{'='*60}")
        print(f"  {'Skills':>8s}  {'Passed':>8s}  {'Total':>6s}  {'Rate':>8s}")
        print(f"  {'─'*36}")
        for r in summary["results"]:
            print(f"  {r['num_skills']:>8d}  {r['passed']:>8d}  {r['total']:>6d}  "
                  f"{r['pass_rate']*100:>7.1f}%")
        print(f"\n  Summary saved to: {summary_path}")

    else:
        # --- Single run mode ---
        top_k = args.num_skills if args.num_skills is not None else 100
        skills_section = sb.get_skills_for_prompt(top_k=top_k)

        actual_count = min(top_k, sb.total_skills)
        skill_bank_name = skill_path.stem
        save_name = args.save_to or f"{skill_bank_name}-{args.domain}-k{actual_count}"

        result = run_single_eval(args, skills_section, save_name, actual_count)
        if result:
            print(f"\n[SkillRL Eval] Result: {result['passed']}/{result['total']} "
                  f"({result['pass_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()


#   python eval_with_skills.py \
#       --domain airline \
#       --agent-url http://localhost:5050/v1 \
#       --user-url http://localhost:5051/v1 \
#       --model Qwen/Qwen3-30B-A3B-Instruct-2507

#   python eval_with_skills.py \
#       --domain retail \
#       --agent-url http://localhost:5050/v1 \
#       --user-url http://localhost:5051/v1 \
#       --model Qwen/Qwen3-30B-A3B-Instruct-2507

# Ablation example:
#   python eval_with_skills.py \
#       --domain airline \
#       --ablation \
#       --skill-bank-file skillrl/data/skills_new.json \
#       --agent-url http://localhost:5050/v1 \
#       --user-url http://localhost:5051/v1

#   python eval_with_skills.py \
#       --domain retail \
#       --ablation \
#       --skill-bank-file skillrl/data/skills_new.json \
#       --agent-url http://localhost:5051/v1 \
#       --user-url http://localhost:5051/v1

#   python eval_with_skills.py \
#       --domain airline \
#       --ablation \
#       --skill-bank-file skillrl/data/skills_new.json \
#       --agent-url http://localhost:5050/v1 \
#       --user-url http://localhost:5050/v1

#   # Retail ablation (port 5051)
#   python eval_with_skills.py \
#       --domain retail \
#       --ablation \
#       --skill-bank-file skillrl/data/skills_new.json \
#       --agent-url http://localhost:5051/v1 \
#       --user-url http://localhost:5051/v1