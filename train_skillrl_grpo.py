#!/usr/bin/env python3
"""
SkillRL GRPO Trainer: GRPO with skill-augmented prompts and recursive evolution.

Wraps train_grpo_optimized.py with:
1. SkillBank loading and injection into the game environment
2. Recursive skill evolution after tau2-bench validation epochs

Usage:
    # First run cold-start SFT, then:
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
    VLLM_BASE_URLS=http://localhost:8080 \
    VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
    torchrun --nproc_per_node=6 train_skillrl_grpo.py \
        --game adversarial_policy_skillrl \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --compact-tools \
        --temperature-range "0.7,0.9,1.0" \
        --groups-per-batch 8 --group-size 16 \
        --skill-bank-dir skillrl/data \
        --skill-evolution \
        --evolution-teacher-url http://localhost:8080/v1 \
        --evolution-teacher-model Qwen/Qwen3-30B-A3B-Instruct-2507
"""
import argparse
import json
import os
import sys
from pathlib import Path

# SkillRL imports
from skillrl.skillbank import SkillBank, load_skillbank_for_domain
from skillrl.skill_evolution import (
    analyze_failures,
    evolve_skills,
    create_vllm_generate_fn,
)


def patch_game_registry_with_skills(skill_bank_dir: str):
    """Patch the adversarial_policy_skillrl game factory to inject SkillBanks.

    The game environment detects the domain at reset() time and uses the
    appropriate SkillBank. We pre-load both and pass them via a closure.
    """
    skillbanks = {}
    for domain in ("airline", "retail"):
        path = Path(skill_bank_dir) / f"{domain}_skills.json"
        if path.exists():
            sb = SkillBank(str(path))
            skillbanks[domain] = sb
            print(f"[SkillRL] Loaded {sb}")

    if not skillbanks:
        print("[SkillRL] WARNING: No skill banks found. Running without skills.")
        return skillbanks

    # Patch the game registry to pass skillbanks to the env
    from game_registry import get_game, _REGISTRY
    from adversarial_policy_game.game import AdversarialPolicyGame

    # The game factory needs to pick the right skillbank based on domain.
    # Since domain is determined at reset() time (random seed -> scenario),
    # we pass BOTH skillbanks and let the env pick at get_system_prompt() time.
    # We create a MultiDomainSkillBank wrapper for this.
    class MultiDomainSkillBank:
        """Routes to the correct SkillBank based on domain."""
        def __init__(self, banks):
            self._banks = banks
            self._current = None

        def set_domain(self, domain):
            self._current = self._banks.get(domain)

        def get_skills_for_prompt(self, **kwargs):
            if self._current is None:
                return ""
            return self._current.get_skills_for_prompt(**kwargs)

    multi_bank = MultiDomainSkillBank(skillbanks)

    # Monkey-patch the game spec's make_env to inject the skillbank
    if "adversarial_policy_skillrl" in _REGISTRY:
        original_make = _REGISTRY["adversarial_policy_skillrl"].make_env

        def make_with_skills(user_client=None, adversarial_ratio=0.2, **kwargs):
            return AdversarialPolicyGame(
                max_steps=30, user_client=user_client,
                adversarial_ratio=adversarial_ratio,
                skillbank=multi_bank,
            )

        _REGISTRY["adversarial_policy_skillrl"] = _REGISTRY["adversarial_policy_skillrl"]._replace(
            make_env=make_with_skills
        ) if hasattr(_REGISTRY["adversarial_policy_skillrl"], '_replace') else None

        # GameSpec is a dataclass not namedtuple, so we do it differently
        spec = _REGISTRY["adversarial_policy_skillrl"]
        # Create a new spec with the patched factory
        from game_registry import GameSpec, register_game
        _REGISTRY["adversarial_policy_skillrl"] = GameSpec(
            name=spec.name,
            make_env=make_with_skills,
            extract_action=spec.extract_action,
            action_space=spec.action_space,
            stop_sequences=spec.stop_sequences,
            system_prompt=spec.system_prompt,
            max_gen_tokens=spec.max_gen_tokens,
        )
        print(f"[SkillRL] Patched adversarial_policy_skillrl game with skill injection")

    # Also hook into reset to set the domain on the multi-bank
    original_reset = AdversarialPolicyGame.reset

    def reset_with_domain_detection(self, seed, user_difficulty=None):
        original_reset(self, seed, user_difficulty)
        if self._skillbank is multi_bank and self._scenario is not None:
            multi_bank.set_domain(self._scenario.domain)

    AdversarialPolicyGame.reset = reset_with_domain_detection
    print(f"[SkillRL] Hooked domain detection into game reset")

    return skillbanks


def run_skill_evolution_step(
    skillbanks: dict,
    eval_results: dict,
    teacher_url: str,
    teacher_model: str,
    threshold: float = 0.4,
    max_new: int = 3,
):
    """Run recursive skill evolution after evaluation.

    Args:
        skillbanks: Dict of domain -> SkillBank
        eval_results: Dict from evaluate_tau2_bench (e.g., eval_tau2/airline_success_rate)
        teacher_url: vLLM base URL for the teacher model
        teacher_model: Model name for generation
        threshold: Don't evolve if accuracy >= threshold
        max_new: Max new skills per evolution
    """
    generate_fn = create_vllm_generate_fn(teacher_url, teacher_model)

    for domain, sb in skillbanks.items():
        key = f"eval_tau2/{domain}_success_rate"
        sr = eval_results.get(key, 1.0)

        if sr >= threshold:
            print(f"[SkillRL Evolution] {domain}: SR={sr:.2f} >= {threshold}, skipping")
            continue

        print(f"[SkillRL Evolution] {domain}: SR={sr:.2f} < {threshold}, evolving...")

        # Load failed simulations from the latest eval output
        # The evaluation saves results that we can parse
        # For now, generate skills based on the low success rate signal
        # In a full implementation, we'd load the actual failed trajectories
        dummy_failures = [{"task_id": "?", "reward_info": {"reward": 0.0},
                           "termination_reason": "low_accuracy",
                           "messages": [{"role": "system", "content": f"Domain {domain} had SR={sr:.2f}"}]}]

        count, new_skills = evolve_skills(
            sb, dummy_failures, generate_fn,
            max_new=max_new, domain=domain,
        )
        if count > 0:
            print(f"[SkillRL Evolution] {domain}: Added {count} new skills (total: {sb.total_skills})")
        else:
            print(f"[SkillRL Evolution] {domain}: No new skills generated")


def main():
    """Entry point: patch game registry, then delegate to train_grpo_optimized.py."""
    # Parse our SkillRL-specific args before passing the rest to GRPO trainer
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skill-bank-dir", type=str,
                        default=str(Path(__file__).parent / "skillrl" / "data"),
                        help="Directory containing *_skills.json files")
    parser.add_argument("--skill-evolution", action="store_true", default=False,
                        help="Enable recursive skill evolution during training")
    parser.add_argument("--evolution-threshold", type=float, default=0.4,
                        help="Min accuracy to skip evolution")
    parser.add_argument("--evolution-max-new", type=int, default=3,
                        help="Max new skills per evolution step")
    parser.add_argument("--evolution-teacher-url", type=str, default=None,
                        help="vLLM URL for teacher model (default: same as VLLM_BASE_URLS)")
    parser.add_argument("--evolution-teacher-model", type=str, default=None,
                        help="Model name for evolution (default: VLLM_MODEL)")

    skillrl_args, remaining = parser.parse_known_args()

    # Load and inject SkillBanks
    print(f"[SkillRL] Loading skill banks from {skillrl_args.skill_bank_dir}")
    skillbanks = patch_game_registry_with_skills(skillrl_args.skill_bank_dir)

    # Set evolution config as environment variables so the GRPO trainer
    # can call our evolution hook after evaluation
    if skillrl_args.skill_evolution:
        teacher_url = (skillrl_args.evolution_teacher_url
                       or os.environ.get("VLLM_BASE_URLS", "http://localhost:8080/v1"))
        if not teacher_url.endswith("/v1"):
            teacher_url = teacher_url.rstrip("/") + "/v1"
        teacher_model = (skillrl_args.evolution_teacher_model
                         or os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"))

        # Store for use in the evaluation callback
        os.environ["SKILLRL_EVOLUTION"] = "1"
        os.environ["SKILLRL_TEACHER_URL"] = teacher_url
        os.environ["SKILLRL_TEACHER_MODEL"] = teacher_model
        os.environ["SKILLRL_THRESHOLD"] = str(skillrl_args.evolution_threshold)
        os.environ["SKILLRL_MAX_NEW"] = str(skillrl_args.evolution_max_new)
        os.environ["SKILLRL_BANK_DIR"] = skillrl_args.skill_bank_dir

        print(f"[SkillRL] Skill evolution enabled:")
        print(f"  Teacher URL: {teacher_url}")
        print(f"  Teacher model: {teacher_model}")
        print(f"  Threshold: {skillrl_args.evolution_threshold}")
        print(f"  Max new skills/step: {skillrl_args.evolution_max_new}")

    # Now delegate to train_grpo_optimized.py
    # We patch sys.argv to pass remaining args
    sys.argv = [sys.argv[0]] + remaining
    print(f"[SkillRL] Delegating to GRPO trainer with args: {remaining}")

    # Import and run
    import train_grpo_optimized
    # The GRPO trainer's main() will pick up the patched game registry


if __name__ == "__main__":
    main()
