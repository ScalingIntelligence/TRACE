#!/usr/bin/env python3
"""
SkyRL on-policy distillation for skill games with per-skill teacher routing.

Student model generates on-policy trajectories across all skill games.
Per-skill teacher models (LoRA adapters on an external vLLM server) grade
each token.  The per-token reverse KL drives a clipped surrogate loss:

    reward_t = -(log pi_student(y_t | ...) - log pi_teacher(y_t | ...))

The teacher vLLM server hosts the base model with ALL skill LoRA adapters
loaded simultaneously (vLLM multi-LoRA).  Each sample is routed to its
skill's teacher LoRA for logprob computation.

Architecture:
    ┌────────────────────────────────────────────────────┐
    │ SkyRL Training (GPUs 0..N-1)                       │
    │  - Student policy (LoRA, FSDP2)                    │
    │  - vLLM inference engine (generation)              │
    │  - NO ref model needed (saves GPU memory)          │
    └──────────────┬─────────────────────────────────────┘
                   │  HTTP queries for teacher logprobs
    ┌──────────────▼─────────────────────────────────────┐
    │ Teacher vLLM Server (1 extra GPU)                  │
    │  - Base model + skill LoRA adapters                │
    │  - /v1/completions with echo=True, logprobs=1      │
    │  - Routes to correct LoRA per skill                │
    └────────────────────────────────────────────────────┘

Usage:
    # 1. Start teacher vLLM server with multi-LoRA:
    #    python -m vllm.entrypoints.openai.api_server \\
    #      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
    #      --enable-lora --max-loras 5 \\
    #      --lora-modules skill_a=/path/to/adapter_a skill_b=/path/to/adapter_b \\
    #      --port 9100

    # 2. Run distillation:
    #    bash run_skill_skyrl_distill.sh
"""

import concurrent.futures
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_GAMES_DIR = os.path.dirname(os.path.abspath(__file__))
if _GAMES_DIR not in sys.path:
    sys.path.insert(0, _GAMES_DIR)

import ray
import requests
import torch
from loguru import logger

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import register_advantage_estimator
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register


# ============================================================================
# Configuration for per-skill teacher routing
# ============================================================================

def parse_teacher_adapters(spec: str) -> Dict[str, str]:
    """Parse teacher adapter specification.

    Format: "skill_name=lora_name,skill_name2=lora_name2"
    Example: "structured_data_reasoning=sdr_teacher,multistep_task=mt_teacher"

    The lora_name must match the name used when launching the vLLM server
    with --lora-modules.

    If a single name is given without '=', it's used as the default for all
    skills (single-teacher mode).
    """
    if not spec:
        return {}
    mapping = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if "=" in entry:
            skill, lora = entry.split("=", 1)
            mapping[skill.strip()] = lora.strip()
        else:
            # Single name → default teacher for all skills
            mapping["__default__"] = entry.strip()
    return mapping


def get_teacher_lora_for_skill(
    skill_name: str,
    teacher_adapters: Dict[str, str],
) -> Optional[str]:
    """Get the teacher LoRA name for a given skill.

    Returns None if the teacher should use the base model (no LoRA).
    """
    if skill_name in teacher_adapters:
        return teacher_adapters[skill_name]
    return teacher_adapters.get("__default__")


# ============================================================================
# External vLLM teacher logprob query
# ============================================================================

def query_teacher_logprobs_vllm(
    teacher_url: str,
    teacher_model: str,
    teacher_lora_name: Optional[str],
    token_ids_list: List[List[int]],
    prompt_lens: List[int],
    response_lens: List[int],
    concurrency: int = 16,
    timeout: int = 120,
) -> List[torch.Tensor]:
    """Query a vLLM teacher server for per-token response logprobs.

    Sends full sequences (prompt + response) to /v1/completions with
    echo=True to get per-token logprobs, then extracts the response region.

    Args:
        teacher_url: Base URL of the teacher vLLM server.
        teacher_model: Model name to use (with LoRA, use the LoRA name).
        teacher_lora_name: LoRA adapter name on the teacher server, or None
            to use the base model.
        token_ids_list: List of full token ID sequences (prompt + response).
        prompt_lens: Length of prompt portion for each sequence.
        response_lens: Length of response portion for each sequence.
        concurrency: Max parallel HTTP requests.
        timeout: Per-request timeout in seconds.

    Returns:
        List of 1-D float tensors, each of length response_lens[i],
        containing per-token teacher logprobs for the response portion.
    """
    N = len(token_ids_list)
    results: List[Optional[torch.Tensor]] = [None] * N

    # Use the LoRA name as the model identifier if provided
    model_name = teacher_lora_name if teacher_lora_name else teacher_model

    def fetch_one(idx: int) -> Tuple[int, torch.Tensor]:
        token_ids = token_ids_list[idx]
        resp = requests.post(
            f"{teacher_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": token_ids,
                "max_tokens": 1,
                "echo": True,
                "logprobs": 1,
                "temperature": 1.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        token_logprobs = data["choices"][0]["logprobs"]["token_logprobs"]

        pl = prompt_lens[idx]
        rl = response_lens[idx]
        response_lps = []
        for t in range(pl, pl + rl):
            if t < len(token_logprobs) and token_logprobs[t] is not None:
                response_lps.append(token_logprobs[t])
            else:
                response_lps.append(0.0)

        return idx, torch.tensor(response_lps, dtype=torch.float32)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(fetch_one, i): i for i in range(N)}
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, lps = future.result()
                results[idx] = lps
            except Exception as e:
                orig_idx = futures[future]
                logger.warning(f"[Teacher vLLM] Error for sample {orig_idx}: {e}")
                # Fall back to zeros (base model equivalent)
                results[orig_idx] = torch.zeros(response_lens[orig_idx], dtype=torch.float32)

    return results


def query_teacher_logprobs_batched(
    teacher_url: str,
    teacher_model: str,
    teacher_adapters: Dict[str, str],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    skill_labels: List[str],
    concurrency: int = 16,
    timeout: int = 120,
) -> torch.Tensor:
    """Compute teacher logprobs for a batch, routing each sample to its
    skill-specific teacher LoRA.

    Groups samples by skill, queries the teacher vLLM server with the
    appropriate LoRA for each group, and assembles the full logprob tensor.

    Args:
        teacher_url: Base URL of the teacher vLLM server.
        teacher_model: Base model name on the teacher server.
        teacher_adapters: Mapping from skill name to LoRA adapter name.
        sequences: [batch, seqlen] token IDs (prompt + response, left-padded).
        attention_mask: [batch, seqlen] attention mask (0=pad, 1=real token).
        response_mask: [batch, padded_response_len] response token mask
            (1=actual response token, 0=padding). Left-aligned: [1,1,1,0,0].
        loss_mask: [batch, padded_response_len] loss mask (same shape as
            response_mask, subset of response tokens to train on).
        skill_labels: Per-sample skill/game name.
        concurrency: Max parallel HTTP requests per skill group.
        timeout: Per-request timeout in seconds.

    Returns:
        [batch, padded_response_len] tensor of teacher logprobs, aligned
        with response_mask positions (zeros where response_mask is 0).
    """
    batch_size = sequences.shape[0]
    seqlen = sequences.shape[1]

    # Find the response region dimensions
    # response_mask is [batch, seqlen], find the padded response length
    # In SkyRL, sequences = [prompt | response] with left-padding
    # response_mask marks the response tokens
    response_length = response_mask.shape[1]

    # Initialize output tensor (same shape as loss_mask for the response region)
    teacher_logprobs = torch.zeros(batch_size, response_length, dtype=torch.float32)

    # Group sample indices by skill
    skill_groups: Dict[str, List[int]] = {}
    for i, skill in enumerate(skill_labels):
        skill_groups.setdefault(skill, []).append(i)

    for skill_name, indices in skill_groups.items():
        lora_name = get_teacher_lora_for_skill(skill_name, teacher_adapters)

        # Extract per-sample data for this skill group
        token_ids_list = []
        prompt_lens = []
        resp_lens = []

        for idx in indices:
            # Get actual (non-padded) tokens
            attn = attention_mask[idx]  # [seqlen]
            seq = sequences[idx]  # [seqlen]
            rmask = response_mask[idx]  # [response_length]

            # Find actual sequence start (skip left padding)
            actual_start = attn.argmax().item()  # first 1 in attention_mask
            actual_seq = seq[actual_start:].tolist()

            # Prompt length = total actual tokens - response tokens
            resp_len = int(rmask.sum().item())
            prompt_len = len(actual_seq) - resp_len

            token_ids_list.append(actual_seq)
            prompt_lens.append(prompt_len)
            resp_lens.append(resp_len)

        # Query teacher for this skill group
        skill_logprobs = query_teacher_logprobs_vllm(
            teacher_url=teacher_url,
            teacher_model=teacher_model,
            teacher_lora_name=lora_name,
            token_ids_list=token_ids_list,
            prompt_lens=prompt_lens,
            response_lens=resp_lens,
            concurrency=concurrency,
            timeout=timeout,
        )

        # Place logprobs into the output tensor (right-aligned to match response_mask)
        for local_j, global_j in enumerate(indices):
            lps = skill_logprobs[local_j]
            rl = resp_lens[local_j]
            rmask = response_mask[global_j]

            # Find where response tokens are in the response_mask
            # response_mask is right-aligned: [..., 0, 0, 1, 1, 1]
            resp_positions = rmask.nonzero(as_tuple=True)[0]
            if len(resp_positions) > 0 and rl > 0:
                # Fill the response positions with teacher logprobs
                n_fill = min(rl, len(resp_positions))
                teacher_logprobs[global_j, resp_positions[:n_fill]] = lps[:n_fill]

    return teacher_logprobs


# ============================================================================
# Custom trainer for on-policy distillation
# ============================================================================

class SkillDistillationTrainer(RayPPOTrainer):
    """On-policy distillation trainer with per-skill teacher routing.

    Key differences from standard RayPPOTrainer:
    1. No ref model (use_kl_loss=False, use_kl_in_reward=False → no ref created)
    2. Overrides fwd_logprobs_values_reward to:
       a. Compute student logprobs via policy forward
       b. Query external vLLM teacher for per-skill teacher logprobs
       c. Set rewards = per-token reverse KL (overwrites task rewards)
    3. Uses no_op advantage estimator (pass-through token-level rewards)
    4. Uses importance_sampling policy loss

    Config requirements:
        trainer.algorithm.use_kl_loss=false
        trainer.algorithm.use_kl_in_reward=false
        trainer.algorithm.advantage_estimator=no_op
        trainer.algorithm.policy_loss_type=importance_sampling

    Environment variables:
        TEACHER_VLLM_URL: Base URL of the teacher vLLM server (default: http://localhost:9100)
        TEACHER_MODEL: Base model name on teacher server
        TEACHER_ADAPTERS: Skill→LoRA mapping (e.g. "skill_a=lora_a,skill_b=lora_b")
        TEACHER_CONCURRENCY: Max parallel HTTP requests (default: 32)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Teacher configuration from environment variables
        self.teacher_url = os.environ.get("TEACHER_VLLM_URL", "http://localhost:9100")
        # Strip trailing /v1 if present (we add /v1 in query URLs)
        if self.teacher_url.endswith("/v1"):
            self.teacher_url = self.teacher_url[:-3]
        self.teacher_model = os.environ.get(
            "TEACHER_MODEL",
            self.cfg.trainer.policy.model.path,
        )
        self.teacher_adapters = parse_teacher_adapters(
            os.environ.get("TEACHER_ADAPTERS", "")
        )
        self.teacher_concurrency = int(os.environ.get("TEACHER_CONCURRENCY", "32"))

        logger.info(f"[Distill] Teacher URL: {self.teacher_url}")
        logger.info(f"[Distill] Teacher model: {self.teacher_model}")
        logger.info(f"[Distill] Teacher adapters: {self.teacher_adapters}")
        logger.info(f"[Distill] Teacher concurrency: {self.teacher_concurrency}")

    @torch.no_grad()
    def fwd_logprobs_values_reward(
        self,
        training_input: TrainingInputBatch,
    ):
        """Compute student logprobs and teacher logprobs, set rewards = reverse KL.

        Replaces the standard flow (which computes ref model logprobs) with:
        1. Policy forward → student logprobs (action_log_probs)
        2. External vLLM query → per-skill teacher logprobs
        3. rewards = -(student_lp - teacher_lp) * loss_mask

        The task rewards from the generator are overwritten with the KL rewards.
        Original task rewards are saved in metadata for logging.
        """
        # --- 1. Policy forward (student logprobs) ---
        fwd_keys = ["sequences", "attention_mask"]
        if training_input.get("rollout_expert_indices") is not None:
            fwd_keys.append("rollout_expert_indices")
        data_fwd_pass = training_input.select(
            keys=fwd_keys, metadata_keys=["response_length"]
        )

        policy_output = self.dispatch.forward("policy", data_fwd_pass)
        action_log_probs = policy_output["output"]

        self.dispatch.empty_cache()

        sequences_all = training_input["sequences"]
        action_log_probs = action_log_probs[: len(sequences_all)]
        training_input["action_log_probs"] = action_log_probs

        # Logprobs diff tracking (same as parent)
        if training_input.get("rollout_logprobs", None) is not None:
            logprobs_diff = (
                training_input["rollout_logprobs"][training_input["loss_mask"] > 0]
                - action_log_probs[training_input["loss_mask"] > 0]
            ).abs()
            self.all_metrics.update({
                "policy/rollout_train_logprobs_abs_diff_mean": logprobs_diff.mean().item(),
                "policy/rollout_train_logprobs_abs_diff_std": logprobs_diff.std().item(),
            })

        # --- 2. Extract skill labels from batch metadata ---
        uids = training_input.metadata.get("uids", [])
        skill_labels = self._extract_skill_labels(training_input, uids)

        # --- 3. Teacher logprobs via external vLLM ---
        teacher_logprobs = query_teacher_logprobs_batched(
            teacher_url=self.teacher_url,
            teacher_model=self.teacher_model,
            teacher_adapters=self.teacher_adapters,
            sequences=training_input["sequences"],
            attention_mask=training_input["attention_mask"],
            response_mask=training_input["response_mask"],
            loss_mask=training_input["loss_mask"],
            skill_labels=skill_labels,
            concurrency=self.teacher_concurrency,
        )

        # Store teacher logprobs (same shape as action_log_probs: [batch, response_len])
        training_input["base_action_log_probs"] = teacher_logprobs.to(
            action_log_probs.device
        )

        # --- 4. Save original task rewards, then overwrite with KL rewards ---
        original_task_rewards = training_input["rewards"].clone()

        loss_masks = training_input["loss_mask"]
        student_lp = action_log_probs
        teacher_lp = training_input["base_action_log_probs"]

        # Per-token reverse KL: -(student - teacher) = teacher - student
        # Positive reward when teacher assigns higher prob → student should learn
        rewards = -(student_lp - teacher_lp) * loss_masks
        training_input["rewards"] = rewards

        # Log metrics
        with torch.no_grad():
            valid_mask = loss_masks.bool()
            if valid_mask.any():
                gap = teacher_lp[valid_mask] - student_lp[valid_mask]
                self.all_metrics.update({
                    "distill/teacher_student_gap_mean": gap.mean().item(),
                    "distill/teacher_student_gap_std": gap.std().item(),
                    "distill/kl_reward_mean": rewards[valid_mask].mean().item(),
                })
            # Log original task reward (sum per sequence, then average)
            task_reward_sums = original_task_rewards.sum(dim=-1)
            self.all_metrics["distill/task_reward_mean"] = task_reward_sums.mean().item()

        # No critic values needed for distillation
        training_input["values"] = None

        return training_input

    def _extract_skill_labels(
        self,
        training_input: TrainingInputBatch,
        uids: List[str],
    ) -> List[str]:
        """Extract per-sample skill labels from training batch.

        Checks metadata sources in order:
        1. training_input.metadata["skill_labels"] (if propagated)
        2. training_input.metadata["env_metrics"] (from SkillGameEnv)
        3. Falls back to "__default__" for single-teacher mode
        """
        batch_size = len(training_input["sequences"])

        if "skill_labels" in training_input.metadata:
            return training_input.metadata["skill_labels"]

        if "env_metrics" in training_input.metadata:
            env_metrics = training_input.metadata["env_metrics"]
            if isinstance(env_metrics, list) and len(env_metrics) == batch_size:
                return [m.get("game_name", "unknown") for m in env_metrics]

        if self.teacher_adapters and "__default__" in self.teacher_adapters:
            return ["__default__"] * batch_size

        logger.warning(
            f"[Distill] Could not extract skill labels for batch of {batch_size}. "
            f"Using '__default__' teacher for all samples."
        )
        return ["__default__"] * batch_size


# ============================================================================
# No-op advantage estimator: pass through token-level rewards
# ============================================================================

@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    """Pass through token-level rewards directly as advantages and returns.

    For on-policy distillation, the per-token reverse KL (set as rewards
    in fwd_logprobs_values_reward) IS the advantage signal.
    """
    return token_level_rewards, token_level_rewards


# ============================================================================
# Experiment and entrypoint
# ============================================================================

class SkillDistillationExp(BasePPOExp):
    """Experiment class that uses our custom distillation trainer."""

    def get_trainer(self, *args, **kwargs):
        return SkillDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    """Ray task that registers skill game envs and runs distillation training."""
    games_dir = os.path.dirname(os.path.abspath(__file__))
    if games_dir not in sys.path:
        sys.path.insert(0, games_dir)

    register(
        id="skill_game",
        entry_point="skill_game_env:SkillGameEnv",
    )

    exp = SkillDistillationExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
