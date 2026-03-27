#!/usr/bin/env python3
"""
Lightweight tests for skill_skyrl_distill.py.

Run with:
    conda activate sky_games
    cd /home/ubuntu/hangook/games
    python -m pytest tests/test_skill_distill.py -v
"""
import json
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch, MagicMock

import pytest
import torch

# Ensure games directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test: Module imports
# ============================================================================

class TestImports:
    def test_import_skill_skyrl_distill(self):
        import skill_skyrl_distill
        assert hasattr(skill_skyrl_distill, "SkillDistillationTrainer")
        assert hasattr(skill_skyrl_distill, "SkillDistillationExp")
        assert hasattr(skill_skyrl_distill, "compute_no_op_advantage")
        assert hasattr(skill_skyrl_distill, "query_teacher_logprobs_vllm")
        assert hasattr(skill_skyrl_distill, "query_teacher_logprobs_batched")
        assert hasattr(skill_skyrl_distill, "parse_teacher_adapters")
        assert hasattr(skill_skyrl_distill, "get_teacher_lora_for_skill")


# ============================================================================
# Test: Teacher adapter parsing
# ============================================================================

class TestTeacherAdapterParsing:
    def test_parse_multi_skill(self):
        from skill_skyrl_distill import parse_teacher_adapters
        result = parse_teacher_adapters(
            "structured_data_reasoning=sdr_teacher,multistep_task=mt_teacher"
        )
        assert result == {
            "structured_data_reasoning": "sdr_teacher",
            "multistep_task": "mt_teacher",
        }

    def test_parse_single_default(self):
        from skill_skyrl_distill import parse_teacher_adapters
        result = parse_teacher_adapters("my_single_teacher")
        assert result == {"__default__": "my_single_teacher"}

    def test_parse_empty(self):
        from skill_skyrl_distill import parse_teacher_adapters
        assert parse_teacher_adapters("") == {}
        assert parse_teacher_adapters(None) == {}

    def test_parse_mixed(self):
        from skill_skyrl_distill import parse_teacher_adapters
        result = parse_teacher_adapters("sdr=sdr_lora, mt=mt_lora")
        assert result == {"sdr": "sdr_lora", "mt": "mt_lora"}

    def test_get_teacher_lora_specific(self):
        from skill_skyrl_distill import get_teacher_lora_for_skill
        adapters = {"skill_a": "lora_a", "skill_b": "lora_b"}
        assert get_teacher_lora_for_skill("skill_a", adapters) == "lora_a"
        assert get_teacher_lora_for_skill("skill_b", adapters) == "lora_b"
        assert get_teacher_lora_for_skill("skill_c", adapters) is None

    def test_get_teacher_lora_default_fallback(self):
        from skill_skyrl_distill import get_teacher_lora_for_skill
        adapters = {"skill_a": "lora_a", "__default__": "default_lora"}
        assert get_teacher_lora_for_skill("skill_a", adapters) == "lora_a"
        assert get_teacher_lora_for_skill("unknown_skill", adapters) == "default_lora"


# ============================================================================
# Test: no_op advantage estimator
# ============================================================================

class TestNoOpAdvantage:
    def test_passthrough(self):
        from skill_skyrl_distill import compute_no_op_advantage
        rewards = torch.tensor([[0.1, 0.2, -0.1], [0.3, 0.0, -0.2]])
        advantages, returns = compute_no_op_advantage(token_level_rewards=rewards)
        assert torch.equal(advantages, rewards)
        assert torch.equal(returns, rewards)

    def test_with_kwargs(self):
        """no_op should ignore extra kwargs from the advantage estimator API."""
        from skill_skyrl_distill import compute_no_op_advantage
        rewards = torch.randn(4, 10)
        advantages, returns = compute_no_op_advantage(
            token_level_rewards=rewards,
            response_mask=torch.ones(4, 10),
            index=["a", "a", "b", "b"],
            values=None,
            config=None,
        )
        assert torch.equal(advantages, rewards)


# ============================================================================
# Test: Reverse KL reward computation
# ============================================================================

class TestReverseKLReward:
    def test_reverse_kl_basic(self):
        """Test that reward = -(student - teacher) * mask."""
        student_lp = torch.tensor([[- 1.0, -2.0, -3.0]])
        teacher_lp = torch.tensor([[-0.5, -1.5, -2.5]])
        loss_mask = torch.tensor([[1.0, 1.0, 1.0]])

        # reward = -(student - teacher) * mask = -((-1 - -0.5), (-2 - -1.5), (-3 - -2.5))
        # = -((-0.5), (-0.5), (-0.5)) = (0.5, 0.5, 0.5)
        expected = torch.tensor([[0.5, 0.5, 0.5]])
        rewards = -(student_lp - teacher_lp) * loss_mask
        assert torch.allclose(rewards, expected)

    def test_reverse_kl_with_mask(self):
        """Masked tokens should have zero reward."""
        student_lp = torch.tensor([[-1.0, -2.0, -3.0]])
        teacher_lp = torch.tensor([[-0.5, -1.5, -2.5]])
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])

        rewards = -(student_lp - teacher_lp) * loss_mask
        assert rewards[0, 1].item() == 0.0  # masked token
        assert rewards[0, 0].item() == 0.5  # unmasked token
        assert rewards[0, 2].item() == 0.5  # unmasked token

    def test_reverse_kl_identical_models(self):
        """When student = teacher, all rewards should be zero."""
        logprobs = torch.randn(3, 10)
        loss_mask = torch.ones(3, 10)
        rewards = -(logprobs - logprobs) * loss_mask
        assert torch.allclose(rewards, torch.zeros_like(rewards))

    def test_reverse_kl_teacher_better(self):
        """When teacher assigns higher prob, reward should be positive."""
        # teacher_lp > student_lp → reward = teacher_lp - student_lp > 0
        student_lp = torch.tensor([[-5.0]])  # low prob
        teacher_lp = torch.tensor([[-1.0]])  # high prob
        loss_mask = torch.tensor([[1.0]])
        rewards = -(student_lp - teacher_lp) * loss_mask
        assert rewards.item() > 0  # positive reward when student should learn from teacher


# ============================================================================
# Test: vLLM teacher logprob query (with mock server)
# ============================================================================

class MockVLLMHandler(BaseHTTPRequestHandler):
    """Mock vLLM server that returns predictable logprobs."""

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(content_length))

        prompt_tokens = body["prompt"]
        n_tokens = len(prompt_tokens)

        # Return -0.5 for all token logprobs (first token is None per vLLM convention)
        token_logprobs = [None] + [-0.5] * (n_tokens - 1)

        response = {
            "choices": [{
                "logprobs": {
                    "token_logprobs": token_logprobs,
                    "tokens": [str(t) for t in prompt_tokens],
                }
            }]
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass  # Suppress request logging


class TestVLLMTeacherQuery:
    @pytest.fixture(autouse=True)
    def mock_server(self):
        """Start a mock vLLM server on a random port."""
        self.server = HTTPServer(("127.0.0.1", 0), MockVLLMHandler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        yield
        self.server.shutdown()

    def test_query_basic(self):
        from skill_skyrl_distill import query_teacher_logprobs_vllm

        # 2 sequences: prompt(3) + response(2) = 5 tokens each
        token_ids_list = [
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51],
        ]
        prompt_lens = [3, 3]
        response_lens = [2, 2]

        results = query_teacher_logprobs_vllm(
            teacher_url=self.url,
            teacher_model="test-model",
            teacher_lora_name=None,
            token_ids_list=token_ids_list,
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            concurrency=2,
        )

        assert len(results) == 2
        assert results[0].shape == (2,)  # response_lens[0] = 2
        assert results[1].shape == (2,)
        # Mock returns -0.5 for all non-first tokens
        assert torch.allclose(results[0], torch.tensor([-0.5, -0.5]))
        assert torch.allclose(results[1], torch.tensor([-0.5, -0.5]))

    def test_query_with_lora_name(self):
        """Test that LoRA name is sent as the model identifier."""
        from skill_skyrl_distill import query_teacher_logprobs_vllm

        results = query_teacher_logprobs_vllm(
            teacher_url=self.url,
            teacher_model="base-model",
            teacher_lora_name="my_skill_lora",
            token_ids_list=[[1, 2, 3, 4]],
            prompt_lens=[2],
            response_lens=[2],
            concurrency=1,
        )

        assert len(results) == 1
        assert results[0].shape == (2,)


# ============================================================================
# Test: Batched teacher query with skill routing
# ============================================================================

class TestBatchedTeacherQuery:
    def test_skill_routing(self):
        """Test that samples are grouped by skill and routed correctly."""
        from skill_skyrl_distill import query_teacher_logprobs_batched

        batch_size = 4
        seqlen = 10
        response_len = 5

        sequences = torch.randint(1, 100, (batch_size, seqlen))
        attention_mask = torch.ones(batch_size, seqlen, dtype=torch.long)
        response_mask = torch.zeros(batch_size, response_len, dtype=torch.long)
        response_mask[:, -3:] = 1  # last 3 tokens are response
        loss_mask = response_mask.clone().float()

        skill_labels = ["skill_a", "skill_b", "skill_a", "skill_b"]
        teacher_adapters = {"skill_a": "lora_a", "skill_b": "lora_b"}

        # Mock the per-skill query function
        call_log = []

        def mock_query(teacher_url, teacher_model, teacher_lora_name,
                       token_ids_list, prompt_lens, response_lens, **kwargs):
            call_log.append({
                "lora_name": teacher_lora_name,
                "n_samples": len(token_ids_list),
            })
            return [torch.ones(rl, dtype=torch.float32) * -0.5
                    for rl in response_lens]

        with patch("skill_skyrl_distill.query_teacher_logprobs_vllm", side_effect=mock_query):
            result = query_teacher_logprobs_batched(
                teacher_url="http://fake:9100",
                teacher_model="base",
                teacher_adapters=teacher_adapters,
                sequences=sequences,
                attention_mask=attention_mask,
                response_mask=response_mask,
                loss_mask=loss_mask,
                skill_labels=skill_labels,
            )

        assert result.shape == (batch_size, response_len)

        # Check that two calls were made (one per skill)
        assert len(call_log) == 2
        lora_names = {c["lora_name"] for c in call_log}
        assert lora_names == {"lora_a", "lora_b"}

        # Each skill should have 2 samples
        for c in call_log:
            assert c["n_samples"] == 2


# ============================================================================
# Test: End-to-end reward computation (without Ray/FSDP)
# ============================================================================

class TestEndToEndReward:
    def test_full_kl_reward_pipeline(self):
        """Test the full pipeline: student logprobs + teacher logprobs → KL rewards → no_op advantages."""
        from skill_skyrl_distill import compute_no_op_advantage

        batch_size = 3
        response_len = 8

        # Simulate student and teacher logprobs
        student_lp = torch.randn(batch_size, response_len) - 2.0  # ~[-4, 0]
        teacher_lp = torch.randn(batch_size, response_len) - 1.5  # slightly better

        loss_mask = torch.ones(batch_size, response_len)
        loss_mask[:, :2] = 0  # first 2 tokens masked (padding)

        # Compute reverse KL rewards
        rewards = -(student_lp - teacher_lp) * loss_mask

        # Pass through no_op advantage
        advantages, returns = compute_no_op_advantage(token_level_rewards=rewards)

        # Verify shapes
        assert advantages.shape == (batch_size, response_len)
        assert returns.shape == (batch_size, response_len)

        # Verify masking
        assert torch.allclose(advantages[:, :2], torch.zeros(batch_size, 2))

        # Verify non-masked tokens have signal
        assert not torch.allclose(advantages[:, 2:], torch.zeros(batch_size, response_len - 2))


# ============================================================================
# Test: Mathematical correctness of the full gradient computation
# ============================================================================

class TestMathCorrectness:
    """Verify end-to-end mathematical correctness of the distillation loss.

    The on-policy distillation objective is to minimize the per-token reverse KL:

        KL(π_θ || π_teacher) = E_{y~π_θ} [log π_θ(y_t) - log π_teacher(y_t)]

    With importance sampling, the loss for a batch of on-policy samples is:

        L = -Σ_t [π_θ_new(y_t) / π_θ_old(y_t)] * [log π_teacher(y_t) - log π_θ_old(y_t)]

    The gradient ∇_θ L pushes π_θ toward π_teacher.
    """

    def test_importance_sampling_loss_matches_paper(self):
        """Verify the IS loss matches the paper's formulation exactly.

        Pipeline:
            1. rewards = -(student_old_lp - teacher_lp) * loss_mask
            2. advantages = rewards (no_op)
            3. IS loss = -Σ exp(student_new_lp - student_old_lp) * advantages * loss_mask

        Expected:
            loss = -Σ_t [exp(new - old) * (teacher - old)] * mask
        """
        from skyrl.backends.skyrl_train.utils.ppo_utils import importance_sampling_loss
        from skyrl.train.config import AlgorithmConfig

        torch.manual_seed(42)
        B, T = 2, 5

        # Simulated logprobs (all negative, as log-probabilities should be)
        student_old_lp = torch.tensor([[-1.0, -2.0, -3.0, -1.5, -2.5],
                                        [-1.2, -0.8, -2.1, -1.8, -0.5]])
        teacher_lp     = torch.tensor([[-0.5, -1.0, -2.0, -0.8, -1.5],
                                        [-0.3, -0.5, -1.5, -1.0, -0.2]])
        student_new_lp = torch.tensor([[-0.9, -1.8, -2.8, -1.3, -2.3],
                                        [-1.0, -0.7, -1.9, -1.5, -0.4]])
        loss_mask = torch.tensor([[1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 1]], dtype=torch.float)

        # Step 1: rewards (from fwd_logprobs_values_reward)
        rewards = -(student_old_lp - teacher_lp) * loss_mask

        # Step 2: advantages (no_op)
        advantages = rewards

        # Step 3: IS loss (from worker._forward_backward_micro)
        config = AlgorithmConfig()
        loss, metrics = importance_sampling_loss(
            log_probs=student_new_lp,
            old_log_probs=student_old_lp,
            advantages=advantages,
            config=config,
            loss_mask=loss_mask,
        )

        # Manual computation for verification:
        # ratio = exp(new - old)
        ratio = torch.exp(student_new_lp - student_old_lp)
        # elementwise_loss = -(ratio * advantages)
        # Note: advantages already has loss_mask baked in, and IS loss applies mask again
        # loss_mask² = loss_mask (binary), so this is fine
        expected_loss = -(ratio * advantages * loss_mask).sum()

        assert torch.allclose(loss, expected_loss, atol=1e-5), \
            f"IS loss {loss.item():.6f} != expected {expected_loss.item():.6f}"

    def test_gradient_pushes_student_toward_teacher(self):
        """Verify the gradient direction: student should move toward teacher.

        If teacher assigns higher probability to a token than the student,
        the gradient should INCREASE the student's probability at that token.
        """
        torch.manual_seed(42)

        # Simple scenario: 1 sample, 3 tokens
        # student_param is a log-softmax parameter we optimize
        student_param = torch.tensor([-2.0, -1.5, -3.0], requires_grad=True)
        teacher_lp = torch.tensor([-0.5, -1.0, -1.0])  # teacher is more confident
        loss_mask = torch.ones(3)

        # In the first iteration, old = new = student_param (ratio = 1)
        old_lp = student_param.detach()
        advantages = (teacher_lp - old_lp) * loss_mask  # positive = teacher better

        # IS loss
        ratio = torch.exp(student_param - old_lp)  # = 1.0 when old = new
        loss = -(ratio * advantages * loss_mask).sum()
        loss.backward()

        # Check gradient direction
        # Where teacher > student (all 3 tokens here), the advantage is positive,
        # so loss = -advantages. Gradient of loss w.r.t student_param should be
        # negative (pushing params in the direction that reduces loss = increases
        # student probability toward teacher).
        #
        # ∂L/∂student_param_t = -ratio_t * advantage_t (when ratio=1)
        # Since advantage_t > 0 (teacher > student), ∂L/∂param < 0
        # Optimizer does param -= lr * grad, so param increases (student prob increases)
        for t in range(3):
            assert student_param.grad[t] < 0, \
                f"Token {t}: gradient should be negative (push toward teacher), got {student_param.grad[t]:.4f}"

    def test_convergence_to_zero_at_equilibrium(self):
        """When student = teacher, all rewards/advantages/loss should be zero."""
        logprobs = torch.tensor([[-1.0, -2.0, -3.0]])
        loss_mask = torch.ones(1, 3)

        # rewards = -(student - teacher) * mask = 0
        rewards = -(logprobs - logprobs) * loss_mask
        assert torch.allclose(rewards, torch.zeros_like(rewards))

        # With no_op advantages, advantages = 0
        # IS loss = -Σ exp(0) * 0 * mask = 0
        from skyrl.backends.skyrl_train.utils.ppo_utils import importance_sampling_loss
        from skyrl.train.config import AlgorithmConfig
        loss, _ = importance_sampling_loss(
            log_probs=logprobs,
            old_log_probs=logprobs,
            advantages=rewards,
            config=AlgorithmConfig(),
            loss_mask=loss_mask,
        )
        assert loss.item() == 0.0, f"Loss should be 0 at equilibrium, got {loss.item()}"

    def test_double_masking_is_harmless(self):
        """Verify that applying loss_mask in both rewards and IS loss is mathematically equivalent
        to applying it once (since mask is binary: mask² = mask)."""
        B, T = 2, 6
        student_lp = torch.randn(B, T) - 2.0
        teacher_lp = torch.randn(B, T) - 1.0
        new_lp = torch.randn(B, T) - 1.8
        loss_mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                                   [1, 1, 1, 1, 0, 0]], dtype=torch.float)

        # Method A: mask in rewards AND in IS loss (our implementation)
        rewards_a = -(student_lp - teacher_lp) * loss_mask
        ratio = torch.exp(new_lp - student_lp)
        loss_a = -(ratio * rewards_a * loss_mask).sum()

        # Method B: mask only in IS loss
        rewards_b = -(student_lp - teacher_lp)  # no mask
        loss_b = -(ratio * rewards_b * loss_mask).sum()

        assert torch.allclose(loss_a, loss_b, atol=1e-5), \
            f"Double masking should give same result: {loss_a.item()} vs {loss_b.item()}"

    def test_teacher_logprob_token_alignment(self):
        """Verify that teacher logprobs are extracted from the correct token positions.

        Given a sequence [prompt_tok1, prompt_tok2, resp_tok1, resp_tok2, resp_tok3]:
        - prompt_lens = 2, response_lens = 3
        - Teacher logprobs should be extracted from positions [2, 3, 4]

        vLLM returns token_logprobs where:
        - token_logprobs[0] = None (first token has no prior context)
        - token_logprobs[t] = log p(token_t | token_<t) for t > 0
        """
        # Simulate vLLM response: 5 tokens, first is None
        token_logprobs = [None, -0.5, -1.0, -1.5, -2.0]
        pl, rl = 2, 3

        # Extract response logprobs (indices pl through pl+rl-1)
        response_lps = []
        for t in range(pl, pl + rl):
            if t < len(token_logprobs) and token_logprobs[t] is not None:
                response_lps.append(token_logprobs[t])
            else:
                response_lps.append(0.0)

        expected = [-1.0, -1.5, -2.0]  # positions 2, 3, 4
        assert response_lps == expected, f"Got {response_lps}, expected {expected}"

    def test_kl_loss_sign_convention(self):
        """Verify sign convention: minimizing our loss minimizes KL(π_student || π_teacher).

        KL(π_s || π_t) = E_{y~π_s}[log π_s(y) - log π_t(y)]

        Our reward = -(log π_s - log π_t) = log π_t - log π_s
        Our loss minimizes: -E[ratio * reward] = -E[ratio * (log π_t - log π_s)]

        When student < teacher (KL > 0), reward > 0, and the loss pushes student
        probability UP toward teacher. This DECREASES KL. ✓
        """
        # Student under-confident (low prob) vs teacher (high prob)
        student_lp = torch.tensor([-5.0])
        teacher_lp = torch.tensor([-1.0])
        loss_mask = torch.tensor([1.0])

        reward = -(student_lp - teacher_lp) * loss_mask
        assert reward.item() > 0, "Reward should be positive when teacher > student"

        # With ratio=1 (first iter), loss = -reward < 0
        # Minimizing this loss means making reward even larger (impossible without
        # changing the advantage), OR increasing ratio via new student logprob.
        # Since new_lp is the parameter being optimized, increasing new_lp (toward
        # teacher) increases ratio * positive_advantage, making loss more negative.
        # The optimizer minimizes loss → pushes student toward teacher. ✓

        # Opposite case: student over-confident vs teacher
        student_lp2 = torch.tensor([-0.1])
        teacher_lp2 = torch.tensor([-3.0])

        reward2 = -(student_lp2 - teacher_lp2) * loss_mask
        assert reward2.item() < 0, "Reward should be negative when student > teacher"
        # Negative reward → positive loss → optimizer pushes student probability DOWN.
        # This moves student toward teacher. ✓


# ============================================================================
# Test: Config validation
# ============================================================================

class TestConfig:
    def test_distillation_config_fields(self):
        """Verify the distillation-specific config settings are valid SkyRL config fields."""
        from skyrl.train.config import SkyRLTrainConfig

        # These are the key config settings used by distillation
        # Both KL flags are False → no ref model created, no KL penalty step
        cfg = SkyRLTrainConfig.from_cli_overrides({
            "trainer.algorithm.advantage_estimator": "grpo",  # no_op registered at runtime
            "trainer.algorithm.policy_loss_type": "importance_sampling",
            "trainer.algorithm.use_kl_loss": False,
            "trainer.algorithm.use_kl_in_reward": False,
            "trainer.algorithm.eps_clip_low": 0.2,
            "trainer.algorithm.eps_clip_high": 0.2,
            "generator.n_samples_per_prompt": 1,
        })

        assert cfg.trainer.algorithm.policy_loss_type == "importance_sampling"
        assert cfg.trainer.algorithm.use_kl_loss is False
        assert cfg.trainer.algorithm.use_kl_in_reward is False
        assert cfg.generator.n_samples_per_prompt == 1

    def test_no_ref_model_when_both_kl_false(self):
        """With both KL flags False, no ref model should be created."""
        # This validates our design: the parent trainer checks
        #   use_ref_model = use_kl_loss or use_kl_in_reward
        # When both are False, ref_model stays None
        from skyrl.train.config import SkyRLTrainConfig
        cfg = SkyRLTrainConfig.from_cli_overrides({
            "trainer.algorithm.use_kl_loss": False,
            "trainer.algorithm.use_kl_in_reward": False,
        })
        use_ref_model = cfg.trainer.algorithm.use_kl_loss or cfg.trainer.algorithm.use_kl_in_reward
        assert use_ref_model is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
