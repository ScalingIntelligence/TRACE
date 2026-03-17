"""
Gumbel Top-K sampling and soft target reconstruction for offline distillation.

Reference: "Categorical Reparameterization with Gumbel-Softmax"
           (Jang, Gu, Poole — 1611.01144)

Usage:
  1. Preprocessing: run teacher model, call sample_gumbel_topk on log-probs,
     save (tokens, log_probs, thresholds) per action-token position.
  2. Training: call soft_targets_from_gumbel_topk to reconstruct dense soft
     targets, then compute cross-entropy distillation loss.
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def sample_gumbel_topk(
    log_prob_XV: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add gumbel noise, then sample top k tokens.

    Args:
        log_prob_XV: [..., vocab_size] log probabilities from the teacher model
        k: number of tokens to sample

    Returns:
        token_Xk:     [..., k] selected token indices
        log_prob_Xk:  [..., k] log probabilities of selected tokens
        threshold_Xk: [..., k] noised-up log-prob of the next-highest token
    """
    K = k + 1
    V = log_prob_XV.size(-1)
    assert K <= V
    noise_XV = -torch.empty_like(log_prob_XV).exponential_().log_()
    noisy_log_prob_XV = noise_XV  # reuse buffer
    noisy_log_prob_XV.add_(log_prob_XV)
    noisy_log_prob_XK, token_XK = torch.topk(noisy_log_prob_XV, K, dim=-1)
    log_prob_XK = torch.gather(log_prob_XV, -1, token_XK)
    token_Xk = token_XK[..., :k].contiguous()
    log_prob_Xk = log_prob_XK[..., :k].contiguous()
    threshold_Xk = noisy_log_prob_XK[..., 1:].contiguous()
    return token_Xk, log_prob_Xk, threshold_Xk


def soft_targets_from_gumbel_topk(
    token_XK: torch.Tensor,
    log_prob_XK: torch.Tensor,
    threshold_XK: torch.Tensor,
    k: int,
    V: int,
    normalize: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct dense soft targets from gumbel topk samples.

    Args:
        token_XK:     [..., K] token indices (K >= k)
        log_prob_XK:  [..., K] teacher log-probs for those tokens
        threshold_XK: [..., K] noised-up log-prob thresholds
        k: number of tokens to use (k <= K)
        V: vocabulary size
        normalize: if True, normalize weights to sum to 1
        dtype: output dtype

    Returns:
        soft_targets_XV: [..., V] dense soft target distribution
    """
    K = log_prob_XK.size(-1)
    assert k <= K
    assert k > 0
    threshold_X1 = threshold_XK[..., k - 1].unsqueeze(-1)
    token_Xk = token_XK[..., :k]
    log_prob_Xk = log_prob_XK[..., :k]
    min_noise_Xk = threshold_X1 - log_prob_Xk
    # Increase precision to avoid NaN
    max_exponential_Xk = (-min_noise_Xk).double().exp()
    log_inclusion_prob_Xk = torch.log1p(-(-max_exponential_Xk).exp())
    weight_Xk = torch.exp(log_prob_Xk - log_inclusion_prob_Xk)
    if normalize:
        weight_Xk = weight_Xk / weight_Xk.sum(dim=-1, keepdim=True)
    weight_Xk = weight_Xk.to(dtype)
    soft_targets_XV = torch.zeros(
        token_XK.size()[:-1] + (V,), device=token_XK.device, dtype=dtype
    )
    soft_targets_XV.scatter_add_(-1, token_Xk.long(), weight_Xk)
    return soft_targets_XV


def gumbel_kl_loss(
    policy_logits: torch.Tensor,
    gumbel_tokens: torch.Tensor,
    gumbel_log_probs: torch.Tensor,
    gumbel_thresholds: torch.Tensor,
    k: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute KL distillation loss from precomputed Gumbel top-k data.

    Reconstructs soft targets from teacher's Gumbel samples, then computes:
        loss = -sum_v T(v) * log pi(v)   (cross-entropy with soft targets)

    This avoids materializing the full [B, T, V] soft target tensor by
    working only with the k nonzero entries per position.

    Args:
        policy_logits:     [N, V] student/policy logits at action positions
        gumbel_tokens:     [N, k] or [N, K>=k] token indices from teacher
        gumbel_log_probs:  [N, k] or [N, K>=k] teacher log-probs
        gumbel_thresholds: [N, k] or [N, K>=k] noised-up thresholds
        k: number of gumbel tokens to use
        normalize: whether to normalize soft target weights

    Returns:
        Scalar loss (mean over N positions)
    """
    V = policy_logits.size(-1)
    N = policy_logits.size(0)

    if N == 0:
        return torch.tensor(0.0, device=policy_logits.device)

    # Reconstruct importance weights for the k tokens
    threshold_1 = gumbel_thresholds[..., k - 1].unsqueeze(-1)  # [N, 1]
    token_k = gumbel_tokens[..., :k]                            # [N, k]
    log_prob_k = gumbel_log_probs[..., :k]                      # [N, k]

    min_noise_k = threshold_1 - log_prob_k
    max_exp_k = (-min_noise_k).double().exp()
    log_incl_k = torch.log1p(-(-max_exp_k).exp())
    weight_k = torch.exp(log_prob_k - log_incl_k).float()  # [N, k]

    if normalize:
        weight_k = weight_k / weight_k.sum(dim=-1, keepdim=True)

    # Gather policy log-probs at the k token positions
    policy_log_probs = F.log_softmax(policy_logits.float(), dim=-1)  # [N, V]
    policy_lp_k = policy_log_probs.gather(1, token_k.long())         # [N, k]

    # Cross-entropy: -sum_k w_k * log pi(k)
    ce = -(weight_k * policy_lp_k).sum(dim=-1)  # [N]
    return ce.mean()
