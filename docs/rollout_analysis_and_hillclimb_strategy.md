# Rollout Analysis & Hill-Climbing Strategy

**Date**: 2026-02-13
**Rollouts file**: `rollouts_grpo_adversarial_policy_20260213_114558.jsonl`
**Model**: Qwen3-30B-A3B with LoRA adversarial policy training (late iteration)
**Context**: This analysis explains WHY the model lost points during training rollouts, connects those patterns to the tau2-bench regression observed in the transfer analysis, and recommends a hill-climbing strategy using an early checkpoint.

---

## 1. Executive Summary

The rollouts show **1,664 games across 4 groups** with mean reward **0.808** (80.8%). The model loses **320 points** (19.2%) from four distinct failure modes. The two most damaging patterns — **over-transfer** (27% of losses) and **missing communication** (38% of losses) — directly explain the tau2-bench regression. The over-transfer behavior trained here propagates to tau2-bench as premature escalation, while the missing communication reflects a model that optimizes for tool-calling correctness at the expense of user-facing explanation.

**Key recommendation**: Use the iter_1-5 checkpoint (which showed tau2-bench improvement) as a base, and train on a **procedural accuracy environment** targeting the "can you say yes correctly" gap — communication quality, multi-step verification workflows, and direct refusal without transfer.

---

## 2. Reward Distribution

| Reward | Games | % | Interpretation |
|---|---|---|---|
| 1.00 | 1,150 | 69.1% | Perfect — correct action, correct communication |
| 0.50 | 245 | 14.7% | Correct action but missing communication |
| 0.40 | 144 | 8.7% | Correct reasoning but transferred instead of refusing |
| 0.10 | 118 | 7.1% | Forbidden action taken OR lazy transfer on valid task |
| 0.00 | 2 | 0.1% | Total failure |
| Other | 5 | 0.3% | Edge cases (0.23, 0.30, 0.37, 0.60) |

---

## 3. Failure Mode Breakdown

### 3.1 Failure Mode 1: Missing Communication (122.5 pts lost, 38% of losses)

**245 games at reward 0.50** — the single largest source of lost reward.

These games follow the pattern: model looks up data (0.1) → performs the correct tool call (0.4) → but fails to communicate required info to the user (0.0 reasoning credit). Total: 0.5.

Breakdown:
- **118 baggage update games**: Model correctly calls `update_reservation_baggages` but doesn't communicate pricing details, free bag allowance, or confirmation of changes.
- **118 flight update games**: Model correctly calls `update_reservation_flights` but doesn't communicate the price difference, new flight details, or payment breakdown.
- **9 other** (cancel + partial)

**Example** (Game (2, 4002000), reward 0.50):
The model correctly adds 1 checked bag to a business class reservation with no charge. It even explains "you are entitled to 2 free checked bags." But the `communicate_info` check requires specific strings (e.g., exact price, bag count wording) that the model's natural language doesn't match.

**Connection to tau2-bench**: Tau2-bench evaluates NL assertions (the agent must communicate specific details). If the training model optimizes for action correctness but not communication quality, this pattern transfers: the model acts correctly but fails NL assertion checks.

### 3.2 Failure Mode 2: Over-Transfer on Refusal Tasks (86.4 pts lost, 27% of losses)

**144 games at reward 0.40** — model gets lookup (0.1) + reasoning (0.3) but transfers instead of directly refusing.

The model correctly identifies the policy violation, correctly explains to the user why the action isn't possible, but then transfers to a human agent instead of ending the conversation with a refusal.

Breakdown:
- **115 games**: Model explains policy → user pushes back → model transfers. The correct behavior was to simply stand firm and refuse.
- **29 games**: Model gets stuck in flight search loops without completing the action, exhausting max steps.

**Example** (Game (3, 3000), reward 0.40):
- Step 5: "Since your booking was made more than 24 hours ago and you do not have travel insurance, cancellation is not allowed under the current policy." (Correct reasoning!)
- Step 6: "I must adhere strictly to the current policy...I am unable to cancel it at this time." (Correct refusal!)
- Step 7: `transfer_to_human_agents` with summary about "requesting human agent review" (Unnecessary transfer — the model already gave the right answer)

**Why this happens**: The reward structure creates an incentive to transfer. On hard refusal tasks, a "safe transfer" earns 0.40, while a correct direct refusal earns 1.0 — but an incorrect action earns 0.0. The model learns that transfer is the "safe middle ground" with 0.40 guaranteed reward. GRPO reinforces this because 0.40 is above the group mean for difficult adversarial scenarios where some rollouts score 0.0 or 0.10.

**Connection to tau2-bench**: This is the exact behavior identified in the transfer analysis. R51 (iter_15 transfers instead of searching other orders), A34 (model gives up on complex requests), and multiple airline tasks where the model prematurely escalates.

### 3.3 Failure Mode 3: Forbidden Cancellations (59.4 pts lost, 19% of losses)

**66 games with forbidden cancel actions at reward 0.10**

The model calls `cancel_reservation` on non-refundable/ineligible reservations.

Breakdown:
- **41 games with 3+ cancels** (Template 12: multi-reservation selective cancel): The user has 5+ reservations and asks to cancel all. Some are eligible, some aren't. The model cancels them ALL indiscriminately. Example: Game (1, 1000) — model retrieves all 5 reservations, then fires 5 `cancel_reservation` calls in sequence without checking each one's eligibility.
- **20 games with 1-2 cancels**: Model cancels a single ineligible reservation despite checking the policy.
- **5 games with cancel + transfer**: Model cancels the wrong reservation AND transfers.

**Root cause**: In multi-cancel scenarios, the model doesn't iterate over each reservation's eligibility. It treats "cancel all" as a bulk operation rather than per-item policy check. This suggests the training scenarios don't sufficiently penalize partial compliance — canceling 3 correct + 2 wrong gives the same 0.10 as canceling 0 correct + 5 wrong.

### 3.4 Failure Mode 4: Lazy Transfer on Valid Tasks (41.4 pts lost, 13% of losses)

**46 games at reward 0.10** — model transfers on tasks where a real action was required.

These are cooperative scenarios (T11-type) or multi-reservation tasks where some reservations ARE eligible for cancellation. The model gives up and transfers instead of completing the valid portion.

**Example** (Game (1, 1001)):
User asks to cancel all reservations. 2 are eligible (7KYHMW, I6KKNF), 3 are not (ABB0M7, GL1CZL, ACE9Z1). Correct behavior: cancel the 2 eligible ones and refuse the 3 ineligible ones. Actual behavior: model retrieves all details, correctly identifies the split, but then transfers instead of acting.

**Root cause**: The model learned from over-transfer (FM2) that "when the user is upset and some actions are forbidden, transfer is safe." It over-generalizes to cases where it should do the valid portion and refuse only the invalid portion.

### 3.5 Other (11.5 pts lost, 3% of losses)

- **2 games at 0.00**: Complete failures (repeated invalid actions)
- **29 games at 0.40 without transfer**: Model got stuck in search loops (searching flights endlessly without completing the update)
- **5 edge cases** (0.23, 0.30, 0.37, 0.60): Unusual partial credit combinations

---

## 4. Per-Group Analysis

| Group | Games | Mean Reward | Transfer Games | Key Character |
|---|---|---|---|---|
| 0 | 416 | 0.839 | 42 (10%) | Balanced mix, moderate transfer rate |
| 1 | 416 | 0.807 | 43 (10%) | Heavy on multi-cancel (151 cancel calls), flight updates |
| 2 | 416 | 0.884 | 1 (0.2%) | Best group — dominated by cooperative retail tasks |
| 3 | 416 | 0.701 | 84 (20%) | Worst group — heavy adversarial, 83 games at 0.40 reward |

**Group 3 is the problem group**: 20% transfer rate, 83 games at 0.40 (over-transfer). This group appears to have a high concentration of adversarial cancel-refusal scenarios where the model defaults to transfer as a coping strategy. This group alone accounts for 124 of the 320 points lost (39% of all losses).

**Group 2 is the success story**: Almost no transfers (1 game), dominated by cooperative retail tasks (cancel_pending_order, exchange, modify). The model handles these well, which aligns with the +7% improvement on non-targeted retail tasks in tau2-bench.

---

## 5. Connection to Tau2-Bench Regression

The rollout patterns directly explain the tau2-bench results:

| Rollout Pattern | Points Lost | Tau2-Bench Effect |
|---|---|---|
| Over-transfer on refusal tasks | 86.4 | Airline non-targeted -9.5% (premature escalation) |
| Over-transfer on refusal tasks | 86.4 | Retail T6 -38% (transfer instead of investigating) |
| Forbidden multi-cancel | 59.4 | Airline T1 marginal improvement (+3.4%) — model sometimes gets it right but often still cancels blindly |
| Lazy transfer on valid tasks | 41.4 | A12, A20, A34 regressed (model gives up on complex valid tasks) |
| Missing communication | 122.5 | Doesn't map directly (tau2-bench evaluates differently) but suggests model deprioritizes user-facing quality |
| Cooperative task competence | — (1,150 perfect games) | Retail non-targeted +7% (tool-calling practice generalizes) |

**The core problem restated**: The model's loss landscape has a "transfer attractor" at 0.40 reward. For any scenario where the model is uncertain, transferring guarantees 0.40, which is often above the GRPO group baseline for hard scenarios. Extended training (iter_15) amplifies this attractor until transfer becomes the default strategy for ambiguity. This is useful within the training distribution (adversarial scenarios where refusal is correct) but catastrophic on tau2-bench (where ambiguity often requires investigation, not escalation).

---

## 6. Hill-Climbing Strategy

### 6.1 Should we use iter_1-5 as a base? YES.

The tau2-bench data shows:
- **iter_0 → iter_1**: Total +1.9% (54→57), airline targeted +6.9%, retail non-targeted +4%
- **iter_1 → iter_15**: Total +0% (57→57), but composition shifts: airline -3.4%, retail targeted -14.3%

Iter_1 captures the "easy wins" — basic policy awareness, improved tool-calling — before the degenerate over-transfer behavior takes hold. By iter_15, the model has reward-hacked the transfer mechanism.

**Recommended checkpoint**: iter_3-5 (if available). Iter_1 may be too early to have internalized the policy reasoning; iter_5 gives a few more iterations of learning without the full degeneration.

### 6.2 What failure mode to target next?

Based on the tau2-bench gap analysis and the rollout failure modes, the priorities are:

#### Priority 1: Procedural Accuracy (addresses 38% of rollout losses + retail gap)

The model does the right action but fails communication/verification. Design tasks that require:
- **Stating specific details before acting**: "Before updating, confirm: the price difference is $X, to be charged to payment_Y."
- **Multi-step verification workflows**: lookup → verify eligibility → confirm with user → act → report result
- **Communication grading**: reward requires the model to mention specific facts (price, policy rule, confirmation number)

This directly targets the 245 games losing 0.5→1.0 on communication failures, AND addresses the tau2-bench NL assertion evaluation criteria.

#### Priority 2: Direct Refusal Training (addresses 27% of rollout losses + over-transfer)

The model transfers when it should refuse. Design tasks that:
- **Explicitly penalize transfer on solvable tasks**: transfer gets 0.0 (not 0.40) when the model already has enough info to refuse
- **Reward firm, explained refusals**: model must say "no" with a specific policy citation and NOT transfer
- **Include scenarios where the user escalates AFTER refusal**: model must hold firm through 3-4 rounds of pushback without transferring

This directly targets the 144 games at 0.40 reward and the tau2-bench over-transfer regression.

#### Priority 3: Selective Action Training (addresses 19% of rollout losses)

For multi-item tasks (cancel 3 of 5 reservations, return 2 of 4 items):
- **Per-item eligibility checking**: model must check each item individually
- **Partial completion**: model should act on eligible items and refuse ineligible ones in the same conversation
- **Strict grading**: canceling even one ineligible item gives 0.0, not partial credit

This directly targets the 66 forbidden-cancel games, especially the T12 multi-cancel scenarios.

### 6.3 Concrete Hill-Climbing Plan

```
Phase 1: iter_5 checkpoint + Procedural Accuracy Environment
  - 5-10 iterations
  - Focus: communicate before acting, verify before executing
  - Expected: improve tau2-bench NL assertions, retail non-targeted
  - Eval: tau2-bench after 1, 3, 5, 10 iters (3 trials each)

Phase 2: best Phase 1 checkpoint + Direct Refusal Environment
  - 5-10 iterations
  - Focus: refuse clearly without transferring, hold firm under pressure
  - Expected: improve airline targeted, fix retail T6 regression
  - Eval: tau2-bench after 1, 3, 5, 10 iters (3 trials each)

Phase 3 (optional): best Phase 2 checkpoint + Selective Action Environment
  - 5 iterations
  - Focus: multi-item per-item eligibility checking
  - Expected: improve multi-cancel/multi-return tasks
```

### 6.4 Reward Structure Recommendations

Based on the failure analysis, the current reward structure has two problems:

1. **Transfer is too rewarding**: 0.40 for "safe transfer" creates a reward-hacking attractor. Consider:
   - 0.40 → 0.20 for transfer on refusal tasks (still above 0.0 for wrong action, but not as attractive)
   - 0.0 for transfer when model has already given the correct refusal (model literally says "no" then transfers anyway)

2. **Communication is binary**: communicate_info is all-or-nothing. Consider:
   - Partial credit for mentioning SOME required info
   - Bonus credit for unprompted proactive communication
   - This provides denser gradient for the communication-heavy rollouts

---

## 7. Reward Structure Deep Dive

For reference, here is how the current reward tiers map to the observed failure modes:

```
Reward Flow:
  1.0  ← Perfect: no forbidden + all required + all communication
  0.50 ← Correct action + missed communication (0.1 lookup + 0.4 action)
  0.40 ← Correct reasoning + transferred on refusal task (0.1 lookup + 0.3 reasoning)
  0.10 ← Forbidden action + had lookup OR lazy transfer on valid task
  0.00 ← Forbidden action + no lookup
```

The problematic incentive is:
- For a hard refusal task: correct refusal = 1.0, transfer = 0.40, wrong action = 0.0
- GRPO group average for hard tasks might be ~0.3 (some get 0.0, some get 0.40, some get 1.0)
- Transfer (0.40) is ABOVE group mean → positive advantage → GRPO reinforces transfer

A potential fix: make transfer reward depend on whether the model already demonstrated it knew the answer:
- Transfer WITHOUT reasoning = 0.10 (current: 0.10 for "unreasoned transfer" — already works)
- Transfer WITH reasoning = 0.20 (current: 0.40 — too high)
- This makes correct refusal (1.0) much more attractive vs transfer (0.20) while still keeping transfer above total failure (0.0).

---

## 8. Summary Table

| Metric | Value |
|---|---|
| Total games | 1,664 |
| Mean reward | 0.808 |
| Points lost | 320 / 1,664 (19.2%) |
| Largest failure: missing communication | 122.5 pts (38%) |
| Second: over-transfer | 86.4 pts (27%) |
| Third: forbidden cancel | 59.4 pts (19%) |
| Fourth: lazy transfer | 41.4 pts (13%) |
| Recommended base checkpoint | iter_3-5 |
| Next training target | Procedural accuracy + communication |
| Reward fix needed | Reduce transfer reward from 0.40 → 0.20 |
