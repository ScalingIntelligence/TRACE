# Adversarial GRPO 5-Iteration Retail Eval Analysis

**Date**: 2026-02-08
**Domain**: tau2-bench retail
**Base model**: Qwen/Qwen3-30B-A3B-Instruct-2507 (untrained)
**Trained model**: adversarial_v1_5iter (5 iterations of adversarial GRPO)
**User simulator**: Qwen3-30B (temperature=0.0)
**Eval files**:
- Base: `data/simulations/qwen-3-30b-retail-qwen3-30b.json`
- Trained: `data/simulations/qwen-3-30b-adv_v1_5iter-retail-qwen3-30b.json`

---

## Overall Accuracy

| Metric | BASE | TRAINED | Delta |
|--------|------|---------|-------|
| Total tasks | 114 | 114 | |
| Pass (reward=1.0) | 46 | 49 | **+3** |
| Fail (reward=0.0) | 68 | 65 | -3 |
| Pass Rate | **40.4%** | **43.0%** | **+2.6pp** |
| Early terminations | 2 | 3 | +1 |

Training yields a modest net improvement of +3 tasks (40.4% to 43.0%).

---

## Per-Task Comparison

| Category | Count | % |
|----------|-------|---|
| Both pass | 35 | 30.7% |
| Both fail | 54 | 47.4% |
| **Regressions** (base pass -> trained fail) | **11** | **9.6%** |
| **Improvements** (base fail -> trained pass) | **14** | **12.3%** |

The net gain of +3 comes from 14 improvements minus 11 regressions. High-volatility result: 25 of 114 tasks (21.9%) changed outcome.

- **Regressions**: Tasks 3, 4, 12, 24, 61, 62, 63, 66, 74, 97, 112
- **Improvements**: Tasks 8, 32, 43, 44, 45, 48, 55, 56, 71, 78, 84, 85, 86, 90

---

## Failure Mode Comparison

| Failure Type | BASE | TRAINED | Delta | Direction |
|-------------|------|---------|-------|-----------|
| ACTION_CHECKS | 61 | 54 | -7 | **Improved** |
| DB | 64 | 58 | -6 | **Improved** |
| COMMUNICATE | 14 | 16 | +2 | Regressed |
| EARLY_TERMINATION | 2 | 3 | +1 | Regressed |
| **Total failing** | **68** | **65** | **-3** | **Improved** |

Training significantly reduced ACTION and DB failures (-7, -6) but slightly increased COMMUNICATE failures (+2).

---

## Trained Model Failure Breakdown (65 tasks)

| Category | Count |
|----------|-------|
| ACTION + DB | 41 |
| ACTION + COMMUNICATE + DB | 11 |
| DB only | 5 |
| EARLY_TERMINATION | 3 |
| ACTION + COMMUNICATE | 2 |
| COMMUNICATE only | 2 |
| COMMUNICATE + DB | 1 |

---

## Regression Analysis (11 Tasks)

**Task 112** -- Modify laptop order to NYC address + change watch + change laptop item.
- BASE: 15 tool calls, completed all modifications.
- TRAINED: Only 6 calls. Diverged at step 3 by fetching wrong product details. Only changed the watch, missed laptop modification and address change.
- **Root cause**: Not thorough enough -- handled one modification instead of all three.

**Tasks 3, 4** -- Modify pending t-shirts to purple, polyester, specific size.
- BASE: 9/12 calls, checked all orders to find the right pending t-shirt.
- TRAINED: 7/6 calls, found one t-shirt and stopped instead of checking all pending orders.
- **Root cause**: Did not exhaustively search all orders.

**Task 12** -- Cancel/return non-gaming items, prefer PayPal refund, transfer to human if needed.
- BASE: Correctly escalated to human agent when PayPal wasn't available.
- TRAINED: Processed a return instead of transferring to human.
- **Root cause**: Failed to recognize PayPal refund requirement should trigger human escalation.

**Task 24** -- Cancel grill (then change mind) + ask about t-shirt materials.
- TRAINED got the DB actions right but failed COMMUNICATE -- didn't relay correct info to user.

**Task 61** -- Change wireless earbuds to blue, price same or lower.
- TRAINED presented options but stopped before executing the modification.

**Task 63** -- Modify bluetooth speaker + communicate order total.
- TRAINED did mental arithmetic instead of using the `calculate` tool, got it wrong ($1,257.87 vs $1,288.65).

**Task 66** -- Change luggage to coat, fallback to return, fallback to cancel.
- BASE: 11 calls, correctly followed fallback chain.
- TRAINED: **71 tool calls** -- stuck in a retry loop. Most dramatic regression.
- **Root cause**: Catastrophic looping behavior introduced by training.

**Task 74** -- Exchange laptop + cancel 5-item pending order.
- TRAINED cancelled the wrong order (W3414433 instead of W3189752).

**Task 97** -- Change address + exchange speaker.
- TRAINED told user to "send the NYC address via email" instead of looking it up from the other order.

**Summary**: 8/11 regressions are ACTION + DB failures (wrong or insufficient tool calls). Dominant pattern is the trained model being **less thorough** -- making fewer calls, not checking all orders, not following through.

---

## Improvement Analysis (14 Tasks)

**Task 55** -- Cancel/return all possible orders (financial hardship).
- BASE: Only 4 calls, then transferred to human.
- TRAINED: 15 calls, systematically cancelled 2 pending orders and returned items from 2 delivered orders.
- **Most dramatic improvement**: TRAINED was dramatically more thorough.

**Task 48** -- Return air purifier, check if vacuum can be returned (don't process).
- BASE returned BOTH items (over-acting).
- TRAINED correctly returned only the air purifier.
- **Improvement**: Better conditional logic adherence.

**Task 8** -- Exchange water bottle and desk lamp, only exchange lamp if asked to confirm.
- BASE exchanged BOTH items (over-acting).
- TRAINED correctly exchanged only the desk lamp after confirmation.
- **Improvement**: Better handling of "only do X if Y" conditions.

**Tasks 44, 45, 56, 71, 78, 85, 86** -- Various multi-step modifications.
- BASE stopped short of executing final tool call.
- TRAINED followed through with the complete workflow.
- **Improvement**: More persistent action completion.

**Task 84** -- Return tablet (then change mind to more expensive one).
- BASE made 14 calls, got confused, transferred to human.
- TRAINED made 7 calls, correctly handled the mind-change.
- **Improvement**: More efficient and correct on changed preferences.

**Task 90** -- Check camera specs, cancel if price > $3000.
- BASE cancelled with wrong reason. TRAINED used correct reason ("ordered by mistake").

**Summary**: 13/14 improvements are ACTION + DB fixes. Dominant pattern is the trained model being **more persistent** -- completing workflows the base model abandoned.

---

## Key Findings

1. **Training helps with action completion**: The trained model is better at following through with multi-step tool call sequences. It more reliably completes modifications, cancellations, and returns that the base model stops short of.

2. **Training sometimes hurts thoroughness**: Paradoxically, while training makes the model more likely to complete individual actions, it sometimes makes it less exhaustive in searching across all orders (Tasks 3, 4) or causes it to skip necessary lookups (Task 97).

3. **One catastrophic regression**: Task 66 saw the trained model make 71 tool calls vs BASE's 11 -- stuck in a retry loop. Adversarial training can occasionally introduce degenerate looping behavior.

4. **Communication regressions**: Training improved DB/action correctness but slightly hurt communication accuracy (+2 COMMUNICATE failures). The model sometimes gets actions right but fails to correctly summarize results (Tasks 24, 63).

5. **The 54 "both fail" tasks are genuinely hard**: These involve complex multi-step reasoning, tricky conditional logic, multi-order coordination, and adversarial user behavior. These are where further training effort should focus.

---

## Comparison with Airline Results

| Metric | Airline BASE | Airline TRAINED | Retail BASE | Retail TRAINED |
|--------|-------------|-----------------|-------------|----------------|
| Pass rate | 34.0% | 32.0% (-2pp) | 40.4% | 43.0% (+2.6pp) |
| Regressions | 4 | | 11 | |
| Improvements | 3 | | 14 | |
| Net change | -1 | | +3 | |

Training **hurt** on airline (-2pp, more over-compliance with adversarial requests) but **helped** on retail (+2.6pp, better action completion). The retail domain benefits because the training signal ("be more action-oriented") aligns with what retail tasks need (follow through on tool calls). In airline, this same signal backfires on adversarial refusal tasks where the correct behavior is to *not* act.
