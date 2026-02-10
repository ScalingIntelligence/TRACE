# Adversarial GRPO 10-Iteration Retail Eval Analysis

**Date**: 2026-02-08
**Domain**: tau2-bench retail
**Models compared**:
- BASE: Qwen/Qwen3-30B-A3B-Instruct-2507 (untrained)
- 5-ITER: adversarial_v1_5iter
- 10-ITER: adversarial_v1_10iter

---

## Overall Accuracy Progression

| Model | Pass | Fail | Pass Rate | Delta vs BASE |
|-------|------|------|-----------|---------------|
| **BASE** | 46/114 | 68 | **40.4%** | -- |
| **5-ITER** | 49/114 | 65 | **43.0%** | +2.6pp |
| **10-ITER** | 46/114 | 68 | **40.4%** | 0.0pp |

5-iter showed modest improvement (+2.6pp). **10-iter regresses back to exact baseline**, completely erasing the 5-iter gains.

---

## Trajectory Analysis: BASE -> 5-ITER -> 10-ITER

| Pattern | Count | Description |
|---------|-------|-------------|
| Stable PASS (P->P->P) | 28 | Reliably solved |
| Stable FAIL (F->F->F) | 47 | Consistently too hard |
| Monotonic improvement (F->F->P or F->P->P) | 12 | 7 F->F->P, 5 F->P->P |
| **Monotonic regression (P->P->F or P->F->F)** | **12** | 7 P->P->F, 5 P->F->F |
| Oscillation P->F->P (recovery) | 6 | Lost at 5-iter, recovered at 10-iter |
| **Oscillation F->P->F (lost gains)** | **9** | Gained at 5-iter, lost at 10-iter |

**12 improvements exactly canceled by 12 regressions = net zero.** Plus 9 tasks that 5-iter learned to solve were "forgotten" by 10-iter (F->P->F).

---

## Failure Categorization for 10-ITER (68 failures)

| Category | Count | % |
|----------|-------|---|
| DB-only failures | 46 | 67.6% |
| DB + COMMUNICATE failures | 12 | 17.6% |
| COMMUNICATE-only failures | 6 | 8.8% |
| Max-steps (looping) | 4 | 5.9% |

---

## Behavioral Failure Categories (10-ITER)

| Category | Count | % |
|----------|-------|---|
| **Wrong tool arguments** | 36 | 52.9% |
| **Wrong tool args + wrong communication** | 9 | 13.2% |
| **Under-action** (didn't complete steps) | 6 | 8.8% |
| **Wrong calculation/communication only** | 6 | 8.8% |
| **Looping/stuck** (max_steps) | 4 | 5.9% |
| Other combinations | 7 | 10.3% |

---

## Most Commonly Failed Actions

| Action | Failure Count (10-ITER) | vs BASE |
|--------|------------------------|---------|
| `get_order_details` | 45 | 37 (+8 worse) |
| `exchange_delivered_order_items` | 21 | -- |
| `return_delivered_order_items` | 18 | -- |
| `modify_pending_order_items` | 17 | -- |
| `calculate` | 14 | -- |
| `cancel_pending_order` | 14 | -- |

`get_order_details` failures **increased** from 37 (BASE) to 45 (10-ITER) -- the model is becoming more eager to act without fully exploring the user's orders first.

---

## Key Regressions: P->P->F (7 tasks -- both BASE and 5-ITER passed, 10-ITER fails)

| Task | Issue | Root Cause |
|------|-------|------------|
| **2** | Return cleaner/headphone/watch | Wrong `return_delivered_order_items` args, didn't call `get_product_details` |
| **15** | Modify pending boots size | Didn't check all orders, wrong item ID |
| **36** | Credit card limit concern | **Hit max_steps** -- looping |
| **67** | Check order total | All 5 action checks failed, didn't communicate $829.43 |
| **83** | Return expensive tablet | Wrong `return_delivered_order_items` arguments |
| **103** | Return bookshelf + jigsaw | DB correct but failed to communicate tracking number |
| **108** | Return everything except tablet | Wrong return args + failed to communicate refund $346.93 |

---

## Lost 5-ITER Gains: F->P->F (9 tasks)

These tasks were solved by 5-iter but broken again at 10-iter:

| Task | What 5-ITER got right | What 10-ITER broke |
|------|----------------------|-------------------|
| 8 | Correctly exchanged only desk lamp after confirmation | Regressed |
| 32 | Correctly cancelled charger order | Regressed |
| 43 | Communicated all values correctly | Regressed |
| 45 | Completed full exchange workflow | Regressed |
| 55 | Systematically processed all 6 orders | Regressed |
| 56 | Executed modification after presenting options | Regressed |
| 84 | Handled mind-change scenario correctly | Regressed |
| 85 | Completed full jacket exchange workflow | Regressed |
| 90 | Used correct cancellation reason | Regressed |

---

## COMMUNICATE Failures Detail (18 total)

All involve the agent failing to state specific numeric values:

| Type | Examples |
|------|----------|
| Missing price/refund | $41.64, $44.08, $918.43, $189.57, $1093.34, $46.66, $3646.68, $1126.04, $1497.65, $164.28, $829.43, $346.93, $1288.65, $9.89, $208.60 |
| Missing tracking number | 746342064230, 286422338955 |
| Missing product spec | 64GB |

---

## Summary

The 10-iteration model **completely erases the modest gains** from 5-iteration training. The pattern is:
- 5-iter improved action completion (+2.6pp)
- 10-iter lost those gains while also introducing new regressions
- Net result: back to baseline

The dominant failure mode is **wrong tool arguments** (53%) -- the model calls correct tools with incorrect parameters. The second issue is the model becoming **less thorough at order exploration** (get_order_details failures increased from 37 to 45).

Unlike airline where the problem is over-compliance on refusals, retail's problem is primarily **execution accuracy degradation** -- the model's ability to correctly parameterize tool calls deteriorates with more training.
