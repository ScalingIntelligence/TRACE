# LoRA Merge Analysis: Why Merging Didn't Work and What to Try

## The Numbers at a Glance

| Model | Airline | Retail |
|-------|---------|--------|
| **Baseline (no LoRA)** | 12/50 (24%) | 42/114 (37%) |
| Best single expert (PRE-40) | 17/50 (34%) | 49/114 (43%) |
| **Union ceiling** | **29/50 (58%)** | **71/114 (62%)** |
| merged-tc-sd-ms-pre (sum w=1.0) | 12/50 (24%) | 43/114 (38%) |
| merged-tc-sd-ms-pre-**linear** | 16/50 (32%) | 47/114 (41%) |
| merged-tc-sd-ms-pre-**stack** | 13/50 (26%) | — |
| distill2 | 14/50 (28%) | 48/114 (42%) |
| mixed-std-ms-25 | 16/50 (32%) | 48/114 (42%) |
| onpolicy-4skill | 13/50 (26%) | 40/114 (35%) |
| **Orchestrator (3-model)** | **23/50 (46%)** | **54/114 (47%)** |

### All Single Expert Results

| Model | Airline | Retail |
|-------|---------|--------|
| Baseline | 12/50 | 42/114 |
| Structured-10 (ST) | 17/50 | 46/114 |
| Structured-Tool-10 (TC) | 17/50 | 48/114 |
| Multistep-10 (MS) | 15/50 | 47/114 |
| Precondition-40 (PRE) | 17/50 | 49/114 |
| Onpolicy-20 (ON) | 14/50 | 46/114 |

### All Merge/Distill Approaches

| Model | Airline | Retail |
|-------|---------|--------|
| merged-tc-sd-ms-pre (sum w=1.0) | 12/50 | 43/114 |
| merged-tc-sd-ms-pre-linear | 16/50 | 47/114 |
| merged-tc-sd-ms-pre-stack | 13/50 | — |
| merged-tc-std | 15/50 | 45/114 |
| distill1 | 14/50 | 47/114 |
| distill2 | 14/50 | 48/114 |
| distill3 | 8/50 | 21/53 |
| distill-r16 | 14/50 | 46/114 |
| onpolicy-4skill | 13/50 | 40/114 |
| mixed-std-ms-25 | 16/50 | 48/114 |
| mixed-std-ms-30 | 12/50 | 45/114 |
| mixed-std-ms-35 | 13/50 | 44/114 |
| mixed-3skill-10 | 16/50 | — |

## Key Finding: The Merged Model Only Captures "Easy" Tasks

Per-task analysis reveals the most damning pattern:

### Airline: Merged-linear success rate by number of expert solvers

| # Experts that solve | Merged-linear captures |
|---------------------|----------------------|
| 1 expert solves | **0/7 (0%)** |
| 2 experts solve | **0/6 (0%)** |
| 3 experts solve | 3/4 (75%) |
| 4+ experts solve | 10/12 (83-100%) |

### Retail: Merged-linear success rate by number of expert solvers

| # Experts that solve | Merged-linear captures | Distill2 captures |
|---------------------|----------------------|-------------------|
| 1 expert solves | 5/15 (33%) | 3/15 (20%) |
| 2 experts solve | 2/3 (67%) | 2/3 (67%) |
| 3 experts solve | 5/13 (38%) | 7/13 (54%) |
| 4 experts solve | 4/5 (80%) | 3/5 (60%) |
| 5 experts solve | 10/12 (83%) | 10/12 (83%) |
| 6 experts solve | 21/23 (91%) | 21/23 (91%) |

**The merged model ONLY succeeds on tasks that most experts already agree on. It completely fails to capture any unique expert knowledge.**

## Airline Per-Task Matrix

```
Task  BL  ST  TC  MS PRE  ON  M-LIN  Status
  0    1   0   1   1   0   1    0     MISSED (4 solvers)
  1    1   1   0   0   1   1    1     GOT
  3    1   1   1   1   1   1    1     GOT (all)
  4    1   1   1   1   1   0    1     GOT
  5    1   0   1   1   1   1    1     GOT
  6    1   1   1   1   1   1    1     GOT (all)
  8    0   0   1   0   1   1    1     GOT
  9    0   1   0   1   0   0    0     MISSED (2 solvers)
 10    0   0   0   1   1   0    0     MISSED (2 solvers)
 12    0   1   0   0   0   0    0     MISSED (unique to ST)
 13    0   1   1   0   1   0    0     MISSED (3 solvers)
 17    0   0   0   0   1   0    0     MISSED (unique to PRE)
 18    0   0   0   0   0   0    1     NEW (no expert solves!)
 20    0   1   1   0   1   1    0     MISSED (4 solvers!)
 22    0   0   0   0   0   0    1     NEW (no expert solves!)
 25    0   0   0   0   0   1    0     MISSED (unique to ON)
 26    0   1   1   0   1   0    1     GOT
 28    1   1   1   1   1   1    1     GOT (all)
 31    0   0   0   0   0   1    0     MISSED (unique to ON)
 34    1   1   1   1   0   1    1     GOT
 36    0   0   1   0   1   0    0     MISSED (2 solvers)
 37    0   1   1   0   0   0    0     MISSED (2 solvers)
 38    0   0   0   1   0   0    0     MISSED (unique to MS)
 40    1   1   1   1   1   1    1     GOT (all)
 41    0   0   0   0   1   0    0     MISSED (unique to PRE)
 42    0   0   1   0   0   1    0     MISSED (2 solvers)
 43    0   0   0   1   0   0    0     MISSED (unique to MS)
 45    1   1   0   0   1   0    1     GOT
 46    1   1   1   1   1   1    1     GOT (all)
 47    0   1   0   1   0   0    0     MISSED (2 solvers)
 48    1   1   1   1   0   0    1     GOT
 49    0   0   0   0   0   0    1     NEW (no expert solves!)
```

Merged-linear captured: 13/29 (45%) of solvable tasks, missed 16/29 (55%).

Notably, merged-linear also gained 3 tasks (T18, T22, T49) that NO individual expert solves — suggesting the merge creates some emergent but unreliable capability.

## Tasks Uniquely Solved by Single Expert (Airline)

- **ST**: Task 12 — merged does NOT capture
- **MS**: Tasks 38, 43 — merged does NOT capture
- **PRE**: Tasks 17, 41 — merged does NOT capture
- **ON**: Tasks 25, 31 — merged does NOT capture

The merge captures **zero** uniquely-expert-solved tasks. This is the core problem.

## Stacked Merge Deep Dive (airline-merged-tc-sd-ms-pre-stack)

The stacked merge (sequential adapter application) performs **13/50 (26%)** — worse than linear merge (16/50), worse than any single expert, and barely above baseline (12/50).

### Stacked vs Linear vs Sum: Head-to-Head

| # Experts that solve | Stack captures | Linear captures | Sum captures |
|---------------------|---------------|----------------|-------------|
| 1 expert solves | 0/7 (0%) | 0/7 (0%) | 1/7 (14%) |
| 2 experts solve | 2/6 (33%) | 0/6 (0%) | 0/6 (0%) |
| 3 experts solve | 1/4 (25%) | 3/4 (75%) | 2/4 (50%) |
| 4 experts solve | 2/4 (50%) | 2/4 (50%) | 1/4 (25%) |
| 5 experts solve | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| 6 experts solve | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |

Stacked is **worse** than linear on 3-solver tasks (25% vs 75%) and comparable elsewhere.

### Stacked vs Linear: Per-Task Differences

- **Stack wins (linear fails)**: Tasks 10, 37 — only 2 tasks
- **Linear wins (stack fails)**: Tasks 8, 18, 22, 26, 49 — 5 tasks

Net: stack loses 3 tasks vs linear.

### Stacked vs Baseline

- **Stack improves**: Tasks 10, 37 (+2)
- **Stack regresses**: Task 0 (-1)

The stacked merge is essentially baseline-level performance with minor noise.

### Trajectory Analysis: Why Stacked Fails

**Task 26 (stack fails, linear passes):** Both models retrieve the same user/reservation data. The stack model proceeds to **cancel the wrong reservation** — it cancels reservation 3FRNFB (MCO→CLT, May 28) when the customer is actually asking about a June 15 flight that doesn't exist. The linear model correctly recognizes the reservation doesn't exist and **transfers to a human agent**. The stacked model is more "action-happy" — it takes a wrong action rather than admitting uncertainty.

**Task 8 (stack fails, linear passes):** Both models follow similar tool call sequences (get_user_details → multiple get_reservation_details → search_direct_flight → book_reservation). The stack model makes the same tool calls but gets the DB state wrong (DB_FAIL), suggesting it passes slightly wrong arguments or makes a subtle procedural error.

### Failure Mode Breakdown

| Failure Mode | Stack | Linear | Baseline |
|-------------|-------|--------|----------|
| DB_FAIL + COMM_OK | 30 | 29 | 30 |
| BOTH_FAIL | 4 | 2 | 4 |
| DB_OK + COMM_FAIL | 0 | 1 | 1 |
| NO_BREAKDOWN | 3 | 2 | 3 |

The stacked model's failure profile is **nearly identical to baseline**. This means stacking effectively destroyed all LoRA improvements — the model reverted to base behavior.

### Why Stacking Is Worse Than Linear

Stacked merging applies adapters sequentially: each adapter's delta is computed on the already-modified weights from the previous adapter. This creates a **compounding distortion** problem:

1. **Adapter 1** shifts weights from W_base to W_1 = W_base + delta_1
2. **Adapter 2** was trained assuming W_base, but now sees W_1. Its delta_2 is computed in the wrong region of weight space.
3. By adapter 4, the base weights have drifted so far from the original training context that each subsequent adapter's corrections are increasingly meaningless.

Linear merge avoids this by computing all deltas independently from the original base weights and summing them. While linear merge still suffers from interference, at least each delta is computed correctly.

**Stacking is strictly worse because it adds order-dependent compounding errors on top of the same interference problem.**

## Root Cause Analysis

### Root Cause #1: LoRA Delta Interference (Destructive Superposition)

The merge computes: `W_final = W_base + delta_TC + delta_ST + delta_MS + delta_PRE`

Each LoRA adapter learned a **different direction** in parameter space. When you sum 4 deltas with weight 1.0 each, the resulting parameter vector lands in a region of weight space that **none of the individual adapters occupy**. It's not an average of their behaviors — it's a fundamentally different model.

The non-linear merge (`merged-tc-sd-ms-pre`) at **12/50** (same as baseline!) confirms this: summing all 4 full-weight deltas essentially destroyed the base model's capabilities. The "linear" variant with reduced weights helped but still loses the specialized behaviors.

### Root Cause #2: Experts Learn Contradictory Strategies

Pairwise overlap analysis shows experts are highly non-overlapping:

| Pair | Both solve | Only 1st | Only 2nd |
|------|-----------|----------|----------|
| ST & TC | 12 | 5 | 5 |
| ST & MS | 10 | 7 | 5 |
| ST & PRE | 11 | 6 | 6 |
| TC & PRE | 12 | 5 | 5 |
| MS & PRE | 8 | 7 | 9 |
| MS & ON | 8 | 7 | 6 |

When you add the delta for "precondition checking" on top of "structured tool calling" on top of "multistep reasoning," the model can't distinguish when to use which strategy. The merged model becomes a **jack of no trades**.

### Root Cause #3: The Task-Level Binary Nature of tau-bench

tau-bench rewards are binary (0 or 1). Even small behavioral degradation from interference causes a complete fail — there's no partial credit. The merged model probably gets "close" on many tasks but makes one wrong tool call or policy decision and gets 0.

### Why Each Approach Failed Similarly

| Approach | Problem |
|----------|---------|
| **Sum merge (w=1.0)** | 4x overshoot — deltas are too large, model diverges |
| **Stacked merge** | Compounding distortion — each adapter computed on wrong base, reverts to baseline behavior |
| **Linear merge (reduced w)** | Dilutes each skill to ~25% strength, losing unique capabilities |
| **Distillation** | Teacher is the orchestrator or mixed data — can't learn what it can't see |
| **Mixed training data** | Training on all skills simultaneously causes gradient interference |
| **Onpolicy-4skill** | RL on all 4 skills at once — same interference problem during training |

## What to Try Next

### 1. TIES-Merging or DARE (most promising for current setup)

Instead of naive linear combination, use **task-specific interference elimination**:
- **TIES**: Trim small deltas, resolve sign conflicts, then merge. Preserves the "direction" of each expert rather than letting them cancel out.
- **DARE**: Randomly drop some delta parameters before merging, relying on the fact that LoRA changes are redundant. Reduces interference while keeping magnitude.

Both are implemented in the `mergekit` library and work with LoRA adapters.

### 2. Model Soup with Validation-Based Weight Search

Instead of fixed weights, **search for optimal merge weights** per adapter:
```
W = W_base + a1*delta_TC + a2*delta_ST + a3*delta_MS + a4*delta_PRE
```
Use a validation set (subset of tau-bench tasks) to grid-search a1,a2,a3,a4 in [0, 0.1, 0.2, ..., 1.0]. The current equal-weight assumption is almost certainly wrong — some adapters may need much higher weight.

### 3. Sequential/Iterative Merging

Instead of summing all 4 at once, merge 2 at a time:
1. Merge the two most compatible adapters (highest pairwise overlap: TC & PRE with 12 shared tasks)
2. Evaluate the 2-merge
3. Then carefully add the third, tuning its weight
4. Add the fourth

This lets you detect when adding an adapter starts hurting.

### 4. SVD-Based Subspace Merging

Decompose each LoRA delta into SVD, find the shared subspace vs. unique subspaces, and merge only the compatible components. This is essentially what methods like "RegMean" or "Fisher merging" do.

### 5. Conditional Activation / Mixture of Experts within a Single Model

Instead of static weight merging, train a **router layer** that dynamically blends LoRA activations at inference time. This is essentially the orchestrator approach but at the parameter level rather than the model level. Libraries like `MoLoRA` or the existing weighted LoRA implementation could be extended this way.

### 6. Progressive Distillation with Per-Task Oracle Routing

Instead of distilling from one merged teacher:
1. For each training example, route it to the **best expert** for that example
2. Use that expert's output as the distillation target
3. Train a single student on these optimally-selected targets

This gives the student the best of each expert rather than a blurred average.

## Recommendation (Priority Order)

1. **TIES-Merge** — quick to try, addresses the core sign-conflict issue, use `mergekit`
2. **Weight search** — grid-search alpha values with existing merge script, just need eval loops
3. **Progressive distillation with oracle routing** — uses existing expert models as teachers, trains a genuinely unified model

## Fundamental Insight

The orchestrator at 23/50 airline (85% of ceiling) is already close to optimal for routing between separate experts. The hard problem is making a *single* model that internalizes all these behaviors, and naive parameter averaging provably doesn't do this when the skills are complementary rather than redundant.
