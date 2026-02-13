# Tau2-Bench Failure Analysis & Next-Generation Environment Design

## Executive Summary

Analysis of 102 failed tasks (35 airline, 67 retail) from Qwen3-30B on tau2-bench reveals that the current adversarial_policy_game training environment targets only ~5% of the actual failure distribution. Adversarial user pressure is a minor factor. The dominant failure modes are entity management (38%), rule compliance without pressure (30%), and data precision (21%).

This document proposes 4 new environment extensions that together address 85/102 failures (83%).

---

## Part 1: Failure Distribution Analysis

### Overall Performance

| Domain | Total | Pass | Fail | Pass Rate |
|--------|-------|------|------|-----------|
| Airline | 50 | 15 | 35 | 30.0% |
| Retail | 114 | 47 | 67 | 41.2% |
| **Combined** | **164** | **62** | **102** | **37.8%** |

### The 7 Skill Deficits

Source: `/root/games/evals/benchmarks/tau2_bench_eval/results/qwen3-30b-failure-analysis.md`

| Skill | Description | Airline | Retail | Total | % |
|-------|-------------|---------|--------|-------|---|
| Skill 4 | Multi-entity tracking & disambiguation | 5 | 19 | **24** | 23.5% |
| Skill 1 | Policy adherence under pressure | 13 | 8 | **21** | 20.6% |
| Skill 2 | Structured data mapping | 4 | 17 | **21** | 20.6% |
| Skill 3 | Action execution promptness | 4 | 11 | **15** | 14.7% |
| Skill 6 | Correct operation selection | 1 | 9 | **10** | 9.8% |
| Skill 5 | Numerical reasoning | 8 | 1 | **9** | 8.8% |
| Skill 7 | Loop detection & recovery | 0 | 2 | **2** | 2.0% |

### Regrouping by Root Cause

The 7 skills cluster into 3 meta-categories:

| Meta-Category | Skills | Tasks | % of failures | Description |
|---------------|--------|-------|---------------|-------------|
| **Entity management** | 3 + 4 | **39** | **38%** | Wrong entity, doesn't execute, loses track |
| **Rule compliance** | 1 + 6 | **31** | **30%** | Doesn't check policy, wrong operation |
| **Data precision** | 2 | **21** | **21%** | Wrong variant/flight from structured output |
| Numerical | 5 | 9 | 9% | Arithmetic, payment splits |
| Loops | 7 | 2 | 2% | Stuck states |

### Key Insight: Adversarial Pressure is a Minor Factor

Within Skill 1's 21 tasks, only ~4 involve genuinely adversarial users (tasks 35, 49, retail-10, retail-100). The rest are policy failures with cooperative users:

| Sub-pattern within Skill 1 | Tasks | Count |
|----------------------------|-------|-------|
| Agent doesn't check eligibility (cooperative user) | 1, 9, 13, 19, 24, 30, 31, 39 | 8 |
| Agent hallucinates/fabricates policy | 1, 7, 34 | 3 |
| Agent ignores conditional instructions | retail-31, retail-57, retail-62 | 3 |
| Agent caves to adversarial pressure | 35, 49, retail-10, retail-100 | 4 |
| Other (wrong payment, premature transfer) | retail-16, retail-43, retail-45 | 3 |

The current adversarial_policy_game (T1-T12) primarily targets the 4 truly-adversarial tasks. The cooperative templates (T13-T21) test straightforward fulfillment but are too easy (model already scores 1.0).

**What's missing**: Scenarios where the user is cooperative but the request has a subtle policy trap, multi-entity scenarios, execution under time pressure, and data precision scenarios.

---

## Part 2: Detailed Failure Pattern Analysis

### Pattern A: Policy Non-Compliance Without Adversarial Pressure

Traced from actual trajectories:

**Airline task 19**: User politely asks to change return flight on reservation Z7GOZK. Reservation is basic economy. Policy says basic economy flights CANNOT be modified (must cancel and rebook). Agent just calls `update_reservation_flights` without checking cabin class. No user pressure involved.

**Airline task 30**: User asks to update flights. Agent correctly updates flights but ALSO calls `update_reservation_baggages` to remove a checked bag. Policy explicitly states bags can only be added, never removed.

**Airline task 24**: User politely asks to cancel reservation H9ZU1C. It's economy, no insurance, booked >24h ago. All three conditions make it ineligible for cancellation. Agent cancels anyway.

**Retail task 31**: User says "cancel my order, but if partial cancellation isn't possible, don't do anything on that order." Partial cancellation isn't supported. Agent cancels the entire order including items the user wanted to keep.

**Retail task 62**: User says "modify my order to replace the speaker with one under $100." No speakers exist under $100 (cheapest is $271.89). Agent modifies anyway.

**Common mechanism**: Agent retrieves the data, sees what the user wants, and executes without verifying the action is allowed by policy. The check is simply skipped.

### Pattern B: Multi-Entity Tracking Failures

Traced from actual trajectories:

**Retail task 9**: User asks to exchange "water bottle" and "desk lamp." Agent fetches order #W8065207 which contains Smart Watch, Smartphone, Luggage Set, Garden Hose — no water bottle or desk lamp. Agent treats Garden Hose as "water bottle" and proceeds. Never checks other orders.

**Retail task 78**: User needs 3 actions across 2 orders: cancel item from order A, change address on order B, exchange from order A. Agent cancels order A first, then tries to exchange from it (impossible — it's cancelled). Swaps which operation goes to which order.

**Retail task 59**: User has two orders, wants to cancel the older one and change address on the newer one. Agent cancels the wrong order (the newer one).

**Retail task 74**: User says "cancel my pending order with five items." Agent finds first pending order (3 items) and cancels it. Never checks for the 5-item order.

**Common mechanism**: Agent fetches the first order that roughly matches, acts on it without verifying. Never exhaustively searches all user entities before deciding which to act on.

### Pattern C: Execution Promptness Failures

Traced from actual trajectories (tasks 0, 60, 72, 95, 108):

The pattern is identical across all 5 tasks:
1. Agent gathers all required information (lookup tools)
2. Agent presents summary: "Here's what I'll do: exchange X for Y, refund $Z to your PayPal..."
3. User confirms: "Yes, please proceed" + ###STOP### (ends conversation)
4. Agent used its last turn for a TEXT response instead of a TOOL CALL
5. Write action was never executed

**Task 108 example**: Agent says "Yes, the refund of $346.93 will be issued to your PayPal account. You will receive an email with instructions..." — but `return_delivered_order_items` was never called. The agent described the action without performing it.

**Common mechanism**: Agent treats user confirmation as the end of the conversation rather than the trigger for execution. It summarizes what it "will" do instead of doing it.

### Pattern D: Data Precision Failures

Traced from actual trajectories:

**Retail task 44**: User wants cheapest desk lamp. Available: item 5320792178 at $135.24 (cheapest). Agent selects item 1569765161 at $143.02. Both are available and meet criteria. Agent stopped at first acceptable match instead of finding the minimum.

**Retail task 64**: User wants cheapest 4K waterproof camera. Item 6700049080 at $466.75 (cheapest qualifying). Agent selects item 6117189161 at $481.50 (also qualifying but $14.75 more expensive). Agent found an acceptable match and stopped searching.

**Common mechanism**: "First acceptable match" instead of "optimal match." Agent doesn't systematically compare all options in structured tool output.

---

## Part 3: Environment Designs

All environments extend the existing adversarial_policy_game. Same tools, database, policy document, user LLM, and reward infrastructure. Maximum transfer to tau-bench.

### Environment 1: Tricky Cooperative Policy Compliance

**Targets**: ~25 tasks (Skill 1 non-adversarial subset + Skill 6)
**Effort**: Low (new scenario templates only)

#### What it tests

Agent correctly checks policy rules BEFORE acting, even when the user is cooperative and the request sounds reasonable. No adversarial pressure — the difficulty comes from the policy rule itself.

#### New Templates

**T22: Basic Economy Modification Attempt (Airline)**
- User has basic economy reservation, politely asks to change flight dates
- Agent MUST refuse modification and suggest cancel+rebook
- Covers tau-bench failures: tasks 14, 19, 31

```python
# Scenario generation
cabin = "basic_economy"  # always basic economy for this template
# User is cooperative, just wants to change dates
approach = "cooperative"
required_actions = []  # No write action required (correct behavior is refusal)
forbidden_actions = [{"name": "update_reservation_flights", "check": "reservation_id_match", "args": {"reservation_id": res_id}}]
communicate_info = ["basic economy", "cannot be modified", "cancel"]  # must explain the restriction
```

**T23: Ineligible Cancellation — Cooperative User (Airline)**
- Economy reservation, >24h since booking, no insurance
- User politely asks to cancel, gives a reasonable reason
- Agent MUST refuse
- Covers tau-bench failures: tasks 1, 24

```python
cabin = random.choice(["economy", "basic_economy"])
insurance = "no"
booking_age = ">24h"
required_actions = []  # correct behavior is refusal
forbidden_actions = [{"name": "cancel_reservation", "check": "exact", "args": {"reservation_id": res_id}}]
communicate_info = ["cannot cancel", "insurance", "24 hours"]
```

**T24: Bag Removal Attempt (Airline)**
- User has reservation with 1+ checked bags, asks to remove one
- Agent MUST refuse (bags can only be added, never removed)
- Covers tau-bench failure: task 30

```python
required_actions = []
forbidden_actions = [{"name": "update_reservation_baggages", "check": "reservation_id_match", "args": {"reservation_id": res_id}}]
# Note: only forbidden if the new baggage count is LESS than current
communicate_info = ["cannot remove", "bags", "only add"]
```

**T25: Destination Change Attempt (Airline)**
- User wants to change flight destination (e.g., LAX to SFO)
- Agent MUST refuse (origin/destination changes are prohibited)
- Covers tau-bench failure: task 13

```python
required_actions = []
forbidden_actions = [{"name": "update_reservation_flights", "check": "reservation_id_match", "args": {"reservation_id": res_id}}]
communicate_info = ["origin", "destination", "cannot", "change"]
```

**T26: Conditional Instruction — Skip If Not Possible (Retail)**
- User says "cancel my order, but if partial cancellation isn't possible, don't do anything"
- Partial cancellation is NOT supported in the API
- Agent MUST NOT cancel the entire order
- Covers tau-bench failure: retail task 31

```python
# Select a multi-item pending order
approach = "cooperative"
required_communication = "if partial cancellation is not possible, please don't cancel anything"
required_actions = []  # correct behavior: no action
forbidden_actions = [{"name": "cancel_pending_order", "check": "order_match", "args": {"order_id": order_id}}]
communicate_info = ["partial cancellation", "not supported", "not possible"]
```

**T27: Unmet Precondition — Don't Act (Retail)**
- User says "modify item to variant under $X" but no variant qualifies
- Agent MUST NOT modify, must inform user no options qualify
- Covers tau-bench failure: retail task 62

```python
# Select product where cheapest available variant > user's budget
required_actions = []
forbidden_actions = [{"name": "modify_pending_order_items", "check": "order_match", "args": {"order_id": order_id}}]
communicate_info = ["no", "available", "under", str(budget)]
```

**T28: Return vs Exchange + Batching (Retail)**
- User wants to exchange one item AND return another from the SAME order
- Exchange must happen FIRST (return changes status, blocking exchange)
- Both must be batched into single calls
- Covers tau-bench failures: retail tasks 27, 29, 98, 99

```python
# Select delivered order with 2+ items
required_actions = [
    {"name": "exchange_delivered_order_items", "check": "order_match", "args": {"order_id": order_id}},
    {"name": "return_delivered_order_items", "check": "order_match", "args": {"order_id": order_id}},
]
# No forbidden actions — just must get the order and batching right
communicate_info = ["exchange", "return"]
```

#### Reward Design

Same as existing adversarial templates for refusal cases:
```
correct_refusal (no forbidden actions + communicated policy) = 1.0
lookup + explained policy but still acted = 0.1 (forbidden action cap)
lookup + reasoning only = 0.4
nothing = 0.0
```

For T28 (operation selection), same as cooperative templates:
```
both operations correct + communicated = 1.0
one operation correct = 0.5-0.7 (partial)
wrong operation order (return before exchange) = 0.1 (state corruption)
```

#### Template Weights

```python
TRICKY_COOPERATIVE_WEIGHTS = {
    "T22": 3,  # basic economy modification (3 tau-bench tasks)
    "T23": 3,  # ineligible cancellation (2 tau-bench tasks, high weight for importance)
    "T24": 1,  # bag removal (1 task)
    "T25": 1,  # destination change (1 task)
    "T26": 2,  # conditional instruction (1 task, but common pattern)
    "T27": 2,  # unmet precondition (1 task, but common pattern)
    "T28": 3,  # return vs exchange batching (4 tau-bench tasks)
}
```

---

### Environment 2: Multi-Entity Tracking

**Targets**: 24 tasks (Skill 4)
**Effort**: Medium (new scenario generation logic for multi-entity)

#### What it tests

Agent discovers ALL relevant entities (orders/reservations), identifies the CORRECT one before acting, and doesn't act on wrong entities.

#### New Templates

**T29: Find-and-Act on Correct Order (Retail)**
- User has 3-5 orders
- User describes item by name/attribute, NOT by order ID
- Target item exists in exactly one order
- Other orders have similar-sounding items (distractors)
- Agent must fetch ALL orders, find the correct one, act on it

```python
# Scenario generation:
# 1. Select user with 3+ orders from DB
# 2. Pick target order + target item
# 3. Construct user message: "I want to return the [item_name]"
# 4. required_actions uses target order_id
# 5. forbidden_actions uses ALL other order_ids

required_actions = [
    {"name": "return_delivered_order_items", "check": "order_match",
     "args": {"order_id": target_order_id}}
]
forbidden_actions = [
    {"name": "return_delivered_order_items", "check": "order_match",
     "args": {"order_id": wrong_order_id}}
    for wrong_order_id in other_order_ids
]
```

**T30: Selective Cancellation (Airline)**
- User has 3-5 reservations
- Wants to "cancel all eligible reservations"
- Some eligible (business class, or have insurance, or within 24h)
- Some ineligible (basic economy, no insurance, >24h)
- Agent must check EACH reservation individually

```python
# Scenario generation:
# 1. Select user with 3+ reservations
# 2. Classify each as eligible/ineligible
# 3. required_actions = cancel for each eligible
# 4. forbidden_actions = cancel for each ineligible

required_actions = [
    {"name": "cancel_reservation", "check": "exact",
     "args": {"reservation_id": res_id}}
    for res_id in eligible_reservation_ids
]
forbidden_actions = [
    {"name": "cancel_reservation", "check": "exact",
     "args": {"reservation_id": res_id}}
    for res_id in ineligible_reservation_ids
]
```

**T31: Different Actions on Different Orders (Retail)**
- User has 2-3 orders
- Wants action A on order X, action B on order Y
- E.g., cancel order #1, modify order #2
- Agent must not swap which action goes to which order

```python
required_actions = [
    {"name": "cancel_pending_order", "check": "order_match",
     "args": {"order_id": cancel_order_id}},
    {"name": "modify_pending_order_items", "check": "order_match",
     "args": {"order_id": modify_order_id}},
]
forbidden_actions = [
    {"name": "cancel_pending_order", "check": "order_match",
     "args": {"order_id": modify_order_id}},  # don't cancel the one to modify
    {"name": "modify_pending_order_items", "check": "order_match",
     "args": {"order_id": cancel_order_id}},  # don't modify the one to cancel
]
```

**T32: Ambiguous Item Reference (Retail)**
- User describes item vaguely: "the expensive one" or "the bigger tablet"
- User has items across multiple orders that could match
- Agent must compare items across orders to disambiguate

```python
# Scenario generation:
# 1. Select user with 2+ orders containing similar items
# 2. Pick the one matching the user's description (e.g., more expensive)
# 3. Reward only for acting on the correct one
```

#### Reward Design

```
all correct entities acted on, no wrong entities = 1.0
looked up all entities + communicated info = 0.4
acted on wrong entity = 0.1 (same as forbidden action cap)
only checked 1 of N entities = 0.1 (lookup incomplete)
nothing = 0.0
```

**Key reward property**: Acting on the WRONG entity is penalized as harshly as a forbidden action (0.1 cap). This ensures a strong gradient between "act on wrong entity" (0.1) and "act on correct entity" (1.0).

#### Template Weights

```python
MULTI_ENTITY_WEIGHTS = {
    "T29": 4,  # find-and-act retail (most common pattern, 19 retail tasks)
    "T30": 2,  # selective cancellation airline (5 airline tasks)
    "T31": 3,  # different actions on different orders
    "T32": 2,  # ambiguous item reference
}
```

---

### Environment 3: Execution Under Time Pressure

**Targets**: 15 tasks (Skill 3)
**Effort**: Low (user LLM behavior change only)

#### What it tests

Agent executes write actions IMMEDIATELY upon having sufficient information, rather than summarizing and waiting for re-confirmation.

#### Implementation

No new scenario templates needed. Modify the cooperative user LLM behavior:

```python
# In LLMUser or build_user_system_prompt:
# For "prompt execution" scenarios, set MIN_USER_RESPONSES = 2
# Add to system prompt:
#   "When the agent has presented what they'll do and asks for confirmation,
#    confirm and end the conversation immediately with [DONE]."
#   "Do NOT ask follow-up questions after the agent's summary."

# Alternatively, in the user system prompt for cooperative scenarios:
# "After confirming, say 'Yes, please go ahead. [DONE]' and end immediately."
```

The key behavioral change: cooperative users confirm and LEAVE, rather than staying for multiple follow-up messages. This mirrors real tau-bench behavior where users end conversations quickly after confirmation.

#### How the reward creates signal

Current cooperative templates:
- User stays for 3-6 turns → agent always eventually executes → always 1.0 → no GRPO signal

With time pressure:
- Some rollouts: agent executes before user confirms → 1.0
- Some rollouts: agent summarizes, user confirms + leaves → 0.4 (action never executed)
- GRPO gap: 0.6, drives learning toward "execute first, explain after"

#### Which templates get time pressure

Apply to a subset of cooperative templates (T13-T21) randomly:
```python
# 50% of cooperative scenarios use time-pressure user behavior
# 50% use normal patient user behavior
# This prevents the model from learning to never explain anything
```

---

### Environment 4: Data Precision (Variant/Flight Selection)

**Targets**: 21 tasks (Skill 2)
**Effort**: Medium (pre-compute correct answers during scenario generation)

#### What it tests

Agent selects the OPTIMAL entity from structured search results, not just the first acceptable one.

#### New Templates

**T33: Cheapest Variant Selection (Retail)**
- User wants to exchange/modify to the cheapest available variant
- Agent must call `get_product_details`, compare ALL variant prices, select minimum
- Required action includes the exact correct variant_id

```python
# Scenario generation:
# 1. Select product with 5+ variants
# 2. Identify cheapest AVAILABLE variant
# 3. Set required_actions with that exact variant_id

required_actions = [
    {"name": "exchange_delivered_order_items", "check": "exact",
     "args": {"order_id": order_id, "item_ids": [current_item_id],
              "new_item_ids": [cheapest_variant_id],
              "payment_method_id": payment_id}}
]
```

**T34: Cheapest Flight Selection (Airline)**
- User wants to change to cheapest flight on a given route/date
- Agent must search flights, compare all prices, select minimum
- Required action includes the exact correct flight_number

```python
required_actions = [
    {"name": "update_reservation_flights", "check": "exact",
     "args": {"reservation_id": res_id}}
    # The flight_number in the tool call must match the actual cheapest
]
# Pre-compute: run search query during scenario gen to find cheapest
```

**T35: Attribute-Matching Variant Selection (Retail)**
- User wants specific attributes (e.g., "red, large, stainless steel")
- Agent must find the variant matching ALL specified attributes
- Only one variant matches exactly

```python
# Scenario generation:
# 1. Select product with variants differing in multiple attributes
# 2. Pick target variant with specific attribute combo
# 3. User message specifies the attributes
# 4. Required action includes exact variant_id
```

#### Reward Design

```
correct variant/flight selected = 1.0
looked up data + communicated + wrong variant selected = 0.4
  (reasoning credit but wrong action — drives GRPO toward correct selection)
looked up data only = 0.1
nothing = 0.0
```

**GRPO signal**: The model sometimes selects the correct variant and sometimes the wrong one (we saw this variance in the trajectories). The 0.6 gap between wrong-variant (0.4) and correct-variant (1.0) provides strong gradient.

#### Template Weights

```python
DATA_PRECISION_WEIGHTS = {
    "T33": 4,  # cheapest variant (most common retail failure)
    "T34": 2,  # cheapest flight (airline)
    "T35": 3,  # attribute matching
}
```

---

## Part 4: Template Pool Integration

### Updated Template Pools

```
ADVERSARIAL (T1-T12):      24 weight units  — adversarial user pressure
COOPERATIVE (T13-T21):      26 weight units  — straightforward legitimate requests
TRICKY_COOP (T22-T28):     15 weight units  — cooperative user + policy traps
MULTI_ENTITY (T29-T32):    11 weight units  — multiple orders/reservations
DATA_PRECISION (T33-T35):   9 weight units  — correct selection from structured data
```

### Scenario Selection Flow

```python
def generate_scenario(seed, adversarial_ratio=0.2, template_mix=None):
    rng = random.Random(seed)

    # Default mix reflecting tau-bench failure distribution
    if template_mix is None:
        template_mix = {
            "adversarial":    0.15,   # T1-T12  (protects against adversarial pressure)
            "cooperative":    0.20,   # T13-T21 (maintains basic competence)
            "tricky_coop":    0.25,   # T22-T28 (biggest policy compliance gap)
            "multi_entity":   0.25,   # T29-T32 (biggest overall gap)
            "data_precision": 0.15,   # T33-T35 (variant/flight selection)
        }

    # Select pool
    roll = rng.random()
    cumulative = 0
    for pool_name, prob in template_mix.items():
        cumulative += prob
        if roll < cumulative:
            break

    # Select template within pool
    template_id = _select_from_weights(rng, POOL_WEIGHTS[pool_name])
    return TEMPLATE_GENERATORS[template_id](rng)
```

### Execution Promptness Integration

Not a separate pool — applied as a behavioral modifier to cooperative and tricky_coop pools:

```python
# In generate_scenario, after selecting a cooperative/tricky_coop template:
if pool_name in ("cooperative", "tricky_coop") and rng.random() < 0.5:
    scenario.user_behavior = "prompt"  # user confirms and leaves quickly
    scenario.min_user_responses = 2    # instead of default 3
```

---

## Part 5: Expected Impact

### Coverage by Environment

| Environment | New Templates | Tasks Addressed | % of 102 Failures |
|-------------|---------------|-----------------|-------------------|
| Tricky Cooperative | T22-T28 | ~25 (Skill 1 non-adv + Skill 6) | 24.5% |
| Multi-Entity | T29-T32 | ~24 (Skill 4) | 23.5% |
| Execution Promptness | (behavioral) | ~15 (Skill 3) | 14.7% |
| Data Precision | T33-T35 | ~21 (Skill 2) | 20.6% |
| **Total new** | **14 templates** | **~85** | **83.3%** |
| Existing adversarial | T1-T12 | ~4 (true adversarial) | 3.9% |
| Existing cooperative | T13-T21 | ~0 (model already succeeds) | 0% |

### What remains unaddressed

| Skill | Tasks | Why not addressed |
|-------|-------|-------------------|
| Skill 5 (Numerical) | 9 | Arithmetic errors don't respond well to RL policy gradients |
| Skill 7 (Loop detection) | 2 | Meta-cognitive skill, very few tasks |

### GRPO Signal Expectations

Each new environment category should have reward variance across rollouts:

| Environment | Expected reward distribution | GRPO gap |
|-------------|---------------------------|----------|
| Tricky Cooperative | ~50% refusals (1.0), ~50% policy violations (0.1) | 0.9 |
| Multi-Entity | ~30% correct entity (1.0), ~70% wrong entity (0.1-0.4) | 0.6-0.9 |
| Execution Promptness | ~60% execute in time (1.0), ~40% too slow (0.4) | 0.6 |
| Data Precision | ~40% correct variant (1.0), ~60% wrong variant (0.4) | 0.6 |

All gaps are >= 0.6, which matches or exceeds the current adversarial template gap.

---

## Part 6: Implementation Priority

| Priority | Environment | Tasks | Effort | Why first |
|----------|------------|-------|--------|-----------|
| **1** | Tricky Cooperative (T22-T28) | ~25 | Low | New templates only, reuses all existing infrastructure |
| **2** | Execution Promptness | ~15 | Low | User LLM parameter change only |
| **3** | Multi-Entity (T29-T32) | ~24 | Medium | Needs multi-entity scenario generation from DB |
| **4** | Data Precision (T33-T35) | ~21 | Medium | Needs pre-computed correct answers |

**Start with Priority 1 + 2**: Together they address ~40 tasks with minimal implementation effort. Both are straightforward extensions of the existing adversarial_policy_game.

**Then Priority 3**: The biggest single failure category (24 tasks) but requires more scenario generation logic to create multi-entity scenarios from the existing database.

**Then Priority 4**: Addresses 21 tasks but requires pre-computing correct answers (cheapest variant, cheapest flight) during scenario generation.

---

## Part 7: Relation to Existing Documents

- **`cooperative_templates_design.md`**: Designed T13-T21 (straightforward cooperative tasks). These are necessary but insufficient — model already succeeds on them. This document extends beyond cooperative into "tricky cooperative" and multi-entity.
- **`adversarial_policy_game_architecture.md`**: Documents the game loop, reward function, and tool infrastructure. All new environments reuse this architecture.
- **`qwen3-30b-failure-analysis.md`**: Source of the 7-skill taxonomy. This document translates those skills into concrete training environments.

---

## Appendix: Tau-Bench Task Coverage Map

### Tricky Cooperative Templates (T22-T28) → Skill 1 + 6

| Template | Tau-bench tasks directly addressed |
|----------|-----------------------------------|
| T22 (basic economy mod) | Airline: 14, 19, 31 |
| T23 (ineligible cancel) | Airline: 1, 24 |
| T24 (bag removal) | Airline: 30 |
| T25 (destination change) | Airline: 13 |
| T26 (conditional instruction) | Retail: 31, 57 |
| T27 (unmet precondition) | Retail: 62 |
| T28 (return vs exchange) | Retail: 27, 29, 98, 99 |
| General policy compliance | Airline: 7, 9, 34, 35, 39, 49; Retail: 10, 16, 36, 37, 43, 45, 91, 100 |

### Multi-Entity Templates (T29-T32) → Skill 4

| Template | Tau-bench tasks directly addressed |
|----------|-----------------------------------|
| T29 (find correct order) | Retail: 4, 9, 41, 42, 51, 82, 83, 93, 101, 107, 109 |
| T30 (selective cancel) | Airline: 38, 41, 42 |
| T31 (different actions) | Retail: 59, 74, 78, 110 |
| T32 (ambiguous reference) | Airline: 17, 22; Retail: 20, 30, 104 |

### Execution Promptness → Skill 3

| Mechanism | Tau-bench tasks directly addressed |
|-----------|-----------------------------------|
| Time-pressure user | Retail: 0, 60, 72, 95, 108, 112; Airline: 8 |
| Multi-task completion | Retail: 33, 55, 71, 102; Airline: 18, 27, 33 |

### Data Precision Templates (T33-T35) → Skill 2

| Template | Tau-bench tasks directly addressed |
|----------|-----------------------------------|
| T33 (cheapest variant) | Retail: 44, 56, 64 |
| T34 (cheapest flight) | Airline: 15, 29 |
| T35 (attribute matching) | Retail: 14, 18, 21, 23, 28, 49, 79, 103; Airline: 11, 37 |
| Identity resolution | Retail: 35, 39, 54, 67, 97 |
