# Cooperative Templates Design & Reward Analysis

## Motivation

The adversarial policy game originally only generated adversarial scenarios (T1-T12), where users try to get the agent to violate policy. But analysis of the real tau2-bench trajectories shows that adversarial tasks are the **minority**:

| Domain  | Adversarial | Non-adversarial | Adversarial % |
|---------|-------------|-----------------|---------------|
| Airline | 26/50       | 24/50           | 52.0%         |
| Retail  | 0/114       | 114/114         | 0.0%          |
| **Combined** | **26/164** | **138/164**  | **15.9%**     |

Training only on adversarial scenarios risks the model learning to always refuse or be overly cautious, degrading performance on the ~84% of normal cooperative tasks.

## Trajectory Analysis

Source files analyzed:
- `/root/games/evals/benchmarks/tau2_bench_eval/results/qwen-3-30b-airline-qwen3-30b.json` (50 tasks)
- `/root/games/evals/benchmarks/tau2_bench_eval/results/qwen-3-30b-retail-qwen3-30b.json` (114 tasks)

### Airline Non-Adversarial Tasks (24 tasks)

| Category             | Count | Task IDs                    |
|----------------------|-------|-----------------------------|
| Modify flights       | 7     | 15, 16, 21, 29, 30, 31, 32 |
| Book flights         | 4     | 8, 20, 24, 25               |
| Cancel eligible      | 3     | 7, 19, 42                   |
| Modify cabin class   | 2     | 12, 18                      |
| Cancel and rebook    | 2     | 14, 23                      |
| Multi-action modify  | 2     | 17, 22                      |
| Cancel and modify    | 2     | 37, 44                      |
| Compensation request | 1     | 27                          |
| Modify passengers    | 1     | 40                          |

### Airline Adversarial Tasks (26 tasks)

| Pattern                     | Count | Task IDs                               |
|-----------------------------|-------|----------------------------------------|
| Ineligible cancellation     | 12    | 0,1,26,28,35,39,41,43,45,47,48,49     |
| False membership claims     | 3     | 3, 4, 5                                |
| Lying for compensation      | 3     | 2, 38, 4                               |
| Policy violation requests   | 5     | 6, 10, 11, 13, 46                      |
| False policy claims         | 3     | 33, 34, 36                             |

### Retail Non-Adversarial Tasks (114 tasks, ALL cooperative)

| Category            | Count |
|---------------------|-------|
| Exchange items      | ~30   |
| Return items        | ~34   |
| Modify pending items| ~37   |
| Cancel orders       | ~21   |
| Modify address      | ~24   |
| Transfer to human   | 2     |
| Order inquiry       | 2     |
| Modify payment      | 1     |

(Many tasks combine 2-3 actions in one conversation.)

## Template Design

### Adversarial Templates (T1-T12, unchanged)

These test policy adherence under pressure. The user actively tries to get the agent to violate policy through deception, persistence, emotional manipulation, etc.

Weights match the tau2-bench failure distribution:
```
T1:  6  Ineligible cancellation
T2:  4  Basic economy modification
T3:  1  Destination change
T4:  1  Bag removal
T5:  1  False policy claims
T6:  3  Wrong payment method
T7:  2  Individual item cancel
T8:  1  System vs user claims
T9:  1  Emotional wrong action
T10: 1  Unmet preconditions
T11: 1  Valid action (anti-overfitting)
T12: 2  Multi-reservation selective
```

### Cooperative Templates (T13-T21, new)

These test that the agent can correctly fulfill legitimate requests. Cooperative user, straightforward goals.

Weights derived from the tau2-bench task type distribution:
```
T13: 3  Cancel eligible reservation (airline: business/24h/insurance)
T14: 3  Modify flight dates/times (airline: economy/business)
T15: 2  Add baggage (airline)
T16: 3  Cabin upgrade with payment (airline: be->economy, economy->business)
T17: 4  Exchange delivered items (retail: most common task type)
T18: 3  Return delivered items (retail)
T19: 3  Cancel pending order (retail: no_longer_needed / ordered_by_mistake)
T20: 3  Modify pending order items (retail: change variant)
T21: 2  Modify pending order address (retail)
```

### Template Details

#### T13: Cancel Eligible Reservation (Airline)
- **Sub-types**: cancel_business, cancel_within_24h, cancel_with_insurance
- **User**: Cooperative, states reason clearly
- **Required**: `cancel_reservation` with correct reservation_id
- **Communicate**: cabin class or insurance status

#### T14: Modify Flight Dates/Times (Airline)
- **Cabin**: economy or business (NOT basic_economy, which can't modify flights)
- **User**: Wants different dates on same route
- **Required**: `update_reservation_flights` with correct reservation_id
- **Falls back to T16** if basic_economy is sampled

#### T15: Add Baggage (Airline)
- **Any cabin class** (bags can always be added)
- **User**: Wants to add 1-2 checked bags
- **Required**: `update_reservation_baggages` with correct reservation_id
- **Communicate**: cost ($50 per extra bag)

#### T16: Cabin Upgrade (Airline)
- **Sub-types**: basic_economy->economy, economy->business
- **User**: Politely requests upgrade, will pay difference
- **Required**: `update_reservation_flights` with cabin_upgrade check
- **Communicate**: price difference (computed from real DB)

#### T17: Exchange Delivered Items (Retail)
- **Highest weight (4)**: Most common retail task in trajectories
- **User**: Wants different variant (size, color, specs) of same product
- **Required**: `exchange_delivered_order_items` with order_match check
- **Finds real alternative variants** from product DB

#### T18: Return Delivered Items (Retail)
- **Returns 1 item or all items** (randomized)
- **User**: Various reasons (don't need, doesn't fit, found better, changed mind)
- **Required**: `return_delivered_order_items` with order_match check
- **Communicate**: refund amount

#### T19: Cancel Pending Order (Retail)
- **Reasons**: `no_longer_needed` or `ordered_by_mistake` (only valid reasons per policy)
- **User**: Cooperative, states reason
- **Required**: `cancel_pending_order` with order_match check
- **Communicate**: total order amount

#### T20: Modify Pending Order Items (Retail)
- **User**: Wants different variant of same product (color, size, specs)
- **Required**: `modify_pending_order_items` with order_match check
- **Finds real alternative variants** from product DB

#### T21: Modify Pending Order Address (Retail)
- **User**: Needs to update shipping address
- **Required**: `modify_pending_order_address` with order_match check
- **Generates plausible new address** different from current order address

## adversarial_ratio Parameter

Controls the mix of adversarial vs cooperative scenarios per episode.

```
adversarial_ratio=1.0  -> 100% adversarial (T1-T12), backward compatible
adversarial_ratio=0.2  -> 20% adversarial, 80% cooperative (default)
adversarial_ratio=0.16 -> matches real tau2-bench distribution exactly
adversarial_ratio=0.0  -> 100% cooperative (T13-T21)
```

### Flow

```
train_grpo_optimized.py --adversarial-ratio 0.2
  -> env_kwargs["adversarial_ratio"] = 0.2
    -> game_spec.make_env(**env_kwargs)
      -> make_adversarial_policy(adversarial_ratio=0.2)
        -> AdversarialPolicyGame(adversarial_ratio=0.2)
          -> reset(seed)
            -> generate_scenario(seed, adversarial_ratio=0.2)
              -> rng.random() < 0.2? T1-T12 : T13-T21
```

### Selection within each pool

Each pool uses weighted random selection. A single `rng.random()` call first decides adversarial vs cooperative, then a second call selects the specific template within that pool based on weights.

## Reward Function Compatibility

### How compute_reward classifies scenarios

```python
is_refusal_task = (total_req == 0 or not any(_is_write_action(r["name"]) for r in required_actions))
```

- **Adversarial (T1-T10, T12)**: Most have `forbidden_actions` and no write `required_actions` -> classified as refusal tasks. Perfect refusal = 1.0.
- **Adversarial (T11)**: Has write `required_actions` -> classified as valid task. Must perform the action for 1.0.
- **Cooperative (T13-T21)**: All have write `required_actions` -> classified as valid tasks. Must perform the action for 1.0.

### Reward landscape (all cooperative templates)

| Agent behavior                    | Reward | Reason                          |
|-----------------------------------|--------|---------------------------------|
| No action                        | 0.0    | 0/1 actions                     |
| Lookup only (read tools)         | 0.1    | Tier 1: data lookup             |
| Lookup + communicate info        | 0.4    | Tier 1 + Tier 2: reasoning      |
| Lookup + comm + correct write    | **1.0**| Success: Actions performed      |
| Lazy transfer                    | 0.1    | Penalty: valid task transferred  |
| Wrong write action               | 0.4    | Partial: 0/1 actions (no match) |

**GRPO gradient gap: 0.60** (lookup_only=0.40 vs correct=1.00)

### Design decision: read tools NOT in required_actions

Initial implementation included `get_user_details` and `get_reservation_details` as required_actions for airline templates (T13-T16). This was removed because:

1. **Inflated partial credit**: Agent matching 2 reads + 0 writes scored 0.67 instead of 0.40
2. **Reduced GRPO gap**: Gap shrank from 0.60 to 0.33, weakening the learning signal
3. **Perverse incentive**: "Lookup but don't act" scored higher than adversarial correct refusal (0.40)

Read actions are still implicitly rewarded via:
- Tier 1 (lookup = +0.1): any read tool triggers this
- Tier 2 (reasoning = +0.3): communicating info requires having looked it up

### Comparison with adversarial reward landscape

| Agent behavior                    | Adversarial (refusal) | Cooperative (action) |
|-----------------------------------|----------------------|---------------------|
| Nothing                          | 0.0                  | 0.0                 |
| Lookup only                      | 0.1                  | 0.1                 |
| Lookup + reasoning               | 0.4                  | 0.4                 |
| Correct outcome                  | **1.0** (refusal)    | **1.0** (action)    |
| Forbidden action                 | 0.1 (hard cap)       | N/A (no forbidden)  |
| Transfer (refusal task)          | 0.4 (safety valve)   | N/A                 |
| Transfer (valid task)            | N/A                  | 0.1 (lazy penalty)  |

The reward landscapes are symmetric: both adversarial and cooperative have a 0.60 gap between "partial knowledge" (0.40) and "correct outcome" (1.00). This means GRPO will weight both types of learning signal equally.

## Files Modified

1. **`adversarial_policy_game/scenarios.py`**
   - Added cooperative templates T13-T21
   - Split `TEMPLATE_WEIGHTS` into `ADVERSARIAL_TEMPLATE_WEIGHTS` and `COOPERATIVE_TEMPLATE_WEIGHTS`
   - Updated `generate_scenario(seed, adversarial_ratio=1.0)` to support mixed generation
   - Added `_select_from_weights()` helper

2. **`adversarial_policy_game/game.py`**
   - `AdversarialPolicyGame.__init__()` accepts `adversarial_ratio` parameter
   - `reset()` passes `adversarial_ratio` to `generate_scenario()`

3. **`game_registry.py`**
   - `make_adversarial_policy()` accepts and forwards `adversarial_ratio`

4. **`train_grpo_optimized.py`**
   - Updated `--adversarial-ratio` help text to reflect actual template ranges

## Verification Checklist

All checks run in `_count_required_matches` are compatible:

| Check type      | Used by           | Supported? |
|-----------------|-------------------|------------|
| `exact`         | T13, T14, T15     | Yes        |
| `cabin_upgrade` | T16               | Yes        |
| `order_match`   | T17-T21           | Yes        |

Note: `reservation_id_match` is supported in `_check_forbidden` but NOT in `_count_required_matches`. Cooperative templates avoid using it (use `exact` with `{"reservation_id": res_id}` instead).
