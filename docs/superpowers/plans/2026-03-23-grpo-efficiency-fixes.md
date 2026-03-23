# GRPO Efficiency Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate zero-gradient GRPO iterations for multistep_task by creating within-group difficulty variation, and add self-play SFT from successful rollouts to teach behavioral patterns the model currently lacks.

**Architecture:** Two independent changes. (1) `generate_scenario` and `RealisticMultiStepGame.reset` gain a `max_ops` parameter that truncates the operation list before building the conversation — creating difficulty variation within GRPO groups so identical-action games get different rewards. (2) `SFTBuffer` gains an `add_from_rollouts` method that ingests successful `GRPOSample` objects; the training loop pipes reward>=1.0 trajectories into it after each rollout collection. Both changes are additive — existing behavior is preserved when the new parameters are unused.

**Tech Stack:** Python, PyTorch, existing game/training infrastructure

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `multistep_task_game.py` | Modify | Add `max_ops` to `generate_scenario`, `_generate_retail_scenario`, and `RealisticMultiStepGame.reset` |
| `sft_buffer.py` | Modify | Add `add_from_rollouts` method to `SFTBuffer` |
| `train_grpo.py` | Modify | Pass `max_ops` per game in `collect_grpo_rollouts`; pipe successful rollouts into SFT buffer |
| `train_grpo_optimized.py` | Modify | Initialize SFT buffer unconditionally; pipe successful rollouts after collection |
| `train_mixed_grpo_cmd.sh` | Modify | Add `--sft-coef 0.1` flag |
| `tests/test_max_ops.py` | Create | Tests for `max_ops` truncation and reward correctness |
| `tests/test_sft_from_rollouts.py` | Create | Tests for SFT buffer ingestion from `GRPOSample` objects |

---

### Task 1: Add `max_ops` to `generate_scenario` (airline path)

**Files:**
- Modify: `multistep_task_game.py:952-1051`
- Test: `tests/test_max_ops.py`

- [ ] **Step 1: Write failing test for max_ops truncation**

```python
# tests/test_max_ops.py
import sys
sys.path.insert(0, ".")

from multistep_task_game import generate_scenario

def test_max_ops_truncates_airline():
    """max_ops=1 on a 3-op airline scenario produces 1 operation."""
    seed = 2087043557  # Known 3-op airline scenario
    full = generate_scenario(seed)
    assert len(full.operations) == 3, f"Expected 3 ops, got {len(full.operations)}"

    truncated = generate_scenario(seed, max_ops=1)
    assert len(truncated.operations) == 1
    assert truncated.operations[0].tool_name == full.operations[0].tool_name
    assert truncated.domain == full.domain

def test_max_ops_none_unchanged():
    """max_ops=None preserves original behavior."""
    seed = 2087043557
    original = generate_scenario(seed)
    with_none = generate_scenario(seed, max_ops=None)
    assert len(original.operations) == len(with_none.operations)

def test_max_ops_larger_than_n_ops():
    """max_ops > actual ops does not crash or add operations."""
    seed = 2087043557  # 3-op scenario
    result = generate_scenario(seed, max_ops=10)
    assert len(result.operations) == 3

def test_max_ops_changes_user_message():
    """Truncated scenario has shorter user message."""
    seed = 2087043557
    full = generate_scenario(seed)
    trunc = generate_scenario(seed, max_ops=1)
    full_user = [m for m in full.messages if m.get("role") == "user"][0]["content"]
    trunc_user = [m for m in trunc.messages if m.get("role") == "user"][0]["content"]
    assert len(trunc_user) < len(full_user)
    # The truncated message should NOT mention the 3rd operation
    assert "MS30465" not in trunc_user
    assert "MS45397" in trunc_user  # First op still present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_max_ops.py -v`
Expected: FAIL — `generate_scenario() got an unexpected keyword argument 'max_ops'`

- [ ] **Step 3: Implement max_ops in generate_scenario (airline path)**

In `multistep_task_game.py`, modify the function signature at line 952 and add truncation after the operations loop:

```python
# Line 952: Change signature
def generate_scenario(seed: int, domain: Optional[str] = None, max_ops: Optional[int] = None) -> MultiStepScenario:
    """Generate a multi-step scenario with 2-5 operations.

    Args:
        seed: Random seed for reproducibility.
        domain: "airline", "retail", or None (50/50 random).
        max_ops: If set, truncate operations to at most this many.
    """
    rng = random.Random(seed)

    # Choose domain
    if domain is None:
        chosen_domain = rng.choice(["airline", "retail"])
    else:
        chosen_domain = domain

    if chosen_domain == "retail":
        return _generate_retail_scenario(rng, max_ops=max_ops)

    # --- Airline path --- (existing code through line 1038 unchanged)
    ...

    # ADD after line 1038 (after operations are fully built, before db/messages):
    # Truncate operations if max_ops is set (minimum 1 to avoid empty scenarios)
    if max_ops is not None:
        max_ops = max(1, max_ops)
        if len(operations) > max_ops:
            operations = operations[:max_ops]

    db = build_airline_db(user, reservations, flights_db)
    messages = _build_prefilled_conversation("airline", user, operations, db)
    ...
```

The key insertion point: after the operations list is finalized (line ~1038) but BEFORE `_build_prefilled_conversation` is called (line 1042). This ensures the conversation reflects the truncated operations.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_max_ops.py -v`
Expected: PASS for airline tests, FAIL for retail (not yet implemented)

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/hangook/games
git add multistep_task_game.py tests/test_max_ops.py
git commit -m "feat: add max_ops parameter to generate_scenario (airline path)"
```

---

### Task 2: Add `max_ops` to retail path and `RealisticMultiStepGame.reset`

**Files:**
- Modify: `multistep_task_game.py:848-949` (retail), `multistep_task_game.py:1219-1238` (reset)
- Test: `tests/test_max_ops.py`

- [ ] **Step 1: Write failing tests for retail max_ops and reset**

```python
# Append to tests/test_max_ops.py

def test_max_ops_truncates_retail():
    """max_ops=1 on a retail scenario truncates correctly."""
    # Find a retail seed
    for seed in range(100):
        s = generate_scenario(seed)
        if s.domain == "retail" and len(s.operations) >= 2:
            break
    full_n = len(s.operations)
    trunc = generate_scenario(seed, max_ops=1)
    assert len(trunc.operations) == 1
    assert trunc.domain == "retail"

def test_reset_with_max_ops():
    """RealisticMultiStepGame.reset accepts max_ops."""
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame()
    game.reset(2087043557, max_ops=1)
    # Should not crash, and scenario should have 1 op
    summary = game.get_summary()
    assert summary["n_ops"] == 1

def test_reward_with_truncated_ops():
    """Reward evaluation uses truncated operation list."""
    from multistep_task_game import compute_reward
    seed = 2087043557
    full = generate_scenario(seed)
    # Model does just the first cancel
    calls = [{"name": "cancel_reservation", "arguments": {"reservation_id": "MS45397"}}]
    # Against 1-op: should be 1.0 (all ops completed)
    r1, _ = compute_reward(calls, full.operations[:1])
    assert r1 == 1.0
    # Against full 3-op: should be 0.0 (only 1/3)
    r3, _ = compute_reward(calls, full.operations)
    assert r3 == 0.0
```

- [ ] **Step 2: Run test to verify failures**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_max_ops.py -v`
Expected: `test_max_ops_truncates_retail` and `test_reset_with_max_ops` FAIL

- [ ] **Step 3: Implement retail truncation and reset parameter**

In `multistep_task_game.py`:

1. Modify `_generate_retail_scenario` (line 848) to accept `max_ops`:

```python
def _generate_retail_scenario(rng: random.Random, max_ops: Optional[int] = None) -> MultiStepScenario:
    ...
    # After line 936 (operations finalized), before line 938 (db build):
    if max_ops is not None:
        max_ops = max(1, max_ops)
        if len(operations) > max_ops:
            operations = operations[:max_ops]

    db = build_retail_db(user, orders, products_db)
    messages = _build_prefilled_conversation("retail", user, operations, db)
    ...
```

2. Modify `RealisticMultiStepGame.reset` (line 1219):

```python
def reset(self, seed: int, max_ops: Optional[int] = None) -> None:
    """Reset with new seed."""
    self._scenario = generate_scenario(seed, domain=self._domain, max_ops=max_ops)
    ...
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_max_ops.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/hangook/games
git add multistep_task_game.py tests/test_max_ops.py
git commit -m "feat: add max_ops to retail path and RealisticMultiStepGame.reset"
```

---

### Task 3: Wire `max_ops` into `collect_grpo_rollouts`

**Files:**
- Modify: `train_grpo.py:192-200` (batch mode), `train_grpo.py:342-350` (sequential mode)
- Test: `tests/test_max_ops.py`

- [ ] **Step 1: Write test for difficulty schedule**

```python
# Append to tests/test_max_ops.py

def test_difficulty_schedule():
    """The difficulty schedule assigns varying max_ops within a group."""
    # Import the schedule function we'll create
    from train_grpo import multistep_difficulty_schedule

    # group_size=8: should return [1, 1, 2, 2, None, None, None, None]
    schedule = multistep_difficulty_schedule(8)
    assert len(schedule) == 8
    assert schedule[0] == 1 and schedule[1] == 1  # easy slots
    assert schedule[2] == 2 and schedule[3] == 2  # medium slots
    assert schedule[4] is None  # full difficulty
    assert schedule[7] is None

    # group_size=4
    schedule4 = multistep_difficulty_schedule(4)
    assert len(schedule4) == 4
    assert schedule4[0] == 1  # at least one easy slot
    assert schedule4[3] is None  # at least one full slot
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_max_ops.py::test_difficulty_schedule -v`
Expected: FAIL — `cannot import name 'multistep_difficulty_schedule'`

- [ ] **Step 3: Implement difficulty schedule and wire into rollout collection**

In `train_grpo.py`, add the schedule function near the top (after imports, before `collect_grpo_rollouts`):

```python
def multistep_difficulty_schedule(group_size: int) -> List[Optional[int]]:
    """Assign max_ops values for each game in a multistep_task group.

    Creates within-group difficulty variation:
      - 25% of games: max_ops=1 (easiest)
      - 25% of games: max_ops=2 (medium)
      - 50% of games: max_ops=None (full difficulty)

    This ensures non-constant rewards even when the model produces
    deterministic actions, because easier variants are more likely
    to succeed than harder ones.
    """
    schedule: List[Optional[int]] = []
    n_easy = max(1, group_size // 4)
    n_medium = max(1, group_size // 4)
    for i in range(group_size):
        if i < n_easy:
            schedule.append(1)
        elif i < n_easy + n_medium:
            schedule.append(2)
        else:
            schedule.append(None)
    return schedule
```

Then modify the two `env.reset(g_seed)` call sites.

**Batch mode (line ~192-200):**

```python
            # Compute multistep difficulty schedule once per group
            ms_schedule = None
            if g_spec.name == "multistep_task":
                ms_schedule = multistep_difficulty_schedule(group_size)

            for s_idx in range(group_size):
                env = g_spec.make_env(**g_kwargs)
                difficulty = None
                if g_spec.name == "adversarial_policy":
                    difficulty = rng.choice(["easy", "medium", "hard"])
                    env.reset(g_seed, user_difficulty=difficulty)
                elif ms_schedule is not None:
                    env.reset(g_seed, max_ops=ms_schedule[s_idx])
                else:
                    env.reset(g_seed)
```

**Sequential mode (line ~342-350):** Apply the same pattern.

- [ ] **Step 4: Run all tests**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_max_ops.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/hangook/games
git add train_grpo.py tests/test_max_ops.py
git commit -m "feat: wire max_ops difficulty schedule into collect_grpo_rollouts"
```

---

### Task 4: Add `add_from_rollouts` to `SFTBuffer`

**Files:**
- Modify: `sft_buffer.py:315-460`
- Test: `tests/test_sft_from_rollouts.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_sft_from_rollouts.py
import sys
sys.path.insert(0, ".")

import torch
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Optional

@dataclass
class FakeGRPOSample:
    prompt_msgs: list
    completion_text: str
    player_id: int
    reward: float
    group_id: int
    game_id: int
    tools: Optional[list] = None
    game_name: str = ""

def test_add_from_rollouts_filters_by_reward():
    """Only samples with reward >= min_reward are added."""
    from sft_buffer import SFTBuffer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507",
                                        trust_remote_code=True)

    buf = SFTBuffer([], tok, compact_tools=True)
    assert len(buf) == 0

    samples = [
        FakeGRPOSample(
            prompt_msgs=[{"role": "system", "content": "You are helpful."},
                         {"role": "user", "content": "Cancel order 123."}],
            completion_text='<tool_call>\n{"name": "cancel_order", "arguments": {"order_id": "123"}}\n</tool_call>',
            player_id=0, reward=1.0, group_id=0, game_id=100,
            game_name="multistep_task",
        ),
        FakeGRPOSample(
            prompt_msgs=[{"role": "system", "content": "You are helpful."},
                         {"role": "user", "content": "Cancel order 456."}],
            completion_text='<tool_call>\n{"name": "cancel_order", "arguments": {"order_id": "456"}}\n</tool_call>',
            player_id=0, reward=0.0, group_id=0, game_id=101,
            game_name="multistep_task",
        ),
    ]

    added = buf.add_from_rollouts(samples, min_reward=1.0)
    assert added == 1  # Only the reward=1.0 sample
    assert len(buf) == 1

def test_add_from_rollouts_deduplicates():
    """Same game_id is not added twice."""
    from sft_buffer import SFTBuffer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507",
                                        trust_remote_code=True)
    buf = SFTBuffer([], tok, compact_tools=True)

    sample = FakeGRPOSample(
        prompt_msgs=[{"role": "system", "content": "test"},
                     {"role": "user", "content": "do thing"}],
        completion_text='<tool_call>\n{"name": "foo", "arguments": {}}\n</tool_call>',
        player_id=0, reward=1.0, group_id=0, game_id=200,
        game_name="multistep_task",
    )

    buf.add_from_rollouts([sample], min_reward=1.0)
    buf.add_from_rollouts([sample], min_reward=1.0)
    assert len(buf) == 1  # Deduped by game_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_sft_from_rollouts.py -v`
Expected: FAIL — `SFTBuffer has no attribute 'add_from_rollouts'`

- [ ] **Step 3: Implement `add_from_rollouts`**

Add to `sft_buffer.py` class `SFTBuffer`, after the `refresh` method (after line 436):

```python
    def add_from_rollouts(self, samples, min_reward: float = 1.0,
                          max_buffer_size: int = 5000) -> int:
        """Add successful GRPO rollout samples to the buffer.

        Args:
            samples: List of GRPOSample (or any object with prompt_msgs,
                     completion_text, tools, game_id, reward attributes).
            min_reward: Minimum reward to include.
            max_buffer_size: Cap buffer at this size (drop oldest when exceeded).

        Returns:
            Number of new samples added.
        """
        new_count = 0
        for s in samples:
            if s.reward < min_reward:
                continue
            # Dedup by game_id (stored as "rollout::{game_id}" in _seen_tasks)
            key = f"rollout::{s.game_id}"
            if key in self._seen_tasks:
                continue
            try:
                seq, pl, al = build_prompt_plus_action(
                    self._tokenizer, s.prompt_msgs, s.completion_text,
                    tools=s.tools,
                )
                total_len = seq.shape[0]
                if self._max_seq_len and total_len > self._max_seq_len:
                    excess = total_len - self._max_seq_len
                    new_pl = pl - excess
                    if new_pl < self._min_prompt_len:
                        continue
                    seq = seq[excess:]
                    pl = new_pl
                self._seqs_cpu.append(seq)
                self._prompt_lens.append(pl)
                self._action_lens.append(al)
                self._seen_tasks.add(key)
                new_count += 1
            except Exception:
                continue

        # Cap buffer size: drop oldest entries.
        # Note: _seen_tasks keys for evicted entries are NOT removed, preventing
        # re-addition of the same game_id. This is acceptable because game_ids
        # include iteration-based offsets and are never reused across iterations.
        if len(self._seqs_cpu) > max_buffer_size:
            excess = len(self._seqs_cpu) - max_buffer_size
            self._seqs_cpu = self._seqs_cpu[excess:]
            self._prompt_lens = self._prompt_lens[excess:]
            self._action_lens = self._action_lens[excess:]

        return new_count
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_sft_from_rollouts.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/hangook/games
git add sft_buffer.py tests/test_sft_from_rollouts.py
git commit -m "feat: add add_from_rollouts to SFTBuffer for self-play SFT"
```

---

### Task 5: Wire SFT buffer into training loop

**Files:**
- Modify: `train_grpo_optimized.py:730-737` (init), `train_grpo_optimized.py:882-886` (after collection)
- Modify: `train_mixed_grpo_cmd.sh`

- [ ] **Step 1: Modify SFT buffer initialization to work without --sft-data**

In `train_grpo_optimized.py`, change the SFT buffer init (lines 730-737) to always create a buffer when `--sft-coef > 0`, even without `--sft-data`:

```python
    # ---- SFT buffer initialization ----
    sft_buffer = None
    if args.sft_coef > 0:
        sft_paths = []
        if args.sft_data:
            sft_paths = [p.strip() for p in args.sft_data.split(",") if p.strip()]
        sft_buffer = SFTBuffer(sft_paths, tokenizer, compact_tools=args.compact_tools,
                               max_seq_len=Config.MAX_SEQ_LENGTH)
        if sft_paths:
            print(f"[SFT] Initialized buffer: {len(sft_buffer)} samples from {len(sft_paths)} files")
        else:
            print(f"[SFT] Initialized empty buffer (will fill from successful rollouts)")
        print(f"[SFT] coef={args.sft_coef}, per_step={args.sft_per_step}")
```

- [ ] **Step 2: Add rollout-to-SFT piping after collection**

In `train_grpo_optimized.py`, after rollout collection completes (after line 885 `samples = accumulated_samples`), add:

```python
            # ---- Pipe successful rollouts into SFT buffer ----
            if sft_buffer is not None and samples:
                n_added = sft_buffer.add_from_rollouts(samples, min_reward=1.0)
                if n_added > 0:
                    print(f"[SFT] Added {n_added} successful rollouts (buffer: {len(sft_buffer)})")
```

- [ ] **Step 3: Update training command**

In `train_mixed_grpo_cmd.sh`, add `--sft-coef 0.5` to the command. Note: the SFT buffer is only populated on rank 0 (pre-existing limitation), so after gradient allreduce across 6 ranks, the effective coefficient is `0.5 / 6 ≈ 0.08`.

```bash
    --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --sft-coef 0.5
```

- [ ] **Step 4: Verify the training script parses correctly**

Run: `cd /home/ubuntu/hangook/games && python train_grpo_optimized.py --help | grep -A1 sft`
Expected: Shows `--sft-coef` and `--sft-data` options

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/hangook/games
git add train_grpo_optimized.py train_mixed_grpo_cmd.sh
git commit -m "feat: wire self-play SFT into training loop, enable --sft-coef 0.1"
```

---

### Task 6: Integration test — full scenario verification

**Files:**
- Test: `tests/test_max_ops.py` (append)

- [ ] **Step 1: Write integration test that simulates a full GRPO group with difficulty variation**

```python
# Append to tests/test_max_ops.py

def test_difficulty_variation_creates_informative_group():
    """A previously constant-zero group becomes informative with max_ops variation."""
    from multistep_task_game import generate_scenario, compute_reward
    from train_grpo import multistep_difficulty_schedule

    seed = 2087043557  # Known all-zero group in original training
    schedule = multistep_difficulty_schedule(8)

    rewards = []
    for max_ops in schedule:
        scenario = generate_scenario(seed, max_ops=max_ops)
        # Simulate the model's known behavior: cancel(A) + baggages(B, wrong) + cancel(C)
        model_calls = [
            {"name": "cancel_reservation", "arguments": {"reservation_id": "MS45397"}},
            {"name": "update_reservation_baggages", "arguments": {
                "reservation_id": "MS24190", "total_baggages": 2,
                "nonfree_baggages": 2, "payment_id": "credit_card_4112902"}},
            {"name": "cancel_reservation", "arguments": {"reservation_id": "MS30465"}},
        ]
        # Only pass calls up to max_ops (model would follow the prompt)
        if max_ops is not None:
            model_calls = model_calls[:max_ops]
        reward, _ = compute_reward(model_calls, scenario.operations)
        rewards.append(reward)

    # The group MUST be informative (non-constant rewards)
    assert len(set(rewards)) > 1, f"Group still constant: {rewards}"
    # The easy slots should succeed
    assert rewards[0] == 1.0, f"1-op easy slot failed: {rewards[0]}"
    print(f"Rewards with difficulty variation: {rewards}")
```

- [ ] **Step 2: Run integration test**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_max_ops.py::test_difficulty_variation_creates_informative_group -v`
Expected: PASS — prints `Rewards with difficulty variation: [1.0, 1.0, ...]` with mixed values

- [ ] **Step 3: Run full test suite**

Run: `cd /home/ubuntu/hangook/games && python -m pytest tests/test_max_ops.py tests/test_sft_from_rollouts.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/hangook/games
git add tests/test_max_ops.py
git commit -m "test: integration test confirming difficulty variation creates informative groups"
```
