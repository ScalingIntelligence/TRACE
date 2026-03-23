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


def test_max_ops_truncates_retail():
    """max_ops=1 on a retail scenario truncates correctly."""
    # Find a retail seed with 2+ ops
    for seed in range(100):
        s = generate_scenario(seed)
        if s.domain == "retail" and len(s.operations) >= 2:
            break
    trunc = generate_scenario(seed, max_ops=1)
    assert len(trunc.operations) == 1
    assert trunc.domain == "retail"


def test_reset_with_max_ops():
    """RealisticMultiStepGame.reset accepts max_ops."""
    from multistep_task_game import RealisticMultiStepGame
    game = RealisticMultiStepGame()
    game.reset(2087043557, max_ops=1)
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


def test_difficulty_schedule():
    """The difficulty schedule assigns varying max_ops within a group."""
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


def test_difficulty_variation_creates_informative_group():
    """A previously constant-zero group becomes informative with max_ops variation."""
    from multistep_task_game import compute_reward
    from train_grpo import multistep_difficulty_schedule

    seed = 2087043557  # Known all-zero group in original training
    schedule = multistep_difficulty_schedule(8)

    rewards = []
    for max_ops in schedule:
        scenario = generate_scenario(seed, max_ops=max_ops)
        # Simulate the model's known behavior: cancel(A) + baggages(B, wrong) + cancel(C)
        all_model_calls = [
            {"name": "cancel_reservation", "arguments": {"reservation_id": "MS45397"}},
            {"name": "update_reservation_baggages", "arguments": {
                "reservation_id": "MS24190", "total_baggages": 2,
                "nonfree_baggages": 2, "payment_id": "credit_card_4112902"}},
            {"name": "cancel_reservation", "arguments": {"reservation_id": "MS30465"}},
        ]
        # Model follows the prompt: only calls up to max_ops writes
        if max_ops is not None:
            model_calls = all_model_calls[:max_ops]
        else:
            model_calls = all_model_calls
        reward, _ = compute_reward(model_calls, scenario.operations)
        rewards.append(reward)

    # The group MUST be informative (non-constant rewards)
    assert len(set(rewards)) > 1, f"Group still constant: {rewards}"
    # The easy slots should succeed
    assert rewards[0] == 1.0, f"1-op easy slot failed: {rewards[0]}"
