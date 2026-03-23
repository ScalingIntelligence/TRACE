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
