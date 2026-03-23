import sys
sys.path.insert(0, ".")

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
    """Same (game_id, prompt_length) is not added twice."""
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
    assert len(buf) == 1  # Deduped by (game_id, prompt_length)


def test_add_from_rollouts_keeps_all_turns():
    """Multiple turns from the same game are all added (different prompt lengths)."""
    from sft_buffer import SFTBuffer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507",
                                        trust_remote_code=True)
    buf = SFTBuffer([], tok, compact_tools=True)

    # Two turns from the same game (game_id=300) with different prompt lengths
    turn1 = FakeGRPOSample(
        prompt_msgs=[{"role": "system", "content": "test"},
                     {"role": "user", "content": "cancel A; add bags B"}],
        completion_text='<tool_call>\n{"name": "cancel", "arguments": {}}\n</tool_call>',
        player_id=0, reward=1.0, group_id=0, game_id=300,
        game_name="multistep_task",
    )
    turn2 = FakeGRPOSample(
        prompt_msgs=[{"role": "system", "content": "test"},
                     {"role": "user", "content": "cancel A; add bags B"},
                     {"role": "assistant", "content": None}],
        completion_text='<tool_call>\n{"name": "update_bags", "arguments": {}}\n</tool_call>',
        player_id=0, reward=1.0, group_id=0, game_id=300,
        game_name="multistep_task",
    )

    added = buf.add_from_rollouts([turn1, turn2], min_reward=1.0)
    assert added == 2  # Both turns added (different prompt lengths)
    assert len(buf) == 2
