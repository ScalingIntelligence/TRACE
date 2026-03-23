"""Tests for per-step reward credit assignment."""
import sys
sys.path.insert(0, ".")

import torch
from train_grpo import GRPOSample, compute_group_advantages


def _sample(group_id, game_id, reward, game_name="multistep_task"):
    return GRPOSample(
        prompt_msgs=[{"role": "user", "content": "test"}],
        completion_text="test",
        player_id=0,
        reward=reward,
        group_id=group_id,
        game_id=game_id,
        game_name=game_name,
    )


def test_step_rewards_game_integration():
    """RealisticMultiStepGame produces per-step rewards."""
    from multistep_task_game import RealisticMultiStepGame

    game = RealisticMultiStepGame()
    game.reset(2087043557)  # 3-op: cancel, baggages, cancel

    # correct cancel → wrong baggages → correct cancel → respond
    game.step('{"name": "cancel_reservation", "arguments": {"reservation_id": "MS45397"}}')
    game.step('{"name": "update_reservation_baggages", "arguments": {"reservation_id": "MS24190", "total_baggages": 2, "nonfree_baggages": 2, "payment_id": "credit_card_4112902"}}')
    game.step('{"name": "cancel_reservation", "arguments": {"reservation_id": "MS30465"}}')
    game.step('{"name": "respond_to_user", "arguments": {"message": "Done"}}')

    assert game.done
    assert len(game._step_rewards) == 4
    assert game._step_rewards[0] == 1.0   # correct cancel
    assert game._step_rewards[1] == -0.5  # wrong baggages (nonfree=2 != expected)
    assert game._step_rewards[2] == 1.0   # correct cancel
    assert game._step_rewards[3] == 0.0   # terminal reward (game failed)


def test_step_rewards_all_correct():
    """All-correct game: all step rewards are 1.0, terminal is 1.0."""
    from multistep_task_game import RealisticMultiStepGame, generate_scenario

    seed = 2087043557
    scenario = generate_scenario(seed)
    # The correct args for baggages: total=2, nonfree=0
    game = RealisticMultiStepGame()
    game.reset(seed)

    game.step('{"name": "cancel_reservation", "arguments": {"reservation_id": "MS45397"}}')
    game.step('{"name": "update_reservation_baggages", "arguments": {"reservation_id": "MS24190", "total_baggages": 2, "nonfree_baggages": 0, "payment_id": "credit_card_4112902"}}')
    game.step('{"name": "cancel_reservation", "arguments": {"reservation_id": "MS30465"}}')
    game.step('{"name": "respond_to_user", "arguments": {"message": "Done"}}')

    assert game._step_rewards == [1.0, 1.0, 1.0, 1.0]
    assert game.rewards[0] == 1.0


def test_step_rewards_lookup_neutral():
    """Lookup turns get step_reward = 0.0."""
    from multistep_task_game import RealisticMultiStepGame

    game = RealisticMultiStepGame()
    game.reset(2087043557)

    # Do a lookup before write
    game.step('{"name": "search_direct_flight", "arguments": {"origin": "DEN", "destination": "PHX", "date": "2024-05-20"}}')
    assert game._step_rewards == [0.0]  # neutral for lookups


def test_per_step_breaks_constant_group():
    """Same-seed identical actions: previously constant, now informative."""
    samples = [
        # Game 0: cancel(1.0), baggages(-0.5), cancel(1.0), respond(0.0)
        _sample(0, 0, 1.0), _sample(0, 0, -0.5), _sample(0, 0, 1.0), _sample(0, 0, 0.0),
        # Game 1: identical
        _sample(0, 1, 1.0), _sample(0, 1, -0.5), _sample(0, 1, 1.0), _sample(0, 1, 0.0),
    ]

    advantages = compute_group_advantages(samples)

    # Must NOT be all-zero (the constant-group problem we're solving)
    assert not torch.all(advantages == 0), "Advantages should not all be zero"

    # Correct turns should have positive advantage
    assert advantages[0].item() > 0  # correct cancel
    assert advantages[4].item() > 0  # correct cancel (game 1)

    # Wrong turns should have negative advantage
    assert advantages[1].item() < 0  # wrong baggages
    assert advantages[5].item() < 0  # wrong baggages (game 1)

    # Sum should be zero
    assert abs(advantages.sum().item()) < 1e-6


def test_backward_compat_terminal_rewards():
    """Games without per-step rewards still work (terminal reward for all turns)."""
    # Simulate old behavior: all turns get same terminal reward
    samples = [
        _sample(0, 0, 1.0), _sample(0, 0, 1.0),  # game 0, all turns reward=1.0
        _sample(0, 1, 0.0), _sample(0, 1, 0.0),  # game 1, all turns reward=0.0
    ]

    advantages = compute_group_advantages(samples)
    assert advantages[0].item() > 0  # game 0 above mean
    assert advantages[2].item() < 0  # game 1 below mean
    assert abs(advantages.sum().item()) < 1e-6
