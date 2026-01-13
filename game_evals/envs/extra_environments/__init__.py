# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import textarena as ta
from textarena.envs.registration import register

# Kuhn Poker (two-player)
register(
    id="KuhnPoker-v1",
    entry_point="spiral.envs.KuhnPoker.env:KuhnPokerEnv",
    ante=1,
    max_rounds=5,
)

# Simple Negotiation (two-player)
register(
    id="SimpleNegotiation-v1",
    entry_point="spiral.envs.SimpleNegotiation.env:SimpleNegotiationEnv",
    max_turns=10,
)

# Liar's Dice (two-player)
register(
    id="LiarsDice-v1",
    entry_point="spiral.envs.LiarsDice.env:LiarsDiceEnv",
)

# Truth and Deception (two-player)
register(
    id="TruthAndDeception-v1",
    entry_point="spiral.envs.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=6,
)

# Pig Dice (two-player)
register(
    id="PigDice-v1",
    entry_point="spiral.envs.PigDice.env:PigDiceEnv",
    winning_score=50,
    max_turns=50,
)

# Connect4 (two-player)
register(
    id="Connect4-v1",
    entry_point="spiral.envs.Connect4.env:Connect4Env",
    rows=6,
    cols=7,
    max_turns=42,
)

# Negotiation (two-player)
register(
    id="Negotiation-v1",
    entry_point="spiral.envs.Negotiation.env:NegotiationEnv",
    num_items=5,
    quantity_per_item=3,
    max_steps=10,
    verbal=False,
)

# TicTacToe (two-player)
register(
    id="TicTacToe-v1",
    entry_point="spiral.envs.TicTacToe.env:TicTacToeEnv",
    max_turns=9,
)

# Battleship (two-player)
register(
    id="Battleship-v1",
    entry_point="spiral.envs.Battleship.env:BattleshipEnv",
    board_size=5,
    ship_length=4,
)

# RockPaperScissors (two-player)
register(
    id="RockPaperScissors-v1",
    entry_point="spiral.envs.RockPaperScissors.env:RockPaperScissorsEnv",
    best_of=5,
)

# Nim (two-player)
register(
    id="Nim-v1",
    entry_point="spiral.envs.Nim.env:NimEnv",
    num_piles=2,
    max_items_per_pile=5,
    max_turns=100,
)

# Game24 (single-player puzzle)
register(
    id="Game24-v1",
    entry_point="spiral.envs.Game24.env:Game24Env",
    num_rounds=5,
    max_turns=100,
)

# DotsAndBoxes (two-player territory control)
register(
    id="DotsAndBoxes-v1",
    entry_point="spiral.envs.DotsAndBoxes.env:DotsAndBoxesEnv",
    rows=5,
    cols=5,
)

# Chomp (two-player combinatorial game)
register(
    id="Chomp-v1",
    entry_point="spiral.envs.Chomp.env:ChompEnv",
    rows=5,
    cols=6,
)

# Auction (two-player simultaneous bidding game)
register(
    id="Auction-v1",
    entry_point="spiral.envs.Auction.env:AuctionEnv",
    starting_money=100,
    num_rounds=10,
)


def make_env(env_id: str, use_llm_obs_wrapper: bool):
    env = ta.make(env_id)
    if use_llm_obs_wrapper:
        logging.info(f"[ENV] Using LLMObservationWrapper for {env_id}")
        env = ta.wrappers.LLMObservationWrapper(env=env)
    else:
        logging.info(f"[ENV] Using FirstLastObservationWrapper for {env_id}")
        env = ta.wrappers.FirstLastObservationWrapper(env=env)
    return env


def make_vec_env(env_id: str, num_envs: int, **kw_args):
    return [make_env(env_id, **kw_args) for _ in range(num_envs)]
