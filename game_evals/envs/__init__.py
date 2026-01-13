import logging
import textarena as ta

# Import all available environments
from .liarsDice_new.env import LiarsDiceEnv
from .pigDice_new.env import PigDiceEnv
from .simpleNegotiation_new.env import SimpleNegotiationEnv
from .kuhnPoker_new.env import KuhnPokerEnv
from .ticTacToe_new.env import TicTacToeEnv
from .connect4_new.env import Connect4Env
from .battleship_new.env import BattleshipEnv
from .rockPaperScissors_new.env import RockPaperScissorsEnv
from .nim_new.env import NimEnv
from .negotiation_new.env import NegotiationEnv
from .dotsAndBoxes_new.env import DotsAndBoxesEnv
from .chomp_new.env import ChompEnv
from .auction_new.env import AuctionEnv
from .extra_environments.Game24.env import Game24Env
from .extra_environments.TruthAndDeception_new.env import TruthAndDeceptionEnv

# Environment registry mapping IDs to (class, default_kwargs)
ENV_REGISTRY = {
	"KuhnPoker-v1": (KuhnPokerEnv, {"ante": 1, "max_rounds": 5}),
	"LiarsDice-v1": (LiarsDiceEnv, {"num_dice": 5}),
	"PigDice-v1": (PigDiceEnv, {"winning_score": 50, "max_turns": 50}),
	"SimpleNegotiation-v1": (SimpleNegotiationEnv, {"max_turns": 10}),
	"TruthAndDeception-v1": (TruthAndDeceptionEnv, {"max_turns": 6}),
	"TicTacToe-v1": (TicTacToeEnv, {"max_turns": 9}),
	"Connect4-v1": (Connect4Env, {"rows": 6, "cols": 7, "max_turns": 42}),
	"Battleship-v1": (BattleshipEnv, {"board_size": 5, "ship_length": 4}),
	"RockPaperScissors-v1": (RockPaperScissorsEnv, {"best_of": 5}),
	"Nim-v1": (NimEnv, {"num_piles": 4, "max_items_per_pile": 5, "max_turns": 100}),
	"Negotiation-v1": (NegotiationEnv, {"num_items": 5, "quantity_per_item": 3, "max_steps": 10, "verbal": False}),
	"DotsAndBoxes-v1": (DotsAndBoxesEnv, {"rows": 5, "cols": 5}),
	"Chomp-v1": (ChompEnv, {"rows": 5, "cols": 6}),
	"Auction-v1": (AuctionEnv, {"starting_money": 100, "num_rounds": 10}),
	"Game24-v1": (Game24Env, {"num_rounds": 5, "max_turns": 100}),
}


def make_env(env_id: str, use_llm_obs_wrapper: bool):
	if env_id not in ENV_REGISTRY:
		available = ", ".join(ENV_REGISTRY.keys())
		raise NotImplementedError(f"Unsupported env_id: {env_id}. Available: {available}")
	
	env_class, default_kwargs = ENV_REGISTRY[env_id]
	env = env_class(**default_kwargs)
	
	if use_llm_obs_wrapper:
		logging.info(f"[ENV] Using LLMObservationWrapper for {env_id}")
		env = ta.wrappers.LLMObservationWrapper(env=env)
	else:
		logging.info(f"[ENV] Using FirstLastObservationWrapper for {env_id}")
		env = ta.wrappers.FirstLastObservationWrapper(env=env)
	return env


def make_vec_env(env_id: str, num_envs: int, **kw_args):
	return [make_env(env_id, **kw_args) for _ in range(num_envs)]


