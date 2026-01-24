from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from config import ACTION_STRS_KUHN, Config
from kuhn_poker import KuhnPoker, extract_action as extract_action_kuhn
from liars_dice import LiarsDice, extract_action as extract_action_liars
from liars_dice_memory import LiarsDiceMemory, extract_action as extract_action_memory
from liars_dice_memory_updated import LiarsDiceMemoryUpdated, extract_action as extract_action_memory_updated

from pathlib import Path


class GameEnv(Protocol):
    """Minimal interface expected by the PPO + self-play loop."""

    done: bool
    current_player: int
    rewards: Dict[int, float]
    invalid_player: Optional[int]

    def reset(self, seed: int) -> None:
        ...

    def observe(self, player_id: int) -> str:
        ...

    def legal_actions(self) -> List[str]:
        ...

    def step(self, action: Optional[str]) -> None:
        ...


ExtractActionFn = Callable[[str, List[str]], Optional[str]]


@dataclass(frozen=True)
class GameSpec:
    """Registry entry for a trainable game."""

    name: str
    make_env: Callable[..., GameEnv]
    extract_action: ExtractActionFn
    # Fixed list of action strings, used for guided decoding when available.
    action_space: List[str] = field(default_factory=list)
    # Optional stop sequences for generation (falls back to action_space when empty).
    stop_sequences: Optional[List[str]] = None
    # System prompt injected into every episode.
    system_prompt: str = ""
    # Maximum new tokens to sample for an action completion.
    max_gen_tokens: int = 8


GAMES: Dict[str, GameSpec] = {}


def register_game(spec: GameSpec) -> None:
    """Register a new game specification."""
    GAMES[spec.name] = spec


def get_game_spec(name: str) -> GameSpec:
    try:
        return GAMES[name]
    except KeyError as e:
        raise KeyError(f"Unknown game '{name}'. Available: {list(GAMES.keys())}") from e


def list_game_names() -> List[str]:
    return sorted(GAMES.keys())


def _register_builtin_games() -> None:
    def make_kuhn(num_rounds: int = Config.NUM_ROUNDS) -> KuhnPoker:
        return KuhnPoker(num_rounds=num_rounds)

    def make_liars_dice(num_dice: int = Config.NUM_DICE) -> LiarsDice:
        return LiarsDice(num_dice=num_dice)

    def make_liars_dice_memory(num_dice: int = Config.NUM_DICE) -> LiarsDiceMemory:
        # Use absolute path so it works from any directory
        games_dir = Path(__file__).resolve().parent
        return LiarsDiceMemory(
            num_dice=num_dice,
            history_source=games_dir / "selfplay_rollouts_ppo_database.jsonl",
            num_history_games=200,
        )

    def make_liars_dice_memory_updated(num_dice: int = Config.NUM_DICE) -> LiarsDiceMemoryUpdated:
        return LiarsDiceMemoryUpdated(num_dice=num_dice, num_fake_games=40)

    register_game(
        GameSpec(
            name="kuhn_poker",
            make_env=make_kuhn,
            extract_action=extract_action_kuhn,
            action_space=ACTION_STRS_KUHN,
            stop_sequences=[] if Config.ENABLE_THINKING else ACTION_STRS_KUHN,
            system_prompt=Config.SYSTEM_PROMPT_KUHN,
            max_gen_tokens=Config.MAX_GEN_TOKENS,
        )
    )

    register_game(
        GameSpec(
            name="liars_dice",
            make_env=make_liars_dice,
            extract_action=extract_action_liars,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["]"],
            system_prompt=Config.SYSTEM_PROMPT_LIARS_DICE,
            max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )

    register_game(
    GameSpec(
        name="liars_dice_memory",
        make_env=make_liars_dice_memory,
        extract_action=extract_action_memory,
        action_space=[],
        stop_sequences=[] if Config.ENABLE_THINKING else ["]"],
        system_prompt=Config.SYSTEM_PROMPT_LIARS_DICE_MEMORY,
        max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )

    register_game(
    GameSpec(
        name="liars_dice_memory_updated",
        make_env=make_liars_dice_memory_updated,
        extract_action=extract_action_memory_updated,
        action_space=[],
        stop_sequences=[] if Config.ENABLE_THINKING else ["]"],
        system_prompt=Config.SYSTEM_PROMPT_LIARS_DICE_MEMORY_UPDATED,
        max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )


def _register_openspiel_games() -> None:
    """Best-effort registration of optional OpenSpiel games."""
    try:
        from openspiel_wrapper import list_openspiel_game_specs
    except Exception:
        return

    try:
        for spec in list_openspiel_game_specs():
            register_game(spec)
    except Exception:
        # Keep core games usable even if OpenSpiel isn't installed.
        return


_register_builtin_games()
_register_openspiel_games()
