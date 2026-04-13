from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


class GameEnv(Protocol):

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
    # Whether this game needs a user LLM client injected into env_kwargs.
    # When True and --user-llm-url is set, the trainer will construct a
    # UserLLMClient and pass it as env_kwargs["user_client"].
    needs_user_llm: bool = False
    # Optional hint/base prompt pair for training-time hint-swap.
    # When both are set, the trainer replaces any sample whose first system
    # message equals hint_prompt with base_prompt before computing gradients,
    # so the gradient reinforces behavior conditioned on the base (eval-time)
    # prompt rather than on the hint that was used during rollout.
    hint_prompt: Optional[str] = None
    base_prompt: Optional[str] = None


GAMES: Dict[str, GameSpec] = {}


def register_game(spec: GameSpec) -> None:
    GAMES[spec.name] = spec


def get_game_spec(name: str) -> GameSpec:
    try:
        return GAMES[name]
    except KeyError as e:
        raise KeyError(f"Unknown game '{name}'. Available: {list(GAMES.keys())}") from e


def list_game_names() -> List[str]:
    return sorted(GAMES.keys())




# ============================================================================
# Multi-environment game mixing
# ============================================================================

@dataclass
class GameMixEntry:
    game_spec: GameSpec
    weight: float  # Proportional weight (normalized to sum=1)
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameMix:
    entries: List[GameMixEntry]

    def assign_groups(self, groups_per_batch: int) -> List[Tuple[GameSpec, Dict[str, Any]]]:
        # Compute raw allocations
        raw = [e.weight * groups_per_batch for e in self.entries]
        allocs = [int(r) for r in raw]
        remainders = [(raw[i] - allocs[i], i) for i in range(len(self.entries))]

        # Distribute remaining slots to entries with largest remainders
        deficit = groups_per_batch - sum(allocs)
        remainders.sort(reverse=True)
        for _, idx in remainders[:deficit]:
            allocs[idx] += 1

        # Build assignment list
        assignments: List[Tuple[GameSpec, Dict[str, Any]]] = []
        for entry, count in zip(self.entries, allocs):
            for _ in range(count):
                assignments.append((entry.game_spec, entry.env_kwargs))
        return assignments

    @property
    def max_gen_tokens(self) -> int:
        return max(e.game_spec.max_gen_tokens for e in self.entries)


# ---------------------------------------------------------------------------
# Auto-register capability environments. Each `capability_*_game.py` lives at
# the project root and calls `register_game(...)` on import.
# ---------------------------------------------------------------------------
