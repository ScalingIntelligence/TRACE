from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from config import Config

try:
    from adversarial_policy_game import (
        AdversarialPolicyGame,
        extract_action as extract_action_adversarial,
        SYSTEM_PROMPT as SYSTEM_PROMPT_ADVERSARIAL,
        UserLLMClient,
    )
except ImportError:
    AdversarialPolicyGame = None
    UserLLMClient = None

try:
    from tau_tool_calling_env import (
        TauToolCallingEnv,
        extract_action as extract_action_tau_tool,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TAU_TOOL,
    )
except ImportError:
    TauToolCallingEnv = None

from structured_data_game import (
    StructuredDataGame,
    extract_action as extract_action_structured_data,
    SYSTEM_PROMPT as SYSTEM_PROMPT_STRUCTURED_DATA,
)

from structured_data_game_deprecated import (
    StructuredDataGame as StructuredDataGameDeprecated,
    extract_action as extract_action_structured_data_deprecated,
    SYSTEM_PROMPT as SYSTEM_PROMPT_STRUCTURED_DATA_DEPRECATED,
)

from structured_data_new_game import (
    StructuredDataGame as StructuredDataGameV2,
    extract_action as extract_action_structured_data_v2,
    SYSTEM_PROMPT as SYSTEM_PROMPT_STRUCTURED_DATA_V2,
)

from multistep_task_game import (
    RealisticMultiStepGame,
    extract_action as extract_action_multistep,
    SYSTEM_PROMPT as SYSTEM_PROMPT_MULTISTEP,
)

from multistep_task_game_deprecated import (
    RealisticMultiStepGame as RealisticMultiStepGameDeprecated,
    extract_action as extract_action_multistep_deprecated,
    SYSTEM_PROMPT as SYSTEM_PROMPT_MULTISTEP_DEPRECATED,
)

try:
    from precondition_game import (
        PreconditionGame,
        extract_action as extract_action_precondition,
        SYSTEM_PROMPT as SYSTEM_PROMPT_PRECONDITION,
    )
except ImportError:
    PreconditionGame = None


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
    # Adversarial Policy Compliance Game — targets Skill 1 failures
    # Trains policy adherence under adversarial user pressure
    # adversarial_ratio controls mix: 0.2 = 20% adversarial, 80% cooperative
    if AdversarialPolicyGame is not None:
        def make_adversarial_policy(user_client=None, adversarial_ratio=0.2) -> AdversarialPolicyGame:
            return AdversarialPolicyGame(
                max_steps=30, user_client=user_client,
                adversarial_ratio=adversarial_ratio,
            )

        register_game(
            GameSpec(
                name="adversarial_policy",
                make_env=make_adversarial_policy,
                extract_action=extract_action_adversarial,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_ADVERSARIAL,
                max_gen_tokens=1024,
            )
        )

    # Tau Bench Tool-Calling Microenvironment — targets Skill 2 failures
    # Trains tool-calling competence on simplified single-action tau bench tasks
    # Uses EXACT same tools, DB, policy as tau2-bench with DB hash verification
    if TauToolCallingEnv is not None:
        def make_tau_tool_calling(user_client=None, domain=None) -> TauToolCallingEnv:
            return TauToolCallingEnv(
                max_steps=30, user_client=user_client, domain=domain,
            )

        register_game(
            GameSpec(
                name="tau_tool_calling",
                make_env=make_tau_tool_calling,
                extract_action=extract_action_tau_tool,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TAU_TOOL,
                max_gen_tokens=1024,
            )
        )

    # Structured Data Reasoning — tau2-bench aligned (airline)
    # Multi-turn with LLM user, uses exact tau2-bench airline tools/policy/system prompt.
    # Trains data reasoning: flight selection, baggage computation, cost comparison, etc.
    def make_structured_data(user_client=None, domain=None) -> StructuredDataGame:
        return StructuredDataGame(user_client=user_client, domain=domain)

    register_game(
        GameSpec(
            name="structured_data_reasoning",
            make_env=make_structured_data,
            extract_action=extract_action_structured_data,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_STRUCTURED_DATA,
            max_gen_tokens=1024,
        )
    )

    # Structured Data Reasoning (DEPRECATED) — old single-turn submit_answers format
    # Kept for backwards compatibility with existing LoRA adapters.
    def make_structured_data_deprecated(difficulty=3) -> StructuredDataGameDeprecated:
        return StructuredDataGameDeprecated(difficulty=difficulty)

    register_game(
        GameSpec(
            name="structured_data_reasoning_deprecated",
            make_env=make_structured_data_deprecated,
            extract_action=extract_action_structured_data_deprecated,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_STRUCTURED_DATA_DEPRECATED,
            max_gen_tokens=512,
        )
    )

    # Structured Data Reasoning v2 — tau2-bench aligned single-turn
    # Model is placed mid-conversation after auth+lookups, produces one action tool call
    # Uses exact tau2-bench system prompt, tools, and message format
    # Tool-calling game: uses generate_with_tools, not generate_text
    def make_structured_data_v2(domain=None) -> StructuredDataGameV2:
        return StructuredDataGameV2(domain=domain)

    register_game(
        GameSpec(
            name="structured_data_v2",
            make_env=make_structured_data_v2,
            extract_action=extract_action_structured_data_v2,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_STRUCTURED_DATA_V2,
            max_gen_tokens=1024,
        )
    )

    # Multi-Step Task — tau2-bench aligned (airline)
    # Multi-turn with LLM user, uses exact tau2-bench airline tools/policy/system prompt.
    # Trains sequential multi-op completion: cancel, change flights, update bags, etc.
    def make_multistep(user_client=None, domain=None) -> RealisticMultiStepGame:
        return RealisticMultiStepGame(max_steps=15, user_client=user_client, domain=domain)

    register_game(
        GameSpec(
            name="multistep_task",
            make_env=make_multistep,
            extract_action=extract_action_multistep,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_MULTISTEP,
            max_gen_tokens=1024,
        )
    )

    # Multi-Step Task (DEPRECATED) — old retail-only custom tool format
    # Kept for backwards compatibility with existing LoRA adapters.
    def make_multistep_deprecated() -> RealisticMultiStepGameDeprecated:
        return RealisticMultiStepGameDeprecated(max_steps=30)

    register_game(
        GameSpec(
            name="multistep_task_deprecated",
            make_env=make_multistep_deprecated,
            extract_action=extract_action_multistep_deprecated,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_MULTISTEP_DEPRECATED,
            max_gen_tokens=1024,
        )
    )

    # Precondition Verification — targets policy compliance failures
    # Tests whether the model checks policy rules before executing actions.
    # Uses same airline tools/DB/policy as tau2-bench. 20 tasks (12 REFUSE + 8 ALLOW).
    if PreconditionGame is not None:
        def make_precondition(user_client=None) -> PreconditionGame:
            return PreconditionGame(
                max_steps=20, user_client=user_client,
            )

        register_game(
            GameSpec(
                name="precondition_check",
                make_env=make_precondition,
                extract_action=extract_action_precondition,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_PRECONDITION,
                max_gen_tokens=1024,
            )
        )


_register_builtin_games()


# ============================================================================
# Multi-environment game mixing
# ============================================================================

@dataclass
class GameMixEntry:
    """One entry in a multi-game mix."""
    game_spec: GameSpec
    weight: float  # Proportional weight (normalized to sum=1)
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameMix:
    """A weighted mix of games for multi-environment GRPO training."""
    entries: List[GameMixEntry]

    def assign_groups(self, groups_per_batch: int) -> List[Tuple[GameSpec, Dict[str, Any]]]:
        """Assign each group index to a (GameSpec, env_kwargs) pair.

        Uses proportional allocation: each entry gets
        round(weight * groups_per_batch) groups, with remainder distributed
        to the highest-weight entries to guarantee the total matches.

        Returns:
            List of length groups_per_batch, each element is
            (GameSpec, env_kwargs) for that group.
        """
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
        """Maximum generation tokens across all games in the mix."""
        return max(e.game_spec.max_gen_tokens for e in self.entries)
