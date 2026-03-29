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

try:
    from tau_tool_calling_env_revised import (
        TauToolCallingEnv as TauToolCallingEnvRevised,
        extract_action as extract_action_tau_tool_revised,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TAU_TOOL_REVISED,
    )
except ImportError:
    TauToolCallingEnvRevised = None

from structured_data_game import (
    StructuredDataGame,
    extract_action as extract_action_structured_data,
    SYSTEM_PROMPT as SYSTEM_PROMPT_STRUCTURED_DATA,
)

from structured_data_game_revised import (
    StructuredDataGame as StructuredDataGameRevised,
    extract_action as extract_action_structured_data_revised,
    SYSTEM_PROMPT as SYSTEM_PROMPT_STRUCTURED_DATA_REVISED,
)

from structured_data_game_deprecated import (
    StructuredDataGame as StructuredDataGameDeprecated,
    extract_action as extract_action_structured_data_deprecated,
    SYSTEM_PROMPT as SYSTEM_PROMPT_STRUCTURED_DATA_DEPRECATED,
)

from multistep_task_game import (
    RealisticMultiStepGame,
    extract_action as extract_action_multistep,
    SYSTEM_PROMPT as SYSTEM_PROMPT_MULTISTEP,
)

from multistep_task_game_revised import (
    RealisticMultiStepGame as RealisticMultiStepGameRevised,
    extract_action as extract_action_multistep_revised,
    SYSTEM_PROMPT as SYSTEM_PROMPT_MULTISTEP_REVISED,
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

try:
    from precondition_game_revised import (
        PreconditionGame as PreconditionGameRevised,
        extract_action as extract_action_precondition_revised,
        SYSTEM_PROMPT as SYSTEM_PROMPT_PRECONDITION_REVISED,
    )
except ImportError:
    PreconditionGameRevised = None


try:
    from tec_game import (
        TECGame,
        extract_action as extract_action_tec,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TEC,
    )
except ImportError:
    TECGame = None

try:
    from tec_game_v2 import (
        TECGameV2,
        extract_action as extract_action_tec_v2,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TEC_V2,
    )
except ImportError:
    TECGameV2 = None

try:
    from toolsandbox_multiturn_game import (
        ToolSandboxMultiTurnGame,
        extract_action as extract_action_ts_multiturn,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TS_MULTITURN,
    )
except ImportError:
    ToolSandboxMultiTurnGame = None

try:
    from datetime_game import (
        DatetimeGame,
        extract_action as extract_action_datetime,
        SYSTEM_PROMPT as SYSTEM_PROMPT_DATETIME,
    )
except ImportError:
    DatetimeGame = None

try:
    from awm_game import (
        AWMGame,
        extract_action as extract_action_awm,
        SYSTEM_PROMPT as SYSTEM_PROMPT_AWM,
    )
except ImportError:
    AWMGame = None

try:
    from tau2_bench_game import (
        Tau2BenchGame,
        extract_action as extract_action_tau2_bench,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TAU2_BENCH,
    )
except ImportError:
    Tau2BenchGame = None

try:
    from tau2_bench_synthetic_game import (
        Tau2BenchSyntheticGame,
        extract_action as extract_action_tau2_bench_synthetic,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TAU2_BENCH_SYNTHETIC,
    )
except ImportError:
    Tau2BenchSyntheticGame = None

try:
    from toolsandbox_game import (
        ToolSandboxGame,
        extract_action as extract_action_toolsandbox,
        SYSTEM_PROMPT as SYSTEM_PROMPT_TOOLSANDBOX,
    )
except ImportError:
    ToolSandboxGame = None


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

    # SkillRL variant — same game with SkillBank injected into system prompt
    if AdversarialPolicyGame is not None:
        def make_adversarial_policy_skillrl(user_client=None, adversarial_ratio=0.2,
                                            skillbank=None) -> AdversarialPolicyGame:
            return AdversarialPolicyGame(
                max_steps=30, user_client=user_client,
                adversarial_ratio=adversarial_ratio,
                skillbank=skillbank,
            )

        register_game(
            GameSpec(
                name="adversarial_policy_skillrl",
                make_env=make_adversarial_policy_skillrl,
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

    # ======================================================================
    # REVISED games — address tau2-bench transferability gaps
    # ======================================================================

    # Tau Tool-Calling (Revised) — adds adversarial users, conditional fallback,
    # cross-entity reasoning, progressive disclosure
    if TauToolCallingEnvRevised is not None:
        def make_tau_tool_calling_revised(user_client=None, domain=None) -> TauToolCallingEnvRevised:
            return TauToolCallingEnvRevised(
                max_steps=30, user_client=user_client, domain=domain,
            )

        register_game(
            GameSpec(
                name="tau_tool_calling_revised",
                make_env=make_tau_tool_calling_revised,
                extract_action=extract_action_tau_tool_revised,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TAU_TOOL_REVISED,
                max_gen_tokens=1024,
            )
        )

    # Structured Data Reasoning (Revised) — adds adversarial users, conditional
    # fallback, DB hash verification, cross-entity, progressive disclosure
    def make_structured_data_revised(user_client=None, domain=None) -> StructuredDataGameRevised:
        return StructuredDataGameRevised(user_client=user_client, domain=domain)

    register_game(
        GameSpec(
            name="structured_data_reasoning_revised",
            make_env=make_structured_data_revised,
            extract_action=extract_action_structured_data_revised,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_STRUCTURED_DATA_REVISED,
            max_gen_tokens=1024,
        )
    )

    # Multi-Step Task (Revised) — adds DB hash verification, communicate check,
    # removes wrong-call penalty
    def make_multistep_revised(user_client=None, domain=None) -> RealisticMultiStepGameRevised:
        return RealisticMultiStepGameRevised(max_steps=15, user_client=user_client, domain=domain)

    register_game(
        GameSpec(
            name="multistep_task_revised",
            make_env=make_multistep_revised,
            extract_action=extract_action_multistep_revised,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_MULTISTEP_REVISED,
            max_gen_tokens=1024,
        )
    )

    # Precondition Check (Revised) — adds retail domain, adversarial users,
    # DB hash verification, dynamic generation, progressive disclosure
    if PreconditionGameRevised is not None:
        def make_precondition_revised(user_client=None) -> PreconditionGameRevised:
            return PreconditionGameRevised(
                max_steps=20, user_client=user_client,
            )

        register_game(
            GameSpec(
                name="precondition_check_revised",
                make_env=make_precondition_revised,
                extract_action=extract_action_precondition_revised,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_PRECONDITION_REVISED,
                max_gen_tokens=1024,
            )
        )

    # AWM Tool Calling — Agent World Model environments
    # 1K pre-generated SQLite-backed tool-use environments from AWM paper.
    # Model interacts with application tools to complete user tasks.
    # Reward from code-based verification of database state changes.
    if AWMGame is not None:
        def make_awm(user_client=None, data_dir=None) -> AWMGame:
            return AWMGame(
                data_dir=data_dir, max_steps=20, user_client=user_client,
            )

        register_game(
            GameSpec(
                name="awm_tool_calling",
                make_env=make_awm,
                extract_action=extract_action_awm,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_AWM,
                max_gen_tokens=1024,
            )
        )

    # Tau2-Bench — full tau2-bench airline + retail evaluation as a training game
    # 50 airline + 114 retail tasks with real tools, DB, policy, and LLM user sim.
    # Reward from DB hash comparison + communicate_info substring matching.
    if Tau2BenchGame is not None:
        def make_tau2_bench(user_client=None, domain=None) -> Tau2BenchGame:
            return Tau2BenchGame(max_steps=30, user_client=user_client, domain=domain)

        register_game(
            GameSpec(
                name="tau2_bench",
                make_env=make_tau2_bench,
                extract_action=extract_action_tau2_bench,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TAU2_BENCH,
                max_gen_tokens=1024,
            )
        )

        def make_tau2_bench_airline(user_client=None) -> Tau2BenchGame:
            return Tau2BenchGame(max_steps=30, user_client=user_client, domain="airline")

        register_game(
            GameSpec(
                name="tau2_bench_airline",
                make_env=make_tau2_bench_airline,
                extract_action=extract_action_tau2_bench,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TAU2_BENCH,
                max_gen_tokens=1024,
            )
        )

        def make_tau2_bench_retail(user_client=None) -> Tau2BenchGame:
            return Tau2BenchGame(max_steps=30, user_client=user_client, domain="retail")

        register_game(
            GameSpec(
                name="tau2_bench_retail",
                make_env=make_tau2_bench_retail,
                extract_action=extract_action_tau2_bench,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TAU2_BENCH,
                max_gen_tokens=1024,
            )
        )


    # Tau2-Bench Synthetic — same mechanics as tau2_bench but with per-episode
    # synthetic DB + tasks for maximum training data diversity.
    if Tau2BenchSyntheticGame is not None:
        def make_tau2_bench_synthetic(user_client=None, domain=None) -> Tau2BenchSyntheticGame:
            return Tau2BenchSyntheticGame(max_steps=30, user_client=user_client, domain=domain)

        register_game(
            GameSpec(
                name="tau2_bench_synthetic",
                make_env=make_tau2_bench_synthetic,
                extract_action=extract_action_tau2_bench_synthetic,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TAU2_BENCH_SYNTHETIC,
                max_gen_tokens=1024,
            )
        )

        def make_tau2_bench_synthetic_airline(user_client=None) -> Tau2BenchSyntheticGame:
            return Tau2BenchSyntheticGame(max_steps=30, user_client=user_client, domain="airline")

        register_game(
            GameSpec(
                name="tau2_bench_synthetic_airline",
                make_env=make_tau2_bench_synthetic_airline,
                extract_action=extract_action_tau2_bench_synthetic,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TAU2_BENCH_SYNTHETIC,
                max_gen_tokens=1024,
            )
        )

        def make_tau2_bench_synthetic_retail(user_client=None) -> Tau2BenchSyntheticGame:
            return Tau2BenchSyntheticGame(max_steps=30, user_client=user_client, domain="retail")

        register_game(
            GameSpec(
                name="tau2_bench_synthetic_retail",
                make_env=make_tau2_bench_synthetic_retail,
                extract_action=extract_action_tau2_bench_synthetic,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TAU2_BENCH_SYNTHETIC,
                max_gen_tokens=1024,
            )
        )


    # Tool-Execute-Communicate v2 (TEC) — focused on two clean skills
    # Skill A: communicate after tool call, Skill B: error recovery chain
    if TECGameV2 is not None:
        def make_tec_v2() -> TECGameV2:
            return TECGameV2()

        register_game(
            GameSpec(
                name="tec_v2",
                make_env=make_tec_v2,
                extract_action=extract_action_tec_v2,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TEC_V2,
                max_gen_tokens=1024,
            )
        )

    # ToolSandbox Multi-Turn Clarification — targets MULTIPLE_USER_TURN failures
    # Trains: (1) asking clarifying questions, (2) communicating after tools,
    # (3) recognizing insufficient information. Matches ToolSandbox format exactly.
    if ToolSandboxMultiTurnGame is not None:
        def make_ts_multiturn() -> ToolSandboxMultiTurnGame:
            return ToolSandboxMultiTurnGame()

        register_game(
            GameSpec(
                name="toolsandbox_multiturn",
                make_env=make_ts_multiturn,
                extract_action=extract_action_ts_multiturn,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TS_MULTITURN,
                max_gen_tokens=1024,
            )
        )

    # Datetime Reasoning — teaches correct Unix timestamp handling
    # Model must call timestamp_to_datetime_info instead of guessing the year
    if DatetimeGame is not None:
        def make_datetime() -> DatetimeGame:
            return DatetimeGame()

        register_game(
            GameSpec(
                name="datetime",
                make_env=make_datetime,
                extract_action=extract_action_datetime,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_DATETIME,
                max_gen_tokens=1024,
            )
        )

    # Tool-Execute-Communicate (TEC v1) — original version
    if TECGame is not None:
        def make_tec() -> TECGame:
            return TECGame()

        register_game(
            GameSpec(
                name="tec",
                make_env=make_tec,
                extract_action=extract_action_tec,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TEC,
                max_gen_tokens=1024,
            )
        )

    # ToolSandbox — Apple's stateful, conversational tool-use benchmark
    # 129 base scenarios across contacts, messaging, settings, reminders, search.
    # Reward from milestone DAG matching + minefield checking.
    if ToolSandboxGame is not None:
        def make_toolsandbox(user_client=None) -> ToolSandboxGame:
            return ToolSandboxGame(max_steps=30, user_client=user_client)

        register_game(
            GameSpec(
                name="toolsandbox",
                make_env=make_toolsandbox,
                extract_action=extract_action_toolsandbox,
                action_space=[],
                stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
                system_prompt=SYSTEM_PROMPT_TOOLSANDBOX,
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
