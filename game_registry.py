from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from config import ACTION_STRS_KUHN, Config
from kuhn_poker import KuhnPoker, extract_action as extract_action_kuhn
from liars_dice import LiarsDice, extract_action as extract_action_liars
from liars_dice_memory import LiarsDiceMemory, extract_action as extract_action_memory
from liars_dice_memory_updated import LiarsDiceMemoryUpdated, extract_action as extract_action_memory_updated
from liars_dice_tools import (
    extract_action_from_tool_call,
    LIARS_DICE_TOOL_REGISTRY_NO_ID,
    LIARS_DICE_TOOL_REGISTRY_WITH_ID,
)
from memory_recall_game import MemoryRecallGame, extract_action as extract_action_recall
from dependency_resolution_game import (
    DependencyResolutionGame,
    extract_action as extract_action_dependency,
    SYSTEM_PROMPT_DEPENDENCY_RESOLUTION,
)
from dependency_resolution_hangoo_game import (
    DependencyResolutionGameHangoo,
    extract_action as extract_action_dependency_hangoo,
    SYSTEM_PROMPT_DEPENDENCY_RESOLUTION as SYSTEM_PROMPT_DEPENDENCY_HANGOO,
)
from multistep_sequence_game import (
    MultiStepSequenceGame,
    extract_action as extract_action_multistep,
    SYSTEM_PROMPT_MULTISTEP,
)
from policy_gated_action_game import (
    PolicyGatedActionGame,
    extract_action as extract_action_policy_gated,
    SYSTEM_PROMPT_POLICY_GATED,
)
from policy_verification_game import (
    PolicyVerificationGame,
    extract_action as extract_action_policy_verify,
    SYSTEM_PROMPT_POLICY_VERIFICATION,
)
from policy_action_game import (
    PolicyActionGame,
    extract_action as extract_action_policy_action,
    SYSTEM_PROMPT_POLICY_ACTION,
)
from conditional_action_game import (
    ConditionalActionGame,
    extract_action as extract_action_conditional,
    SYSTEM_PROMPT_CONDITIONAL_ACTION,
)
from tool_chain_game import (
    ToolChainGame,
    extract_action as extract_action_tool_chain,
    SYSTEM_PROMPT_TOOL_CHAIN,
)
from progressive_service_agent_env import (
    ProgressiveServiceAgentEnv,
    extract_action as extract_action_progressive_service,
    SYSTEM_PROMPT as SYSTEM_PROMPT_PROGRESSIVE_SERVICE,
    TaskComplexity,
)
from adversarial_policy_game import (
    AdversarialPolicyGame,
    extract_action as extract_action_adversarial,
    SYSTEM_PROMPT as SYSTEM_PROMPT_ADVERSARIAL,
)
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
    def extract_tool_with_id(text: str, legal_actions: List[str]) -> Optional[str]:
        return extract_action_from_tool_call(text, legal_actions, LIARS_DICE_TOOL_REGISTRY_WITH_ID)

    def extract_tool_no_id(text: str, legal_actions: List[str]) -> Optional[str]:
        return extract_action_from_tool_call(text, legal_actions, LIARS_DICE_TOOL_REGISTRY_NO_ID)

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

    def make_memory_recall(num_dice: int = Config.NUM_DICE) -> MemoryRecallGame:
        return MemoryRecallGame(num_dice=num_dice, num_fake_games=100)


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

    register_game(
        GameSpec(
            name="liars_dice_tool",
            make_env=make_liars_dice,
            extract_action=extract_tool_with_id,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=Config.SYSTEM_PROMPT_LIARS_DICE_TOOL,
            max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )

    register_game(
        GameSpec(
            name="liars_dice_memory_tool",
            make_env=make_liars_dice_memory,
            extract_action=extract_tool_no_id,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=Config.SYSTEM_PROMPT_LIARS_DICE_MEMORY_TOOL,
            max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )

    register_game(
        GameSpec(
            name="liars_dice_memory_updated_tool",
            make_env=make_liars_dice_memory_updated,
            extract_action=extract_tool_no_id,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=Config.SYSTEM_PROMPT_LIARS_DICE_MEMORY_UPDATED_TOOL,
            max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )

    register_game(
    GameSpec(
        name="memory_recall",
        make_env=make_memory_recall,
        extract_action=extract_action_recall,
        action_space=[],
        stop_sequences=[] if Config.ENABLE_THINKING else ["]"],
        system_prompt=Config.SYSTEM_PROMPT_MEMORY_RECALL,
        max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )

    def make_dependency_resolution() -> DependencyResolutionGame:
        return DependencyResolutionGame(
            min_complexity=1,  # Always use conditionals (consistent difficulty)
            max_complexity=3,  # Up to conditional formulas
            max_steps=20,  # Enough room to explore but not infinite
        )

    register_game(
        GameSpec(
            name="dependency_resolution",
            make_env=make_dependency_resolution,
            extract_action=extract_action_dependency,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_DEPENDENCY_RESOLUTION,
            max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )
    
    def make_dependency_resolution_hangoo() -> DependencyResolutionGameHangoo:
        return DependencyResolutionGameHangoo(
            min_complexity=1,
            max_complexity=3,
            max_steps=20,
        )

    register_game(
        GameSpec(
            name="dependency_resolution_hangoo",
            make_env=make_dependency_resolution_hangoo,
            extract_action=extract_action_dependency_hangoo,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_DEPENDENCY_HANGOO,
            max_gen_tokens=Config.MAX_GEN_TOKENS_LIARS_DICE,
        )
    )

    # Multi-step sequence game - tests pure multi-step planning
    def make_multistep_sequence() -> MultiStepSequenceGame:
        return MultiStepSequenceGame(max_steps=30)

    register_game(
        GameSpec(
            name="multistep_sequence",
            make_env=make_multistep_sequence,
            extract_action=extract_action_multistep,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_MULTISTEP,
            max_gen_tokens=512,
        )
    )

    # Policy-gated action game - trains state verification and constraint compliance
    # Binary rewards only, targets tau-bench core failure modes
    def make_policy_gated_action() -> PolicyGatedActionGame:
        return PolicyGatedActionGame(max_steps=15)

    register_game(
        GameSpec(
            name="policy_gated_action",
            make_env=make_policy_gated_action,
            extract_action=extract_action_policy_gated,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_POLICY_GATED,
            max_gen_tokens=256,  # Concise: tool calls are ~50-120 tokens max
        )
    )

    # Simplified policy verification - single-step classification
    # Use this FIRST to train policy reasoning, then graduate to policy_gated_action
    def make_policy_verification() -> PolicyVerificationGame:
        return PolicyVerificationGame()

    register_game(
        GameSpec(
            name="policy_verification",
            make_env=make_policy_verification,
            extract_action=extract_action_policy_verify,
            action_space=["[approve]", "[deny]"],
            stop_sequences=[] if Config.ENABLE_THINKING else ["]"],
            system_prompt=SYSTEM_PROMPT_POLICY_VERIFICATION,
            max_gen_tokens=64,  # Very short: just [approve] or [deny]
        )
    )

    # Policy Action Game - single-step tool calling with policy reasoning
    # All info provided, model must output correct tool call or refuse
    def make_policy_action() -> PolicyActionGame:
        return PolicyActionGame()

    register_game(
        GameSpec(
            name="policy_action",
            make_env=make_policy_action,
            extract_action=extract_action_policy_action,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_POLICY_ACTION,
            max_gen_tokens=256,
        )
    )

    # Conditional Action Game - trains strict constraint adherence
    # Key tau-bench failure mode: "if X unavailable, ONLY do Y" but model does both
    def make_conditional_action() -> ConditionalActionGame:
        return ConditionalActionGame(max_steps=10)

    register_game(
        GameSpec(
            name="conditional_action",
            make_env=make_conditional_action,
            extract_action=extract_action_conditional,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_CONDITIONAL_ACTION,
            max_gen_tokens=256,
        )
    )

    # Tool Chain Game - 2-step tool chaining
    def make_tool_chain() -> ToolChainGame:
        return ToolChainGame(max_steps=5)

    register_game(
        GameSpec(
            name="tool_chain",
            make_env=make_tool_chain,
            extract_action=extract_action_tool_chain,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_TOOL_CHAIN,
            max_gen_tokens=256,
        )
    )

    # Progressive Service Agent - comprehensive tau-bench skill training
    # Trains: authentication, one-shot constraints, conditional logic,
    # multi-item batching, information discovery, entity disambiguation, confirmation
    def make_progressive_service_agent() -> ProgressiveServiceAgentEnv:
        return ProgressiveServiceAgentEnv(max_steps=20)

    register_game(
        GameSpec(
            name="progressive_service_agent",
            make_env=make_progressive_service_agent,
            extract_action=extract_action_progressive_service,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_PROGRESSIVE_SERVICE,
            max_gen_tokens=512,  # Longer: multi-turn dialogue with tool calls
        )
    )

    # Progressive Service Agent with CURRICULUM LEARNING
    # Auto-advances difficulty when win rate exceeds 70% over last 3 iterations
    def make_progressive_service_curriculum() -> ProgressiveServiceAgentEnv:
        return ProgressiveServiceAgentEnv(
            max_steps=20,
            curriculum=True,
            curriculum_threshold=0.7,
            games_per_iter=512,      # Matches Config.GAMES_PER_ITER
            curriculum_iters=3,       # Consider last 3 iterations for win rate
        )

    register_game(
        GameSpec(
            name="progressive_service_curriculum",
            make_env=make_progressive_service_curriculum,
            extract_action=extract_action_progressive_service,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_PROGRESSIVE_SERVICE,
            max_gen_tokens=512,
        )
    )

    # Progressive Service Agent variants for each complexity level
    def make_progressive_service_simple() -> ProgressiveServiceAgentEnv:
        return ProgressiveServiceAgentEnv(max_steps=15, complexity=TaskComplexity.SIMPLE_QUERY)

    def make_progressive_service_conditional() -> ProgressiveServiceAgentEnv:
        return ProgressiveServiceAgentEnv(max_steps=20, complexity=TaskComplexity.CONDITIONAL)

    def make_progressive_service_full() -> ProgressiveServiceAgentEnv:
        return ProgressiveServiceAgentEnv(max_steps=25, complexity=TaskComplexity.FULL_COMPLEXITY)

    register_game(
        GameSpec(
            name="progressive_service_simple",
            make_env=make_progressive_service_simple,
            extract_action=extract_action_progressive_service,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_PROGRESSIVE_SERVICE,
            max_gen_tokens=512,
        )
    )

    register_game(
        GameSpec(
            name="progressive_service_conditional",
            make_env=make_progressive_service_conditional,
            extract_action=extract_action_progressive_service,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_PROGRESSIVE_SERVICE,
            max_gen_tokens=512,
        )
    )

    register_game(
        GameSpec(
            name="progressive_service_full",
            make_env=make_progressive_service_full,
            extract_action=extract_action_progressive_service,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_PROGRESSIVE_SERVICE,
            max_gen_tokens=512,
        )
    )

    # Adversarial Policy Compliance Game — targets Skill 1 failures
    # Trains policy adherence under adversarial user pressure
    def make_adversarial_policy() -> AdversarialPolicyGame:
        return AdversarialPolicyGame(max_steps=20)

    register_game(
        GameSpec(
            name="adversarial_policy",
            make_env=make_adversarial_policy,
            extract_action=extract_action_adversarial,
            action_space=[],
            stop_sequences=[] if Config.ENABLE_THINKING else ["}"],
            system_prompt=SYSTEM_PROMPT_ADVERSARIAL,
            max_gen_tokens=512,
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
