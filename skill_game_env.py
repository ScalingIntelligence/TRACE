"""
SkillGameEnv: SkyRL-Gym BaseTextEnv wrapper for all skill games.

Bridges the GameEnv protocol (game_registry.py) to SkyRL's BaseTextEnv
interface, enabling GRPO/PPO training via SkyRL for all registered games:
  - adversarial_policy (multi-turn, tool-calling)
  - tau_tool_calling (multi-turn, tool-calling)
  - multistep_task (multi-turn, tool-calling)
  - structured_data_reasoning (single-turn, answer extraction)
  - structured_data_single_turn (single-turn, tool-calling)
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)

# Ensure games directory is importable
_GAMES_DIR = os.path.dirname(os.path.abspath(__file__))
if _GAMES_DIR not in sys.path:
    sys.path.insert(0, _GAMES_DIR)

from game_registry import GAMES, GameSpec, get_game_spec

# Games that need an LLM-based user simulator
_USER_LLM_GAMES = {"adversarial_policy", "tau_tool_calling", "tau2_bench", "tau2_bench_airline", "tau2_bench_retail"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SkillGameEnvConfig:
    """Config for SkillGameEnv.

    ``user_llm_base_url`` and ``user_llm_model`` are required only when
    training adversarial_policy or tau_tool_calling (games that need
    an LLM to simulate user responses).
    """
    user_llm_base_url: Optional[str] = None
    user_llm_model: Optional[str] = None


# ---------------------------------------------------------------------------
# UserLLMClient helper
# ---------------------------------------------------------------------------

_cached_user_client = None


def _get_user_client(cfg: SkillGameEnvConfig):
    """Lazily create a shared UserLLMClient (one per worker process)."""
    global _cached_user_client
    if _cached_user_client is not None:
        return _cached_user_client

    base_url = cfg.user_llm_base_url if cfg else None
    model = cfg.user_llm_model if cfg else None

    # Fall back to env vars
    if not base_url:
        base_url = os.environ.get("USER_LLM_BASE_URL")
    if not model:
        model = os.environ.get("USER_LLM_MODEL")

    if not base_url or not model:
        raise ValueError(
            "adversarial_policy / tau_tool_calling require a user LLM. "
            "Set environment.skyrl_gym.skill_game.user_llm_base_url and "
            "user_llm_model in your config, or export USER_LLM_BASE_URL "
            "and USER_LLM_MODEL env vars."
        )

    from adversarial_policy_game.llm_user import UserLLMClient

    _cached_user_client = UserLLMClient(base_url=base_url, model=model)
    return _cached_user_client


# ---------------------------------------------------------------------------
# Game factory
# ---------------------------------------------------------------------------

def _make_game(game_name: str, cfg: SkillGameEnvConfig = None):
    """Create a game, injecting UserLLMClient when required."""
    spec = get_game_spec(game_name)
    if game_name in _USER_LLM_GAMES:
        client = _get_user_client(cfg)
        return spec.make_env(user_client=client)
    return spec.make_env()


# ---------------------------------------------------------------------------
# Prompt builders  (used by both the env and the dataset script)
# ---------------------------------------------------------------------------

def build_initial_prompt(
    game_name: str,
    seed: int,
    cfg: SkillGameEnvConfig = None,
) -> ConversationType:
    """Build the initial chat prompt for a game episode.

    Returns ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    spec = get_game_spec(game_name)
    game = _make_game(game_name, cfg)
    game.reset(seed)
    return _extract_prompt_from_game(game, spec)


def _extract_prompt_from_game(game, spec: GameSpec) -> ConversationType:
    """Extract (system_msg, user_msg) from an already-reset game instance."""

    # --- structured-message games: adversarial_policy, tau_tool_calling, structured_data_single_turn
    if getattr(game, "supports_structured_messages", False):
        system_content = game.get_system_prompt()
        tools = game.get_tool_schemas()
        system_content += (
            "\n\n<available_tools>\n"
            + json.dumps(tools, indent=2)
            + "\n</available_tools>"
        )
        system_content += (
            '\n\nRespond with a JSON tool call: '
            '{"name": "<tool_name>", "arguments": {<args>}}'
        )

        # First user message from get_messages()
        user_content = ""
        for msg in game.get_messages():
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # --- non-structured multi-turn games with _conversation: multistep_task
    if hasattr(game, "_conversation") and game._conversation:
        system_content = spec.system_prompt

        # Try to pull tool schemas from the game module
        tool_schemas = _get_tool_schemas_for_game(game)
        if tool_schemas:
            system_content += (
                "\n\n<available_tools>\n"
                + json.dumps(tool_schemas, indent=2)
                + "\n</available_tools>"
            )

        system_content += (
            '\n\nRespond with a JSON tool call: '
            '{"name": "<tool_name>", "arguments": {<args>}}'
        )

        # First user message from internal conversation
        user_content = game._conversation[0].get("text", "")

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # --- single-turn games without conversation: structured_data_reasoning
    return [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": game.observe(0)},
    ]


def _get_tool_schemas_for_game(game) -> Optional[List[Dict]]:
    """Attempt to retrieve tool schemas from various sources."""
    if hasattr(game, "get_tool_schemas"):
        return game.get_tool_schemas()

    # multistep_task_game stores TOOL_SCHEMAS at module level
    game_module = sys.modules.get(type(game).__module__)
    if game_module and hasattr(game_module, "TOOL_SCHEMAS"):
        return game_module.TOOL_SCHEMAS

    return None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SkillGameEnv(BaseTextEnv):
    """SkyRL-Gym environment wrapping any registered skill GameEnv.

    Required keys in *extras*:
        game_name (str): Name from game_registry (e.g. "adversarial_policy").
        seed (int): Deterministic episode seed.

    Optional keys:
        max_turns (int): Override for max environment turns (default 30).
    """

    def __init__(
        self,
        env_config: SkillGameEnvConfig = None,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        game_name = extras.get("game_name")
        if game_name is None:
            raise ValueError("extras must contain 'game_name'")
        seed = extras.get("seed")
        if seed is None:
            raise ValueError("extras must contain 'seed'")

        self._game_name: str = game_name
        self._seed: int = int(seed)
        self._cfg = env_config

        # Create and reset game
        self._spec: GameSpec = get_game_spec(game_name)
        self._game = _make_game(game_name, env_config)
        self._game.reset(self._seed)
        self._extract_action = self._spec.extract_action

        # Multi-turn bookkeeping
        self.max_turns = extras.get("max_turns", 30)
        self._prev_conv_len: int = self._conv_len()
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # BaseTextEnv interface
    # ------------------------------------------------------------------

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict]:
        """Return the pre-generated prompt from the dataset."""
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Process raw LLM text: parse → game.step → return observations."""
        self.turns += 1
        self._total_steps += 1

        # 1. Parse LLM output into a game action
        legal = self._game.legal_actions()
        parsed_action = self._extract_action(action, legal)

        # 2. Snapshot conversation length
        prev_len = self._conv_len()

        # 3. Execute
        self._game.step(parsed_action)

        # 4. Termination & reward
        done = self._game.done
        reward = self._game.rewards.get(0, 0.0) if done else 0.0

        # 5. Build observations from conversation delta (skip on done)
        observations: ConversationType = []
        if not done:
            delta = self._conv_delta(prev_len)
            obs_text = self._format_delta(delta)
            if obs_text:
                observations = [{"role": "user", "content": obs_text}]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata={
                "game_name": self._game_name,
                "steps": self._total_steps,
            },
        )

    def close(self):
        pass

    def get_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "game_name": self._game_name,
            "reward": self._game.rewards.get(0, 0.0),
            "steps": self._total_steps,
        }
        if hasattr(self._game, "get_summary"):
            summary = self._game.get_summary()
            for k, v in summary.items():
                if isinstance(v, (int, float, bool)):
                    metrics[k] = v
        return metrics

    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics_list:
            return {}

        rewards = [m.get("reward", 0.0) for m in metrics_list]
        steps = [m.get("steps", 0) for m in metrics_list]

        result: Dict[str, Any] = {
            "mean_reward": sum(rewards) / len(rewards),
            "mean_steps": sum(steps) / len(steps),
            "num_episodes": len(metrics_list),
        }

        # Per-game breakdown
        per_game: Dict[str, List[float]] = {}
        for m in metrics_list:
            gn = m.get("game_name", "unknown")
            per_game.setdefault(gn, []).append(m.get("reward", 0.0))
        for gn, rews in per_game.items():
            result[f"mean_reward/{gn}"] = sum(rews) / len(rews)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conv_len(self) -> int:
        conv = getattr(self._game, "_conversation", None)
        return len(conv) if conv is not None else 0

    def _conv_delta(self, prev_len: int) -> List[Dict[str, str]]:
        conv = getattr(self._game, "_conversation", None)
        if conv is None:
            return []
        return conv[prev_len:]

    @staticmethod
    def _format_delta(delta: List[Dict[str, str]]) -> str:
        """Format new conversation entries as observation text.

        Skips tool_call / assistant roles (already in the LLM's output).
        Includes tool_result and user messages.
        """
        parts: List[str] = []
        for msg in delta:
            role = msg.get("role", "")
            text = msg.get("text", msg.get("content", ""))
            if role in ("tool_call", "assistant"):
                continue
            elif role == "tool_result":
                parts.append(f"[TOOL_RESULT]: {text}")
            elif role == "user":
                parts.append(f"[USER]: {text}")
            else:
                parts.append(f"[{role.upper()}]: {text}")
        return "\n\n".join(parts)
