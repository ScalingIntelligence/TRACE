from __future__ import annotations

"""
Lightweight OpenSpiel adapter.

Add entries to `OPENSPIEL_GAME_CONFIGS` to make new OpenSpiel games available to
the existing PPO self-play pipeline without changing training logic.

Each config maps an OpenSpiel game name (pyspiel.load_game) to:
- a fixed, stringified action space (used for guided decoding),
- a system prompt, and
- a simple text observation built from the OpenSpiel state.
"""

from pickletools import int4
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from game_registry import GameEnv, GameSpec
from config import Config


@dataclass
class OpenSpielGameConfig:
    """Declarative config for an OpenSpiel game."""

    name: str  # Registry key / --game flag value.
    openspiel_name: str  # pyspiel.load_game argument.
    system_prompt: str
    # Optional map: OpenSpiel action id -> string token exposed to the model.
    # If omitted, we create tokens like "[action_0]" up to num_distinct_actions().
    action_map: Optional[Dict[int, str]] = None
    # Optionally restrict the usable action ids to this subset.
    allowed_action_ids: Optional[List[int]] = None
    # Max new tokens to sample for one action completion.

    max_gen_tokens = Config.MAX_GEN_TOKENS
    # Which observation string to expose ("observation" or "information").
    observation_type: str = "observation"
    # Append legal actions to the observation text.
    include_legal_actions: bool = True
    # Reward to assign to the player who makes an invalid move.
    invalid_reward: float = -1.0
    # Optional stop sequences; defaults to the action strings.
    stop_sequences: Optional[List[str]] = None


class OpenSpielEnv(GameEnv):
    """Wrap a two-player OpenSpiel game to match the training interface."""

    def __init__(self, cfg: OpenSpielGameConfig):
        self.cfg = cfg
        try:
            import pyspiel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "OpenSpiel is not installed. Install with `pip install open_spiel` to use OpenSpiel games."
            ) from e

        self.pyspiel = pyspiel
        self.game = pyspiel.load_game(self.cfg.openspiel_name)
        if int(self.game.num_players()) != 2:
            raise ValueError(f"Game '{self.cfg.openspiel_name}' is not 2-player; only 2-player games are supported.")

        self._action_map = self._build_action_map()
        self._reverse_action_map = {v: k for k, v in self._action_map.items()}
        self.action_space = [self._action_map[a] for a in sorted(self._action_map)]

        self._rng = random.Random()
        self.reset(0)

    # --- lifecycle ---
    def reset(self, seed: int) -> None:
        self._rng.seed(int(seed))
        # Some OpenSpiel games expose seeded initial states; fall back to default.
        if hasattr(self.game, "new_initial_state_with_random_seed"):
            try:
                self.state = self.game.new_initial_state_with_random_seed(int(seed))
            except Exception:
                self.state = self.game.new_initial_state()
        else:
            self.state = self.game.new_initial_state()

        self.done = False
        self.invalid_player: Optional[int] = None
        self.rewards: Dict[int, float] = {0: 0.0, 1: 0.0}

        self._advance_chance_nodes()
        self.current_player = int(self.state.current_player())

    # --- helpers ---
    def _build_action_map(self) -> Dict[int, str]:
        """Create a stable mapping from OpenSpiel action ids -> string tokens."""
        if self.cfg.action_map:
            ids = self.cfg.allowed_action_ids or list(self.cfg.action_map.keys())
            return {int(a): str(self.cfg.action_map[a]) for a in ids if a in self.cfg.action_map}

        try:
            num_actions = int(self.game.num_distinct_actions())
            action_ids = self.cfg.allowed_action_ids or list(range(num_actions))
        except Exception:
            # Fallback: use legal actions from the initial state.
            action_ids = self.cfg.allowed_action_ids or list(self.game.new_initial_state().legal_actions())

        action_map: Dict[int, str] = {}
        for a in action_ids:
            token = f"[action_{int(a)}]"
            try:
                s = self.game.action_to_string(0, int(a))
                if s:
                    normalized = re.sub(r"[^A-Za-z0-9_]", "_", s).lower()
                    token = f"[{normalized}]" if not normalized.startswith("[") else normalized
            except Exception:
                pass
            action_map[int(a)] = token
        return action_map

    def _advance_chance_nodes(self) -> None:
        """Auto-play chance nodes so training only sees player turns."""
        while hasattr(self, "state") and self.state.is_chance_node():
            outcomes = self.state.chance_outcomes()
            if not outcomes:
                break
            actions, probs = zip(*outcomes)
            a = self._rng.choices(actions, weights=probs, k=1)[0]
            self.state.apply_action(int(a))

    # --- interface ---
    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        legal_ids = list(self.state.legal_actions())
        out = []
        for a in legal_ids:
            if a in self._action_map:
                out.append(self._action_map[a])
            else:
                out.append(f"[action_{int(a)}]")
        return out

    def observe(self, player_id: int) -> str:
        if self.cfg.observation_type == "information":
            base = self.state.information_state_string(player_id)
        else:
            base = self.state.observation_string(player_id)

        base = f"You are Player {player_id}.\n\n{base}"

        if self.cfg.include_legal_actions:
            legal = ", ".join(self.legal_actions())
            base = f"{base}\nLegal actions now: {legal}"
        return base

    def step(self, action: Optional[str]) -> None:
        if self.done:
            return
        if action is None or action not in self._reverse_action_map:
            self._terminate_invalid(self.current_player)
            return

        a_id = int(self._reverse_action_map[action])
        if a_id not in self.state.legal_actions():
            self._terminate_invalid(self.current_player)
            return

        self.state.apply_action(a_id)
        self._advance_chance_nodes()

        self.done = bool(self.state.is_terminal())
        if self.done:
            rets = self.state.returns()
            self.rewards = {i: float(rets[i]) for i in range(len(rets))}
            return

        self.current_player = int(self.state.current_player())

    # --- termination ---
    def _terminate_invalid(self, player_id: int) -> None:
        self.done = True
        self.invalid_player = int(player_id)
        other = 1 - player_id
        penalty = float(self.cfg.invalid_reward)
        self.rewards = {0: 0.0, 1: 0.0}
        self.rewards[player_id] = penalty
        if penalty < 0:
            self.rewards[other] = -penalty


def _extract_openspiel_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Return the first legal action token present in the text."""
    if not text:
        return None
    lower = text.lower()
    for a in legal_actions:
        if a.lower() in lower:
            return a
    # Fallback: pick the first integer that matches a legal token.
    for num in re.findall(r"-?\d+", lower):
        if num in legal_actions:
            return num
    return None


# Default configs: add new entries here to expose more OpenSpiel games.
OPENSPIEL_GAME_CONFIGS: Dict[str, OpenSpielGameConfig] = {
    # Example: simple, deterministic game with a small action space.
    "openspiel_tictactoe": OpenSpielGameConfig(
        name="openspiel_tictactoe",
        openspiel_name="tic_tac_toe",
        system_prompt=(
            "You are playing Tic-Tac-Toe. Respond including an action token and nothing else. "
            "Use the legal action tokens shown in your observation verbatim."
        ),
        max_gen_tokens=Config.MAX_GEN_TOKENS,
    ),
        "openspiel_breakthrough": OpenSpielGameConfig(
        name="openspiel_breakthrough",
        openspiel_name="breakthrough(rows=4,columns=4)",
        system_prompt=(
            "You are playing Breakthrough on a 4x4 board.\n"
            "\n"
            "RULES:\n"
            "- You control pieces shown as 'b' (black, moves down) or 'w' (white, moves up).\n"
            "- Goal: Get ANY one of your pieces to the opponent's back row.\n"
            "- Movement: Pieces move forward one square (straight or diagonal).\n"
            "- Capture: You can capture opponent pieces by moving diagonally into them.\n"
            "- You cannot move straight into an opponent's piece, only diagonally.\n"
            "\n"
            "ACTIONS:\n"
            "- Format: 'a2a3' means move from column a, row 2 to column a, row 3.\n"
            "- A '*' at the end (like 'a2b3*') indicates a capture.\n"
            "\n"
            "Choose one legal action from the list provided.\n"
        ),
        max_gen_tokens=Config.MAX_GEN_TOKENS,
    ),
    "openspiel_dots_and_boxes": OpenSpielGameConfig(
    name="openspiel_dots_and_boxes",
    openspiel_name="dots_and_boxes(num_rows=4,num_cols=4)",
    system_prompt=(
        "You are playing Dots and Boxes on a 4x4 grid.\n"
        "\n"
        "RULES:\n"
        "- The board is a grid of dots. Players take turns drawing lines between adjacent dots.\n"
        "- When you complete the 4th side of a box, you score 1 point and get another turn.\n"
        "- The game ends when all lines are drawn. Player with the most boxes wins.\n"
        "\n"
        "STRATEGY:\n"
        "- Avoid completing the 3rd side of a box (giving opponent an easy score).\n"
        "- Try to force opponent into giving you chains of boxes.\n"
        "\n"
        "ACTIONS:\n"
        "- P1(h,row,col): Draw a horizontal line at position (row, col).\n"
        "- P1(v,row,col): Draw a vertical line at position (row, col).\n"
        "- 'h' = horizontal, 'v' = vertical.\n"
        "\n"
        "Choose one legal action from the list provided.\n"
    ),
    max_gen_tokens=Config.MAX_GEN_TOKENS,
    ),
}


def _build_spec_from_config(cfg: OpenSpielGameConfig) -> GameSpec:
    try:
        preview_env = OpenSpielEnv(cfg)
        action_space = list(preview_env.action_space)
    except Exception:
        action_space = list(cfg.action_map.values()) if cfg.action_map else []

    def make_env(cfg: OpenSpielGameConfig = cfg) -> OpenSpielEnv:
        return OpenSpielEnv(cfg)

    stop_sequences = cfg.stop_sequences or (action_space if action_space else None)

    return GameSpec(
        name=cfg.name,
        make_env=make_env,
        extract_action=_extract_openspiel_action,
        action_space=action_space,
        stop_sequences=stop_sequences,
        system_prompt=cfg.system_prompt,
        max_gen_tokens=cfg.max_gen_tokens,
    )


def list_openspiel_game_specs() -> List[GameSpec]:
    """Return GameSpec entries for all configured OpenSpiel games."""
    specs: List[GameSpec] = []
    for cfg in OPENSPIEL_GAME_CONFIGS.values():
        try:
            specs.append(_build_spec_from_config(cfg))
        except Exception:
            # Don't block core training if a config is mis-specified.
            continue
    return specs
