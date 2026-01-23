"""
Liar's Dice with Memory Augmentation.

This variant interleaves dice roll reveals throughout game history,
testing the model's ability to track state across long, distracting context.
"""

import re
import random
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Global cache - loaded once, never updated during training
_CACHED_HISTORIES: Optional[List[str]] = None
_MAX_CACHED_GAMES = 10000

_BID_RE = re.compile(r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]", re.IGNORECASE)
_CALL_RE = re.compile(r"\[call\]", re.IGNORECASE)


def extract_action(text: str, legal_actions: List[str]):
    """Extract action from model output, check if valid."""
    if not text:
        return None

    candidates = []

    for match in _BID_RE.finditer(text):
        quantity, face = match.groups()
        action = f"[bid: {quantity}, {face}]"
        candidates.append((match.start(), action))

    for match in _CALL_RE.finditer(text):
        candidates.append((match.start(), "[call]"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, action in candidates:
        if action in legal_actions:
            return action

    return None


def _load_and_cache_histories(source_path: Path, min_games: int = 10) -> List[str]:
    """
    Load game histories from file ONCE and cache globally.
    Only keeps up to _MAX_CACHED_GAMES games.
    
    Handles interleaved games (from parallel collection) by grouping entries
    by game_id before formatting.
    
    Raises:
        FileNotFoundError: If source_path doesn't exist.
        ValueError: If fewer than min_games valid games are found.
    """
    global _CACHED_HISTORIES

    if _CACHED_HISTORIES is not None:
        return _CACHED_HISTORIES

    if not source_path.exists():
        raise FileNotFoundError(
            f"[LiarsDiceMemory] History file not found: {source_path}\n"
            f"Please create a game history file first by running normal Liar's Dice training, "
            f"then copy the rollouts file: cp selfplay_rollouts_ppo.jsonl {source_path}"
        )

    print(f"[LiarsDiceMemory] Loading game histories from {source_path}...")

    # First pass: collect all entries grouped by game_id
    # Each entry is (turn_idx or large number for game_end, entry_dict)
    games_dict: Dict[Any, List[tuple]] = {}
    
    with open(source_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            game_id = entry.get("game_id")
            if game_id is None:
                continue
            
            entry_type = entry.get("type", "")
            
            # Use turn_idx for ordering, game_end goes last (use large number)
            if entry_type == "step":
                turn_idx = entry.get("turn_idx", 0)
                order_key = turn_idx
            elif entry_type == "game_end":
                order_key = 999999  # Ensure game_end comes last
            else:
                continue
            
            if game_id not in games_dict:
                games_dict[game_id] = []
            games_dict[game_id].append((order_key, entry))
            
            # Stop if we have enough games
            if len(games_dict) >= _MAX_CACHED_GAMES:
                break

    print(f"[LiarsDiceMemory] Found {len(games_dict)} unique game IDs, formatting...")

    # Second pass: format each game into readable text
    games = []
    for game_id in sorted(games_dict.keys()):
        entries = games_dict[game_id]
        # Sort by turn_idx to get correct order
        entries.sort(key=lambda x: x[0])
        
        lines = []
        for _, entry in entries:
            entry_type = entry.get("type", "")
            
            if entry_type == "step":
                pid = entry.get("player_id", "?")
                action = entry.get("action", "unknown")
                lines.append(f"Player {pid} played {action}.")
            
            elif entry_type == "game_end":
                rewards = entry.get("rewards", {})
                r0 = rewards.get("0", rewards.get(0, 0))
                r1 = rewards.get("1", rewards.get(1, 0))
                if r0 > r1:
                    lines.append("Result: Player 0 wins.")
                elif r1 > r0:
                    lines.append("Result: Player 1 wins.")
                else:
                    lines.append("Result: Draw.")
        
        # Only keep complete Liar's Dice games
        # Must have a result AND contain valid liar's dice actions (bid or call)
        game_text = "\n".join(lines)
        is_complete = "Result:" in game_text
        is_liars_dice = "[bid:" in game_text.lower() or "[call]" in game_text.lower()
        
        if lines and is_complete and is_liars_dice:
            games.append(game_text)

    if len(games) < min_games:
        raise ValueError(
            f"[LiarsDiceMemory] Only found {len(games)} valid games in {source_path}, "
            f"but need at least {min_games}.\n"
            f"Please run normal Liar's Dice training first to generate more game histories."
        )

    # Trim to max
    games = games[:_MAX_CACHED_GAMES]

    print(f"[LiarsDiceMemory] Cached {len(games)} game histories.")
    _CACHED_HISTORIES = games
    return _CACHED_HISTORIES


class LiarsDiceMemory:
    """
    Liar's Dice with memory-augmented observations.

    Dice are revealed one at a time via "interruptions" randomly
    placed within historical game logs, testing the model's ability
    to track and remember all dice values from noisy context.
    """

    def __init__(
        self,
        num_dice: int = 5,
        history_source: Optional[Path] = None,
        num_history_games: int = 4,
    ):
        self.num_dice = num_dice
        self.num_history_games = num_history_games
        self.history_source = history_source or Path("selfplay_rollouts_ppo.jsonl")

        # Load histories (uses global cache)
        _load_and_cache_histories(self.history_source)

        # Game state
        self.dice: Dict[int, List[int]] = {}
        self.current_bid_quantity = 0
        self.current_bid_face = 0
        self.last_bidder: Optional[int] = None
        self.current_player = 0
        self.done = False
        self.invalid_player: Optional[int] = None
        self.rewards: Dict[int, float] = {0: 0.0, 1: 0.0}

        # RNG for this instance
        self._rng = random.Random()

    def reset(self, seed: int):
        """Reset the game with a seed."""
        self._rng = random.Random(int(seed))

        self.dice = {
            0: [self._rng.randint(1, 6) for _ in range(self.num_dice)],
            1: [self._rng.randint(1, 6) for _ in range(self.num_dice)],
        }

        self.current_bid_quantity = 0
        self.current_bid_face = 0
        self.last_bidder = None
        self.current_player = self._rng.randint(0, 1)
        self.done = False
        self.invalid_player = None
        self.rewards = {0: 0.0, 1: 0.0}

    def legal_actions(self) -> List[str]:
        """Return list of legal actions for current player."""
        if self.done:
            return []

        actions = []
        curr_q = self.current_bid_quantity
        curr_f = self.current_bid_face
        total_dice = self.num_dice * 2

        if self.last_bidder is not None:
            actions.append("[call]")

        if curr_q > 0 and curr_f < 6:
            for f in range(curr_f + 1, 7):
                actions.append(f"[bid: {curr_q}, {f}]")

        for q in range(curr_q + 1, total_dice + 1):
            for f in range(1, 7):
                actions.append(f"[bid: {q}, {f}]")

        return actions

    def _build_augmented_observation(self, player_id: int) -> str:
        """Build observation with dice interleaved in historical games."""
        player_dice = self.dice[player_id]

        # Sample historical games from cache
        histories = _CACHED_HISTORIES or []
        if len(histories) >= self.num_history_games:
            selected = self._rng.sample(histories, self.num_history_games)
        else:
            selected = histories[:self.num_history_games]

        # Combine all history text
        combined = "\n\n".join([
            f"=== Historical Game {i+1} ===\n{hist}"
            for i, hist in enumerate(selected)
        ])

        # Split into lines for interleaving
        lines = combined.split("\n")

        # Create dice interruption messages
        interruptions = [
            f">>> INTERRUPTION: You just rolled a {die}. <<<"
            for die in player_dice
        ]

        # Choose random positions to insert (avoid very start)
        valid_positions = list(range(2, max(3, len(lines))))
        num_to_insert = min(len(valid_positions), len(interruptions))
        insert_positions = sorted(self._rng.sample(valid_positions, num_to_insert))

        # Build result with interruptions
        result_lines = []
        interrupt_idx = 0

        for i, line in enumerate(lines):
            result_lines.append(line)
            if interrupt_idx < len(insert_positions) and i == insert_positions[interrupt_idx]:
                result_lines.append("")
                result_lines.append(interruptions[interrupt_idx])
                result_lines.append("")
                interrupt_idx += 1

        # Add any remaining interruptions at end
        while interrupt_idx < len(interruptions):
            result_lines.append("")
            result_lines.append(interruptions[interrupt_idx])
            interrupt_idx += 1

        augmented = "\n".join(result_lines)

        # Build full observation
        intro = (
            "We're going to roll the current dice for you live now. "
            "As we do so, we'll show histories of past games, interrupted by the results of your dice rolls.\n\n"
        )

        outro = (
            "\n\n" + "=" * 50 + "\n" +
            "You've now been told all your dice.\n" +
            "=" * 50 + "\n\n"
        )

        game_state = self._format_game_state(player_id)

        return intro + augmented + outro + game_state

    def _format_game_state(self, player_id: int) -> str:
        """Format current game state."""
        opp_id = 1 - player_id
        total_dice = self.num_dice * 2

        legal = self.legal_actions()[:8]
        legal_str = ", ".join(legal)
        if len(self.legal_actions()) > 8:
            legal_str += f", ... ({len(self.legal_actions())} total)"

        lines = [
            "=== CURRENT GAME ===",
            f"You are Player {player_id} in a 2-player Liar's Dice game.",
            "",
            "Game Rules:",
            f"- Each player has {self.num_dice} dice ({total_dice} total in play)",
            "- Bids claim a MINIMUM count of a face value across ALL dice",
            "- '[bid: X, Y]': Claim at least X dice show face Y (both players combined)",
            "- '[call]': Challenge the last bid",
            "- If called and bid is WRONG (actual < bid): bidder loses",
            "- If called and bid is CORRECT (actual >= bid): caller loses",
            "- New bids must be HIGHER: more quantity OR same quantity with higher face",
            "",
            "IMPORTANT: Recall the dice you were told during the INTERRUPTIONS above!",
            f"Opponent (Player {opp_id}) has {self.num_dice} dice (hidden from you).",
            "",
        ]

        if self.last_bidder is not None:
            lines.append(
                f"Current bid: {self.current_bid_quantity} of face {self.current_bid_face} "
                f"(by Player {self.last_bidder})"
            )
        else:
            lines.append("No bid yet - you must make the first bid!")

        lines.append(f"Your available actions: {legal_str}")

        return "\n".join(lines)

    def observe(self, player_id: int) -> str:
        """Return augmented observation for the player."""
        return self._build_augmented_observation(player_id)

    def step(self, action: Optional[str]):
        """Process an action."""
        if self.done:
            return

        legal = self.legal_actions()
        if action not in legal:
            self._terminate_invalid(self.current_player)
            return

        if action == "[call]":
            self._resolve_call()
            return

        match = _BID_RE.search(action)
        if match:
            quantity = int(match.group(1))
            face = int(match.group(2))
            self.current_bid_quantity = quantity
            self.current_bid_face = face
            self.last_bidder = self.current_player
            self.current_player = 1 - self.current_player

    def _resolve_call(self):
        """Resolve a call action."""
        caller = self.current_player
        bidder = self.last_bidder

        target_face = self.current_bid_face
        actual_count = (
            self.dice[0].count(target_face) + self.dice[1].count(target_face)
        )

        if actual_count < self.current_bid_quantity:
            winner, loser = caller, bidder
        else:
            winner, loser = bidder, caller

        self.done = True
        self.rewards = {winner: 1.0, loser: -1.0}

    def _terminate_invalid(self, player_id: int):
        """End game due to invalid move."""
        self.done = True
        self.invalid_player = player_id
        other = 1 - player_id
        self.rewards = {player_id: -1.5, other: 0.5}
