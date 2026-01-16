#!/usr/bin/env python3
"""
Kuhn Poker game environment implementation.
"""
import re
import random
from typing import List, Optional

from config import ACTION_STRS


# =========================
# Action extraction
# =========================
_ACTION_RE = re.compile(r"\[(check|bet|call|fold)\]", re.IGNORECASE)
_CARD_RANK = {"J": 0, "Q": 1, "K": 2}


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """
    Take the last legal action found.
    With strict decoding, the completion should be exactly one action anyway.
    """
    candidates = []
    for match in _ACTION_RE.finditer(text or ""):
        action = f"[{match.group(1).lower()}]"
        candidates.append((match.start(), action))
    
    # Sort by position (latest first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, action in candidates:
        if action in legal_actions:
            return action
    return None


# =========================
# Kuhn Poker Environment
# =========================
class KuhnPoker:
    def __init__(self, num_rounds: int = 5):
        self.num_rounds = num_rounds
        self.reset(0)

    def reset(self, seed: int):
        rng = random.Random(int(seed))
        self.start_player0 = rng.randint(0, 1)
        self.round_cards = [rng.sample(["J", "Q", "K"], 2) for _ in range(self.num_rounds)]
        self.round_idx = 1
        self.chips = [0, 0]
        self.history = []
        self.actions_in_round = []
        self.bet_by = None
        self.current_player = self._round_start_player()
        self.done = False
        self.invalid_player = None
        self.rewards = {0: 0.0, 1: 0.0}

    def _round_start_player(self) -> int:
        return (self.start_player0 + (self.round_idx - 1)) % 2

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return ["[check]", "[bet]"] if self.bet_by is None else ["[call]", "[fold]"]

    def _card(self, player_id: int) -> str:
        return self.round_cards[self.round_idx - 1][player_id]

    def _round_history_str(self) -> str:
        if not self.actions_in_round:
            return "None"
        return " ".join([f"P{p}:{a}" for p, a in self.actions_in_round])

    def _full_history_str(self) -> str:
        if not self.history:
            return "None"
        return " ".join([f"R{r}P{p}:{a}" for r, p, a in self.history])

    def observe(self, player_id: int) -> str:
        legal = ", ".join(self.legal_actions())
        return (
            f"[GAME] You are Player {player_id} in a {self.num_rounds} round game of Kuhn Poker.\n"
            "Game Rules:\n"
            "- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)\n"
            "- Each player antes 1 chip and receives 1 card each round\n"
            f"- Game continues for {self.num_rounds} rounds\n"
            "- The player with the most chips after all rounds wins\n"
            "Action Rules:\n"
            "- '[check]': Pass without betting (only if no bet is on the table)\n"
            "- '[bet]': Add 1 chip to the pot (only if no bet is on the table)\n"
            "- '[call]': Match an opponent's bet by adding 1 chip to the pot\n"
            "- '[fold]': Surrender your hand and let your opponent win the pot\n"
            f"[GAME] Scores (chips won so far): Player 0: {self.chips[0]}, Player 1: {self.chips[1]}\n"
            f"[GAME] Starting round {self.round_idx} out of {self.num_rounds} rounds.\n"
            f"Your card is: {self._card(player_id)}\n"
            f"[GAME] Betting history this round: {self._round_history_str()}\n"
            f"[GAME] Full game history: {self._full_history_str()}\n"
            f"Your available actions are: {legal}\n"
        )

    def _end_round_showdown(self, pot_win: int):
        c0, c1 = self._card(0), self._card(1)
        winner = 0 if _CARD_RANK[c0] > _CARD_RANK[c1] else 1
        payoff = pot_win if winner == 0 else -pot_win
        self.chips[0] += payoff
        self.chips[1] -= payoff
        self.round_idx += 1
        self.actions_in_round = []
        self.bet_by = None
        if self.round_idx > self.num_rounds:
            self.done = True
            outcome = 0
            if self.chips[0] > self.chips[1]:
                outcome = 1
            elif self.chips[0] < self.chips[1]:
                outcome = -1
            self.rewards = {0: float(outcome), 1: float(-outcome)}
        else:
            self.current_player = self._round_start_player()

    def _terminate_invalid(self, player_id: int):
        self.done = True
        self.invalid_player = player_id
        other = 1 - player_id
        self.rewards = {0: 0.5, 1: 0.5}
        self.rewards[player_id] = -1.5
        self.rewards[other] = 0.5

    def step(self, action: Optional[str]):
        if self.done:
            return
        if action not in self.legal_actions():
            self._terminate_invalid(self.current_player)
            return

        p = self.current_player
        self.actions_in_round.append((p, action))
        self.history.append((self.round_idx, p, action))

        if self.bet_by is None:
            if action == "[bet]":
                self.bet_by = p
                self.current_player = 1 - p
                return
            if len(self.actions_in_round) == 2:
                self._end_round_showdown(1)
                return
            self.current_player = 1 - p
            return

        if action == "[fold]":
            bettor = self.bet_by
            payoff = 1 if bettor == 0 else -1
            self.chips[0] += payoff
            self.chips[1] -= payoff
            self.round_idx += 1
            self.actions_in_round = []
            self.bet_by = None
            if self.round_idx > self.num_rounds:
                self.done = True
                outcome = 0
                if self.chips[0] > self.chips[1]:
                    outcome = 1
                elif self.chips[0] < self.chips[1]:
                    outcome = -1
                self.rewards = {0: float(outcome), 1: float(-outcome)}
            else:
                self.current_player = self._round_start_player()
            return

        self._end_round_showdown(2)

