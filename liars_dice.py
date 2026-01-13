"""
Liar's Dice game environment implementation.
"""


import re
import random
from typing import List, Optional, Dict, Tuple

_BID_RE = re.compile(r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]", re.IGNORECASE)
_CALL_RE = re.compile(r"\[call\]", re.IGNORECASE)

def extract_action(text: str, legal_actions: List[str]):
    """ Extract action from model output, check if valid """
    if not text:
        return None

    if _CALL_RE.search(text):
        if "[call]" in legal_actions:
            return "[call]"
        return None

    match = _BID_RE.search(text)
    if match:
        quantity = int(match.group(1))
        face = int(match.group(2))
        normalized = f"[bid: {quantity}, {face}]"
        if normalized in legal_actions:
            return normalized
    
    return None

class LiarsDice:
    """Single-round Liar's Dice """

    def __init__(self, num_dice: int = 5):
        self.num_dice = num_dice
        self.reset(0)

    def reset(self, seed: int):
        """ Reset the game. Takes only seed (like KuhnPoker) """

        rng = random.Random(int(seed))

        self.dice = {
            0: [rng.randint(1,6) for _ in range(self.num_dice)],
            1: [rng.randint(1,6) for _ in range(self.num_dice)],
        }

        self.current_bid_quantity = 0
        self.current_bid_face = 0
        self.last_bidder = None
        self.current_player = rng.randint(0,1)
        self.done = False
        self.invalid_player = None
        self.rewards = {0: 0.0, 1: 0.0}

    def legal_actions(self):
        """ Return list of legal actions for current player """
        if self.done:
            return []

        actions = []
        curr_q = self.current_bid_quantity
        curr_f = self.current_bid_face
        total_dice = self.num_dice * 2

        if self.last_bidder is not None:
            actions.append("[call]")

        if curr_q > 0 and curr_f  < 6:
            for f in range(curr_f + 1, 7):
                actions.append(f"[bid: {curr_q}, {f}]")
        
        for q in range(curr_q + 1, total_dice + 1):
            for f in range(1, 7):
                actions.append(f"[bid: {q}, {f}]")

        return actions


    def observe(self, player_id: int) -> str:
        """Generate observation string for a player."""
        my_dice = ", ".join(map(str, self.dice[player_id]))
        opp_id = 1 - player_id
        total_dice = self.num_dice * 2
        legal = ", ".join(self.legal_actions()[: 8])
        if len(self.legal_actions()) > 8:
            legal += f", ... ({len(self.legal_actions())} total)"

        obs = (
            f"[GAME] You are Player {player_id} in a 2-player Liar's Dice game.\n"
            f"Game Rules:\n"
            f"- Each player has {self.num_dice} dice ({total_dice} total in play)\n"
            f"- Bids claim a MINIMUM count of a face value across ALL dice\n"
            f"- '[bid: X, Y]': Claim at least X dice show face Y (both players combined)\n"
            f"- '[call]': Challenge the last bid\n"
            f"- If called and bid is WRONG (actual < bid): bidder loses\n"
            f"- If called and bid is CORRECT (actual >= bid): caller loses\n"
            f"- New bids must be HIGHER: more quantity OR same quantity with higher face\n\n"
            f"- If you think your new bid isn't likely to be correct, you should call the last bid\n"
            f"[GAME] Your dice (hidden from opponent): {my_dice}\n"
            f"[GAME] Opponent (Player {opp_id}) has {self.num_dice} dice (hidden)\n"
        )  

        if self.last_bidder is not None:
            obs += (
                f"[GAME] Current bid: {self.current_bid_quantity} of face {self.current_bid_face} "
                f"(by Player {self.last_bidder})\n"
            )
        else:
            obs += f"[GAME] No bid yet - you must make the first bid!\n"
        
        obs += f"Your available actions: {legal}\n"
        return obs

    def step(self, action: Optional[str]):
        """Process an action. No return value (like KuhnPoker)."""
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
            self.current_player = 1- self.current_player

    def _resolve_call(self):
        """Resolve a call action."""

        caller = self.current_player
        bidder = self.last_bidder

        target_face = self.current_bid_face
        actual_count = (
            self.dice[0].count(target_face) + self.dice[1].count(target_face)
        ) 

        bid_quantity = self.current_bid_quantity

        if actual_count < bid_quantity:
            winner, loser = caller, bidder

        else:
            winner, loser = bidder, caller
        
        self.done = True
        self.rewards = {winner: 1.0, loser: -1.0}

    def _terminate_invalid(self, player_id: int):
        """End game due to invalid move (like KuhnPoker)."""
        self.done = True
        self.invalid_player = player_id
        other = 1 - player_id
        self.rewards = {player_id: -1.5, other: 0.5}


        

    
