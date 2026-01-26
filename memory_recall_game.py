"""
Memory Recall Game - Tests LLM's ability to recall dice from noisy logs.

Each player sees their dice revealed one-by-one, interleaved with noise
from fake games. They must recall their dice correctly.

Both players take one turn each, then game ends.
Reward is based on correctness of recall.
"""

import re
import random
from typing import List, Optional, Dict
from collections import Counter


# Pattern to match dice recall: [dice: 1, 2, 3, 4, 5] or [dice: 1,2,3,4,5]
_DICE_RE = re.compile(r"\[dice\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", re.IGNORECASE)


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract dice recall from model output (finds LAST valid match)."""
    if not text:
        return None
    
    # Find all matches and take the last one (to avoid matching examples in prompt)
    matches = list(_DICE_RE.finditer(text))
    
    # Check matches from last to first
    for match in reversed(matches):
        dice = [int(d) for d in match.groups()]
        # Validate all dice are 1-6
        if all(1 <= d <= 6 for d in dice):
            return f"[dice: {dice[0]}, {dice[1]}, {dice[2]}, {dice[3]}, {dice[4]}]"
    
    return None


def _generate_user_id(rng: random.Random) -> str:
    """Generate a random user ID like '8X92-Alpha'."""
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    prefix = ''.join(rng.choices(chars, k=4))
    suffix = rng.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Omega', 'Sigma', 'Theta', 'Zeta'])
    return f"{prefix}-{suffix}"


class FakeGame:
    """A fake game used for generating realistic noise."""
    
    def __init__(self, rng: random.Random, num_dice: int, user1_id: str, user2_id: str):
        self.rng = rng
        self.num_dice = num_dice
        self.user1_id = user1_id
        self.user2_id = user2_id
        
        # Generate dice for fake players
        self.user1_dice = [rng.randint(1, 6) for _ in range(num_dice)]
        self.user2_dice = [rng.randint(1, 6) for _ in range(num_dice)]
        
        # Track reveal progress
        self.user1_revealed = 0
        self.user2_revealed = 0
        
        # Generate a realistic bid sequence
        self.bids = self._generate_bid_sequence()
        self.bid_idx = 0
    
    def _generate_bid_sequence(self) -> List[Dict]:
        """Generate a realistic sequence of bids for this fake game."""
        bids = []
        current_qty = 0
        current_face = 0
        current_bidder = self.rng.choice([self.user1_id, self.user2_id])
        
        # Generate 3-8 bids
        num_bids = self.rng.randint(3, 8)
        for _ in range(num_bids):
            # Increase bid
            if current_qty == 0:
                current_qty = self.rng.randint(1, 3)
                current_face = self.rng.randint(1, 6)
            else:
                if self.rng.random() < 0.5 and current_face < 6:
                    current_face = self.rng.randint(current_face + 1, 6)
                else:
                    current_qty += self.rng.randint(1, 2)
                    current_face = self.rng.randint(1, 6)
            
            bids.append({
                "type": "bid",
                "user": current_bidder,
                "qty": current_qty,
                "face": current_face,
            })
            # Alternate bidder
            current_bidder = self.user2_id if current_bidder == self.user1_id else self.user1_id
        
        # End with a call
        bids.append({
            "type": "call",
            "user": current_bidder,
        })
        return bids
    
    def get_next_event(self) -> Optional[str]:
        """Get next event from this fake game (die reveal or bid)."""
        # First reveal all dice
        if self.user1_revealed < self.num_dice or self.user2_revealed < self.num_dice:
            # Randomly choose which user reveals next
            if self.user1_revealed >= self.num_dice:
                return self._reveal_die(2)  # User1 done, reveal for user2
            elif self.user2_revealed >= self.num_dice:
                return self._reveal_die(1)  # User2 done, reveal for user1
            else:
                return self._reveal_die(self.rng.choice([1, 2]))
        
        # Then play out bids
        if self.bid_idx < len(self.bids):
            bid = self.bids[self.bid_idx]
            self.bid_idx += 1
            
            if bid["type"] == "bid":
                return f"User {bid['user']} bid: {bid['qty']}x[Face:{bid['face']}]."
            else:
                return f"User {bid['user']} called."
        
        return None
    
    def _reveal_die(self, user_num: int) -> str:
        """Reveal the next die for the specified user (1 or 2)."""
        if user_num == 1:
            die_value = self.user1_dice[self.user1_revealed]
            self.user1_revealed += 1
            user_id = self.user1_id
        else:
            die_value = self.user2_dice[self.user2_revealed]
            self.user2_revealed += 1
            user_id = self.user2_id
        
        return f"User {user_id} die initialized: Face:{die_value}"
    
    def is_done(self) -> bool:
        """Check if this fake game has no more events."""
        return (self.user1_revealed >= self.num_dice and 
                self.user2_revealed >= self.num_dice and 
                self.bid_idx >= len(self.bids))


class MemoryRecallGame:
    """
    Memory Recall Game - both players recall their dice from noisy logs.
    
    Turn 0: Player 0 recalls their dice
    Turn 1: Player 1 recalls their dice
    Game ends after both players have recalled.
    """
    
    def __init__(
        self,
        num_dice: int = 5,
        num_fake_games: int = 40,
    ):
        self.num_dice = num_dice
        self.num_fake_games = num_fake_games
        self._rng = random.Random()
        
        # Game state
        self.dice: Dict[int, List[int]] = {}
        self.player_ids: Dict[int, str] = {}
        self.current_player = 0
        self.turns_taken = 0
        self.done = False
        self.invalid_player: Optional[int] = None
        self.rewards: Dict[int, float] = {0: 0.0, 1: 0.0}
        
        # Fake games for noise
        self.fake_games: List[FakeGame] = []
    
    def reset(self, seed: int):
        """Reset the game with a new seed."""
        self._rng = random.Random(seed)
        
        # Generate dice for both players
        self.dice = {
            0: [self._rng.randint(1, 6) for _ in range(self.num_dice)],
            1: [self._rng.randint(1, 6) for _ in range(self.num_dice)],
        }
        
        # Generate user IDs
        self.player_ids = {
            0: _generate_user_id(self._rng),
            1: _generate_user_id(self._rng),
        }
        
        # Generate fake games
        self.fake_games = []
        all_used_ids = set(self.player_ids.values())
        
        for _ in range(self.num_fake_games):
            fake_user1 = _generate_user_id(self._rng)
            while fake_user1 in all_used_ids:
                fake_user1 = _generate_user_id(self._rng)
            all_used_ids.add(fake_user1)
            
            fake_user2 = _generate_user_id(self._rng)
            while fake_user2 in all_used_ids:
                fake_user2 = _generate_user_id(self._rng)
            all_used_ids.add(fake_user2)
            
            self.fake_games.append(FakeGame(
                rng=self._rng,
                num_dice=self.num_dice,
                user1_id=fake_user1,
                user2_id=fake_user2,
            ))
        
        # Reset game state
        self.current_player = 0
        self.turns_taken = 0
        self.done = False
        self.invalid_player = None
        self.rewards = {0: 0.0, 1: 0.0}
    
    def _get_fake_event(self) -> Optional[str]:
        """Get a random event from an active fake game."""
        active_games = [g for g in self.fake_games if not g.is_done()]
        if not active_games:
            return None
        game = self._rng.choice(active_games)
        return game.get_next_event()
    
    def _add_fake_events(self, lines: List[str], count: int):
        """Add multiple fake events to the lines list."""
        for _ in range(count):
            event = self._get_fake_event()
            if event:
                lines.append(event)
    
    def _regenerate_fake_games(self):
        """Regenerate fake games for fresh noise each observation."""
        self.fake_games = []
        all_used_ids = set(self.player_ids.values())
        
        for _ in range(self.num_fake_games):
            fake_user1 = _generate_user_id(self._rng)
            while fake_user1 in all_used_ids:
                fake_user1 = _generate_user_id(self._rng)
            all_used_ids.add(fake_user1)
            
            fake_user2 = _generate_user_id(self._rng)
            while fake_user2 in all_used_ids:
                fake_user2 = _generate_user_id(self._rng)
            all_used_ids.add(fake_user2)
            
            self.fake_games.append(FakeGame(
                rng=self._rng,
                num_dice=self.num_dice,
                user1_id=fake_user1,
                user2_id=fake_user2,
            ))
        
        # Pre-advance some fake games so we get bids/calls mixed in
        for game in self.fake_games:
            advances = self._rng.randint(3, 8)
            for _ in range(advances):
                if not game.is_done():
                    game.get_next_event()
    
    def _build_observation(self, player_id: int) -> str:
        """Build the log-style observation with interleaved fake game events."""
        my_id = self.player_ids[player_id]
        opp_id = self.player_ids[1 - player_id]
        player_dice = self.dice[player_id]
        
        # Regenerate fresh fake games for each observation
        self._regenerate_fake_games()
        
        lines = []
        
        # Header
        lines.append(f"Authentication Successful. Your User ID is: {my_id}.")
        lines.append("")
        
        # Scale event counts based on num_fake_games
        base_events = max(5, self.num_fake_games // 5)
        per_die_events = max(3, self.num_fake_games // 10)
        
        # Add initial fake events
        self._add_fake_events(lines, self._rng.randint(base_events, base_events * 2))
        
        # Reveal dice one by one with fake events in between
        for i, die in enumerate(player_dice):
            # Add fake events before each die reveal
            self._add_fake_events(lines, self._rng.randint(per_die_events, per_die_events * 2))
            
            # Reveal just this one die
            lines.append(f"User {my_id} die initialized: Face:{die}")
        
        # Add fake events after all dice revealed
        self._add_fake_events(lines, self._rng.randint(base_events, base_events * 2))
        
        # Drain remaining fake events
        self._add_fake_events(lines, self._rng.randint(base_events, base_events * 3))
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("=== MEMORY RECALL TASK ===")
        lines.append(f"You are User ID: {my_id}")
        lines.append("")
        lines.append("Task: Recall YOUR dice values from the log above.")
        lines.append("Your dice were shown as 'User {your_id} die initialized: Face:X' entries.")
        lines.append("Ignore all other users' dice - only recall YOUR OWN.")
        lines.append("")
        lines.append(f"You have {self.num_dice} dice. State them in the format:")
        lines.append("[dice: A, B, C, D, E]")
        lines.append("")
        lines.append("Example: [dice: 3, 1, 4, 6, 2]")
        
        return "\n".join(lines)
    
    def legal_actions(self) -> List[str]:
        """Return all possible dice recall actions."""
        if self.done:
            return []
        
        # Generate all possible 5-dice combinations (6^5 = 7776 options)
        # This is too many, so we just validate in step() instead
        # Return a placeholder that indicates format
        return ["[dice: ?, ?, ?, ?, ?]"]
    
    def observe(self, player_id: int) -> str:
        """Get observation for a player."""
        return self._build_observation(player_id)
    
    def step(self, action: Optional[str]):
        """Process Player 0's recall action. Game ends immediately after."""
        if self.done:
            return
        
        actual_dice = self.dice[0]
        
        # Parse the recalled dice
        if action is None:
            recalled_dice = None
        else:
            match = _DICE_RE.search(action)
            if match:
                recalled_dice = [int(d) for d in match.groups()]
            else:
                recalled_dice = None
        
        # Calculate reward based on correctness (order doesn't matter)
        if recalled_dice is None:
            # Failed to parse - penalty
            self.rewards = {0: -1.0, 1: 1.0}
        elif sorted(recalled_dice) == sorted(actual_dice):
            # Correct recall (same set of dice)
            self.rewards = {0: 1.0, 1: -1.0}
        else:
            # Wrong - no partial credit
            self.rewards = {0: -1.0, 1: 1.0}
        
        # Game ends after Player 0's turn
        self.done = True
