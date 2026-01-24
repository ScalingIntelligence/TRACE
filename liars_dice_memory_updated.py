"""
Liar's Dice with Synthetic Log-Style Memory Challenge.

This variant presents the game as a system log with User IDs.
Dice are revealed one by one, interleaved with noise from fake users/games.
Game actions (bids, calls) are also interleaved with distracting log entries.
No external history file needed - all noise is generated synthetically.
"""

import re
import random
from typing import List, Optional, Dict

# Pattern to match [dice_numbers][bid: X, Y] or [dice_numbers][call]
# dice_numbers is like [1,2,3,4,5] or [1, 2, 3, 4, 5]
_DICE_BID_RE = re.compile(r"\[([\d,\s]+)\]\s*\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]", re.IGNORECASE)
_DICE_CALL_RE = re.compile(r"\[([\d,\s]+)\]\s*\[call\]", re.IGNORECASE)


def extract_action(text: str, legal_actions: List[str]):
    """Extract action from model output, check if valid."""
    if not text:
        return None

    candidates = []

    for match in _DICE_BID_RE.finditer(text):
        dice_str, quantity, face = match.groups()
        # Parse dice numbers
        dice_numbers = [int(x.strip()) for x in dice_str.split(',') if x.strip()]
        dice_numbers_str = '[' + ','.join(map(str, sorted(dice_numbers))) + ']'
        action = f"{dice_numbers_str}[bid: {quantity}, {face}]"
        candidates.append((match.start(), action))

    for match in _DICE_CALL_RE.finditer(text):
        dice_str = match.group(1)
        # Parse dice numbers
        dice_numbers = [int(x.strip()) for x in dice_str.split(',') if x.strip()]
        dice_numbers_str = '[' + ','.join(map(str, sorted(dice_numbers))) + ']'
        action = f"{dice_numbers_str}[call]"
        candidates.append((match.start(), action))

    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, action in candidates:
        if action in legal_actions:
            return action

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
        
        # Generate dice for both fake players
        self.user1_dice = [rng.randint(1, 6) for _ in range(num_dice)]
        self.user2_dice = [rng.randint(1, 6) for _ in range(num_dice)]
        
        # Track dice reveal progress
        self.user1_revealed = 0
        self.user2_revealed = 0
        
        # Generate a sequence of bids and possibly a call
        self.bids = self._generate_bid_sequence()
        self.bid_idx = 0
        
    def _generate_bid_sequence(self) -> List[Dict]:
        """Generate a realistic sequence of bids ending in a call."""
        bids = []
        current_qty = 0
        current_face = 0
        current_bidder = self.rng.choice([self.user1_id, self.user2_id])
        
        # Generate 2-6 bids
        num_bids = self.rng.randint(2, 6)
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
            
            # Switch bidder
            current_bidder = self.user2_id if current_bidder == self.user1_id else self.user1_id
        
        # End with a call
        bids.append({
            "type": "call",
            "user": current_bidder,
        })
        
        return bids
    
    def get_next_event(self) -> Optional[str]:
        """Get the next event from this fake game, or None if done."""
        # First, reveal all dice for both users (interleaved)
        if self.user1_revealed < self.num_dice or self.user2_revealed < self.num_dice:
            # Decide which user reveals next
            if self.user1_revealed >= self.num_dice:
                return self._reveal_die(2)
            elif self.user2_revealed >= self.num_dice:
                return self._reveal_die(1)
            else:
                return self._reveal_die(self.rng.choice([1, 2]))
        
        # Then, play out the bids
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


class LiarsDiceMemoryUpdated:
    """
    Liar's Dice with synthetic log-style observations.
    
    The observation looks like a system log with multiple concurrent games.
    Your dice are revealed one by one, mixed with events from other fake games.
    Each fake game has two users who both initialize all their dice, then bid.
    """

    def __init__(
        self,
        num_dice: int = 5,
        num_fake_games: int = 3,  # Number of concurrent fake games
    ):
        self.num_dice = num_dice
        self.num_fake_games = num_fake_games

        # Game state
        self.dice: Dict[int, List[int]] = {}
        self.current_bid_quantity = 0
        self.current_bid_face = 0
        self.last_bidder: Optional[int] = None
        self.current_player = 0
        self.done = False
        self.invalid_player: Optional[int] = None
        self.rewards: Dict[int, float] = {0: 0.0, 1: 0.0}
        
        # User IDs (generated per game)
        self.player_ids: Dict[int, str] = {}
        
        # Fake games for noise
        self.fake_games: List[FakeGame] = []
        
        # Action history for display
        self.action_history: List[Dict] = []

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
        
        # Generate unique user IDs for real players
        self.player_ids = {
            0: _generate_user_id(self._rng),
            1: _generate_user_id(self._rng),
        }
        
        # Generate fake games with their own user pairs
        all_used_ids = set(self.player_ids.values())
        self.fake_games = []
        
        for _ in range(self.num_fake_games):
            # Generate two unique user IDs for this fake game
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
        
        self.action_history = []

    def _get_fake_event(self) -> Optional[str]:
        """Get a random event from one of the fake games."""
        active_games = [g for g in self.fake_games if not g.is_done()]
        if not active_games:
            return None
        game = self._rng.choice(active_games)
        return game.get_next_event()

    def _add_fake_events(self, lines: List[str], count: int):
        """Add random events from fake games."""
        for _ in range(count):
            event = self._get_fake_event()
            if event:
                lines.append(event)

    def _regenerate_fake_games(self):
        """
        Regenerate fake games fresh for each observation.
        This ensures we always have distractions available.
        """
        all_used_ids = set(self.player_ids.values())
        self.fake_games = []
        
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
        
        # Pre-advance each fake game through some of its dice reveals
        for game in self.fake_games:
            advances = self._rng.randint(6, 9)
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
        lines.append(f"Authentication Successful. Your User ID is: {my_id}. Your Opponent's User ID is: {opp_id}.")
        lines.append("")
        
        # Scale event counts based on num_fake_games
        base_events = max(5, self.num_fake_games // 5)
        per_die_events = max(3, self.num_fake_games // 10)
        
        # Add some initial fake events (now includes bids/calls from pre-advanced games)
        self._add_fake_events(lines, self._rng.randint(base_events, base_events * 2))
        
        # Reveal dice one by one with fake events in between
        for i, die in enumerate(player_dice):
            # Add fake events before each die reveal
            self._add_fake_events(lines, self._rng.randint(per_die_events, per_die_events * 2))
            
            # Reveal just this one die
            lines.append(f"User {my_id} die initialized: Face:{die}")
        
        # Add fake events after all dice revealed
        self._add_fake_events(lines, self._rng.randint(base_events, base_events * 2))
        
        # Show game action history interleaved with fake events
        for action_info in self.action_history:
            self._add_fake_events(lines, self._rng.randint(per_die_events, per_die_events * 2))
            
            acting_player = action_info["player"]
            action = action_info["action"]
            acting_id = self.player_ids[acting_player]
            
            # Parse action to extract dice_numbers and action type
            match = _DICE_CALL_RE.search(action)
            if match:
                lines.append(f"User {acting_id} called.")
            else:
                # Try bid format
                match = _DICE_BID_RE.search(action)
                if match:
                    dice_str, q, f = match.groups()
                    lines.append(f"User {acting_id} bid: {q}x[Face:{f}].")
        
        # Drain remaining fake events (use up to 3x base_events)
        self._add_fake_events(lines, self._rng.randint(base_events, base_events * 3))
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("=== CURRENT GAME ===")
        lines.append(f"You are Player {player_id} (User ID: {my_id}).")
        lines.append("")
        lines.append("Game Rules:")
        lines.append(f"- You are only playing a game with your opponent (User ID: {opp_id})")
        lines.append(f"- Each player has {self.num_dice} dice ({self.num_dice * 2} total in play)")
        lines.append("- Bids claim a MINIMUM count of a face value across ALL dice")
        lines.append("- You should preface your action with the dice you have.")
        lines.append("- '[dice_numbers][bid: X, Y]': Claim at least X dice show face Y (both players combined)")
        lines.append("- '[dice_numbers][call]': Challenge the last bid")
        lines.append("- If called and bid is WRONG (actual < bid): bidder loses")
        lines.append("- If called and bid is CORRECT (actual >= bid): caller loses")
        lines.append("- New bids must be HIGHER: more quantity OR same quantity with higher face")
        lines.append("")
        lines.append(f"Opponent (User ID: {opp_id}) has {self.num_dice} dice (hidden from you).")
        lines.append("")
        
        if self.last_bidder is not None:
            lines.append("Determine an optimal legal current bid based on the action so far.")
        else:
            lines.append("Determine an optimal legal opening bid based on your dice.")
        
        # Legal actions hint
        legal = self.legal_actions()[:8]
        legal_str = ", ".join(legal)
        if len(self.legal_actions()) > 8:
            legal_str += f", ... ({len(self.legal_actions())} total)"
        lines.append(f"Your available actions: {legal_str}")
        
        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        """Return list of legal actions for current player."""
        if self.done:
            return []

        # Get current player's dice and format as [dice_numbers]
        player_dice = sorted(self.dice[self.current_player])
        dice_numbers_str = '[' + ','.join(map(str, player_dice)) + ']'

        actions = []
        curr_q = self.current_bid_quantity
        curr_f = self.current_bid_face
        total_dice = self.num_dice * 2

        if self.last_bidder is not None:
            actions.append(f"{dice_numbers_str}[call]")

        if curr_q > 0 and curr_f < 6:
            for f in range(curr_f + 1, 7):
                actions.append(f"{dice_numbers_str}[bid: {curr_q}, {f}]")

        for q in range(curr_q + 1, total_dice + 1):
            for f in range(1, 7):
                actions.append(f"{dice_numbers_str}[bid: {q}, {f}]")

        return actions

    def observe(self, player_id: int) -> str:
        """Return observation for the player."""
        return self._build_observation(player_id)

    def step(self, action: Optional[str]):
        """Process an action."""
        if self.done:
            return

        if not action:
            self._terminate_invalid(self.current_player)
            return

        # Extract and validate dice_numbers
        player_dice = sorted(self.dice[self.current_player])
        dice_numbers = None
        action_type = None
        quantity = None
        face = None

        # Try to match bid format
        match = _DICE_BID_RE.search(action)
        if match:
            dice_str, qty_str, face_str = match.groups()
            dice_numbers = [int(x.strip()) for x in dice_str.split(',') if x.strip()]
            quantity = int(qty_str)
            face = int(face_str)
            action_type = "bid"
        else:
            # Try to match call format
            match = _DICE_CALL_RE.search(action)
            if match:
                dice_str = match.group(1)
                dice_numbers = [int(x.strip()) for x in dice_str.split(',') if x.strip()]
                action_type = "call"
            else:
                # Invalid format - no dice_numbers prefix
                self._terminate_invalid(self.current_player)
                return

        # Validate dice_numbers match player's actual dice
        if sorted(dice_numbers) != player_dice:
            self._terminate_invalid(self.current_player)
            return

        # Check if action is legal (bid quantity/face or call)
        legal = self.legal_actions()
        if action not in legal:
            self._terminate_invalid(self.current_player)
            return
        
        # Record action in history
        self.action_history.append({
            "player": self.current_player,
            "action": action,
        })

        if action_type == "call":
            self._resolve_call()
            return

        # action_type == "bid"
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
