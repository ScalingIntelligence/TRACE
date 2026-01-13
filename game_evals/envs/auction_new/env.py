# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auction game environment for SPIRAL."""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType


class AuctionEnv(ta.Env):
    """
    Auction game environment.
    
    Simultaneous bidding game where players compete for points across multiple rounds.
    Each round is worth increasing points - strategic resource allocation is key.
    
    Game mechanics:
    - Both players start with $100
    - 10 rounds of simultaneous bidding
    - Round J awards J points to the highest bidder
    - Bids are subtracted from player budgets
    - Player with most total points wins
    
    Strategic depth:
    - Resource management over time
    - Valuation of different rounds
    - Opponent modeling and bluffing
    - All-pay auction dynamics
    """
    
    def __init__(self, starting_money: int = 100, num_rounds: int = 10):
        """
        Initialize Auction environment.
        
        Args:
            starting_money: Starting budget for each player (default: $100)
            num_rounds: Number of auction rounds (default: 10)
        """
        super().__init__()
        self.starting_money = starting_money
        self.num_rounds = num_rounds
        
        # Bid action pattern: [bid_amount] - INTEGER ONLY
        self.action_pattern = re.compile(r"\[(\d+)\]")
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            random.seed(seed)
        
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.num_rounds * 2,  # Each round has 2 moves (one per player)
        )
        
        # Initialize game state
        game_state = {
            "current_round": 1,
            "budgets": {0: float(self.starting_money), 1: float(self.starting_money)},
            "points": {0: 0, 1: 0},
            "pending_bids": {},  # Stores bids before both players submit
            "round_history": [],  # List of (round, bid0, bid1, winner)
        }
        
        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Show initial state to Player 0
        state_msg = self._format_game_state()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=0,
            message=state_msg,
            observation_type=ObservationType.GAME_MESSAGE
        )
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate initial instructions for a player."""
        prompt = (
            f"You are Player {player_id} in an Auction game.\n\n"
            f"═══════════════════════════════════════\n"
            f"GAME SETUP:\n"
            f"═══════════════════════════════════════\n"
            f"- Starting budget: ${self.starting_money} per player\n"
            f"- Number of rounds: {self.num_rounds}\n"
            f"- Goal: Win the most points across all rounds\n\n"
            f"GAME RULES:\n\n"
            f"1. Each round, both players simultaneously submit a bid\n"
            f"2. Round J (1 to {self.num_rounds}) awards J points to the winner\n"
            f"   - Round 1 = 1 point, Round 2 = 2 points, ..., Round {self.num_rounds} = {self.num_rounds} points\n"
            f"3. The player with the HIGHER bid wins the round\n"
            f"4. BOTH players pay their bid (subtracted from budget)\n"
            f"   - Even the loser pays their bid!\n"
            f"   - This is an 'all-pay auction'\n"
            f"5. If both players bid the same amount:\n"
            f"   - It's a TIE - nobody gets the points\n"
            f"   - Both still pay their bids\n"
            f"6. After {self.num_rounds} rounds, player with most points WINS\n"
            f"7. If tied on points, player with more money remaining wins\n\n"
            f"ACTION FORMAT:\n\n"
            f"⚠️  YOU MUST RESPOND WITH EXACTLY THIS FORMAT: [bid_amount]\n\n"
            f"⚠️  BID FORMAT: [INTEGER]\n"
            f"   - INTEGER = your bid in dollars (MUST BE A WHOLE NUMBER)\n"
            f"   - NO DECIMALS ALLOWED! (e.g., [10] is valid, [10.5] is INVALID)\n"
            f"   - Must be between 0 and your remaining budget (inclusive)\n"
            f"   - Examples: [10] = $10 bid\n"
            f"   - Examples: [25] = $25 bid\n"
            f"   - Examples: [0] = $0 bid (valid but you won't win)\n\n"
            f"⚠️  IMPORTANT RULES:\n"
            f"   - You CANNOT bid more than your remaining budget\n"
            f"   - You CANNOT bid negative amounts\n"
            f"   - Both players submit bids simultaneously (hidden from each other)\n"
            f"   - After both submit, bids are revealed and winner determined\n"
            f"   - Invalid moves cause you to LOSE immediately\n\n"
            f"EXAMPLES:\n\n"
            f"  [5] = Bid $5 on this round\n"
            f"        • You pay $5 regardless of outcome\n"
            f"        • If opponent bids less, you win the round's points\n"
            f"        • If opponent bids more, you lose and still pay $5\n\n"
            f"  [0] = Bid $0 (forfeit this round)\n"
            f"        • You pay nothing\n"
            f"        • You will lose unless opponent also bids $0\n"
            f"        • Strategic when saving for later rounds\n\n"
            f"  [50] = Bid $50 (INTEGER ONLY!)\n"
            f"         • Large bid for an important round\n"
            f"         • Leaves you with less budget for future rounds\n\n"
            f"STRATEGY TIPS:\n\n"
            f"- Later rounds are worth more points (Round 10 = 10 points!)\n"
            f"- You can't win every round - choose which rounds to contest\n"
            f"- Save budget for high-value rounds\n"
            f"- Consider opponent's remaining budget\n"
            f"- Even losing bids cost money\n"
            f"- Total points available: {sum(range(1, self.num_rounds + 1))} points\n"
            f"- Need to win rounds worth > {sum(range(1, self.num_rounds + 1)) // 2} points to guarantee victory\n"
        )
        return prompt
    
    def get_observation(self) -> Tuple[int, str]:
        """Get current player ID and their observation."""
        player_id = self.state.current_player_id
        observation = self.state.get_current_player_observation()
        return player_id, observation
    
    def get_valid_actions(self, player_id: int) -> List[str]:
        """Get list of valid actions for a player.
        
        Args:
            player_id: ID of the player (0 or 1)
            
        Returns:
            List of valid action strings in format ["[0]", "[1]", ..., "[budget]"]
        """
        budget = int(self.state.game_state['budgets'][player_id])
        # Generate ALL valid integer bids from 0 to budget (inclusive)
        return [f"[{bid}]" for bid in range(0, budget + 1)]
    
    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Process an action and update game state."""
        player_id = self.state.current_player_id
        opponent_id = 1 - player_id
        
        # Parse the bid
        match = self.action_pattern.search(action)
        if not match:
            self.state.set_invalid_move(reason=(
                    f"⚠️  INVALID FORMAT! You must use [bid_amount] where:\n\n"
                    f"  REQUIRED FORMAT: [INTEGER]\n"
                    f"  • INTEGER = your bid in dollars (WHOLE NUMBER ONLY)\n"
                    f"  • NO DECIMALS! Use integers like [10] not [10.5]\n"
                    f"  • Must be between 0 and your budget (${int(self.state.game_state['budgets'][player_id])})\n\n"
                    f"  Examples:\n"
                    f"  [10] = bid $10 ✅\n"
                    f"  [0] = bid $0 (forfeit round) ✅\n"
                    f"  [5.50] = INVALID (no decimals!) ❌\n\n"
                    f"  Your response must be EXACTLY in this format with brackets!"
                )
            )
            return self.state.step()
        
        try:
            bid = int(match.group(1))
        except ValueError:
            self.state.set_invalid_move(reason="Invalid bid amount - must be an INTEGER (whole number)."
            )
            return self.state.step()
        
        # Validate bid is non-negative (should always pass since regex only matches \d+)
        if bid < 0:
            self.state.set_invalid_move(reason=f"⚠️  INVALID: Cannot bid negative amounts!\n\nYou tried to bid ${bid}.\nBids must be >= 0."
            )
            return self.state.step()
        
        # Validate bid doesn't exceed budget
        budget = int(self.state.game_state["budgets"][player_id])
        if bid > budget:
            self.state.set_invalid_move(reason=(
                    f"⚠️  INVALID: Bid exceeds your budget!\n\n"
                    f"You tried to bid ${bid}\n"
                    f"Your remaining budget: ${budget}\n"
                    f"Maximum you can bid: ${budget}"
                )
            )
            return self.state.step()
        
        # Store the bid (privately - opponent doesn't see it yet)
        self.state.game_state["pending_bids"][player_id] = bid
        
        # Log action privately (only this player sees their own bid)
        self.state.add_observation(
            from_id=player_id,
            to_id=player_id,
            message=f"You bid ${bid}. Waiting for opponent..."
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Check if both players have submitted bids
        if len(self.state.game_state["pending_bids"]) == 2:
            # Both bids submitted - resolve the round
            self._resolve_round()
            
            # Check if game is over
            if self.state.game_state["current_round"] > self.num_rounds:
                # All rounds complete - determine winner
                self._determine_final_winner()
                return self.state.step()
            
            # Show updated state to both players
            state_msg = self._format_game_state()
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=state_msg,
                observation_type=ObservationType.GAME_MESSAGE
            )
            
            return self.state.step()
        else:
            # Only one player has bid - advance turn to opponent
            return self.state.step()
    
    def _resolve_round(self):
        """Resolve the current round after both players have bid."""
        bids = self.state.game_state["pending_bids"]
        bid_0 = bids[0]
        bid_1 = bids[1]
        current_round = self.state.game_state["current_round"]
        
        # Deduct bids from budgets
        self.state.game_state["budgets"][0] -= bid_0
        self.state.game_state["budgets"][1] -= bid_1
        
        # Determine winner and award points
        points_at_stake = current_round
        
        if bid_0 > bid_1:
            winner = 0
            self.state.game_state["points"][0] += points_at_stake
            result_msg = f"Player 0 wins Round {current_round}! (Bid ${bid_0} vs ${bid_1}) +{points_at_stake} point(s)"
        elif bid_1 > bid_0:
            winner = 1
            self.state.game_state["points"][1] += points_at_stake
            result_msg = f"Player 1 wins Round {current_round}! (Bid ${bid_1} vs ${bid_0}) +{points_at_stake} point(s)"
        else:
            winner = None
            result_msg = f"Round {current_round} is a TIE! (Both bid ${bid_0}) No points awarded"
        
        # Broadcast result
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=result_msg
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Record history
        self.state.game_state["round_history"].append((current_round, bid_0, bid_1, winner))
        
        # Clear pending bids
        self.state.game_state["pending_bids"] = {}
        
        # Move to next round
        self.state.game_state["current_round"] += 1
    
    def _determine_final_winner(self):
        """Determine the final winner after all rounds."""
        points_0 = self.state.game_state["points"][0]
        points_1 = self.state.game_state["points"][1]
        budget_0 = self.state.game_state["budgets"][0]
        budget_1 = self.state.game_state["budgets"][1]
        
        # Show final summary
        summary = (
            f"\n{'='*60}\n"
            f"GAME OVER - FINAL RESULTS\n"
            f"{'='*60}\n\n"
            f"Player 0: {points_0} points, ${int(budget_0)} remaining\n"
            f"Player 1: {points_1} points, ${int(budget_1)} remaining\n\n"
        )
        
        if points_0 > points_1:
            summary += f"Player 0 WINS with {points_0} points!"
            self.state.set_winner(
                player_id=0,
                reason=f"Player 0 wins with {points_0} points vs Player 1's {points_1} points!"
            )
        elif points_1 > points_0:
            summary += f"Player 1 WINS with {points_1} points!"
            self.state.set_winner(
                player_id=1,
                reason=f"Player 1 wins with {points_1} points vs Player 0's {points_0} points!"
            )
        else:
            # Tied on points - check budget
            if budget_0 > budget_1:
                summary += f"Tied on points! Player 0 WINS with more money remaining (${int(budget_0)} vs ${int(budget_1)})"
                self.state.set_winner(
                    player_id=0,
                    reason=f"Tied at {points_0} points. Player 0 wins with ${int(budget_0)} remaining vs ${int(budget_1)}!"
                )
            elif budget_1 > budget_0:
                summary += f"Tied on points! Player 1 WINS with more money remaining (${int(budget_1)} vs ${int(budget_0)})"
                self.state.set_winner(
                    player_id=1,
                    reason=f"Tied at {points_0} points. Player 1 wins with ${int(budget_1)} remaining vs ${int(budget_0)}!"
                )
            else:
                summary += f"Perfect TIE! Both have {points_0} points and ${int(budget_0)} remaining!"
                self.state.set_draw(
                    reason=f"Perfect draw: both players have {points_0} points and ${int(budget_0)} remaining."
                )
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=summary
        ,
            observation_type=ObservationType.GAME_MESSAGE)
    
    def _format_game_state(self) -> str:
        """Format current game state."""
        current_round = self.state.game_state["current_round"]
        budgets = self.state.game_state["budgets"]
        points = self.state.game_state["points"]
        history = self.state.game_state["round_history"]
        
        msg = f"\n{'='*60}\n"
        msg += f"AUCTION GAME - Round {current_round}/{self.num_rounds}\n"
        msg += f"{'='*60}\n\n"
        
        msg += f"CURRENT STANDINGS:\n"
        msg += f"  Player 0: {points[0]} points, ${int(budgets[0])} remaining\n"
        msg += f"  Player 1: {points[1]} points, ${int(budgets[1])} remaining\n\n"
        
        msg += f"THIS ROUND: Worth {current_round} point(s)\n\n"
        
        # Show recent history
        if history:
            msg += f"RECENT HISTORY (last 5 rounds):\n"
            for round_num, bid_0, bid_1, winner in history[-5:]:
                if winner == 0:
                    result = f"P0 wins (${int(bid_0)} > ${int(bid_1)})"
                elif winner == 1:
                    result = f"P1 wins (${int(bid_1)} > ${int(bid_0)})"
                else:
                    result = f"TIE (${int(bid_0)} = ${int(bid_1)})"
                msg += f"  Round {round_num} ({round_num}pts): {result}\n"
            msg += "\n"
        
        msg += f"Submit your bid: [amount]\n"
        
        return msg

