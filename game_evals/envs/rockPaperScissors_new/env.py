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

"""Rock Paper Scissors game environment for SPIRAL."""

import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType


class RockPaperScissorsEnv(ta.Env):
    """
    Rock Paper Scissors game environment with best-of-X format.
    
    Players compete in multiple rounds, with the first to win the majority
    winning the overall game. Moves are SIMULTANEOUS - both players choose
    without seeing the opponent's choice, then both choices are revealed.
    Full history of all rounds is included in observations to enable 
    strategic adaptation and pattern recognition.
    
    Actions: [Rock], [Paper], [Scissors]
    
    Rules:
    - Rock beats Scissors
    - Scissors beats Paper
    - Paper beats Rock
    - Same choice = Draw (round is replayed)
    - Moves are simultaneous (no player sees the other's choice first)
    """
    
    def __init__(self, best_of: int = 5):
        """
        Initialize Rock Paper Scissors environment.
        
        Args:
            best_of: Number of rounds to play (first to win majority wins)
                    For example, best_of=5 means first to win 3 rounds wins
        """
        super().__init__()
        self.best_of = best_of
        self.rounds_to_win = (best_of // 2) + 1  # Ceiling division
        
        # Valid choices
        self.choices = ["Rock", "Paper", "Scissors"]
        
        # Regex to match [Rock], [Paper], or [Scissors]
        self.action_pattern = re.compile(
            r"\[(Rock|Paper|Scissors)\]", 
            re.IGNORECASE
        )
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.best_of * 2,  # Max turns (in case of many draws)
        )
        
        # Initialize game state
        game_state = {
            "scores": {0: 0, 1: 0},  # Rounds won by each player
            "round_history": [],  # List of (p0_choice, p1_choice, winner) tuples
            "current_round": 0,
            "pending_choices": {},  # Store choices as players submit them
        }
        
        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Send initial game info to both players
        initial_msg = self._format_game_state()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=initial_msg
        ,
            observation_type=ObservationType.GAME_MESSAGE)
    
    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate initial instructions for a player."""
        prompt = (
            f"You are Player {player_id} in a Rock Paper Scissors game.\n\n"
            f"═══════════════════════════════════════\n"
            f"IMPORTANT: MOVES ARE SIMULTANEOUS\n"
            f"═══════════════════════════════════════\n"
            f"Both players choose at the same time without seeing the opponent's choice.\n"
            f"Choices are revealed only after both players have submitted.\n\n"
            f"GAME FORMAT:\n"
            f"- Best of {self.best_of} rounds\n"
            f"- First to win {self.rounds_to_win} rounds wins the game\n"
            f"- If both players choose the same, the round is a draw and replayed\n\n"
            f"GAME RULES:\n"
            f"- Rock beats Scissors\n"
            f"- Scissors beats Paper\n"
            f"- Paper beats Rock\n\n"
            f"ACTION FORMAT:\n"
            f"- To make your choice, write [Rock], [Paper], or [Scissors]\n"
            f"- Examples: [Rock], [Paper], [Scissors]\n\n"
            f"STRATEGY TIPS:\n"
            f"- Pay attention to your opponent's patterns from previous rounds\n"
            f"- Mix up your choices to be unpredictable\n"
            f"- Adapt based on game history\n"
            f"- All previous rounds are shown in your observation\n"
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
            List of valid action strings: ["[Rock]", "[Paper]", "[Scissors]"]
        """
        return ["[Rock]", "[Paper]", "[Scissors]"]
    
    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Process an action and update game state."""
        player_id = self.state.current_player_id
        
        # Parse the choice
        match = self.action_pattern.search(action)
        if not match:
            self.state.set_invalid_move(reason="Action must be [Rock], [Paper], or [Scissors]."
            )
            return self.state.step()
        
        choice = match.group(1).capitalize()  # Normalize to "Rock", "Paper", "Scissors"
        
        # Store this player's choice (PRIVATE - not shown to opponent yet)
        self.state.game_state["pending_choices"][player_id] = choice
        
        # Send confirmation ONLY to this player (not broadcast)
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=player_id,
            message=f"You chose {choice}. Waiting for opponent..."
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Check if both players have submitted choices
        if len(self.state.game_state["pending_choices"]) == 2:
            # Both players have chosen - NOW reveal and resolve the round
            self._resolve_round()
            
            # Check if game is over
            if not self.state.done:
                # Show updated game state to both players
                game_state_msg = self._format_game_state()
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=game_state_msg
                ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        return self.state.step()
    
    def _resolve_round(self):
        """Resolve the current round and update scores."""
        choices = self.state.game_state["pending_choices"]
        choice_0 = choices[0]
        choice_1 = choices[1]
        
        # Determine winner
        winner = self._determine_winner(choice_0, choice_1)
        
        # Update game state
        self.state.game_state["current_round"] += 1
        self.state.game_state["round_history"].append((choice_0, choice_1, winner))
        
        # Safety check: prevent infinite loops from too many rounds
        if self.state.game_state["current_round"] > 500:
            raise RuntimeError(
                f"Rock Paper Scissors game exceeded 500 rounds! "
                f"Current scores: P0={self.state.game_state['scores'][0]}, "
                f"P1={self.state.game_state['scores'][1]}. "
                f"This likely indicates an issue with the game logic or too many draws."
            )
        
        # Announce round result
        if winner is None:
            # Draw
            result_msg = (
                f"Round {self.state.game_state['current_round']}: "
                f"Player 0 chose {choice_0}, Player 1 chose {choice_1} - DRAW! "
                f"This round will be replayed."
            )
        else:
            # Someone won
            self.state.game_state["scores"][winner] += 1
            loser = 1 - winner
            result_msg = (
                f"Round {self.state.game_state['current_round']}: "
                f"Player 0 chose {choice_0}, Player 1 chose {choice_1} - "
                f"Player {winner} wins! ({choice_0 if winner == 0 else choice_1} beats "
                f"{choice_1 if winner == 0 else choice_0})"
            )
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=result_msg
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Check if someone has won the game
        if winner is not None:
            if self.state.game_state["scores"][winner] >= self.rounds_to_win:
                # Game over!
                final_score = f"{self.state.game_state['scores'][0]}-{self.state.game_state['scores'][1]}"
                self.state.set_winner(
                    player_id=winner,
                    reason=f"Player {winner} won {self.rounds_to_win} rounds first! Final score: {final_score}"
                )
        
        # Clear pending choices for next round
        self.state.game_state["pending_choices"] = {}
    
    def _determine_winner(self, choice_0: str, choice_1: str) -> Optional[int]:
        """
        Determine the winner of a round.
        
        Returns:
            0 if player 0 wins, 1 if player 1 wins, None if draw
        """
        if choice_0 == choice_1:
            return None  # Draw
        
        # Define winning combinations
        wins = {
            ("Rock", "Scissors"): 0,
            ("Scissors", "Paper"): 0,
            ("Paper", "Rock"): 0,
            ("Scissors", "Rock"): 1,
            ("Paper", "Scissors"): 1,
            ("Rock", "Paper"): 1,
        }
        
        return wins.get((choice_0, choice_1))
    
    def _format_game_state(self) -> str:
        """Format current game state with full history."""
        scores = self.state.game_state["scores"]
        history = self.state.game_state["round_history"]
        
        msg = f"{'='*60}\n"
        msg += f"ROCK PAPER SCISSORS - Best of {self.best_of}\n"
        msg += f"{'='*60}\n\n"
        
        msg += f"CURRENT SCORE: Player 0 has {scores[0]} | Player 1 has {scores[1]}\n"
        msg += f"(First to {self.rounds_to_win} round wins)\n\n"
        
        # Show complete game history
        msg += f"{'='*60}\n"
        msg += f"COMPLETE GAME HISTORY\n"
        msg += f"{'='*60}\n"
        
        if history:
            for i, (choice_0, choice_1, winner) in enumerate(history, 1):
                if winner is None:
                    result = "DRAW (replayed)"
                elif winner == 0:
                    result = f"Player 0 WINS ({choice_0} beats {choice_1})"
                else:
                    result = f"Player 1 WINS ({choice_1} beats {choice_0})"
                
                msg += f"Round {i:2d}: [P0: {choice_0:8s}] vs [P1: {choice_1:8s}] → {result}\n"
            
            # Add summary stats
            msg += f"\nSUMMARY: {scores[0]} wins for P0, {scores[1]} wins for P1, "
            draws = sum(1 for _, _, w in history if w is None)
            msg += f"{draws} draw(s)\n"
        else:
            msg += "No rounds played yet.\n"
        
        msg += f"{'='*60}\n\n"
        msg += "Choose your move: [Rock], [Paper], or [Scissors]\n"
        
        return msg
    
    def _analyze_patterns(self) -> str:
        """Analyze patterns in game history (visible to both players)."""
        history = self.state.game_state["round_history"]
        
        # Count choices for each player
        p0_choices = {"Rock": 0, "Paper": 0, "Scissors": 0}
        p1_choices = {"Rock": 0, "Paper": 0, "Scissors": 0}
        
        for choice_0, choice_1, _ in history:
            p0_choices[choice_0] += 1
            p1_choices[choice_1] += 1
        
        msg = "PATTERN ANALYSIS:\n"
        msg += f"  Player 0 frequency: Rock={p0_choices['Rock']}, Paper={p0_choices['Paper']}, Scissors={p0_choices['Scissors']}\n"
        msg += f"  Player 1 frequency: Rock={p1_choices['Rock']}, Paper={p1_choices['Paper']}, Scissors={p1_choices['Scissors']}\n"
        
        # Show last 3 moves for each player
        if len(history) >= 3:
            last_3_p0 = [h[0] for h in history[-3:]]
            last_3_p1 = [h[1] for h in history[-3:]]
            msg += f"  Player 0 last 3: {', '.join(last_3_p0)}\n"
            msg += f"  Player 1 last 3: {', '.join(last_3_p1)}\n"
        
        msg += "\n"
        return msg

