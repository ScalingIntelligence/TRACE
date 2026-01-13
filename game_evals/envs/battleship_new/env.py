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

"""Battleship game environment for SPIRAL."""

import random
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import textarena as ta
from textarena.core import ObservationType


class BattleshipEnv(ta.Env):
    """
    Battleship game environment.
    
    Simplified naval battle game on a 5x5 grid where players try to sink
    opponent's hidden ship by guessing coordinates.
    
    Each player has ONE ship that is 4 squares long.
    
    Game phases:
    1. Placement phase: Players place their ship with [row,col,orientation]
       - row,col: top-left corner of the ship
       - orientation: 0 for horizontal (extends right), 1 for vertical (extends down)
       Example: [1,0,0] places horizontal ship at row 1, columns 0-3
       Example: [0,2,1] places vertical ship at column 2, rows 0-3
       
    2. Attack phase: Players take turns attacking coordinates [row,col]
       Example: [2,3] attacks row 2, column 3
    """
    
    def __init__(self, board_size: int = 5, ship_length: int = 4):
        """
        Initialize Battleship environment.
        
        Args:
            board_size: Size of the square board (default: 5 for 5x5)
            ship_length: Length of the ship (default: 4)
        """
        super().__init__()
        self.board_size = board_size
        self.ship_length = ship_length
        # Max moves per player = total squares on board
        self.max_moves_per_player = board_size * board_size
        
        # Regex patterns
        self.attack_pattern = re.compile(r"\[(\d+)\s*,\s*(\d+)\]")
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            random.seed(seed)
        
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.max_moves_per_player * 2 + 2,  # +2 for placement phase
        )
        
        # Initialize game state
        game_state = {
            "phase": "placement",  # "placement" or "attack"
            "ships_placed": {
                0: False,
                1: False,
            },
            "ships": {
                0: None,  # Set of (row, col) positions
                1: None,
            },
            "hits": {
                0: set(),  # Positions where player 0 has been hit
                1: set(),  # Positions where player 1 has been hit
            },
            "attacks": {
                0: set(),  # Positions player 0 has attacked
                1: set(),  # Positions player 1 has attacked
            },
            "attack_turn_count": 0,
            "moves_made": {
                0: 0,  # Number of attack moves made by player 0
                1: 0,  # Number of attack moves made by player 1
            },
        }
        
        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Ask Player 0 to place their ship
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=0,
            message=(
                "═══ SHIP PLACEMENT PHASE ═══\n\n"
                f"Place your ship ({self.ship_length} squares long) using format:\n"
                "  [row,col,orientation]\n\n"
                "Where:\n"
                f"  row, col: top-left corner (0-{self.board_size-1})\n"
                "  orientation: 0 for horizontal, 1 for vertical\n\n"
                "Examples:\n"
                f"  [1,0,0] = horizontal ship at row 1, columns 0-{self.ship_length-1}\n"
                f"  [0,2,1] = vertical ship at column 2, rows 0-{self.ship_length-1}"
            ),
            observation_type=ObservationType.GAME_MESSAGE
        )
    
    def _validate_ship_placement(
        self, row: int, col: int, orientation: int
    ) -> Tuple[bool, Optional[str], Optional[Set[Tuple[int, int]]]]:
        """
        Validate if a ship can be placed at the given position.
        
        Args:
            row: Starting row
            col: Starting column
            orientation: 0 for horizontal, 1 for vertical
        
        Returns:
            (is_valid, error_message, ship_positions)
        """
        positions = []
        
        for i in range(self.ship_length):
            if orientation == 0:  # Horizontal
                r, c = row, col + i
            else:  # Vertical
                r, c = row + i, col
            
            # Check bounds
            if r >= self.board_size or c >= self.board_size or r < 0 or c < 0:
                orient_name = "horizontal" if orientation == 0 else "vertical"
                if orientation == 0:
                    return False, f"{orient_name.capitalize()} ship would go out of bounds (needs columns {col} to {col + self.ship_length - 1}, but grid is only 0-{self.board_size-1})", None
                else:
                    return False, f"{orient_name.capitalize()} ship would go out of bounds (needs rows {row} to {row + self.ship_length - 1}, but grid is only 0-{self.board_size-1})", None
            
            positions.append((r, c))
        
        return True, None, set(positions)
    
    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate initial instructions for a player."""
        prompt = (
            f"You are Player {player_id} in a Battleship game.\n\n"
            f"═══════════════════════════════════════\n"
            f"GAME SETUP:\n"
            f"═══════════════════════════════════════\n"
            f"- Board size: {self.board_size}x{self.board_size} grid\n"
            f"- Each player has ONE ship that is {self.ship_length} squares long\n"
            f"- Coordinates: rows and columns are numbered 0-{self.board_size-1}\n\n"
            f"GAME PHASES:\n\n"
            f"1. PLACEMENT PHASE:\n"
            f"   - Place your ship with: [row,col,orientation]\n"
            f"   - row,col: top-left corner position\n"
            f"   - orientation: 0 for horizontal, 1 for vertical\n"
            f"   - Example: [1,0,0] = horizontal ship at row 1, columns 0-{self.ship_length-1}\n"
            f"   - Example: [0,2,1] = vertical ship at column 2, rows 0-{self.ship_length-1}\n\n"
            f"2. ATTACK PHASE:\n"
            f"   - Players take turns attacking coordinates\n"
            f"   - Format: [row,col]\n"
            f"   - Example: [2,3] attacks row 2, column 3\n"
            f"   - You'll see: HIT (X) or MISS (M)\n"
            f"   - First to sink opponent's ship wins!\n"
            f"   - Max moves: {self.max_moves_per_player} per player\n"
            f"   - If max moves reached, player with most hits wins (tie = draw)\n\n"
            f"MAP LEGEND:\n"
            f"- YOUR ATTACK MAP (targeting opponent):\n"
            f"  . = Not yet attacked\n"
            f"  M = Miss\n"
            f"  X = Hit\n\n"
            f"- YOUR DEFENSE MAP (your board):\n"
            f"  . = Empty water\n"
            f"  S = Your ship\n"
            f"  M = Opponent missed\n"
            f"  X = Opponent hit your ship\n"
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
            List of valid action strings depending on game phase:
            - Placement phase: ["[row,col,orientation]", ...]
            - Attack phase: ["[row,col]", ...]
        """
        if self.state.game_state["phase"] == "placement":
            # Return ALL valid ship placements
            # Format: [row,col,orientation] where orientation is 0 (horizontal) or 1 (vertical)
            actions = []
            for row in range(self.board_size):
                for col in range(self.board_size):
                    for orient in [0, 1]:
                        actions.append(f"[{row},{col},{orient}]")
            return actions
        else:
            # Attack phase - return all grid positions
            actions = []
            for row in range(self.board_size):
                for col in range(self.board_size):
                    actions.append(f"[{row},{col}]")
            return actions
    
    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Process an action and update game state."""
        player_id = self.state.current_player_id
        
        # Log the raw action
        # During placement phase, keep ship positions private
        # During attack phase, broadcast attacks to both players
        if self.state.game_state["phase"] == "placement":
            # Ship placement is PRIVATE - only the placing player should see it
            self.state.add_observation(
                from_id=player_id,
                to_id=player_id,
                message=action
            ,
            observation_type=ObservationType.GAME_MESSAGE)
        else:
            # Attacks are PUBLIC - both players can see them
            self.state.add_observation(
                from_id=player_id,
                to_id=-1,
                message=action
            ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Check which phase we're in
        if self.state.game_state["phase"] == "placement":
            return self._handle_placement(player_id, action)
        else:  # attack phase
            return self._handle_attack(player_id, action)
    
    def _handle_placement(self, player_id: int, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Handle ship placement actions in format [row,col,orientation]."""
        # Parse placement: [row,col,orientation]
        placement_pattern = re.compile(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]")
        match = placement_pattern.search(action)
        
        if not match:
            self.state.set_invalid_move(reason=(
                    f"Invalid placement format. Use [row,col,orientation] where:\n"
                    f"  row, col: 0-{self.board_size-1}\n"
                    f"  orientation: 0 for horizontal, 1 for vertical\n"
                    f"Example: [1,0,0] for horizontal ship at row 1, starting at column 0\n"
                    f"Example: [0,2,1] for vertical ship at column 2, starting at row 0"
                )
            )
            return self.state.step()
        
        try:
            row = int(match.group(1))
            col = int(match.group(2))
            orientation = int(match.group(3))
        except ValueError:
            self.state.set_invalid_move(reason="Invalid coordinates."
            )
            return self.state.step()
        
        # Validate orientation
        if orientation not in [0, 1]:
            self.state.set_invalid_move(reason="Orientation must be 0 (horizontal) or 1 for vertical)."
            )
            return self.state.step()
        
        # Validate position
        is_valid, error, positions = self._validate_ship_placement(row, col, orientation)
        
        if not is_valid:
            self.state.set_invalid_move(reason=error
            )
            return self.state.step()
        
        # Place the ship
        self.state.game_state["ships"][player_id] = positions
        self.state.game_state["ships_placed"][player_id] = True
        
        orient_str = "horizontal" if orientation == 0 else "vertical"
        if orientation == 0:
            coverage = f"columns {col}-{col + self.ship_length - 1}"
        else:
            coverage = f"rows {row}-{row + self.ship_length - 1}"
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=player_id,
            message=f"Ship placed successfully! {orient_str.capitalize()} at [{row},{col}], covering {coverage}",
            observation_type=ObservationType.GAME_MESSAGE,
        )
        
        # Check if both players have placed ships
        if all(self.state.game_state["ships_placed"][p] for p in [0, 1]):
            # Start attack phase
            self.state.game_state["phase"] = "attack"
            
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message="\n═══ ATTACK PHASE BEGINS ═══\n"
            ,
            observation_type=ObservationType.GAME_MESSAGE)
            
            # Show initial board to Player 0 (don't log to history)
            board_msg = self._format_game_state_for_player(0)
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=0,
                message=board_msg,
                observation_type=ObservationType.GAME_BOARD
            )
        else:
            # Ask next player to place ship
            next_player = 1 - player_id
            if not self.state.game_state["ships_placed"][next_player]:
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=next_player,
                    message=(
                        "═══ SHIP PLACEMENT PHASE ═══\n\n"
                        f"Place your ship ({self.ship_length} squares long) using format:\n"
                        "  [row,col,orientation]\n\n"
                        "Where:\n"
                        "  orientation: 0 = horizontal, 1 = vertical\n"
                        f"  row, col: top-left corner (0-{self.board_size-1})\n\n"
                        "Examples:\n"
                        f"  [1,0,0] = horizontal ship at row 1, columns 0-{self.ship_length-1}\n"
                        f"  [0,2,1] = vertical ship at column 2, rows 0-{self.ship_length-1}"
                    ),
                    observation_type=ObservationType.GAME_MESSAGE
                )
        
        return self.state.step()
    
    def _handle_attack(self, player_id: int, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Handle attack actions."""
        opponent_id = 1 - player_id
        
        # Parse the attack coordinates
        match = self.attack_pattern.search(action)
        if not match:
            self.state.set_invalid_move(reason=f"Action must be in format [row,col] where row and col are 0-{self.board_size-1}."
            )
            return self.state.step()
        
        try:
            row = int(match.group(1))
            col = int(match.group(2))
        except ValueError:
            self.state.set_invalid_move(reason="Invalid coordinates."
            )
            return self.state.step()
        
        # Validate coordinates are in range
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            self.state.set_invalid_move(reason=f"Coordinates must be between 0 and {self.board_size-1}."
            )
            return self.state.step()
        
        # Record the attack (allow re-attacking same position)
        self.state.game_state["attacks"][player_id].add((row, col))
        self.state.game_state["attack_turn_count"] += 1
        self.state.game_state["moves_made"][player_id] += 1
        
        # Check if it's a hit
        opponent_ship = self.state.game_state["ships"][opponent_id]
        is_hit = (row, col) in opponent_ship
        
        if is_hit:
            # It's a hit!
            self.state.game_state["hits"][opponent_id].add((row, col))
            
            # Check if ship is sunk (all positions hit)
            if opponent_ship.issubset(self.state.game_state["hits"][opponent_id]):
                # Ship is sunk!
                message = f"Player {player_id} attacks [{row},{col}] - HIT! Ship is SUNK!"
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=message
                ,
            observation_type=ObservationType.GAME_MESSAGE)
                
                # Game over!
                self.state.set_winner(
                    player_id=player_id,
                    reason=f"Player {player_id} sunk Player {opponent_id}'s ship!"
                )
                return self.state.step()
            else:
                # Hit but not sunk
                message = f"Player {player_id} attacks [{row},{col}] - HIT!"
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=message
                ,
            observation_type=ObservationType.GAME_MESSAGE)
        else:
            # It's a miss
            message = f"Player {player_id} attacks [{row},{col}] - MISS"
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=message
            ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Check if both players have reached max moves
        if (self.state.game_state["moves_made"][0] >= self.max_moves_per_player and 
            self.state.game_state["moves_made"][1] >= self.max_moves_per_player):
            # Game ends - determine winner by hit count
            hits_by_player_0 = len(self.state.game_state["hits"][1])  # Hits on player 1's ship
            hits_by_player_1 = len(self.state.game_state["hits"][0])  # Hits on player 0's ship
            
            if hits_by_player_0 > hits_by_player_1:
                # Player 0 wins
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Max moves reached! Player 0 landed {hits_by_player_0} hits vs Player 1's {hits_by_player_1} hits."
                ,
            observation_type=ObservationType.GAME_MESSAGE)
                self.state.set_winner(
                    player_id=0,
                    reason=f"Player 0 wins with {hits_by_player_0} hits!"
                )
            elif hits_by_player_1 > hits_by_player_0:
                # Player 1 wins
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Max moves reached! Player 1 landed {hits_by_player_1} hits vs Player 0's {hits_by_player_0} hits."
                ,
            observation_type=ObservationType.GAME_MESSAGE)
                self.state.set_winner(
                    player_id=1,
                    reason=f"Player 1 wins with {hits_by_player_1} hits!"
                )
            else:
                # Tie
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Max moves reached! Both players landed {hits_by_player_0} hits. It's a draw!"
                ,
            observation_type=ObservationType.GAME_MESSAGE)
                self.state.set_draw(
                    reason="Draw - both players landed the same number of hits."
                )
            
            return self.state.step()
        
        # Show updated board state to next player (don't log to history)
        next_player_state = self._format_game_state_for_player(opponent_id)
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=opponent_id,
            message=next_player_state,
            observation_type=ObservationType.GAME_BOARD
        )
        
        return self.state.step()
    
    def _format_game_state_for_player(self, player_id: int) -> str:
        """Format current game state for a player showing both maps."""
        opponent_id = 1 - player_id
        
        # Game status
        msg = f"\n{'='*50}\n"
        msg += f"ATTACK TURN {self.state.game_state['attack_turn_count'] + 1}\n"
        msg += f"{'='*50}\n\n"
        
        # Calculate hits and moves
        my_ship = self.state.game_state["ships"][player_id]
        opp_ship = self.state.game_state["ships"][opponent_id]
        my_hits_received = len(self.state.game_state["hits"][player_id])
        opp_hits_received = len(self.state.game_state["hits"][opponent_id])
        my_moves_made = self.state.game_state["moves_made"][player_id]
        my_moves_remaining = self.max_moves_per_player - my_moves_made
        
        msg += f"YOUR SHIP: {self.ship_length - my_hits_received}/{self.ship_length} squares remaining\n"
        msg += f"OPPONENT SHIP: {self.ship_length - opp_hits_received}/{self.ship_length} squares remaining\n"
        msg += f"YOUR MOVES: {my_moves_made}/{self.max_moves_per_player} used ({my_moves_remaining} remaining)\n\n"
        
        # Render attack map (where you're attacking opponent)
        msg += self._render_attack_map(player_id)
        msg += "\n\n"
        
        # Render defense map (where opponent is attacking you)
        msg += self._render_defense_map(player_id)
        
        return msg
    
    def _render_attack_map(self, player_id: int) -> str:
        """
        Render attack map showing where player has attacked opponent.
        This is where the player is GUESSING.
        . = Not attacked
        M = Miss
        X = Hit
        """
        opponent_id = 1 - player_id
        
        lines = []
        lines.append("┌─────────────────────────────────────┐")
        lines.append("│  YOUR ATTACK MAP (opponent's grid) │")
        lines.append("│  Where YOU are attacking           │")
        lines.append("└─────────────────────────────────────┘")
        lines.append("")
        lines.append("   " + " ".join([str(i) for i in range(self.board_size)]))
        lines.append("  " + "─" * (self.board_size * 2 - 1))
        
        for row in range(self.board_size):
            line = f"{row}│ "
            for col in range(self.board_size):
                if (row, col) in self.state.game_state["attacks"][player_id]:
                    if (row, col) in self.state.game_state["hits"][opponent_id]:
                        line += "X "  # Hit
                    else:
                        line += "M "  # Miss
                else:
                    line += ". "  # Not attacked
            lines.append(line)
        
        lines.append("")
        lines.append("Legend: . = not attacked, M = miss, X = hit")
        
        return "\n".join(lines)
    
    def _render_defense_map(self, player_id: int) -> str:
        """
        Render defense map showing player's own ship and opponent's attacks.
        This is YOUR board showing YOUR ship.
        . = Empty water
        S = Your ship
        M = Opponent missed
        X = Opponent hit your ship
        """
        opponent_id = 1 - player_id
        my_ship = self.state.game_state["ships"][player_id]
        
        lines = []
        lines.append("┌─────────────────────────────────────┐")
        lines.append("│  YOUR DEFENSE MAP (your grid)      │")
        lines.append("│  Where OPPONENT is attacking       │")
        lines.append("└─────────────────────────────────────┘")
        lines.append("")
        lines.append("   " + " ".join([str(i) for i in range(self.board_size)]))
        lines.append("  " + "─" * (self.board_size * 2 - 1))
        
        for row in range(self.board_size):
            line = f"{row}│ "
            for col in range(self.board_size):
                is_my_ship = (row, col) in my_ship
                opp_attacked = (row, col) in self.state.game_state["attacks"][opponent_id]
                opp_hit = (row, col) in self.state.game_state["hits"][player_id]
                
                if opp_hit:
                    line += "X "  # Opponent hit your ship
                elif opp_attacked:
                    line += "M "  # Opponent missed
                elif is_my_ship:
                    line += "S "  # Your ship (not hit yet)
                else:
                    line += ". "  # Empty water
            lines.append(line)
        
        lines.append("")
        lines.append("Legend: . = water, S = your ship, M = opponent miss, X = opponent hit")
        
        return "\n".join(lines)

