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

"""Chomp game environment for SPIRAL."""

import random
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import textarena as ta
from textarena.core import ObservationType


class ChompEnv(ta.Env):
    """
    Chomp game environment.
    
    Classic combinatorial game on a rectangular chocolate bar grid.
    Players take turns "chomping" (removing) squares.
    When you chomp a square, you also remove all squares above and to the right.
    The bottom-left square (0,0) is POISONED - whoever is forced to eat it loses!
    
    This is a solved game with optimal strategy, demonstrating:
    - Perfect information
    - Deterministic outcomes
    - Strategic complexity from simple rules
    - Nim-like structure in endgame
    """
    
    def __init__(self, rows: int = 5, cols: int = 6):
        """
        Initialize Chomp environment.
        
        Args:
            rows: Number of rows in the chocolate bar (default: 5)
            cols: Number of columns in the chocolate bar (default: 6)
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        
        # Action pattern: [row,col] to chomp a square
        self.action_pattern = re.compile(r"\[(\d+)\s*,\s*(\d+)\]")
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            random.seed(seed)
        
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.rows * self.cols,  # Max possible moves
        )
        
        # Initialize game state
        # Grid: True = square exists, False = square eaten
        game_state = {
            "grid": [[True for _ in range(self.cols)] for _ in range(self.rows)],
            "move_count": 0,
        }
        
        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Show initial board to Player 0
        board_msg = self._format_game_state()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=0,
            message=board_msg,
            observation_type=ObservationType.GAME_BOARD
        )
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate initial instructions for a player."""
        prompt = (
            f"You are Player {player_id} in a Chomp game.\n\n"
            f"═══════════════════════════════════════\n"
            f"GAME SETUP:\n"
            f"═══════════════════════════════════════\n"
            f"- Chocolate bar: {self.rows}×{self.cols} grid of squares\n"
            f"- Coordinates: rows 0-{self.rows-1} (bottom to top), cols 0-{self.cols-1} (left to right)\n"
            f"- ☠️  POISONED SQUARE: Bottom-left corner at (row 0, col 0)\n"
            f"- Goal: Force your opponent to eat the poisoned square\n\n"
            f"GAME RULES:\n\n"
            f"1. Players take turns chomping (eating) a square\n"
            f"2. When you chomp a square at [row,col]:\n"
            f"   - That square is removed\n"
            f"   - ALL squares above it (higher row numbers) are removed\n"
            f"   - ALL squares to the right of it (higher column numbers) are removed\n"
            f"   - You remove the entire 'upper-right' rectangular region\n"
            f"3. ☠️  The poisoned square (0,0) can NEVER be chomped directly\n"
            f"4. The player forced to take the last remaining square (the poisoned one) LOSES\n\n"
            f"ACTION FORMAT:\n\n"
            f"⚠️  YOU MUST RESPOND WITH EXACTLY THIS FORMAT: [row,col]\n\n"
            f"⚠️  COORDINATE FORMAT: [ROW, COLUMN]\n"
            f"   - FIRST number = ROW (vertical position)\n"
            f"     • Row 0 = BOTTOM row (where poisoned square is)\n"
            f"     • Row {self.rows-1} = TOP row\n"
            f"     • Rows go from 0 (bottom) to {self.rows-1} (top)\n"
            f"   - SECOND number = COLUMN (horizontal position)\n"
            f"     • Column 0 = LEFT column (where poisoned square is)\n"
            f"     • Column {self.cols-1} = RIGHT column\n"
            f"     • Columns go from 0 (left) to {self.cols-1} (right)\n\n"
            f"⚠️  IMPORTANT RULES:\n"
            f"   - You CANNOT chomp the poisoned square [0,0] directly\n"
            f"   - You CANNOT chomp a square that has already been eaten (shown as ·)\n"
            f"   - You can ONLY chomp available squares (shown as ■)\n"
            f"   - Chomping removes the square AND all squares above and to the right\n"
            f"   - Invalid moves cause you to LOSE immediately\n\n"
            f"EXAMPLES WITH DETAILED EXPLANATIONS:\n\n"
            f"  [{self.rows-1},{self.cols-1}] = Chomp top-right corner\n"
            f"                     • Row {self.rows-1} (top), Column {self.cols-1} (right)\n"
            f"                     • Safest move - removes only that one square\n"
            f"                     • Nothing above or to the right of it\n\n"
            f"  [2,3] = Chomp square at row 2, column 3\n"
            f"          • Removes (2,3) and everything above and right:\n"
            f"          • Rows 2, 3, 4, ... up to {self.rows-1}\n"
            f"          • Columns 3, 4, 5, ... up to {self.cols-1}\n"
            f"          • Creates an L-shaped remaining board\n\n"
            f"  [0,1] = Chomp square at row 0 (bottom), column 1\n"
            f"          • Removes entire bottom row EXCEPT [0,0] (poisoned square)\n"
            f"          • Also removes everything above in columns 1 and higher\n"
            f"          • Dangerous - leaves opponent in strong position\n\n"
            f"  [1,0] = Chomp square at row 1, column 0 (left)\n"
            f"          • Removes entire left column EXCEPT [0,0] (poisoned square)\n"
            f"          • Also removes everything to the right in rows 1 and higher\n\n"
            f"BOARD DISPLAY:\n\n"
            f"- ☠️  = Poisoned square (row 0, col 0)\n"
            f"- ■ = Available chocolate square\n"
            f"- · = Eaten square (removed)\n\n"
            f"STRATEGY TIPS:\n\n"
            f"- This is a solved game - first player can always win with perfect play\n"
            f"- Think about symmetry and mirror moves\n"
            f"- Avoid leaving your opponent with only the poisoned square\n"
            f"- Consider the shape you leave behind\n"
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
            List of valid action strings: ["[row,col]", ...] for all board positions
        """
        actions = []
        for row in range(self.rows):
            for col in range(self.cols):
                actions.append(f"[{row},{col}]")
        return actions
    
    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Process an action and update game state."""
        player_id = self.state.current_player_id
        opponent_id = 1 - player_id
        
        # Log the raw action
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Parse the chomp coordinates
        match = self.action_pattern.search(action)
        if not match:
            self.state.set_invalid_move(reason=(
                    f"⚠️  INVALID FORMAT! You must use [row,col] where:\n\n"
                    f"  REQUIRED FORMAT: [ROW, COLUMN]\n"
                    f"  • FIRST number = ROW (0 = bottom, {self.rows-1} = top)\n"
                    f"  • SECOND number = COLUMN (0 = left, {self.cols-1} = right)\n\n"
                    f"  Valid range:\n"
                    f"  • row: 0 to {self.rows-1}\n"
                    f"  • col: 0 to {self.cols-1}\n\n"
                    f"  Examples:\n"
                    f"  [{self.rows-1},{self.cols-1}] = top-right corner\n"
                    f"  [1,1] = row 1, column 1\n"
                    f"  [0,2] = bottom row, column 2\n\n"
                    f"  Your response must be EXACTLY in this format with brackets and comma!"
                )
            )
            return self.state.step()
        
        try:
            row = int(match.group(1))
            col = int(match.group(2))
        except ValueError:
            self.state.set_invalid_move(reason="Invalid coordinates."
            )
            return self.state.step()
        
        # Validate coordinates are in bounds
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self.state.set_invalid_move(reason=f"Coordinates out of bounds. Must be 0-{self.rows-1} for rows, 0-{self.cols-1} for cols."
            )
            return self.state.step()
        
        # Cannot chomp the poisoned square directly
        if row == 0 and col == 0:
            self.state.set_invalid_move(reason=(
                    f"☠️  INVALID: Cannot chomp the poisoned square [0,0] directly!\n\n"
                    f"The poisoned square at [row 0, column 0] (bottom-left corner) is POISONED.\n"
                    f"It can only be eaten as the very last remaining square.\n"
                    f"You must choose a different square that is available (shown as ■ on the board).\n\n"
                    f"Look at the board and chomp any square EXCEPT [0,0]."
                )
            )
            return self.state.step()
        
        # Check if square exists (hasn't been eaten)
        if not self.state.game_state["grid"][row][col]:
            self.state.set_invalid_move(reason=(
                    f"⚠️  INVALID: Square at [row {row}, column {col}] has already been eaten!\n\n"
                    f"You tried to chomp [{row},{col}] but this square no longer exists.\n"
                    f"It was removed by a previous chomp.\n\n"
                    f"Look at the board:\n"
                    f"  ■ = Available squares (you CAN chomp these)\n"
                    f"  · = Eaten squares (you CANNOT chomp these)\n"
                    f"  ☠️  = Poisoned square (you CANNOT chomp this)\n\n"
                    f"Choose a square that is still available (shown as ■)."
                )
            )
            return self.state.step()
        
        # Valid move - chomp the square and everything above and to the right
        chomped_count = self._chomp_square(row, col)
        
        self.state.game_state["move_count"] += 1
        
        # Announce the chomp
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id} chomps [{row},{col}] - removed {chomped_count} square(s)",
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        # Check if only the poisoned square remains
        if self._only_poisoned_remains():
            # Opponent is forced to take the poisoned square and loses
            self.state.set_winner(
                player_id=player_id,
                reason=f"Player {player_id} wins! Only the poisoned square remains - Player {opponent_id} must eat it."
            )
            return self.state.step()
        
        # Show updated board to next player (don't log to history)
        board_msg = self._format_game_state()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=opponent_id,
            message=board_msg,
            observation_type=ObservationType.GAME_BOARD
        )
        
        return self.state.step()
    
    def _chomp_square(self, row: int, col: int) -> int:
        """
        Chomp a square and all squares above and to the right.
        
        Returns the number of squares removed.
        """
        count = 0
        for r in range(row, self.rows):
            for c in range(col, self.cols):
                if self.state.game_state["grid"][r][c]:
                    self.state.game_state["grid"][r][c] = False
                    count += 1
        return count
    
    def _only_poisoned_remains(self) -> bool:
        """Check if only the poisoned square (0,0) remains."""
        for r in range(self.rows):
            for c in range(self.cols):
                # If any square other than (0,0) exists, game continues
                if self.state.game_state["grid"][r][c] and not (r == 0 and c == 0):
                    return False
        return True
    
    def _format_game_state(self) -> str:
        """Format current game state showing the chocolate bar."""
        msg = f"\n{'='*60}\n"
        msg += f"CHOMP - {self.rows}×{self.cols} Chocolate Bar\n"
        msg += f"{'='*60}\n\n"
        
        msg += f"Moves made: {self.state.game_state['move_count']}\n\n"
        
        # Render the board
        msg += self._render_board()
        
        msg += f"\n☠️  = Poisoned square (0,0) - DO NOT CHOMP!\n"
        msg += f"■ = Available chocolate squares\n"
        msg += f"· = Eaten squares\n"
        
        return msg
    
    def _render_board(self) -> str:
        """Render the chocolate bar grid."""
        grid = self.state.game_state["grid"]
        
        lines = []
        
        # Header with column numbers
        header = "    "  # Offset for row labels
        for c in range(self.cols):
            header += f"{c:2} "
        lines.append(header)
        lines.append("   " + "─" * (self.cols * 3))
        
        # Render from top to bottom (highest row first for visual clarity)
        for r in range(self.rows - 1, -1, -1):
            row_line = f"{r:2} │"
            for c in range(self.cols):
                if r == 0 and c == 0:
                    # Poisoned square
                    if grid[r][c]:
                        row_line += " ☠️"
                    else:
                        row_line += " ·"
                else:
                    # Regular square
                    if grid[r][c]:
                        row_line += " ■"
                    else:
                        row_line += " ·"
            lines.append(row_line)
        
        return "\n".join(lines) + "\n"

