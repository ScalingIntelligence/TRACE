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

"""Connect 4 game environment for SPIRAL."""

import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType


class Connect4Env(ta.Env):
    """
    Connect 4 game environment.
    
    Standard 6x7 board where players try to get 4 pieces in a row.
    Actions are column numbers (0-6) where pieces drop to the lowest position.
    """
    
    def __init__(self, rows: int = 6, cols: int = 7, max_turns: int = 42):
        """
        Initialize Connect 4 environment.
        
        Args:
            rows: Number of rows (default: 6)
            cols: Number of columns (default: 7)
            max_turns: Maximum turns before draw (default: 42, full board)
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.max_turns = max_turns
        
        # Regex to match [0], [1], ..., [6] format
        self.action_pattern = re.compile(r"\[(\d+)\]")
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.max_turns,
        )
        
        # Initialize game state
        game_state = {
            "board": [[-1 for _ in range(self.cols)] for _ in range(self.rows)],
            "move_count": 0,
            "last_move": None,  # (player_id, row, col)
        }
        
        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
    
    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate initial instructions for a player."""
        prompt = (
            f"You are Player {player_id} in a Connect 4 game.\n\n"
            f"GAME RULES:\n"
            f"- The board has {self.rows} rows and {self.cols} columns\n"
            f"- Players take turns dropping pieces into columns\n"
            f"- Pieces fall to the lowest available position in the chosen column\n"
            f"- Win by getting 4 of your pieces in a row (horizontal, vertical, or diagonal)\n"
            f"- The game ends in a draw if the board is full with no winner\n\n"
            f"BOARD REPRESENTATION:\n"
            f"- Your pieces are shown as '{player_id}'\n"
            f"- Opponent's pieces are shown as '{1 - player_id}'\n"
            f"- Empty spaces are shown as ' '\n\n"
            f"ACTION FORMAT:\n"
            f"- To place a piece, write [N] where N is the column number (0-{self.cols - 1})\n"
            f"- Example: [3] means place your piece in column 3\n"
            f"- Only columns that aren't full are valid moves\n\n"
            f"STRATEGY TIPS:\n"
            f"- Try to create multiple threats at once\n"
            f"- Block your opponent if they're close to winning\n"
            f"- Control the center columns for more opportunities\n\n"
            f"The game lasts up to {self.state.max_turns} turns.\n"
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
            List of valid action strings: ["[0]", "[1]", ..., "[6]"] for columns
        """
        return [f"[{col}]" for col in range(self.cols)]
    
    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Process an action and update game state."""
        player_id = self.state.current_player_id
        
        # Log the raw action
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Parse the column number
        match = self.action_pattern.search(action)
        if not match:
            self.state.set_invalid_move(reason=f"Action must be in format [N] where N is a column number (0-{self.cols - 1})."
            )
            return self.state.step()
        
        try:
            column = int(match.group(1))
        except ValueError:
            self.state.set_invalid_move(reason=f"Invalid column number in action."
            )
            return self.state.step()
        
        # Validate column is in range
        if not (0 <= column < self.cols):
            self.state.set_invalid_move(reason=f"Column must be between 0 and {self.cols - 1}."
            )
            return self.state.step()
        
        # Get legal actions (columns that aren't full)
        legal_actions = self._get_legal_columns()
        if column not in legal_actions:
            legal_str = ", ".join([f"[{c}]" for c in legal_actions])
            self.state.set_invalid_move(reason=f"Column {column} is full. Legal moves: {legal_str}"
            )
            return self.state.step()
        
        # Find the lowest empty row in this column
        row = self._find_empty_row(column)
        if row is None:
            self.state.set_invalid_move(reason=f"Column {column} is full (should not happen)."
            )
            return self.state.step()
        
        # Place the piece
        self.state.game_state["board"][row][column] = player_id
        self.state.game_state["last_move"] = (player_id, row, column)
        self.state.game_state["move_count"] += 1
        
        # Show updated board to both players
        board_display = self._render_board()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id} placed a piece in column {column}.\n\n{board_display}"
        ,
            observation_type=ObservationType.GAME_BOARD)
        
        # Check for win
        if self._check_win(row, column, player_id):
            self.state.set_winner(
                player_id=player_id,
                reason=f"Player {player_id} got 4 in a row!"
            )
            return self.state.step()
        
        # Check for draw (board full)
        if len(self._get_legal_columns()) == 0:
            self.state.set_draw(
                reason="Board is full. Game ends in a draw."
            )
            return self.state.step()
        
        # Check for max turns truncation
        if self.state.turn >= self.max_turns - 1:
            self.state.set_draw(
                reason=f"Maximum turns ({self.max_turns}) reached. Game ends in a draw."
            )
            return self.state.step()
        
        # Show legal moves to next player
        legal_actions = self._get_legal_columns()
        legal_str = ", ".join([f"[{c}]" for c in legal_actions])
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=1 - player_id,
            message=f"Your available moves: {legal_str}"
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        return self.state.step()
    
    def _get_legal_columns(self) -> List[int]:
        """Get list of columns that aren't full."""
        board = self.state.game_state["board"]
        return [col for col in range(self.cols) if board[0][col] == -1]
    
    def _find_empty_row(self, column: int) -> Optional[int]:
        """Find the lowest empty row in a column (gravity effect)."""
        board = self.state.game_state["board"]
        for row in range(self.rows - 1, -1, -1):
            if board[row][column] == -1:
                return row
        return None
    
    def _check_win(self, row: int, col: int, player_id: int) -> bool:
        """Check if the last move resulted in a win."""
        # Check all four directions
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal down-right
            (1, -1),  # Diagonal down-left
        ]
        
        for dr, dc in directions:
            if self._check_direction(row, col, dr, dc, player_id):
                return True
        
        return False
    
    def _check_direction(self, row: int, col: int, dr: int, dc: int, player_id: int) -> bool:
        """Check if there are 4 in a row in a given direction."""
        board = self.state.game_state["board"]
        count = 1  # Count the piece we just placed
        
        # Check in positive direction
        r, c = row + dr, col + dc
        while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player_id:
            count += 1
            r += dr
            c += dc
        
        # Check in negative direction
        r, c = row - dr, col - dc
        while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player_id:
            count += 1
            r -= dr
            c -= dc
        
        return count >= 4
    
    def _render_board(self) -> str:
        """Render the board as a text string."""
        board = self.state.game_state["board"]
        lines = []
        lines.append("=" * (self.cols * 4 + 1))
        
        # Column numbers
        header = " "
        for col in range(self.cols):
            header += f" {col}  "
        lines.append(header)
        lines.append("=" * (self.cols * 4 + 1))
        
        # Board rows (top to bottom)
        for row in range(self.rows):
            line = "|"
            for col in range(self.cols):
                cell = board[row][col]
                if cell == -1:
                    line += "   |"
                else:
                    line += f" {cell} |"
            lines.append(line)
            lines.append("-" * (self.cols * 4 + 1))
        
        return "\n".join(lines)

