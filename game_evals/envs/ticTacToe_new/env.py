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

"""Tic Tac Toe game environment for SPIRAL."""

import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType


class TicTacToeEnv(ta.Env):
    """
    Tic Tac Toe game environment.
    
    Classic 3x3 board where players try to get 3 in a row.
    Actions are position numbers (0-8) in row-major order:
    
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
    """
    
    def __init__(self, max_turns: int = 9):
        """
        Initialize Tic Tac Toe environment.
        
        Args:
            max_turns: Maximum turns before draw (default: 9, full board)
        """
        super().__init__()
        self.board_size = 3
        self.max_turns = max_turns
        
        # Regex to match [0], [1], ..., [8] format
        self.action_pattern = re.compile(r"\[(\d)\]")
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.max_turns,
        )
        
        # Initialize game state
        # Board: -1 = empty, 0 = player 0 (X), 1 = player 1 (O)
        game_state = {
            "board": [-1] * 9,  # 3x3 board flattened
            "move_count": 0,
            "last_move": None,  # (player_id, position)
        }
        
        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Show initial board to both players
        board_display = self._render_board()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Game starting!\n\n{board_display}"
        ,
            observation_type=ObservationType.GAME_BOARD)
        
        # Show available moves to first player (don't log to history)
        legal_actions = self._get_legal_positions()
        legal_str = ", ".join([f"[{pos}]" for pos in legal_actions])
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=0,
            message=f"YOUR SYMBOL: X\nYour available moves: {legal_str}",
            observation_type=ObservationType.GAME_MESSAGE
        )
    
    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate initial instructions for a player."""
        symbol = "X" if player_id == 0 else "O"
        opponent_symbol = "O" if player_id == 0 else "X"
        
        prompt = (
            f"You are Player {player_id} in a Tic Tac Toe game.\n\n"
            f"═══════════════════════════════════════\n"
            f"YOUR SYMBOL: {symbol}\n"
            f"OPPONENT'S SYMBOL: {opponent_symbol}\n"
            f"═══════════════════════════════════════\n\n"
            f"GAME RULES:\n"
            f"- The board is a 3x3 grid with positions numbered 0-8:\n"
            f"  0 | 1 | 2\n"
            f"  ---------\n"
            f"  3 | 4 | 5\n"
            f"  ---------\n"
            f"  6 | 7 | 8\n\n"
            f"- Players take turns placing their symbol on the board\n"
            f"- Win by getting 3 of YOUR symbols ({symbol}) in a row (horizontal, vertical, or diagonal)\n"
            f"- The game ends in a draw if the board is full with no winner\n\n"
            f"BOARD REPRESENTATION:\n"
            f"- YOUR pieces are shown as '{symbol}'\n"
            f"- OPPONENT'S pieces are shown as '{opponent_symbol}'\n"
            f"- Empty spaces are shown as position numbers in parentheses, e.g. (0), (1), etc.\n\n"
            f"ACTION FORMAT:\n"
            f"- To place your symbol ({symbol}), write [N] where N is the position number (0-8)\n"
            f"- Example: [4] means place {symbol} in the center position\n"
            f"- Only empty positions are valid moves\n\n"
            f"STRATEGY TIPS:\n"
            f"- Control the center (position 4) for maximum opportunities\n"
            f"- Block your opponent if they're about to win\n"
            f"- Try to create multiple winning threats at once\n"
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
            List of valid action strings: ["[0]", "[1]", ..., "[8]"] for positions
        """
        return [f"[{pos}]" for pos in range(9)]
    
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
        
        # Parse the position number
        match = self.action_pattern.search(action)
        if not match:
            self.state.set_invalid_move(reason=f"Action must be in format [N] where N is a position number (0-8)."
            )
            return self.state.step()
        
        try:
            position = int(match.group(1))
        except ValueError:
            self.state.set_invalid_move(reason=f"Invalid position number in action."
            )
            return self.state.step()
        
        # Validate position is in range
        if not (0 <= position <= 8):
            self.state.set_invalid_move(reason=f"Position must be between 0 and 8."
            )
            return self.state.step()
        
        # Check if position is empty
        if self.state.game_state["board"][position] != -1:
            legal_actions = self._get_legal_positions()
            legal_str = ", ".join([f"[{pos}]" for pos in legal_actions])
            self.state.set_invalid_move(reason=f"Position {position} is already occupied. Legal moves: {legal_str}"
            )
            return self.state.step()
        
        # Place the piece
        self.state.game_state["board"][position] = player_id
        self.state.game_state["last_move"] = (player_id, position)
        self.state.game_state["move_count"] += 1
        
        # Show updated board to both players
        board_display = self._render_board()
        symbol = "X" if player_id == 0 else "O"
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id} ({symbol}) placed at position {position}.\n\n{board_display}",
            observation_type=ObservationType.GAME_MESSAGE,
        )
        
        # Check for win
        if self._check_win(position, player_id):
            self.state.set_winner(
                player_id=player_id,
                reason=f"Player {player_id} ({symbol}) got 3 in a row!"
            )
            return self.state.step()
        
        # Check for draw (board full)
        if len(self._get_legal_positions()) == 0:
            self.state.set_draw(
                reason="Board is full. Game ends in a draw."
            )
            return self.state.step()
        
        # Show legal moves to next player (don't log to history)
        next_player = 1 - player_id
        next_symbol = "X" if next_player == 0 else "O"
        legal_actions = self._get_legal_positions()
        legal_str = ", ".join([f"[{pos}]" for pos in legal_actions])
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=next_player,
            message=f"YOUR SYMBOL: {next_symbol}\nYour available moves: {legal_str}",
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        return self.state.step()
    
    def _get_legal_positions(self) -> List[int]:
        """Get list of empty positions."""
        board = self.state.game_state["board"]
        return [pos for pos in range(9) if board[pos] == -1]
    
    def _check_win(self, position: int, player_id: int) -> bool:
        """Check if the last move resulted in a win."""
        board = self.state.game_state["board"]
        
        # Define all winning combinations (rows, columns, diagonals)
        winning_combinations = [
            [0, 1, 2],  # Top row
            [3, 4, 5],  # Middle row
            [6, 7, 8],  # Bottom row
            [0, 3, 6],  # Left column
            [1, 4, 7],  # Middle column
            [2, 5, 8],  # Right column
            [0, 4, 8],  # Diagonal top-left to bottom-right
            [2, 4, 6],  # Diagonal top-right to bottom-left
        ]
        
        # Check if player has any winning combination
        for combo in winning_combinations:
            if all(board[pos] == player_id for pos in combo):
                return True
        
        return False
    
    def _render_board(self) -> str:
        """Render the board as a text string."""
        board = self.state.game_state["board"]
        
        def cell_str(pos: int) -> str:
            """Convert cell value to display string."""
            val = board[pos]
            if val == -1:
                return f"({pos})"  # Show position number for empty cells
            elif val == 0:
                return " X "
            else:  # val == 1
                return " O "
        
        lines = []
        lines.append("=" * 13)
        
        # Render the 3x3 board
        for row in range(3):
            row_cells = []
            for col in range(3):
                pos = row * 3 + col
                row_cells.append(cell_str(pos))
            lines.append("|".join(row_cells))
            
            # Add separator between rows (but not after last row)
            if row < 2:
                lines.append("-" * 13)
        
        lines.append("=" * 13)
        
        return "\n".join(lines)

