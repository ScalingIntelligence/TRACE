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

"""Dots and Boxes game environment for SPIRAL."""

import random
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import textarena as ta
from textarena.core import ObservationType


class DotsAndBoxesEnv(ta.Env):
    """
    Dots and Boxes game environment.
    
    Classic two-player territory control game on a dot grid.
    Players alternate drawing edges between adjacent dots.
    Completing a box (1x1 square) scores a point and grants another turn.
    Player with most boxes when all are claimed wins.
    
    The game demonstrates emergent complexity from simple rules:
    - Local actions (drawing lines) create strategic territory patterns
    - Endgame involves chain control and combinatorial foresight
    - Optimal play requires sacrifice and long-term planning
    """
    
    def __init__(self, rows: int = 5, cols: int = 5):
        """
        Initialize Dots and Boxes environment.
        
        Args:
            rows: Number of dot rows (default: 5, creates 4x4 boxes)
            cols: Number of dot columns (default: 5, creates 4x4 boxes)
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_boxes = (rows - 1) * (cols - 1)
        
        # Edge action pattern: [r1,c1,r2,c2] for adjacent dots
        self.action_pattern = re.compile(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]")
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            random.seed(seed)
        
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.num_boxes * 4,  # Max possible edges
        )
        
        # Initialize game state
        game_state = {
            # Edges: set of tuples ((r1,c1), (r2,c2)) where r1,c1 <= r2,c2
            "edges": set(),
            # Box ownership: dict from (box_row, box_col) -> player_id
            "boxes": {},
            # Scores
            "scores": {0: 0, 1: 0},
            # Last move completed a box (grants extra turn)
            "extra_turn": False,
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
            f"You are Player {player_id} in a Dots and Boxes game.\n\n"
            f"═══════════════════════════════════════\n"
            f"GAME SETUP:\n"
            f"═══════════════════════════════════════\n"
            f"- Grid: {self.rows}×{self.cols} dots (makes {self.rows-1}×{self.cols-1} boxes)\n"
            f"- Dot coordinates: rows 0-{self.rows-1}, cols 0-{self.cols-1}\n"
            f"- Goal: Complete more boxes than your opponent\n\n"
            f"GAME RULES:\n\n"
            f"1. Players alternate drawing edges between adjacent dots\n"
            f"2. ⚠️  IMPORTANT: You CANNOT draw an edge that already exists!\n"
            f"   - Each edge can only be drawn ONCE\n"
            f"   - Drawing an existing edge is an INVALID MOVE and you will LOSE\n"
            f"3. If your edge completes a 1×1 box:\n"
            f"   - You claim that box and score +1 point\n"
            f"   - You immediately get another turn (extra turn)\n"
            f"4. Game ends when all {self.num_boxes} boxes are claimed\n"
            f"5. Player with most boxes wins\n\n"
            f"ACTION FORMAT:\n\n"
            f"Draw an edge with: [row1,col1,row2,col2]\n\n"
            f"⚠️  COORDINATE FORMAT: [ROW, COLUMN, ROW, COLUMN]\n"
            f"   - FIRST number = ROW (vertical position, 0 = top)\n"
            f"   - SECOND number = COLUMN (horizontal position, 0 = left)\n"
            f"   - THIRD number = ROW of adjacent dot\n"
            f"   - FOURTH number = COLUMN of adjacent dot\n\n"
            f"- Must connect two adjacent dots (horizontally or vertically, exactly 1 step apart)\n"
            f"- Cannot draw an edge that already exists (shown as — or │)\n\n"
            f"EXAMPLES:\n"
            f"  [0,0,0,1] = horizontal edge from (row 0, col 0) to (row 0, col 1)\n"
            f"              This connects top-left dot to the dot on its RIGHT\n"
            f"  [0,0,1,0] = vertical edge from (row 0, col 0) to (row 1, col 0)\n"
            f"              This connects top-left dot to the dot BELOW it\n"
            f"  [2,3,2,4] = horizontal edge from (row 2, col 3) to (row 2, col 4)\n"
            f"  [1,2,2,2] = vertical edge from (row 1, col 2) to (row 2, col 2)\n\n"
            f"BOARD DISPLAY:\n\n"
            f"- Dots: ●\n"
            f"- Edges: — (horizontal) or │ (vertical)\n"
            f"- Empty space: · (horizontal) or   (vertical)\n"
            f"- Boxes: 0 (yours), 1 (opponent's), · (unclaimed)\n\n"
            f"STRATEGY TIPS:\n\n"
            f"- Early game: Avoid giving opponent opportunities to complete boxes\n"
            f"- Mid game: Control chains of nearly-complete boxes\n"
            f"- Late game: Strategic sacrifice can lead to capturing larger chains\n"
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
            List of valid action strings: ["[r1,c1,r2,c2]", ...] for all possible edges
        """
        actions = []
        
        # Horizontal edges: connect (r,c) to (r,c+1)
        for row in range(self.rows):
            for col in range(self.cols - 1):
                actions.append(f"[{row},{col},{row},{col+1}]")
        
        # Vertical edges: connect (r,c) to (r+1,c)
        for row in range(self.rows - 1):
            for col in range(self.cols):
                actions.append(f"[{row},{col},{row+1},{col}]")
        
        return actions
    
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
        
        # Parse the edge coordinates
        match = self.action_pattern.search(action)
        if not match:
            self.state.set_invalid_move(reason=(
                    f"Invalid action format. Use [row1,col1,row2,col2] where:\n"
                    f"  FORMAT: [ROW, COLUMN, ROW, COLUMN]\n"
                    f"  row1,col1 and row2,col2 are adjacent dots\n"
                    f"  Coordinates must be 0-{self.rows-1} (rows) and 0-{self.cols-1} (cols)\n"
                    f"Example: [0,0,0,1] = horizontal edge (same row 0, columns 0→1)\n"
                    f"Example: [0,0,1,0] = vertical edge (rows 0→1, same column 0)"
                )
            )
            return self.state.step()
        
        try:
            r1, c1, r2, c2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        except ValueError:
            self.state.set_invalid_move(reason="Invalid coordinates."
            )
            return self.state.step()
        
        # Validate coordinates are in bounds
        if not (0 <= r1 < self.rows and 0 <= c1 < self.cols and 
                0 <= r2 < self.rows and 0 <= c2 < self.cols):
            self.state.set_invalid_move(reason=f"Coordinates out of bounds. Must be 0-{self.rows-1} for rows, 0-{self.cols-1} for cols."
            )
            return self.state.step()
        
        # Validate adjacency (must be exactly 1 step away horizontally or vertically)
        if not ((r1 == r2 and abs(c1 - c2) == 1) or (c1 == c2 and abs(r1 - r2) == 1)):
            self.state.set_invalid_move(reason="Dots must be adjacent (horizontally or vertically, exactly 1 step apart)."
            )
            return self.state.step()
        
        # Normalize edge (always store with smaller coordinate first)
        edge = tuple(sorted([(r1, c1), (r2, c2)]))
        
        # Check if edge already exists
        if edge in self.state.game_state["edges"]:
            self.state.set_invalid_move(reason=(
                    f"⚠️  INVALID: Edge from (row {r1}, col {c1}) to (row {r2}, col {c2}) ALREADY EXISTS!\n"
                    f"You cannot draw an edge that has already been drawn.\n"
                    f"Look at the board - edges are shown as — (horizontal) or │ (vertical).\n"
                    f"Choose a different edge where you see · or empty space."
                )
            )
            return self.state.step()
        
        # Valid move - add the edge
        self.state.game_state["edges"].add(edge)
        
        # Check if this edge completed any boxes
        completed_boxes = self._check_completed_boxes(edge)
        
        if completed_boxes:
            # Player claims the completed boxes
            for box_pos in completed_boxes:
                self.state.game_state["boxes"][box_pos] = player_id
                self.state.game_state["scores"][player_id] += 1
            
            boxes_word = "box" if len(completed_boxes) == 1 else "boxes"
            message = f"Player {player_id} draws edge [{r1},{c1},{r2},{c2}] - Completed {len(completed_boxes)} {boxes_word}! +{len(completed_boxes)} point(s). EXTRA TURN!"
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=message
            ,
            observation_type=ObservationType.GAME_MESSAGE)
            
            # Grant extra turn
            self.state.game_state["extra_turn"] = True
            
            # Check if game is over (all boxes claimed)
            if len(self.state.game_state["boxes"]) == self.num_boxes:
                score_0 = self.state.game_state["scores"][0]
                score_1 = self.state.game_state["scores"][1]
                
                if score_0 > score_1:
                    self.state.set_winner(
                        player_id=0,
                        reason=f"Player 0 wins with {score_0} boxes vs Player 1's {score_1} boxes!"
                    )
                elif score_1 > score_0:
                    self.state.set_winner(
                        player_id=1,
                        reason=f"Player 1 wins with {score_1} boxes vs Player 0's {score_0} boxes!"
                    )
                else:
                    self.state.set_draw(
                        reason=f"Draw! Both players claimed {score_0} boxes."
                    )
                
                return self.state.step()
            
            # Show updated board (don't change turn yet)
            board_msg = self._format_game_state()
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=player_id,
                message=board_msg,
                observation_type=ObservationType.GAME_BOARD
            )
            
            # Return without changing turn (same player goes again)
            return False, {}
        else:
            # No box completed - normal turn
            message = f"Player {player_id} draws edge [{r1},{c1},{r2},{c2}]"
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=message
            ,
            observation_type=ObservationType.GAME_MESSAGE)
            
            self.state.game_state["extra_turn"] = False
            
            # Show updated board to next player (don't log to history)
            opponent_id = 1 - player_id
            board_msg = self._format_game_state()
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=opponent_id,
                message=board_msg,
                observation_type=ObservationType.GAME_BOARD
            )
            
            return self.state.step()
    
    def _check_completed_boxes(self, edge: Tuple[Tuple[int, int], Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Check if adding this edge completes any boxes.
        
        Returns list of (box_row, box_col) positions for completed boxes.
        A box at (r, c) is defined by corners: (r,c), (r,c+1), (r+1,c), (r+1,c+1)
        """
        (r1, c1), (r2, c2) = edge
        completed = []
        
        # Determine if edge is horizontal or vertical
        if r1 == r2:  # Horizontal edge
            row = r1
            col_min = min(c1, c2)
            
            # Check box above (if exists)
            if row > 0:
                box_pos = (row - 1, col_min)
                if self._is_box_complete(box_pos) and box_pos not in self.state.game_state["boxes"]:
                    completed.append(box_pos)
            
            # Check box below (if exists)
            if row < self.rows - 1:
                box_pos = (row, col_min)
                if self._is_box_complete(box_pos) and box_pos not in self.state.game_state["boxes"]:
                    completed.append(box_pos)
        
        else:  # Vertical edge (c1 == c2)
            col = c1
            row_min = min(r1, r2)
            
            # Check box to the left (if exists)
            if col > 0:
                box_pos = (row_min, col - 1)
                if self._is_box_complete(box_pos) and box_pos not in self.state.game_state["boxes"]:
                    completed.append(box_pos)
            
            # Check box to the right (if exists)
            if col < self.cols - 1:
                box_pos = (row_min, col)
                if self._is_box_complete(box_pos) and box_pos not in self.state.game_state["boxes"]:
                    completed.append(box_pos)
        
        return completed
    
    def _is_box_complete(self, box_pos: Tuple[int, int]) -> bool:
        """
        Check if a box at (row, col) is complete (all 4 edges present).
        
        Box at (r, c) has corners: (r,c), (r,c+1), (r+1,c), (r+1,c+1)
        Edges needed:
        - Top: (r,c) to (r,c+1)
        - Bottom: (r+1,c) to (r+1,c+1)
        - Left: (r,c) to (r+1,c)
        - Right: (r,c+1) to (r+1,c+1)
        """
        r, c = box_pos
        edges = self.state.game_state["edges"]
        
        top = tuple(sorted([(r, c), (r, c+1)]))
        bottom = tuple(sorted([(r+1, c), (r+1, c+1)]))
        left = tuple(sorted([(r, c), (r+1, c)]))
        right = tuple(sorted([(r, c+1), (r+1, c+1)]))
        
        return all(edge in edges for edge in [top, bottom, left, right])
    
    def _format_game_state(self) -> str:
        """Format current game state showing the board."""
        msg = f"\n{'='*60}\n"
        msg += f"DOTS AND BOXES - {self.rows}×{self.cols} Grid\n"
        msg += f"{'='*60}\n\n"
        
        # Scores
        msg += f"SCORE: Player 0 = {self.state.game_state['scores'][0]} | Player 1 = {self.state.game_state['scores'][1]}\n"
        msg += f"Boxes remaining: {self.num_boxes - len(self.state.game_state['boxes'])}/{self.num_boxes}\n\n"
        
        # Render the board
        msg += self._render_board()
        
        return msg
    
    def _render_board(self) -> str:
        """Render the game board with dots, edges, and boxes."""
        edges = self.state.game_state["edges"]
        boxes = self.state.game_state["boxes"]
        
        lines = []
        
        for r in range(self.rows):
            # Row of dots and horizontal edges
            row_line = ""
            for c in range(self.cols):
                # Add dot
                row_line += "●"
                
                # Add horizontal edge or space (if not last column)
                if c < self.cols - 1:
                    h_edge = tuple(sorted([(r, c), (r, c+1)]))
                    if h_edge in edges:
                        row_line += "———"
                    else:
                        row_line += "···"
            
            lines.append(row_line)
            
            # Row of vertical edges and box interiors (if not last row)
            if r < self.rows - 1:
                v_line = ""
                for c in range(self.cols):
                    # Add vertical edge or space
                    v_edge = tuple(sorted([(r, c), (r+1, c)]))
                    if v_edge in edges:
                        v_line += "│"
                    else:
                        v_line += " "
                    
                    # Add box interior (if not last column)
                    if c < self.cols - 1:
                        box_pos = (r, c)
                        if box_pos in boxes:
                            owner = boxes[box_pos]
                            v_line += f" {owner} "
                        else:
                            v_line += " · "
                
                lines.append(v_line)
        
        return "\n".join(lines) + "\n"

