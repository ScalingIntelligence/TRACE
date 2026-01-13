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

"""Nim game environment for SPIRAL."""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType


class NimEnv(ta.Env):
    """
    Nim game environment (normal play variant).
    
    Classic combinatorial game where players take turns removing items from piles.
    The player who takes the last item wins (normal play).
    
    Game Setup:
    - n piles with random number of items (1 to max_items_per_pile)
    - Players alternate turns
    
    Actions: [pile_index, num_items]
    - Take num_items from pile_index
    - Example: [0,2] takes 2 items from pile 0
    
    Win Condition: Player who empties the last pile (takes final item) wins
    """
    
    def __init__(
        self, 
        num_piles: int = 4, 
        max_items_per_pile: int = 5,
        max_turns: int = 100
    ):
        """
        Initialize Nim environment.
        
        Args:
            num_piles: Number of piles (default: 4)
            max_items_per_pile: Maximum items per pile at start (default: 5)
            max_turns: Maximum turns before draw (default: 100)
        """
        super().__init__()
        self.num_piles = num_piles
        self.max_items_per_pile = max_items_per_pile
        self.max_turns = max_turns
        
        # Regex to match [pile_index, num_items] format
        self.action_pattern = re.compile(r"\[(\d+)\s*,\s*(\d+)\]")
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            random.seed(seed)
        
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.max_turns,
        )
        
        # Initialize piles with random number of items (1 to max_items_per_pile)
        piles = [random.randint(1, self.max_items_per_pile) for _ in range(self.num_piles)]
        
        # Initialize game state
        game_state = {
            "piles": piles,
            "initial_piles": piles.copy(),  # Store initial state for display
            "move_history": [],  # List of (player_id, pile_index, num_items) tuples
            "turn_count": 0,
        }
        
        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Send initial game state to both players
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
            f"You are Player {player_id} in a Nim game.\n\n"
            f"GAME SETUP:\n"
            f"- There are {self.num_piles} piles of items\n"
            f"- Each pile starts with 1-{self.max_items_per_pile} items (randomly determined)\n"
            f"- Players take turns removing items from piles\n\n"
            f"GAME RULES:\n"
            f"- On your turn, you must take at least 1 item from exactly one pile\n"
            f"- You can take as many items as you want from that pile (up to all remaining)\n"
            f"- The player who takes the LAST item (empties all piles) WINS\n\n"
            f"ACTION FORMAT:\n"
            f"- To take items, write [pile_index, num_items]\n"
            f"- pile_index: which pile to take from (0 to {self.num_piles-1})\n"
            f"- num_items: how many items to take (1 to remaining items in that pile)\n"
            f"- Example: [0,2] takes 2 items from pile 0\n"
            f"- Example: [1,1] takes 1 item from pile 1\n\n"
            f"STRATEGY TIPS:\n"
            f"- Think about the total number of items remaining\n"
            f"- Consider what position you leave your opponent in\n"
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
            List of valid action strings: ["[pile_index, num_items]", ...]
        """
        actions = []
        piles = self.state.game_state["piles"]
        
        for pile_idx, count in enumerate(piles):
            if count > 0:
                # Return ALL valid actions: taking 1 to all items from this pile
                for take in range(1, count + 1):
                    actions.append(f"[{pile_idx},{take}]")
        
        return actions if actions else ["[0,1]"]  # Fallback
    
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
        
        # Parse the action
        match = self.action_pattern.search(action)
        if not match:
            self.state.set_invalid_move(reason=f"Action must be in format [pile_index, num_items] where pile_index is 0-{self.num_piles-1} and num_items is positive."
            )
            return self.state.step()
        
        try:
            pile_index = int(match.group(1))
            num_items = int(match.group(2))
        except ValueError:
            self.state.set_invalid_move(reason="Invalid pile index or number of items."
            )
            return self.state.step()
        
        # Validate pile index
        if not (0 <= pile_index < self.num_piles):
            self.state.set_invalid_move(reason=f"Pile index must be between 0 and {self.num_piles-1}."
            )
            return self.state.step()
        
        # Validate num_items is positive
        if num_items <= 0:
            self.state.set_invalid_move(reason="You must take at least 1 item."
            )
            return self.state.step()
        
        # Check if pile has enough items
        current_pile_size = self.state.game_state["piles"][pile_index]
        if num_items > current_pile_size:
            self.state.set_invalid_move(reason=f"Pile {pile_index} only has {current_pile_size} items. You tried to take {num_items}."
            )
            return self.state.step()
        
        # Check if pile is already empty
        if current_pile_size == 0:
            self.state.set_invalid_move(reason=f"Pile {pile_index} is already empty. Choose a different pile."
            )
            return self.state.step()
        
        # Valid move - execute it
        self.state.game_state["piles"][pile_index] -= num_items
        self.state.game_state["move_history"].append((player_id, pile_index, num_items))
        self.state.game_state["turn_count"] += 1
        
        # Announce the move
        items_word = "item" if num_items == 1 else "items"
        remaining = self.state.game_state["piles"][pile_index]
        move_msg = (
            f"Player {player_id} takes {num_items} {items_word} from pile {pile_index}. "
            f"Pile {pile_index} now has {remaining} items."
        )
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=move_msg
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Check if game is over (all piles empty)
        if all(pile == 0 for pile in self.state.game_state["piles"]):
            # Current player took the last item and wins!
            total_items = sum(self.state.game_state["initial_piles"])
            self.state.set_winner(
                player_id=player_id,
                reason=f"Player {player_id} took the last item and wins! (Total items: {total_items})"
            )
            return self.state.step()
        
        # Show updated game state to next player
        game_state_msg = self._format_game_state()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=1 - player_id,
            message=game_state_msg
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        return self.state.step()
    
    def _format_game_state(self) -> str:
        """Format current game state for display."""
        piles = self.state.game_state["piles"]
        initial_piles = self.state.game_state["initial_piles"]
        history = self.state.game_state["move_history"]
        
        msg = f"=== Nim Game: {self.num_piles} Piles ===\n\n"
        
        # Show recent move history (last 5 moves) FIRST
        if history:
            msg += f"RECENT MOVES (last {min(5, len(history))}):\n"
            for player_id, pile_idx, num_taken in history[-5:]:
                items_word = "item" if num_taken == 1 else "items"
                msg += f"  Player {player_id} took {num_taken} {items_word} from pile {pile_idx}\n"
        else:
            msg += "No moves yet.\n"
        
        # Show current piles AFTER recent moves
        msg += "\nCURRENT PILES:\n"
        
        # Show each pile with visual representation
        for i, count in enumerate(piles):
            items_visual = "●" * count if count > 0 else "(empty)"
            initial = initial_piles[i]
            msg += f"  Pile {i}: {count} items {items_visual} (started with {initial})\n"
        
        msg += f"\nYour turn! Choose [pile_index, num_items]"
        
        return msg

