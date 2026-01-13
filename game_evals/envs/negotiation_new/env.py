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

"""Negotiation game environment for SPIRAL (adapted from weaver_for_RL)."""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType


class NegotiationEnv(ta.Env):
    """
    Multi-issue negotiation environment.
    
    Players negotiate over multiple items with different utility values.
    Actions: [Propose: allocation] or [Accept]
    Goal: Reach agreement to maximize utility (both players can win!)
    
    NOTE: This is a NON-ZERO-SUM game - both players can benefit from good deals.
    """
    
    def __init__(
        self,
        num_items: int = 5,
        quantity_per_item: int = 3,
        max_steps: int = 10,
        reward_range: Tuple[int, int] = (1, 5),
        verbal: bool = False,
    ):
        """
        Initialize negotiation environment.
        
        Args:
            num_items: Number of item types to negotiate over
            quantity_per_item: How many of each item available
            max_steps: Maximum negotiation rounds before deadlock
            reward_range: Range for utility values (min, max)
            verbal: If True, players can send messages with proposals
        """
        super().__init__()
        self.num_items = num_items
        self.quantity_per_item = quantity_per_item
        self.max_steps = max_steps
        self.reward_range = reward_range
        self.verbal = verbal
        
        # Regex to match actions
        self.accept_pattern = re.compile(r"\[Accept\]", re.IGNORECASE)
        
        # Update propose pattern to optionally capture message if verbal mode enabled
        if self.verbal:
            # Format: [Propose: 1,2,3] with optional message after
            self.propose_pattern = re.compile(
                r"\[Propose:\s*([0-9,\s]+)\]\s*(.*)",
                re.IGNORECASE | re.DOTALL
            )
        else:
            self.propose_pattern = re.compile(
                r"\[Propose:\s*([0-9,\s]+)\]",
                re.IGNORECASE
            )
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=self.max_steps,
        )
        
        # Generate random utilities for this game
        if seed is not None:
            random.seed(seed)
        
        player_utilities = self._generate_utility_vectors()
        item_pool = [self.quantity_per_item] * self.num_items
        item_names = [f"Item {chr(65 + i)}" for i in range(self.num_items)]
        
        # Initialize game state
        game_state = {
            "item_pool": item_pool,
            "player_utilities": player_utilities,
            "item_names": item_names,
            "last_proposal": None,
            "last_proposer": None,
            "agreement": None,
            "step_count": 0,
        }
        
        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Send initial information to both players
        for player_id in range(2):
            initial_msg = self._format_game_state_for_player(player_id)
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=player_id,
                message=initial_msg
            ,
            observation_type=ObservationType.GAME_MESSAGE)
    
    def _generate_utility_vectors(self) -> List[List[int]]:
        """Generate random utility vectors for both players."""
        # Positive and negative items
        num_positive = (self.num_items + 1) // 2  # Roughly half positive
        num_negative = self.num_items - num_positive
        
        def generate_single_utility() -> List[int]:
            indices = list(range(self.num_items))
            random.shuffle(indices)
            positive_indices = set(indices[:num_positive])
            
            utility = []
            for i in range(self.num_items):
                if i in positive_indices:
                    utility.append(random.randint(*self.reward_range))
                else:
                    utility.append(-random.randint(*self.reward_range))
            return utility
        
        return [generate_single_utility(), generate_single_utility()]
    
    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate initial instructions for a player."""
        utilities = game_state["player_utilities"][player_id]
        item_names = game_state["item_names"]
        
        # Find which items are valuable
        positive_items = [
            (item_names[i], utilities[i]) 
            for i in range(self.num_items) 
            if utilities[i] > 0
        ]
        negative_items = [
            (item_names[i], utilities[i]) 
            for i in range(self.num_items) 
            if utilities[i] < 0
        ]
        
        positive_items.sort(key=lambda x: x[1], reverse=True)
        
        prompt = (
            f"You are Player {player_id} in a multi-issue negotiation game.\n\n"
            f"GAME RULES:\n"
            f"- You negotiate with an opponent over {self.num_items} types of items\n"
            f"- Available items: {', '.join(item_names)}\n"
            f"- Each item type has {self.quantity_per_item} units available\n"
            f"- Each item has DIFFERENT VALUE to you vs your opponent\n"
            f"- You can make proposals OR accept the opponent's current proposal\n"
            f"- The game ends when someone accepts OR max rounds ({self.max_steps}) reached\n"
            f"- If no agreement: BOTH players get 0 utility (deadlock = worst outcome!)\n\n"
            f"YOUR UTILITY VALUES:\n"
        )
        
        for i, name in enumerate(item_names):
            util = utilities[i]
            prompt += f"  - {name}: {util:+d} points per unit\n"
        
        prompt += (
            f"\nYOUR GOAL:\n"
            f"Maximize your utility by proposing allocations and accepting good offers.\n"
        )
        
        if positive_items:
            top_items = [name for name, _ in positive_items[:2]]
            prompt += f"Prioritize getting {' and '.join(top_items)}.\n"
        
        if negative_items:
            avoid_items = [name for name, _ in negative_items[:2]]
            prompt += f"Avoid {' and '.join(avoid_items)} (negative value!).\n"
        
        prompt += f"\nACTION FORMAT:\n"
        
        if self.verbal:
            prompt += (
                f"1. TO PROPOSE an allocation (with optional message):\n"
                f"   [Propose: a,b,c,d,e] where a,b,c,d,e are quantities for items {','.join(item_names)}\n"
                f"   - Optionally, add a message AFTER the proposal to communicate with your opponent\n"
                f"   - Your message can be reasonably long - make a good effort to convince them!\n"
                f"   - Explain your reasoning, appeal to fairness, suggest why your proposal benefits both parties\n"
                f"   - Example: [Propose: 3,2,1,0,2] I think this is fair for both of us!\n"
                f"   - Example: [Propose: 2,2,2,1,1] Let's split evenly and compromise - we both benefit from avoiding deadlock.\n\n"
                f"   ⚠️  WARNING: Your message CANNOT contain ANY of your utility numbers!\n"
                f"   - If your message mentions {', '.join(str(abs(u)) for u in utilities)}, you will INSTANTLY LOSE!\n"
                f"   - Do NOT reveal your utility values to the opponent\n"
                f"   - Do NOT say things like 'I value Item A at 5 points'\n"
                f"   - You CAN say general things like 'I prefer Item A' or 'This seems fair'\n\n"
                f"   Your proposal format:\n"
                f"   - Your proposal specifies what YOU get\n"
                f"   - Opponent automatically gets the remainder\n"
                f"   - All quantities must be ≤ available quantities\n\n"
                f"2. TO ACCEPT the opponent's proposal:\n"
                f"   [Accept]\n"
                f"   - This ENDS the game immediately\n"
                f"   - You receive the utility from their proposed allocation\n\n"
            )
        else:
            prompt += (
                f"1. TO PROPOSE an allocation:\n"
                f"   [Propose: a,b,c,d,e] where a,b,c,d,e are quantities for items {','.join(item_names)}\n"
                f"   Example: [Propose: 3,2,1,0,2] means you want 3×{item_names[0]}, 2×{item_names[1]}, etc.\n"
                f"   - Your proposal specifies what YOU get\n"
                f"   - Opponent automatically gets the remainder\n"
                f"   - All quantities must be ≤ available quantities\n\n"
                f"2. TO ACCEPT the opponent's proposal:\n"
                f"   [Accept]\n"
                f"   - This ENDS the game immediately\n"
                f"   - You receive the utility from their proposed allocation\n\n"
            )
        
        prompt += (
            f"STRATEGY TIPS:\n"
            f"- Start with proposals favoring items you value highly\n"
            f"- Look for win-win allocations (give them items they value, keep items you value)\n"
            f"- ACCEPT reasonable offers to avoid deadlock!\n"
            f"- Early rounds: Be ambitious (70-80%% of max utility)\n"
            f"- Later rounds: Be more flexible to reach agreement\n"
            f"- Deadlock = 0 utility for everyone (worst outcome!)\n"
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
            List of valid action strings: ["[Propose: 0,0,0,0,0]", ..., "[Accept]"]
        """
        import itertools
        
        actions = []
        
        # Get item pool quantities
        item_pool = self.state.game_state["item_pool"]
        
        # Generate all possible combinations of quantities (0 to max for each item)
        ranges = [range(q + 1) for q in item_pool]
        
        # Generate all valid proposals
        for combo in itertools.product(*ranges):
            proposal_str = ','.join(map(str, combo))
            actions.append(f"[Propose: {proposal_str}]")
        
        # Only add Accept action if there's a proposal from the opponent
        last_proposal = self.state.game_state["last_proposal"]
        last_proposer = self.state.game_state["last_proposer"]
        if last_proposal is not None and last_proposer != player_id:
            actions.append("[Accept]")
        
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
        
        # Check for Accept
        if self.accept_pattern.search(action):
            return self._handle_accept(player_id)
        
        # Check for Propose
        propose_match = self.propose_pattern.search(action)
        if propose_match:
            allocation_str = propose_match.group(1)
            message = None
            if self.verbal and len(propose_match.groups()) > 1:
                message = propose_match.group(2).strip()
                if not message:
                    message = None
            return self._handle_propose(player_id, allocation_str, message)
        
        # Invalid action format
        self.state.set_invalid_move(reason="Action must be [Accept] or [Propose: a,b,c,d,e] where a,b,c,d,e are numbers."
        )
        return self.state.step()
    
    def _handle_accept(self, player_id: int) -> Tuple[bool, Dict[str, Any]]:
        """Handle accept action."""
        last_proposal = self.state.game_state["last_proposal"]
        last_proposer = self.state.game_state["last_proposer"]
        
        # Check if there's something to accept
        if last_proposal is None:
            self.state.set_invalid_move(reason="Nothing to accept - no proposal has been made yet."
            )
            return self.state.step()
        
        # Check if trying to accept own proposal
        if last_proposer == player_id:
            self.state.set_invalid_move(reason="You cannot accept your own proposal."
            )
            return self.state.step()
        
        # Agreement reached!
        self.state.game_state["agreement"] = last_proposal
        
        # Calculate utilities
        utilities = self._calculate_utilities(last_proposal, last_proposer)
        
        # Announce agreement
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id} accepted the proposal! Agreement reached."
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Determine outcome based on utilities
        # Since this is non-zero-sum, we use custom rewards
        self._set_outcome_with_utilities(utilities)
        
        done, info = self.state.step()
        
        # Add rewards to info dict if they exist in game_state
        if "final_rewards" in self.state.game_state:
            info["rewards"] = self.state.game_state["final_rewards"]
        
        return done, info
    
    def _handle_propose(
        self, player_id: int, allocation_str: str, message: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Handle propose action with optional verbal message."""
        # Parse allocation
        try:
            allocation = [int(x.strip()) for x in allocation_str.split(',')]
        except ValueError:
            self.state.set_invalid_move(reason=f"Invalid allocation format: {allocation_str}. Must be comma-separated numbers."
            )
            return self.state.step()
        
        # Validate length
        if len(allocation) != self.num_items:
            self.state.set_invalid_move(reason=f"Allocation must have {self.num_items} values, got {len(allocation)}."
            )
            return self.state.step()
        
        # Validate quantities
        item_pool = self.state.game_state["item_pool"]
        for i, (proposed, available) in enumerate(zip(allocation, item_pool)):
            if proposed < 0 or proposed > available:
                self.state.set_invalid_move(reason=f"Invalid quantity for {self.state.game_state['item_names'][i]}: "
                           f"proposed {proposed}, available {available}."
                )
                return self.state.step()
        
        # Validate message doesn't contain player's utility numbers (if verbal mode)
        if self.verbal and message:
            player_utilities = self.state.game_state["player_utilities"][player_id]
            forbidden_numbers = [str(abs(u)) for u in player_utilities]
            
            # Check if any utility number appears in the message
            for num_str in forbidden_numbers:
                if num_str in message:
                    # Player revealed their utility - instant loss!
                    opponent_id = 1 - player_id
                    self.state.set_invalid_move(reason=(
                            f"⚠️  CRITICAL VIOLATION: Your message contained '{num_str}', which is one of your utility values!\n"
                            f"You are NOT allowed to reveal your utility numbers to the opponent.\n"
                            f"Your utility values are: {', '.join(str(u) for u in player_utilities)}\n"
                            f"You have lost the game immediately."
                        )
                    )
                    # Set opponent as winner for revealing secret information
                    self.state.set_winner(
                        player_id=opponent_id,
                        reason=f"Player {player_id} revealed their utility values in a message. Player {opponent_id} wins!"
                    )
                    return self.state.step()
        
        # Record proposal
        self.state.game_state["last_proposal"] = allocation
        self.state.game_state["last_proposer"] = player_id
        self.state.game_state["step_count"] += 1
        
        # Format proposal for display
        item_names = self.state.game_state["item_names"]
        item_pool = self.state.game_state["item_pool"]
        opponent_id = 1 - player_id
        
        # Calculate what each player would get
        proposer_gets = []
        accepter_gets = []
        
        for i, (prop_qty, total_qty, name) in enumerate(zip(allocation, item_pool, item_names)):
            if prop_qty > 0:
                proposer_gets.append(f"{prop_qty}×{name}")
            
            accepter_qty = total_qty - prop_qty
            if accepter_qty > 0:
                accepter_gets.append(f"{accepter_qty}×{name}")
        
        # Format the proposal announcement
        proposer_str = ", ".join(proposer_gets) if proposer_gets else "nothing"
        accepter_str = ", ".join(accepter_gets) if accepter_gets else "nothing"
        
        # Calculate utilities for both players
        proposer_utility, accepter_utility = self._calculate_utilities_for_proposal(allocation, player_id)
        
        # Announce proposal with clear breakdown
        proposal_announcement = (
            f"Player {player_id} proposes:\n"
            f"  • Player {player_id} gets: {proposer_str}\n"
            f"  • Player {opponent_id} gets: {accepter_str}"
        )
        
        if message:
            proposal_announcement += f"\n\nMessage from Player {player_id}: \"{message}\""
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=proposal_announcement
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Send private utility info to proposer
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=player_id,
            message=f"(This proposal would give you {proposer_utility:.1f} utility)",
            observation_type=ObservationType.GAME_MESSAGE,
        )
        
        # Send private utility info to accepter
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=1 - player_id,
            message=f"(This proposal would give you {accepter_utility:.1f} utility)",
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        # Check if max steps reached
        if self.state.game_state["step_count"] >= self.max_steps:
            # Deadlock!
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"Maximum rounds ({self.max_steps}) reached without agreement. Deadlock!",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            # Both players get 0
            self._set_deadlock()
            done, info = self.state.step()
            
            # Add rewards to info dict if they exist in game_state
            if "final_rewards" in self.state.game_state:
                info["rewards"] = self.state.game_state["final_rewards"]
            
            return done, info
        
        # Tell next player they can accept or counter-propose
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=1 - player_id,
            message=f"You can [Accept] this proposal or make a counter-proposal."
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        return self.state.step()
    
    def _calculate_utilities_for_proposal(
        self, allocation: List[int], proposer: int
    ) -> Tuple[float, float]:
        """Calculate what utilities both players would get from this allocation."""
        accepter = 1 - proposer
        item_pool = self.state.game_state["item_pool"]
        utilities = self.state.game_state["player_utilities"]
        
        # Proposer gets the allocation
        proposer_allocation = allocation
        # Accepter gets the remainder
        accepter_allocation = [total - prop for total, prop in zip(item_pool, allocation)]
        
        # Calculate utilities
        proposer_utility = sum(
            alloc * util
            for alloc, util in zip(proposer_allocation, utilities[proposer])
        )
        accepter_utility = sum(
            alloc * util
            for alloc, util in zip(accepter_allocation, utilities[accepter])
        )
        
        return float(proposer_utility), float(accepter_utility)
    
    def _calculate_utilities(
        self, allocation: List[int], proposer: int
    ) -> Tuple[float, float]:
        """Calculate utilities for both players (in player order)."""
        proposer_util, accepter_util = self._calculate_utilities_for_proposal(allocation, proposer)
        
        # Return in player order (not proposer order)
        if proposer == 0:
            return (proposer_util, accepter_util)
        else:
            return (accepter_util, proposer_util)
    
    def _set_outcome_with_utilities(self, utilities: Tuple[float, float]):
        """Set game outcome based on utilities (NON-ZERO-SUM)."""
        # Store utilities in game state for close() to retrieve
        self.state.game_state["final_utilities"] = utilities
        
        # Calculate rewards immediately and store in game_state
        rewards = self._calculate_rewards(utilities)
        self.state.game_state["final_rewards"] = rewards
        
        # Determine who "won" for display purposes
        if utilities[0] > 0 and utilities[1] > 0:
            # Both benefit - determine winner by who got more utility
            if utilities[0] > utilities[1]:
                self.state.set_winner(
                    player_id=0,
                    reason=f"Agreement reached! Both benefit. P0 utility: {utilities[0]:.1f}, P1 utility: {utilities[1]:.1f}"
                )
            elif utilities[1] > utilities[0]:
                self.state.set_winner(
                    player_id=1,
                    reason=f"Agreement reached! Both benefit. P0 utility: {utilities[0]:.1f}, P1 utility: {utilities[1]:.1f}"
                )
            else:
                # Equal utilities - true draw
                self.state.set_draw(
                    reason=f"Agreement reached! Both players gained equal utility: {utilities[0]:.1f}"
                )
        elif utilities[0] > 0:
            self.state.set_winner(
                player_id=0,
                reason=f"Agreement reached! P0 utility: {utilities[0]:.1f}, P1 utility: {utilities[1]:.1f}"
            )
        elif utilities[1] > 0:
            self.state.set_winner(
                player_id=1,
                reason=f"Agreement reached! P0 utility: {utilities[0]:.1f}, P1 utility: {utilities[1]:.1f}"
            )
        else:
            # Both negative (bad deal for both, but better than deadlock)
            self.state.set_draw(
                reason=f"Agreement reached but poor for both. P0 utility: {utilities[0]:.1f}, P1 utility: {utilities[1]:.1f}"
            )
    
    def _set_deadlock(self):
        """Set deadlock outcome (both get 0)."""
        utilities = (0.0, 0.0)
        self.state.game_state["final_utilities"] = utilities
        
        # Calculate rewards immediately and store in game_state
        rewards = self._calculate_rewards(utilities)
        self.state.game_state["final_rewards"] = rewards
        
        self.state.set_draw(
            reason="No agreement reached - deadlock! Both players receive 0 utility."
        )
    
    def _calculate_rewards(self, utilities: Tuple[float, float]) -> Dict[int, float]:
        """
        Calculate rewards for each player based on utilities (NON-ZERO-SUM).
        
        Reward = utility_received / max_possible_utility
        where max_possible_utility = sum of (quantity * utility) for positive utilities only.
        
        Args:
            utilities: Tuple of (player_0_utility, player_1_utility)
            
        Returns:
            Dict mapping player_id to reward value
        """
        player_utilities = self.state.game_state["player_utilities"]
        
        # Calculate max possible utility for each player
        # (sum of positive utilities * max quantity available)
        rewards = {}
        for player_id in [0, 1]:
            max_possible_utility = sum(
                self.quantity_per_item * util 
                for util in player_utilities[player_id] 
                if util > 0
            )
            
            # Error if no positive utilities (would make reward undefined)
            if max_possible_utility == 0:
                raise ValueError(
                    f"Player {player_id} has no positive utility values! "
                    f"Utilities: {player_utilities[player_id]}. "
                    f"Cannot calculate reward as utility_received / max_possible_utility. "
                    f"This indicates a bug in utility generation."
                )
            
            # Calculate reward
            rewards[player_id] = utilities[player_id] / max_possible_utility
        
        return rewards
    
    def close(self) -> Dict[int, float]:
        """
        Return final rewards for each player (NON-ZERO-SUM).
        
        This method is called by TextArena to get the final rewards.
        We override it to return custom non-zero-sum rewards based on utilities.
        """
        # Get utilities from game state
        utilities = self.state.game_state.get("final_utilities", (0.0, 0.0))
        return self._calculate_rewards(utilities)
    
    def _format_game_state_for_player(self, player_id: int) -> str:
        """Format current game state for a player."""
        item_names = self.state.game_state["item_names"]
        item_pool = self.state.game_state["item_pool"]
        utilities = self.state.game_state["player_utilities"][player_id]
        step_count = self.state.game_state["step_count"]
        
        msg = f"Round {step_count + 1}/{self.max_steps}\n"
        msg += f"Available items:\n"
        for i, name in enumerate(item_names):
            msg += f"  - {name}: {item_pool[i]} available (your value: {utilities[i]:+d} per unit)\n"
        
        return msg

