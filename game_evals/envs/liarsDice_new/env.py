import random
import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType
from textarena.envs.LiarsDice.renderer import create_board_str


class LiarsDiceEnv(ta.Env):
    """
    Two-player version of Liar's Dice - SINGLE ROUND.
    
    Each player starts with dice and can see only their own dice.
    Players take turns bidding on the TOTAL number of dice showing a specific face
    value ACROSS BOTH PLAYERS' DICE COMBINED.
    
    When someone calls:
    - All dice are revealed
    - If the bid was WRONG (actual < bid): The bidder LOSES, caller WINS
    - If the bid was CORRECT (actual >= bid): The caller LOSES, bidder WINS
    
    Game ends immediately after the first call!
    """

    def __init__(self, num_dice: int = 5):
        """
        Initialize the Liar's Dice game environment.

        Args:
            num_dice (int): Initial number of dice each player starts with.
        """
        self.initial_num_dice = num_dice

        self.bid_pattern = re.compile(r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]", re.IGNORECASE)
        self.call_pattern = re.compile(r"\[call\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the Liar's Dice game to its initial state"""
        self.state = ta.TwoPlayerState(num_players=num_players,
            seed=seed
        )

        remaining_dice = {
            pid: self.initial_num_dice for pid in range(self.state.num_players)
        }
        dice_rolls = {
            pid: [random.randint(1, 6) for _ in range(self.initial_num_dice)]
            for pid in range(self.state.num_players)
        }

        # Keep track of the last bidder so we know who wins/loses on a call
        # We'll store None when no bids have yet been made
        game_state = {
            "current_bid": {"quantity": 0, "face_value": 0},
            "last_bidder_id": None,
            "remaining_dice": remaining_dice,
            "dice_rolls": dice_rolls,
            "active_players": list(range(num_players)),
        }

        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )

        self._ensure_active_player()

    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate the prompt for a given player, showing their dice and opponent's dice count"""
        my_dice = game_state["dice_rolls"][player_id]
        # Build a listing of opponent's dice count
        others_info = []
        total_dice_count = 0
        for pid in range(self.state.num_players):
            if game_state["remaining_dice"][pid] > 0:  # Only show active players
                count = game_state["remaining_dice"][pid]
                total_dice_count += count
                if pid != player_id:
                    others_info.append(f"Player {pid} has {count} dice (hidden from you)")

        others_text = "\n".join(others_info)

        current_quantity = game_state["current_bid"]["quantity"]
        current_face_value = game_state["current_bid"]["face_value"]
        last_bidder_id = game_state["last_bidder_id"]
        
        # Check if calling is allowed (only when there's an existing bid this round)
        can_call = last_bidder_id is not None and current_quantity > 0

        prompt = (
            f"You are Player {player_id} in a TWO-PLAYER Liar's Dice game.\n\n"
            f"═══════════════════════════════════════════════════════════\n"
            f"YOUR DICE (only you can see these): {', '.join(map(str, my_dice))}\n"
            f"═══════════════════════════════════════════════════════════\n\n"
            f"TOTAL DICE IN PLAY (both players combined): {total_dice_count} dice\n"
            f"{others_text}\n\n"
            f"═══════════════════════════════════════════════════════════\n"
            f"CRITICAL: BIDS COUNT ALL DICE FROM BOTH PLAYERS COMBINED!\n"
            f"═══════════════════════════════════════════════════════════\n"
            f"When you bid \"3 fours\", you're claiming there are AT LEAST\n"
            f"3 dice showing '4' across YOUR dice + YOUR OPPONENT'S dice.\n\n"
            "GAME RULES:\n"
            "- On your turn, you can either:\n"
            "  [1] Make a NEW BID: Must be higher quantity OR higher face value\n"
            "      Format: [Bid: quantity, face_value]\n"
            "      Example: [Bid: 3, 4] means \"at least 3 fours across all dice\"\n"
            "  [2] CALL the last bid: Type [Call]\n"
            "      NOTE: You can ONLY call if someone has already made a bid!\n\n"
            "WHAT HAPPENS WHEN YOU CALL:\n"
            "- ALL dice from BOTH players are revealed and counted\n"
            "- If the actual count is LESS than the bid → last bidder LOSES, you WIN!\n"
            "- If the actual count is >= the bid → you (the caller) LOSE, bidder WINS!\n"
            "- GAME ENDS IMMEDIATELY after the call - this is a single-round game!\n\n"
        )
        
        # Show current bid status with clear indication of whether calling is allowed
        if can_call:
            prompt += (
                f"CURRENT BID: {current_quantity} of face value {current_face_value}\n"
                f"(This means: \"At least {current_quantity} dice showing '{current_face_value}' across all {total_dice_count} dice\")\n\n"
                "Your action? (e.g. '[Bid: 4, 5]' or '[Call]')"
            )
        else:
            prompt += (
                f"═══════════════════════════════════════════════════════════\n"
                f"NO BID YET - YOU MUST MAKE A BID!\n"
                f"CALLING IS NOT ALLOWED until someone makes a bid.\n"
                f"═══════════════════════════════════════════════════════════\n\n"
                "Your action? (e.g. '[Bid: 3, 4]')"
            )
        
        return prompt

    def _ensure_active_player(self):
        """Ensure current player is active, skip to next active player if not"""
        active_players = self.state.game_state["active_players"]
        if self.state.current_player_id not in active_players:
            # Find next active player
            current = self.state.current_player_id
            for _ in range(self.state.num_players):
                current = (current + 1) % self.state.num_players
                if current in active_players:
                    self.state.manually_set_current_player_id(current)
                    break


    def get_observation(self) -> Tuple[int, str]:
        """Get current player ID and their observation."""
        player_id = self.state.current_player_id
        observation = self.state.get_current_player_observation()
        return player_id, observation
    
    def get_valid_actions(self, player_id: int) -> list[str]:
        """Get list of valid actions for a player.
        
        Args:
            player_id: ID of the player (0 or 1)
            
        Returns:
            List of valid action strings: ["[Bid: quantity, face]", ..., "[Call]"]
        """
        actions = []
        
        # Get current bid and total dice count
        current_bid = self.state.game_state["current_bid"]
        curr_quantity = current_bid["quantity"]
        curr_face = current_bid["face_value"]
        last_bidder_id = self.state.game_state["last_bidder_id"]
        
        # Can only call if someone has made a bid this round
        if last_bidder_id is not None and curr_quantity > 0:
            actions.append("[Call]")
        
        # Count total dice in play
        remaining_dice = self.state.game_state["remaining_dice"]
        total_dice = sum(remaining_dice.values())
        
        # Generate ALL valid higher bids
        # Same quantity, higher face value
        if curr_face < 6:
            for f in range(curr_face + 1, 7):
                actions.append(f"[Bid: {curr_quantity}, {f}]")
        
        # Higher quantity with any face value
        for q in range(curr_quantity + 1, total_dice + 1):
            for f in range(1, 7):
                actions.append(f"[Bid: {q}, {f}]")
        
        return actions

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process one action from the current player"""
        player_id = self.state.current_player_id

        if player_id not in self.state.game_state["active_players"]:
            self._ensure_active_player()
            return self.state.step()

        # Log the action for the record
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            observation_type=ObservationType.GAME_MESSAGE
        )

        # 1. Check if action is '[Call]'
        if self.call_pattern.search(action):
            current_bid = self.state.game_state["current_bid"]
            last_bidder_id = self.state.game_state["last_bidder_id"]

            if last_bidder_id is None or current_bid["quantity"] == 0:
                # No existing bid to call - this is invalid
                self.state.set_invalid_move(
                    reason="INVALID: Cannot call when no bid has been made yet. You must make a bid first!"
                )
                return self.state.step()

            quantity = current_bid["quantity"]
            face_value = current_bid["face_value"]
            # Count how many dice across all players match face_value
            total_face_count = 0
            for pid, dice_list in self.state.game_state["dice_rolls"].items():
                if (
                    self.state.game_state["remaining_dice"][pid] > 0
                ):  # Only count dice from active players
                    total_face_count += dice_list.count(face_value)

            if total_face_count < quantity:
                loser_id = last_bidder_id
                winner_id = player_id
                msg = (
                    f"Player {player_id} calls!\n"
                    f"Bid was: {quantity} of face value {face_value}\n"
                    f"Actual count: {total_face_count}\n\n"
                    f"The bid was WRONG (actual < bid)!\n"
                    f"Player {winner_id} (the caller) WINS!"
                )
            else:
                # Otherwise, the caller loses
                loser_id = player_id
                winner_id = last_bidder_id
                msg = (
                    f"Player {player_id} calls!\n"
                    f"Bid was: {quantity} of face value {face_value}\n"
                    f"Actual count: {total_face_count}\n\n"
                    f"The bid was CORRECT (actual >= bid)!\n"
                    f"Player {winner_id} (the bidder) WINS!"
                )

            self._end_game_with_winner(winner_id, loser_id, msg)
            # Call step() to properly complete the state transition
            return self.state.step()

        # 2. Otherwise, check if it is a valid '[Bid: X, Y]'
        bid_match = self.bid_pattern.search(action)
        if bid_match:
            try:
                new_quantity = int(bid_match.group(1))
                new_face_value = int(bid_match.group(2))
            except ValueError:
                self.state.set_invalid_move(
                    reason="Bid values must be valid integers."
                )
                return self.state.step()

            current_bid = self.state.game_state["current_bid"]
            # Validate it is strictly higher
            is_valid, reason = self._is_valid_bid(
                new_quantity, new_face_value, current_bid
            )
            if is_valid:
                self.state.game_state["current_bid"] = {
                    "quantity": new_quantity,
                    "face_value": new_face_value,
                }
                self.state.game_state["last_bidder_id"] = player_id
                message = (
                    f"Player {player_id} bids {new_quantity} of face {new_face_value}."
                )
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=message,
                    observation_type=ObservationType.GAME_MESSAGE
                )

                done, info = self.state.step()
                if not done:
                    self._ensure_active_player()
                    next_player_id = self.state.current_player_id
                    bid_info = (
                        f"Current bid: {new_quantity} of face {new_face_value}\n"
                        f"Your turn, Player {next_player_id}. Make a higher bid with '[Bid: X, Y]' or challenge with '[Call]'."
                    )
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=next_player_id,
                        message=bid_info,
                        observation_type=ObservationType.GAME_MESSAGE
                    )
                return done, info
            else:
                self.state.set_invalid_move(
                    reason=f"Invalid bid: {reason}"
                )
                return self.state.step()

        # 3. If neither a valid call nor bid, it's invalid
        reason = "Action not recognized as either a valid '[Bid: X, Y]' or '[Call]'."
        self.state.set_invalid_move(reason=reason)
        return self.state.step()

    def _end_game_with_winner(self, winner_id: int, loser_id: int, message: str):
        """
        End the game immediately with a winner after a call is made.
        Reveals all dice and declares the winner.
        """
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=message,
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        # Reveal all dice to show the final count
        all_dice_msg = "\n\n═══════════════════════════════════════\n"
        all_dice_msg += "DICE REVEAL:\n"
        all_dice_msg += "═══════════════════════════════════════\n"
        for pid in range(self.state.num_players):
            if pid in self.state.game_state["dice_rolls"]:
                dice = self.state.game_state["dice_rolls"][pid]
                all_dice_msg += f"Player {pid}: {', '.join(map(str, dice))}\n"
        all_dice_msg += "═══════════════════════════════════════\n"
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=all_dice_msg,
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        # Declare winner and end the game
        reason = f"Player {winner_id} wins the call!"
        self.state.set_winner(player_id=winner_id, reason=reason)

    def _is_valid_bid(
        self, new_quantity: int, new_face_value: int, current_bid: Dict[str, int]
    ) -> Tuple[bool, str]:
        """
        Check if the new bid is strictly higher than the current bid,
        and if face_value is between 1 and 6.
        """
        old_quantity = current_bid["quantity"]
        old_face_value = current_bid["face_value"]

        if new_quantity <= 0:
            return False, "Quantity must be positive."
        if not (1 <= new_face_value <= 6):
            return False, "Face value must be between 1 and 6."

        if new_quantity < old_quantity:
            return False, (
                f"New quantity {new_quantity} is lower than current {old_quantity}."
            )
        if new_quantity == old_quantity and new_face_value < old_face_value:
            return False, (
                f"With same quantity, face value must be higher than current {old_face_value}."
            )
        if new_quantity == old_quantity and new_face_value == old_face_value:
            return False, "Bid is identical to the current bid."

        total_dice = sum(
            self.state.game_state["remaining_dice"][pid]
            for pid in self.state.game_state["active_players"]
        )
        if new_quantity > total_dice:
            return (
                False,
                f"Bid quantity {new_quantity} exceeds total dice count {total_dice}.",
            )

        return True, ""
