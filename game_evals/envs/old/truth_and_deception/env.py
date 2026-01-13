import importlib.resources
import json
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.TruthAndDeception.renderer import create_board_str


class TruthAndDeceptionEnv(ta.Env):
    """Environment for Truth and Deception Game"""

    def __init__(self, max_turns: Optional[int] = 5, data_path: Optional[str] = None):
        """
        Initialize the Truth and Deception game.

        Roles:
            - Player 0 is the deceiver
            - Player 1 is the guesser

        Args:
            max_turns (int): Maximum number of conversation turns.
            data_path (str): Path to the JSON file containing the facts.
        """
        assert (
            max_turns % 2 == 0
        ), f"Please use an even number of max turns. Current max_turns: {max_turns}"

        self.max_turns = max_turns
        self._load_facts(data_path=data_path)

        self.guess_fact1_pattern = re.compile(r"\[Fact 1\]", re.IGNORECASE)
        self.guess_fact2_pattern = re.compile(r"\[Fact 2\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def _load_facts(self, data_path: Optional[str]) -> None:
        """Load the facts from the specified JSON file.

        Args:
            data_path (str): Path to the JSON file containing the facts.
        """
        try:
            if data_path is not None:
                if not os.path.exists(data_path):
                    raise FileNotFoundError(
                        f"Facts data file not found at: {data_path}"
                    )
                with open(data_path, "r", encoding="utf-8") as file:
                    self.facts_data = json.load(file)
            else:
                with (
                    importlib.resources.files("textarena.envs.TruthAndDeception")
                    .joinpath("facts.json")
                    .open("r") as file
                ):
                    self.facts_data = json.load(file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load facts data: {str(e)}")

    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[int, Any]
    ) -> str:
        """Generate the initial prompt for a player"""
        prompt = f"You are Player {player_id}, and you are the {self.state.role_mapping[player_id]} in the 'Truth and Deception' game.\n"
        if self.state.role_mapping[player_id] == "Deceiver":
            prompt += (
                f"You know which of the following two facts is correct:\n"
                f"Fact 1: {game_state['fact1']['fact']} ({'correct' if game_state['fact1']['is_correct'] else 'wrong'})\n"
                f"Fact 2: {game_state['fact2']['fact']} ({'correct' if game_state['fact2']['is_correct'] else 'wrong'})\n"
                "Your goal is to convince the other player to choose the incorrect fact.\n"
                f"You will have a conversation for {self.state.max_turns - 1} turns.\n"
                "After that, the other player will make their guess.\n"
                "On your turn, simply type your message."
            )
        elif self.state.role_mapping[player_id] == "Guesser":
            prompt += (
                f"You need to determine which of the following two facts is correct:\n"
                f"Fact 1: {game_state['fact1']['fact']}\n"
                f"Fact 2: {game_state['fact2']['fact']}\n"
                f"You will have a conversation with the other player for {self.state.max_turns - 2} turns.\n"
                "After that, you will make your guess.\n"
                "On your turn, simply type your message."
            )
        else:
            raise ValueError(
                f"Unexpected role mapping: {self.state.role_mapping[player_id]}. Expected 'Deceiver' or 'Guesser'."
            )
        return prompt

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the game to its initial state."""
        # Initialize game state
        self.state = ta.State(
            num_players=2,
            min_players=2,
            max_players=2,
            max_turns=self.max_turns,
            check_truncated=False,
            role_mapping={0: "Deceiver", 1: "Guesser"},
        )

        selected_facts = random.choice(self.facts_data)

        facts = [
            (
                selected_facts["facts"]["fact1"],
                selected_facts["correct_fact"] == "fact1",
            ),
            (
                selected_facts["facts"]["fact2"],
                selected_facts["correct_fact"] == "fact2",
            ),
        ]
        random.shuffle(facts)

        self.state.reset(
            seed=seed,
            game_state={
                "fact1": {"fact": facts[0][0], "is_correct": facts[0][1]},
                "fact2": {"fact": facts[1][0], "is_correct": facts[1][1]},
                "correct_fact": facts[0][0] if facts[0][1] else facts[1][0],
                "wrong_fact": facts[0][0] if facts[1][1] else facts[1][0],
            },
            player_prompt_function=self._generate_player_prompt,
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process the player's action"""

        conversation_turns = self.state.max_turns - 2

        # Normal conversation phase
        if self.state.turn < conversation_turns:
            self.state.add_observation(
                from_id=self.state.current_player_id,
                to_id=-1,
                message=action,
                for_logging=True,
            )
            return self.state.step()

        # Transition to guessing phase after conversation turns
        elif (
            self.state.turn == conversation_turns and self.state.current_player_id == 0
        ):
            # Add Deceiver's final conversation message
            self.state.add_observation(
                from_id=self.state.current_player_id,
                to_id=-1,
                message=action,
                for_logging=True,
            )

            # Normal step to rotate to Guesser
            done, info = self.state.step()

            # Show guessing prompt to Guesser
            if not done:
                message = "Now guess which of the two facts are correct by returning '[Fact 1]' or '[Fact 2]'."
                self.state.add_observation(
                    from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=True
                )

            return done, info

        elif self.state.turn == conversation_turns + 1:
            if self.state.current_player_id != 1:
                self.state.set_invalid_move(
                    player_id=self.state.current_player_id,
                    reason="Only the Guesser can make the final guess.",
                )
                return self.state.step()

            self.state.add_observation(
                from_id=self.state.current_player_id,
                to_id=-1,
                message=action,
                for_logging=True,
            )

            if self.guess_fact1_pattern.search(
                action
            ) or self.guess_fact2_pattern.search(action):
                # evaluate guess
                if (
                    self.guess_fact1_pattern.search(action)
                    and self.state.game_state["fact1"]["is_correct"]
                ) or (
                    self.guess_fact2_pattern.search(action)
                    and self.state.game_state["fact2"]["is_correct"]
                ):
                    # correct guess
                    winner_ids = [self.state.current_player_id]
                    reason = (
                        f"Player {self.state.current_player_id} guessed correct fact."
                    )
                else:
                    # wrong guess
                    winner_ids = [1 - self.state.current_player_id]
                    reason = (
                        f"Player {self.state.current_player_id} guessed the wrong fact."
                    )

                # set state winner
                self.state.set_winners(player_ids=winner_ids, reason=reason)
            else:
                reason = f"Player {self.state.current_player_id} did not make their guess in the correct format."
                self.state.set_invalid_move(
                    player_id=self.state.current_player_id, reason=reason
                )

            return self.state.step()

        # Should not reach here
        else:
            self.state.add_observation(
                from_id=self.state.current_player_id,
                to_id=-1,
                message=action,
                for_logging=True,
            )
            return self.state.step()
