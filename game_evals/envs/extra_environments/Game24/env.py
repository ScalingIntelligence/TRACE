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

"""24 Game environment for SPIRAL."""

import ast
import itertools
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType


class Game24Env(ta.Env):
    """
    24 Game environment (single player puzzle).
    
    Players receive 4 numbers and must use +, -, *, / to make exactly 24.
    Each number must be used exactly once.
    
    Action format: [expression]
    Example: [8*3-8+3] or [(8-3)*(8-3)]
    
    The game consists of multiple rounds. Player sees history of all
    previous rounds including their attempts and correct solutions.
    """
    
    def __init__(self, num_rounds: int = 5, max_turns: int = 100):
        """
        Initialize Game24 environment.
        
        Args:
            num_rounds: Number of rounds to play (default: 5)
            max_turns: Maximum turns before game ends (default: 100)
        """
        super().__init__()
        self.num_rounds = num_rounds
        self.max_turns = max_turns
        
        # Regex to match expressions in brackets
        self.action_pattern = re.compile(r"\[(.*?)\]")
        
        # Pre-generate some solvable combinations (numbers that can make 24)
        self.solvable_sets = self._generate_solvable_sets()
    
    def _generate_solvable_sets(self) -> List[Tuple[int, ...]]:
        """Generate a list of 4-number combinations that can make 24."""
        # Some known solvable combinations
        solvable = [
            (3, 3, 8, 8),
            (1, 2, 3, 4),
            (2, 3, 4, 5),
            (1, 3, 4, 6),
            (1, 5, 5, 5),
            (2, 2, 6, 6),
            (3, 4, 4, 8),
            (1, 1, 12, 12),
            (2, 4, 6, 8),
            (1, 2, 8, 9),
            (3, 6, 6, 6),
            (4, 4, 5, 5),
            (1, 3, 8, 8),
            (2, 3, 5, 12),
            (1, 4, 5, 6),
            (3, 3, 7, 7),
            (2, 5, 5, 6),
            (1, 6, 6, 8),
            (4, 6, 6, 6),
            (3, 4, 6, 6),
        ]
        return solvable
    
    def reset(self, num_players: int = 1, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            random.seed(seed)
        
        self.state = ta.State(
            num_players=num_players,
            max_turns=self.max_turns,
        )
        
        # Initialize game state
        game_state = {
            "current_round": 0,
            "rounds_won": 0,
            "round_history": [],  # List of round results
            "current_numbers": None,
            "current_solution": None,
        }
        
        self.state.reset(            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Start first round
        self._start_new_round()
    
    def _start_new_round(self):
        """Start a new round with fresh numbers."""
        # Select random solvable set
        numbers = list(random.choice(self.solvable_sets))
        random.shuffle(numbers)  # Shuffle order
        
        self.state.game_state["current_numbers"] = numbers
        
        # Find a solution for reference
        solution = self._find_solution(numbers)
        self.state.game_state["current_solution"] = solution
        
        self.state.game_state["current_round"] += 1
        
        # Show round info
        round_msg = self._format_round_state()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=0,
            message=round_msg
        ,
            observation_type=ObservationType.GAME_MESSAGE)
    
    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate initial instructions for the player."""
        prompt = (
            f"You are playing the 24 Game!\n\n"
            f"OBJECTIVE:\n"
            f"- You'll receive 4 numbers each round\n"
            f"- Use +, -, *, / to make exactly 24\n"
            f"- Each number must be used exactly once\n"
            f"- You can use parentheses for order of operations\n\n"
            f"GAME FORMAT:\n"
            f"- {self.num_rounds} rounds total\n"
            f"- Win a round by finding a correct solution\n"
            f"- After each round, you'll see if you were correct\n"
            f"- If wrong, you'll see what the correct answer was\n"
            f"- Final score is how many rounds you won\n\n"
            f"ACTION FORMAT:\n"
            f"- Write your expression in brackets: [expression]\n"
            f"- Examples:\n"
            f"  - [8*3-8+3] evaluates to 24\n"
            f"  - [(8-3)*(8-3)] uses parentheses\n"
            f"  - [6/(1-3/4)] uses division\n\n"
            f"RULES:\n"
            f"- Use each number exactly once\n"
            f"- Only use +, -, *, / operators\n"
            f"- Parentheses are allowed\n"
            f"- Result must equal exactly 24\n\n"
            f"STRATEGY TIPS:\n"
            f"- Try to make factors of 24: 3×8, 4×6, 2×12\n"
            f"- Look for ways to make intermediate results\n"
            f"- Consider using division creatively\n"
        )
        return prompt
    
    def get_observation(self) -> Tuple[int, str]:
        """Get current player ID and their observation."""
        player_id = self.state.current_player_id
        observation = self.state.get_current_player_observation()
        return player_id, observation
    
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
        
        # Parse the expression
        match = self.action_pattern.search(action)
        if not match:
            self.state.set_invalid_move(
                player_id=player_id,
                reason="Expression must be in brackets like [8*3] or [(8-3)*6]"
            )
            return self.state.step()
        
        expression = match.group(1).strip()
        
        # Validate and evaluate the expression
        is_correct, result_msg, evaluation = self._validate_expression(
            expression, 
            self.state.game_state["current_numbers"]
        )
        
        # Record round result
        round_result = {
            "round": self.state.game_state["current_round"],
            "numbers": self.state.game_state["current_numbers"].copy(),
            "player_expression": expression,
            "is_correct": is_correct,
            "evaluation": evaluation,
            "correct_solution": self.state.game_state["current_solution"],
        }
        self.state.game_state["round_history"].append(round_result)
        
        if is_correct:
            self.state.game_state["rounds_won"] += 1
            feedback = f"✓ CORRECT! {expression} = 24"
        else:
            feedback = f"✗ {result_msg}"
            if evaluation is not None:
                feedback += f" (Your expression evaluated to {evaluation:.4f})"
            feedback += f"\nCorrect solution: {self.state.game_state['current_solution']}"
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=0,
            message=feedback
        ,
            observation_type=ObservationType.GAME_MESSAGE)
        
        # Check if game is over
        if self.state.game_state["current_round"] >= self.num_rounds:
            # Game over!
            rounds_won = self.state.game_state["rounds_won"]
            self.state.set_winners(
                player_ids=[0],
                reason=f"Game complete! You solved {rounds_won}/{self.num_rounds} rounds correctly."
            )
            return self.state.step()
        
        # Start next round
        self._start_new_round()
        
        return self.state.step()
    
    def _validate_expression(
        self, 
        expression: str, 
        numbers: List[int]
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Validate and evaluate a mathematical expression.
        
        Returns:
            (is_correct, message, evaluation)
        """
        try:
            # Extract numbers from expression
            used_numbers = self._extract_numbers(expression)
            
            # Check if correct numbers used
            if sorted(used_numbers) != sorted(numbers):
                return False, f"Must use {numbers} exactly once, you used {used_numbers}", None
            
            # Safely evaluate the expression
            result = self._safe_eval(expression)
            
            # Check if equals 24 (with floating point tolerance)
            if abs(result - 24) < 0.0001:
                return True, "Correct!", 24.0
            else:
                return False, f"Expression evaluates to {result:.4f}, not 24", result
                
        except ZeroDivisionError:
            return False, "Division by zero error", None
        except SyntaxError:
            return False, "Invalid syntax in expression", None
        except ValueError as e:
            return False, str(e), None
        except Exception as e:
            return False, f"Error evaluating expression: {str(e)}", None
    
    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate mathematical expression using AST.
        Only allows basic arithmetic operations.
        """
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Whitelist of allowed node types
        allowed_nodes = (
            ast.Expression,
            ast.Constant,  # Python 3.8+
            ast.Num,       # Python 3.7
            ast.BinOp,
            ast.UnaryOp,
            ast.USub,
            ast.UAdd,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
        )
        
        # Check all nodes are allowed
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Operation not allowed: {type(node).__name__}")
        
        # Evaluate safely
        return float(eval(compile(tree, '<string>', 'eval')))
    
    def _extract_numbers(self, expression: str) -> List[int]:
        """Extract all numbers from the expression."""
        # Find all integer numbers in the expression
        numbers = re.findall(r'\b\d+\b', expression)
        return [int(n) for n in numbers]
    
    def _find_solution(self, numbers: List[int]) -> str:
        """
        Find a solution that makes 24 from the given numbers.
        Uses brute force to try all permutations and operations.
        """
        ops = ['+', '-', '*', '/']
        
        # Try all permutations of numbers
        for perm in itertools.permutations(numbers):
            # Try all combinations of operations
            for op_combo in itertools.product(ops, repeat=3):
                # Try different parenthesizations
                expressions = [
                    f"{perm[0]}{op_combo[0]}{perm[1]}{op_combo[1]}{perm[2]}{op_combo[2]}{perm[3]}",
                    f"({perm[0]}{op_combo[0]}{perm[1]}){op_combo[1]}{perm[2]}{op_combo[2]}{perm[3]}",
                    f"{perm[0]}{op_combo[0]}({perm[1]}{op_combo[1]}{perm[2]}){op_combo[2]}{perm[3]}",
                    f"{perm[0]}{op_combo[0]}{perm[1]}{op_combo[1]}({perm[2]}{op_combo[2]}{perm[3]})",
                    f"({perm[0]}{op_combo[0]}{perm[1]}){op_combo[1]}({perm[2]}{op_combo[2]}{perm[3]})",
                    f"(({perm[0]}{op_combo[0]}{perm[1]}){op_combo[1]}{perm[2]}){op_combo[2]}{perm[3]}",
                    f"({perm[0]}{op_combo[0]}({perm[1]}{op_combo[1]}{perm[2]})){op_combo[2]}{perm[3]}",
                    f"{perm[0]}{op_combo[0]}(({perm[1]}{op_combo[1]}{perm[2]}){op_combo[2]}{perm[3]})",
                    f"{perm[0]}{op_combo[0]}({perm[1]}{op_combo[1]}({perm[2]}{op_combo[2]}{perm[3]}))",
                ]
                
                for expr in expressions:
                    try:
                        result = self._safe_eval(expr)
                        if abs(result - 24) < 0.0001:
                            return expr
                    except:
                        continue
        
        # Fallback (shouldn't happen with pre-vetted sets)
        return "No solution found"
    
    def _format_round_state(self) -> str:
        """Format the current round state with history."""
        current_round = self.state.game_state["current_round"]
        rounds_won = self.state.game_state["rounds_won"]
        numbers = self.state.game_state["current_numbers"]
        history = self.state.game_state["round_history"]
        
        msg = f"=== 24 Game: Round {current_round}/{self.num_rounds} ===\n\n"
        msg += f"SCORE: {rounds_won} rounds won so far\n\n"
        msg += f"YOUR NUMBERS: {numbers}\n"
        msg += f"Make exactly 24 using +, -, *, /\n"
        msg += f"Each number must be used exactly once.\n\n"
        
        # Show history of previous rounds
        if history:
            msg += f"PREVIOUS ROUNDS:\n"
            for result in history:
                status = "✓" if result["is_correct"] else "✗"
                msg += f"  Round {result['round']}: {result['numbers']}\n"
                msg += f"    Your answer: {result['player_expression']} {status}\n"
                if not result["is_correct"]:
                    msg += f"    Correct answer: {result['correct_solution']}\n"
            msg += "\n"
        
        msg += "Submit your answer in brackets: [your_expression]"
        
        return msg

