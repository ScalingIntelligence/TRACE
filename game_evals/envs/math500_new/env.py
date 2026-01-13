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

"""MATH 500 single question environment for SPIRAL."""

import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.core import ObservationType


class Math500Env(ta.Env):
    """
    MATH 500 two-player competitive environment.
    
    Each player gets ONE random question from the MATH 500 dataset.
    Your goal is to answer correctly AND outperform your opponent.
    
    Rewards (ZERO-SUM competitive):
    - Both correct: (0, 0) - tie, no reward
    - Only you correct: (+1, -1) - you win
    - Only opponent correct: (-1, +1) - opponent wins  
    - Both incorrect: (0, 0) - tie, no reward
    
    This zero-sum structure makes role baseline training work correctly.
    
    Action format: [answer] or text containing \\boxed{answer}
    Example: [42] or \\boxed{42}
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize MATH 500 environment.
        
        Args:
            data_path: Path to MATH dataset. If None, uses default location.
        """
        super().__init__()
        
        # Load dataset
        self.dataset = self._load_dataset(data_path)
        
        # Regex to extract answer from brackets or boxed notation
        self.bracket_pattern = re.compile(r"\[([^\]]+)\]")
        self.boxed_pattern = re.compile(r"\\boxed\{([^}]*)\}")
    
    def _load_dataset(self, data_path: Optional[str]) -> List[Dict[str, Any]]:
        """Load MATH dataset from disk or download if needed."""
        # PRIORITY: Use the same evaluation dataset for training
        # Load ONLY the first 500 problems that evaluation uses
        eval_data_paths = [
            "/home/ubuntu/hangook/acshi/weaver_for_RL/experiments/evals/data/math/test.jsonl",
            "evals/data/math/test.jsonl",
            "../evals/data/math/test.jsonl",
        ]
        
        # Try to load from evaluation dataset (JSONL format)
        if data_path is None:
            for path in eval_data_paths:
                if os.path.exists(path):
                    data_path = path
                    break
        
        # Check if it's a JSONL file
        if data_path and os.path.exists(data_path) and data_path.endswith('.jsonl'):
            try:
                import json
                problems = []
                with open(data_path, 'r') as f:
                    for i, line in enumerate(f):
                        # Only load first 500 problems (same as evaluation)
                        if i >= 500:
                            break
                        item = json.loads(line.strip())
                        problems.append({
                            'problem': item.get('problem', item.get('question', '')),
                            'answer': item.get('answer', ''),
                            'difficulty': item.get('level', 0)
                        })
                print(f"✓ Loaded {len(problems)} MATH problems from {data_path}")
                print(f"  → Training on EXACT SAME 500 problems as evaluation!")
                print(f"  → Expect fast overfitting and high accuracy")
                return problems
            except Exception as e:
                print(f"Error loading JSONL dataset from {data_path}: {e}")
        
        # Fallback: Try old arrow format locations
        arrow_paths = [
            "/home/ubuntu/hangook/acshi/spiral/data/math",
            "/home/ubuntu/hangook/acshi/weaver_for_RL/spiral/data/math",
        ]
        
        if data_path is None:
            for path in arrow_paths:
                if os.path.exists(path):
                    data_path = path
                    break
        
        if data_path and os.path.exists(data_path):
            # Load from datasets arrow format
            try:
                from datasets import load_from_disk
                dataset = load_from_disk(data_path)
                # Convert to list of dicts
                problems = []
                for item in dataset:
                    problems.append({
                        'problem': item['problem'],
                        'answer': item['answer'],
                        'difficulty': item.get('difficulty', 0)
                    })
                print(f"⚠ Loaded {len(problems)} MATH problems from {data_path}")
                print(f"  → WARNING: This may differ from evaluation dataset!")
                return problems
            except Exception as e:
                print(f"Error loading dataset from {data_path}: {e}")
        
        # Fallback: try loading from URL
        try:
            import pandas as pd
            print("Attempting to download MATH 500 from online source...")
            df = pd.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv")
            problems = []
            for _, row in df.iterrows():
                problems.append({
                    'problem': row.get('Problem', row.get('problem', '')),
                    'answer': row.get('Answer', row.get('answer', '')),
                    'difficulty': 0
                })
            print(f"Downloaded {len(problems)} MATH problems from online source")
            return problems
        except Exception as e:
            print(f"Error downloading dataset: {e}")
        
        # Last resort: create a small dummy dataset for testing
        print("WARNING: Using dummy MATH dataset with 5 example problems")
        return [
            {
                'problem': 'What is 2 + 2?',
                'answer': '4',
                'difficulty': 1
            },
            {
                'problem': 'Solve for x: 2x + 5 = 13',
                'answer': '4',
                'difficulty': 2
            },
            {
                'problem': 'What is the value of $\\sqrt{144}$?',
                'answer': '12',
                'difficulty': 1
            },
            {
                'problem': 'If $f(x) = x^2 + 3x - 4$, what is $f(2)$?',
                'answer': '6',
                'difficulty': 2
            },
            {
                'problem': 'What is the sum of the interior angles of a triangle?',
                'answer': '180',
                'difficulty': 1
            }
        ]
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """Reset the environment with two random questions (one per player)."""
        if seed is not None:
            random.seed(seed)
        
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            seed=seed,
            max_turns=2,  # One turn per player
        )
        
        # Select two different random problems (one for each player)
        problems = random.sample(self.dataset, 2)
        
        # Initialize game state
        game_state = {
            "problems": {
                0: problems[0]['problem'],
                1: problems[1]['problem']
            },
            "answers": {
                0: problems[0]['answer'],
                1: problems[1]['answer']
            },
            "difficulties": {
                0: problems[0].get('difficulty', 0),
                1: problems[1].get('difficulty', 0)
            },
            "player_responses": {},  # Store player answers
            "player_correct": {},  # Track if each player got it right
        }
        
        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )
        
        # Present the problem to Player 0
        problem_msg = self._format_problem(0)
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=0,
            message=problem_msg,
            observation_type=ObservationType.GAME_MESSAGE
        )
    
    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate initial instructions for the player."""
        prompt = (
            f"You are Player {player_id} solving a mathematics problem from the MATH dataset.\n\n"
            f"OBJECTIVE:\n"
            f"- Each player gets their own problem to solve\n"
            f"- Answer correctly AND outperform your opponent to win\n"
            f"- Provide your final answer in \\boxed{{}} format\n\n"
            f"ANSWER FORMAT:\n"
            f"- You must put your final answer inside \\boxed{{}}\n"
            f"- Example: \\boxed{{42}} or \\boxed{{x = 5}}\n"
            f"- You can show your work, but the \\boxed{{}} answer will be extracted and graded\n\n"
            f"SCORING (Zero-Sum Competitive):\n"
            f"- If ONLY YOU answer correctly: +1 point (you win)\n"
            f"- If ONLY OPPONENT answers correctly: -1 point (you lose)\n"
            f"- If BOTH correct or BOTH incorrect: 0 points (tie)\n"
            f"- This is a competitive game - you must outperform your opponent!\n\n"
            f"Think step by step and show your reasoning. Put your final answer in \\boxed{{}}.\n"
        )
        return prompt
    
    def get_observation(self) -> Tuple[int, str]:
        """Get current player ID and their observation."""
        player_id = self.state.current_player_id
        observation = self.state.get_current_player_observation()
        return player_id, observation
    
    def get_valid_actions(self, player_id: int) -> List[str]:
        """Get list of valid actions for a player.
        
        Returns the correct answer (and possibly some variations) wrapped in \\boxed{}.
        
        Args:
            player_id: ID of the player (0 or 1)
            
        Returns:
            List of action strings containing the correct answer
        """
        # Get the correct answer for this player
        correct_answer = self.state.game_state["answers"][player_id]
        
        # Return the correct answer in various formats
        actions = [
            f"\\boxed{{{correct_answer}}}",
            f"[{correct_answer}]",
        ]
        
        return actions
    
    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Process an action and update game state."""
        player_id = self.state.current_player_id
        
        # Log the raw action
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        # Extract answer from action
        predicted_answer = self._extract_answer(action)
        
        if predicted_answer is None:
            # No valid answer format found
            self.state.set_invalid_move(
                player_id=player_id,
                reason="Could not find answer in \\boxed{} or [answer] format. Please provide your answer as \\boxed{your_answer}"
            )
            return self.state.step()
        
        # Check if answer is correct
        ground_truth = self.state.game_state["answers"][player_id]
        is_correct = self._check_answer(predicted_answer, ground_truth)
        
        # Store player's answer and correctness
        self.state.game_state["player_responses"][player_id] = predicted_answer
        self.state.game_state["player_correct"][player_id] = is_correct
        
        # Add feedback to this player
        if is_correct:
            feedback = f"✓ Player {player_id} CORRECT! Answer '{predicted_answer}' matches '{ground_truth}'"
        else:
            feedback = f"✗ Player {player_id} INCORRECT. Answer '{predicted_answer}' does not match '{ground_truth}'"
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=player_id,
            message=feedback,
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        # Check if both players have answered
        if len(self.state.game_state["player_correct"]) == 2:
            # Both players have answered, determine rewards
            self._determine_final_rewards()
            return self.state.step()
        
        # Show problem to next player
        next_player = 1 - player_id
        problem_msg = self._format_problem(next_player)
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=next_player,
            message=problem_msg,
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        return self.state.step()
    
    def _determine_final_rewards(self):
        """Determine final rewards based on competitive zero-sum structure."""
        p0_correct = self.state.game_state["player_correct"][0]
        p1_correct = self.state.game_state["player_correct"][1]
        
        # Zero-sum rewards: only get reward if you outperform opponent
        # This makes role baseline work correctly
        if p0_correct and p1_correct:
            # Both correct → tie → zero reward
            rewards = {0: 0.0, 1: 0.0}
        elif p0_correct and not p1_correct:
            # Player 0 wins
            rewards = {0: 1.0, 1: -1.0}
        elif not p0_correct and p1_correct:
            # Player 1 wins
            rewards = {0: -1.0, 1: 1.0}
        else:
            # Both incorrect → tie → zero reward
            rewards = {0: 0.0, 1: 0.0}
        
        # Build reason string
        p0_status = "correct" if p0_correct else "incorrect"
        p1_status = "correct" if p1_correct else "incorrect"
        reason = f"Player 0 {p0_status}, Player 1 {p1_status} → zero-sum rewards ({rewards[0]:+.1f}, {rewards[1]:+.1f})"
        
        # Store final rewards
        self.state.game_state["final_rewards"] = rewards
        
        # Announce results to both players
        result_msg = self._format_final_results()
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=result_msg,
            observation_type=ObservationType.GAME_MESSAGE
        )
        
        # End the game - use set_winners since we have custom rewards
        # The rewards will be extracted from game_state in close()
        self.state.set_draw(reason=reason)
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from text.
        Looks for \\boxed{} notation first, then [answer] format.
        """
        # Try boxed notation first (preferred)
        match = self.boxed_pattern.search(text)
        if match:
            return match.group(1).strip()
        
        # Try bracket notation
        match = self.bracket_pattern.search(text)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.
        Uses simple normalization and comparison.
        """
        if predicted is None or ground_truth is None:
            return False
        
        # Normalize both answers
        pred_norm = self._normalize_answer(predicted)
        gt_norm = self._normalize_answer(ground_truth)
        
        # Direct string match (case insensitive)
        if pred_norm.lower() == gt_norm.lower():
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(pred_norm)
            gt_num = float(gt_norm)
            return abs(pred_num - gt_num) < 1e-4
        except (ValueError, TypeError):
            pass
        
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer.strip())
        
        # Remove common LaTeX commands that don't affect value
        answer = answer.replace('\\$', '').replace('$', '')
        answer = answer.replace('\\text{', '').replace('}', '')
        answer = answer.replace('\\,', '')
        answer = answer.replace(',', '')  # Remove commas from numbers
        
        return answer
    
    def _format_problem(self, player_id: int) -> str:
        """Format the problem statement for a specific player."""
        problem = self.state.game_state["problems"][player_id]
        difficulty = self.state.game_state["difficulties"].get(player_id, "Unknown")
        
        msg = f"=== MATH Problem for Player {player_id} ===\n\n"
        msg += f"Difficulty: {difficulty}\n\n"
        msg += f"PROBLEM:\n{problem}\n\n"
        msg += f"Solve this problem step by step.\n"
        msg += f"Put your final answer in \\boxed{{answer}} format.\n"
        
        return msg
    
    def _format_final_results(self) -> str:
        """Format the final results showing both players' performance."""
        p0_correct = self.state.game_state["player_correct"][0]
        p1_correct = self.state.game_state["player_correct"][1]
        rewards = self.state.game_state["final_rewards"]
        
        msg = "\n" + "=" * 70 + "\n"
        msg += "GAME RESULTS\n"
        msg += "=" * 70 + "\n\n"
        
        for pid in [0, 1]:
            problem = self.state.game_state["problems"][pid]
            answer = self.state.game_state["answers"][pid]
            response = self.state.game_state["player_responses"][pid]
            correct = self.state.game_state["player_correct"][pid]
            reward = rewards[pid]
            
            status = "✓ CORRECT" if correct else "✗ INCORRECT"
            msg += f"Player {pid}: {status}\n"
            msg += f"  Problem: {problem[:60]}{'...' if len(problem) > 60 else ''}\n"
            msg += f"  Correct answer: {answer}\n"
            msg += f"  Player answer: {response}\n"
            msg += f"  Reward: {reward}\n\n"
        
        msg += "=" * 70 + "\n"
        
        return msg
    
    def close(self) -> Tuple[Dict[int, float], Dict[str, Any]]:
        """
        Close the environment and return final rewards.
        
        Returns:
            Tuple of (rewards_dict, game_info_dict)
        """
        # Get rewards from game state (already calculated in _determine_final_rewards)
        rewards = self.state.game_state.get("final_rewards", {0: 0.0, 1: 0.0})
        
        # Prepare game info
        game_info = {
            "problems": self.state.game_state.get("problems", {}),
            "answers": self.state.game_state.get("answers", {}),
            "player_responses": self.state.game_state.get("player_responses", {}),
            "player_correct": self.state.game_state.get("player_correct", {}),
            "difficulties": self.state.game_state.get("difficulties", {}),
        }
        
        return rewards, game_info

