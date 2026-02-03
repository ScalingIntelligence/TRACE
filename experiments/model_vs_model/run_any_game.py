#!/usr/bin/env python3
"""
Run model-vs-model evaluation for ANY registered game.

Usage:
    python run_any_game.py --game liars_dice_memory --num_games 5 --verbose
    python run_any_game.py --game liars_dice --vllm_url http://localhost:8076 --num_games 10
    python run_any_game.py --game dependency_resolution --num_games 100 --parallel 20
"""
import sys
from pathlib import Path

# Add games directory to path
GAMES_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GAMES_PATH))

import argparse
import json
import random
import requests
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, TextIO
from transformers import AutoTokenizer

from config import Config
from game_registry import get_game_spec, list_game_names
from liars_dice_tools import extract_tool_call

_TOOL_CALL_GAMES = {
    "liars_dice_tool",
    "liars_dice_memory_tool",
    "liars_dice_memory_updated_tool",
}

class VLLMBackend:
    """Backend for calling vLLM server using /completions endpoint (like training code)."""

    def __init__(self, base_url: str, model_name: str = "default", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()

        # Auto-detect model name from server
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=10)
            if r.status_code == 200:
                models = r.json().get("data", [])
                if models:
                    self.model_name = models[0].get("id", model_name)
        except Exception:
            pass

        # Load tokenizer for prompt formatting (same as training code)
        print(f"Loading tokenizer from {Config.MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    def _format_prompt(self, messages: List[Dict]) -> str:
        """Format messages using tokenizer's chat template with thinking enabled."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=Config.ENABLE_THINKING,
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
            return self.tokenizer.decode(ids, skip_special_tokens=False)

    def generate(self, messages: List[Dict], temperature: float, max_new_tokens: int) -> str:
        # Format prompt locally (with thinking enabled), then use /completions
        prompt = self._format_prompt(messages)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        }

        r = self.session.post(
            f"{self.base_url}/completions",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["text"]


def play_game(
    game_name: str,
    backend: VLLMBackend,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    verbose: bool = False,
    log_file: Optional[TextIO] = None,
) -> Dict:
    """Play a single game with the model against itself.

    Args:
        log_file: Optional file handle to write verbose logs to instead of stdout.
                  When provided, verbose output goes to this file instead of print().
    """

    def log(msg: str = ""):
        """Write to log file if provided, otherwise print."""
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()
        elif verbose:
            print(msg)

    game_spec = get_game_spec(game_name)
    env = game_spec.make_env()
    env.reset(seed)

    game_log = []
    invalid_tool_calls = 0
    tool_enabled = game_spec.name in _TOOL_CALL_GAMES

    if verbose or log_file:
        log(f"\n    --- Game (seed={seed}) ---")
        log(f"\n--- SYSTEM PROMPT ---")
        log(game_spec.system_prompt)

    while not env.done:
        player = env.current_player
        observation = env.observe(player)
        legal_actions = env.legal_actions()

        messages = [
            {"role": "system", "content": game_spec.system_prompt},
            {"role": "user", "content": observation},
        ]

        raw_output = backend.generate(
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        tool_call = extract_tool_call(raw_output) if tool_enabled else None
        action = game_spec.extract_action(raw_output, legal_actions)

        if action is None:
            action = random.choice(legal_actions)
            if tool_enabled:
                action_status = "INVALID_TOOL" if tool_call is not None else "PARSE_FAIL"
            else:
                action_status = "INVALID"
        else:
            action_status = "OK"

        if tool_call is not None and action_status == "INVALID_TOOL":
            invalid_tool_calls += 1

        if verbose or log_file:
            log(f"\n{'='*70}")
            log(f"Player {player}'s turn")
            log(f"{'='*70}")
            log(f"\n--- OBSERVATION (what player sees) ---")
            log(observation)
            log(f"\n--- LEGAL ACTIONS ({len(legal_actions)} total) ---")
            log(str(legal_actions[:8]))
            if len(legal_actions) > 8:
                log(f"... and {len(legal_actions) - 8} more")
            log(f"\n--- MODEL OUTPUT ---")
            log(raw_output)
            if tool_call is not None:
                log(f"\n--- TOOL CALL (parsed JSON) ---")
                log(json.dumps(tool_call, indent=2, sort_keys=True))
            log(f"\n--- EXTRACTED ACTION: {action} ({action_status}) ---")

        game_log.append({
            "player": player,
            "legal_actions": legal_actions,
            "raw_output": raw_output,
            "tool_call": tool_call,
            "action": action,
            "action_status": action_status,
        })

        env.step(action)

    # Determine winner
    winner = None
    if env.rewards.get(0, 0) > env.rewards.get(1, 0):
        winner = 0
    elif env.rewards.get(1, 0) > env.rewards.get(0, 0):
        winner = 1

    if verbose or log_file:
        if env.invalid_player is not None:
            log(f"    Result: Player {env.invalid_player} made INVALID move -> Player {1 - env.invalid_player} WINS")
        elif winner is not None:
            log(f"    Result: Player {winner} WINS")
        else:
            log(f"    Result: DRAW")

        # For dependency resolution game, log summary info
        if hasattr(env, 'get_summary'):
            summary = env.get_summary()
            log(f"\n--- GAME SUMMARY ---")
            log(f"Formula: {summary.get('formula', 'N/A')}")
            log(f"Correct answer: {summary.get('correct_answer', 'N/A')}")
            log(f"Submitted answer: {summary.get('submitted_answer', 'N/A')}")
            log(f"Required info: {summary.get('required_info', [])}")
            log(f"Acquired info: {summary.get('acquired_info', [])}")
            log(f"Sufficiency turn: {summary.get('sufficiency_turn', 'N/A')}")
            log(f"Total turns: {summary.get('total_turns', 'N/A')}")
            log(f"Reward: {summary.get('reward', 'N/A')}")

    result = {
        "seed": seed,
        "winner": winner,
        "rewards": env.rewards,
        "invalid_player": env.invalid_player,
        "invalid_tool_calls": invalid_tool_calls,
        "game_log": game_log,
    }

    # Include game-specific summary if available
    if hasattr(env, 'get_summary'):
        result['game_summary'] = env.get_summary()

    return result


def play_game_wrapper(args_tuple):
    """Wrapper for parallel execution - unpacks arguments."""
    game_name, backend, temperature, max_new_tokens, seed, verbose, log_dir = args_tuple
    try:
        log_file = None
        if log_dir:
            log_path = os.path.join(log_dir, f"game_seed_{seed}.log")
            log_file = open(log_path, 'w')
        try:
            result = play_game(game_name, backend, temperature, max_new_tokens, seed, verbose, log_file)
            if log_dir:
                result['log_file'] = log_path
            return result
        finally:
            if log_file:
                log_file.close()
    except Exception as e:
        import traceback
        return {
            "seed": seed,
            "winner": None,
            "rewards": {},
            "invalid_player": None,
            "invalid_tool_calls": 0,
            "game_log": [],
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_games_sequential(
    game_name: str,
    backend: VLLMBackend,
    num_games: int,
    temperature: float,
    max_tokens: int,
    start_seed: int,
    verbose: bool,
) -> List[Dict]:
    """Run games sequentially (original behavior)."""
    results = []
    wins = {0: 0, 1: 0}
    draws = 0
    invalid_count = 0
    invalid_tool_calls = 0

    for i in range(num_games):
        seed = start_seed + i
        result = play_game(
            game_name=game_name,
            backend=backend,
            temperature=temperature,
            max_new_tokens=max_tokens,
            seed=seed,
            verbose=verbose,
        )
        results.append(result)

        if result["winner"] is not None:
            wins[result["winner"]] += 1
        else:
            draws += 1

        if result["invalid_player"] is not None:
            invalid_count += 1
        invalid_tool_calls += int(result.get("invalid_tool_calls", 0))

        # Progress
        if (i + 1) % 5 == 0 or i == num_games - 1:
            print(
                f"Game {i+1}/{num_games}: P0={wins[0]} ({wins[0]/(i+1):.1%}), "
                f"P1={wins[1]} ({wins[1]/(i+1):.1%}), Draws={draws}, "
                f"Invalid={invalid_count}, InvalidToolCalls={invalid_tool_calls}"
            )

    return results


def run_games_parallel(
    game_name: str,
    backend: VLLMBackend,
    num_games: int,
    temperature: float,
    max_tokens: int,
    start_seed: int,
    num_workers: int,
    verbose: bool,
    log_dir: Optional[str] = None,
) -> List[Dict]:
    """Run games in parallel using ThreadPoolExecutor.

    Args:
        log_dir: Directory to write individual game logs. If None and verbose=True,
                 creates a timestamped directory.
    """
    results = []
    wins = {0: 0, 1: 0}
    draws = 0
    invalid_count = 0
    invalid_tool_calls = 0
    errors = 0
    completed = 0

    # Create log directory for parallel execution with verbose
    if verbose and log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            os.path.dirname(__file__),
            "logs",
            f"{game_name}_{timestamp}"
        )

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        print(f"Writing game logs to: {log_dir}")

    # Prepare arguments for each game
    game_args = [
        (game_name, backend, temperature, max_tokens, start_seed + i, verbose, log_dir)
        for i in range(num_games)
    ]

    print(f"Running {num_games} games with {num_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_seed = {
            executor.submit(play_game_wrapper, args): args[4]  # args[4] is seed
            for args in game_args
        }

        # Process results as they complete
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                results.append(result)

                if "error" in result:
                    errors += 1
                elif result["winner"] is not None:
                    wins[result["winner"]] += 1
                else:
                    draws += 1

                if result.get("invalid_player") is not None:
                    invalid_count += 1
                invalid_tool_calls += int(result.get("invalid_tool_calls", 0))

            except Exception as e:
                errors += 1
                results.append({
                    "seed": seed,
                    "winner": None,
                    "rewards": {},
                    "invalid_player": None,
                    "invalid_tool_calls": 0,
                    "game_log": [],
                    "error": str(e),
                })

            completed += 1

            # Progress update every 5 games or at the end
            if completed % 5 == 0 or completed == num_games:
                total_valid = wins[0] + wins[1] + draws
                if total_valid > 0:
                    print(
                        f"Completed {completed}/{num_games}: P0={wins[0]} ({wins[0]/total_valid:.1%}), "
                        f"P1={wins[1]} ({wins[1]/total_valid:.1%}), Draws={draws}, "
                        f"Invalid={invalid_count}, Errors={errors}"
                    )
                else:
                    print(f"Completed {completed}/{num_games}: No valid results yet, Errors={errors}")

    # Sort results by seed for consistent ordering
    results.sort(key=lambda x: x.get("seed", 0))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run model-vs-model for any game")
    parser.add_argument("--game", type=str, required=True,
                        help=f"Game to play. Available: {list_game_names()}")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8076",
                        help="vLLM server URL")
    parser.add_argument("--num_games", type=int, default=10,
                        help="Number of games to play")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max tokens for generation")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed game logs")
    parser.add_argument("--parallel", type=int, default=0,
                        help="Number of parallel workers (0 = sequential)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to write detailed game results as JSON")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to write individual game logs (for parallel execution)")

    args = parser.parse_args()

    # Validate game
    try:
        game_spec = get_game_spec(args.game)
    except KeyError as e:
        print(f"Error: {e}")
        print(f"Available games: {list_game_names()}")
        return

    print("=" * 60)
    print(f"Game: {args.game}")
    print(f"vLLM URL: {args.vllm_url}")
    print(f"Num games: {args.num_games}")
    print(f"Temperature: {args.temperature}")
    print(f"Parallel workers: {args.parallel if args.parallel > 0 else 'sequential'}")
    print("=" * 60)

    # Connect to vLLM
    backend = VLLMBackend(args.vllm_url)
    print(f"Connected to vLLM, model: {backend.model_name}")

    # Run games
    if args.parallel > 0:
        results = run_games_parallel(
            game_name=args.game,
            backend=backend,
            num_games=args.num_games,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            start_seed=args.seed,
            num_workers=args.parallel,
            verbose=args.verbose,
            log_dir=args.log_dir,
        )
    else:
        results = run_games_sequential(
            game_name=args.game,
            backend=backend,
            num_games=args.num_games,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            start_seed=args.seed,
            verbose=args.verbose,
        )

    # Aggregate final results
    wins = {0: 0, 1: 0}
    draws = 0
    invalid_count = 0
    invalid_tool_calls = 0
    errors = 0

    for result in results:
        if "error" in result:
            errors += 1
            continue
        if result["winner"] is not None:
            wins[result["winner"]] += 1
        else:
            draws += 1
        if result["invalid_player"] is not None:
            invalid_count += 1
        invalid_tool_calls += int(result.get("invalid_tool_calls", 0))

    total_valid = wins[0] + wins[1] + draws

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if total_valid > 0:
        print(f"Player 0 wins: {wins[0]} ({wins[0]/total_valid:.1%})")
        print(f"Player 1 wins: {wins[1]} ({wins[1]/total_valid:.1%})")
        print(f"Draws: {draws} ({draws/total_valid:.1%})")
    else:
        print("No valid games completed")
    print(f"Invalid moves: {invalid_count}")
    print(f"Invalid tool calls: {invalid_tool_calls}")
    if errors > 0:
        print(f"Errors: {errors}")

    # Save results to JSON if log_dir was specified
    if args.parallel > 0 and args.verbose:
        # Determine log directory from results or create one
        log_dir = args.log_dir
        if log_dir is None:
            # Find from results
            for r in results:
                if 'log_file' in r:
                    log_dir = os.path.dirname(r['log_file'])
                    break

        if log_dir:
            results_file = os.path.join(log_dir, "results_summary.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {results_file}")

            # Print failure analysis for dependency resolution game
            if args.game == "dependency_resolution":
                print("\n" + "=" * 60)
                print("FAILURE ANALYSIS (Player 0)")
                print("=" * 60)

                early_submissions = 0  # Submitted before sufficient info
                late_submissions = 0   # Had sufficient info but submitted later
                wrong_answers = 0      # Wrong answer
                max_steps_reached = 0  # Hit max steps
                perfect_games = 0      # Correct and timely

                for r in results:
                    if 'error' in r:
                        continue
                    summary = r.get('game_summary', {})
                    reward = summary.get('reward', r.get('rewards', {}).get(0, 0))
                    suff_turn = summary.get('sufficiency_turn')
                    total_turns = summary.get('total_turns', 0)
                    correct = summary.get('correct_answer')
                    submitted = summary.get('submitted_answer')

                    if submitted is None:
                        max_steps_reached += 1
                    elif suff_turn is None:
                        # Submitted before sufficient
                        early_submissions += 1
                    elif submitted != correct:
                        wrong_answers += 1
                    elif total_turns > suff_turn + 1:
                        late_submissions += 1
                    else:
                        perfect_games += 1

                print(f"Perfect games (correct & timely): {perfect_games}")
                print(f"Early submissions (before sufficient info): {early_submissions}")
                print(f"Late submissions (correct but too many turns): {late_submissions}")
                print(f"Wrong answers: {wrong_answers}")
                print(f"Max steps reached (no submission): {max_steps_reached}")


if __name__ == "__main__":
    main()
