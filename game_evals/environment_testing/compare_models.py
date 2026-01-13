#!/usr/bin/env python3
"""
Script to compare two OpenAI models across multiple game environments.

Usage:
    cd weaver_for_RL/experiments/environment_testing
    python compare_models.py --model-a gpt-4o-mini --model-b gpt-4o --envs all --trials 10
    
    # Or specify specific environments by name or number:
    python compare_models.py --model-a gpt-4o-mini --model-b gpt-4o --envs TicTacToe-v1,Connect4-v1
    python compare_models.py --model-a gpt-4o-mini --model-b gpt-4o --envs 1,2,3 --trials 5 --verbose

This script will run head-to-head matches between two models on each game,
printing win rates after each game environment is completed.
"""

import sys
import os
import time
import argparse
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

# Add parent directory to path to import envs module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import environment utilities
from envs import make_env

# Import inference backends
from inference_backends import InferenceBackend, OpenAIBackend, VLLMBackend

from config import OPENAI_API_KEY, OPENAI_MODEL_A, OPENAI_MODEL_B
OPENAI_AVAILABLE = True
OPENAI_CONFIGURED = OPENAI_API_KEY and OPENAI_API_KEY.strip() and not OPENAI_API_KEY.startswith("your-")



# =========================
# API Key Management
# =========================

def prompt_for_api_key(api_key_arg: str = None):
    """Prompt user for OpenAI API key if not configured.
    
    Args:
        api_key_arg: Optional API key from command line argument
    """
    global OPENAI_API_KEY, OPENAI_CONFIGURED
    
    # If API key provided via argument, use it
    if api_key_arg and api_key_arg.startswith("sk-"):
        OPENAI_API_KEY = api_key_arg
        OPENAI_CONFIGURED = True
        return True
    
    # If already configured, return True
    if OPENAI_CONFIGURED:
        return True
    else:
        print("⚠️  Cannot continue without API key.")
        return False


def get_llm_action(
    observation: str,
    env_id: str,
    model: str,
    backend: InferenceBackend,
    temperature: float = 0.7,
    max_tokens: int = None
) -> Tuple[Optional[str], Optional[str]]:
    """Get action from LLM using the specified backend.
    
    Args:
        observation: The observation string to send to the LLM
        env_id: Environment ID
        model: Model name
        backend: Inference backend to use
        temperature: Temperature for sampling
        max_tokens: Maximum tokens in response
        
    Returns:
        Tuple of (action, error_message)
    """
    return backend.generate_action(observation, env_id, model, temperature, max_tokens)


def get_num_players(env_id: str) -> int:
    """Determine number of players for the environment."""
    if env_id in ["Game24-v1"]:
        return 1
    return 2


def play_game(
    env_id: str,
    model_a: str,
    model_b: str,
    backend_a: InferenceBackend,
    backend_b: InferenceBackend,
    verbose: bool = False
) -> Dict:
    """Play one game between two models.
    
    Args:
        env_id: Environment ID
        model_a: First model name (plays as player 0 then player 1)
        model_b: Second model name (plays as player 1 then player 0)
        backend_a: Backend for model A
        backend_b: Backend for model B
        verbose: Whether to print game progress
        
    Returns:
        Dictionary with game results
    """
    # Use LLM observation wrapper since we're testing LLMs
    env = make_env(env_id, use_llm_obs_wrapper=True)
    
    num_players = get_num_players(env_id)
    env.reset(num_players=num_players)
    
    models = {0: model_a, 1: model_b}
    backends = {0: backend_a, 1: backend_b}
    
    if verbose:
        print(f"  Model A ({model_a}) playing as Player 0")
        print(f"  Model B ({model_b}) playing as Player 1")
    
    turn_count = 0
    max_turns = 200  # Safety limit
    invalid_player = None
    turns = []  # Track actions per turn
    
    while turn_count < max_turns:
        # Get current observation
        player_id, observation = env.get_observation()
        current_model = models[player_id]
        current_backend = backends[player_id]
        
        # Get action from LLM
        action, error = get_llm_action(observation, env_id, current_model, current_backend)
        
        if error:
            raise RuntimeError(error)
        
        if verbose:
            action_preview = action[:100] if len(action) <= 100 else action[:100] + "..."
            print(f"  Turn {turn_count}: Player {player_id} ({current_model}) → {action_preview}")
        
        # Take step
        done, info = env.step(action)
        
        # Record turn information
        turn_info = {
            "turn": turn_count,
            "player_id": player_id,
            "model": current_model,
            "action": action,
            "invalid_move": info.get('invalid_move', False),
            "reason": info.get('reason', '') if 'reason' in info else ''
        }
        turns.append(turn_info)
        
        # Debug: Print info when game ends
        if done and verbose:
            print(f"    Game ended. Info: {info}")
            if hasattr(env.state, 'rewards'):
                print(f"    Rewards: {env.state.rewards}")
        
        # Check for invalid move (check both flag and reason string)
        is_invalid = info.get('invalid_move', False) or \
                     ('reason' in info and 'Invalid Move' in str(info.get('reason', '')))
        
        if is_invalid:
            if verbose:
                print(f"    ❌ Invalid move by Player {player_id} ({current_model}): {info.get('reason', 'Unknown')}")
            invalid_player = player_id
            break
        
        turn_count += 1
        
        if done:
            break
    
    # Get final rewards
    rewards = env.state.rewards if hasattr(env.state, 'rewards') else {0: 0, 1: 0}
    
    # Determine outcome from player 0's perspective
    if invalid_player is not None:
        if invalid_player == 0:  # Player 0 made invalid move
            outcome = "loss"
            player_0_reward = -1
            player_1_reward = 1
        else:  # Player 1 made invalid move
            outcome = "win"
            player_0_reward = 1
            player_1_reward = -1
    elif turn_count >= max_turns:
        outcome = "draw"
        player_0_reward = 0
        player_1_reward = 0
    else:
        player_0_reward = rewards.get(0, 0)
        player_1_reward = rewards.get(1, 0)
        
        if player_0_reward > player_1_reward:
            outcome = "win"
        elif player_0_reward < player_1_reward:
            outcome = "loss"
        else:
            outcome = "draw"
    
    return {
        "outcome": outcome,  # From player 0's perspective
        "player_0_reward": player_0_reward,
        "player_1_reward": player_1_reward,
        "turns": turn_count,
        "invalid_player": invalid_player,
        "reason": info.get('reason', '') if 'info' in locals() else '',
        "turn_details": turns  # Actions per turn
    }


def run_single_trial(
    trial_num: int,
    env_id: str,
    model_a: str,
    model_b: str,
    backend_a: InferenceBackend,
    backend_b: InferenceBackend,
    is_negotiation: bool,
    verbose: bool = False
) -> Dict:
    """Run a single trial game.
    
    Args:
        trial_num: Trial number (for logging)
        env_id: Environment ID
        model_a: First model name
        model_b: Second model name
        backend_a: Backend for model A
        backend_b: Backend for model B
        is_negotiation: Whether this is a negotiation game
        verbose: Whether to print detailed game progress
        
    Returns:
        Dictionary with trial results
    """
    # Randomize starting positions
    model_a_played_as_player_0 = random.random() < 0.5
    if model_a_played_as_player_0:
        # Model A plays as player 0, Model B as player 1
        result = play_game(env_id, model_a, model_b, backend_a, backend_b, verbose=verbose)
        # Outcome is from player 0 (Model A)'s perspective
        model_a_outcome = result["outcome"]
        model_a_reward = result["player_0_reward"]
        model_b_reward = result["player_1_reward"]
        invalid_a = result["invalid_player"] == 0
        invalid_b = result["invalid_player"] == 1
        # Map turn details: player 0 = model_a, player 1 = model_b
        turn_details = result.get("turn_details", [])
    else:
        # Model B plays as player 0, Model A as player 1
        result = play_game(env_id, model_b, model_a, backend_b, backend_a, verbose=verbose)
        # Outcome is from player 0 (Model B)'s perspective - FLIP for Model A
        if result["outcome"] == "win":
            model_a_outcome = "loss"  # Player 0 (Model B) won, so Model A lost
        elif result["outcome"] == "loss":
            model_a_outcome = "win"  # Player 0 (Model B) lost, so Model A won
        else:
            model_a_outcome = "draw"
        # Rewards: Player 0 is Model B, Player 1 is Model A
        model_a_reward = result["player_1_reward"]
        model_b_reward = result["player_0_reward"]
        invalid_a = result["invalid_player"] == 1
        invalid_b = result["invalid_player"] == 0
        # Map turn details: player 0 = model_b, player 1 = model_a
        turn_details = result.get("turn_details", [])
        # Update turn details to show which model (A or B) made each action
        for turn in turn_details:
            if turn["player_id"] == 0:
                turn["model"] = model_b  # Player 0 was Model B
            else:
                turn["model"] = model_a  # Player 1 was Model A
    
    return {
        "trial_num": trial_num,
        "model_a_outcome": model_a_outcome,
        "model_a_reward": model_a_reward,
        "model_b_reward": model_b_reward,
        "invalid_a": invalid_a,
        "invalid_b": invalid_b,
        "turns": result["turns"],
        "is_negotiation": is_negotiation,
        "turn_details": turn_details,  # Actions per turn
        "model_a_played_as_player_0": model_a_played_as_player_0
    }


def compare_models_on_env(
    env_id: str,
    model_a: str,
    model_b: str,
    backend_a: InferenceBackend,
    backend_b: InferenceBackend,
    n_trials: int,
    max_workers: int,
    verbose: bool = False
):
    """Compare two models on a single environment.
    
    Args:
        env_id: Environment ID
        model_a: First model name
        model_b: Second model name
        backend_a: Backend for model A
        backend_b: Backend for model B
        n_trials: Number of games to play
        max_workers: Maximum number of parallel workers
        verbose: Whether to print detailed game progress
    """
    print(f"\n{'='*70}")
    print(f"ENVIRONMENT: {env_id}")
    print(f"{'='*70}")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"Number of trials: {n_trials}")
    print(f"Max workers: {max_workers}")

    if env_id == 'Battleship-v1':
        return None
    
    # Check if single-player game
    if get_num_players(env_id) == 1:
        print(f"\n⚠️  WARNING: {env_id} is a single-player game.")
        print(f"    Head-to-head comparison not meaningful. Skipping.")
        print(f"{'='*70}\n")
        return None
    
    print(f"{'='*70}\n")
    
    # Check if this is a negotiation game (non-zero-sum)
    is_negotiation = env_id.startswith("Negotiation")
    
    results = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "invalid_a": 0,
        "invalid_b": 0,
        "total_turns": 0,
        "model_a_total_reward": 0.0,
        "model_b_total_reward": 0.0,
    }
    
    # Run trials in parallel
    trial_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_trial,
                trial_num + 1,
                env_id,
                model_a,
                model_b,
                backend_a,
                backend_b,
                is_negotiation,
                verbose
            )
            for trial_num in range(n_trials)
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            trial_result = future.result()
            trial_results.append(trial_result)
            
            # Update results based on Model A's perspective
            if trial_result["model_a_outcome"] == "win":
                results["wins"] += 1
            elif trial_result["model_a_outcome"] == "loss":
                results["losses"] += 1
            else:
                results["draws"] += 1
            
            if trial_result["invalid_a"]:
                results["invalid_a"] += 1
            if trial_result["invalid_b"]:
                results["invalid_b"] += 1
            
            results["total_turns"] += trial_result["turns"]
            results["model_a_total_reward"] += trial_result["model_a_reward"]
            results["model_b_total_reward"] += trial_result["model_b_reward"]
            
            # Print progress
            if not verbose:
                if is_negotiation:
                    print(f"Trial {trial_result['trial_num']}/{n_trials}: Rewards A={trial_result['model_a_reward']:.3f}, B={trial_result['model_b_reward']:.3f} (Turns: {trial_result['turns']})", end="")
                else:
                    print(f"Trial {trial_result['trial_num']}/{n_trials}: {trial_result['model_a_outcome'].upper()} (Turns: {trial_result['turns']})", end="")
                if trial_result["invalid_a"]:
                    print(f" - INVALID: Model A", end="")
                elif trial_result["invalid_b"]:
                    print(f" - INVALID: Model B", end="")
                print()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {env_id}")
    print(f"{'='*70}")
    print(f"Model A ({model_a}) vs Model B ({model_b})")
    
    if is_negotiation:
        # For negotiation games, show average rewards (non-zero-sum)
        avg_a = results["model_a_total_reward"] / n_trials
        avg_b = results["model_b_total_reward"] / n_trials
        print(f"\nAverage Reward (Model A): {avg_a:.3f}")
        print(f"Average Reward (Model B): {avg_b:.3f}")
    else:
        # For competitive games, show win/loss/draw rates
        print(f"\nWin Rate (Model A): {results['wins']}/{n_trials} = {results['wins']/n_trials*100:.1f}%")
        print(f"Loss Rate (Model A): {results['losses']}/{n_trials} = {results['losses']/n_trials*100:.1f}%")
        print(f"Draw Rate: {results['draws']}/{n_trials} = {results['draws']/n_trials*100:.1f}%")
    
    print(f"\nInvalid Moves (Model A): {results['invalid_a']}")
    print(f"\nInvalid Moves (Model B): {results['invalid_b']}")
    print(f"Average turns per game: {results['total_turns']/n_trials:.1f}")
    print(f"{'='*70}\n")
    
    # Save results to JSON file
    results_base_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Create directory structure: results/{model_a}_vs_{model_b}/
    safe_model_a = model_a.replace("/", "_").replace("\\", "_")
    safe_model_b = model_b.replace("/", "_").replace("\\", "_")
    model_pair_dir = os.path.join(results_base_dir, f"{safe_model_a}_vs_{safe_model_b}")
    os.makedirs(model_pair_dir, exist_ok=True)
    
    # Create JSON output with all trial details
    json_output = {
        "env_id": env_id,
        "model_a": model_a,
        "model_b": model_b,
        "n_trials": n_trials,
        "is_negotiation": is_negotiation,
        "summary": {
            "wins": results["wins"],
            "losses": results["losses"],
            "draws": results["draws"],
            "invalid_a": results["invalid_a"],
            "invalid_b": results["invalid_b"],
            "total_turns": results["total_turns"],
            "average_turns": results["total_turns"] / n_trials,
            "model_a_total_reward": results["model_a_total_reward"],
            "model_b_total_reward": results["model_b_total_reward"],
            "model_a_avg_reward": results["model_a_total_reward"] / n_trials,
            "model_b_avg_reward": results["model_b_total_reward"] / n_trials,
        },
        "trials": sorted(trial_results, key=lambda x: x["trial_num"])
    }
    
    # Save to JSON file: results/{model_a}_vs_{model_b}/{env_id}.json
    safe_env_id = env_id.replace("/", "_").replace("\\", "_")
    json_filename = f"{safe_env_id}.json"
    json_path = os.path.join(model_pair_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {json_path}\n")
    
    return results


def parse_args():
    """Parse command line arguments."""
    available_envs = [
        "TicTacToe-v1",
        "Connect4-v1",
        "Battleship-v1",
        "RockPaperScissors-v1",
        "Nim-v1",
        "DotsAndBoxes-v1",
        "Chomp-v1",
        "Auction-v1",
        "Negotiation-v1",
        "SimpleNegotiation-v1",
        "PigDice-v1",
        "LiarsDice-v1",
        "KuhnPoker-v1",
    ]
    
    parser = argparse.ArgumentParser(
        description="Compare two OpenAI models across multiple game environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available environments:
{chr(10).join(f'  {i}. {env}' for i, env in enumerate(available_envs, 1))}

Examples:
  # Compare models on all environments with defaults from config
  python compare_models.py
  
  # Compare specific models on all environments
  python compare_models.py --model-a gpt-4o-mini --model-b gpt-4o --envs all
  
  # Compare on specific environments by name
  python compare_models.py --model-a gpt-4o-mini --model-b gpt-4o --envs TicTacToe-v1,Connect4-v1
  
  # Compare on specific environments by number
  python compare_models.py --model-a gpt-4o-mini --model-b gpt-4o --envs 1,2,3 --trials 5 --verbose
        """
    )
    
    parser.add_argument(
        '--model-a',
        type=str,
        default=None,
        help=f'Model A name (default: {OPENAI_MODEL_A} from config.py)'
    )
    
    parser.add_argument(
        '--model-b',
        type=str,
        default=None,
        help=f'Model B name (default: {OPENAI_MODEL_B} from config.py)'
    )
    
    parser.add_argument(
        '--envs',
        type=str,
        default='all',
        help='Environments to test: "all", comma-separated names (e.g., "TicTacToe-v1,Connect4-v1"), or comma-separated numbers (e.g., "1,2,3") (default: all)'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Number of trials per environment (default: 10)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed game progress'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (can also be set via OPENAI_API_KEY env var or config.py)'
    )
    
    parser.add_argument(
        '--inference-backend',
        type=str,
        choices=['openai', 'vllm'],
        default='openai',
        help='Inference backend to use: "openai" or "vllm" (default: openai)'
    )
    
    parser.add_argument(
        '--vllm-address-1',
        type=str,
        default=None,
        help='vLLM server address for model A (required when using vllm backend)'
    )
    
    parser.add_argument(
        '--vllm-address-2',
        type=str,
        default=None,
        help='vLLM server address for model B (required when using vllm backend)'
    )
    
    parser.add_argument(
        '--vllm-api-key',
        type=str,
        default=None,
        help='vLLM API key (optional, for authenticated vLLM servers)'
    )
    
    parser.add_argument(
        '--vllm-timeout',
        type=float,
        default=120.0,
        help='vLLM request timeout in seconds (default: 120.0)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help=f'Maximum number of parallel workers for game execution (default: {os.cpu_count()})'
    )
    
    parser.add_argument(
        '--enable-thinking',
        action='store_true',
        help='Enable thinking mode for vLLM backend (default: False)'
    )
    
    return parser.parse_args()


def parse_env_selection(env_choice: str, available_envs: List[str]) -> List[str]:
    """Parse environment selection string into list of environment IDs.
    
    Args:
        env_choice: "all", comma-separated names, or comma-separated numbers
        available_envs: List of available environment IDs
        
    Returns:
        List of selected environment IDs
    """
    env_choice = env_choice.strip()
    if env_choice.lower() == 'all':
        return available_envs
    
    # Try parsing as comma-separated numbers first
    try:
        indices = [int(x.strip()) - 1 for x in env_choice.split(',')]
        env_ids = [available_envs[i] for i in indices]
        return env_ids
    except (ValueError, IndexError):
        pass
    
    # Try parsing as comma-separated environment names
    env_names = [name.strip() for name in env_choice.split(',')]
    env_ids = []
    for name in env_names:
        if name in available_envs:
            env_ids.append(name)
        else:
            print(f"Warning: Unknown environment '{name}', skipping.")
    
    
    if not env_ids:
        print(f"Warning: No valid environments found. Using all environments.")
        return available_envs
    
    return env_ids


def create_backend(backend_type: str, address: Optional[str], api_key: Optional[str], timeout: float, model_name: str = None, max_workers: int = None, enable_thinking: bool = False) -> InferenceBackend:
    """Create an inference backend based on the specified type.
    
    Args:
        backend_type: Either "openai" or "vllm"
        address: Server address (for vLLM) or None (for OpenAI)
        api_key: API key for authentication
        timeout: Timeout in seconds (for vLLM)
        model_name: Model name (for vLLM)
        max_workers: Maximum number of workers (used for pool_connections and pool_maxsize)
        enable_thinking: Enable thinking mode for vLLM backend
        
    Returns:
        InferenceBackend instance
    """
    if max_workers is None:
        max_workers = os.cpu_count()
    
    pool_size = max_workers
    
    if backend_type == "openai":
        if not api_key:
            api_key = OPENAI_API_KEY
        if not api_key or not api_key.strip() or api_key.startswith("your-"):
            raise ValueError("OpenAI API key is required when using openai backend")
        return OpenAIBackend(api_key, pool_connections=pool_size, pool_maxsize=pool_size)
    elif backend_type == "vllm":
        if not address:
            raise ValueError("vLLM server address is required when using vllm backend")
        return VLLMBackend(address, api_key, timeout, model_name, pool_connections=pool_size, pool_maxsize=pool_size, enable_thinking=enable_thinking)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def main():
    """Main function to compare models across environments."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("MODEL COMPARISON TOOL")
    print("="*70)
    
    # Get number of trials early
    n_trials = args.trials
    
    # Get max_workers early so we can use it for backend connection pools
    # Set to minimum of max_workers and number of trials
    initial_max_workers = args.max_workers if args.max_workers is not None else os.cpu_count()
    max_workers = min(initial_max_workers, n_trials)
    
    # Initialize backends
    if args.inference_backend == "openai":
        # Handle API key for OpenAI
        if not prompt_for_api_key(args.api_key):
            print("\n❌ ERROR: OpenAI API key not configured!")
            print("Please provide --api-key, set OPENAI_API_KEY env var, or configure in config.py")
            return
        backend_a = create_backend("openai", None, args.api_key or OPENAI_API_KEY, 0.0, max_workers=max_workers, enable_thinking=args.enable_thinking)
        backend_b = create_backend("openai", None, args.api_key or OPENAI_API_KEY, 0.0, max_workers=max_workers, enable_thinking=args.enable_thinking)
    elif args.inference_backend == "vllm":
        if not args.vllm_address_1:
            print("\n❌ ERROR: --vllm-address-1 is required when using vllm backend")
            return
        if not args.vllm_address_2:
            print("\n❌ ERROR: --vllm-address-2 is required when using vllm backend")
            return
        backend_a = create_backend("vllm", args.vllm_address_1, args.vllm_api_key, args.vllm_timeout, args.model_a, max_workers=max_workers, enable_thinking=args.enable_thinking)
        backend_b = create_backend("vllm", args.vllm_address_2, args.vllm_api_key, args.vllm_timeout, args.model_b, max_workers=max_workers, enable_thinking=args.enable_thinking)
    else:
        print(f"\n❌ ERROR: Unknown inference backend: {args.inference_backend}")
        return
    
    # Get model names
    available_envs = [
        "TicTacToe-v1",
        "Connect4-v1",
        "Battleship-v1",
        "RockPaperScissors-v1",
        "Nim-v1",
        "DotsAndBoxes-v1",
        "Chomp-v1",
        "Auction-v1",
        "Negotiation-v1",
        "SimpleNegotiation-v1",
        "PigDice-v1",
        "LiarsDice-v1",
        "KuhnPoker-v1",
    ]
    
    model_a = args.model_a if args.model_a else OPENAI_MODEL_A
    model_b = args.model_b if args.model_b else OPENAI_MODEL_B
    
    # Parse environment selection
    env_ids = parse_env_selection(args.envs, available_envs)
    
    # Get verbose flag
    verbose = args.verbose
    # max_workers and n_trials already calculated above
    
    # Run comparisons
    print(f"\n{'='*70}")
    print("STARTING COMPARISON")
    print(f"{'='*70}")
    print(f"Inference Backend: {args.inference_backend}")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"Environments: {', '.join(env_ids)}")
    print(f"Trials per environment: {n_trials}")
    print(f"Max workers: {max_workers}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    all_results = {}
    
    for env_id in env_ids:
        results = compare_models_on_env(env_id, model_a, model_b, backend_a, backend_b, n_trials, max_workers, verbose=verbose)
        if results is not None:  # Skip single-player games
            all_results[env_id] = results
    
    # Print overall summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"{'='*70}\n")
    
    total_wins = sum(r["wins"] for r in all_results.values())
    total_losses = sum(r["losses"] for r in all_results.values())
    total_draws = sum(r["draws"] for r in all_results.values())
    total_games = total_wins + total_losses + total_draws
    
    if total_games > 0:
        print(f"Overall Win Rate (Model A): {total_wins}/{total_games} = {total_wins/total_games*100:.1f}%")
        print(f"Overall Loss Rate (Model A): {total_losses}/{total_games} = {total_losses/total_games*100:.1f}%")
        print(f"Overall Draw Rate: {total_draws}/{total_games} = {total_draws/total_games*100:.1f}%")
    
    print("\nPer-Environment Win Rates (Model A):")
    for env_id, results in all_results.items():
        total = results["wins"] + results["losses"] + results["draws"]
        if total > 0:
            win_pct = results["wins"] / total * 100
            print(f"  {env_id:25s}: {results['wins']:2d}/{total:2d} = {win_pct:5.1f}%")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

