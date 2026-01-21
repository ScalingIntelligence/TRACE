#!/usr/bin/env python3
"""
Run all model-vs-model evaluations for Kuhn Poker.

Usage (local HuggingFace):
    python run_evaluation.py
    python run_evaluation.py --num_games 50 --models qwen-4b qwen-8b
    python run_evaluation.py --output results.json

Usage (vLLM - much faster):
    # First start vLLM servers in separate terminals:
    # Terminal 1: CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 --max-model-len 10000
    # Terminal 2: CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-30B-A3B-Instruct --port 8001 --max-model-len 10000
    
    # Then run evaluation:
    python run_evaluation.py --models qwen-4b-instruct qwen-30b-instruct \\
        --vllm_urls qwen-4b-instruct=http://localhost:8000 qwen-30b-instruct=http://localhost:8001 \\
        --num_games 50
"""

import argparse
import json
import torch
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional

from eval_config import (
    AVAILABLE_MODELS,
    DEFAULT_NUM_GAMES,
    DEFAULT_NUM_ROUNDS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_NEW_TOKENS,
)
from game_runner import load_model, play_game, load_vllm_backend, VLLMBackend


def run_matchup_with_progress(
    model_p0,
    tokenizer_p0,
    model_p1,
    tokenizer_p1,
    model_p0_name: str,
    model_p1_name: str,
    num_games: int,
    num_rounds: int,
    temperature: float,
    max_new_tokens: int,
    start_seed: int = 0,
    print_interval: int = 10,
    verbose: bool = False,
    use_vllm: bool = False,
) -> Dict:
    """
    Run multiple games between two models with progress printing.
    Model P0 always goes first.
    
    If use_vllm=True, model_p0 and model_p1 should be VLLMBackend instances.
    """
    wins = {0: 0, 1: 0}
    draws = 0
    invalid_counts = {0: 0, 1: 0}
    all_games = []
    
    print(f"\n  [{model_p0_name} (P0/first)] vs [{model_p1_name} (P1/second)]")
    print(f"  " + "-" * 50)
    
    for i in range(num_games):
        seed = start_seed + i
        result = play_game(
            model_p0=model_p0,
            tokenizer_p0=tokenizer_p0,
            model_p1=model_p1,
            tokenizer_p1=tokenizer_p1,
            num_rounds=num_rounds,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
            verbose=verbose,
            model_p0_name=model_p0_name,
            model_p1_name=model_p1_name,
            use_vllm=use_vllm,
        )
        
        if result["winner"] is not None:
            wins[result["winner"]] += 1
        else:
            draws += 1
        
        if result["invalid_player"] is not None:
            invalid_counts[result["invalid_player"]] += 1
        
        all_games.append(result)
        
        # Print progress every print_interval games
        game_num = i + 1
        if game_num % print_interval == 0 or game_num == num_games:
            p0_wins = wins[0]
            p1_wins = wins[1]
            p0_rate = p0_wins / game_num if game_num > 0 else 0
            p1_rate = p1_wins / game_num if game_num > 0 else 0
            print(f"  Game {game_num:4d}/{num_games}: "
                  f"{model_p0_name}={p0_wins} ({p0_rate:.1%}) | "
                  f"{model_p1_name}={p1_wins} ({p1_rate:.1%}) | "
                  f"Draws={draws}")
    
    return {
        "wins": wins,
        "draws": draws,
        "invalid_counts": invalid_counts,
        "p0_win_rate": wins[0] / num_games,
        "p1_win_rate": wins[1] / num_games,
        "draw_rate": draws / num_games,
        "all_games": all_games,
    }


def run_all_matchups(
    models_to_test: List[str],
    num_games: int,
    num_rounds: int,
    temperature: float,
    max_new_tokens: int,
    output_path: Optional[str] = None,
    print_interval: int = 10,
    verbose: bool = False,
    vllm_urls: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Run all ordered pairwise matchups between the specified models.
    Each (A, B) pair is tested separately from (B, A).
    
    If vllm_urls is provided, it should be a dict mapping model names to vLLM server URLs.
    """
    use_vllm = vllm_urls is not None and len(vllm_urls) > 0
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_games_per_matchup": num_games,
            "num_rounds_per_game": num_rounds,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "models_tested": models_to_test,
            "backend": "vllm" if use_vllm else "local_hf",
        },
        "matchups": {},
        "summary": {},
    }
    
    # Load all models first
    print("=" * 60)
    if use_vllm:
        print("Connecting to vLLM servers...")
    else:
        print("Loading models locally...")
    print("=" * 60)
    
    loaded_models = {}
    for model_name in models_to_test:
        if use_vllm and model_name in vllm_urls:
            url = vllm_urls[model_name]
            # Get the actual HF model name for vLLM
            hf_model_name = AVAILABLE_MODELS.get(model_name, model_name)
            backend = load_vllm_backend(url, hf_model_name)
            loaded_models[model_name] = (backend, None)  # No tokenizer needed for vLLM
        else:
            model, tokenizer = load_model(model_name)
            loaded_models[model_name] = (model, tokenizer)
    
    print(f"\nLoaded {len(loaded_models)} models: {list(loaded_models.keys())}")
    
    # Generate all ordered pairs (permutations, not combinations)
    matchup_pairs = [(a, b) for a, b in permutations(models_to_test, 2)]
    total_matchups = len(matchup_pairs)
    
    print(f"\nRunning {total_matchups} matchups ({num_games} games each)...")
    print("=" * 60)
    
    # Run all matchups
    for idx, (model_first, model_second) in enumerate(matchup_pairs, 1):
        print(f"\n>>> Matchup {idx}/{total_matchups}: {model_first} (first) vs {model_second} (second)")
        
        model_first_obj, tokenizer_first = loaded_models[model_first]
        model_second_obj, tokenizer_second = loaded_models[model_second]
        
        result = run_matchup_with_progress(
            model_p0=model_first_obj,
            tokenizer_p0=tokenizer_first,
            model_p1=model_second_obj,
            tokenizer_p1=tokenizer_second,
            model_p0_name=model_first,
            model_p1_name=model_second,
            num_games=num_games,
            num_rounds=num_rounds,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            start_seed=idx * num_games,
            print_interval=print_interval,
            verbose=verbose,
            use_vllm=use_vllm,
        )
        
        matchup_key = f"{model_first}_vs_{model_second}"
        results["matchups"][matchup_key] = {
            "first_player": model_first,
            "second_player": model_second,
            "first_player_wins": result["wins"][0],
            "second_player_wins": result["wins"][1],
            "draws": result["draws"],
            "first_player_win_rate": result["p0_win_rate"],
            "second_player_win_rate": result["p1_win_rate"],
            "draw_rate": result["draw_rate"],
            "first_player_invalid_count": result["invalid_counts"][0],
            "second_player_invalid_count": result["invalid_counts"][1],
        }
    
    # Calculate summary statistics for each model
    print("\n" + "=" * 60)
    print("FINAL RESULTS - All (First, Second) Matchups")
    print("=" * 60)
    
    # Print detailed matchup table
    print(f"\n{'First Player':<15} {'Second Player':<15} {'First Wins':<12} {'Second Wins':<12} {'Draws':<8} {'First WR':<10}")
    print("-" * 82)
    
    for matchup_key, data in results["matchups"].items():
        print(f"{data['first_player']:<15} {data['second_player']:<15} "
              f"{data['first_player_wins']:<12} {data['second_player_wins']:<12} "
              f"{data['draws']:<8} {data['first_player_win_rate']:<10.1%}")
    
    # Calculate overall stats per model
    print("\n" + "=" * 60)
    print("SUMMARY - Overall Performance")
    print("=" * 60)
    
    model_stats = {name: {"wins": 0, "losses": 0, "draws": 0, "games": 0,
                          "wins_as_first": 0, "games_as_first": 0,
                          "wins_as_second": 0, "games_as_second": 0}
                   for name in models_to_test}
    
    for matchup_key, data in results["matchups"].items():
        first = data["first_player"]
        second = data["second_player"]
        
        # First player stats
        model_stats[first]["wins"] += data["first_player_wins"]
        model_stats[first]["losses"] += data["second_player_wins"]
        model_stats[first]["draws"] += data["draws"]
        model_stats[first]["games"] += num_games
        model_stats[first]["wins_as_first"] += data["first_player_wins"]
        model_stats[first]["games_as_first"] += num_games
        
        # Second player stats
        model_stats[second]["wins"] += data["second_player_wins"]
        model_stats[second]["losses"] += data["first_player_wins"]
        model_stats[second]["draws"] += data["draws"]
        model_stats[second]["games"] += num_games
        model_stats[second]["wins_as_second"] += data["second_player_wins"]
        model_stats[second]["games_as_second"] += num_games
    
    # Store and sort by overall win rate
    summary_list = []
    for model_name, stats in model_stats.items():
        overall_wr = stats["wins"] / stats["games"] if stats["games"] > 0 else 0
        first_wr = stats["wins_as_first"] / stats["games_as_first"] if stats["games_as_first"] > 0 else 0
        second_wr = stats["wins_as_second"] / stats["games_as_second"] if stats["games_as_second"] > 0 else 0
        
        results["summary"][model_name] = {
            "total_wins": stats["wins"],
            "total_losses": stats["losses"],
            "total_draws": stats["draws"],
            "total_games": stats["games"],
            "overall_win_rate": overall_wr,
            "wins_as_first": stats["wins_as_first"],
            "games_as_first": stats["games_as_first"],
            "win_rate_as_first": first_wr,
            "wins_as_second": stats["wins_as_second"],
            "games_as_second": stats["games_as_second"],
            "win_rate_as_second": second_wr,
        }
        summary_list.append((model_name, overall_wr, first_wr, second_wr, stats))
    
    summary_list.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6}{'Model':<15}{'Overall WR':<12}{'As First':<12}{'As Second':<12}{'W/L/D':<15}")
    print("-" * 72)
    for rank, (name, overall_wr, first_wr, second_wr, stats) in enumerate(summary_list, 1):
        wld = f"{stats['wins']}/{stats['losses']}/{stats['draws']}"
        print(f"{rank:<6}{name:<15}{overall_wr:<12.1%}{first_wr:<12.1%}{second_wr:<12.1%}{wld:<15}")
    
    # Save results
    if output_path:
        results_to_save = {
            "metadata": results["metadata"],
            "matchups": results["matchups"],
            "summary": results["summary"],
        }
        
        with open(output_path, "w") as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run model-vs-model Kuhn Poker evaluations")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to test. Available: {list(AVAILABLE_MODELS.keys())}. Default: all",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=DEFAULT_NUM_GAMES,
        help=f"Number of games per matchup (default: {DEFAULT_NUM_GAMES})",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=DEFAULT_NUM_ROUNDS,
        help=f"Number of rounds per game (default: {DEFAULT_NUM_ROUNDS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max new tokens for generation (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=10,
        help="Print progress every N games (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game logs",
    )
    parser.add_argument(
        "--vllm_urls",
        nargs="+",
        default=None,
        help="vLLM server URLs in format 'model_name=url'. E.g., --vllm_urls qwen-4b-instruct=http://localhost:8000 qwen-30b-instruct=http://localhost:8001",
    )
    
    args = parser.parse_args()
    
    # Determine which models to test
    if args.models:
        models_to_test = args.models
        for m in models_to_test:
            if m not in AVAILABLE_MODELS:
                print(f"Warning: '{m}' not in AVAILABLE_MODELS, treating as full path")
    else:
        models_to_test = list(AVAILABLE_MODELS.keys())
    
    if len(models_to_test) < 2:
        print("Error: Need at least 2 models to run matchups")
        return
    
    # Parse vLLM URLs if provided
    vllm_urls = None
    if args.vllm_urls:
        vllm_urls = {}
        for item in args.vllm_urls:
            if "=" in item:
                model_name, url = item.split("=", 1)
                vllm_urls[model_name] = url
            else:
                print(f"Warning: Invalid vllm_url format '{item}', expected 'model_name=url'")
    
    print("=" * 60)
    print("Kuhn Poker Model-vs-Model Evaluation")
    print("=" * 60)
    print(f"Models: {models_to_test}")
    print(f"Games per matchup: {args.num_games}")
    print(f"Rounds per game: {args.num_rounds}")
    print(f"Temperature: {args.temperature}")
    print(f"Backend: {'vLLM' if vllm_urls else 'Local HuggingFace'}")
    if vllm_urls:
        print(f"vLLM URLs: {vllm_urls}")
    print(f"Total matchups: {len(models_to_test) * (len(models_to_test) - 1)}")
    print()
    
    run_all_matchups(
        models_to_test=models_to_test,
        num_games=args.num_games,
        num_rounds=args.num_rounds,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        print_interval=args.print_interval,
        verbose=args.verbose,
        vllm_urls=vllm_urls,
    )


if __name__ == "__main__":
    main()
