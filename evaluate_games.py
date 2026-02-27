"""Evaluate multistep_task and structured_data_reasoning games against a vLLM model.

Generates rollouts by sending game observations to a vLLM chat completions
endpoint, collecting responses, and computing rewards.

Usage:
    # Evaluate multistep_task:
    python evaluate_games.py \
        --game multistep_task \
        --base-url http://localhost:9000/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --num-seeds 50

    # Evaluate structured_data_reasoning at difficulty 4:
    python evaluate_games.py \
        --game structured_data_reasoning \
        --difficulty 4 \
        --base-url http://localhost:9000/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --num-seeds 50
"""

import json
import time
import argparse
import threading
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

import requests


# ---------------------------------------------------------------------------
# vLLM chat client
# ---------------------------------------------------------------------------

class VLLMChatClient:
    def __init__(self, base_url: str, model: str, max_tokens: int = 2048,
                 temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=128, pool_connections=128)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def generate(self, system_prompt: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_multistep_episode(client: VLLMChatClient, seed: int) -> Dict[str, Any]:
    from multistep_task_game import (
        RealisticMultiStepGame, extract_action, SYSTEM_PROMPT,
    )

    game = RealisticMultiStepGame(max_steps=30)
    game.reset(seed)
    raw_responses = []

    while not game.done:
        obs = game.observe(0)
        try:
            response = client.generate(SYSTEM_PROMPT, obs)
            raw_responses.append(response)
        except Exception as e:
            game.step(None)
            break

        action = extract_action(response, game.legal_actions())
        game.step(action)

    summary = game.get_summary()
    summary["seed"] = seed
    summary["raw_responses"] = raw_responses
    return summary


def run_structured_data_episode(client: VLLMChatClient, seed: int,
                                 difficulty: int) -> Dict[str, Any]:
    from structured_data_game import (
        StructuredDataGame, extract_action, SYSTEM_PROMPT,
    )

    game = StructuredDataGame(difficulty=difficulty)
    game.reset(seed)

    obs = game.observe(0)
    try:
        response = client.generate(SYSTEM_PROMPT, obs)
    except Exception as e:
        game.step(None)
        summary = game.get_summary()
        summary["seed"] = seed
        summary["error"] = str(e)
        return summary

    action = extract_action(response, game.legal_actions())
    game.step(action)

    summary = game.get_summary()
    summary["seed"] = seed
    summary["raw_response"] = response
    return summary


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def analyze_multistep_results(results: List[Dict]) -> Dict[str, Any]:
    """Detailed breakdown of multistep task game failures."""
    total = len(results)
    rewards = [r["reward"] for r in results]
    perfect = sum(1 for r in rewards if r >= 1.0)
    avg = sum(rewards) / total if total else 0

    # Failure categorization
    failures = [r for r in results if r["reward"] < 1.0]
    failure_reasons = defaultdict(int)
    for f in failures:
        reason = f.get("reason", "")
        if "one-shot" in reason.lower():
            failure_reasons["one_shot_violation"] += 1
        if "wrong tool" in reason.lower():
            failure_reasons["wrong_tool_type"] += 1
        if "wrong call" in reason.lower():
            failure_reasons["wrong_calls"] += 1
        if "loop" in reason.lower():
            failure_reasons["loop_detected"] += 1
        if "max steps" in reason.lower():
            failure_reasons["max_steps"] += 1
        if "invalid" in reason.lower():
            failure_reasons["invalid_format"] += 1

        # Count missed operations
        for part in reason.split(";"):
            if "MISSED" in part:
                failure_reasons["missed_ops"] += 1

    # Op completion rates
    op_done = 0
    op_total = 0
    for r in results:
        reason = r.get("reason", "")
        # Parse "X/Y ops completed"
        if "/" in reason and "ops" in reason:
            parts = reason.split("/")
            try:
                done = int(parts[0].split()[-1])
                tot = int(parts[1].split()[0])
                op_done += done
                op_total += tot
            except (ValueError, IndexError):
                pass

    return {
        "total": total,
        "avg_reward": round(avg, 4),
        "perfect_rate": round(perfect / total, 4) if total else 0,
        "perfect_count": perfect,
        "num_failures": len(failures),
        "failure_reasons": dict(failure_reasons),
        "op_completion_rate": round(op_done / op_total, 4) if op_total else 0,
        "reward_distribution": {
            "1.0": sum(1 for r in rewards if r >= 1.0),
            "0.5-0.99": sum(1 for r in rewards if 0.5 <= r < 1.0),
            "0.01-0.49": sum(1 for r in rewards if 0 < r < 0.5),
            "0.0": sum(1 for r in rewards if r == 0),
            "<0": sum(1 for r in rewards if r < 0),
        },
    }


def analyze_structured_data_results(results: List[Dict]) -> Dict[str, Any]:
    """Detailed breakdown of structured data reasoning failures."""
    total = len(results)
    rewards = [r["reward"] for r in results]
    perfect = sum(1 for r in rewards if r >= 1.0)
    avg = sum(rewards) / total if total else 0

    # Per-question accuracy
    q_correct = 0
    q_total = 0
    for r in results:
        reason = r.get("reason", "")
        if "/" in reason:
            try:
                parts = reason.split("/")
                correct = int(parts[0].split()[-1])
                q_correct += correct
                q_total += 3
            except (ValueError, IndexError):
                q_total += 3

    # Question type breakdown from results
    q_type_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        questions = r.get("questions", [])
        reason = r.get("reason", "")
        # Parse individual Q results from reason
        for i, q in enumerate(questions, 1):
            q_text = q.get("text", "")
            q_type = q.get("type", "")
            if "in_stock=true" in q_text or "seats_available > 0" in q_text:
                qcat = "cheapest_available"
            elif "Among" in q_text and ("cheapest" in q_text or "expensive" in q_text):
                qcat = "multi_step_filter"
            elif "between $" in q_text:
                qcat = "conditional_count"
            elif "cheapest" in q_text.lower() and "AND" in q_text:
                qcat = "multi_constraint"
            elif "cheapest" in q_text.lower():
                qcat = "select_cheapest"
            elif any(x in q_text for x in ["2nd", "3rd", "4th", "5th"]):
                qcat = "select_nth"
            elif "total price" in q_text:
                qcat = "compute_sum"
            elif "nonfree" in q_text:
                qcat = "compute_derived"
            elif "How many" in q_text:
                qcat = "count"
            else:
                qcat = "other"

            q_type_results[qcat]["total"] += 1
            # Check if this question was correct from reason
            q_marker = f"Q{i}:"
            if q_marker in reason and "CORRECT" in reason.split(q_marker)[1].split(";")[0]:
                q_type_results[qcat]["correct"] += 1

    return {
        "total": total,
        "avg_reward": round(avg, 4),
        "perfect_rate": round(perfect / total, 4) if total else 0,
        "perfect_count": perfect,
        "question_accuracy": round(q_correct / q_total, 4) if q_total else 0,
        "reward_distribution": {
            "1.0": sum(1 for r in rewards if r >= 1.0),
            "0.67": sum(1 for r in rewards if abs(r - 0.67) < 0.01),
            "0.33": sum(1 for r in rewards if abs(r - 0.33) < 0.01),
            "0.0": sum(1 for r in rewards if r == 0),
        },
        "per_question_type": {
            k: {"accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
                 "total": v["total"]}
            for k, v in sorted(q_type_results.items())
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate games against a vLLM-served model"
    )
    parser.add_argument("--game", required=True,
                        choices=["multistep_task", "structured_data_reasoning"])
    parser.add_argument("--difficulty", type=int, nargs="+", default=[3],
                        help="Difficulty level(s) for structured_data_reasoning (default: 3)")
    parser.add_argument("--base-url", default="http://localhost:9000/v1",
                        help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="Model name on the vLLM server")
    parser.add_argument("--num-seeds", type=int, default=50,
                        help="Number of seeds to evaluate")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--workers", "-w", type=int, default=32,
                        help="Parallel workers")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default="/root/games/game_eval_results",
                        help="Output directory for results")

    args = parser.parse_args()

    client = VLLMChatClient(
        args.base_url, args.model, args.max_tokens, args.temperature,
    )

    # Test connection
    print(f"Model:   {args.model}")
    print(f"Server:  {args.base_url}")
    print(f"Game:    {args.game}")
    print(f"Seeds:   {args.num_seeds} (start={args.start_seed})")
    print(f"Workers: {args.workers}")

    try:
        test = client.generate("You are helpful.", "Say OK.")
        print(f"Connection OK: {test[:80]}\n")
    except Exception as e:
        print(f"Connection FAILED: {e}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.game == "multistep_task":
        difficulties = [None]
    else:
        difficulties = args.difficulty

    for difficulty in difficulties:
        diff_label = f"difficulty={difficulty}" if difficulty is not None else ""
        print(f"\n{'=' * 70}")
        print(f"  {args.game}{' @ ' + diff_label if diff_label else ''}")
        print(f"{'=' * 70}\n")

        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

        if args.game == "multistep_task":
            run_fn = lambda s: run_multistep_episode(client, s)
            analyze_fn = analyze_multistep_results
        else:
            run_fn = lambda s: run_structured_data_episode(client, s, difficulty)
            analyze_fn = analyze_structured_data_results

        results = []
        print_lock = threading.Lock()
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_fn, seed): seed for seed in seeds}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    with print_lock:
                        n_done = len(results)
                        reward = result.get("reward", 0.0)
                        elapsed = time.time() - t0
                        avg_r = sum(r.get("reward", 0) for r in results) / n_done
                        reason_short = result.get("reason", "")[:60]
                        print(f"  [{n_done:3d}/{len(seeds)}] seed={seed:4d} "
                              f"reward={reward:+.3f} avg={avg_r:.3f} "
                              f"({elapsed:.0f}s) {reason_short}")
                except Exception as e:
                    print(f"  ERROR seed={seed}: {e}")

        elapsed = time.time() - t0
        analysis = analyze_fn(results)

        # Print summary
        print(f"\n{'─' * 50}")
        print(f"  SUMMARY: {args.game}{' ' + diff_label if diff_label else ''}")
        print(f"{'─' * 50}")
        print(f"  Episodes:      {analysis['total']}")
        print(f"  Avg reward:    {analysis['avg_reward']:.4f}")
        print(f"  Perfect rate:  {analysis['perfect_rate']:.2%} "
              f"({analysis['perfect_count']}/{analysis['total']})")
        print(f"  Distribution:  {analysis['reward_distribution']}")
        if "failure_reasons" in analysis:
            print(f"  Failure types: {analysis['failure_reasons']}")
        if "question_accuracy" in analysis:
            print(f"  Q accuracy:    {analysis['question_accuracy']:.2%}")
        if "per_question_type" in analysis:
            print(f"  Per Q-type:")
            for qt, info in analysis["per_question_type"].items():
                print(f"    {qt:30s} {info['accuracy']:.2%} ({info['total']})")
        print(f"  Time:          {elapsed:.1f}s")

        # Save
        model_short = args.model.split("/")[-1]
        if difficulty is not None:
            output_path = output_dir / f"{args.game}_d{difficulty}_{model_short}.json"
        else:
            output_path = output_dir / f"{args.game}_{model_short}.json"
        save_config = {
            "game": args.game,
            "model": args.model,
            "num_seeds": args.num_seeds,
            "temperature": args.temperature,
        }
        if difficulty is not None:
            save_config["difficulty"] = difficulty
        with open(output_path, "w") as f:
            json.dump({
                "config": save_config,
                "analysis": analysis,
                "results": results,
            }, f, indent=2, default=str)
        print(f"  Saved:         {output_path}")


if __name__ == "__main__":
    main()
