import argparse
import glob
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Create a thread-local storage for tokenizer
thread_local = threading.local()


def get_tokenizer(model_name):
    """Get or create thread-local tokenizer"""
    if not hasattr(thread_local, "tokenizer"):
        thread_local.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return thread_local.tokenizer


def normalize_model_name(path):
    """Extract and normalize model name from path"""
    parts = path.split("/")
    for part in parts[::-1]:
        if "checkpoint" in part:
            idx = parts.index(part)
            model_name = parts[idx - 1]
            checkpoint = part
            return f"{model_name}-{checkpoint}"
        if "global_step" in part:
            idx = parts.index(part)
            model_name = parts[idx - 1]
            return f"{model_name}-{part}"
    for part in reversed(parts):
        if any(x in part.lower() for x in ["llama", "qwen", "gpt", "mistral"]):
            return f"{part}-checkpoint-final"
    return "unknown_model"


def get_benchmark_name(path):
    """Extract benchmark name from path"""
    parts = path.split("/")
    return parts[-2]


def get_jsonl_path(metrics_file):
    """Get corresponding jsonl file path"""
    # Get the directory containing the metrics file
    metric_folder = os.path.dirname(metrics_file)

    # The JSONL file should be in the same directory with a .jsonl extension
    # and without the '_metrics' suffix
    base_name = os.path.basename(metrics_file).replace("_metrics.json", "")
    jsonl_file = os.path.join(metric_folder, f"{base_name}.jsonl")

    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")

    return jsonl_file


def calculate_avg_tokens(jsonl_path, tokenizer):
    """Calculate average tokens in the first code element"""
    if not os.path.exists(jsonl_path):
        print(f"Warning: JSONL file not found: {jsonl_path}")
        return 0

    total_tokens = 0
    count = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if (
                    "code" in data
                    and isinstance(data["code"], list)
                    and len(data["code"]) > 0
                ):
                    tokens = len(tokenizer.encode(data["code"][0]))
                    total_tokens += tokens
                    count += 1
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")
        return 0

    return total_tokens / count if count > 0 else 0


def process_file(args):
    """Process a single metrics file"""
    metrics_file, model_name = args
    try:
        # Get model and benchmark names
        model_name_norm = normalize_model_name(metrics_file)
        benchmark = get_benchmark_name(metrics_file)

        # Read metrics
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
            acc = metrics.get("acc", 0)
            invalid_count = metrics.get("empty_samples", 0) / metrics.get(
                "num_samples", 1
            )

        # Get corresponding jsonl file
        jsonl_file = get_jsonl_path(metrics_file)
        tokenizer = get_tokenizer(model_name)
        avg_tokens = calculate_avg_tokens(jsonl_file, tokenizer)

        return (
            model_name_norm,
            benchmark,
            {"Score": acc, "Avg Tokens": avg_tokens, "Invalid Format": invalid_count},
        )

    except Exception as e:
        print(f"Error processing {metrics_file}: {e}")
        return None


def collect_results(base_dir, model_name, num_threads=8):
    # Initialize results storage
    results = defaultdict(lambda: defaultdict(dict))

    # Find all metrics.json files
    metrics_files = glob.glob(f"{base_dir}/**/test_*metrics.json", recursive=True)

    # Create arguments for parallel processing
    process_args = [(f, model_name) for f in metrics_files]

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list(
            tqdm(
                executor.map(process_file, process_args),
                total=len(metrics_files),
                desc="Processing files",
            )
        )

        # Collect results
        for result in futures:
            if result is not None:
                model_name, benchmark, metrics = result
                results[model_name][benchmark] = metrics

    return results


def write_to_csv(args, results, df, csv_path):
    """Sync results, CSV table and plots to wandb"""
    # Initialize wandb run
    # Save metrics to CSV with eval-{benchmark}/metrics format
    metrics_data = []
    for model, benchmarks in results.items():
        for benchmark, metrics in benchmarks.items():
            for metric, value in metrics.items():
                metrics_data.append(
                    {
                        "key": f"eval-{benchmark}/{metric}",
                        "value": value,
                    }
                )

    # Convert to DataFrame and save
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(csv_path, index=False)


def main(args):
    base_dir = args.base_dir
    model_name = args.model_name

    # Parse benchmarks if specified
    benchmarks = None
    if args.benchmarks:
        benchmarks = set(args.benchmarks.split(","))

    # Collect results
    print("Collecting results...")
    results = collect_results(base_dir, model_name, args.num_threads)

    # Filter results if benchmarks specified
    if benchmarks:
        filtered_results = defaultdict(lambda: defaultdict(dict))
        for model, model_results in results.items():
            for benchmark, metrics in model_results.items():
                if benchmark in benchmarks:
                    filtered_results[model][benchmark] = metrics
        results = filtered_results

    # Flatten results for DataFrame
    rows = []
    for model, benchmarks in results.items():
        for benchmark, metrics in benchmarks.items():
            row = {"model": model, "benchmark": benchmark}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)

    # Save to CSV
    # output_file = args.output_path
    # df.to_csv(output_file, index=False)
    # print(f"\nResults saved to {output_file}")

    write_to_csv(args, results, df, args.output_path)
    print("Wandb sync completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir", type=str, default="outputs/project/reasonshort/weiliu/cot/output"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen-math-7B-S100-qwq-fs-7k8-8192len-5e-6-rope10-bsz64",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="gsm8k,math,minerva_math,olympiadbench,aime24,amc23",
        help="Comma-separated list of benchmarks to include",
    )

    args = parser.parse_args()

    main(args)
