#!/usr/bin/env python3
"""Preprocess tau2-bench trajectory JSONs to wrap assistant reasoning in <think> tags.

Qwen3-Max puts reasoning text in the `content` field of assistant messages that
also have `tool_calls`.  The SFT buffer's `_make_completion_text` only keeps
the tool-call JSON and discards that content, so the reasoning is lost from
training.

This script wraps such content in `<think>...</think>` tags so that
`_extract_thinking()` in sft_buffer.py picks it up and prepends it to the
JSON action in the completion text.  The model then learns:

    <think>reasoning...</think>
    {"name": "tool_name", "arguments": {...}}

Usage:
    python preprocess_think_tags.py \
        --input-dir /path/to/trajectories \
        --output-dir /path/to/output

    # Or specific files:
    python preprocess_think_tags.py \
        --input-files file1.json file2.json \
        --output-dir /path/to/output
"""
import argparse
import json
import copy
from pathlib import Path


def wrap_think_tags(data: dict) -> dict:
    """Wrap assistant reasoning content in <think> tags.

    For every assistant message that has BOTH non-empty `content` AND
    `tool_calls`, wraps the content as `<think>\n{content}\n</think>`.

    Returns a deep copy with modifications applied.
    """
    data = copy.deepcopy(data)
    stats = {"total_assistant": 0, "wrapped": 0, "text_only": 0, "tool_only": 0}

    for sim in data.get("simulations", []):
        for msg in sim.get("messages", []):
            if msg["role"] != "assistant":
                continue

            stats["total_assistant"] += 1
            content = msg.get("content", "") or ""
            has_tool_calls = bool(msg.get("tool_calls"))
            has_content = bool(content.strip())

            if has_tool_calls and has_content:
                # Wrap reasoning in <think> tags
                msg["content"] = f"<think>\n{content.strip()}\n</think>"
                stats["wrapped"] += 1
            elif has_tool_calls and not has_content:
                stats["tool_only"] += 1
            else:
                stats["text_only"] += 1

    return data, stats


def process_file(input_path: Path, output_path: Path) -> dict:
    """Process a single trajectory JSON file."""
    with open(input_path, "r") as f:
        data = json.load(f)

    processed, stats = wrap_think_tags(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(processed, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Wrap assistant reasoning in <think> tags for SFT training"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=str,
                       help="Directory containing trajectory JSON files")
    group.add_argument("--input-files", nargs="+", type=str,
                       help="Specific trajectory JSON files to process")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        input_files = sorted(input_dir.glob("*.json"))
    else:
        input_files = [Path(f) for f in args.input_files]

    if not input_files:
        print("No JSON files found.")
        return

    total_stats = {"total_assistant": 0, "wrapped": 0, "text_only": 0, "tool_only": 0}

    for input_path in input_files:
        output_path = output_dir / input_path.name
        stats = process_file(input_path, output_path)

        for k in total_stats:
            total_stats[k] += stats[k]

        print(f"  {input_path.name}: "
              f"{stats['wrapped']} wrapped, "
              f"{stats['text_only']} text-only, "
              f"{stats['tool_only']} tool-only "
              f"(of {stats['total_assistant']} assistant turns)")

    print(f"\nTotal: {total_stats['wrapped']} assistant turns wrapped with <think> tags "
          f"(out of {total_stats['total_assistant']} total)")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
