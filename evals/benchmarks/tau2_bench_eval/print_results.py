#!/usr/bin/env python3
"""Print a summary table of tau2 bench simulation results."""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SIMULATIONS_DIR = Path(__file__).parent / "data" / "simulations"
MIN_AIRLINE = 50
MIN_RETAIL = 114
TOTAL_PROBLEMS = 164


def parse_filename(filename):
    """Extract (identifier, domain) from a simulation filename.

    E.g. 'qwen-3-30b-airline-qwen3-30b.json' -> ('qwen-3-30b-qwen3-30b', 'airline')
    """
    stem = filename.removesuffix(".json")
    for domain in ("airline", "retail"):
        pattern = f"-{domain}-"
        idx = stem.find(pattern)
        if idx != -1:
            before = stem[:idx]
            after = stem[idx + len(pattern):]
            identifier = f"{before}-{after}"
            return identifier, domain
    return None, None


def compute_success_rate(filepath):
    """Return (num_successes, num_total) from a simulation JSON."""
    with open(filepath) as f:
        data = json.load(f)
    sims = data["simulations"]
    total = len(sims)
    successes = sum(1 for s in sims if s["reward_info"]["reward"] == 1.0)
    return successes, total


def main():
    results = defaultdict(dict)  # identifier -> {domain: (successes, total)}

    for fname in sorted(os.listdir(SIMULATIONS_DIR)):
        if not fname.endswith(".json"):
            continue
        identifier, domain = parse_filename(fname)
        if identifier is None:
            print(f"WARNING: could not parse filename: {fname}", file=sys.stderr)
            continue

        filepath = SIMULATIONS_DIR / fname
        successes, total = compute_success_rate(filepath)

        # Skip files with fewer problems than expected
        if domain == "airline" and total < MIN_AIRLINE:
            print(f"SKIPPING {fname}: only {total} problems (need {MIN_AIRLINE})", file=sys.stderr)
            continue
        if domain == "retail" and total < MIN_RETAIL:
            print(f"SKIPPING {fname}: only {total} problems (need {MIN_RETAIL})", file=sys.stderr)
            continue

        results[identifier][domain] = (successes, total)

    # Build rows
    rows = []
    for identifier, domains in results.items():
        airline_str = ""
        retail_str = ""
        airline_successes = 0
        retail_successes = 0
        has_airline = "airline" in domains
        has_retail = "retail" in domains

        if has_airline:
            s, t = domains["airline"]
            airline_successes = s
            airline_str = f"{s / t * 100:.1f}"

        if has_retail:
            s, t = domains["retail"]
            retail_successes = s
            retail_str = f"{s / t * 100:.1f}"

        if has_airline and has_retail:
            overall = (airline_successes + retail_successes) / TOTAL_PROBLEMS * 100
            overall_str = f"{overall:.1f}"
        else:
            overall_str = ""

        rows.append((identifier, airline_str, retail_str, overall_str))

    # Sort by overall descending; entries without overall go to the bottom
    def sort_key(row):
        if row[3]:
            return (0, -float(row[3]))
        # For entries with only one domain, sort by whichever percentage exists
        val = float(row[1]) if row[1] else float(row[2]) if row[2] else 0
        return (1, -val)

    rows.sort(key=sort_key)

    # Print CSV (paste-friendly for Google Sheets)
    headers = ("Identifier", "Airline (%)", "Retail (%)", "Overall (%)")
    print(",".join(headers))
    for row in rows:
        print(",".join(row))


if __name__ == "__main__":
    main()
