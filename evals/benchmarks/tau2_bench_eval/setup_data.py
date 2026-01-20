#!/usr/bin/env python3
"""
Setup script to download only the necessary data files from tau2-bench.
Downloads ~2MB of data files instead of cloning the entire repository.
"""

import urllib.request
from pathlib import Path


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL to output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {output_path.name}...")
    urllib.request.urlretrieve(url, output_path)


def setup_tau2_data():
    """Download essential tau2-bench data files."""
    base_url = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2"

    data_dir = Path(__file__).parent / "data" / "tau2"

    print("=" * 80)
    print("Setting up τ²-bench data files")
    print("=" * 80)
    print(f"Data directory: {data_dir.absolute()}")
    print()

    # User simulator guidelines
    print("Downloading user simulator files...")
    user_sim_files = [
        "user_simulator/simulation_guidelines.md",
    ]

    for filepath in user_sim_files:
        url = f"{base_url}/{filepath}"
        output_path = data_dir / filepath
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"  Warning: Could not download {filepath}: {e}")

    # Domain-specific files
    print("\nDownloading domain files...")
    domains = {
        "airline": ["tasks.json", "split_tasks.json", "policy.md", "db.json"],
        "retail": ["tasks.json", "split_tasks.json", "policy.md", "db.json"],
        "telecom": ["tasks.json", "split_tasks.json", "main_policy.md", "tech_support_workflow.md", "db.toml"],
        "mock": ["tasks.json", "split_tasks.json", "policy.md", "db.json"],
    }

    for domain, filenames in domains.items():
        print(f"  Domain: {domain}")
        for filename in filenames:
            url = f"{base_url}/domains/{domain}/{filename}"
            output_path = data_dir / "domains" / domain / filename
            try:
                download_file(url, output_path)
            except Exception as e:
                print(f"    Warning: Could not download {domain}/{filename}: {e}")

    # Create simulations directory (tau2 saves to data/tau2/simulations/)
    print("\nCreating output directories...")
    # tau2-bench saves results under data/tau2/simulations/
    simulations_dir = data_dir / "simulations"
    simulations_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {simulations_dir}")

    print()
    print("=" * 80)
    print("✓ Setup complete!")
    print("=" * 80)
    print(f"\nData files installed to: {data_dir.parent.parent.absolute()}")
    print(f"Simulations will be saved to: {simulations_dir.absolute()}")
    print("\nYou can now run evaluations:")
    print("  python main.py --config config.yml --num-tasks 5")


if __name__ == "__main__":
    setup_tau2_data()
