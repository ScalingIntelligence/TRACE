# Model vs Model Evaluation

Run Kuhn Poker games between different LLMs to compare their performance.

## Quick Start

```bash
cd experiments/model_vs_model
python run_evaluation.py
```

## Usage

```bash
# Run with all available models (default: 100 games per matchup)
python run_evaluation.py

# Specify models and number of games
python run_evaluation.py --models qwen-4b qwen-8b --num_games 50

# Save results to JSON
python run_evaluation.py --output results.json

# Enable verbose logging (see each move)
python run_evaluation.py --verbose
```

## Available Models

| Name | HuggingFace Path |
|------|------------------|
| `qwen-4b` | `Qwen/Qwen3-4B` |
| `qwen-4b-instruct` | `Qwen/Qwen3-4B-Instruct-2507` |
| `qwen-8b` | `Qwen/Qwen3-8B` |
| `qwen-14b` | `Qwen/Qwen3-14B` |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | all | Space-separated list of models to test |
| `--num_games` | 100 | Games per matchup |
| `--num_rounds` | 1 | Rounds per game |
| `--temperature` | 0.7 | Sampling temperature |
| `--output` | None | Save results to JSON file |
| `--verbose` | False | Print detailed game logs |

## Output

The evaluation runs all ordered pairwise matchups (A vs B and B vs A separately, since position matters). Results include:

- **Per-matchup stats**: Win rates for first/second player
- **Overall summary**: Each model's total win rate, win rate as first player, win rate as second player

## Configuration

Edit `eval_config.py` to:
- Add/modify available models
- Change default parameters
- Toggle thinking mode (models can reason before answering)

## Requirements

- PyTorch with CUDA
- `unsloth` for fast model loading
- Models are loaded in 4-bit quantization
