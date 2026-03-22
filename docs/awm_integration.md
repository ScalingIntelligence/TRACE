# AWM (Agent World Model) Integration

Paper: [Agent World Model: Infinity Synthetic Environments for Agentic RL](https://arxiv.org/abs/2602.10090)
Repo: [Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model)

## Overview

AWM provides 1,000 pre-generated, SQLite-backed, tool-use environments exposed via a unified interface for large-scale multi-turn agentic RL training. Each environment is a complete FastAPI application with CRUD endpoints, backed by a SQLite database with realistic sample data.

Our integration wraps these environments into the existing `GameEnv` protocol so they work seamlessly with the GRPO/PPO training infrastructure.

## Files

| File | Purpose |
|------|---------|
| `awm_game.py` | Game environment wrapper (`AWMGame` class) |
| `setup_awm.sh` | One-time setup: downloads data, creates databases |
| `game_registry.py` | Registers `awm_tool_calling` game |

## Setup (one-time)

```bash
bash setup_awm.sh
```

This does three things:
1. Installs Python dependencies (`sqlalchemy`, `httpx`)
2. Downloads the pre-generated 1K environments from HuggingFace (`Snowflake/AgentWorldModel-1K`) into `./outputs/`
3. Creates SQLite databases from schemas + sample data in `./outputs/databases/`

Custom data directory:
```bash
bash setup_awm.sh /path/to/custom/data_dir
```

## Training

### GRPO training
```bash
# With vLLM server (recommended for speed):
VLLM_BASE_URL=http://localhost:8000 python train_grpo.py --game awm_tool_calling

# Without vLLM (HF local generation):
python train_grpo.py --game awm_tool_calling
```

### Collect rollouts (for inspection/SFT)
```bash
python collect_rollouts.py --env_type awm_tool_calling
```

### Custom data directory
```bash
AWM_DATA_DIR=/path/to/outputs python train_grpo.py --game awm_tool_calling
```

## Architecture

### Episode Flow

```
reset(seed)
  |
  v
Pick random scenario + task (from 1K environments)
  |
  v
Copy SQLite DB to temp dir (isolation)
  |
  v
exec() FastAPI code -> TestClient (no HTTP server)
  |
  v
Extract tool schemas from OpenAPI spec
  |
  v
[Training loop]
  Model generates tool call -> step(action)
    -> Execute via TestClient
    -> Return result to model
    -> Repeat until respond_to_user or max_steps
  |
  v
Run code-based verifier on initial vs final DB
  |
  v
Reward: 1.0 (complete) or 0.0 (incomplete)
  |
  v
Cleanup temp files
```

### Key Design Decisions

1. **FastAPI TestClient instead of MCP servers**: The AWM paper uses MCP servers over HTTP for each environment. For RL training with hundreds of parallel rollouts, starting/stopping HTTP servers would be prohibitively slow. Instead, we `exec()` the generated FastAPI code and use `TestClient` for direct in-process calls. This gives zero HTTP overhead while reusing the exact same generated environment code.

2. **Code-based verifier for rewards**: AWM supports two verification modes:
   - **SQL mode** (recommended by AWM): Runs SQL queries on the DB, then uses an LLM judge to classify the result. Requires an external LLM API call per verification.
   - **Code mode**: Deterministic Python function that checks DB state and returns `"complete"` or `"others"`. No external LLM needed.

   We default to code mode for RL training because it's deterministic, fast, and doesn't require an external API. Falls back to SQL mode with heuristic scoring if code verifier is unavailable.

3. **Qwen3 native tool-calling format**: The model generates tool calls using Qwen3's standard format (via `apply_chat_template(tools=...)`), identical to how existing games (multistep_task, tau_tool_calling) work. This means no format adaptation is needed for training.

4. **Database isolation**: Each episode copies the scenario's database to a temp directory. This ensures parallel rollouts don't interfere with each other, and each episode starts from a clean state.

### Data Files

All data lives in the `AWM_DATA_DIR` directory (default: `./outputs/`):

| File | Content |
|------|---------|
| `gen_envs.jsonl` | Generated FastAPI environment code per scenario |
| `gen_tasks.jsonl` | User tasks (10 per scenario) |
| `gen_db.jsonl` | Database schemas (DDL statements) |
| `gen_sample.jsonl` | Sample data for each database |
| `gen_verifier.pure_code.jsonl` | Code-based verifiers (preferred) |
| `gen_verifier.jsonl` | SQL-based verifiers (fallback) |
| `databases/` | Pre-built SQLite `.db` files |

### AWMGame API

```python
from awm_game import AWMGame

game = AWMGame(data_dir="./outputs", max_steps=20)

# GameEnv protocol
game.reset(seed=42)
game.observe(player_id=0)    # Returns placeholder (use get_messages instead)
game.legal_actions()          # Returns generic JSON template
game.step(action_json)        # Process tool call

# Structured messages interface (used by training loop)
game.get_system_prompt()      # System instructions
game.get_tool_schemas()       # OpenAI-format tool definitions
game.get_messages()           # Conversation history
game.get_tool_schemas_compact()  # Schemas without descriptions (saves tokens)
game.get_summary()            # Episode summary dict
```

### Reward Structure

| Outcome | Reward |
|---------|--------|
| Task verified as complete | 1.0 |
| Task incomplete / verification failed | 0.0 |
| No verifier available, but tools were called | 0.3 |
| No verifier and no tool calls | 0.0 |
| Invalid action (bad JSON) | 0.0 |

## AWM Synthesis Pipeline (optional)

The AWM repo also includes a full synthesis pipeline to generate new environments from scratch. This is **not required** for training (we use pre-generated environments), but can be used to create more:

```bash
# Install the full AWM package
pip install git+https://github.com/Snowflake-Labs/agent-world-model.git

# Set LLM provider
export AWM_SYN_LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key
export AWM_SYN_OVERRIDE_MODEL=gpt-4o
export EMBEDDING_OPENAI_API_KEY=your-key

# Run full pipeline (generates ~1000 environments)
awm gen all --input outputs/seed_scenario.jsonl --output_dir outputs

# Or run individual steps:
awm gen scenario --input_path outputs/seed_scenario.jsonl --output_path outputs/gen_scenario.jsonl --target_count 1000
awm gen task --input outputs/gen_scenario.jsonl --output outputs/gen_tasks.jsonl
awm gen db --input outputs/gen_tasks.jsonl --output outputs/gen_db.jsonl
awm gen sample --input_task outputs/gen_tasks.jsonl --input_db outputs/gen_db.jsonl --output outputs/gen_sample.jsonl
awm gen spec --input_task outputs/gen_tasks.jsonl --input_db outputs/gen_db.jsonl --output outputs/gen_spec.jsonl
awm gen env --input_spec outputs/gen_spec.jsonl --input_db outputs/gen_db.jsonl --output outputs/gen_envs.jsonl
awm gen verifier --mode code --input_task outputs/gen_tasks.jsonl --output outputs/gen_verifier.pure_code.jsonl
```

## AWM Pre-trained Models

Snowflake provides pre-trained models on HuggingFace:
- `Snowflake/Arctic-AWM-4B`
- `Snowflake/Arctic-AWM-8B`
- `Snowflake/Arctic-AWM-14B`

These can be served with vLLM and used as baselines:
```bash
vllm serve Snowflake/Arctic-AWM-4B --host 127.0.0.1 --port 8000
```
