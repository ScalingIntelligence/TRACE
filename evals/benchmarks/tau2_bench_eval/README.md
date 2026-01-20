# τ²-bench Evaluation Harness

Config-driven evaluation harness for [τ²-bench](https://github.com/sierra-research/tau2-bench) with vLLM support.

## Quick Start

```bash
# 1. Install
cd eval/tau2_bench_eval
uv sync

# 2. Download data files (~2MB)
python setup_data.py

# 3. Run with OpenAI
python main.py --config config.yml --num-tasks 5

# 4. Run with vLLM (start server first)
vllm serve <model> --port 8000 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768
python main.py --config config-vllm-example.yml
```

## Configuration

**config.yml:**
```yaml
domain: airline                    # airline, retail, telecom, mock
agent_llm: gpt-4o                 # Model for agent
user_llm: gpt-4o                  # Model for user simulator
num_trials: 1                     # Trials per task
num_tasks: null                   # Number of tasks (null = all)
max_concurrency: 10               # Concurrent simulations
save_to: null                     # Custom filename prefix
```

**CLI overrides:**
```bash
python main.py --config config.yml --domain retail --num-trials 3
python main.py --config config.yml --agent-llm vllm://my-model --user-llm gpt-4o-mini
```

## vLLM Support

Use `vllm://` prefix for local models:

```yaml
agent_llm: vllm://hazyresearch/qwen-3b-ot3-6k-qwq-r1-complete-rr2
user_llm: vllm://hazyresearch/qwen-3b-ot3-6k-qwq-r1-complete-rr2

vllm:
  base_url: http://localhost:8000/v1
```

**Required vLLM flags:**
- `--enable-auto-tool-choice` - Enable tool calling
- `--tool-call-parser hermes` - Tool call format
- `--max-model-len 32768` - Context for multi-turn conversations

**Mixed configurations** (recommended):
```yaml
agent_llm: vllm://my-model        # Your model being evaluated
user_llm: gpt-4o-mini             # Strong model for user simulation
```

## View Results

```bash
tau2 view  # Interactive result browser
```

Results saved to `data/simulations/`.

## Troubleshooting

**tau2 command not found:** Run `uv sync` or `pip install -e .`

**Model not found:** Use actual model name: `vllm://hazyresearch/qwen-3b-ot3-6k-qwq-r1-complete-rr2`

**Tool choice error:** Add `--enable-auto-tool-choice --tool-call-parser hermes` to vLLM

**Context length errors:** Increase `--max-model-len 32768` or higher

## References

- [τ²-bench GitHub](https://github.com/sierra-research/tau2-bench)
- [Paper](https://arxiv.org/abs/2506.07982)
- [Leaderboard](https://taubench.com)
