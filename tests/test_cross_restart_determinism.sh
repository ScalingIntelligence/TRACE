#!/bin/bash
# Cross-restart determinism test for weighted LoRA
#
# Runs tau2 airline tasks twice (with server restart in between),
# then compares the conversation logs for exact match.
#
# Usage: bash tests/test_cross_restart_determinism.sh

set -e
cd /home/ubuntu/hangook/games

PYTHON=/home/ubuntu/miniconda3/envs/games/bin/python3
CONFIG=tests/config-determinism-test.yml
RUN1_DIR=/tmp/determinism-run1
RUN2_DIR=/tmp/determinism-run2
SERVER_LOG=/tmp/vllm_server_determinism.log
PORT=8080

# Clean previous runs
rm -rf "$RUN1_DIR" "$RUN2_DIR"

start_server() {
    echo "=== Starting vLLM server ==="
    HF_HOME=/workspace/.cache/huggingface \
    VLLM_RPC_TIMEOUT=2000 \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
    PYTHONDONTWRITEBYTECODE=1 \
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --host 0.0.0.0 --port $PORT --dtype bfloat16 \
        --max-model-len 32000 --enable-lora --max-loras 2 \
        --gpu-memory-utilization 0.9 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --no-enable-prefix-caching --max-num-seqs 1 \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    # Wait for server to be ready
    for i in $(seq 1 60); do
        if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "data"; then
            echo "Server ready after ${i}0 seconds"
            return 0
        fi
        sleep 10
    done
    echo "ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null
    return 1
}

stop_server() {
    echo "=== Stopping vLLM server ==="
    # Kill all processes on port 8080
    for pid in $(lsof -ti:$PORT 2>/dev/null); do
        kill $pid 2>/dev/null
    done
    # Wait for port to be free
    for i in $(seq 1 30); do
        if ! ss -tlnp | grep -q ":$PORT "; then
            echo "Server stopped"
            return 0
        fi
        sleep 2
    done
    echo "WARNING: Port $PORT still in use"
}

run_tau2() {
    local save_dir=$1
    echo "=== Running tau2 bench (saving to $save_dir) ==="
    cd /home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval
    $PYTHON main.py --config /home/ubuntu/hangook/games/$CONFIG --save-to "$save_dir"
    cd /home/ubuntu/hangook/games
}

compare_results() {
    echo ""
    echo "============================================================"
    echo "=== Comparing results ==="
    echo "============================================================"

    # Find result files
    local file1=$(find "$RUN1_DIR" -name "*.json" -type f | head -1)
    local file2=$(find "$RUN2_DIR" -name "*.json" -type f | head -1)

    if [ -z "$file1" ] || [ -z "$file2" ]; then
        echo "ERROR: Could not find result files"
        echo "Run1 files: $(find $RUN1_DIR -name '*.json' -type f)"
        echo "Run2 files: $(find $RUN2_DIR -name '*.json' -type f)"
        return 1
    fi

    echo "File 1: $file1"
    echo "File 2: $file2"

    # Compare using Python to extract and compare conversations
    $PYTHON -u -c "
import json, sys

with open('$file1') as f:
    run1 = json.load(f)
with open('$file2') as f:
    run2 = json.load(f)

# Compare task results
all_match = True

# Handle both list and dict formats
if isinstance(run1, list):
    results1 = run1
    results2 = run2
else:
    results1 = run1.get('results', run1.get('simulations', []))
    results2 = run2.get('results', run2.get('simulations', []))

print(f'Run 1: {len(results1)} results')
print(f'Run 2: {len(results2)} results')

if len(results1) != len(results2):
    print('FAIL: Different number of results')
    sys.exit(1)

for i, (r1, r2) in enumerate(zip(results1, results2)):
    # Get task ID
    tid1 = r1.get('task_id', r1.get('id', i))
    tid2 = r2.get('task_id', r2.get('id', i))

    # Get conversations
    conv1 = r1.get('conversation', r1.get('chat_history', r1.get('messages', [])))
    conv2 = r2.get('conversation', r2.get('chat_history', r2.get('messages', [])))

    # Get reward/score
    reward1 = r1.get('reward', r1.get('score', None))
    reward2 = r2.get('reward', r2.get('score', None))

    # Compare conversations
    conv1_str = json.dumps(conv1, sort_keys=True)
    conv2_str = json.dumps(conv2, sort_keys=True)

    if conv1_str == conv2_str:
        print(f'Task {tid1}: MATCH (reward={reward1})')
    else:
        all_match = False
        print(f'Task {tid1}: DIFFER (reward1={reward1}, reward2={reward2})')
        # Find first difference
        msgs1 = conv1 if isinstance(conv1, list) else []
        msgs2 = conv2 if isinstance(conv2, list) else []
        for j, (m1, m2) in enumerate(zip(msgs1, msgs2)):
            content1 = m1.get('content', '') if isinstance(m1, dict) else str(m1)
            content2 = m2.get('content', '') if isinstance(m2, dict) else str(m2)
            if content1 != content2:
                print(f'  First diff at message {j}:')
                print(f'    Run1: {str(content1)[:200]}')
                print(f'    Run2: {str(content2)[:200]}')
                break
        else:
            if len(msgs1) != len(msgs2):
                print(f'  Different number of messages: {len(msgs1)} vs {len(msgs2)}')

if all_match:
    print()
    print('============================================================')
    print('PASS: All tasks produce identical results across restarts')
    print('============================================================')
else:
    print()
    print('============================================================')
    print('FAIL: Some tasks differ across restarts')
    print('============================================================')
    sys.exit(1)
"
}

# Main
echo "============================================================"
echo "Cross-Restart Determinism Test"
echo "============================================================"

# Check if server is already running
if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "data"; then
    echo "Server already running, using it for Run 1"
else
    start_server
fi

# Run 1
run_tau2 "$RUN1_DIR"

# Stop server
stop_server
sleep 5

# Start fresh server
start_server

# Run 2
run_tau2 "$RUN2_DIR"

# Compare
compare_results
