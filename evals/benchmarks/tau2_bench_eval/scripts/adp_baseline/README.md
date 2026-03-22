# ADP Baseline: End-to-End Pipeline

Train Qwen3-30B-A3B-Instruct on ADP data and evaluate on tau-bench.

## Steps

```bash
# Step 1: Download & convert ADP data
python download_and_convert_adp.py

# Step 2: Install LLaMA-Factory
bash setup_llamafactory.sh

# Step 3: Train (LoRA, matching our expert config)
bash train.sh

# Step 4: Evaluate on tau-bench
bash eval.sh
```

See each script for details.
