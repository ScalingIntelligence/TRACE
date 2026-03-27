#!/bin/bash
# Step 2: Install LLaMA-Factory
set -e

cd /home/ubuntu

if [ ! -d "LLaMA-Factory" ]; then
    echo "Cloning LLaMA-Factory..."
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi

cd LLaMA-Factory
pip install -e ".[torch,metrics]"

echo "LLaMA-Factory installed successfully."
echo "Verify: llamafactory-cli version"
llamafactory-cli version
