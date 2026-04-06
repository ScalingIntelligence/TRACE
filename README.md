# TRACE: Capability-Targeted Agentic Training

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](#)
[![Status](https://img.shields.io/badge/Status-Code_Coming_Soon-blue.svg)](#)

This repository contains the official implementation for **TRACE (Turning Recurrent Agent failures into Capability-targeted training Environments)**. TRACE is an end-to-end system for environment-specific agent self-improvement. 

Large Language Models (LLMs) deployed in complex agentic environments often fail because they lack specific, underlying capabilities. Existing approaches typically rely on generic synthetic data or direct reinforcement learning (RL) on the target environment, which forces the model to learn these capabilities implicitly and inefficiently. TRACE solves this by automatically identifying the exact capabilities an agent lacks, synthesizing targeted micro-environments to isolate and train those capabilities, and routing to the appropriate adapter at inference time.

## 🚀 How TRACE Works

The TRACE pipeline consists of four automated steps:

1. **Capability Selection:** An analysis agent contrasts successful and failed trajectories from the base model in the target environment. It identifies and ranks the specific capabilities that meaningfully distinguish successes from failures.
2. **Synthetic Environment Generation:** For each identified deficit, a generation agent constructs a capability-targeted synthetic training environment. This environment preserves the target environment's interface (tool schemas, interaction protocols) while isolating the missing capability for verifiable training.
3. **GRPO Training:** We train a separate Low-Rank Adaptation (LoRA) module for each capability-specific synthetic environment using Group Relative Policy Optimization (GRPO).
4. **Select & Adapt:** At inference, a lightweight router uses the base model to classify the task and activate only the single, most relevant LoRA adapter for generation. 

## 📊 Key Results

TRACE demonstrates significant improvements and generalization across different complex environments:

* **$\tau^{2}$-Bench (Customer Service):** Improves upon the base agent by +14.1 points, achieving an overall pass rate of 47.0%. It scales more efficiently than baselines, outperforming GRPO and GEPA by +9.2 and +7.4 points given the same rollout budget.
* **ToolSandBox (Tool Use):** Achieves a mean similarity score of 0.552, improving over the base model by +0.141 points and +7 perfect scores.

## ⏳ Code Release Status

The codebase is currently undergoing final cleanup and formatting for public release. We will be publishing the complete TRACE pipeline soon, which will include:

* The prompt templates and agentic pipeline for **Contrastive Capability Identification**.
* The **Synthetic Environment Generator** to create isolated micro-environments.
* The **GRPO Training Scripts** for optimizing capability-specific LoRA adapters.
* The **Inference Router** for dynamic adapter selection during evaluation.

Please star or watch this repository to be notified when the code is pushed!

## 📝 Citation

If you find this work helpful in your research, please consider citing our paper:

```bibtex
@article{kang2026trace,
  title={TRACE: Capability-Targeted Agentic Training},
  author={Kang, Hangoo and Suresh, Tarun and Saad-Falcon, Jon and Mirhoseini, Azalia},
  journal={arXiv preprint},
  year={2026}
}
