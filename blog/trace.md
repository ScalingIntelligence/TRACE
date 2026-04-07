---
title: "TRACE: Capability-Targeted Agentic Training"
authors:
  - key: hangookang
    equal: true
  - key: tarunsuresh
    equal: true
  - key: jonsaadfalcon
  - key: azaliamirhoseini
tags:
  - Agents
  - Reinforcement Learning
  - Self-improving AI
venue: none
year: 2025
date: 2025-07-07
teaser: "TRACE automatically identifies capability deficits in LLM agents and synthesizes targeted training environments to fix them."
materials:
  - name: Codebase
    url: https://github.com/ScalingIntelligence/TRACE
    type: code
  - name: Paper
    url: https://arxiv.org/abs/2504.04736
    type: file-pdf
---

# TL;DR

LLM agents fail in complex environments because they lack specific capabilities — but standard RL training doesn't tell the model *which* ones are missing. **TRACE** (Turning Recurrent Agent failures into Capability-targeted training Environments) automatically identifies an agent's capability deficits, synthesizes targeted training environments for each, and trains lightweight LoRA adapters via RL. TRACE improves over the base agent by **+14.1 points** on &tau;<sup>2</sup>-Bench and **+7 perfect scores** on ToolSandBox, outperforming the strongest baseline by **+7.4 points** and **+4 perfect scores**.

---

## Table of Contents

- [Overview](#overview)
- [Main Results](#main-results)
- [Scaling Behavior](#scaling-behavior)
- [Citation](#citation)

---

## Overview

LLM agents in environments like customer service or tool use must exercise multiple capabilities across different tasks. Training directly on the target environment forces the model to learn these capabilities implicitly — making learning sparse and sample-inefficient. TRACE decomposes the problem into four steps:

1. **Capability Selection.** Roll out the base agent, then contrast successful and failed trajectories to identify the specific capabilities the agent lacks.
2. **Synthetic Environment Generation.** For each identified capability deficit, synthesize a targeted training environment that isolates and rewards exercising that capability.
3. **GRPO Training.** Train a lightweight LoRA adapter on each capability-specific synthetic environment via RL.
4. **Select & Adapt.** At inference, route each task to the relevant LoRA adapter using the base model as a classifier.

<img src="../figures/diagram.png" alt="Overview of TRACE: an end-to-end system with four steps — capability selection, synthetic environment generation, GRPO training, and select-and-adapt routing at inference." style="width: 100%; height: auto;">
<p style="text-align: center;"><i>Overview of TRACE. An analysis agent identifies capability deficits from the agent's trajectories. For each deficit, a generation agent synthesizes a targeted training environment. A LoRA adapter is trained via GRPO on each environment, and a router selects the appropriate adapter at inference.</i></p>

---

## Main Results

We evaluate TRACE using Qwen3-30B-A3B on two benchmarks: **&tau;<sup>2</sup>-Bench** (customer service with Airline and Retail domains) and **ToolSandBox** (stateful tool use, 129 scenarios).

### &tau;<sup>2</sup>-Bench: Pass Rate

<table style="width: 100%; text-align: center;">
<thead>
<tr>
<th style="text-align: left;">Approach</th>
<th>Airline (%)</th>
<th>Retail (%)</th>
<th>Overall (%)</th>
</tr>
</thead>
<tbody>
<tr><td style="text-align: left;">Base Model</td><td>24.0</td><td>36.8</td><td>32.9</td></tr>
<tr><td style="text-align: left;">ADP</td><td>28.0</td><td>34.2</td><td>32.3</td></tr>
<tr><td style="text-align: left;">GRPO on Target</td><td>32.0</td><td>40.4</td><td>37.8</td></tr>
<tr><td style="text-align: left;">AWM</td><td>32.0</td><td>41.2</td><td>38.4</td></tr>
<tr><td style="text-align: left;">GEPA</td><td>38.0</td><td>40.4</td><td>39.6</td></tr>
<tr><td style="text-align: left;">Single Capability GRPO (Ours)</td><td>34.0</td><td>43.0</td><td>40.3</td></tr>
<tr style="font-weight: bold;"><td style="text-align: left;">TRACE (Ours)</td><td>44.0</td><td>48.2</td><td>47.0</td></tr>
</tbody>
</table>

### ToolSandBox: Perfect Score & Mean Similarity

<table style="width: 100%; text-align: center;">
<thead>
<tr>
<th style="text-align: left;">Model</th>
<th>Perfect</th>
<th>Mean Sim.</th>
</tr>
</thead>
<tbody>
<tr><td style="text-align: left;">Base (Qwen3-30B-A3B)</td><td>19/129</td><td>0.411</td></tr>
<tr><td style="text-align: left;">ADP</td><td>19/129</td><td>0.422</td></tr>
<tr><td style="text-align: left;">AWM</td><td>20/129</td><td>0.504</td></tr>
<tr><td style="text-align: left;">GRPO on Target</td><td>22/129</td><td>0.519</td></tr>
<tr><td style="text-align: left;">GEPA</td><td>22/129</td><td>0.520</td></tr>
<tr><td style="text-align: left;">Single Capability GRPO (Ours)</td><td>22/129</td><td>0.514</td></tr>
<tr style="font-weight: bold;"><td style="text-align: left;">TRACE (Ours)</td><td>26/129</td><td>0.552</td></tr>
</tbody>
</table>

<br>

Even a **single adapter** trained on one synthesized capability environment surpasses large-scale general-purpose methods like AWM and ADP — while using only a quarter of the optimization steps. Combining multiple capability-specific adapters with routing, TRACE achieves the strongest results on both benchmarks.

---

## Scaling Behavior

### Scaling with Number of Rollouts

TRACE scales more efficiently with training rollouts than both GRPO (direct RL on the target environment) and GEPA (evolutionary prompt optimization). On &tau;<sup>2</sup>-Bench, TRACE shows consistent, monotonic improvement up to **47.0%** at 5,120 rollouts, while GRPO stalls at 37.8% and GEPA plateaus at 39.6%. A consistent trend appears on ToolSandBox.

<div style="display: flex; gap: 16px; align-items: center;">
  <img src="../figures/scale_rollouts_taubench.png" alt="Pass rate scaling with number of rollouts on tau2-Bench" style="max-width: 49%; height: auto; display: block;">
  <img src="../figures/scale_rollouts_toolsandbox.png" alt="Mean similarity scaling with number of rollouts on ToolSandBox" style="max-width: 49%; height: auto; display: block;">
</div>
<p style="text-align: center;"><i>Performance scaling with number of rollouts on &tau;<sup>2</sup>-Bench (left) and ToolSandBox (right). TRACE scales consistently while baselines plateau or become unstable.</i></p>

### Scaling with Number of Capabilities

TRACE continues to improve as more capability-specific adapters are added, reaching **47.0%** with 4 capabilities. In contrast, GEPA's prompt-based approach plateaus quickly — demonstrating that explicitly training on capability-targeted environments is necessary for continued gains.

<img src="../figures/cap_scaling.png" alt="Overall pass rate scaling with number of capabilities on tau2-Bench" style="width: 60%; height: auto; display: block; margin: 0 auto;">
<p style="text-align: center;"><i>Overall pass rate on &tau;<sup>2</sup>-Bench as the number of capabilities increases. TRACE with trained LoRA adapters scales steadily, while GEPA's prompt-based approach saturates.</i></p>

---

## Citation

If you find TRACE useful, please use the following citation:

```bibtex
@misc{kang2025trace,
  title={TRACE: Capability-Targeted Agentic Training},
  author={Hangoo Kang and Tarun Suresh and Jon Saad-Falcon and Azalia Mirhoseini},
  year={2025},
  howpublished={\url{https://github.com/ScalingIntelligence/TRACE}}
}
```
