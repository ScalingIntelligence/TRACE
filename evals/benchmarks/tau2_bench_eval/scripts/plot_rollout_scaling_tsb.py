#!/usr/bin/env python3
"""Plot overall pass rate vs number of rollouts for GRPO, GEPA, and TRACE."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'font.family': 'sans-serif', 'figure.dpi': 200,
    'lines.linewidth': 2.5, 'lines.markersize': 9,
})

# Data
rollouts = [int(1280/2 * i) for i in range(5)]  # [0, 1280, 2560, 3840, 5120]
grpo  = [0.411, 0.466, 0.501, 0.509, 0.519]
gepa  = [0.411, 0.443, 0.497, 0.517, 0.520]
trace = [0.411, 0.483, 0.514, 0.541, 0.552]

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(rollouts, grpo,  'o-', color='#e74c3c', label='GRPO',  zorder=3)
ax.plot(rollouts, gepa,  's-', color='#3498db', label='GEPA',  zorder=4)
ax.plot(rollouts, trace, 'D-', color='#2ecc71', label='TRACE', zorder=5)

ax.set_xlabel('Number of Rollouts')
ax.set_ylabel('Mean Sim. Score')
ax.set_xticks(rollouts)
ax.set_xticklabels([str(r) for r in rollouts])
ax.set_xlim(-200, max(rollouts) + 200)
ax.set_ylim(0.4, 0.6)
ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
ax.grid(alpha=0.3, zorder=0)

# Annotate GRPO
for xi, y in zip(rollouts, grpo):
    ax.annotate(f'{y}', (xi, y), textcoords="offset points",
                xytext=(0, -16), ha='center', fontsize=10, color='#c0392b')

# Annotate GEPA
for xi, y in zip(rollouts, gepa):
    ax.annotate(f'{y}', (xi, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=10, color='#2980b9')

# Annotate TRACE
for xi, y in zip(rollouts, trace):
    ax.annotate(f'{y}', (xi, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=10, fontweight='bold', color='#27ae60')

plt.tight_layout()
out = '/home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval/scripts/rollout_scaling_plot.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
