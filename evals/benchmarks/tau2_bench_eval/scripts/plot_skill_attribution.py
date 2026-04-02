#!/usr/bin/env python3
"""Plot skill attribution scaling: oracle coverage vs TRACE Router accuracy."""
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

# Data from sweep
K = [1, 2, 4, 8, 16]
oracle    = [41, 66, 71, 94, 110]
router    = [22, 38, 60, 72,  80]
fig, ax = plt.subplots(figsize=(8, 5.5))

# Oracle (ceiling)
ax.plot(K, oracle, 's-', color='#3498db', label='Oracle', zorder=3)

# Router
ax.plot(K, router, 'o-', color='#2ecc71', label='TRACE Router', zorder=4)

ax.set_xlabel('Number of Capabilities')
ax.set_ylabel('Instances Correctly Routed')
ax.set_xticks(K)
ax.set_xticklabels([str(k) for k in K])
ax.set_xlim(1, max(K) + 0.5)
ax.set_ylim(0, 120)
ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
ax.grid(alpha=0.3, zorder=0)

# Annotate oracle
for xi, y in zip(K, oracle):
    ax.annotate(f'{y}', (xi, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=10, color='#2980b9')

# Annotate router
for xi, y_r in zip(K, router):
    ax.annotate(f'{y_r}', (xi, y_r), textcoords="offset points",
                xytext=(0, -16), ha='center', fontsize=10, fontweight='bold', color='#27ae60')

plt.tight_layout()
out = '/home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval/scripts/skill_attribution_plot.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
