#!/usr/bin/env python3
"""Plot capability ablation results as line graphs."""
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
# Capability ICL (prompt-only, no training) — full range up to 16
icl_caps = [1, 2, 4, 8, 16]
icl_overall = [35.4, 37.8, 39.6, 39.6, 38.4]

# Our method (training)
ours_caps = [1, 2, 4]
ours_overall = [40.3, 43.9, 47.0]

# Baseline
baseline = 32.9

fig, ax = plt.subplots(figsize=(8, 5.5))

# Baseline
ax.axhline(y=baseline, color='#e74c3c', linestyle='--', linewidth=1.5,
           label=f'Base', zorder=2)

# Capability ICL
ax.plot(icl_caps, icl_overall, 's--', color='#95a5a6',
        label='GEPA', zorder=3, markersize=8)

# Our method
ax.plot(ours_caps, ours_overall, 'o-', color='#2ecc71',
        label='TRACE', zorder=4)

ax.set_xlabel('Number of Capabilities')
ax.set_ylabel('Overall Pass Rate (%)')
ax.set_xticks(sorted(set(icl_caps + ours_caps)))
ax.set_xticklabels([str(s) for s in sorted(set(icl_caps + ours_caps))])
ax.set_ylim(30, 55)
ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
ax.grid(alpha=0.3, zorder=0)

# Annotate our method
for x, y in zip(ours_caps, ours_overall):
    ax.annotate(f'{y}', (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=10, fontweight='bold', color='#27ae60')

# Annotate ICL
for x, y in zip(icl_caps, icl_overall):
    ax.annotate(f'{y}', (x, y), textcoords="offset points",
                xytext=(0, -16), ha='center', fontsize=9, color='#7f8c8d')

plt.tight_layout()
out = '/home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval/scripts/skill_ablation_plot.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
