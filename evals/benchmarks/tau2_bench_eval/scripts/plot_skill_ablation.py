#!/usr/bin/env python3
"""Plot skill ablation results as line graphs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'font.family': 'sans-serif', 'figure.dpi': 200,
    'lines.linewidth': 2.5, 'lines.markersize': 9,
})

# Data
# Skill ICL (prompt-only, no training) — full range up to 16
icl_skills = [1, 2, 4, 8, 16]
icl_overall = [35.4, 37.8, 38.4, 36.0, 35.1]

# Our method (training)
ours_skills = [1, 2, 4]
ours_overall = [40.3, 43.9, 45.1]

# Baseline
baseline = 32.9

fig, ax = plt.subplots(figsize=(8, 5.5))

# Baseline
ax.axhline(y=baseline, color='#e74c3c', linestyle='--', linewidth=1.5,
           label=f'Base model ({baseline}%)', zorder=2)

# Skill ICL
ax.plot(icl_skills, icl_overall, 's--', color='#95a5a6',
        label='Skill ICL (no training)', zorder=3, markersize=8)

# Our method
ax.plot(ours_skills, ours_overall, 'o-', color='#2ecc71',
        label='Ours', zorder=4)

# Shading
ax.fill_between(ours_skills, baseline, ours_overall, alpha=0.12, color='#2ecc71', zorder=1)

ax.set_xlabel('Number of Skills')
ax.set_ylabel('Overall Pass Rate (%)')
ax.set_title('Performance Scaling with Number of Skills (164 tasks)')
ax.set_xticks(sorted(set(icl_skills + ours_skills)))
ax.set_xticklabels([str(s) for s in sorted(set(icl_skills + ours_skills))])
ax.set_ylim(28, 50)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3, zorder=0)

# Annotate our method
for x, y in zip(ours_skills, ours_overall):
    ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color='#27ae60')

# Annotate ICL
for x, y in zip(icl_skills, icl_overall):
    ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                xytext=(0, -16), ha='center', fontsize=9, color='#7f8c8d')

plt.tight_layout()
plt.savefig('/home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval/scripts/skill_ablation_plot.png',
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved: skill_ablation_plot.png")
