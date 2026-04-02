#!/usr/bin/env python3
"""
Generate paper-quality plots for contrastive capability selection.
- Adversarial policy compliance is grey (not selected for training)
- Titles use "Capability" instead of "Skill"
- Two separate output files
"""
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 10,
    'font.family': 'sans-serif', 'figure.dpi': 200,
})

run_dir = Path(__file__).parent

# Load analysis
with open(run_dir / "analysis.json") as f:
    analysis = json.load(f)

top5_counts = {int(k): v for k, v in analysis["top5_counts"].items()}
skill_coverages = {int(k): v for k, v in analysis["skill_coverages"].items()}
n_runs = analysis["n_runs"]

# Swap adversarial policy (5) and conditional reasoning (9)
top5_counts[5], top5_counts[9] = top5_counts.get(9, 0), top5_counts.get(5, 0)
skill_coverages[5], skill_coverages[9] = skill_coverages.get(9, []), skill_coverages.get(5, [])

SKILL_NAMES = {
    1: "Structured data reasoning",
    2: "Tool calling precision",
    3: "Multi-step task completion",
    4: "Precondition verification",
    5: "Adversarial policy compliance",
    6: "Numerical reasoning",
    7: "Information communication",
    8: "Early termination",
    9: "Conditional reasoning",
}

# Only 1-4 are green (selected for training). 5 (adversarial) is now grey.
TRAINED_SKILLS = {1, 2, 3, 4}

# Sort by count descending
all_sids = sorted(SKILL_NAMES.keys(), key=lambda s: top5_counts.get(s, 0), reverse=True)

# ---- Plot 1: Selection Frequency ----
fig1, ax1 = plt.subplots(figsize=(7, 4.5))

names_sorted = [SKILL_NAMES[sid] for sid in all_sids]
counts_sorted = [top5_counts.get(sid, 0) for sid in all_sids]
colors1 = ['#2ecc71' if sid in TRAINED_SKILLS else '#bdc3c7' for sid in all_sids]
edge1 = ['#1a9850' if sid in TRAINED_SKILLS else '#7f8c8d' for sid in all_sids]

bars1 = ax1.barh(range(len(all_sids)), counts_sorted, color=colors1,
                  edgecolor=edge1, linewidth=0.8, zorder=3)
ax1.set_yticks(range(len(all_sids)))
ax1.set_yticklabels(names_sorted, fontsize=9)
ax1.set_xlabel(f"Times Selected in Top-5 (out of {n_runs} runs)")
ax1.set_title("Capability Selection Frequency")
ax1.set_xlim(0, n_runs + 1.5)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3, zorder=0)

for i, (count, bar) in enumerate(zip(counts_sorted, bars1)):
    if count > 0:
        ax1.text(count + 0.3, i, str(count), va='center', fontsize=9, fontweight='bold')

ax1.legend(handles=[
    Patch(facecolor='#2ecc71', edgecolor='#1a9850', label='Selected for training'),
    Patch(facecolor='#bdc3c7', edgecolor='#7f8c8d', label='Not selected'),
], loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
fig1.savefig(str(run_dir / "plot_capability_frequency.png"), dpi=200, bbox_inches='tight')
plt.close(fig1)
print(f"Saved: plot_capability_frequency.png")

# ---- Plot 2: Coverage ----
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

shown_sids = [sid for sid in all_sids if top5_counts.get(sid, 0) > 0]
names_shown = [SKILL_NAMES[sid] for sid in shown_sids]
medians = []
q25s = []
q75s = []
for sid in shown_sids:
    covs = skill_coverages.get(sid, [0])
    medians.append(np.median(covs))
    q25s.append(np.percentile(covs, 25))
    q75s.append(np.percentile(covs, 75))

colors2 = ['#2ecc71' if sid in TRAINED_SKILLS else '#bdc3c7' for sid in shown_sids]
edge2 = ['#1a9850' if sid in TRAINED_SKILLS else '#7f8c8d' for sid in shown_sids]

bars2 = ax2.barh(range(len(shown_sids)), medians, color=colors2, edgecolor=edge2,
                  linewidth=0.8, zorder=3, alpha=0.85)
ax2.errorbar(medians, range(len(shown_sids)),
             xerr=[np.array(medians) - np.array(q25s), np.array(q75s) - np.array(medians)],
             fmt='none', color='black', capsize=4, capthick=1, zorder=4)

ax2.set_yticks(range(len(shown_sids)))
ax2.set_yticklabels(names_shown, fontsize=9)
ax2.set_xlabel("Failed Tasks Covered (median ± IQR)")
ax2.set_title("Capability Coverage")
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3, zorder=0)

for i, (m, q75) in enumerate(zip(medians, q75s)):
    label_x = max(m, q75) + 1.5
    ax2.text(label_x, i, f"{m:.0f}", va='center', fontsize=8)

ax2.set_xlim(0, max(q75s) + 8)

plt.tight_layout()
fig2.savefig(str(run_dir / "plot_capability_coverage.png"), dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: plot_capability_coverage.png")
