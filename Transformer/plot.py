"""
plot_metrics.py
---------------
Reads flash_attention_training_metrics.json and produces:

  1. Accuracy vs Epoch
  2. Training Time vs Epoch
  3. Average Memory Delta per Layer  (mean across epochs)
  4. Average Time per Layer          (mean across epochs)
  5. Average Peak Memory per Layer   (mean across epochs)

Usage:
    python plot_metrics.py
    python plot_metrics.py --metrics flash_attention_training_metrics.json --out plots.png
"""

import json
import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--metrics",
    default="normal_attention_training_metrics.json",
    help="Path to the training metrics JSON file",
)
parser.add_argument(
    "--out",
    default="normal_attention_plots.png",
    help="Output image path (PNG / PDF / SVG etc.)",
)
args = parser.parse_args()


# ──────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────
if not os.path.exists(args.metrics):
    raise FileNotFoundError(
        f"Metrics file not found: {args.metrics}\n"
        "Run train_test.py first to generate it."
    )

with open(args.metrics) as f:
    data = json.load(f)

epochs = list(range(1, len(data["training_time_per_epoch_sec"]) + 1))
accuracy    = data.get("accuracy_per_epoch", [])
train_times = data["training_time_per_epoch_sec"]


# ──────────────────────────────────────────────
# Aggregate per-layer stats
# ──────────────────────────────────────────────
layer_profiles = data.get("layer_profiles_per_epoch", [])

agg_time = defaultdict(list)
agg_mem  = defaultdict(list)
agg_peak = defaultdict(list)

for epoch_profile in layer_profiles:
    for layer_name, stats in epoch_profile.items():
        agg_time[layer_name].append(stats["mean_time_ms"])
        agg_mem[layer_name].append(stats["mean_mem_delta_MB"])
        agg_peak[layer_name].append(stats["mean_peak_MB"])

layer_names = sorted(agg_time.keys())

avg_time_per_layer = [np.mean(agg_time[n]) for n in layer_names]
avg_mem_per_layer  = [np.mean(agg_mem[n])  for n in layer_names]
avg_peak_per_layer = [np.mean(agg_peak[n]) for n in layer_names]


# ──────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────
COLORS = {
    "blue":   "#3A86FF",
    "green":  "#06D6A0",
    "orange": "#FB8500",
    "red":    "#EF233C",
    "purple": "#8338EC",
}

fig, axes = plt.subplots(3, 2, figsize=(15, 13))
fig.suptitle("Flash Attention – Training Metrics",
             fontsize=16, fontweight="bold", y=0.98)
fig.patch.set_facecolor("#F8F9FA")

for ax in axes.flat:
    ax.set_facecolor("#FFFFFF")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.8, linestyle="--")


# ── 1. Accuracy vs Epoch ──────────────────────
ax1 = axes[0, 0]
if accuracy:
    ax1.plot(epochs, accuracy, marker="o", linewidth=2.2,
             color=COLORS["blue"])
    ax1.fill_between(epochs, accuracy,
                     alpha=0.12, color=COLORS["blue"])
    ax1.set_ylim(max(0, min(accuracy) - 0.05),
                 min(1.0, max(accuracy) + 0.05))

    for x, y in zip(epochs, accuracy):
        ax1.annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

ax1.set_title("Accuracy per Epoch", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# ── 2. Training Time vs Epoch ─────────────────
ax2 = axes[0, 1]
ax2.bar(epochs, train_times,
        color=COLORS["green"], alpha=0.85, width=0.55)
ax2.plot(epochs, train_times,
         marker="o", linewidth=1.8, color=COLORS["green"])

for x, y in zip(epochs, train_times):
    ax2.annotate(f"{y:.1f}s", (x, y),
                 textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=9)

ax2.set_title("Training Time per Epoch", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Time (s)")
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.set_ylim(0, max(train_times) * 1.18)


# ── 3. Avg Memory Delta per Layer ─────────────
ax3 = axes[1, 0]
if layer_names:
    y_pos = np.arange(len(layer_names))
    bars = ax3.barh(y_pos, avg_mem_per_layer,
                    color=COLORS["orange"], alpha=0.85)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(layer_names)

    for bar, val in zip(bars, avg_mem_per_layer):
        ax3.text(bar.get_width() + max(avg_mem_per_layer) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f} MB",
                 va="center", fontsize=8.5)

    ax3.set_xlim(0, max(avg_mem_per_layer) * 1.2)

ax3.set_title("Average VRAM Delta per Layer",
              fontweight="bold")
ax3.set_xlabel("Avg Memory Delta (MB)")
ax3.grid(axis="x", linestyle="--")
ax3.grid(axis="y", visible=False)


# ── 4. Avg Execution Time per Layer ───────────
ax4 = axes[1, 1]
if layer_names:
    y_pos = np.arange(len(layer_names))
    bars = ax4.barh(y_pos, avg_time_per_layer,
                    color=COLORS["red"], alpha=0.85)

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(layer_names)

    for bar, val in zip(bars, avg_time_per_layer):
        ax4.text(bar.get_width() + max(avg_time_per_layer) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f} ms",
                 va="center", fontsize=8.5)

    ax4.set_xlim(0, max(avg_time_per_layer) * 1.2)

ax4.set_title("Average Execution Time per Layer",
              fontweight="bold")
ax4.set_xlabel("Avg Time (ms)")
ax4.grid(axis="x", linestyle="--")
ax4.grid(axis="y", visible=False)


# ── 5. Avg Peak Memory per Layer ──────────────
ax5 = axes[2, 0]
if layer_names:
    y_pos = np.arange(len(layer_names))
    bars = ax5.barh(y_pos, avg_peak_per_layer,
                    color=COLORS["purple"], alpha=0.85)

    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(layer_names)

    for bar, val in zip(bars, avg_peak_per_layer):
        ax5.text(bar.get_width() + max(avg_peak_per_layer) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f} MB",
                 va="center", fontsize=8.5)

    ax5.set_xlim(0, max(avg_peak_per_layer) * 1.15)

ax5.set_title("Average Peak VRAM per Layer",
              fontweight="bold")
ax5.set_xlabel("Peak Memory (MB)")
ax5.grid(axis="x", linestyle="--")
ax5.grid(axis="y", visible=False)


# Remove unused last subplot
fig.delaxes(axes[2, 1])


# ──────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────
fig.tight_layout()
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Plot saved → {args.out}")
plt.show()