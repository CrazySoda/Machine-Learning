"""
plot_linformer_metrics.py
=========================
Generates 5 matplotlib plots from the JSON files saved by train_test.py:

  1. Accuracy vs Epoch          (real per-epoch values)
  2. Training Time vs Epoch
  3. Memory Usage vs Epoch
  4. Average Memory per Layer   (bar chart, last epoch)
  5. Average Time per Layer     (bar chart, last epoch)

Expected input files (same directory):
  - linformer_training_metrics.json
  - linformer_eval_metrics.json

Output:
  - linformer_plots.png   (all 5 in one figure)
  - 5 individual PNGs
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAINING_JSON = "linformer_training_metrics.json"
EVAL_JSON     = "linformer_eval_metrics.json"
OUT_DIR       = "."

# ── Style ─────────────────────────────────────────────────────────────────────
ACCENT  = "#4FC3F7"   # sky blue
ACCENT2 = "#FF8A65"   # warm orange
ACCENT3 = "#A5D6A7"   # sage green
BG      = "#0F1923"
PANEL   = "#182535"
GRID_C  = "#1E3248"
TEXT    = "#E8F0F7"
SUBTEXT = "#8BA5BF"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID_C,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "grid.color":       GRID_C,
    "grid.linewidth":   0.7,
    "text.color":       TEXT,
    "font.family":      "monospace",
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID_C,
})


# ── Load data ─────────────────────────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run train_test.py first to generate the JSON files."
        )
    with open(path) as f:
        return json.load(f)


train_data = load_json(TRAINING_JSON)
eval_data  = load_json(EVAL_JSON)

epochs             = list(range(1, len(train_data["loss_per_epoch"]) + 1))
accuracy_per_epoch = train_data["accuracy_per_epoch"]   # real measured values
losses             = train_data["loss_per_epoch"]
times_sec          = train_data["training_time_per_epoch_sec"]
peak_mem_MB        = train_data["peak_memory_per_epoch_MB"]
layer_profiles     = train_data["layer_profiles_per_epoch"]

# Per-layer stats from the last epoch
last_profile = layer_profiles[-1]
layer_names  = list(last_profile.keys())
mean_times   = [last_profile[n]["mean_time_ms"]      for n in layer_names]
mean_mems    = [last_profile[n]["mean_mem_delta_MB"]  for n in layer_names]


# ── Helper ────────────────────────────────────────────────────────────────────
def label_bars(ax, bars, fmt="{:.2f}", pad_frac=0.03):
    vals = [b.get_height() for b in bars]
    pad  = max(abs(v) for v in vals) * pad_frac or 0.01
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + pad,
            fmt.format(v),
            ha="center", va="bottom",
            fontsize=7, color=SUBTEXT,
        )


def annotate_points(ax, xs, ys, color, fmt="{:.4f}"):
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.5, color=color)


# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), dpi=130)
fig.suptitle(
    "Linformer — Training & GPU Profile Report",
    fontsize=15, fontweight="bold", color=TEXT, y=0.98,
)

gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])   # Accuracy
ax2 = fig.add_subplot(gs[0, 1])   # Training Time
ax3 = fig.add_subplot(gs[0, 2])   # Memory
ax4 = fig.add_subplot(gs[1, :2])  # Memory per Layer
ax5 = fig.add_subplot(gs[1, 2])   # Time per Layer


# ── 1. Accuracy vs Epoch ──────────────────────────────────────────────────────
ax1.plot(epochs, accuracy_per_epoch, color=ACCENT, marker="o",
         linewidth=2, markersize=7, markerfacecolor=BG, markeredgewidth=2)
ax1.fill_between(epochs, accuracy_per_epoch, alpha=0.12, color=ACCENT)
annotate_points(ax1, epochs, accuracy_per_epoch, ACCENT)
ax1.set_title("Accuracy vs Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_xticks(epochs)
ax1.set_ylim(0, 1.08)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax1.grid(True, axis="y")


# ── 2. Training Time vs Epoch ─────────────────────────────────────────────────
ax2.bar(epochs, times_sec, color=ACCENT2, alpha=0.85, width=0.5, zorder=3)
ax2.plot(epochs, times_sec, color=ACCENT2, marker="D",
         linewidth=1.5, markersize=5, zorder=4)
for e, t in zip(epochs, times_sec):
    ax2.text(e, t + max(times_sec) * 0.02, f"{t:.1f}s",
             ha="center", fontsize=7.5, color=ACCENT2)
ax2.set_title("Training Time vs Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Time (seconds)")
ax2.set_xticks(epochs)
ax2.set_ylim(0, max(times_sec) * 1.2)
ax2.grid(True, axis="y")


# ── 3. Memory Usage vs Epoch ──────────────────────────────────────────────────
ax3.plot(epochs, peak_mem_MB, color=ACCENT3, marker="s",
         linewidth=2, markersize=7, markerfacecolor=BG, markeredgewidth=2)
ax3.fill_between(epochs, peak_mem_MB, alpha=0.12, color=ACCENT3)
annotate_points(ax3, epochs, peak_mem_MB, ACCENT3, fmt="{:.0f}")
ax3.set_title("Peak VRAM Usage vs Epoch")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Peak Memory (MB)")
ax3.set_xticks(epochs)
ax3.set_ylim(0, max(peak_mem_MB) * 1.2)
ax3.grid(True, axis="y")


# ── 4. Average Memory per Layer ───────────────────────────────────────────────
x4    = np.arange(len(layer_names))
cols4 = [ACCENT] * len(layer_names)
max_mem_idx = int(np.argmax(mean_mems))
cols4[max_mem_idx] = ACCENT2

bars4 = ax4.bar(x4, mean_mems, color=cols4, alpha=0.85, width=0.55, zorder=3)
ax4.set_title("Avg VRAM Δ per Layer  (last epoch)")
ax4.set_xlabel("Layer")
ax4.set_ylabel("Mean Memory Δ (MB)")
ax4.set_xticks(x4)
ax4.set_xticklabels(layer_names, rotation=30, ha="right")
ax4.grid(True, axis="y")
label_bars(ax4, bars4)
ax4.annotate("↑ peak", (max_mem_idx, mean_mems[max_mem_idx]),
             textcoords="offset points", xytext=(0, 22),
             ha="center", fontsize=7.5, color=ACCENT2)


# ── 5. Average Time per Layer ─────────────────────────────────────────────────
x5    = np.arange(len(layer_names))
cols5 = [ACCENT3] * len(layer_names)
max_t_idx = int(np.argmax(mean_times))
cols5[max_t_idx] = ACCENT2

bars5 = ax5.bar(x5, mean_times, color=cols5, alpha=0.85, width=0.55, zorder=3)
ax5.set_title("Avg Time per Layer  (last epoch)")
ax5.set_xlabel("Layer")
ax5.set_ylabel("Mean Time (ms)")
ax5.set_xticks(x5)
ax5.set_xticklabels(layer_names, rotation=30, ha="right")
ax5.grid(True, axis="y")
label_bars(ax5, bars5)
ax5.annotate("↑ slowest", (max_t_idx, mean_times[max_t_idx]),
             textcoords="offset points", xytext=(0, 22),
             ha="center", fontsize=7.5, color=ACCENT2)


# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    f"Linformer (scale=4, 512→128 keys) · IMDB · "
    f"Final Acc={eval_data['accuracy']:.4f}  "
    f"F1={eval_data['f1']:.4f}  "
    f"ROC-AUC={eval_data['roc_auc']:.4f}",
    ha="center", fontsize=8, color=SUBTEXT,
)


# ── Save ──────────────────────────────────────────────────────────────────────
combined = os.path.join(OUT_DIR, "linformer_plots.png")
fig.savefig(combined, bbox_inches="tight", facecolor=BG)
print(f"Saved combined figure → {combined}")

individual = [
    (ax1, "linformer_accuracy_vs_epoch.png"),
    (ax2, "linformer_time_vs_epoch.png"),
    (ax3, "linformer_memory_vs_epoch.png"),
    (ax4, "linformer_memory_per_layer.png"),
    (ax5, "linformer_time_per_layer.png"),
]
for ax, fname in individual:
    extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted()
    )
    fig.savefig(
        os.path.join(OUT_DIR, fname),
        bbox_inches=extent.expanded(1.12, 1.18),
        facecolor=BG,
    )
    print(f"Saved → {fname}")

plt.show()