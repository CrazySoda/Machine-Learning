"""
plot_comparison.py
------------------
Compares Normal Attention vs Flash Attention training metrics side by side.

Reads:
    Flash_Attention/flash_attention_training_metrics.json
    Transformer/normal_attention_training_metrics.json

Produces:
    1.  Accuracy vs Epoch
    2.  Training Time vs Epoch
    3.  Peak VRAM per Epoch
    4.  Loss vs Epoch
    5.  Avg Execution Time per Layer  (attention layer highlighted)
    6.  Avg VRAM Delta per Layer
    7.  Avg Peak VRAM per Layer

Usage:
    python plot_comparison.py
    python plot_comparison.py --flash Flash_Attention/flash_attention_training_metrics.json
                              --normal Transformer/normal_attention_training_metrics.json
                              --out comparison.png
"""

import json
import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--flash",  default="Flash_Attention/flash_attention_training_metrics.json")
parser.add_argument("--normal", default="Transformer/normal_attention_training_metrics.json")
parser.add_argument("--out",    default="attention_comparison.png")
args = parser.parse_args()


# ──────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path) as f:
        return json.load(f)

flash  = load(args.flash)
normal = load(args.normal)


# ──────────────────────────────────────────────────────────────
# NORMALISE LAYER NAMES
# The two scripts emit slightly different names for attention:
#   Flash  → "Multi-Head Attention"
#   Normal → "MultiHeadAttention"
# We map both to a common label so the bar charts align.
# ──────────────────────────────────────────────────────────────
NAME_MAP = {
    "MultiHeadAttention":   "Attention",
    "Multi-Head Attention": "Attention",
    "FeedForward":          "FeedForward",
    "Input Embedding":      "Input Embedding",
    "Positional Encoding":  "Positional Encoding",
    "Projection":           "Projection",
}

def normalise(name):
    return NAME_MAP.get(name, name)


# ──────────────────────────────────────────────────────────────
# AGGREGATE PER-LAYER STATS  (mean over epochs)
# ──────────────────────────────────────────────────────────────
def aggregate(data):
    agg_time = defaultdict(list)
    agg_mem  = defaultdict(list)
    agg_peak = defaultdict(list)

    for epoch_profile in data.get("layer_profiles_per_epoch", []):
        for raw_name, stats in epoch_profile.items():
            name = normalise(raw_name)
            agg_time[name].append(stats["mean_time_ms"])
            agg_mem[name].append(stats["mean_mem_delta_MB"])
            agg_peak[name].append(stats["mean_peak_MB"])

    layers = sorted(agg_time.keys())
    return (
        layers,
        [np.mean(agg_time[n]) for n in layers],
        [np.mean(agg_mem[n])  for n in layers],
        [np.mean(agg_peak[n]) for n in layers],
    )


flash_layers,  f_time,  f_mem,  f_peak  = aggregate(flash)
normal_layers, n_time,  n_mem,  n_peak  = aggregate(normal)

# Union of layer names (in case one run has layers the other doesn't)
all_layers = sorted(set(flash_layers) | set(normal_layers))

def pad(layers, values, full_list):
    lookup = dict(zip(layers, values))
    return [lookup.get(l, 0.0) for l in full_list]

f_time_p  = pad(flash_layers,  f_time,  all_layers)
f_mem_p   = pad(flash_layers,  f_mem,   all_layers)
f_peak_p  = pad(flash_layers,  f_peak,  all_layers)
n_time_p  = pad(normal_layers, n_time,  all_layers)
n_mem_p   = pad(normal_layers, n_mem,   all_layers)
n_peak_p  = pad(normal_layers, n_peak,  all_layers)


# ──────────────────────────────────────────────────────────────
# EPOCH-LEVEL SERIES
# ──────────────────────────────────────────────────────────────
f_epochs = list(range(1, len(flash["training_time_per_epoch_sec"]) + 1))
n_epochs = list(range(1, len(normal["training_time_per_epoch_sec"]) + 1))

f_acc    = flash.get("accuracy_per_epoch", [])
n_acc    = normal.get("accuracy_per_epoch", [])
f_loss   = flash.get("loss_per_epoch", [])
n_loss   = normal.get("loss_per_epoch", [])
f_times  = flash["training_time_per_epoch_sec"]
n_times  = normal["training_time_per_epoch_sec"]
f_vmem   = flash["peak_memory_per_epoch_MB"]
n_vmem   = normal["peak_memory_per_epoch_MB"]


# ──────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────
C_FLASH  = "#3A86FF"   # blue
C_NORMAL = "#FF6B6B"   # red-ish
BG       = "#F8F9FA"
AX_BG    = "#FFFFFF"

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.8, linestyle="--")

def label_line(ax, xs, ys, fmt="{:.4f}", dy=8):
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y),
                    textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8.5)

flash_patch  = mpatches.Patch(color=C_FLASH,  label="Flash Attention")
normal_patch = mpatches.Patch(color=C_NORMAL, label="Normal Attention")


# ──────────────────────────────────────────────────────────────
# GROUPED HORIZONTAL BAR HELPER
# ──────────────────────────────────────────────────────────────
def grouped_hbar(ax, labels, flash_vals, normal_vals,
                 xlabel, title, unit=""):
    n = len(labels)
    y   = np.arange(n)
    h   = 0.35

    bars_f = ax.barh(y + h/2, flash_vals,  height=h, color=C_FLASH,  alpha=0.88, label="Flash")
    bars_n = ax.barh(y - h/2, normal_vals, height=h, color=C_NORMAL, alpha=0.88, label="Normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    max_val = max(max(flash_vals, default=0), max(normal_vals, default=0))
    ax.set_xlim(0, max_val * 1.25)

    for bar, val in zip(bars_f, flash_vals):
        if val > 0:
            ax.text(bar.get_width() + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}{unit}", va="center", fontsize=8, color=C_FLASH)

    for bar, val in zip(bars_n, normal_vals):
        if val > 0:
            ax.text(bar.get_width() + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}{unit}", va="center", fontsize=8, color=C_NORMAL)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", color="#E0E0E0")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(AX_BG)


# ──────────────────────────────────────────────────────────────
# FIGURE  (4 rows × 2 cols, last cell used for legend)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.patch.set_facecolor(BG)
fig.suptitle("Normal Attention vs Flash Attention — Training Comparison",
             fontsize=17, fontweight="bold", y=0.995)


# ── 1. Accuracy vs Epoch ──────────────────────────────────────
ax = axes[0, 0];  style_ax(ax)
if f_acc:
    ax.plot(f_epochs, f_acc, marker="o", lw=2.2, color=C_FLASH,  label="Flash")
    ax.fill_between(f_epochs, f_acc, alpha=0.10, color=C_FLASH)
    label_line(ax, f_epochs, f_acc)
if n_acc:
    ax.plot(n_epochs, n_acc, marker="s", lw=2.2, color=C_NORMAL, label="Normal", linestyle="--")
    ax.fill_between(n_epochs, n_acc, alpha=0.10, color=C_NORMAL)
    label_line(ax, n_epochs, n_acc, dy=-14)

all_acc = f_acc + n_acc
if all_acc:
    ax.set_ylim(max(0, min(all_acc) - 0.05), min(1.0, max(all_acc) + 0.05))

ax.set_title("Accuracy per Epoch", fontweight="bold")
ax.set_xlabel("Epoch");  ax.set_ylabel("Accuracy")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=9)


# ── 2. Loss vs Epoch ──────────────────────────────────────────
ax = axes[0, 1];  style_ax(ax)
if f_loss:
    ax.plot(f_epochs, f_loss, marker="o", lw=2.2, color=C_FLASH,  label="Flash")
    ax.fill_between(f_epochs, f_loss, alpha=0.10, color=C_FLASH)
    label_line(ax, f_epochs, f_loss, fmt="{:.4f}", dy=8)
if n_loss:
    ax.plot(n_epochs, n_loss, marker="s", lw=2.2, color=C_NORMAL, label="Normal", linestyle="--")
    ax.fill_between(n_epochs, n_loss, alpha=0.10, color=C_NORMAL)
    label_line(ax, n_epochs, n_loss, fmt="{:.4f}", dy=-14)

ax.set_title("Loss per Epoch", fontweight="bold")
ax.set_xlabel("Epoch");  ax.set_ylabel("Loss")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=9)


# ── 3. Training Time per Epoch ────────────────────────────────
ax = axes[1, 0];  style_ax(ax)
w = 0.35
x = np.arange(len(f_epochs))

ax.bar(x - w/2, f_times, width=w, color=C_FLASH,  alpha=0.88, label="Flash")
ax.bar(x + w/2, n_times, width=w, color=C_NORMAL, alpha=0.88, label="Normal")

for xi, (ft, nt) in enumerate(zip(f_times, n_times)):
    ax.text(xi - w/2, ft + max(f_times+n_times)*0.01, f"{ft:.0f}s",
            ha="center", fontsize=8.5, color=C_FLASH)
    ax.text(xi + w/2, nt + max(f_times+n_times)*0.01, f"{nt:.0f}s",
            ha="center", fontsize=8.5, color=C_NORMAL)

ax.set_xticks(x);  ax.set_xticklabels([f"Epoch {e}" for e in f_epochs])
ax.set_title("Training Time per Epoch", fontweight="bold")
ax.set_ylabel("Time (s)")
ax.set_ylim(0, max(f_times + n_times) * 1.20)
ax.legend(fontsize=9)

# Speedup annotation
for xi, (ft, nt) in enumerate(zip(f_times, n_times)):
    speedup = nt / ft
    ax.annotate(f"  {speedup:.2f}× faster",
                xy=(xi, max(ft, nt) + max(f_times+n_times)*0.04),
                ha="center", fontsize=8, color="#555555",
                fontweight="bold")


# ── 4. Peak VRAM per Epoch ────────────────────────────────────
ax = axes[1, 1];  style_ax(ax)
ax.bar(x - w/2, f_vmem, width=w, color=C_FLASH,  alpha=0.88, label="Flash")
ax.bar(x + w/2, n_vmem, width=w, color=C_NORMAL, alpha=0.88, label="Normal")

for xi, (fv, nv) in enumerate(zip(f_vmem, n_vmem)):
    ax.text(xi - w/2, fv + max(f_vmem+n_vmem)*0.01, f"{fv:.0f}",
            ha="center", fontsize=8.5, color=C_FLASH)
    ax.text(xi + w/2, nv + max(f_vmem+n_vmem)*0.01, f"{nv:.0f}",
            ha="center", fontsize=8.5, color=C_NORMAL)

ax.set_xticks(x);  ax.set_xticklabels([f"Epoch {e}" for e in f_epochs])
ax.set_title("Peak VRAM per Epoch", fontweight="bold")
ax.set_ylabel("Peak Memory (MB)")
ax.set_ylim(0, max(f_vmem + n_vmem) * 1.20)
ax.legend(fontsize=9)

for xi, (fv, nv) in enumerate(zip(f_vmem, n_vmem)):
    saving = nv - fv
    ax.annotate(f"  -{saving:.0f} MB",
                xy=(xi, max(fv, nv) + max(f_vmem+n_vmem)*0.04),
                ha="center", fontsize=8, color="#555555",
                fontweight="bold")


# ── 5. Avg Execution Time per Layer ───────────────────────────
grouped_hbar(axes[2, 0], all_layers, f_time_p, n_time_p,
             xlabel="Avg Time (ms)",
             title="Avg Execution Time per Layer",
             unit=" ms")


# ── 6. Avg VRAM Delta per Layer ───────────────────────────────
grouped_hbar(axes[2, 1], all_layers, f_mem_p, n_mem_p,
             xlabel="Avg Memory Delta (MB)",
             title="Avg VRAM Delta per Layer",
             unit=" MB")


# ── 7. Avg Peak VRAM per Layer ────────────────────────────────
grouped_hbar(axes[3, 0], all_layers, f_peak_p, n_peak_p,
             xlabel="Peak Memory (MB)",
             title="Avg Peak VRAM per Layer",
             unit=" MB")


# ── 8. Summary stats box ──────────────────────────────────────
ax_summary = axes[3, 1]
ax_summary.axis("off")

avg_f_time  = np.mean(f_times)
avg_n_time  = np.mean(n_times)
speedup     = avg_n_time / avg_f_time

avg_f_vram  = np.mean(f_vmem)
avg_n_vram  = np.mean(n_vmem)
vram_saving = avg_n_vram - avg_f_vram

avg_f_acc   = np.mean(f_acc) if f_acc else float("nan")
avg_n_acc   = np.mean(n_acc) if n_acc else float("nan")

# Attention layer specifically
attn = "Attention"
f_attn_time = f_time_p[all_layers.index(attn)] if attn in all_layers else float("nan")
n_attn_time = n_time_p[all_layers.index(attn)] if attn in all_layers else float("nan")
attn_speedup = n_attn_time / f_attn_time if f_attn_time else float("nan")

summary_text = (
    "━━━━━━  Summary  ━━━━━━\n\n"
    f"{'Metric':<26}{'Flash':>10}{'Normal':>10}\n"
    f"{'─'*46}\n"
    f"{'Avg epoch time (s)':<26}{avg_f_time:>10.1f}{avg_n_time:>10.1f}\n"
    f"{'Epoch speedup':<26}{'—':>10}{speedup:>9.2f}×\n\n"
    f"{'Avg peak VRAM (MB)':<26}{avg_f_vram:>10.0f}{avg_n_vram:>10.0f}\n"
    f"{'VRAM saved (MB)':<26}{'—':>10}{vram_saving:>9.0f}\n\n"
    f"{'Avg accuracy':<26}{avg_f_acc:>10.4f}{avg_n_acc:>10.4f}\n\n"
    f"{'Attention time (ms)':<26}{f_attn_time:>10.2f}{n_attn_time:>10.2f}\n"
    f"{'Attention speedup':<26}{'—':>10}{attn_speedup:>9.2f}×\n"
)

ax_summary.text(
    0.05, 0.95, summary_text,
    transform=ax_summary.transAxes,
    fontsize=10, verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#EEF2FF",
              edgecolor="#AABBFF", linewidth=1.5)
)

# Global legend
fig.legend(handles=[flash_patch, normal_patch],
           loc="lower center", ncol=2,
           fontsize=11, frameon=True,
           bbox_to_anchor=(0.5, -0.01))


# ──────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Comparison plot saved → {args.out}")
plt.show()