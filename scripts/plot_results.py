#!/usr/bin/env python3
"""
plot_results.py — EdgeDrive Perception benchmark visualization

Usage:
    python3 scripts/plot_results.py           # show interactive
    python3 scripts/plot_results.py --save    # save to benchmarks/plots/
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

# ── Benchmark data ─────────────────────────────────────────────────────────────

LABELS = ["Python\nFP32", "Python\nTRT FP16", "Python\nTRT INT8", "C++\nTRT INT8"]
FPS         = [29.1,  87.6,  78.5, 193.3]
PREPROCESS  = [5.2,   5.6,   5.6,   1.7]
INFERENCE   = [27.6,  4.2,   3.5,   3.5]
POSTPROCESS = [1.1,   1.2,   3.2,   0.0]
POWER       = [9.9,  11.0,   9.7,  12.2]

# ── Theme ──────────────────────────────────────────────────────────────────────

BG          = "#0d1117"
PANEL       = "#161b22"
BORDER      = "#30363d"
TEXT_PRI    = "#e6edf3"
TEXT_SEC    = "#8b949e"
ACCENT      = "#58a6ff"      # blue — Python bars
HIGHLIGHT   = "#f85149"      # red — C++ bar
GRID        = "#21262d"

C_PRE       = "#388bfd"      # blue
C_INF       = "#f0883e"      # orange
C_POST      = "#3fb950"      # green

BAR_COLORS  = [ACCENT, ACCENT, ACCENT, HIGHLIGHT]

def apply_theme():
    rcParams.update({
        "figure.facecolor":     BG,
        "axes.facecolor":       PANEL,
        "axes.edgecolor":       BORDER,
        "axes.labelcolor":      TEXT_SEC,
        "axes.titlecolor":      TEXT_PRI,
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.color":           GRID,
        "grid.linewidth":       0.8,
        "xtick.color":          TEXT_SEC,
        "ytick.color":          TEXT_SEC,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "text.color":           TEXT_PRI,
        "font.family":          "monospace",
        "legend.facecolor":     PANEL,
        "legend.edgecolor":     BORDER,
        "legend.labelcolor":    TEXT_SEC,
        "legend.fontsize":      8.5,
        "figure.dpi":           150,
    })

def spine_style(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

def val_label(ax, bar, text, color=TEXT_PRI, offset=0, fontsize=11):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        text,
        ha="center", va="bottom",
        color=color, fontsize=fontsize, fontweight="bold",
        fontfamily="monospace"
    )

# ── Plot 1: FPS ────────────────────────────────────────────────────────────────

def plot_fps(ax):
    x = np.arange(len(LABELS))
    bars = ax.bar(x, FPS, color=BAR_COLORS, width=0.55,
                  edgecolor=BG, linewidth=1.2, zorder=3)

    # Subtle gradient effect via alpha overlay
    for bar, fps, col in zip(bars, FPS, BAR_COLORS):
        ax.bar(bar.get_x(), fps, bar.get_width(),
               color="white", alpha=0.04, zorder=4)

    for bar, fps, col in zip(bars, FPS, BAR_COLORS):
        val_label(ax, bar, f"{fps:.0f}", color=col, offset=2.5)

    # 30 FPS line
    ax.axhline(30, color=TEXT_SEC, linestyle="--", linewidth=1.0,
               alpha=0.6, zorder=2, label="30 FPS target")

    # C++ annotation
    ax.annotate(
        "6.6× faster\nthan FP32",
        xy=(x[3], FPS[3] * 0.55),
        xytext=(x[3] - 1.3, FPS[3] * 0.68),
        color=HIGHLIGHT, fontsize=8, fontfamily="monospace",
        arrowprops=dict(arrowstyle="->", color=HIGHLIGHT,
                        lw=1.2, connectionstyle="arc3,rad=-0.2"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("Frames per Second (FPS)", color=TEXT_SEC, fontsize=9)
    ax.set_title("Inference Throughput", color=TEXT_PRI,
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, 230)
    ax.legend(loc="upper left", framealpha=0.6)
    spine_style(ax)

# ── Plot 2: Latency ────────────────────────────────────────────────────────────

def plot_latency(ax):
    x = np.arange(len(LABELS))
    w = 0.55
    pre = np.array(PREPROCESS)
    inf = np.array(INFERENCE)
    pst = np.array(POSTPROCESS)

    b1 = ax.bar(x, pre, w, label="Preprocess",   color=C_PRE,  edgecolor=BG, lw=1.2, zorder=3)
    b2 = ax.bar(x, inf, w, bottom=pre,            label="TRT Inference", color=C_INF, edgecolor=BG, lw=1.2, zorder=3)
    b3 = ax.bar(x, pst, w, bottom=pre+inf,        label="Postprocess",  color=C_POST, edgecolor=BG, lw=1.2, zorder=3)

    totals = pre + inf + pst
    for i, (xi, t) in enumerate(zip(x, totals)):
        col = HIGHLIGHT if i == 3 else TEXT_PRI
        ax.text(xi, t + 0.4, f"{t:.1f}ms",
                ha="center", va="bottom",
                color=col, fontsize=10, fontweight="bold",
                fontfamily="monospace")

    # TRT parity bracket
    trt_y_py  = PREPROCESS[2] + INFERENCE[2] / 2
    trt_y_cpp = PREPROCESS[3] + INFERENCE[3] / 2
    mid_y = (PREPROCESS[2] + INFERENCE[2] + PREPROCESS[3] + INFERENCE[3]) / 2 - 1.5
    ax.annotate("", xy=(x[3] + 0.05, trt_y_cpp),
                xytext=(x[2] - 0.05, trt_y_py),
                arrowprops=dict(arrowstyle="<->", color=TEXT_SEC,
                                lw=1.0, connectionstyle="arc3,rad=0"))
    ax.text(2.5, mid_y + 1.5,
            "TRT = 3.5ms\n(identical)",
            ha="center", va="bottom", color=TEXT_SEC,
            fontsize=7.5, style="italic", fontfamily="monospace")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("Latency per Frame (ms)", color=TEXT_SEC, fontsize=9)
    ax.set_title("Latency Breakdown", color=TEXT_PRI,
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, 42)
    ax.legend(loc="upper right", framealpha=0.6)
    spine_style(ax)

# ── Plot 3: Power ──────────────────────────────────────────────────────────────

def plot_power(ax):
    x = np.arange(len(LABELS))
    bars = ax.bar(x, POWER, color=BAR_COLORS, width=0.55,
                  edgecolor=BG, linewidth=1.2, zorder=3)

    for bar, pw, col in zip(bars, POWER, BAR_COLORS):
        val_label(ax, bar, f"{pw:.1f}W", color=col, offset=0.15)

    # FPS/W labels inside bars
    fps_per_w = [f / p for f, p in zip(FPS, POWER)]
    for bar, ratio in zip(bars, fps_per_w):
        if bar.get_height() > 3:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.45,
                    f"{ratio:.1f}\nFPS/W",
                    ha="center", va="center",
                    color="white", alpha=0.55,
                    fontsize=7.5, fontfamily="monospace")

    # Reference lines
    ax.axhline(15, color="#f85149", linestyle="--",
               linewidth=0.9, alpha=0.7, zorder=2, label="15W TDP limit")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("Total Board Power VDD_IN (W)", color=TEXT_SEC, fontsize=9)
    ax.set_title("Power Consumption", color=TEXT_PRI,
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, 17)
    ax.legend(loc="upper left", framealpha=0.6)
    spine_style(ax)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    apply_theme()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.97,
                        top=0.84, bottom=0.14)

    # Suptitle
    fig.text(0.5, 0.97,
             "EdgeDrive Perception — Jetson Orin Nano Super 8GB Benchmarks",
             ha="center", va="top",
             color=TEXT_PRI, fontsize=13, fontweight="bold",
             fontfamily="monospace")
    fig.text(0.5, 0.93,
             "YOLO26n-det TRT INT8  |  404 nuScenes CAM_FRONT images  |  jetson_clocks",
             ha="center", va="top",
             color=TEXT_SEC, fontsize=9, fontfamily="monospace")

    plot_fps(axes[0])
    plot_latency(axes[1])
    plot_power(axes[2])

    # Thin top border line
    fig.add_artist(
        plt.Line2D([0.03, 0.97], [0.895, 0.895],
                   transform=fig.transFigure,
                   color=BORDER, linewidth=0.8))

    if args.save:
        out_dir = os.path.join(
            os.path.dirname(__file__), "..", "benchmarks", "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "benchmark_results.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()