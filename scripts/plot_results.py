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
from matplotlib import rcParams

# ── Benchmark data ─────────────────────────────────────────────────────────────
# Source: benchmarks/results/

# Two groups: Python baselines + C++ deployment variants
LABELS = [
    "Python\nFP32",
    "Python\nTRT FP16",
    "Python\nTRT INT8",
    "C++ INT8\n(bench)",
    "C++ INT8\n(USB cam)",
    "C++ INT8\n(video)",
]

FPS         = [29.1,  87.6,  78.5,  193.3, 202.6, 195.2]
PREPROCESS  = [5.2,   5.6,   5.6,   1.7,   1.1,   1.2  ]
INFERENCE   = [27.6,  4.2,   3.5,   3.5,   3.7,   4.1  ]
POSTPROCESS = [1.1,   1.2,   3.2,   0.0,   0.0,   0.0  ]
POWER       = [9.9,   11.0,  9.7,   12.2,  8.03,  10.2 ]

# ── Theme ──────────────────────────────────────────────────────────────────────

BG        = "#0d1117"
PANEL     = "#161b22"
BORDER    = "#30363d"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
GRID      = "#21262d"

# Colors: Python = blue, C++ bench = red, C++ deployment = green
C_PYTHON  = "#388bfd"
C_BENCH   = "#f85149"
C_DEPLOY  = "#3fb950"

BAR_COLORS = [C_PYTHON, C_PYTHON, C_PYTHON, C_BENCH, C_DEPLOY, C_DEPLOY]

C_PRE     = "#388bfd"
C_INF     = "#f0883e"
C_POST    = "#3fb950"

def apply_theme():
    rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_SEC,
        "axes.titlecolor":   TEXT_PRI,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        GRID,
        "grid.linewidth":    0.8,
        "xtick.color":       TEXT_SEC,
        "ytick.color":       TEXT_SEC,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        TEXT_PRI,
        "font.family":       "monospace",
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": TEXT_SEC,
        "legend.fontsize":   7.5,
        "figure.dpi":        150,
    })

def spine_style(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

def val_label(ax, bar, text, color, offset=2.5, fontsize=9):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            text, ha="center", va="bottom",
            color=color, fontsize=fontsize,
            fontweight="bold", fontfamily="monospace")

# ── Plot 1: FPS ────────────────────────────────────────────────────────────────

def plot_fps(ax):
    x = np.arange(len(LABELS))
    bars = ax.bar(x, FPS, color=BAR_COLORS, width=0.6,
                  edgecolor=BG, linewidth=1.0, zorder=3)

    for bar, fps, col in zip(bars, FPS, BAR_COLORS):
        val_label(ax, bar, f"{fps:.1f}", color=col)

    # 30 FPS line
    ax.axhline(30, color=TEXT_SEC, linestyle="--",
               linewidth=0.9, alpha=0.5, zorder=2, label="30 FPS target")

    # Group separator
    ax.axvline(2.5, color=BORDER, linewidth=0.8, linestyle=":", alpha=0.7)

    # Group labels
    ax.text(1.0,  110, "Python", ha="center", color=C_PYTHON,
            fontsize=8, fontfamily="monospace", alpha=0.8)
    ax.text(4.0,  219, "C++ TensorRT", ha="center", color=C_DEPLOY,
            fontsize=8, fontfamily="monospace", alpha=0.8)

    # USB cam annotation
    ax.annotate("best\ndeployment",
                xy=(4, FPS[4] * 0.62),
                xytext=(3.1, FPS[4] * 0.76),
                color=TEXT_PRI, fontsize=7.5, fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color=TEXT_PRI, lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=7.5)
    ax.set_ylabel("Frames per Second", color=TEXT_SEC, fontsize=9)
    ax.set_title("Inference Throughput", color=TEXT_PRI,
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, 235)
    ax.legend(loc="upper left", framealpha=0.6)
    spine_style(ax)

# ── Plot 2: Latency breakdown ──────────────────────────────────────────────────

def plot_latency(ax):
    x = np.arange(len(LABELS))
    w = 0.6
    pre = np.array(PREPROCESS)
    inf = np.array(INFERENCE)
    pst = np.array(POSTPROCESS)

    ax.bar(x, pre, w, label="Preprocess",    color=C_PRE,  edgecolor=BG, lw=1.0, zorder=3)
    ax.bar(x, inf, w, bottom=pre,            label="TRT Inference", color=C_INF, edgecolor=BG, lw=1.0, zorder=3)
    ax.bar(x, pst, w, bottom=pre+inf,        label="Postprocess",   color=C_POST, edgecolor=BG, lw=1.0, zorder=3)

    totals = pre + inf + pst
    for i, (xi, t) in enumerate(zip(x, totals)):
        col = C_DEPLOY if i == 4 else (C_BENCH if i == 3 else TEXT_PRI)
        ax.text(xi, t + 0.3, f"{t:.1f}ms",
                ha="center", va="bottom",
                color=col, fontsize=8.5, fontweight="bold",
                fontfamily="monospace")

    # Group separator
    ax.axvline(2.5, color=BORDER, linewidth=0.8, linestyle=":", alpha=0.7)

    # TRT parity annotation — show all C++ TRT times are ~3.5-4.1ms
    ax.annotate("TRT: 3.5–4.1ms\nacross all C++",
                xy=(4.5, 5.0),
                xytext=(2.7, 15),
                color=TEXT_SEC, fontsize=7, style="italic",
                fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color=TEXT_SEC,
                                lw=0.9, connectionstyle="arc3,rad=0.2"))

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=7.5)
    ax.set_ylabel("Latency per Frame (ms)", color=TEXT_SEC, fontsize=9)
    ax.set_title("Latency Breakdown", color=TEXT_PRI,
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, 44)
    ax.legend(loc="upper right", framealpha=0.6)
    spine_style(ax)

# ── Plot 3: Power ──────────────────────────────────────────────────────────────

def plot_power(ax):
    x = np.arange(len(LABELS))
    bars = ax.bar(x, POWER, color=BAR_COLORS, width=0.6,
                  edgecolor=BG, linewidth=1.0, zorder=3)

    for bar, pw, col in zip(bars, POWER, BAR_COLORS):
        val_label(ax, bar, f"{pw:.1f}W", color=col, offset=0.15, fontsize=9)

    # FPS/W inside bars
    fps_per_w = [f / p for f, p in zip(FPS, POWER)]
    for bar, ratio in zip(bars, fps_per_w):
        if bar.get_height() > 3:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.45,
                    f"{ratio:.1f}\nFPS/W",
                    ha="center", va="center",
                    color="white", alpha=0.55,
                    fontsize=7, fontfamily="monospace")

    # Reference lines
    ax.axhline(10, color="#d29922", linestyle="--",
               linewidth=0.9, alpha=0.7, zorder=2, label="10W deployment target")
    ax.axhline(15, color=C_BENCH,   linestyle="--",
               linewidth=0.8, alpha=0.5, zorder=2, label="15W TDP limit")

    # Group separator
    ax.axvline(2.5, color=BORDER, linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=7.5)
    ax.set_ylabel("Total Board Power VDD_IN (W)", color=TEXT_SEC, fontsize=9)
    ax.set_title("Power Consumption", color=TEXT_PRI,
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, 17)
    ax.legend(loc="upper left", framealpha=0.6, fontsize=7)
    spine_style(ax)

# ── Legend patches ─────────────────────────────────────────────────────────────

def add_group_legend(fig):
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=C_PYTHON, label="Python baseline"),
        mpatches.Patch(color=C_BENCH,  label="C++ benchmark (jetson_clocks, 404 images)"),
        mpatches.Patch(color=C_DEPLOY, label="C++ live deployment (real-world load)"),
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=3, framealpha=0.6,
               facecolor=PANEL, edgecolor=BORDER,
               labelcolor=TEXT_SEC, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    apply_theme()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.97,
                        top=0.83, bottom=0.18)

    fig.text(0.5, 0.97,
             "EdgeDrive Perception — Jetson Orin Nano Super 8GB Benchmarks",
             ha="center", va="top",
             color=TEXT_PRI, fontsize=13, fontweight="bold",
             fontfamily="monospace")
    fig.text(0.5, 0.93,
             "YOLO26n-det TRT INT8  |  nuScenes + Live Camera  |  jetson_clocks",
             ha="center", va="top",
             color=TEXT_SEC, fontsize=9, fontfamily="monospace")

    fig.add_artist(
        plt.Line2D([0.03, 0.97], [0.895, 0.895],
                   transform=fig.transFigure,
                   color=BORDER, linewidth=0.8))

    plot_fps(axes[0])
    plot_latency(axes[1])
    plot_power(axes[2])
    add_group_legend(fig)

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