#!/usr/bin/env python3
"""
plot_results.py — Visualize EdgeDrive Perception benchmark results

Generates two plots from hardcoded benchmark data:
  1. FPS comparison (bar chart)
  2. Latency breakdown (stacked bar chart)

Usage:
    python3 scripts/plot_results.py
    python3 scripts/plot_results.py --save  # save to benchmarks/plots/

Requirements:
    pip install matplotlib numpy
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Benchmark data ────────────────────────────────────────────────────────────
# Source: benchmarks/results/jetson_python.md + jetson_cpp.md
# Conditions: Jetson Orin Nano Super 8GB, jetson_clocks, 404 images, 60s run

FORMATS = [
    "Python\nFP32",
    "Python\nTRT FP16",
    "Python\nTRT INT8",
    "C++\nTRT INT8",
]

FPS = [29.1, 87.6, 78.5, 193.3]

# Latency breakdown per frame (ms)
PREPROCESS  = [5.2,  5.6,  5.6,  1.7]
INFERENCE   = [27.6, 4.2,  3.5,  3.5]
POSTPROCESS = [1.1,  1.2,  3.2,  0.0]  # C++ postprocess excluded from timer

POWER_W = [9.9, 11.0, 9.7, 12.2]  # VDD_IN steady state (W)

COLORS = {
    'preprocess':  '#4C72B0',
    'inference':   '#DD8452',
    'postprocess': '#55A868',
    'fps_bar':     ['#4C72B0', '#4C72B0', '#4C72B0', '#C44E52'],
}

# Highlight C++ bar
BAR_COLORS = ['#7FB3D3', '#7FB3D3', '#7FB3D3', '#C44E52']


def plot_fps(ax):
    """Bar chart: FPS comparison."""
    x = np.arange(len(FORMATS))
    bars = ax.bar(x, FPS, color=BAR_COLORS, width=0.6,
                  edgecolor='white', linewidth=0.8)

    # Value labels on bars
    for bar, fps in zip(bars, FPS):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f'{fps:.0f}',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # 30 FPS target line
    ax.axhline(30, color='gray', linestyle='--', linewidth=1.0,
               label='30 FPS target')

    ax.set_xticks(x)
    ax.set_xticklabels(FORMATS, fontsize=10)
    ax.set_ylabel('Frames per Second (FPS)', fontsize=11)
    ax.set_title('Inference Throughput\n(Jetson Orin Nano Super, jetson_clocks)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 220)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate C++ bar
    ax.annotate('6.6× faster\nthan FP32',
                xy=(3, FPS[3]), xytext=(2.4, 170),
                fontsize=9, color='#C44E52',
                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5))


def plot_latency(ax):
    """Stacked bar chart: latency breakdown."""
    x = np.arange(len(FORMATS))
    width = 0.55

    p1 = ax.bar(x, PREPROCESS, width,
                label='Preprocess', color='#4C72B0', edgecolor='white')
    p2 = ax.bar(x, INFERENCE, width, bottom=PREPROCESS,
                label='TRT Inference', color='#DD8452', edgecolor='white')
    p3 = ax.bar(x, POSTPROCESS, width,
                bottom=np.array(PREPROCESS) + np.array(INFERENCE),
                label='Postprocess', color='#55A868', edgecolor='white')

    # Total labels
    totals = np.array(PREPROCESS) + np.array(INFERENCE) + np.array(POSTPROCESS)
    for i, (xi, total) in enumerate(zip(x, totals)):
        ax.text(xi, total + 0.4,
                f'{total:.1f}ms',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color='#C44E52' if i == 3 else 'black')

    # TRT parity annotation
    ax.annotate('',
                xy=(3, 3.5 + 1.7), xytext=(2, 3.5 + 5.6),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(2.5, 7.5, 'TRT identical\n(3.5ms)', ha='center',
            fontsize=8, style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(FORMATS, fontsize=10)
    ax.set_ylabel('Latency per Frame (ms)', fontsize=11)
    ax.set_title('Latency Breakdown\n(preprocess + inference + postprocess)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 42)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def plot_power(ax):
    """Bar chart: power consumption."""
    x = np.arange(len(FORMATS))
    bars = ax.bar(x, POWER_W, color=BAR_COLORS, width=0.6,
                  edgecolor='white', linewidth=0.8)

    for bar, power in zip(bars, POWER_W):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f'{power:.1f}W',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # 10W budget line
    ax.axhline(10, color='orange', linestyle='--', linewidth=1.2,
               label='10W budget')
    ax.axhline(15, color='red', linestyle='--', linewidth=1.0,
               label='15W TDP limit', alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(FORMATS, fontsize=10)
    ax.set_ylabel('Total Board Power VDD_IN (W)', fontsize=11)
    ax.set_title('Power Consumption\n(tegrastats VDD_IN, steady state)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 16)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help='Save plots to benchmarks/plots/')
    args = parser.parse_args()

    if args.save:
        matplotlib.use('Agg')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'EdgeDrive Perception — Jetson Orin Nano Super 8GB Benchmarks\n'
        'YOLO26n-det TRT INT8 | 404 nuScenes CAM_FRONT images | jetson_clocks',
        fontsize=13, fontweight='bold', y=1.02
    )

    plot_fps(axes[0])
    plot_latency(axes[1])
    plot_power(axes[2])

    plt.tight_layout()

    if args.save:
        out_dir = os.path.join(
            os.path.dirname(__file__), '..', 'benchmarks', 'plots')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'benchmark_results.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()