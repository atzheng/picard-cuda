#!/usr/bin/env python3
"""
Plot wall time and speedup vs. number of Picard workers.

Usage:
    python3 scripts/plot_benchmark.py results/benchmark_summary.csv results/benchmark_plot.png
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Reference palette (see the dataviz skill's references/palette.md)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"


def load_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["workers"]))
    return rows


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <benchmark_summary.csv> <output.png>", file=sys.stderr)
        sys.exit(1)
    in_csv, out_png = sys.argv[1], sys.argv[2]

    rows = load_rows(in_csv)
    workers = [int(r["workers"]) for r in rows]
    wall = [float(r["wall_time_s"]) for r in rows]
    speedup = [float(r["speedup"]) for r in rows]
    seq_time = float(rows[0]["sequential_time_s"])
    n_events = int(rows[0]["n_events"])
    mismatches = sum(int(r["mismatches"]) for r in rows)

    fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    fig.suptitle(
        f"Picard runtime vs. worker count  ({n_events:,} events)",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.02, ha="left",
    )

    # --- left: wall time (log-log), sequential baseline as a reference line ---
    style_axes(ax_time)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.plot(workers, wall, color=BLUE, linewidth=2, marker="o", markersize=6,
                 zorder=3, label="Picard (parallel)")
    ax_time.axhline(seq_time, color=INK_MUTED, linewidth=1.5, linestyle="--",
                     zorder=2, label="Sequential (single-thread)")
    for x, y in zip(workers, wall):
        ax_time.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                          xytext=(0, 8), ha="center", fontsize=8, color=INK_SECONDARY)
    ax_time.set_xlabel("Workers")
    ax_time.set_ylabel("Wall time (s, log scale)")
    ax_time.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_time.xaxis.set_minor_formatter(mticker.NullFormatter())
    legend = ax_time.legend(frameon=True, fontsize=9, loc="center right")
    legend.get_frame().set_facecolor(SURFACE)
    legend.get_frame().set_edgecolor("none")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # --- right: speedup vs. sequential (log-x, linear y) ---
    style_axes(ax_speedup)
    ax_speedup.set_xscale("log")
    ax_speedup.plot(workers, speedup, color=BLUE, linewidth=2, marker="o", markersize=6,
                     zorder=3)
    for x, y in zip(workers, speedup):
        ax_speedup.annotate(f"{y:.0f}×", (x, y), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8, color=INK_SECONDARY)
    ax_speedup.set_xlabel("Workers")
    ax_speedup.set_ylabel("Speedup vs. sequential (×)")
    ax_speedup.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_speedup.xaxis.set_minor_formatter(mticker.NullFormatter())

    footer = f"seed 42 · {mismatches} mismatch(es) vs. sequential across all worker counts"
    fig.text(0.02, -0.02, footer, color=INK_MUTED, fontsize=8, ha="left")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
