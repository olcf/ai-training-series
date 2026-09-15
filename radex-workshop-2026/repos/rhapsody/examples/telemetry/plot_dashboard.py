#!/usr/bin/env python3
"""plot-final.py — RHAPSODY telemetry JSONL → multi-panel figure.

Layout
------
Row 0 (4 panels): Task Duration · Concurrency · Cumulative Completions · Throughput
Row 1 (4 panels): CPU Utilization · Memory Utilization · GPU Utilization · Event Distribution
Row 2 (full row): Aggregated resource utilization (mean CPU/Mem/GPU across all nodes) + concurrency
Row 3 (full row): Per-node / per-GPU utilization + concurrency
"""

import collections
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Global font / rcParams ─────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "lines.linewidth": 2.0,
        "patch.linewidth": 0.8,
        "figure.dpi": 130,
    }
)

# ── Colours ────────────────────────────────────────────────────────────────────
BG = "#f4f6f9"  # figure background
PANEL_BG = "#ffffff"  # axes face
BORDER = "#c8d0da"  # spine / border colour
GRID_CLR = "#e2e6ea"  # grid lines

C = {
    "duration": "#2e86de",
    "concurrency": "#ee5a24",
    "cumulative": "#009432",
    "throughput": "#006266",
    "mean": "#ee5a24",
    "median": "#f9ca24",
    "incomplete": "#c8d6e5",
}
NODE_PALETTE = [
    "#ee5a24",
    "#2e86de",
    "#009432",
    "#f9ca24",
    "#8854d0",
    "#00b894",
    "#e17055",
    "#2d3436",
]
GPU_PALETTE = [
    "#fd79a8",
    "#a29bfe",
    "#55efc4",
    "#fdcb6e",
    "#e17055",
    "#6c5ce7",
    "#00cec9",
    "#ffeaa7",
]

# Wide-row specific line colours (distinct from concurrency)
AGG_CPU_CLR = "#00b894"
AGG_MEM_CLR = "#0984e3"
AGG_GPU_CLR = "#d63031"


# ── I/O ───────────────────────────────────────────────────────────────────────


def load(path: str):
    events, metrics, spans = [], [], []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sec = rec.get("section")
            if sec == "event":
                events.append(rec)
            elif sec == "metric":
                metrics.append(rec)
            elif sec == "span":
                spans.append(rec)
    if not events:
        raise ValueError("No events found in file")
    return events, metrics, spans


def session_t0(events):
    ss = [e for e in events if e.get("event_type") == "SessionStarted"]
    return ss[0]["event_time"] if ss else min(e["event_time"] for e in events)


def build_task_timeline(events, t0):
    tl = collections.defaultdict(dict)
    for e in events:
        tid = e.get("task_id")
        if tid is None:
            continue
        key = e["event_type"].replace("Task", "")
        tl[tid][key] = e["event_time"] - t0
    return tl


def smart_bins(data, max_bins=50):
    return min(max_bins, max(10, len(data) // 5))


# ── Style helpers ─────────────────────────────────────────────────────────────


def _decorate(ax, title, xlabel, ylabel, grid=True, rotate_x=True):
    """Apply uniform title / label / spine / border styling to any axes."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, pad=8, color="#2d3436")
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    if rotate_x:
        ax.tick_params(axis="x", rotation=15)

    # All four spines visible → panel border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor(BORDER)

    if grid:
        ax.grid(color=GRID_CLR, linewidth=0.7, linestyle="-", zorder=0)
        ax.set_axisbelow(True)


def _decorate_wide(ax, title, xlabel, ylabel):
    _decorate(ax, title, xlabel, ylabel, rotate_x=False)


def _style_twin_right(ax2, ylabel):
    """Style the right-axis twin — only right/bottom spines shown."""
    ax2.set_ylabel(ylabel, labelpad=10, color="#555")
    ax2.tick_params(axis="y", labelcolor="#555")
    for side, vis in [("top", False), ("left", False), ("right", True), ("bottom", True)]:
        ax2.spines[side].set_visible(vis)
        if vis:
            ax2.spines[side].set_linewidth(1.2)
            ax2.spines[side].set_edgecolor(BORDER)
    ax2.set_ylim(0, 110)


def _vline(ax, val, color, label):
    ax.axvline(val, color=color, linestyle="--", linewidth=1.6, label=label, alpha=0.9)


def _legend_small(ax, **kw):
    leg = ax.legend(frameon=True, **kw)
    leg.get_frame().set_linewidth(0.8)
    return leg


def _legend_below(ax, lines, labs, ncol=None):
    ncol = ncol or min(len(labs), 8)
    leg = ax.legend(
        lines,
        labs,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=True,
        handlelength=2.0,
        columnspacing=1.2,
    )
    leg.get_frame().set_linewidth(0.8)
    return leg


# ── Concurrency helpers ───────────────────────────────────────────────────────


def _build_concurrency(events, t0):
    deltas = []
    for e in events:
        et = e.get("event_type", "")
        t = e.get("event_time", 0) - t0
        if et == "TaskStarted":
            deltas.append((t, +1))
        elif et in ("TaskCompleted", "TaskFailed", "TaskCanceled"):
            deltas.append((t, -1))
    if not deltas:
        return [], []
    deltas.sort(key=lambda x: (x[0], -x[1]))
    t_c, y_c, running = [], [], 0
    for tc, d in deltas:
        t_c += [tc, tc]
        y_c += [running, running + d]
        running += d
    if t_c:
        t_c.append(t_c[-1])
        y_c.append(0)
    return t_c, y_c


def _draw_conc(ax, t_c, y_c):
    """Concurrency step on left axis (no fill — keeps wide rows readable)."""
    if not t_c:
        return
    ax.step(
        t_c,
        y_c,
        color=C["concurrency"],
        linewidth=2.5,
        where="post",
        label="Tasks running",
        zorder=5,
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0, top=max(y_c) * 1.18 if y_c else 1)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(path: str):
    events, metric_lines, spans = load(path)
    t0 = session_t0(events)
    session_name = Path(path).stem

    ev = collections.defaultdict(list)
    for e in events:
        ev[e["event_type"]].append(e)

    ev_terminal = sorted(
        ev["TaskCompleted"] + ev["TaskFailed"],
        key=lambda e: e["event_time"],
    )
    tl = build_task_timeline(events, t0)
    nodes = sorted({e["node_id"] for e in ev["ResourceUpdate"] if e.get("node_id")})

    all_durations = [
        e.get("duration_seconds")
        for e in (ev["TaskCompleted"] + ev["TaskFailed"])
        if e.get("duration_seconds") is not None
    ]
    durations_valid = [d for d in all_durations if d > 0]
    durations_incomplete = [d for d in all_durations if d == 0.0]

    times_completed = sorted(e["event_time"] - t0 for e in ev_terminal)
    cum_completed = list(range(1, len(times_completed) + 1))

    t_c, y_c = _build_concurrency(events, t0)

    # ── Figure & grid ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 32), facecolor=BG)
    fig.suptitle(
        f"RHAPSODY Telemetry  ·  {session_name}",
        fontsize=17,
        fontweight="bold",
        y=0.999,
        color="#2d3436",
    )

    gs = gridspec.GridSpec(
        4,
        4,
        figure=fig,
        hspace=0.82,
        wspace=0.44,
        top=0.973,
        bottom=0.063,
        left=0.07,
        right=0.95,
        height_ratios=[1, 1, 1.35, 1.35],
    )

    small = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    i = 0

    # ── 1. Task Duration Histogram ────────────────────────────────────────────
    ax = small[i]
    i += 1
    if durations_valid or durations_incomplete:
        nb = smart_bins(durations_valid + durations_incomplete)
        ax.hist(
            [durations_valid, durations_incomplete],
            bins=nb,
            stacked=True,
            color=[C["duration"], C["incomplete"]],
            edgecolor="#f4f6f9",
            linewidth=0.6,
            label=["valid", f"incomplete ({len(durations_incomplete)})"],
            zorder=3,
        )
        if durations_valid:
            mu = np.mean(durations_valid)
            _vline(ax, mu, C["mean"], f"mean {mu:.2f}s")
            _vline(
                ax,
                np.median(durations_valid),
                C["median"],
                f"med {np.median(durations_valid):.2f}s",
            )
        _legend_small(ax, fontsize=8, loc="upper right")
    _decorate(ax, "Task Duration", "Duration (s)", "Count")

    # ── 2. Concurrency ────────────────────────────────────────────────────────
    ax = small[i]
    i += 1
    if t_c:
        ax.step(t_c, y_c, color=C["concurrency"], linewidth=2.0, where="post", zorder=3)
        ax.fill_between(t_c, y_c, step="post", alpha=0.18, color=C["concurrency"], zorder=2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _decorate(ax, "Concurrency", "Time (s)", "Active tasks")

    # ── 3. Cumulative Completions ─────────────────────────────────────────────
    ax = small[i]
    i += 1
    if times_completed:
        ax.step(
            times_completed,
            cum_completed,
            color=C["cumulative"],
            linewidth=2.0,
            where="post",
            zorder=3,
        )
        ax.fill_between(
            times_completed, cum_completed, step="post", alpha=0.18, color=C["cumulative"], zorder=2
        )
    _decorate(ax, "Cumulative Completed", "Time (s)", "Tasks")

    # ── 4. Throughput ─────────────────────────────────────────────────────────
    ax = small[i]
    i += 1
    if len(times_completed) > 1:
        tmin, tmax = min(times_completed), max(times_completed)
        dur = tmax - tmin
        if dur > 0:
            bin_w = dur / 30
            bins_arr = np.arange(tmin, tmax + bin_w, bin_w)
            counts, _ = np.histogram(times_completed, bins=bins_arr)
            centers = (bins_arr[:-1] + bins_arr[1:]) / 2
            ax.bar(
                centers,
                counts / bin_w,
                width=bin_w * 0.85,
                color=C["throughput"],
                edgecolor="#f4f6f9",
                linewidth=0.6,
                zorder=3,
            )
    _decorate(ax, "Completion Throughput", "Time (s)", "Tasks / s")

    # ── 5. CPU Utilization ────────────────────────────────────────────────────
    ax = small[i]
    i += 1
    if ev["ResourceUpdate"]:
        for k, node in enumerate(nodes):
            ne = sorted(
                [
                    e
                    for e in ev["ResourceUpdate"]
                    if e["node_id"] == node and e.get("gpu_id") is None
                ],
                key=lambda e: e["event_time"],
            )
            if ne:
                ax.plot(
                    [e["event_time"] - t0 for e in ne],
                    [e.get("cpu_percent") or 0.0 for e in ne],
                    color=NODE_PALETTE[k % len(NODE_PALETTE)],
                    linewidth=2.0,
                    marker="o",
                    markersize=3,
                    label=node.split(".")[0],
                    zorder=3,
                )
        ax.set_ylim(0, 110)
        if nodes:
            _legend_small(ax, loc="lower right", fontsize=8)
    _decorate(ax, "CPU Utilization", "Time (s)", "CPU %")

    # ── 6. Memory Utilization ─────────────────────────────────────────────────
    ax = small[i]
    i += 1
    if ev["ResourceUpdate"]:
        for k, node in enumerate(nodes):
            ne = sorted(
                [
                    e
                    for e in ev["ResourceUpdate"]
                    if e["node_id"] == node and e.get("gpu_id") is None
                ],
                key=lambda e: e["event_time"],
            )
            if ne:
                ax.plot(
                    [e["event_time"] - t0 for e in ne],
                    [e.get("memory_percent") or 0.0 for e in ne],
                    color=NODE_PALETTE[k % len(NODE_PALETTE)],
                    linewidth=2.0,
                    marker="s",
                    markersize=3,
                    label=node.split(".")[0],
                    zorder=3,
                )
        ax.set_ylim(0, 110)
        if nodes:
            _legend_small(ax, loc="lower right", fontsize=8)
    _decorate(ax, "Memory Utilization", "Time (s)", "Mem %")

    # ── 7. GPU Utilization ────────────────────────────────────────────────────
    ax = small[i]
    i += 1
    has_gpu = any(e.get("gpu_percent") is not None for e in ev["ResourceUpdate"])
    if has_gpu:
        cidx = 0
        for node in nodes:
            node_evs = [e for e in ev["ResourceUpdate"] if e.get("node_id") == node]
            gpu_ids = sorted({e["gpu_id"] for e in node_evs if e.get("gpu_id") is not None})
            for gid in gpu_ids:
                gev = sorted(
                    [e for e in node_evs if e.get("gpu_id") == gid],
                    key=lambda e: e["event_time"],
                )
                ax.plot(
                    [e["event_time"] - t0 for e in gev],
                    [e.get("gpu_percent") or 0.0 for e in gev],
                    color=NODE_PALETTE[cidx % len(NODE_PALETTE)],
                    linewidth=2.0,
                    marker="^",
                    markersize=3,
                    label=f"{node.split('.')[0]} G{gid}",
                    zorder=3,
                )
                cidx += 1
        ax.set_ylim(0, 110)
        _legend_small(ax, loc="lower right", fontsize=8, ncol=2)
    else:
        ax.text(
            0.5,
            0.5,
            "No GPU data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="#b2bec3",
        )
        ax.set_axis_off()
    _decorate(ax, "GPU Utilization (per device)", "Time (s)", "GPU %")

    # ── 8. Event Type Distribution ────────────────────────────────────────────
    ax = small[i]
    i += 1
    items = sorted(
        collections.Counter(e["event_type"] for e in events).items(),
        key=lambda x: x[1],
    )
    if items:
        labels_e, vals_e = zip(*items)
        bar_colors = [NODE_PALETTE[j % len(NODE_PALETTE)] for j in range(len(vals_e))]
        bars = ax.barh(
            labels_e,
            vals_e,
            color=bar_colors,
            edgecolor="#f4f6f9",
            linewidth=0.6,
            height=0.65,
            zorder=3,
        )
        for bar, v in zip(bars, vals_e):
            ax.text(
                v + max(vals_e) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                str(v),
                va="center",
                fontsize=9,
                fontweight="bold",
            )
    ax.tick_params(axis="y", labelsize=8.5)
    _decorate(ax, "Event Type Distribution", "Count", "", grid=False, rotate_x=False)

    # ── Row 2: Aggregated resource + concurrency ──────────────────────────────
    ax_agg = fig.add_subplot(gs[2, :])
    ax_agg2 = ax_agg.twinx()

    _draw_conc(ax_agg, t_c, y_c)

    if nodes and ev["ResourceUpdate"]:
        from collections import defaultdict as _dd

        buckets_cpu: dict = _dd(list)
        buckets_mem: dict = _dd(list)
        buckets_gpu: dict = _dd(list)
        for e in ev["ResourceUpdate"]:
            if e.get("gpu_id") is not None:
                continue
            tb = round((e["event_time"] - t0) * 2) / 2
            if e.get("cpu_percent") is not None:
                buckets_cpu[tb].append(e["cpu_percent"])
            if e.get("memory_percent") is not None:
                buckets_mem[tb].append(e["memory_percent"])
            if e.get("gpu_percent") is not None:
                buckets_gpu[tb].append(e["gpu_percent"])

        def _mean_series(b):
            ts = sorted(b)
            return ts, [np.mean(b[t]) for t in ts]

        if buckets_cpu:
            ts, vs = _mean_series(buckets_cpu)
            ax_agg2.plot(
                ts,
                vs,
                color=AGG_CPU_CLR,
                linewidth=3.0,
                linestyle="-",
                label="Mean CPU %",
                zorder=4,
            )
        if buckets_mem:
            ts, vs = _mean_series(buckets_mem)
            ax_agg2.plot(
                ts,
                vs,
                color=AGG_MEM_CLR,
                linewidth=3.0,
                linestyle="--",
                label="Mean Mem %",
                zorder=4,
            )
        if buckets_gpu:
            ts, vs = _mean_series(buckets_gpu)
            ax_agg2.plot(
                ts,
                vs,
                color=AGG_GPU_CLR,
                linewidth=3.0,
                linestyle=":",
                label="Mean GPU %",
                zorder=4,
            )

    _decorate_wide(
        ax_agg,
        "Aggregated Resource Utilization (all nodes)  +  Concurrency",
        "Time since start (s)",
        "Active tasks",
    )
    _style_twin_right(ax_agg2, "Utilization %  (mean, all nodes)")

    lines1, labs1 = ax_agg.get_legend_handles_labels()
    lines2, labs2 = ax_agg2.get_legend_handles_labels()
    _legend_below(ax_agg, lines1 + lines2, labs1 + labs2)

    # ── Row 3: Per-node / per-GPU + concurrency ───────────────────────────────
    ax_det = fig.add_subplot(gs[3, :])
    ax_det2 = ax_det.twinx()

    _draw_conc(ax_det, t_c, y_c)

    for k, node in enumerate(nodes):
        nc = NODE_PALETTE[k % len(NODE_PALETTE)]
        short = node.split(".")[0]
        per_node = sorted(
            [
                e
                for e in ev["ResourceUpdate"]
                if e.get("node_id") == node and e.get("gpu_id") is None
            ],
            key=lambda e: e["event_time"],
        )
        if per_node:
            ts = [e["event_time"] - t0 for e in per_node]
            ax_det2.plot(
                ts,
                [e.get("cpu_percent") or 0 for e in per_node],
                color=nc,
                linewidth=2.5,
                linestyle="-",
                label=f"{short} CPU%",
                zorder=4,
            )
            ax_det2.plot(
                ts,
                [e.get("memory_percent") or 0 for e in per_node],
                color=nc,
                linewidth=2.5,
                linestyle="--",
                alpha=0.80,
                label=f"{short} Mem%",
                zorder=4,
            )

    gpu_idx = 0
    for node in nodes:
        short = node.split(".")[0]
        node_evs = [e for e in ev["ResourceUpdate"] if e.get("node_id") == node]
        gpu_ids = sorted({e["gpu_id"] for e in node_evs if e.get("gpu_id") is not None})
        for gid in gpu_ids:
            gev = sorted(
                [e for e in node_evs if e.get("gpu_id") == gid],
                key=lambda e: e["event_time"],
            )
            ax_det2.plot(
                [e["event_time"] - t0 for e in gev],
                [e.get("gpu_percent") or 0 for e in gev],
                color=GPU_PALETTE[gpu_idx % len(GPU_PALETTE)],
                linewidth=2.5,
                linestyle=":",
                label=f"{short} GPU{gid}%",
                zorder=4,
            )
            gpu_idx += 1

    _decorate_wide(
        ax_det,
        "Per-Node / Per-GPU Utilization  +  Concurrency",
        "Time since start (s)",
        "Active tasks",
    )
    _style_twin_right(ax_det2, "Utilization %  (per node / per GPU)")

    lines1, labs1 = ax_det.get_legend_handles_labels()
    lines2, labs2 = ax_det2.get_legend_handles_labels()
    _legend_below(ax_det, lines1 + lines2, labs1 + labs2)

    out_path = Path(path).with_suffix(".png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        candidates = glob.glob("**/*.jsonl", recursive=True)
        if not candidates:
            print("No JSONL file found", file=sys.stderr)
            sys.exit(1)
        target = max(candidates, key=os.path.getmtime)

    main(target)
