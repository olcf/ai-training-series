#!/usr/bin/env python3
"""plot_session_snapshot.py — RHAPSODY telemetry JSONL → multi-panel figure."""

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

EMIT_SCALE = 1e6  # seconds → microseconds

# ── Palette ───────────────────────────────────────────────────────────────────
# One accent colour per subplot for instant visual separation
C = {
    "duration": "#3498db",  # blue
    "cdf": "#2980b9",  # darker blue
    "concurrency": "#e74c3c",  # red
    "cumulative": "#27ae60",  # green
    "wait": "#9b59b6",  # purple
    "routing": "#f39c12",  # amber
    "e2e": "#1abc9c",  # teal
    "emit": "#e67e22",  # orange
    "cpu": "#2ecc71",  # bright green
    "mem": "#3498db",  # blue (reused — different subplot)
    "events": "#8e44ad",  # violet
    "throughput": "#16a085",  # dark teal
    "mean": "#e74c3c",  # red for mean lines
    "median": "#f39c12",  # amber for median lines
    "incomplete": "#bdc3c7",  # light grey for incomplete-lifecycle bar
}
NODE_PALETTE = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#34495e",
]


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


# ── Anomaly tracking ──────────────────────────────────────────────────────────


class AnomalyTracker:
    def __init__(self):
        self.negative_wait = []
        self.negative_routing = []
        self.negative_e2e = []
        self.negative_emit = []
        self.negative_concurrency = False

    def report(self):
        print("\n=== ANOMALY REPORT ===")
        print(f"Negative queue wait:      {len(self.negative_wait)}")
        print(f"Negative routing:         {len(self.negative_routing)}")
        print(f"Negative end-to-end:      {len(self.negative_e2e)}")
        print(f"Negative emit latency:    {len(self.negative_emit)}")
        print(f"Negative concurrency:     {self.negative_concurrency}")
        print("======================\n")


# ── Style helpers ─────────────────────────────────────────────────────────────


def _style(ax, title, xlabel, ylabel, grid=True):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    if grid:
        ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _vline(ax, val, color, label):
    ax.axvline(val, color=color, linestyle="--", linewidth=1.4, label=label, alpha=0.9)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(path: str):
    events, metric_lines, spans = load(path)
    t0 = session_t0(events)
    session_name = Path(path).stem
    anomalies = AnomalyTracker()

    ev = collections.defaultdict(list)
    for e in events:
        ev[e["event_type"]].append(e)

    ev_terminal = sorted(
        ev["TaskCompleted"] + ev["TaskFailed"],
        key=lambda e: e["event_time"],
    )
    tl = build_task_timeline(events, t0)
    nodes = sorted({e["node_id"] for e in ev["ResourceUpdate"] if e.get("node_id")})

    # FIX 1 (duration): separate valid durations from incomplete-lifecycle (0.0)
    all_durations = [
        e.get("duration_seconds")
        for e in (ev["TaskCompleted"] + ev["TaskFailed"])
        if e.get("duration_seconds") is not None
    ]
    durations_valid = [d for d in all_durations if d > 0]
    durations_incomplete = [d for d in all_durations if d == 0.0]

    times_completed = sorted(e["event_time"] - t0 for e in ev_terminal)
    cum_completed = list(range(1, len(times_completed) + 1))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18), facecolor="#f8f9fa")
    fig.suptitle(
        f"RHAPSODY Telemetry  ·  {session_name}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
        color="#2c3e50",
    )

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(4)]

    for ax in axes:
        ax.set_facecolor("#ffffff")

    i = 0

    # ── 1. Task Duration Histogram ────────────────────────────────────────────
    ax = axes[i]
    i += 1
    if durations_valid or durations_incomplete:
        combined = durations_valid + durations_incomplete
        nb = smart_bins(combined)
        # valid durations in blue, incomplete in grey (stacked)
        ax.hist(
            [durations_valid, durations_incomplete],
            bins=nb,
            stacked=True,
            color=[C["duration"], C["incomplete"]],
            edgecolor="white",
            linewidth=0.4,
            label=["valid", f"incomplete ({len(durations_incomplete)})"],
        )
        if durations_valid:
            mu = np.mean(durations_valid)
            _vline(ax, mu, C["mean"], f"mean {mu:.2f}s")
            _vline(
                ax,
                np.median(durations_valid),
                C["median"],
                f"median {np.median(durations_valid):.2f}s",
            )
        ax.legend(fontsize=7)
    _style(ax, "Task Duration", "Duration (seconds)", "Task count")

    # ── 2. Duration CDF ───────────────────────────────────────────────────────
    ax = axes[i]
    i += 1
    if durations_valid:
        s = np.sort(durations_valid)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, cdf, color=C["cdf"], linewidth=2, where="post")
        ax.fill_between(s, cdf, step="post", alpha=0.12, color=C["cdf"])
        for pct, col, lbl in [
            (0.50, "#e74c3c", "p50"),
            (0.90, "#f39c12", "p90"),
            (0.95, "#9b59b6", "p95"),
            (0.99, "#7f8c8d", "p99"),
        ]:
            ax.axhline(pct, color=col, linestyle="--", linewidth=1, alpha=0.7, label=lbl)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=2)
    _style(ax, "Duration CDF", "Duration (seconds)", "Cumulative probability")

    # ── 3. Concurrency Over Time ──────────────────────────────────────────────
    ax = axes[i]
    i += 1
    conc = [(e["event_time"] - t0, +1) for e in ev["TaskStarted"]] + [
        (e["event_time"] - t0, -1) for e in ev_terminal
    ]
    if conc:
        conc.sort(key=lambda x: (x[0], -x[1]))
        t_c = [x[0] for x in conc]
        y_c = np.cumsum([x[1] for x in conc])
        if np.any(y_c < 0):
            anomalies.negative_concurrency = True
        ax.step(t_c, y_c, color=C["concurrency"], linewidth=1.5, where="post")
        ax.fill_between(t_c, y_c, step="post", alpha=0.15, color=C["concurrency"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax, "Concurrency", "Time since start (seconds)", "Active tasks")

    # ── 4. Cumulative Completions ─────────────────────────────────────────────
    ax = axes[i]
    i += 1
    if times_completed:
        ax.step(times_completed, cum_completed, color=C["cumulative"], linewidth=2, where="post")
        ax.fill_between(
            times_completed, cum_completed, step="post", alpha=0.15, color=C["cumulative"]
        )
    _style(ax, "Cumulative Completed", "Time since start (seconds)", "Completed tasks")

    # ── 5. Queue Wait (Queued → Started) ─────────────────────────────────────
    ax = axes[i]
    i += 1
    wait = []
    for ts in tl.values():
        if "Queued" in ts and "Started" in ts:
            val = ts["Started"] - ts["Queued"]
            if val < 0:
                anomalies.negative_wait.append(val)
                continue  # FIX 2: exclude negatives from histogram
            wait.append(val)
    if wait:
        ax.hist(wait, bins=smart_bins(wait), color=C["wait"], edgecolor="white", linewidth=0.4)
        mu = np.mean(wait)
        _vline(ax, mu, C["mean"], f"mean {mu:.3f}s")
        ax.legend(fontsize=7)
    _style(ax, "Queue Wait\n(TaskQueued → TaskStarted)", "Wait time (seconds)", "Task count")

    # ── 6. Routing Latency (Submitted → Queued) ───────────────────────────────
    ax = axes[i]
    i += 1
    routing = []
    for ts in tl.values():
        if "Submitted" in ts and "Queued" in ts:
            val = ts["Queued"] - ts["Submitted"]
            if val < 0:
                anomalies.negative_routing.append(val)
                continue  # FIX 2: exclude negatives
            routing.append(val)
    if routing:
        ax.hist(
            routing, bins=smart_bins(routing), color=C["routing"], edgecolor="white", linewidth=0.4
        )
        mu = np.mean(routing)
        _vline(ax, mu, C["mean"], f"mean {mu:.4f}s")
        ax.legend(fontsize=7)
    _style(
        ax, "Routing Latency\n(TaskSubmitted → TaskQueued)", "Routing time (seconds)", "Task count"
    )

    # ── 7. End-to-End Latency (Submitted → Completed/Failed) ─────────────────
    ax = axes[i]
    i += 1
    e2e = []
    for ts in tl.values():
        if "Submitted" in ts and ("Completed" in ts or "Failed" in ts):
            end = ts.get("Completed", ts.get("Failed"))
            val = end - ts["Submitted"]
            if val < 0:
                anomalies.negative_e2e.append(val)
                continue  # FIX 2: exclude negatives
            e2e.append(val)
    if e2e:
        ax.hist(e2e, bins=smart_bins(e2e), color=C["e2e"], edgecolor="white", linewidth=0.4)
        mu = np.mean(e2e)
        _vline(ax, mu, C["mean"], f"mean {mu:.3f}s")
        ax.legend(fontsize=7)
    _style(
        ax, "End-to-End Latency\n(TaskSubmitted → TaskCompleted)", "Latency (seconds)", "Task count"
    )

    # ── 8. Emit Latency ───────────────────────────────────────────────────────
    ax = axes[i]
    i += 1
    emit = []
    for e in events:
        if e.get("emit_time") is not None:
            val = (e["emit_time"] - e["event_time"]) * EMIT_SCALE
            if val < 0:
                anomalies.negative_emit.append(val)
                continue  # FIX 2: exclude negatives
            emit.append(val)
    if emit:
        ax.hist(emit, bins=50, color=C["emit"], edgecolor="white", linewidth=0.4)
        mu = np.mean(emit)
        _vline(ax, mu, C["mean"], f"mean {mu:.1f}μs")
        ax.legend(fontsize=7)
    _style(ax, "Emit Latency\n(emit_time − event_time)", "Latency (μs)", "Event count")

    # ── 9. CPU Utilization ────────────────────────────────────────────────────
    ax = axes[i]
    i += 1
    if ev["ResourceUpdate"]:
        for k, node in enumerate(nodes):
            # FIX 1 (resource): only per-node events (gpu_id is None)
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
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                    label=node,
                )
        ax.set_ylim(0, 105)
        if nodes:
            ax.legend(fontsize=6)
    _style(ax, "CPU Utilization", "Time since start (seconds)", "CPU %")

    # ── 10. Memory Utilization ────────────────────────────────────────────────
    ax = axes[i]
    i += 1
    if ev["ResourceUpdate"]:
        for k, node in enumerate(nodes):
            # FIX 1 (resource): only per-node events (gpu_id is None)
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
                    linewidth=1.5,
                    marker="s",
                    markersize=3,
                    label=node,
                )
        ax.set_ylim(0, 105)
        if nodes:
            ax.legend(fontsize=6)
    _style(ax, "Memory Utilization", "Time since start (seconds)", "Memory %")

    # ── 11. GPU Utilization ───────────────────────────────────────────────────
    ax = axes[i]
    i += 1
    has_gpu = any(e.get("gpu_percent") is not None for e in ev["ResourceUpdate"])
    if has_gpu:
        color_idx = 0
        for node in nodes:
            node_events = [e for e in ev["ResourceUpdate"] if e.get("node_id") == node]
            gpu_ids = sorted({e["gpu_id"] for e in node_events if e.get("gpu_id") is not None})
            for gid in gpu_ids:
                gev = sorted(
                    [e for e in node_events if e.get("gpu_id") == gid],
                    key=lambda e: e["event_time"],
                )
                ax.plot(
                    [e["event_time"] - t0 for e in gev],
                    [e.get("gpu_percent") or 0.0 for e in gev],
                    color=NODE_PALETTE[color_idx % len(NODE_PALETTE)],
                    linewidth=1.5,
                    marker="^",
                    markersize=3,
                    label=f"{node} GPU {gid}",
                )
                color_idx += 1
        ax.set_ylim(0, 105)
        ax.legend(fontsize=6, ncol=2)
        _style(ax, "GPU Utilization (per device)", "Time since start (seconds)", "GPU %")
    else:
        ax.text(
            0.5,
            0.5,
            "No GPU data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="#95a5a6",
        )
        ax.set_axis_off()
        ax.set_title("GPU Utilization", fontsize=10, fontweight="bold", pad=6)

    # ── 12. Event Type Distribution ───────────────────────────────────────────
    ax = axes[i]
    i += 1
    items = sorted(
        collections.Counter(e["event_type"] for e in events).items(),
        key=lambda x: x[1],
    )
    if items:
        labels_e, vals_e = zip(*items)
        bar_colors = [NODE_PALETTE[j % len(NODE_PALETTE)] for j in range(len(vals_e))]
        bars = ax.barh(labels_e, vals_e, color=bar_colors, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals_e):
            ax.text(
                v + max(vals_e) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(v),
                va="center",
                fontsize=7,
            )
    _style(ax, "Event Type Distribution", "Count", "", grid=False)

    # ── 13. Throughput ────────────────────────────────────────────────────────
    ax = axes[i]
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
                width=bin_w * 0.9,
                color=C["throughput"],
                edgecolor="white",
                linewidth=0.4,
            )
    _style(ax, "Task Completion Throughput", "Time since start (seconds)", "Tasks per second")

    # ── 14–16. OTel metric counters bar chart ─────────────────────────────────
    ax = axes[i]
    i += 1
    if metric_lines:
        m = metric_lines[-1]
        keys = [
            "tasks_submitted",
            "tasks_started",
            "tasks_completed",
            "tasks_failed",
            "tasks_running",
        ]
        labels = [k.replace("tasks_", "").capitalize() for k in keys]
        vals = [m.get(k, 0) for k in keys]
        bcolors = ["#3498db", "#9b59b6", "#2ecc71", "#e74c3c", "#f39c12"]
        bars = ax.bar(labels, vals, color=bcolors, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals, default=1) * 0.015,
                str(int(v)),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax, "OTel Metric Snapshot", "", "Count")

    # ── 15. Concurrency + Node Resources (combined overview) ─────────────────
    ax = axes[i]
    i += 1
    ax2 = ax.twinx()
    # Concurrency step curve (left axis)
    deltas = []
    for e in events:
        et = e.get("event_type", "")
        t = e.get("event_time", 0) - t0
        if et == "TaskStarted":
            deltas.append((t, +1))
        elif et in ("TaskCompleted", "TaskFailed", "TaskCanceled"):
            deltas.append((t, -1))
    if deltas:
        deltas.sort()
        t_c, y_c, running = [], [], 0
        for tc, d in deltas:
            t_c += [tc, tc]
            y_c += [running, running + d]
            running += d
        if t_c:
            t_c.append(t_c[-1])
            y_c.append(0)
        ax.step(
            t_c,
            y_c,
            color=C["concurrency"],
            linewidth=1.5,
            where="post",
            label="Concurrency",
            alpha=0.9,
        )
        ax.fill_between(t_c, y_c, step="post", alpha=0.10, color=C["concurrency"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    # Per-node CPU / Mem / GPU lines (right axis)
    for k, node in enumerate(nodes):
        nc = NODE_PALETTE[k % len(NODE_PALETTE)]
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
            ax2.plot(
                ts,
                [e.get("cpu_percent") or 0 for e in per_node],
                color=nc,
                linewidth=1.2,
                linestyle="-",
                label=f"{node} CPU%",
            )
            ax2.plot(
                ts,
                [e.get("memory_percent") or 0 for e in per_node],
                color=nc,
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                label=f"{node} Mem%",
            )
        gpu_ids = sorted(
            {
                e["gpu_id"]
                for e in ev["ResourceUpdate"]
                if e.get("node_id") == node and e.get("gpu_id") is not None
            }
        )
        for gid in gpu_ids:
            gev = sorted(
                [
                    e
                    for e in ev["ResourceUpdate"]
                    if e.get("node_id") == node and e.get("gpu_id") == gid
                ],
                key=lambda e: e["event_time"],
            )
            gc = NODE_PALETTE[(k + 2) % len(NODE_PALETTE)]
            ax2.plot(
                [e["event_time"] - t0 for e in gev],
                [e.get("gpu_percent") or 0 for e in gev],
                color=gc,
                linewidth=1.2,
                linestyle=":",
                label=f"{node} GPU{gid}%",
            )
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Utilization (%)", fontsize=7)
    ax2.tick_params(axis="y", labelsize=6)
    ax2.spines[["top"]].set_visible(False)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=5, ncol=3, loc="upper right")
    _style(ax, "Concurrency + Node Resources", "Time since start (s)", "Active tasks")

    # Hide unused panels
    while i < len(axes):
        axes[i].set_visible(False)
        i += 1

    out_path = Path(path).with_suffix(".png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.close(fig)

    anomalies.report()


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        candidates = glob.glob("**/*.telemetry.jsonl", recursive=True)
        if not candidates:
            print("No telemetry file found", file=sys.stderr)
            sys.exit(1)
        target = max(candidates, key=os.path.getmtime)

    main(target)
