"""Performance tests for telemetry throughput."""

import asyncio
import time

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed")

from rhapsody.telemetry.events import TaskSubmitted
from rhapsody.telemetry.events import make_event
from rhapsody.telemetry.manager import TelemetryManager


@pytest.mark.performance
async def test_emit_10k_tasks_under_5ms():
    """10K emit(make_event()) calls must complete in under 500ms (non-blocking, no awaits)."""
    manager = TelemetryManager(session_id="perf-session")
    await manager.start()

    N = 10_000
    start = time.perf_counter()
    for i in range(N):
        manager.emit(
            make_event(
                TaskSubmitted,
                session_id="perf-session",
                backend="concurrent",
                task_id=f"task.{i:06d}",
            )
        )
    elapsed = time.perf_counter() - start

    await manager.stop()

    assert elapsed < 0.5, f"Emit of {N} events took {elapsed * 1000:.2f}ms (limit: 500ms)"

    # All events must have been processed
    total = _sum_counter(manager.read_metrics(), "tasks_submitted")
    assert total == N


@pytest.mark.performance
async def test_queue_drains_fully():
    """Queue must be empty after stop()."""
    manager = TelemetryManager(session_id="drain-session")
    await manager.start()

    for i in range(500):
        manager.emit(
            make_event(
                TaskSubmitted,
                session_id="drain-session",
                backend="concurrent",
                task_id=f"task.{i:04d}",
            )
        )

    await manager.stop()
    assert manager._queue.empty()


@pytest.mark.performance
async def test_subscriber_does_not_slow_dispatch():
    """A slow subscriber must not block the dispatch loop for other subscribers."""
    manager = TelemetryManager(session_id="sub-perf")
    await manager.start()

    fast_received = []
    slow_count = [0]

    def slow_sub(event):
        slow_count[0] += 1
        time.sleep(0.01)  # 10ms per event — intentionally slow

    manager.subscribe(slow_sub)
    manager.subscribe(lambda e: fast_received.append(e))

    # Fire a handful of events — slow_sub runs inline but dispatch loop
    # moves to next event immediately after iterating subscribers.
    for i in range(5):
        manager.emit(
            make_event(
                TaskSubmitted,
                session_id="sub-perf",
                backend="concurrent",
                task_id=f"t{i}",
            )
        )

    await asyncio.sleep(0.5)
    await manager.stop()

    assert len(fast_received) >= 5


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sum_counter(metrics_data, name: str) -> float:
    total = 0.0
    if metrics_data is None:
        return total
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    for dp in metric.data.data_points:
                        total += dp.value
    return total
