"""End-to-end integration test: ConcurrentBackend + Session + telemetry."""

import asyncio
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed")

from rhapsody.api.session import Session
from rhapsody.api.task import ComputeTask
from rhapsody.backends.execution.concurrent import ConcurrentExecutionBackend
from rhapsody.telemetry.events import ResourceUpdate
from rhapsody.telemetry.events import SessionEnded
from rhapsody.telemetry.events import SessionStarted
from rhapsody.telemetry.events import TaskCompleted
from rhapsody.telemetry.events import TaskQueued
from rhapsody.telemetry.events import TaskSubmitted

N = 20


async def test_telemetry_e2e_20_tasks():
    backend = await ConcurrentExecutionBackend(executor=ThreadPoolExecutor(max_workers=8))
    session = Session(backends=[backend])
    telemetry = await session.start_telemetry(resource_poll_interval=0.1)

    received = []
    telemetry.subscribe(lambda e: received.append(e))

    tasks = [ComputeTask(executable="/bin/echo", arguments=[str(i)]) for i in range(N)]

    await session.submit_tasks(tasks)
    await session.wait_tasks(tasks, timeout=30.0)
    await session.close()
    await asyncio.sleep(0.15)

    # --- Event assertions ---
    event_types = {e.event_type for e in received}
    assert "SessionStarted" in event_types
    assert "TaskSubmitted" in event_types
    assert "TaskCompleted" in event_types or "TaskFailed" in event_types
    assert "SessionEnded" in event_types

    submitted_events = [e for e in received if isinstance(e, TaskSubmitted)]
    assert len(submitted_events) == N
    # backend must always be populated after routing fix
    assert all(e.backend != "" for e in submitted_events)
    # task_type must be set (not empty string)
    assert all(e.attributes.get("task_type") == "ComputeTask" for e in submitted_events)

    queued_events = [e for e in received if isinstance(e, TaskQueued)]
    assert len(queued_events) == N
    assert all(
        e.attributes.get("executable") != "" or e.attributes.get("task_type") != ""
        for e in queued_events
    )

    # TaskStarted must be emitted for every task (RUNNING state now added to ConcurrentBackend)
    started_events = [e for e in received if e.event_type == "TaskStarted"]
    assert len(started_events) == N

    completed_events = [e for e in received if isinstance(e, TaskCompleted)]
    failed_events = [e for e in received if e.event_type == "TaskFailed"]
    assert len(completed_events) + len(failed_events) == N

    for ev in completed_events:
        # duration must be > 0 now that RUNNING is always emitted
        assert ev.duration_seconds > 0.0
        assert not ev.attributes.get("incomplete_lifecycle", False)

    # --- Metric assertions ---
    metrics = telemetry.read_metrics()
    assert _sum_counter(metrics, "tasks_submitted") == N
    assert _sum_counter(metrics, "tasks_completed") == N
    assert _sum_counter(metrics, "tasks_failed") == 0
    assert _sum_gauge(metrics, "tasks_running") == 0

    # --- Span assertions ---
    spans = telemetry.read_traces()
    task_spans = [s for s in spans if s.name == "task"]
    session_spans = [s for s in spans if s.name == "session"]

    assert len(task_spans) == N
    assert len(session_spans) == 1

    assert all(s.end_time is not None for s in task_spans)
    assert all(s.attributes.get("status") in ("completed", "failed") for s in task_spans)
    assert session_spans[0].end_time is not None


async def test_subscriber_receives_all_events():
    backend = await ConcurrentExecutionBackend(executor=ThreadPoolExecutor(max_workers=4))
    session = Session(backends=[backend])
    telemetry = await session.start_telemetry()

    received = []
    telemetry.subscribe(lambda e: received.append(e))

    tasks = [ComputeTask(executable="/bin/echo", arguments=["x"]) for _ in range(5)]
    await session.submit_tasks(tasks)
    await session.wait_tasks(tasks, timeout=10.0)
    await session.close()
    await asyncio.sleep(0.1)

    # At minimum: SessionStarted + 5×TaskSubmitted + 5×TaskStarted/Completed + SessionEnded
    assert len(received) >= 12


async def test_start_telemetry_after_backend_added():
    """start_telemetry() called after add_backend() must still attach adapters."""
    backend = await ConcurrentExecutionBackend(executor=ThreadPoolExecutor(max_workers=2))
    session = Session()
    session.add_backend(backend)
    telemetry = await session.start_telemetry()

    # ConcurrentTelemetryAdapter should have been registered
    assert len(telemetry._adapters) >= 1

    tasks = [ComputeTask(executable="/bin/echo", arguments=["ok"])]
    await session.submit_tasks(tasks)
    await session.wait_tasks(tasks, timeout=5.0)
    await session.close()


async def test_checkpoint_file_e2e():
    """Session with checkpoint_path produces a valid JSONL file with all three sections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = await ConcurrentExecutionBackend(executor=ThreadPoolExecutor(max_workers=4))
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry(
            resource_poll_interval=0.1,
            checkpoint_path=tmpdir,
        )

        tasks = [ComputeTask(executable="/bin/echo", arguments=[str(i)]) for i in range(5)]
        await session.submit_tasks(tasks)
        await session.wait_tasks(tasks, timeout=10.0)
        await asyncio.sleep(0.15)
        await session.close()

        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".telemetry.jsonl")

        filepath = os.path.join(tmpdir, files[0])
        lines = [json.loads(l) for l in open(filepath)]
        sections = {l["section"] for l in lines}
        assert "event" in sections
        assert "metric" in sections
        assert "span" in sections

        event_types = {l["event_type"] for l in lines if l["section"] == "event"}
        assert "SessionStarted" in event_types
        assert "TaskSubmitted" in event_types
        assert "TaskQueued" in event_types
        assert "SessionEnded" in event_types

        task_events = [
            l
            for l in lines
            if l["section"] == "event" and l["event_type"] in ("TaskCompleted", "TaskFailed")
        ]
        assert len(task_events) == 5
        # executable is now in the attributes dict, not a top-level field
        assert all(l.get("attributes", {}).get("executable") is not None for l in task_events)

        spans = [l for l in lines if l["section"] == "span" and l["name"] == "task"]
        assert len(spans) == 5


async def test_summary_and_task_spans_e2e():
    """Summary() and task_spans() return correct plain-dict data after session."""
    backend = await ConcurrentExecutionBackend(executor=ThreadPoolExecutor(max_workers=4))
    session = Session(backends=[backend])
    telemetry = await session.start_telemetry()

    tasks = [ComputeTask(executable="/bin/echo", arguments=[str(i)]) for i in range(4)]
    tasks.append(ComputeTask(executable="/bin/false"))  # intentional failure
    await session.submit_tasks(tasks)
    await session.wait_tasks(tasks, timeout=10.0)
    await session.close()

    summary = telemetry.summary()
    assert summary["tasks"]["submitted"] == 5
    assert summary["tasks"]["completed"] == 4
    assert summary["tasks"]["failed"] == 1
    assert summary["tasks"]["running"] == 0
    assert summary["duration"]["mean_seconds"] is not None

    spans = telemetry.task_spans()
    assert len(spans) == 5
    assert all("task_id" in s for s in spans)
    assert all("status" in s for s in spans)
    assert any(s["status"] == "failed" for s in spans)


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


def _sum_gauge(metrics_data, name: str) -> float:
    return _sum_counter(metrics_data, name)
