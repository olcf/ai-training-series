"""Unit tests for Session.start_telemetry() integration."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed")

from rhapsody.api.session import Session
from rhapsody.api.task import ComputeTask
from rhapsody.backends.execution.concurrent import ConcurrentExecutionBackend
from rhapsody.telemetry.manager import TelemetryManager


@pytest.fixture
async def backend():
    b = await ConcurrentExecutionBackend(executor=ThreadPoolExecutor(max_workers=4))
    yield b
    await b.shutdown()


class TestStartTelemetry:
    async def test_returns_telemetry_manager(self, backend):
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry()
        assert isinstance(telemetry, TelemetryManager)
        assert session._telemetry is telemetry
        await session.close()

    async def test_observer_wired(self, backend):
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry()
        assert session._state_manager._telemetry_observer == telemetry._on_task_state_change
        await session.close()

    async def test_no_otel_import_before_start_telemetry(self, backend):
        """Session creation must not import opentelemetry before start_telemetry() is called."""
        import sys

        before = set(sys.modules.keys())
        session = Session(backends=[backend])
        after = set(sys.modules.keys())
        new_modules = after - before
        assert not any("opentelemetry.sdk" in m for m in new_modules)

    async def test_already_started_on_return(self, backend):
        """start_telemetry() must return an already-running manager (no separate start needed)."""
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry()
        assert telemetry._running is True
        await session.close()

    async def test_get_telemetry_returns_manager(self, backend):
        session = Session(backends=[backend])
        await session.start_telemetry()
        telemetry = session.get_telemetry()
        assert isinstance(telemetry, TelemetryManager)
        await session.close()

    async def test_get_telemetry_raises_if_not_started(self, backend):
        session = Session(backends=[backend])
        with pytest.raises(RuntimeError, match="start_telemetry"):
            session.get_telemetry()

    async def test_task_submitted_event(self, backend):
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry()

        received = []
        telemetry.subscribe(lambda e: received.append(e))

        tasks = [ComputeTask(executable="echo", arguments=["hi"])]
        await session.submit_tasks(tasks)
        await asyncio.sleep(0.05)

        types = [e.event_type for e in received]
        assert "TaskSubmitted" in types

        await session.close()

    async def test_full_lifecycle_events(self, backend):
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry()

        received = []
        telemetry.subscribe(lambda e: received.append(e))

        tasks = [
            ComputeTask(executable="/bin/echo", arguments=["hello"]),
            ComputeTask(executable="/bin/echo", arguments=["world"]),
        ]
        await session.submit_tasks(tasks)
        await session.wait_tasks(tasks, timeout=10.0)
        await session.close()
        await asyncio.sleep(0.1)

        types = {e.event_type for e in received}
        assert "SessionStarted" in types
        assert "TaskSubmitted" in types
        assert "TaskCompleted" in types or "TaskFailed" in types
        assert "SessionEnded" in types

    async def test_read_metrics_after_tasks(self, backend):
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry()

        tasks = [ComputeTask(executable="/bin/echo", arguments=[str(i)]) for i in range(5)]
        await session.submit_tasks(tasks)
        await session.wait_tasks(tasks, timeout=10.0)
        await session.close()
        await asyncio.sleep(0.1)

        metrics = telemetry.read_metrics()
        submitted = _sum_counter(metrics, "tasks_submitted")
        completed = _sum_counter(metrics, "tasks_completed")
        running = _sum_gauge(metrics, "tasks_running")

        assert submitted == 5
        assert completed == 5
        assert running == 0

    async def test_read_traces_after_tasks(self, backend):
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry()

        tasks = [ComputeTask(executable="/bin/echo", arguments=[str(i)]) for i in range(3)]
        await session.submit_tasks(tasks)
        await session.wait_tasks(tasks, timeout=10.0)
        await session.close()
        await asyncio.sleep(0.1)

        spans = telemetry.read_traces()
        task_spans = [s for s in spans if s.name == "task"]
        assert len(task_spans) == 3
        assert all(s.end_time is not None for s in task_spans)
        assert all(s.attributes.get("status") in ("completed", "failed") for s in task_spans)

    async def test_close_stops_telemetry(self, backend):
        session = Session(backends=[backend])
        telemetry = await session.start_telemetry()
        assert telemetry._running is True
        await session.close()
        assert telemetry._running is False
        assert session._telemetry is None

    async def test_adapter_start_failure_propagates(self, backend):
        """An adapter whose start() raises must propagate through start_telemetry()."""
        from rhapsody.telemetry.adapters.base import TelemetryAdapter

        class BrokenAdapter(TelemetryAdapter):
            def start(self) -> None:
                raise RuntimeError("adapter broken")

            def stop(self) -> None:
                pass

        from rhapsody.telemetry.manager import TelemetryManager

        manager = TelemetryManager(session_id="test-broken")
        manager.register_adapter(BrokenAdapter())
        with pytest.raises(RuntimeError, match="adapter broken"):
            await manager.start()
        # Manager must not be left in a running state on failure
        # (dispatch task may have been created — cancel it to avoid task leak)
        if manager._dispatch_task and not manager._dispatch_task.done():
            manager._dispatch_task.cancel()
            try:
                await manager._dispatch_task
            except asyncio.CancelledError:
                pass


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
