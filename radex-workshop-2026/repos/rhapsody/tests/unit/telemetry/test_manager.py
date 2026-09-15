"""Unit tests for TelemetryManager (OTel wiring, EventBus, subscriber dispatch)."""

import asyncio
import json
import os
import tempfile
import time

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed")

from rhapsody.telemetry.events import TaskCanceled
from rhapsody.telemetry.events import TaskCompleted
from rhapsody.telemetry.events import TaskCreated
from rhapsody.telemetry.events import TaskFailed
from rhapsody.telemetry.events import TaskQueued
from rhapsody.telemetry.events import TaskStarted
from rhapsody.telemetry.events import TaskSubmitted
from rhapsody.telemetry.events import make_event
from rhapsody.telemetry.manager import TelemetryManager


@pytest.fixture
async def manager():
    m = TelemetryManager(session_id="test-session")
    await m.start()
    yield m
    if m._running:
        await m.stop()


class TestEmit:
    async def test_emit_is_nonblocking(self, manager):
        """put_nowait must not block — 10K calls should complete essentially instantly."""
        start = time.perf_counter()
        for i in range(10_000):
            manager.emit(
                make_event(
                    TaskSubmitted,
                    session_id="test-session",
                    backend="concurrent",
                    task_id=f"task.{i:06d}",
                )
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5  # generous upper bound — real overhead is <5ms

    async def test_queue_drains_on_stop(self, manager):
        N = 50
        for i in range(N):
            manager.emit(
                make_event(
                    TaskSubmitted,
                    session_id="test-session",
                    backend="concurrent",
                    task_id=f"task.{i:06d}",
                )
            )
        await manager.stop()
        assert manager._queue.empty()


class TestOtelInstruments:
    async def test_task_submitted_counter(self, manager):
        manager.emit(
            make_event(TaskSubmitted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        await asyncio.sleep(0.1)
        metrics = manager.read_metrics()
        total = _sum_counter(metrics, "tasks_submitted")
        assert total >= 1

    async def test_task_lifecycle_counters(self, manager):
        now = time.time()
        manager.emit(
            make_event(TaskStarted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(
                TaskCompleted,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.5,
            )
        )
        await asyncio.sleep(0.1)
        metrics = manager.read_metrics()
        assert _sum_counter(metrics, "tasks_started") >= 1
        assert _sum_counter(metrics, "tasks_completed") >= 1

    async def test_tasks_running_gauge(self, manager):
        manager.emit(
            make_event(TaskStarted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(TaskStarted, session_id="test-session", backend="concurrent", task_id="t2")
        )
        manager.emit(
            make_event(
                TaskCompleted,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.1,
            )
        )
        await asyncio.sleep(0.1)
        metrics = manager.read_metrics()
        running = _sum_gauge(metrics, "tasks_running")
        assert running == 1  # t2 still running

    async def test_failed_task(self, manager):
        manager.emit(
            make_event(TaskStarted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(
                TaskFailed,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.2,
                error_type="RuntimeError",
            )
        )
        await asyncio.sleep(0.1)
        metrics = manager.read_metrics()
        assert _sum_counter(metrics, "tasks_failed") >= 1
        assert _sum_gauge(metrics, "tasks_running") == 0


class TestSpans:
    async def test_task_span_created_and_closed(self, manager):
        manager.emit(
            make_event(TaskStarted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(
                TaskCompleted,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.1,
            )
        )
        await asyncio.sleep(0.1)
        from opentelemetry.trace import StatusCode

        spans = [s for s in manager.read_traces() if s.name == "task"]
        assert len(spans) == 1
        assert spans[0].end_time is not None
        assert spans[0].attributes.get("status") == "completed"
        assert spans[0].status.status_code == StatusCode.OK

    async def test_failed_span_status(self, manager):
        manager.emit(
            make_event(TaskStarted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(
                TaskFailed,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.05,
                error_type="ValueError",
            )
        )
        await asyncio.sleep(0.1)
        from opentelemetry.trace import StatusCode

        spans = [s for s in manager.read_traces() if s.name == "task"]
        assert spans[0].attributes.get("status") == "failed"
        assert spans[0].attributes.get("error_type") == "ValueError"
        assert spans[0].status.status_code == StatusCode.ERROR
        assert spans[0].status.description == "ValueError"

    async def test_session_span(self, manager):
        await manager.stop()
        session_spans = [s for s in manager.read_traces() if s.name == "session"]
        assert len(session_spans) == 1
        assert session_spans[0].end_time is not None
        # prevent double-stop in fixture teardown
        manager._telemetry = None  # type: ignore[attr-defined]

    async def test_session_ended_has_trace_and_span_ids(self):
        """SessionEnded must carry full trace+span IDs even though the span ends in the same
        batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(session_id="se-ids", checkpoint_path=tmpdir)
            await m.start()
            await m.stop()

            filepath = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            events = [json.loads(l) for l in open(filepath) if json.loads(l)["section"] == "event"]
            ended = next(e for e in events if e["event_type"] == "SessionEnded")
            assert ended["trace_id"] is not None, "SessionEnded: trace_id must not be null"
            assert ended["span_id"] is not None, "SessionEnded: span_id must not be null"

    async def test_task_span_created_at_task_created(self, manager):
        """Span must exist with valid trace/span context immediately after TaskCreated."""
        manager.emit(
            make_event(TaskCreated, session_id="test-session", backend="concurrent", task_id="t1")
        )
        await asyncio.sleep(0.1)
        assert "t1" in manager._active_spans
        ctx = manager._active_spans["t1"].get_span_context()
        assert ctx.is_valid
        assert ctx.trace_id != 0
        assert ctx.span_id != 0

    async def test_full_lifecycle_span_ids_consistent(self, manager):
        """All lifecycle events for a task must reference the same span_id."""
        received_ids: dict[str, tuple] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(session_id="ids-test", checkpoint_path=tmpdir)
            await m.start()
            for etype, kwargs in [
                (TaskCreated, {}),
                (TaskSubmitted, {}),
                (TaskQueued, {}),
                (TaskStarted, {}),
                (TaskCompleted, {"duration_seconds": 0.1}),
            ]:
                m.emit(
                    make_event(
                        etype, session_id="ids-test", backend="concurrent", task_id="t1", **kwargs
                    )
                )
            await asyncio.sleep(0.1)
            await m.stop()

            filepath = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            events = [
                json.loads(line)
                for line in open(filepath)
                if json.loads(line)["section"] == "event"
            ]
            lifecycle = {
                e["event_type"]: e
                for e in events
                if e["event_type"]
                in ("TaskCreated", "TaskSubmitted", "TaskQueued", "TaskStarted", "TaskCompleted")
            }

            for etype, e in lifecycle.items():
                assert e["trace_id"] is not None, f"{etype}: trace_id must not be null"
                assert e["span_id"] is not None, f"{etype}: span_id must not be null"

            span_ids = {e["span_id"] for e in lifecycle.values()}
            assert len(span_ids) == 1, (
                f"All lifecycle events must share one span_id, got {span_ids}"
            )

    @pytest.mark.parametrize(
        "terminal_cls,terminal_kwargs,terminal_etype",
        [
            (TaskFailed, {"duration_seconds": 0.1, "error_type": "RuntimeError"}, "TaskFailed"),
            (TaskCanceled, {"duration_seconds": 0.1}, "TaskCanceled"),
        ],
    )
    async def test_failed_and_canceled_same_batch_span_ids(
        self, terminal_cls, terminal_kwargs, terminal_etype
    ):
        """TaskFailed and TaskCanceled must carry the same span_id as the rest of the lifecycle even
        when all events land in the same batch (same-batch ordering hazard)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(session_id="fc-ids", checkpoint_path=tmpdir)
            await m.start()
            for etype, kwargs in [
                (TaskCreated, {}),
                (TaskSubmitted, {}),
                (TaskQueued, {}),
                (TaskStarted, {}),
                (terminal_cls, terminal_kwargs),
            ]:
                m.emit(
                    make_event(
                        etype, session_id="fc-ids", backend="concurrent", task_id="t1", **kwargs
                    )
                )
            await asyncio.sleep(0.1)
            await m.stop()

            filepath = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            events = [
                json.loads(line)
                for line in open(filepath)
                if json.loads(line)["section"] == "event"
            ]
            lifecycle = {
                e["event_type"]: e
                for e in events
                if e["event_type"]
                in ("TaskCreated", "TaskSubmitted", "TaskQueued", "TaskStarted", terminal_etype)
            }

            for etype, e in lifecycle.items():
                assert e["trace_id"] is not None, f"{etype}: trace_id must not be null"
                assert e["span_id"] is not None, f"{etype}: span_id must not be null"

            span_ids = {e["span_id"] for e in lifecycle.values()}
            assert len(span_ids) == 1, (
                f"All lifecycle events must share one span_id, got {span_ids}"
            )


class TestSubscriber:
    async def test_sync_subscriber_receives_events(self, manager):
        received = []
        manager.subscribe(received.append)
        manager.emit(
            make_event(TaskSubmitted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        await asyncio.sleep(0.1)
        assert any(e.event_type == "TaskSubmitted" for e in received)

    async def test_async_subscriber(self, manager):
        received = []

        async def handler(event):
            received.append(event)

        manager.subscribe(handler)
        manager.emit(
            make_event(TaskSubmitted, session_id="test-session", backend="concurrent", task_id="t2")
        )
        await asyncio.sleep(0.15)
        assert any(e.event_type == "TaskSubmitted" for e in received)

    async def test_subscriber_exception_does_not_crash_loop(self, manager):
        def bad_sub(event):
            raise RuntimeError("subscriber error")

        good_received = []
        manager.subscribe(bad_sub)
        manager.subscribe(good_received.append)
        manager.emit(
            make_event(TaskSubmitted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        await asyncio.sleep(0.1)
        # dispatch loop still alive and good subscriber still called
        assert manager._dispatch_task and not manager._dispatch_task.done()
        assert len(good_received) >= 1


class TestConvenienceMethods:
    async def test_task_count(self, manager):
        manager.emit(
            make_event(TaskSubmitted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(TaskSubmitted, session_id="test-session", backend="concurrent", task_id="t2")
        )
        manager.emit(
            make_event(
                TaskCompleted,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.1,
            )
        )
        manager.emit(
            make_event(
                TaskFailed,
                session_id="test-session",
                backend="concurrent",
                task_id="t2",
                duration_seconds=0.1,
                error_type="Err",
            )
        )
        await asyncio.sleep(0.1)
        counts = manager.task_count()
        assert counts["submitted"] == 2
        assert counts["completed"] == 1
        assert counts["failed"] == 1
        assert counts["running"] == 0

    async def test_task_spans_plain_dicts(self, manager):
        manager.emit(
            make_event(
                TaskCompleted,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.5,
                attributes={"executable": "/bin/echo", "task_type": "ComputeTask"},
            )
        )
        await asyncio.sleep(0.1)
        spans = manager.task_spans()
        assert len(spans) == 1
        s = spans[0]
        assert s["task_id"] == "t1"
        assert s["status"] == "completed"
        assert s["executable"] == "/bin/echo"
        assert s["task_type"] == "ComputeTask"
        assert s["duration_seconds"] is not None
        assert "span_id" in s
        assert "trace_id" in s

    async def test_task_spans_failed_has_error_type(self, manager):
        manager.emit(
            make_event(
                TaskFailed,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.1,
                error_type="ValueError",
                attributes={"error_type": "ValueError"},
            )
        )
        await asyncio.sleep(0.1)
        spans = manager.task_spans()
        assert spans[0]["status"] == "failed"
        assert spans[0]["error_type"] == "ValueError"

    async def test_session_duration_after_stop(self, manager):
        await manager.stop()
        dur = manager.session_duration()
        assert dur > 0.0
        manager._telemetry = None  # prevent double-stop in fixture

    async def test_summary_structure(self, manager):
        manager.emit(
            make_event(TaskSubmitted, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(
                TaskCompleted,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.3,
            )
        )
        await asyncio.sleep(0.1)
        s = manager.summary()
        assert "session_id" in s
        assert "tasks" in s
        assert "duration" in s
        assert s["tasks"]["submitted"] == 1
        assert s["tasks"]["completed"] == 1
        assert s["duration"]["mean_seconds"] is not None


class TestCheckpointFile:
    async def test_file_created_with_correct_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(
                session_id="cp-test",
                checkpoint_path=tmpdir,
            )
            await m.start()
            m.emit(
                make_event(TaskSubmitted, session_id="cp-test", backend="concurrent", task_id="t1")
            )
            await asyncio.sleep(0.05)
            await m.stop()

            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].startswith("cp-test.")
            assert files[0].endswith(".telemetry.jsonl")

    async def test_file_contains_all_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(session_id="cp-sections", checkpoint_path=tmpdir)
            await m.start()
            m.emit(
                make_event(
                    TaskSubmitted, session_id="cp-sections", backend="concurrent", task_id="t1"
                )
            )
            m.emit(
                make_event(
                    TaskCompleted,
                    session_id="cp-sections",
                    backend="concurrent",
                    task_id="t1",
                    duration_seconds=0.1,
                )
            )
            await asyncio.sleep(0.1)
            await m.stop()

            filepath = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            lines = [json.loads(l) for l in open(filepath)]
            sections = {l["section"] for l in lines}
            assert "event" in sections
            assert "metric" in sections
            assert "span" in sections

    async def test_events_have_event_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(session_id="cp-events", checkpoint_path=tmpdir)
            await m.start()
            m.emit(
                make_event(
                    TaskQueued,
                    session_id="cp-events",
                    backend="concurrent",
                    task_id="t1",
                    attributes={"executable": "/bin/echo", "task_type": "ComputeTask"},
                )
            )
            await asyncio.sleep(0.1)
            await m.stop()

            filepath = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            events = [json.loads(l) for l in open(filepath) if json.loads(l)["section"] == "event"]
            types = {e["event_type"] for e in events}
            assert "TaskQueued" in types
            queued = next(e for e in events if e["event_type"] == "TaskQueued")
            # executable and task_type are now in the attributes dict
            assert queued["attributes"]["executable"] == "/bin/echo"
            assert queued["attributes"]["task_type"] == "ComputeTask"

    async def test_no_file_when_path_is_none(self):
        m = TelemetryManager(session_id="no-file")
        await m.start()
        m.emit(make_event(TaskSubmitted, session_id="no-file", backend="concurrent", task_id="t1"))
        await asyncio.sleep(0.05)
        await m.stop()
        assert m._checkpoint_file is None

    async def test_custom_task_event_has_span_ids(self):
        """Custom events with task_id must carry the same trace/span as the task span."""
        from rhapsody.telemetry.events import define_event

        TaskResolved = define_event("asyncflow.TaskResolved")

        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(session_id="custom-task-ev", checkpoint_path=tmpdir)
            await m.start()
            m.emit(
                make_event(
                    TaskCreated, session_id="custom-task-ev", backend="concurrent", task_id="t1"
                )
            )
            m.emit(
                make_event(
                    TaskResolved, session_id="custom-task-ev", backend="concurrent", task_id="t1"
                )
            )
            m.emit(
                make_event(
                    TaskCompleted,
                    session_id="custom-task-ev",
                    backend="concurrent",
                    task_id="t1",
                    duration_seconds=0.1,
                )
            )
            await asyncio.sleep(0.1)
            await m.stop()

            filepath = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            rows = [json.loads(l) for l in open(filepath)]
            evs = {r["event_type"]: r for r in rows if r.get("section") == "event"}

            assert evs["asyncflow.TaskResolved"]["trace_id"] is not None
            assert evs["asyncflow.TaskResolved"]["span_id"] is not None
            assert (
                evs["asyncflow.TaskResolved"]["span_id"]
                == evs["TaskCreated"]["span_id"]
                == evs["TaskCompleted"]["span_id"]
            )

    async def test_custom_session_event_has_trace_id(self):
        """Custom events without task_id must carry the session trace/span IDs."""
        from rhapsody.telemetry.events import define_event

        SessionNote = define_event("user.SessionNote")

        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(session_id="custom-sess-ev", checkpoint_path=tmpdir)
            await m.start()
            m.emit(make_event(SessionNote, session_id="custom-sess-ev", backend="concurrent"))
            await asyncio.sleep(0.1)
            await m.stop()

            filepath = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            rows = [json.loads(l) for l in open(filepath)]
            evs = {r["event_type"]: r for r in rows if r.get("section") == "event"}

            assert evs["user.SessionNote"]["trace_id"] is not None
            assert evs["user.SessionNote"]["span_id"] is not None

    async def test_span_records_have_status_code(self):
        """Span JSONL records must include status_code field for OTel tool compatibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            m = TelemetryManager(session_id="sc-test", checkpoint_path=tmpdir)
            await m.start()
            m.emit(
                make_event(TaskCreated, session_id="sc-test", backend="concurrent", task_id="t1")
            )
            m.emit(
                make_event(
                    TaskCompleted,
                    session_id="sc-test",
                    backend="concurrent",
                    task_id="t1",
                    duration_seconds=0.1,
                )
            )
            m.emit(
                make_event(TaskCreated, session_id="sc-test", backend="concurrent", task_id="t2")
            )
            m.emit(
                make_event(
                    TaskFailed,
                    session_id="sc-test",
                    backend="concurrent",
                    task_id="t2",
                    duration_seconds=0.1,
                    error_type="RuntimeError",
                )
            )
            await asyncio.sleep(0.1)
            await m.stop()

            filepath = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            rows = [json.loads(l) for l in open(filepath)]
            spans = {
                r["attributes"]["task_id"]: r
                for r in rows
                if r.get("section") == "span" and r.get("name") == "task"
            }

            assert spans["t1"]["status_code"] == "OK"
            assert spans["t2"]["status_code"] == "ERROR"
            assert spans["t2"]["status_description"] == "RuntimeError"


class TestTaskStateTransitions:
    """Verify _on_task_state_change handles all RUNNING / DONE / FAILED orderings.

    These tests exercise TelemetryManager directly — no backend required. The RUNNING → DONE/FAILED
    path reflects how backends that emit an explicit RUNNING callback (e.g. Dragon V3 after
    build_task) are expected to behave. The DONE/FAILED-without-RUNNING path verifies the
    incomplete_lifecycle fallback.
    """

    def _make_task(self, uid: str, exc=None) -> dict:
        t: dict = {"uid": uid, "backend": "test", "task_type": "FunctionTask"}
        if exc is not None:
            t["exception"] = exc
        return t

    async def test_running_then_done_gives_accurate_duration(self, manager):
        """RUNNING → DONE: duration_seconds must reflect real elapsed time."""
        task = self._make_task("t-rd")
        manager._on_task_state_change(task, "RUNNING")
        await asyncio.sleep(0.05)
        manager._on_task_state_change(task, "DONE")
        await asyncio.sleep(0.1)

        spans = [s for s in manager.read_traces() if s.name == "task"]
        assert len(spans) == 1
        dur = (spans[0].end_time - spans[0].start_time) / 1e9
        assert 0.04 < dur < 5.0, f"Unexpected duration: {dur:.4f}s"
        assert spans[0].attributes.get("status") == "completed"
        assert not spans[0].attributes.get("incomplete_lifecycle", False)

    async def test_running_then_failed_gives_accurate_duration(self, manager):
        """RUNNING → FAILED: duration_seconds must reflect real elapsed time."""
        task = self._make_task("t-rf", exc=RuntimeError("boom"))
        manager._on_task_state_change(task, "RUNNING")
        await asyncio.sleep(0.05)
        manager._on_task_state_change(task, "FAILED")
        await asyncio.sleep(0.1)

        spans = [s for s in manager.read_traces() if s.name == "task"]
        assert len(spans) == 1
        dur = (spans[0].end_time - spans[0].start_time) / 1e9
        assert 0.04 < dur < 5.0, f"Unexpected duration: {dur:.4f}s"
        assert spans[0].attributes.get("status") == "failed"
        assert spans[0].attributes.get("error_type") == "RuntimeError"
        assert not spans[0].attributes.get("incomplete_lifecycle", False)

    async def test_done_without_running_sets_incomplete_lifecycle(self, manager):
        """DONE without prior RUNNING: incomplete_lifecycle=True is set."""
        task = self._make_task("t-d-no-run")
        manager._on_task_state_change(task, "DONE")
        await asyncio.sleep(0.1)

        spans = manager.task_spans()
        assert len(spans) == 1
        assert spans[0].get("incomplete_lifecycle") is True
        assert _sum_counter(manager.read_metrics(), "tasks_completed") >= 1

    async def test_failed_without_running_sets_incomplete_lifecycle(self, manager):
        """FAILED without prior RUNNING: incomplete_lifecycle=True, error_type preserved."""
        task = self._make_task("t-f-no-run", exc=ValueError("no start seen"))
        manager._on_task_state_change(task, "FAILED")
        await asyncio.sleep(0.1)

        spans = manager.task_spans()
        assert len(spans) == 1
        assert spans[0].get("incomplete_lifecycle") is True
        assert spans[0]["status"] == "failed"
        assert spans[0]["error_type"] == "ValueError"

    async def test_start_time_cleared_after_done(self, manager):
        """_task_start_times must not leak entries after DONE."""
        task = self._make_task("t-leak-done")
        manager._on_task_state_change(task, "RUNNING")
        assert "t-leak-done" in manager._task_start_times
        manager._on_task_state_change(task, "DONE")
        await asyncio.sleep(0.05)
        assert "t-leak-done" not in manager._task_start_times

    async def test_start_time_cleared_after_failed(self, manager):
        """_task_start_times must not leak entries after FAILED."""
        task = self._make_task("t-leak-fail", exc=Exception("x"))
        manager._on_task_state_change(task, "RUNNING")
        assert "t-leak-fail" in manager._task_start_times
        manager._on_task_state_change(task, "FAILED")
        await asyncio.sleep(0.05)
        assert "t-leak-fail" not in manager._task_start_times

    async def test_concurrent_tasks_tracked_independently(self, manager):
        """Multiple in-flight tasks must not cross-contaminate durations."""
        t1 = self._make_task("conc-1")
        t2 = self._make_task("conc-2")
        manager._on_task_state_change(t1, "RUNNING")
        await asyncio.sleep(0.03)
        manager._on_task_state_change(t2, "RUNNING")
        await asyncio.sleep(0.05)
        manager._on_task_state_change(t1, "DONE")
        await asyncio.sleep(0.04)
        manager._on_task_state_change(t2, "DONE")
        await asyncio.sleep(0.1)

        assert manager._task_start_times == {}

        spans = {s["task_id"]: s for s in manager.task_spans()}
        assert "conc-1" in spans
        assert "conc-2" in spans
        for tid, span in spans.items():
            assert not span.get("incomplete_lifecycle"), f"{tid} wrongly marked incomplete"


# ------------------------------------------------------------------
# Helpers to navigate OTel MetricsData
# ------------------------------------------------------------------


class TestOtelInjection:
    async def test_extra_span_processor_receives_spans(self):
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        m = TelemetryManager(
            session_id="inj-spans",
            span_processors=[SimpleSpanProcessor(exporter)],
        )
        await m.start()
        m.emit(make_event(TaskStarted, session_id="inj-spans", backend="concurrent", task_id="t1"))
        m.emit(
            make_event(
                TaskCompleted,
                session_id="inj-spans",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.1,
            )
        )
        await asyncio.sleep(0.1)
        await m.stop()

        assert any(s.name == "task" for s in exporter.get_finished_spans()), (
            "injected exporter did not receive the task span"
        )
        assert any(s.name == "task" for s in m.read_traces()), (
            "internal SpanBuffer must still be populated"
        )

    async def test_extra_metric_reader_receives_metrics(self):
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader as OtelReader

        reader = OtelReader()
        m = TelemetryManager(session_id="inj-metrics", metric_readers=[reader])
        await m.start()
        m.emit(
            make_event(TaskSubmitted, session_id="inj-metrics", backend="concurrent", task_id="t1")
        )
        await asyncio.sleep(0.1)

        metrics = reader.get_metrics_data()
        all_names = {
            metric.name
            for rm in (metrics.resource_metrics if metrics else [])
            for sm in rm.scope_metrics
            for metric in sm.metrics
        }
        assert "tasks_submitted" in all_names, (
            f"tasks_submitted not in injected reader metrics: {all_names}"
        )

        await m.stop()


class TestExportAsOtlp:
    async def test_custom_resource_attributes_in_otlp(self):
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": "my-app"})
        m = TelemetryManager(session_id="otlp-res", resource=resource)
        await m.start()
        m.emit(
            make_event(
                TaskCompleted,
                session_id="otlp-res",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.1,
            )
        )
        await asyncio.sleep(0.1)
        await m.stop()

        data = json.loads(m.export_as_otlp())
        res_attrs = {
            a["key"]: a["value"] for a in data["resourceSpans"][0]["resource"]["attributes"]
        }
        assert res_attrs.get("service.name") == {"stringValue": "my-app"}, (
            f"Expected 'my-app', got {res_attrs.get('service.name')}"
        )

    async def test_array_attribute_serialized_as_array_value(self):
        """Tuple span attribute must produce arrayValue in OTLP JSON."""
        m = TelemetryManager(session_id="otlp-arr")
        await m.start()
        m.register_span_enricher(lambda e: {"tags": ("alpha", "beta", "gamma")})
        m.emit(make_event(TaskCreated, session_id="otlp-arr", backend="concurrent", task_id="t1"))
        m.emit(
            make_event(
                TaskCompleted,
                session_id="otlp-arr",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.1,
            )
        )
        await asyncio.sleep(0.1)
        await m.stop()

        data = json.loads(m.export_as_otlp())
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        task_span = next(s for s in spans if s["name"] == "task")
        attrs = {a["key"]: a["value"] for a in task_span["attributes"]}

        assert "arrayValue" in attrs.get("tags", {}), (
            f"Expected arrayValue for 'tags', got {attrs.get('tags')}"
        )
        values = attrs["tags"]["arrayValue"]["values"]
        assert [v["stringValue"] for v in values] == ["alpha", "beta", "gamma"]


class TestTaskQueuedSpanEvent:
    async def test_task_queued_adds_span_event(self, manager):
        manager.emit(
            make_event(TaskCreated, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(TaskQueued, session_id="test-session", backend="concurrent", task_id="t1")
        )
        manager.emit(
            make_event(
                TaskCompleted,
                session_id="test-session",
                backend="concurrent",
                task_id="t1",
                duration_seconds=0.1,
            )
        )
        await asyncio.sleep(0.1)

        spans = [s for s in manager.read_traces() if s.name == "task"]
        assert len(spans) == 1
        span_event_names = [e.name for e in spans[0].events]
        assert "TaskQueued" in span_event_names

        queued_ev = next(e for e in spans[0].events if e.name == "TaskQueued")
        assert queued_ev.attributes.get("task_id") == "t1"

    async def test_task_queued_without_prior_created_is_ignored(self, manager):
        """TaskQueued for an unknown task_id must not crash — span not in _active_spans."""
        manager.emit(
            make_event(
                TaskQueued, session_id="test-session", backend="concurrent", task_id="unknown-t"
            )
        )
        await asyncio.sleep(0.1)
        assert manager._dispatch_task and not manager._dispatch_task.done()


# ------------------------------------------------------------------
# Helpers to navigate OTel MetricsData
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
