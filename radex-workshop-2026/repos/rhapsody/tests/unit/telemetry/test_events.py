"""Unit tests for rhapsody.telemetry.events."""

import dataclasses
import time

import pytest

from rhapsody.telemetry.events import BaseEvent
from rhapsody.telemetry.events import ResourceUpdate
from rhapsody.telemetry.events import SessionEnded
from rhapsody.telemetry.events import SessionStarted
from rhapsody.telemetry.events import TaskCompleted
from rhapsody.telemetry.events import TaskFailed
from rhapsody.telemetry.events import TaskQueued
from rhapsody.telemetry.events import TaskStarted
from rhapsody.telemetry.events import TaskSubmitted
from rhapsody.telemetry.events import make_event


def _base_kwargs():
    now = time.time()
    return {
        "event_id": "abc123",
        "event_time": now,
        "emit_time": now,
        "session_id": "session.001",
        "backend": "concurrent",
    }


class TestBaseEvent:
    def test_frozen(self):
        e = SessionStarted(**_base_kwargs())
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            e.session_id = "other"  # type: ignore[misc]

    def test_event_type_discriminator(self):
        assert SessionStarted(**_base_kwargs()).event_type == "SessionStarted"
        assert SessionEnded(**_base_kwargs(), duration_seconds=1.0).event_type == "SessionEnded"
        assert TaskSubmitted(**_base_kwargs(), task_id="t1").event_type == "TaskSubmitted"
        assert TaskQueued(**_base_kwargs(), task_id="t1").event_type == "TaskQueued"
        assert TaskStarted(**_base_kwargs(), task_id="t1").event_type == "TaskStarted"
        assert (
            TaskCompleted(**_base_kwargs(), task_id="t1", duration_seconds=0.5).event_type
            == "TaskCompleted"
        )
        assert (
            TaskFailed(
                **_base_kwargs(), task_id="t1", duration_seconds=0.1, error_type="ValueError"
            ).event_type
            == "TaskFailed"
        )
        assert (
            ResourceUpdate(
                **_base_kwargs(), node_id="n1", cpu_percent=10.0, memory_percent=20.0
            ).event_type
            == "ResourceUpdate"
        )


class TestAttributes:
    def test_task_submitted_carries_attributes(self):
        e = make_event(
            TaskSubmitted,
            session_id="s",
            backend="b",
            task_id="t1",
            attributes={"executable": "/bin/echo", "task_type": "ComputeTask"},
        )
        assert e.attributes["executable"] == "/bin/echo"
        assert e.attributes["task_type"] == "ComputeTask"

    def test_task_queued_carries_attributes(self):
        e = make_event(
            TaskQueued,
            session_id="s",
            backend="b",
            task_id="t1",
            attributes={"executable": "/bin/echo", "task_type": "ComputeTask"},
        )
        assert e.attributes["executable"] == "/bin/echo"
        assert e.attributes["task_type"] == "ComputeTask"

    def test_task_started_carries_attributes(self):
        e = make_event(
            TaskStarted,
            session_id="s",
            backend="b",
            task_id="t1",
            attributes={"executable": "my_func", "task_type": "FunctionTask"},
        )
        assert e.attributes["executable"] == "my_func"
        assert e.attributes["task_type"] == "FunctionTask"

    def test_task_completed_carries_attributes(self):
        e = make_event(
            TaskCompleted,
            session_id="s",
            backend="b",
            task_id="t1",
            duration_seconds=0.5,
            attributes={"executable": "/bin/echo", "task_type": "ComputeTask"},
        )
        assert e.attributes["executable"] == "/bin/echo"
        assert e.duration_seconds == 0.5

    def test_task_failed_carries_attributes(self):
        e = make_event(
            TaskFailed,
            session_id="s",
            backend="b",
            task_id="t1",
            duration_seconds=0.1,
            error_type="ValueError",
            attributes={
                "executable": "/bin/false",
                "task_type": "ComputeTask",
                "error_type": "ValueError",
            },
        )
        assert e.error_type == "ValueError"
        assert e.attributes.get("error_type") == "ValueError"

    def test_attributes_default_empty(self):
        e = make_event(SessionStarted, session_id="s", backend="b")
        assert e.attributes == {}

    def test_incomplete_lifecycle_flag(self):
        e = make_event(
            TaskCompleted,
            session_id="s",
            backend="b",
            task_id="t1",
            duration_seconds=0.0,
            attributes={"incomplete_lifecycle": True},
        )
        assert e.attributes.get("incomplete_lifecycle") is True
        assert e.duration_seconds == 0.0


class TestClockConsistency:
    def test_make_event_uses_wall_clock(self):
        """emit_time and event_time must both be wall-clock (time.time()) values."""
        before = time.time()
        e = make_event(SessionStarted, session_id="s", backend="b")
        after = time.time()
        # Both timestamps are wall-clock — they should be in [before, after]
        assert before <= e.event_time <= after
        assert before <= e.emit_time <= after

    def test_emit_time_ge_event_time(self):
        """emit_time >= event_time always holds (queue entry >= occurrence)."""
        e = make_event(
            TaskCompleted, session_id="s", backend="b", task_id="t1", duration_seconds=0.5
        )
        assert e.emit_time >= e.event_time

    def test_same_clock_subtractable(self):
        """emit_time - event_time gives a meaningful (non-negative) latency in seconds."""
        past = time.time() - 0.1
        e = make_event(TaskStarted, session_id="s", backend="b", task_id="t1", event_time=past)
        latency = e.emit_time - e.event_time
        assert latency >= 0.0
        assert latency < 1.0  # should be well under 1 second in tests


class TestMakeEvent:
    def test_unique_ids(self):
        ids = {
            make_event(TaskSubmitted, session_id="s", backend="b", task_id=f"t{i}").event_id
            for i in range(1000)
        }
        assert len(ids) == 1000

    def test_immutable_fan_out(self):
        e = make_event(
            TaskCompleted, session_id="s", backend="b", task_id="t1", duration_seconds=1.0
        )
        refs = [e, e, e]
        assert all(r is e for r in refs)


class TestResourceUpdate:
    def test_io_fields_optional(self):
        e = ResourceUpdate(**_base_kwargs(), node_id="n1", cpu_percent=5.0, memory_percent=10.0)
        assert e.disk_read_bytes is None
        assert e.disk_write_bytes is None
        assert e.net_sent_bytes is None
        assert e.net_recv_bytes is None
        assert e.gpu_percent is None

    def test_io_fields_set(self):
        e = make_event(
            ResourceUpdate,
            session_id="s",
            backend="b",
            node_id="n1",
            cpu_percent=5.0,
            memory_percent=10.0,
            disk_read_bytes=1024.0,
            disk_write_bytes=512.0,
            net_sent_bytes=2048.0,
            net_recv_bytes=4096.0,
        )
        assert e.disk_read_bytes == 1024.0
        assert e.net_recv_bytes == 4096.0

    def test_task_id_and_node_id_optional(self):
        e = SessionStarted(**_base_kwargs())
        assert e.task_id is None
        assert e.node_id is None
