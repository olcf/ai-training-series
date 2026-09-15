"""Dragon telemetry adapter tests.

Run with Dragon launcher (required for Dragon runtime):
    dragon /home/aymen/ve/rhapsody/bin/python3 -m pytest tests/unit/telemetry/test_adapters_dragon.py -v

Tests cover:
  - No-op guard when DRAGON_TELEMETRY_LEVEL < 1
  - Adapter starts, spawns collectors, emits ResourceUpdate events
  - stop() cleanly terminates collector processes
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from unittest.mock import MagicMock

import pytest

# Skip the whole module if Dragon is not installed
pytest.importorskip("dragon", reason="Dragon runtime required for this test module")

import os as _os

# DRAGON_DEFAULT_PD is set by the Dragon launcher — absent in plain pytest runs.
# Tests that exercise real Dragon APIs (Queue, ProcessGroup, MemoryPool) must be
# skipped when Dragon's runtime is not active, but the no-op guard test can run
# without the runtime (it exits before touching any Dragon API).
_dragon_runtime_available = _os.getenv("DRAGON_DEFAULT_PD") is not None
_requires_dragon_runtime = pytest.mark.skipif(
    not _dragon_runtime_available,
    reason="Dragon runtime not active — run with: dragon python3 -m pytest",
)

from rhapsody.telemetry.adapters.dragon import DragonTelemetryAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal TelemetryManager stand-in
# ---------------------------------------------------------------------------


class _FakeManager:
    """Minimal stand-in for TelemetryManager — records emitted events."""

    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def emit(self, event):
        with self._lock:
            self.events.append(event)

    def register_adapter(self, adapter) -> None:
        """Inject self as adapter._manager (mirrors TelemetryManager.register_adapter)."""
        adapter._manager = self

    def resource_updates(self):
        with self._lock:
            from rhapsody.telemetry.events import ResourceUpdate

            return [e for e in self.events if isinstance(e, ResourceUpdate)]


# ---------------------------------------------------------------------------
# Helper: run an async coroutine in a fresh event loop from a sync test
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test: no-op when DRAGON_TELEMETRY_LEVEL is 0
# ---------------------------------------------------------------------------


def test_noop_when_telemetry_level_zero():
    """Adapter must silently do nothing when DRAGON_TELEMETRY_LEVEL < 1."""
    original = os.environ.get("DRAGON_TELEMETRY_LEVEL")
    os.environ["DRAGON_TELEMETRY_LEVEL"] = "0"
    try:
        loop = asyncio.new_event_loop()
        adapter = DragonTelemetryAdapter(
            session_id="test-session",
            backend_name="dragon",
            interval=0.5,
        )
        manager = _FakeManager()

        async def _check():
            manager.register_adapter(adapter)
            adapter.start()
            # Give thread a moment to detect level < 1 and exit
            await asyncio.sleep(0.3)
            adapter.stop()

        loop.run_until_complete(_check())
        loop.close()

        assert manager.resource_updates() == [], (
            "No ResourceUpdate events should be emitted when DRAGON_TELEMETRY_LEVEL=0"
        )
    finally:
        if original is None:
            os.environ.pop("DRAGON_TELEMETRY_LEVEL", None)
        else:
            os.environ["DRAGON_TELEMETRY_LEVEL"] = original


# ---------------------------------------------------------------------------
# Test: adapter emits ResourceUpdate within reasonable time
# ---------------------------------------------------------------------------


@_requires_dragon_runtime
def test_adapter_emits_resource_update():
    """Full adapter lifecycle: start → collect → at least one ResourceUpdate → stop.

    DRAGON_TELEMETRY_LEVEL is set to 1 so CPU/RAM are collected.
    GPU would require level >= 3.
    """
    original = os.environ.get("DRAGON_TELEMETRY_LEVEL")
    os.environ["DRAGON_TELEMETRY_LEVEL"] = "1"
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        adapter = DragonTelemetryAdapter(
            session_id="test-dragon-adapter",
            backend_name="dragon",
            interval=1.0,
        )
        manager = _FakeManager()

        async def _run_adapter():
            manager.register_adapter(adapter)
            adapter.start()
            # Wait up to 15 seconds for at least one ResourceUpdate
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                if manager.resource_updates():
                    break
            adapter.stop()

        loop.run_until_complete(_run_adapter())
        loop.close()

        updates = manager.resource_updates()
        assert len(updates) >= 1, (
            f"Expected at least one ResourceUpdate event, got {len(updates)}. "
            f"All events: {[type(e).__name__ for e in manager.events]}"
        )

        ev = updates[0]
        assert ev.session_id == "test-dragon-adapter"
        assert ev.backend == "dragon"
        assert ev.node_id is not None and ev.node_id != ""

        # cpu_percent and memory_percent must be collected at level=1
        assert ev.cpu_percent is not None, f"cpu_percent should not be None; event={ev}"
        assert ev.memory_percent is not None, f"memory_percent should not be None; event={ev}"
        assert 0.0 <= ev.cpu_percent <= 100.0, f"cpu_percent out of range: {ev.cpu_percent}"
        assert 0.0 <= ev.memory_percent <= 100.0, (
            f"memory_percent out of range: {ev.memory_percent}"
        )

    finally:
        if original is None:
            os.environ.pop("DRAGON_TELEMETRY_LEVEL", None)
        else:
            os.environ["DRAGON_TELEMETRY_LEVEL"] = original


# ---------------------------------------------------------------------------
# Test: stop() terminates the adapter cleanly
# ---------------------------------------------------------------------------


@_requires_dragon_runtime
def test_adapter_stops_cleanly():
    """Stop() must cause the daemon thread to exit within a reasonable timeout."""
    original = os.environ.get("DRAGON_TELEMETRY_LEVEL")
    os.environ["DRAGON_TELEMETRY_LEVEL"] = "1"
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        adapter = DragonTelemetryAdapter(
            session_id="test-stop",
            backend_name="dragon",
            interval=1.0,
        )
        manager = _FakeManager()

        async def _lifecycle():
            manager.register_adapter(adapter)
            adapter.start()
            # Wait until at least spawned (give up to 10s)
            await asyncio.sleep(5.0)
            adapter.stop()

        loop.run_until_complete(_lifecycle())
        loop.close()

        # Thread should exit within a few seconds of stop()
        if adapter._thread is not None:
            adapter._thread.join(timeout=10.0)
            assert not adapter._thread.is_alive(), (
                "DragonTelemetryAdapter daemon thread did not exit after stop()"
            )

    finally:
        if original is None:
            os.environ.pop("DRAGON_TELEMETRY_LEVEL", None)
        else:
            os.environ["DRAGON_TELEMETRY_LEVEL"] = original


# ---------------------------------------------------------------------------
# Test: worker + collector in isolation
# ---------------------------------------------------------------------------


@_requires_dragon_runtime
def test_collector_worker_puts_on_queue():
    """_rhapsody_telemetry_worker should put at least one dps packet on the queue.

    Runs the worker directly (in-process, no ProcessGroup) to verify that psutil CPU/RAM collection
    and the DragonQueue.put() path work correctly.
    """
    import threading

    from dragon.native.event import Event as DragonEvent
    from dragon.native.queue import Queue as DragonQueue

    from rhapsody.telemetry.adapters.dragon import _rhapsody_telemetry_worker

    result_queue = DragonQueue(maxsize=100)
    _shutdown_event = DragonEvent()
    received = []

    def _drain():
        """Drain the queue; stop as soon as we get one packet."""
        import queue as stdlib_queue

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                data = result_queue.get(timeout=0.5)
                received.append(data)
                return
            except stdlib_queue.Empty:
                continue

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()

    def _run_worker():
        try:
            _rhapsody_telemetry_worker(result_queue, _shutdown_event, 0.5)
        except Exception:
            pass

    worker_thread = threading.Thread(target=_run_worker, daemon=True)
    worker_thread.start()

    drain_thread.join(timeout=15.0)
    _shutdown_event.set()  # signal worker to exit cleanly

    assert len(received) >= 1, "No data arrived from _rhapsody_telemetry_worker within 15 seconds"

    data = received[0]
    assert "host" in data, f"Missing 'host' key: {data}"
    assert "dps" in data, f"Missing 'dps' key: {data}"
    assert len(data["dps"]) > 0, f"Empty dps list: {data}"

    metrics = {dp["metric"] for dp in data["dps"]}
    assert "cpu_percent" in metrics, f"cpu_percent missing from dps metrics: {metrics}"
    assert "used_RAM" in metrics, f"used_RAM missing from dps metrics: {metrics}"
