"""Unit tests for ConcurrentTelemetryAdapter."""

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed")

from rhapsody.telemetry.adapters.concurrent import ConcurrentTelemetryAdapter
from rhapsody.telemetry.events import ResourceUpdate
from rhapsody.telemetry.manager import TelemetryManager


@pytest.fixture
async def manager():
    m = TelemetryManager(session_id="test-session")
    await m.start()
    yield m
    if m._running:
        await m.stop()


class TestConcurrentTelemetryAdapter:
    async def test_emits_resource_update(self, manager):
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        adapter = ConcurrentTelemetryAdapter(
            session_id="test-session",
            backend_name="concurrent",
            interval=0.05,  # fast for testing
        )

        with (
            patch("psutil.cpu_percent", return_value=42.0),
            patch("psutil.virtual_memory", return_value=MagicMock(percent=55.0)),
        ):
            manager.register_adapter(adapter)
            adapter.start()
            await asyncio.sleep(0.3)  # give the daemon thread time to fire
            adapter.stop()

        assert len(received) >= 1
        ev = received[0]
        assert ev.cpu_percent == 42.0
        assert ev.memory_percent == 55.0
        assert ev.session_id == "test-session"
        assert ev.backend == "concurrent"

    async def test_gpu_none_when_unavailable(self, manager):
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        adapter = ConcurrentTelemetryAdapter(
            session_id="test-session",
            backend_name="concurrent",
            interval=0.05,
        )

        with (
            patch("psutil.cpu_percent", return_value=10.0),
            patch("psutil.virtual_memory", return_value=MagicMock(percent=20.0)),
            patch("pynvml.nvmlInit", side_effect=Exception("no NVIDIA GPU")),
        ):
            manager.register_adapter(adapter)
            adapter.start()
            await asyncio.sleep(0.3)
            adapter.stop()

        assert len(received) >= 1
        # Filter to node-level event (gpu_id=None); gpu_percent must be None when pynvml fails
        node_events = [e for e in received if e.gpu_id is None]
        assert len(node_events) >= 1
        assert node_events[0].gpu_percent is None

    async def test_uses_daemon_thread(self, manager):
        """Adapter must use a daemon thread (not an asyncio task) for Dragon compatibility."""
        adapter = ConcurrentTelemetryAdapter(
            session_id="test-session",
            backend_name="concurrent",
            interval=100.0,
        )
        with (
            patch("psutil.cpu_percent", return_value=0.0),
            patch("psutil.virtual_memory", return_value=MagicMock(percent=0.0)),
        ):
            manager.register_adapter(adapter)
            adapter.start()
            assert adapter._thread is not None
            assert adapter._thread.is_alive()
            assert adapter._thread.daemon  # must be daemon so it doesn't block process exit
            adapter.stop()

    async def test_no_psutil_is_noop(self, manager):
        """If psutil is not importable, adapter should not crash."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        adapter = ConcurrentTelemetryAdapter(
            session_id="test-session",
            backend_name="concurrent",
            interval=0.05,
        )
        with patch("builtins.__import__", side_effect=mock_import):
            manager.register_adapter(adapter)
            adapter.start()
            await asyncio.sleep(0.15)
            adapter.stop()
        # No exception raised — adapter is a silent no-op
