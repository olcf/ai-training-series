"""Unit tests for DaskTelemetryAdapter."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed")

from rhapsody.telemetry.adapters.dask import DaskTelemetryAdapter
from rhapsody.telemetry.events import ResourceUpdate
from rhapsody.telemetry.manager import TelemetryManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(workers: dict) -> MagicMock:
    """Return a minimal Dask client mock with a fixed scheduler_info()."""
    client = MagicMock()
    client.scheduler_info.return_value = {"workers": workers}
    return client


def _worker(
    host: str = "node0",
    cpu: float = 50.0,
    memory: int = 512 * 1024 * 1024,
    memory_limit: int = 1024 * 1024 * 1024,
    read_bytes: float = 0.0,
    write_bytes: float = 0.0,
) -> dict:
    return {
        "host": host,
        "memory_limit": memory_limit,
        "metrics": {
            "cpu": cpu,
            "memory": memory,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
        },
    }


@pytest.fixture
async def manager():
    m = TelemetryManager(session_id="test-session")
    await m.start()
    yield m
    if m._running:
        await m.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDaskTelemetryAdapter:
    async def test_emits_resource_update(self, manager):
        """Adapter emits at least one ResourceUpdate per poll."""
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        client = _make_client({"tcp://w0:8000": _worker(host="node0", cpu=60.0)})
        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="test-session",
            backend_name="dask",
            interval=0.05,
        )
        manager.register_adapter(adapter)
        adapter.start()
        await asyncio.sleep(0.25)
        adapter.stop()

        assert len(received) >= 1
        ev = received[0]
        assert ev.session_id == "test-session"
        assert ev.backend == "dask"
        assert ev.node_id == "node0"

    async def test_cpu_and_memory_values(self, manager):
        """cpu_percent and memory_percent are derived correctly from Dask metrics."""
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        # memory = 256 MB, limit = 1 GB → 25 %
        client = _make_client(
            {
                "tcp://w0:8000": _worker(
                    host="node0",
                    cpu=75.0,
                    memory=256 * 1024 * 1024,
                    memory_limit=1024 * 1024 * 1024,
                )
            }
        )
        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="test-session",
            backend_name="dask",
            interval=0.05,
        )
        manager.register_adapter(adapter)
        adapter.start()
        await asyncio.sleep(0.2)
        adapter.stop()

        assert len(received) >= 1
        ev = received[0]
        assert ev.cpu_percent == 75.0
        assert abs(ev.memory_percent - 25.0) < 0.1

    async def test_gpu_always_none(self, manager):
        """gpu_percent and gpu_id are always None — Dask does not expose GPU."""
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        client = _make_client({"tcp://w0:8000": _worker()})
        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="s",
            backend_name="dask",
            interval=0.05,
        )
        manager.register_adapter(adapter)
        adapter.start()
        await asyncio.sleep(0.2)
        adapter.stop()

        assert all(e.gpu_percent is None for e in received)
        assert all(e.gpu_id is None for e in received), (
            "gpu_id must be None (node-level aggregate, OTel-compatible)"
        )

    async def test_disk_delta_on_second_poll(self, manager):
        """Disk I/O bytes are per-interval deltas, not cumulative totals.

        First poll → disk_read/write = None (no baseline yet).
        Second poll → delta = new_cumulative - prev_cumulative.
        """
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        # First poll: cumulative = 1 MB read / 512 KB write
        # Second poll: cumulative = 3 MB read / 1 MB write  → delta = 2 MB / 512 KB
        call_count = 0
        base_read, base_write = 1 * 1024 * 1024, 512 * 1024

        def scheduler_info_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "workers": {
                        "tcp://w0:8000": _worker(
                            read_bytes=float(base_read), write_bytes=float(base_write)
                        )
                    }
                }
            return {
                "workers": {
                    "tcp://w0:8000": _worker(
                        read_bytes=float(base_read + 2 * 1024 * 1024),
                        write_bytes=float(base_write + 512 * 1024),
                    )
                }
            }

        client = MagicMock()
        client.scheduler_info.side_effect = scheduler_info_side_effect

        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="s",
            backend_name="dask",
            interval=0.05,
        )
        manager.register_adapter(adapter)
        adapter.start()
        await asyncio.sleep(0.2)
        adapter.stop()

        assert len(received) >= 2, f"Expected ≥2 events, got {len(received)}"

        # First event — no baseline yet, disk values must be None
        assert received[0].disk_read_bytes is None
        assert received[0].disk_write_bytes is None

        # Second event — delta
        assert received[1].disk_read_bytes == pytest.approx(2 * 1024 * 1024)
        assert received[1].disk_write_bytes == pytest.approx(512 * 1024)

    async def test_net_always_none(self, manager):
        """net_sent/recv_bytes are None — Dask scheduler_info does not expose net I/O."""
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        client = _make_client({"tcp://w0:8000": _worker()})
        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="s",
            backend_name="dask",
            interval=0.05,
        )
        manager.register_adapter(adapter)
        adapter.start()
        await asyncio.sleep(0.2)
        adapter.stop()

        assert all(e.net_sent_bytes is None for e in received)
        assert all(e.net_recv_bytes is None for e in received)

    async def test_multi_worker_emits_per_worker(self, manager):
        """One ResourceUpdate is emitted per Dask worker per poll."""
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        client = _make_client(
            {
                "tcp://w0:8000": _worker(host="node0", cpu=10.0),
                "tcp://w1:8000": _worker(host="node1", cpu=90.0),
            }
        )
        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="s",
            backend_name="dask",
            interval=0.05,
        )
        manager.register_adapter(adapter)
        adapter.start()
        await asyncio.sleep(0.2)
        adapter.stop()

        node_ids = {e.node_id for e in received}
        assert "node0" in node_ids
        assert "node1" in node_ids

    async def test_scheduler_error_does_not_crash(self, manager):
        """A transient scheduler_info() failure must not crash the adapter."""
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        client = MagicMock()
        client.scheduler_info.side_effect = RuntimeError("scheduler unavailable")

        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="s",
            backend_name="dask",
            interval=0.05,
        )
        manager.register_adapter(adapter)
        adapter.start()
        await asyncio.sleep(0.2)
        adapter.stop()
        # No crash — received may be empty but no exception propagated

    async def test_stop_cancels_task(self, manager):
        """Stop() must cancel the asyncio Task cleanly."""
        client = _make_client({"tcp://w0:8000": _worker()})
        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="s",
            backend_name="dask",
            interval=100.0,
        )
        manager.register_adapter(adapter)
        adapter.start()
        assert adapter._task is not None
        assert not adapter._task.done()

        adapter.stop()
        await asyncio.sleep(0.05)  # let the event loop process the cancel
        assert adapter._task.done()

    async def test_otel_structure_node_level(self, manager):
        """Every emitted event must have gpu_id=None and gpu_percent=None (OTel node-level)."""
        received = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        client = _make_client({"tcp://w0:8000": _worker(host="nodeA", cpu=33.0)})
        adapter = DaskTelemetryAdapter(
            client=client,
            session_id="test-session",
            backend_name="dask",
            interval=0.05,
        )
        manager.register_adapter(adapter)
        adapter.start()
        await asyncio.sleep(0.2)
        adapter.stop()

        assert len(received) >= 1
        for ev in received:
            # OTel hierarchy: gpu_id=None means node-level aggregate, not per-device
            assert ev.gpu_id is None, f"Expected gpu_id=None, got {ev.gpu_id}"
            assert ev.gpu_percent is None, f"Expected gpu_percent=None, got {ev.gpu_percent}"
            # Must carry session context for OTel parent-child correlation
            assert ev.session_id == "test-session"
            assert ev.backend == "dask"
