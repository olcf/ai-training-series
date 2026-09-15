"""OTel-aligned ResourceUpdate contract tests for all telemetry adapters.

Each adapter is parametrized with its declared capabilities.  A single shared
assertion function (imported from conftest) verifies every emitted event
against those capabilities.

Adding a new adapter:
    1. Implement a factory function that returns a started-ready adapter.
    2. Declare its AdapterCapabilities.
    3. Add one entry to ADAPTER_PARAMS.

Dragon tests are skipped when the Dragon runtime is not active.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed")

from rhapsody.telemetry.events import ResourceUpdate
from rhapsody.telemetry.manager import TelemetryManager

from .conftest import AdapterCapabilities
from .conftest import assert_resource_update_contract

# ---------------------------------------------------------------------------
# Dragon runtime detection (same pattern as test_adapters_dragon.py)
# ---------------------------------------------------------------------------

_dragon_available = False
try:
    import dragon  # noqa: F401

    _dragon_available = os.getenv("DRAGON_DEFAULT_PD") is not None
except ImportError:
    pass

_requires_dragon = pytest.mark.skipif(
    not _dragon_available,
    reason="Dragon runtime not active — run with: dragon python3 -m pytest",
)

# ---------------------------------------------------------------------------
# Adapter capabilities declarations
# ---------------------------------------------------------------------------

CAPS_CONCURRENT = AdapterCapabilities(
    name="concurrent",
    has_node_id=True,
    has_gpu_aggregate=False,  # pynvml absent on this machine → no GPU
    has_gpu_per_device=False,  # same
    has_disk_io=True,
    has_net_io=True,
)

CAPS_DASK = AdapterCapabilities(
    name="dask",
    has_node_id=True,
    has_gpu_aggregate=False,  # Dask scheduler_info never exposes GPU
    has_gpu_per_device=False,
    has_disk_io=True,  # Dask exposes read_bytes/write_bytes; first poll = None
    has_net_io=False,  # Dask does not expose net I/O
)

CAPS_DRAGON = AdapterCapabilities(
    name="dragon",
    has_node_id=True,
    has_gpu_aggregate=False,  # GPU absent or driver "Not Supported" on test machine
    has_gpu_per_device=False,  # same
    has_disk_io=True,
    has_net_io=True,
)

# ---------------------------------------------------------------------------
# Adapter factories
# ---------------------------------------------------------------------------


def _make_concurrent_adapter(session_id: str = "contract-session") -> Any:
    from rhapsody.telemetry.adapters.concurrent import ConcurrentTelemetryAdapter

    return ConcurrentTelemetryAdapter(
        session_id=session_id,
        backend_name="concurrent",
        interval=0.05,
    )


def _make_dask_adapter(session_id: str = "contract-session") -> Any:
    from rhapsody.telemetry.adapters.dask import DaskTelemetryAdapter

    workers = {
        "tcp://w0:8000": {
            "host": "node0",
            "memory_limit": 1024 * 1024 * 1024,
            "metrics": {
                "cpu": 42.0,
                "memory": 256 * 1024 * 1024,
                "read_bytes": 1024.0,
                "write_bytes": 512.0,
            },
        }
    }
    client = MagicMock()
    client.scheduler_info.return_value = {"workers": workers}
    return DaskTelemetryAdapter(
        client=client,
        session_id=session_id,
        backend_name="dask",
        interval=0.05,
    )


def _make_dragon_adapter(session_id: str = "contract-session") -> Any:
    from rhapsody.telemetry.adapters.dragon import DragonTelemetryAdapter

    return DragonTelemetryAdapter(
        session_id=session_id,
        backend_name="dragon",
        interval=1.0,
    )


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------

ADAPTER_PARAMS = [
    pytest.param("concurrent", _make_concurrent_adapter, CAPS_CONCURRENT, id="concurrent"),
    pytest.param("dask", _make_dask_adapter, CAPS_DASK, id="dask"),
    pytest.param(
        "dragon",
        _make_dragon_adapter,
        CAPS_DRAGON,
        id="dragon",
        marks=_requires_dragon,
    ),
]

# ---------------------------------------------------------------------------
# Shared manager fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def manager():
    m = TelemetryManager(session_id="contract-session")
    await m.start()
    yield m
    if m._running:
        await m.stop()


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestResourceUpdateOTelContract:
    """Verify that every adapter emits ResourceUpdate events in the correct OTel structure."""

    @pytest.mark.parametrize("name,make_adapter,caps", ADAPTER_PARAMS)
    async def test_all_events_satisfy_contract(self, manager, name, make_adapter, caps):
        """Every ResourceUpdate from every adapter must pass assert_resource_update_contract."""
        received: list[ResourceUpdate] = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        wait = 0.3 if name != "dragon" else 15.0

        with patch("pynvml.nvmlInit", side_effect=Exception("no GPU")):
            adapter = make_adapter()
            manager.register_adapter(adapter)
            adapter.start()
            await asyncio.sleep(wait)
            adapter.stop()

        assert len(received) >= 1, f"{name}: no ResourceUpdate events emitted within {wait}s"

        for ev in received:
            assert_resource_update_contract(ev, caps)

    @pytest.mark.parametrize("name,make_adapter,caps", ADAPTER_PARAMS)
    async def test_node_level_events_have_no_per_device_gpu_id(
        self, manager, name, make_adapter, caps
    ):
        """Node-level ResourceUpdate events (gpu_id=None) must never carry a device index."""
        received: list[ResourceUpdate] = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        wait = 0.3 if name != "dragon" else 15.0

        with patch("pynvml.nvmlInit", side_effect=Exception("no GPU")):
            adapter = make_adapter()
            manager.register_adapter(adapter)
            adapter.start()
            await asyncio.sleep(wait)
            adapter.stop()

        node_events = [e for e in received if e.gpu_id is None]
        assert len(node_events) >= 1, f"{name}: no node-level events (gpu_id=None) emitted"

    @pytest.mark.parametrize("name,make_adapter,caps", ADAPTER_PARAMS)
    async def test_session_id_and_backend_always_set(self, manager, name, make_adapter, caps):
        """session_id and backend must be set on every event — required for OTel parent linkage."""
        received: list[ResourceUpdate] = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        wait = 0.3 if name != "dragon" else 15.0

        with patch("pynvml.nvmlInit", side_effect=Exception("no GPU")):
            adapter = make_adapter(session_id="contract-session")
            manager.register_adapter(adapter)
            adapter.start()
            await asyncio.sleep(wait)
            adapter.stop()

        assert len(received) >= 1
        for ev in received:
            assert ev.session_id == "contract-session", (
                f"{name}: session_id mismatch — got {ev.session_id!r}"
            )
            assert ev.backend is not None and ev.backend != "", (
                f"{name}: backend must be non-empty string, got {ev.backend!r}"
            )

    @pytest.mark.parametrize(
        "name,make_adapter,caps",
        [
            pytest.param("concurrent", _make_concurrent_adapter, CAPS_CONCURRENT, id="concurrent"),
            pytest.param("dask", _make_dask_adapter, CAPS_DASK, id="dask"),
        ],
    )
    async def test_per_device_events_have_null_io(self, manager, name, make_adapter, caps):
        """When gpu_id=N events are emitted, disk/net fields must all be None.

        Only runs for adapters that can emit per-device GPU events (pynvml path). Uses a mocked
        pynvml with 2 devices to force per-device emission.
        """
        received: list[ResourceUpdate] = []
        manager.subscribe(lambda e: received.append(e) if isinstance(e, ResourceUpdate) else None)

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=77.0)

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            adapter = make_adapter()
            manager.register_adapter(adapter)
            adapter.start()
            await asyncio.sleep(0.3)
            adapter.stop()

        per_device = [e for e in received if e.gpu_id is not None]
        if not per_device:
            pytest.skip(f"{name}: no per-device GPU events emitted (pynvml mock may not apply)")

        for ev in per_device:
            assert ev.disk_read_bytes is None, (
                f"{name}: gpu_id={ev.gpu_id} event must have disk_read_bytes=None"
            )
            assert ev.disk_write_bytes is None, (
                f"{name}: gpu_id={ev.gpu_id} event must have disk_write_bytes=None"
            )
            assert ev.net_sent_bytes is None, (
                f"{name}: gpu_id={ev.gpu_id} event must have net_sent_bytes=None"
            )
            assert ev.net_recv_bytes is None, (
                f"{name}: gpu_id={ev.gpu_id} event must have net_recv_bytes=None"
            )
