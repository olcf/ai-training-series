"""Telemetry adapter for DaskExecutionBackend.

Relies purely on Dask's native scheduler_info() API — no psutil, no extra deps.

Per-worker metrics exposed by Dask:

    workers[addr]["metrics"]["cpu"]         → CPU utilization %
    workers[addr]["metrics"]["memory"]      → bytes used
    workers[addr]["memory_limit"]           → bytes total
    workers[addr]["metrics"]["read_bytes"]  → cumulative disk read bytes
    workers[addr]["metrics"]["write_bytes"] → cumulative disk write bytes
    workers[addr]["host"]                   → hostname (used as node_id)

GPU utilization is not exposed by Dask scheduler_info — gpu_percent is always None.
Disk read/write are cumulative counters; this adapter tracks per-worker deltas.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from typing import Any

from rhapsody.telemetry.adapters.base import TelemetryAdapter
from rhapsody.telemetry.events import ResourceUpdate
from rhapsody.telemetry.events import make_event

if TYPE_CHECKING:
    from rhapsody.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


class DaskTelemetryAdapter(TelemetryAdapter):
    """Collects per-worker resource metrics from Dask's scheduler.

    Emits one :class:`~rhapsody.telemetry.events.ResourceUpdate` per Dask worker
    per poll interval. ``gpu_percent`` is always ``None`` — Dask does not expose
    GPU utilization via ``scheduler_info()``. ``gpu_id`` is always ``None``
    (node-level aggregate semantics, OTel-compatible).

    Disk I/O bytes are cumulative in Dask's metrics dict; this adapter converts
    them to per-interval deltas, consistent with the Dragon and Concurrent adapters.

    Args:
        client:       An active Dask ``distributed.Client`` instance.
        session_id:   Session identifier for emitted events.
        backend_name: Backend name for emitted events.
        interval:     Polling interval in seconds (default: 10.0).
    """

    def __init__(
        self,
        client: Any,
        session_id: str,
        backend_name: str,
        interval: float = 10.0,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._backend_name = backend_name
        self._interval = interval
        self._manager: TelemetryManager | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        # Per-worker disk I/O baseline for delta calculations: addr → (read, write)
        self._prev_disk: dict[str, tuple[float, float]] = {}

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._collect_loop(), name="telemetry-dask-adapter")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _collect_loop(self) -> None:
        async def _poll() -> None:
            try:
                workers = self._client.scheduler_info().get("workers", {})
                for addr, w in workers.items():
                    m = w.get("metrics", {})
                    mem_limit = w.get("memory_limit", 1) or 1
                    cpu = m.get("cpu", 0.0)
                    mem_pct = (m.get("memory", 0) / mem_limit) * 100.0
                    node_id = w.get("host", addr)

                    # Disk I/O — cumulative counters → per-interval deltas
                    raw_read = float(m.get("read_bytes", 0) or 0)
                    raw_write = float(m.get("write_bytes", 0) or 0)
                    prev = self._prev_disk.get(addr)
                    if prev is not None:
                        disk_read = max(raw_read - prev[0], 0.0)
                        disk_write = max(raw_write - prev[1], 0.0)
                    else:
                        disk_read = disk_write = None  # first poll — no baseline yet
                    self._prev_disk[addr] = (raw_read, raw_write)

                    self._manager.emit(
                        make_event(
                            ResourceUpdate,
                            session_id=self._session_id,
                            backend=self._backend_name,
                            node_id=node_id,
                            cpu_percent=cpu,
                            memory_percent=mem_pct,
                            gpu_percent=None,  # Dask does not expose GPU via scheduler_info
                            gpu_id=None,  # node-level aggregate — OTel-compatible
                            disk_read_bytes=disk_read,
                            disk_write_bytes=disk_write,
                            net_sent_bytes=None,  # not exposed by Dask scheduler_info
                            net_recv_bytes=None,
                        )
                    )
            except Exception:
                logger.debug("DaskTelemetryAdapter collection error", exc_info=True)

        # Emit one sample immediately so resource coverage starts at t=0,
        # not one full interval later.
        await _poll()

        while self._running:
            await asyncio.sleep(self._interval)
            await _poll()
