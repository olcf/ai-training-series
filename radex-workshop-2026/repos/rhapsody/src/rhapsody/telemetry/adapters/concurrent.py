"""Telemetry adapter for ConcurrentExecutionBackend.

Uses psutil for CPU, memory, disk I/O, and network I/O metrics on the local node.
GPU utilization is collected via nvidia-smi if available; falls back to None gracefully.

The concurrent backend runs on a single local node with no native telemetry
infrastructure, so psutil is the only correct option here.

The collection loop runs in a **daemon thread** (not an asyncio task) so it is
immune to any event-loop interference from the Dragon runtime when RHAPSODY is
launched via `dragon -T N`.  Events are posted back to the asyncio loop via
`loop.call_soon_threadsafe`, keeping the emit path thread-safe.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from typing import TYPE_CHECKING

from rhapsody.telemetry.adapters.base import TelemetryAdapter
from rhapsody.telemetry.events import ResourceUpdate
from rhapsody.telemetry.events import make_event

if TYPE_CHECKING:
    from rhapsody.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


class ConcurrentTelemetryAdapter(TelemetryAdapter):
    """Collects node-level resource metrics for the concurrent backend via psutil.

    Runs a daemon thread that wakes up every *interval* seconds, reads CPU /
    memory / disk / network counters via psutil, then posts a ResourceUpdate event
    back to the asyncio event loop via ``loop.call_soon_threadsafe``.  Using a
    thread (rather than an asyncio task) makes the adapter robust under Dragon's
    runtime, which can interfere with ``asyncio.sleep`` inside tasks.

    Args:
        session_id:   Session identifier to attach to emitted events.
        backend_name: Backend name to attach to emitted events.
        interval:     Polling interval in seconds (default: 5.0).
    """

    def __init__(
        self,
        session_id: str,
        backend_name: str,
        interval: float = 5.0,
    ) -> None:
        self._session_id = session_id
        self._backend_name = backend_name
        self._interval = interval
        self._manager: TelemetryManager | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._node_id = socket.gethostname()

    def start(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        self._running = True
        self._thread = threading.Thread(
            target=self._collect_loop,
            name="telemetry-concurrent-adapter",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # Thread is a daemon — it will exit when the process exits; no join needed.

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_loop(self) -> None:
        """Blocking loop that runs in the daemon thread."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available — ConcurrentTelemetryAdapter is a no-op")
            return

        # Warm up cpu_percent (first call always returns 0.0 on Linux)
        psutil.cpu_percent(interval=None)

        # GPU setup via pynvml — initialized once, not on every poll
        _nvml_ok = False
        _gpu_count = 0
        _gpu_failed: set[int] = set()
        try:
            import pynvml

            pynvml.nvmlInit()
            _gpu_count = pynvml.nvmlDeviceGetCount()
            _nvml_ok = True
        except ImportError:
            pass  # pynvml not installed — GPU metrics skipped silently
        except Exception as _exc:
            logger.warning(
                "pynvml initialization failed on %s — GPU metrics disabled: %s",
                self._node_id,
                _exc,
            )

        # Seed disk / net baselines for delta calculations
        prev_disk: tuple | None = None
        prev_net: tuple | None = None
        try:
            dc = psutil.disk_io_counters()
            prev_disk = (dc.read_bytes, dc.write_bytes) if dc else None
        except Exception:  # noqa: S110
            pass
        try:
            nc = psutil.net_io_counters()
            prev_net = (nc.bytes_sent, nc.bytes_recv) if nc else None
        except Exception:  # noqa: S110
            pass

        def _poll() -> None:
            """Collect one sample and emit ResourceUpdate events."""
            nonlocal prev_disk, prev_net
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent

                # Per-device GPU utilization via pynvml
                per_device_gpu: dict[int, float] = {}
                if _nvml_ok:
                    for dev_idx in range(_gpu_count):
                        if dev_idx in _gpu_failed:
                            continue
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            per_device_gpu[dev_idx] = float(util.gpu)
                        except Exception as _exc:
                            _gpu_failed.add(dev_idx)
                            logger.warning(
                                "GPU %d unavailable on %s (%s) — disabling",
                                dev_idx,
                                self._node_id,
                                _exc,
                            )
                gpu_aggregate = max(per_device_gpu.values()) if per_device_gpu else None

                disk_read = disk_write = net_sent = net_recv = None

                try:
                    dc = psutil.disk_io_counters()
                    if dc and prev_disk is not None:
                        disk_read = float(dc.read_bytes - prev_disk[0])
                        disk_write = float(dc.write_bytes - prev_disk[1])
                    prev_disk = (dc.read_bytes, dc.write_bytes) if dc else None
                except Exception:  # noqa: S110
                    pass

                try:
                    nc = psutil.net_io_counters()
                    if nc and prev_net is not None:
                        net_sent = float(nc.bytes_sent - prev_net[0])
                        net_recv = float(nc.bytes_recv - prev_net[1])
                    prev_net = (nc.bytes_sent, nc.bytes_recv) if nc else None
                except Exception:  # noqa: S110
                    pass

                if self._loop and self._loop.is_running():
                    # Node-level aggregate event (gpu_id=None — backward-compatible)
                    node_event = make_event(
                        ResourceUpdate,
                        session_id=self._session_id,
                        backend=self._backend_name,
                        node_id=self._node_id,
                        cpu_percent=cpu,
                        memory_percent=mem,
                        gpu_percent=gpu_aggregate,
                        disk_read_bytes=disk_read,
                        disk_write_bytes=disk_write,
                        net_sent_bytes=net_sent,
                        net_recv_bytes=net_recv,
                    )
                    self._loop.call_soon_threadsafe(self._manager.emit, node_event)

                    # Per-device GPU events (cpu/mem/disk/net are per_node scope only)
                    for dev_idx, gpu_pct in per_device_gpu.items():
                        gpu_event = make_event(
                            ResourceUpdate,
                            session_id=self._session_id,
                            backend=self._backend_name,
                            node_id=self._node_id,
                            cpu_percent=None,
                            memory_percent=None,
                            gpu_percent=gpu_pct,
                            gpu_id=dev_idx,
                            disk_read_bytes=None,
                            disk_write_bytes=None,
                            net_sent_bytes=None,
                            net_recv_bytes=None,
                        )
                        self._loop.call_soon_threadsafe(self._manager.emit, gpu_event)

            except Exception:
                logger.debug("ConcurrentTelemetryAdapter collection error", exc_info=True)

        # Emit one sample immediately so resource coverage starts at t=0,
        # not one full interval later. Disk/net deltas are None on this first
        # call (baselines were just seeded above) which is correct.
        _poll()

        while self._running:
            time.sleep(self._interval)  # plain OS sleep — no event loop involved
            if not self._running:
                break
            _poll()
