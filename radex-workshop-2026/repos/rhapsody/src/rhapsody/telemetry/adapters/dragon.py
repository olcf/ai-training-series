"""Telemetry adapter for DragonExecutionBackend.

Architecture
------------
Spawns one lightweight worker process per Dragon node using Dragon's own
ProcessGroup + Policy(HOST_NAME) infrastructure.  Each worker imports metric
collection functions directly from ``dragon.telemetry.collector`` (which
triggers the module-level ``find_accelerators()`` call and the conditional
``pynvml`` / ``rocm_smi`` imports — identical to what Dragon's own Collector
does).  The worker collects CPU%, RAM%, and GPU% every ``interval`` seconds and
puts a ``{host, dps, ts}`` dict on a shared Dragon Queue.

A head-node daemon thread reads the queue, aggregates the ``dps`` list, and
emits one ``ResourceUpdate`` event per node per interval.

Data flow
---------
  ProcessGroup on each node
    └─ _rhapsody_telemetry_worker()
         imports from dragon.telemetry.collector:
           identify_gpu()  get_nvidia_metrics()  get_amd_metrics()  get_intel_metrics()
         psutil for CPU% and RAM%
         result_queue.put({"host": ..., "dps": [...], "ts": ...})

  Head-node daemon thread
    └─ result_queue.get() → aggregate dps → emit ResourceUpdate per node

Requirements
------------
- DRAGON_TELEMETRY_LEVEL >= 1:  CPU/RAM collected on all nodes
- DRAGON_TELEMETRY_LEVEL >= 1:  GPU collected if hardware present
  (the level guard is only at the adapter entry; the worker always collects
  whatever hardware is available)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue as stdlib_queue
import threading
import time
from typing import TYPE_CHECKING

from rhapsody.telemetry.adapters.base import TelemetryAdapter
from rhapsody.telemetry.events import ResourceUpdate
from rhapsody.telemetry.events import make_event

if TYPE_CHECKING:
    from rhapsody.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-node worker (top-level so cloudpickle can serialize it)
# ---------------------------------------------------------------------------


def _rhapsody_telemetry_worker(
    result_queue,
    shutdown_event,
    interval: float,
) -> None:
    """Collect CPU/RAM/GPU on one Dragon node and push dps to a Dragon Queue.

    Runs inside a Dragon ProcessGroup subprocess on the target node.
    Uses os.uname().nodename to label results — matches Dragon's own node
    identification, avoiding FQDN vs short-name mismatches on clusters.

    The ``shutdown_event`` (dragon.native.event.Event) is set by the head node
    when the adapter stops; the worker then exits cleanly with code 0, avoiding
    the DragonUserCodeError that SIGINT-based stop() would cause.
    """
    import logging as _logging  # noqa: PLC0415
    import os as _os  # noqa: PLC0415
    import time as _time  # noqa: PLC0415

    import psutil  # noqa: PLC0415

    hostname = _os.uname().nodename
    _log = _logging.getLogger(__name__)

    # Import GPU helpers from Dragon's collector.
    # FIXME: We should not do that, but due to a bug in Dragon where if we
    # enable telemetry everything hangs (specifically adding the AnalysisClient).
    # Once it is fixed we will rely on Dragon telemetry and AnalysisClient
    # to get the data from each node
    try:
        from dragon.infrastructure.gpu_desc import AccVendor  # noqa: PLC0415
        from dragon.telemetry.collector import get_amd_metrics  # noqa: PLC0415
        from dragon.telemetry.collector import get_intel_metrics  # noqa: PLC0415
        from dragon.telemetry.collector import get_nvidia_metrics  # noqa: PLC0415
        from dragon.telemetry.collector import identify_gpu  # noqa: PLC0415

        _collector_available = True
    except Exception:
        _collector_available = False

    gpu_vendor = None
    gpu_count = 0
    if _collector_available:
        try:
            gpu_vendor, gpu_count = identify_gpu()
        except Exception:
            gpu_vendor, gpu_count = None, 0

    # Per-device failure tracking: device index → logged once, then skipped forever
    _gpu_failed: set = set()
    # Intel: if the single bulk call fails, disable the whole Intel branch
    _intel_disabled = False

    # Seed disk / net baselines for delta calculations (bytes-per-interval)
    _prev_disk: tuple = (0, 0)
    _prev_net: tuple = (0, 0)
    try:
        dc = psutil.disk_io_counters()
        if dc:
            _prev_disk = (dc.read_bytes, dc.write_bytes)
    except Exception:  # noqa: S110
        pass
    try:
        nc = psutil.net_io_counters()
        if nc:
            _prev_net = (nc.bytes_sent, nc.bytes_recv)
    except Exception:  # noqa: S110
        pass

    def _collect_and_push() -> None:
        nonlocal _prev_disk, _prev_net, _intel_disabled
        dps = []

        # CPU and RAM — always collected; psutil does not raise here
        dps.append({"metric": "cpu_percent", "value": psutil.cpu_percent()})
        dps.append({"metric": "used_RAM", "value": psutil.virtual_memory().percent})

        # Disk I/O — bytes transferred since last interval
        try:
            dc = psutil.disk_io_counters()
            if dc:
                dps.append(
                    {"metric": "disk_read_bytes", "value": float(dc.read_bytes - _prev_disk[0])}
                )
                dps.append(
                    {"metric": "disk_write_bytes", "value": float(dc.write_bytes - _prev_disk[1])}
                )
                _prev_disk = (dc.read_bytes, dc.write_bytes)
        except Exception:  # noqa: S110
            pass

        # Network I/O — bytes transferred since last interval
        try:
            nc = psutil.net_io_counters()
            if nc:
                dps.append(
                    {"metric": "net_sent_bytes", "value": float(nc.bytes_sent - _prev_net[0])}
                )
                dps.append(
                    {"metric": "net_recv_bytes", "value": float(nc.bytes_recv - _prev_net[1])}
                )
                _prev_net = (nc.bytes_sent, nc.bytes_recv)
        except Exception:  # noqa: S110
            pass

        # GPU — vendor-dispatched; per-device failure logged once, then skipped
        if _collector_available and gpu_vendor is not None:
            try:
                if gpu_vendor == AccVendor.NVIDIA:
                    for i in range(gpu_count):
                        if i in _gpu_failed:
                            continue
                        try:
                            metrics = get_nvidia_metrics(i, telemetry_level=3)
                            if metrics:
                                for m in metrics:
                                    dps.append({**m, "device_id": i})
                        except Exception as _exc:
                            _gpu_failed.add(i)
                            _log.warning(
                                "GPU %d metrics unavailable on %s (%s) — disabling",
                                i,
                                hostname,
                                _exc,
                            )
                elif gpu_vendor == AccVendor.AMD:
                    for idx, device in enumerate(gpu_count):  # listDevices() returns a list
                        if device in _gpu_failed:
                            continue
                        try:
                            metrics = get_amd_metrics(device, telemetry_level=3)
                            if metrics:
                                for m in metrics:
                                    dps.append({**m, "device_id": idx})
                        except Exception as _exc:
                            _gpu_failed.add(device)
                            _log.warning(
                                "GPU %s metrics unavailable on %s (%s) — disabling",
                                device,
                                hostname,
                                _exc,
                            )
                elif gpu_vendor == AccVendor.INTEL and not _intel_disabled:
                    try:
                        all_metrics = get_intel_metrics(telemetry_level=3)
                        for dev_idx, metrics_list in enumerate(all_metrics.values()):
                            for m in metrics_list:
                                dps.append({**m, "device_id": dev_idx})
                    except Exception as _exc:
                        _intel_disabled = True
                        _log.warning(
                            "Intel GPU metrics unavailable on %s (%s) — disabling",
                            hostname,
                            _exc,
                        )
            except Exception:  # noqa: S110
                pass  # outer guard: GPU block failure never silences CPU/RAM

        try:
            result_queue.put(
                {"host": hostname, "dps": dps, "ts": int(_time.time())},
                timeout=5.0,
            )
        except Exception:  # noqa: S110
            pass  # queue full or closed — drop silently

    # Emit one sample immediately so resource coverage starts at t=0,
    # not one full interval later.
    _collect_and_push()

    # shutdown_event.wait(timeout) blocks for interval seconds, returns True when set.
    # Replaces time.sleep() so the worker exits cleanly (code 0) on adapter stop.
    while not shutdown_event.wait(timeout=interval):
        _collect_and_push()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DragonTelemetryAdapter(TelemetryAdapter):
    """Collects node-level resource metrics for Dragon backends.

    Spawns one subprocess per Dragon node (using Dragon's own ProcessGroup +
    Policy infrastructure) that collects CPU%, RAM%, and GPU% by calling
    Dragon's own metric functions from ``dragon.telemetry.collector``.
    Results are delivered via a Dragon native Queue; the head-node daemon
    thread reads the queue and emits one ``ResourceUpdate`` per node per
    interval.

    When Dragon telemetry is not active (``DRAGON_TELEMETRY_LEVEL < 1``) or
    the Dragon runtime is unavailable, the adapter is a no-op.

    Args:
        session_id:   Session identifier for emitted events.
        backend_name: Backend name for emitted events.
        interval:     Collection interval in seconds (default: 10.0).
    """

    def __init__(
        self,
        session_id: str,
        backend_name: str,
        interval: float = 10.0,
    ) -> None:
        self._session_id = session_id
        self._backend_name = backend_name
        self._interval = interval
        self._manager: TelemetryManager | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._node_id = os.uname().nodename
        self._group = None  # Dragon ProcessGroup — set in _collect_loop, used in stop()

    def start(self) -> None:
        # Guard: Dragon must be importable.
        try:
            import dragon.telemetry  # noqa: F401, PLC0415
        except ImportError:
            logger.debug("Dragon not available — DragonTelemetryAdapter is a no-op")
            return

        self._running = True

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

        self._thread = threading.Thread(
            target=self._collect_loop,
            daemon=True,
            name="telemetry-dragon-adapter",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # ProcessGroup cleanup happens inside _collect_loop() after the while loop exits.

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_loop(self) -> None:
        """Daemon thread: spawns one worker per Dragon node, reads ResourceUpdates."""

        # 1. Discover Dragon nodes + get one Policy per node
        try:
            from dragon.native.machine import System  # noqa: PLC0415

            policies = System().hostname_policies()
        except Exception:
            logger.warning(
                "DragonTelemetryAdapter: node discovery failed — adapter is a no-op",
                exc_info=True,
            )
            return

        # 2. Create shared Dragon result queue + cooperative shutdown event
        try:
            from dragon.native.event import Event as DragonEvent  # noqa: PLC0415
            from dragon.native.queue import Queue as DragonQueue  # noqa: PLC0415

            result_queue = DragonQueue(maxsize=max(len(policies) * 40, 400))
            _shutdown_event = DragonEvent()
        except Exception:
            logger.warning(
                "DragonTelemetryAdapter: Dragon Queue/Event creation failed — adapter is a no-op",
                exc_info=True,
            )
            return

        # 3. Spawn one _rhapsody_telemetry_worker per node
        try:
            from dragon.native.process import ProcessTemplate  # noqa: PLC0415
            from dragon.native.process_group import ProcessGroup  # noqa: PLC0415

            grp = ProcessGroup(restart=False, pmi=None)
            for policy in policies:
                grp.add_process(
                    nproc=1,
                    template=ProcessTemplate(
                        target=_rhapsody_telemetry_worker,
                        args=(result_queue, _shutdown_event, self._interval),
                        policy=policy,
                    ),
                )
            grp.init()
            grp.start()
            self._group = grp
            logger.debug(f"DragonTelemetryAdapter: workers running on {len(policies)} nodes")
        except Exception:
            logger.warning(
                "DragonTelemetryAdapter: ProcessGroup spawn failed — adapter is a no-op",
                exc_info=True,
            )
            return

        # 4. Read loop — emit ResourceUpdate at most once per interval per node
        last_emit: dict = {}  # host → last emit wall-clock time

        while self._running:
            try:
                data = result_queue.get(timeout=self._interval)
            except stdlib_queue.Empty:
                continue
            except Exception:
                logger.debug("DragonTelemetryAdapter: queue read error", exc_info=True)
                continue

            host = data.get("host", self._node_id)
            now = time.time()

            # Throttle: emit at most once per interval per node
            if now - last_emit.get(host, 0.0) < self._interval:
                continue
            last_emit[host] = now

            # Separate node-level metrics from per-device GPU metrics
            node_vals: dict = {}
            per_device: dict[int, dict] = {}  # device_id → {metric: value}
            for dp in data.get("dps", []):
                metric = dp.get("metric")
                value = dp.get("value")
                dev_id = dp.get("device_id")  # None for cpu/ram/disk/net
                if value is None or metric is None:
                    continue
                if dev_id is not None:
                    per_device.setdefault(dev_id, {})[metric] = float(value)
                elif metric not in node_vals:
                    node_vals[metric] = float(value)

            # GPU aggregate (max across devices) for the node-level event
            if per_device:
                node_vals["DeviceUtilization"] = max(
                    d.get("DeviceUtilization", 0.0) for d in per_device.values()
                )

            if self._loop and self._loop.is_running():
                # Node-level aggregate event (gpu_id=None, backward-compatible)
                node_event = make_event(
                    ResourceUpdate,
                    session_id=self._session_id,
                    backend=self._backend_name,
                    node_id=host,
                    cpu_percent=node_vals.get("cpu_percent"),
                    memory_percent=node_vals.get("used_RAM"),
                    gpu_percent=node_vals.get("DeviceUtilization"),
                    disk_read_bytes=node_vals.get("disk_read_bytes"),
                    disk_write_bytes=node_vals.get("disk_write_bytes"),
                    net_sent_bytes=node_vals.get("net_sent_bytes"),
                    net_recv_bytes=node_vals.get("net_recv_bytes"),
                )
                self._loop.call_soon_threadsafe(self._manager.emit, node_event)

                # Per-GPU events — one per device (cpu/mem/disk/net are per_node scope only)
                for dev_id, dvals in per_device.items():
                    gpu_event = make_event(
                        ResourceUpdate,
                        session_id=self._session_id,
                        backend=self._backend_name,
                        node_id=host,
                        cpu_percent=None,
                        memory_percent=None,
                        gpu_percent=dvals.get("DeviceUtilization"),
                        gpu_id=dev_id,
                        disk_read_bytes=None,
                        disk_write_bytes=None,
                        net_sent_bytes=None,
                        net_recv_bytes=None,
                    )
                    self._loop.call_soon_threadsafe(self._manager.emit, gpu_event)

        # 5. Cooperative shutdown: set event so workers exit their while loop cleanly
        #    (exit code 0), then join. This avoids DragonUserCodeError that grp.stop()
        #    would cause (SIGINT → exit code -2 → Dragon raises on join).
        try:
            _shutdown_event.set()
            grp.join(timeout=30.0)
            grp.close()
        except Exception:
            logger.debug("DragonTelemetryAdapter: ProcessGroup cleanup error", exc_info=True)
