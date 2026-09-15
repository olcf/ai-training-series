"""Shared fixtures and contract helpers for telemetry adapter tests."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from rhapsody.telemetry.events import ResourceUpdate


@dataclass
class AdapterCapabilities:
    """Declares what a telemetry adapter is expected to emit.

    Fields that are False/None mean the adapter either cannot collect that metric (e.g. Dask has no
    net I/O) or the hardware is absent.  The contract assertions use these flags to distinguish
    intentional gaps from bugs.
    """

    name: str
    # Node identification
    has_node_id: bool = True
    # GPU
    has_gpu_aggregate: bool = False  # emits node-level gpu_percent (gpu_id=None)
    has_gpu_per_device: bool = False  # emits per-device events (gpu_id=N)
    # I/O
    has_disk_io: bool = True
    has_net_io: bool = True
    # Extra fields expected on all events
    required_fields: list[str] = field(
        default_factory=lambda: [
            "event_id",
            "event_type",
            "event_time",
            "emit_time",
            "session_id",
            "backend",
        ]
    )


def assert_resource_update_contract(event: Any, caps: AdapterCapabilities) -> None:
    """Assert that a ResourceUpdate event satisfies the OTel-aligned contract.

    Hard failures (apply to ALL adapters regardless of capabilities):
      - Required fields present and non-None
      - event_type is "ResourceUpdate"
      - cpu_percent and memory_percent are floats

    Capability-gated failures (only checked when the adapter declared support):
      - node_id non-empty when has_node_id=True
      - gpu_percent non-None on node-level events when has_gpu_aggregate=True
      - gpu_id=N events only emitted when has_gpu_per_device=True
      - Per-device events must have disk/net = None (node-level only)
      - disk/net None when adapter declared no I/O capability
    """
    assert isinstance(event, ResourceUpdate), (
        f"{caps.name}: expected ResourceUpdate, got {type(event).__name__}"
    )

    # ── Hard constraints ──────────────────────────────────────────────────
    for f in caps.required_fields:
        assert getattr(event, f, None) is not None, (
            f"{caps.name}: required field '{f}' is None or missing on {event}"
        )

    assert event.event_type == "ResourceUpdate", (
        f"{caps.name}: event_type must be 'ResourceUpdate', got '{event.event_type}'"
    )

    # resource_scope must be set and consistent with gpu_id
    assert event.resource_scope in ("per_node", "per_gpu"), (
        f"{caps.name}: resource_scope must be 'per_node' or 'per_gpu', got {event.resource_scope!r}"
    )
    if event.resource_scope == "per_gpu":
        assert event.gpu_id is not None, f"{caps.name}: resource_scope='per_gpu' but gpu_id is None"
        assert event.cpu_percent is None, (
            f"{caps.name}: resource_scope='per_gpu' must have cpu_percent=None"
        )
        assert event.memory_percent is None, (
            f"{caps.name}: resource_scope='per_gpu' must have memory_percent=None"
        )
    else:
        assert event.gpu_id is None, (
            f"{caps.name}: resource_scope='per_node' but gpu_id={event.gpu_id}"
        )
        assert isinstance(event.cpu_percent, float), (
            f"{caps.name}: cpu_percent must be float on per_node events, got {type(event.cpu_percent)}"
        )
        assert 0.0 <= event.cpu_percent <= 100.0, (
            f"{caps.name}: cpu_percent out of range: {event.cpu_percent}"
        )
        assert isinstance(event.memory_percent, float), (
            f"{caps.name}: memory_percent must be float on per_node events, got {type(event.memory_percent)}"
        )
        assert 0.0 <= event.memory_percent <= 100.0, (
            f"{caps.name}: memory_percent out of range: {event.memory_percent}"
        )

    # ── Node id ───────────────────────────────────────────────────────────
    if caps.has_node_id:
        assert event.node_id is not None and event.node_id != "", (
            f"{caps.name}: node_id must be a non-empty string, got {event.node_id!r}"
        )

    # ── GPU fields ────────────────────────────────────────────────────────
    if event.gpu_id is None:
        # Node-level aggregate event
        if not caps.has_gpu_aggregate:
            assert event.gpu_percent is None, (
                f"{caps.name}: adapter declared no GPU support but gpu_percent={event.gpu_percent}"
            )
        else:
            assert isinstance(event.gpu_percent, float), (
                f"{caps.name}: gpu_percent must be float on gpu-capable adapter, got {event.gpu_percent}"
            )
    else:
        # Per-device event — adapter must have declared per-device GPU support
        assert caps.has_gpu_per_device, (
            f"{caps.name}: emitted gpu_id={event.gpu_id} but declared has_gpu_per_device=False"
        )
        assert isinstance(event.gpu_id, int) and event.gpu_id >= 0, (
            f"{caps.name}: gpu_id must be a non-negative int, got {event.gpu_id}"
        )
        assert isinstance(event.gpu_percent, float), (
            f"{caps.name}: per-device event must have float gpu_percent, got {event.gpu_percent}"
        )
        # Per-device events must NOT carry node-level I/O (those belong to the aggregate event)
        assert event.disk_read_bytes is None, (
            f"{caps.name}: per-device event (gpu_id={event.gpu_id}) must have disk_read_bytes=None"
        )
        assert event.disk_write_bytes is None, (
            f"{caps.name}: per-device event (gpu_id={event.gpu_id}) must have disk_write_bytes=None"
        )
        assert event.net_sent_bytes is None, (
            f"{caps.name}: per-device event (gpu_id={event.gpu_id}) must have net_sent_bytes=None"
        )
        assert event.net_recv_bytes is None, (
            f"{caps.name}: per-device event (gpu_id={event.gpu_id}) must have net_recv_bytes=None"
        )

    # ── I/O fields (node-level events only) ──────────────────────────────
    if event.gpu_id is None:
        if not caps.has_disk_io:
            assert event.disk_read_bytes is None, (
                f"{caps.name}: declared no disk I/O but disk_read_bytes={event.disk_read_bytes}"
            )
            assert event.disk_write_bytes is None, (
                f"{caps.name}: declared no disk I/O but disk_write_bytes={event.disk_write_bytes}"
            )
        if not caps.has_net_io:
            assert event.net_sent_bytes is None, (
                f"{caps.name}: declared no net I/O but net_sent_bytes={event.net_sent_bytes}"
            )
            assert event.net_recv_bytes is None, (
                f"{caps.name}: declared no net I/O but net_recv_bytes={event.net_recv_bytes}"
            )
