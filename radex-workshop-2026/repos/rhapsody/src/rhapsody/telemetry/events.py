"""RHAPSODY telemetry events.

All telemetry originates from these normalized events. Frozen dataclasses ensure
immutability and safe zero-copy fan-out to multiple subscribers.

Canonical lifecycle:
    (TaskCreated) → TaskSubmitted → (TaskQueued) → TaskStarted → TaskCompleted | TaskFailed | TaskCanceled

TaskCreated is emitted by Session.submit_tasks() when the future is created.
TaskQueued is optional. TaskStarted is the authoritative "execution began" event.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass
from dataclasses import field


def _new_event_id() -> str:
    return uuid.uuid4().hex


def _now_wall() -> float:
    return time.time()


@dataclass(frozen=True)
class BaseEvent:
    """Base class for all RHAPSODY telemetry events.

    Attributes:
        event_id:   Unique event identifier (uuid4 hex, no hyphens).
        event_type: Discriminator string matching the subclass name.
        event_time: Wall-clock time when the event actually occurred (from task
                    history where available; otherwise time of emission).
        emit_time:  Wall-clock time at the point of emission into the event bus.
                    Both event_time and emit_time use time.time() — their
                    difference is the queue latency in seconds.
        session_id: Owning session identifier.
        backend:    Name of the execution backend that produced the event.
        task_id:    Task identifier, None for session-level events.
        node_id:    Node/worker identifier, None when not applicable.
        attributes: Optional metadata dict. Carries task-context fields such as
                    executable, task_type, error_type (for lifecycle events) and
                    incomplete_lifecycle when STARTED was not recorded.
    """

    event_id: str
    event_type: str
    event_time: float
    emit_time: float
    session_id: str
    backend: str
    task_id: str | None = None
    node_id: str | None = None
    attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SessionStarted(BaseEvent):
    """Emitted when a session begins (telemetry enabled and started)."""

    event_type: str = field(default="SessionStarted", init=False)


@dataclass(frozen=True)
class SessionEnded(BaseEvent):
    """Emitted when a session ends (telemetry stopped)."""

    duration_seconds: float = 0.0
    event_type: str = field(default="SessionEnded", init=False)


@dataclass(frozen=True)
class TaskCreated(BaseEvent):
    """Emitted when a task object is created and registered, before backend submission.

    Used by DAG-aware runtimes (e.g. AsyncFlow) where a future is created at DAG
    node definition time — before dependencies are resolved or the task is submitted.
    Also useful in Session contexts for tracking the registration-to-submission latency.

    attributes keys: executable, task_type, dag_dependencies (DAG contexts only)
    """

    event_type: str = field(default="TaskCreated", init=False)


@dataclass(frozen=True)
class TaskSubmitted(BaseEvent):
    """Emitted when a task is submitted to the session.

    attributes keys: executable, task_type
    """

    event_type: str = field(default="TaskSubmitted", init=False)


@dataclass(frozen=True)
class TaskQueued(BaseEvent):
    """Emitted when a task is handed off to a backend's execution queue.

    Optional event. Marks the boundary between session-level submission and
    backend-level scheduling. Gap between TaskSubmitted.event_time and
    TaskQueued.event_time is the session routing latency.

    attributes keys: executable, task_type
    """

    event_type: str = field(default="TaskQueued", init=False)


@dataclass(frozen=True)
class TaskStarted(BaseEvent):
    """Emitted when a task transitions to RUNNING state (execution physically begins).

    attributes keys: executable, task_type
    """

    event_type: str = field(default="TaskStarted", init=False)


@dataclass(frozen=True)
class TaskCompleted(BaseEvent):
    """Emitted when a task reaches DONE state.

    attributes keys: executable, task_type
    If incomplete_lifecycle=True in attributes, duration_seconds is 0.0
    because the STARTED timestamp was not recorded.
    """

    duration_seconds: float = 0.0
    event_type: str = field(default="TaskCompleted", init=False)


@dataclass(frozen=True)
class TaskFailed(BaseEvent):
    """Emitted when a task reaches FAILED state.

    attributes keys: executable, task_type, error_type
    If incomplete_lifecycle=True in attributes, duration_seconds is 0.0
    because the STARTED timestamp was not recorded.
    """

    duration_seconds: float = 0.0
    error_type: str = "unknown"
    event_type: str = field(default="TaskFailed", init=False)


@dataclass(frozen=True)
class TaskCanceled(BaseEvent):
    """Emitted when a task reaches CANCELED state.

    Cancellation is intentional (user-initiated or backend-level) — distinct from TaskFailed, which
    represents an unexpected error. No error_type field.

    duration_seconds is 0.0 when RUNNING was never recorded (task canceled before execution
    started); incomplete_lifecycle=True is set in attributes in that case.
    """

    duration_seconds: float = 0.0
    event_type: str = field(default="TaskCanceled", init=False)


@dataclass(frozen=True)
class ResourceUpdate(BaseEvent):
    """Emitted periodically by backend adapters with node resource utilization.

    Two scopes are distinguished by ``resource_scope`` (computed automatically from ``gpu_id``):

    ``"per_node"`` (``gpu_id=None``)
        Node-level aggregate: cpu_percent, memory_percent, disk/net deltas, and the
        aggregate gpu_percent (max across devices) are all populated.

    ``"per_gpu"`` (``gpu_id=N``)
        Per-device GPU event: only gpu_percent and gpu_id are set; cpu_percent,
        memory_percent, disk/net bytes are all None.  Consumers that need CPU/RAM
        for a GPU node should join on (session_id, node_id, event_time) to the
        corresponding per_node event.

    cpu/memory values are point-in-time percentages (not deltas).
    disk/net values are byte deltas since the previous poll.
    """

    cpu_percent: float | None = None
    memory_percent: float | None = None
    gpu_percent: float | None = None
    gpu_id: int | None = None
    disk_read_bytes: float | None = None
    disk_write_bytes: float | None = None
    net_sent_bytes: float | None = None
    net_recv_bytes: float | None = None
    event_type: str = field(default="ResourceUpdate", init=False)
    resource_scope: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_scope",
            "per_gpu" if self.gpu_id is not None else "per_node",
        )


def define_event(name: str, **fields) -> type:
    """Dynamically create a custom BaseEvent subclass — no RHAPSODY edits required.

    Designed for higher-layer runtimes (AsyncFlow, user applications) that need
    domain-specific events without modifying RHAPSODY's event schema.

    Rules
    -----
    ``name`` **must be namespaced** (contain at least one ``"."``) to prevent
    collisions across systems::

        # bad  — rejected
        define_event("DataQualityChecked", ...)

        # good — accepted
        define_event("asyncflow.DataQualityChecked", ...)

    Field names **must not shadow** any core :class:`BaseEvent` field
    (``event_id``, ``event_time``, ``emit_time``, ``session_id``, ``backend``,
    ``task_id``, ``node_id``, ``attributes``, ``event_type``).
    Doing so raises :exc:`ValueError` at definition time.

    Field values can be:
      - A type alone (``float``, ``int``, ``str``, ``bool``): a zero/empty default
        is inferred.
      - A ``(type, default)`` tuple: explicit default value.

    The returned class is a proper frozen dataclass that inherits ``BaseEvent``.
    It is compatible with :func:`make_event` and any
    :class:`~rhapsody.telemetry.manager.TelemetryManager`.

    Example::

        DataQualityChecked = define_event(
            "asyncflow.DataQualityChecked",
            score=float,
            rows_checked=int,
            label=(str, "unknown"),
        )

        telemetry.emit(
            make_event(
                DataQualityChecked,
                session_id=telemetry._session_id,
                backend="application",
                score=0.97,
                rows_checked=1000,
            )
        )
    """
    if "." not in name:
        raise ValueError(
            f"define_event: name {name!r} must be namespaced (e.g. 'asyncflow.{name}'). "
            "Flat names risk collisions across systems."
        )

    base_fields = frozenset(f.name for f in dataclasses.fields(BaseEvent))
    reserved = base_fields.intersection(fields)
    if reserved:
        raise ValueError(
            f"define_event: fields {sorted(reserved)} shadow core BaseEvent fields. "
            "Custom events must not redefine the base schema."
        )

    primitive_defaults = {float: 0.0, int: 0, str: "", bool: False}

    field_specs = []
    for fname, spec in fields.items():
        if isinstance(spec, tuple):
            ftype, default = spec
            field_specs.append((fname, ftype, dataclasses.field(default=default)))
        else:
            ftype = spec
            default = primitive_defaults.get(ftype)  # None for unknown types
            field_specs.append((fname, ftype, dataclasses.field(default=default)))

    # Override event_type — init=False, mirrors every concrete BaseEvent subclass.
    field_specs.append(("event_type", str, dataclasses.field(default=name, init=False)))

    return dataclasses.make_dataclass(name, fields=field_specs, bases=(BaseEvent,), frozen=True)


def make_event(cls: type, *, session_id: str, backend: str, **kwargs) -> BaseEvent:
    """Construct any event with auto-generated id and consistent wall-clock timestamps.

    Both event_time and emit_time use time.time() so their difference is interpretable as queue
    latency in seconds (emit_time >= event_time always holds when event_time is taken from a recent
    task history entry).
    """
    now = time.time()
    return cls(
        event_id=_new_event_id(),
        event_time=kwargs.pop("event_time", now),
        emit_time=now,
        session_id=session_id,
        backend=backend,
        **kwargs,
    )
