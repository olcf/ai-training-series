"""RHAPSODY TelemetryManager.

Central orchestrator: owns the asyncio EventBus (Queue), wires OpenTelemetry SDK
instruments and tracers, dispatches events to metrics, traces, and external subscribers,
and writes a single JSONL checkpoint file containing events, metric snapshots, and spans.

OTel imports are deferred inside start() so that importing this module never pulls
in opentelemetry at startup unless telemetry is explicitly enabled.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from rhapsody.telemetry.events import BaseEvent
from rhapsody.telemetry.events import SessionEnded
from rhapsody.telemetry.events import SessionStarted
from rhapsody.telemetry.events import TaskCanceled
from rhapsody.telemetry.events import TaskCompleted
from rhapsody.telemetry.events import TaskCreated
from rhapsody.telemetry.events import TaskFailed
from rhapsody.telemetry.events import TaskQueued
from rhapsody.telemetry.events import TaskStarted
from rhapsody.telemetry.events import TaskSubmitted
from rhapsody.telemetry.events import make_event

if TYPE_CHECKING:
    from opentelemetry.sdk.metrics.export import MetricReader
    from opentelemetry.sdk.resources import Resource as OtelResource
    from opentelemetry.sdk.trace import SpanProcessor

    from rhapsody.telemetry.adapters.base import TelemetryAdapter

logger = logging.getLogger(__name__)

# Pushed onto the queue by stop() after queue.join() to unblock the dispatch loop cleanly.
_STOP_SENTINEL = object()


class SpanBuffer:
    """O(1)-append in-process span store.

    InMemorySpanExporter (an OTel test utility) does `_finished_spans += tuple(spans)` on every
    export call — O(n) per call, O(n²) total for n spans. For 40K spans that costs ~4 seconds inside
    the dispatch loop. list.extend() keeps append at O(1) amortized.

    Implements the SpanExporter interface so it can be passed to any SpanProcessor.
    """

    def __init__(self) -> None:
        import threading

        self._spans: list = []
        self._lock = threading.Lock()

    def export(self, spans) -> None:
        with self._lock:
            self._spans.extend(spans)

    def get_finished_spans(self) -> tuple:
        with self._lock:
            return tuple(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def shutdown(self) -> None:
        pass  # keep spans accessible via get_finished_spans() after tracer_provider.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class TelemetryManager:
    """Central telemetry orchestrator for a RHAPSODY session.

    Owns:
    - asyncio.Queue EventBus (non-blocking producer via put_nowait)
    - OpenTelemetry MeterProvider + TracerProvider (in-memory, no exporters)
    - Background dispatch loop
    - External subscriber list
    - Optional JSONL checkpoint file

    Usage::

        telemetry = TelemetryManager(
            session_id="session.0001",
            checkpoint_interval=30.0,
            checkpoint_path="./telemetry/",
        )
        await telemetry.start()
        telemetry.subscribe(my_callback)
        # ... session runs ...
        await telemetry.stop()

        print(telemetry.summary())
        print(telemetry.task_spans())
    """

    def __init__(
        self,
        session_id: str,
        checkpoint_interval: float | None = None,
        checkpoint_path: str | None = None,
        span_processors: list[SpanProcessor] | None = None,
        metric_readers: list[MetricReader] | None = None,
        resource: OtelResource | None = None,
    ) -> None:
        self._session_id = session_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: list[Callable[[BaseEvent], Any]] = []
        self._adapters: list[TelemetryAdapter] = []
        self._running = False
        self._dispatch_task: asyncio.Task | None = None
        self._checkpoint_task: asyncio.Task | None = None
        self._session_start_time: float = 0.0

        # Checkpoint config
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_path = checkpoint_path
        self._checkpoint_file: Any | None = None  # open file handle

        # Raw event log (in-memory, for convenience methods)
        self._event_log: list[BaseEvent] = []

        # OTel objects — set in start()
        self._meter_provider: Any = None
        self._tracer_provider: Any = None
        self._metric_reader: Any = None
        self._span_exporter: Any = None
        self._meter: Any = None
        self._tracer: Any = None

        # OTel instruments — set in _register_instruments()
        self._c_submitted: Any = None
        self._c_started: Any = None
        self._c_completed: Any = None
        self._c_failed: Any = None
        self._g_running: Any = None
        self._h_duration: Any = None
        self._g_cpu: Any = None
        self._g_memory: Any = None
        self._g_gpu: Any = None

        # Active OTel spans keyed by task_id
        self._active_spans: dict[str, Any] = {}
        self._session_span: Any = None
        self._completed_session_ctx: Any = None  # snapshot after session span ends

        # Wall-clock time recorded when RUNNING is first seen for each task.
        self._task_start_times: dict[str, float] = {}

        # Temporary span context store: task_id → SpanContext, populated in
        # _process_event() on TaskCompleted/Failed, consumed in _span_ids_for_event().
        self._completed_span_ctx: dict[str, Any] = {}

        # Last-seen resource values for UpDownCounter delta tracking
        self._last: dict[str, dict[str, float]] = {"cpu": {}, "mem": {}, "gpu": {}}

        # Registered by higher-layer runtimes to inject custom OTel span attributes.
        # Each callable receives the event and returns a dict of extra attributes.
        self._span_attribute_extractors: list[Callable[[BaseEvent], dict]] = []

        # OTel context snapshot captured at emit() time for each TaskCreated event.
        # Keyed by task_id. Consumed (popped) in _process_event() to restore the
        # correct parent span context in the dispatch loop.
        self._task_contexts: dict[str, Any] = {}

        # Token returned by context.attach() when the session span is activated in start().
        # Stored so it can be properly detached in stop().
        self._session_ctx_token: Any = None

        # Caller-supplied OTel extension points wired into RHAPSODY's internal providers
        # at start() time alongside SpanBuffer / InMemoryMetricReader.
        # Callers own exporter construction; RHAPSODY never imports specific exporters.
        self._extra_span_processors: list = list(span_processors or [])
        self._extra_metric_readers: list = list(metric_readers or [])
        # Optional Resource override. None → Resource.create() in start(), which reads
        # OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES from the environment automatically.
        self._resource: Any = resource

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The session identifier shared by all events from this manager."""
        return self._session_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the telemetry manager: set up OTel SDK, open checkpoint file, start loops."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Resource.create() reads OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES from the
        # environment automatically; the passed dict provides fallback defaults only.
        _resource = self._resource or Resource.create(
            {"service.name": "rhapsody", "session_id": self._session_id}
        )

        self._metric_reader = InMemoryMetricReader()
        self._span_exporter = SpanBuffer()

        self._meter_provider = MeterProvider(
            metric_readers=[self._metric_reader] + self._extra_metric_readers,
            resource=_resource,
        )
        self._tracer_provider = TracerProvider(resource=_resource)
        self._tracer_provider.add_span_processor(SimpleSpanProcessor(self._span_exporter))
        for proc in self._extra_span_processors:
            self._tracer_provider.add_span_processor(proc)

        self._meter = self._meter_provider.get_meter("rhapsody")
        self._tracer = self._tracer_provider.get_tracer("rhapsody")

        self._register_instruments()

        # Open checkpoint file
        if self._checkpoint_path is not None:
            self._open_checkpoint_file()

        self._session_start_time = time.time()

        # Create the session span synchronously so it is available before any
        # workflow_scope() / span_scope() call — _dispatch_loop runs in a separate
        # asyncio task and would otherwise process SessionStarted too late.
        self._session_span = self._tracer.start_span(
            "session", attributes={"session_id": self._session_id, "backend": "rhapsody"}
        )
        from opentelemetry import context as otel_context
        from opentelemetry import trace as trace_mod

        self._session_ctx_token = otel_context.attach(
            trace_mod.set_span_in_context(self._session_span)
        )

        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop(), name="telemetry-dispatch")

        if self._checkpoint_interval is not None and self._checkpoint_file is not None:
            self._checkpoint_task = asyncio.create_task(
                self._checkpoint_loop(), name="telemetry-checkpoint"
            )

        for adapter in self._adapters:
            adapter.start()

        self.emit(
            make_event(
                SessionStarted,
                session_id=self._session_id,
                backend="rhapsody",
            )
        )

    async def stop(self) -> None:
        """Stop: emit SessionEnded, drain queue, flush file, stop adapters, shutdown OTel."""
        duration = time.time() - self._session_start_time
        self.emit(
            make_event(
                SessionEnded,
                session_id=self._session_id,
                backend="rhapsody",
                duration_seconds=duration,
            )
        )

        self._running = False
        # Drain all in-flight events before signalling the dispatch loop to exit.
        # emit() is now a no-op (guard on _running), so no new events can arrive.
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Telemetry queue did not drain cleanly within timeout")
        self._queue.put_nowait(_STOP_SENTINEL)

        for adapter in self._adapters:
            try:
                adapter.stop()
            except Exception:
                logger.exception("Adapter %s failed to stop", type(adapter).__name__)

        if self._checkpoint_task and not self._checkpoint_task.done():
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass

        if self._dispatch_task and not self._dispatch_task.done():
            try:
                await asyncio.wait_for(self._dispatch_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Telemetry dispatch loop did not drain cleanly within timeout")

        # Final flush of metrics + spans to file
        if self._checkpoint_file is not None:
            self._write_metric_snapshot()
            self._write_span_snapshot()
            self._checkpoint_file.flush()
            self._checkpoint_file.close()
            self._checkpoint_file = None

        from opentelemetry import context as otel_context

        if self._session_ctx_token is not None:
            otel_context.detach(self._session_ctx_token)
            self._session_ctx_token = None

        if self._meter_provider:
            self._meter_provider.shutdown()
        if self._tracer_provider:
            self._tracer_provider.shutdown()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit(self, event: BaseEvent) -> None:
        """Enqueue an event for async processing (JSONL, metrics, spans).

        Non-blocking, O(1) for all event types. For TaskCreated events, captures the caller's OTel
        context so _process_event() can restore it in the dispatch loop and create the task span
        with the correct structural parent.
        """
        if not self._running:
            return
        self._queue.put_nowait(event)
        if self._tracer is not None and event.event_type == "TaskCreated":
            from opentelemetry import context as _ctx

            self._task_contexts[event.task_id] = _ctx.get_current()

    def subscribe(self, callback: Callable[[BaseEvent], Any]) -> None:
        """Register a callback to receive every event.

        Callbacks may be sync or async. Async callbacks are fire-and-forget tasks. Exceptions in
        callbacks never crash the dispatch loop.
        """
        self._subscribers.append(callback)

    def annotate_current_span(self, event: BaseEvent) -> None:
        """Add this event as a timestamped annotation on the currently active OTel span.

        Must be called from a context where a span is active (e.g. inside span_scope()). No-op if no
        valid span is active or telemetry has not been started.
        """
        if self._tracer is None:
            return
        from opentelemetry import trace as _trace

        active = _trace.get_current_span()
        if active is None or not active.get_span_context().is_valid:
            return
        attrs = event.attributes
        active.add_event(
            event.event_type,
            attributes={k: str(v) for k, v in attrs.items()} if attrs else {},
        )

    def register_span_enricher(self, func: Callable[[BaseEvent], dict]) -> None:
        """Register a callable that enriches OTel span attributes at span creation time.

        Called for every TaskCreated event (and TaskStarted fallback). The callable receives the
        event and must return a dict of additional OTel attributes to merge. Exceptions are logged
        at DEBUG level so faulty extractors cannot crash the dispatch loop.
        """
        self._span_attribute_extractors.append(func)

    @contextmanager
    def span_scope(self, name: str, attributes: dict | None = None):
        """Create a named span, make it the active OTel context for the duration.

        Any emit(TaskCreated) calls made while this scope is active will capture this span as the
        parent, so task spans become structural children.

        No-op when telemetry has not been started (self._tracer is None).
        """
        if self._tracer is None:
            yield
            return
        from opentelemetry import context as otel_context
        from opentelemetry import trace as trace_mod

        _active = trace_mod.get_current_span()
        _ctx = (
            None
            if _active.get_span_context().is_valid
            else (trace_mod.set_span_in_context(self._session_span) if self._session_span else None)
        )
        span = self._tracer.start_span(name, context=_ctx, attributes=attributes or {})
        token = otel_context.attach(trace_mod.set_span_in_context(span))
        try:
            yield
        finally:
            otel_context.detach(token)
            span.end()

    def register_adapter(self, adapter: TelemetryAdapter) -> None:
        """Register a backend telemetry adapter.

        Injects ``self`` as the adapter's manager so :meth:`TelemetryAdapter.start`
        requires no arguments — the back-reference is set here, once.
        """
        adapter._manager = self
        self._adapters.append(adapter)

    def attach_backend(
        self,
        backend,
        *,
        session_id: str,
        backend_name: str,
        interval: float = 5.0,
    ) -> None:
        """Detect backend type and register the appropriate telemetry adapter.

        Silently skips backends with no adapter (LocalExecutionBackend, NoopExecutionBackend,
        unknown types). Safe to call before or after start(); if already running the adapter is
        started immediately.
        """
        try:
            from rhapsody.backends.execution.concurrent import ConcurrentExecutionBackend
            from rhapsody.telemetry.adapters.concurrent import ConcurrentTelemetryAdapter

            if isinstance(backend, ConcurrentExecutionBackend):
                adapter = ConcurrentTelemetryAdapter(
                    session_id=session_id,
                    backend_name=backend_name,
                    interval=interval,
                )
                self.register_adapter(adapter)
                if self._running:
                    adapter.start()
                return
        except Exception:
            logger.warning("Could not attach ConcurrentTelemetryAdapter", exc_info=True)

        backend_cls = type(backend).__name__

        if backend_cls == "DaskExecutionBackend":
            try:
                from rhapsody.telemetry.adapters.dask import DaskTelemetryAdapter

                client = getattr(backend, "_client", None) or getattr(backend, "client", None)
                if client is not None:
                    adapter = DaskTelemetryAdapter(
                        client=client,
                        session_id=session_id,
                        backend_name=backend_name,
                        interval=interval,
                    )
                    self.register_adapter(adapter)
                    if self._running:
                        adapter.start()
            except Exception:
                logger.warning("Could not attach DaskTelemetryAdapter", exc_info=True)
            return

        if "Dragon" in backend_cls:
            try:
                from rhapsody.telemetry.adapters.dragon import DragonTelemetryAdapter

                adapter = DragonTelemetryAdapter(
                    session_id=session_id,
                    backend_name=backend_name,
                    interval=interval,
                )
                self.register_adapter(adapter)
                if self._running:
                    adapter.start()
            except Exception:
                logger.warning("Could not attach DragonTelemetryAdapter", exc_info=True)
            return

        # LocalExecutionBackend, NoopExecutionBackend, unknown — silently skip.

    def read_metrics(self) -> Any:
        """Return current OTel MetricsData snapshot (InMemoryMetricReader)."""
        if self._metric_reader is None:
            return None
        return self._metric_reader.get_metrics_data()

    def read_traces(self) -> tuple:
        """Return all finished OTel ReadableSpan objects (InMemorySpanExporter)."""
        if self._span_exporter is None:
            return ()
        return self._span_exporter.get_finished_spans()

    def export_as_otlp(self, path: str | None = None) -> str:
        """Export finished spans as a valid OTLP JSON string.

        Produces output compatible with Jaeger, Tempo, Grafana, and any standard
        OTel visualizer. The JSONL checkpoint is RHAPSODY's internal analysis format
        and is intentionally not OTLP-compliant; this method provides the standard
        export path alongside it.

        Args:
            path: If given, write the JSON to this file (parent dirs created as needed).

        Returns:
            OTLP JSON string (``{"resourceSpans": [...]}``)
        """
        spans = self.read_traces()

        def _typed_kv(k: str, v) -> dict:
            if isinstance(v, bool):
                return {"key": k, "value": {"boolValue": v}}
            if isinstance(v, int):
                return {"key": k, "value": {"intValue": str(v)}}
            if isinstance(v, float):
                return {"key": k, "value": {"doubleValue": v}}
            if isinstance(v, (list, tuple)):
                return {
                    "key": k,
                    "value": {"arrayValue": {"values": [_typed_kv("", e)["value"] for e in v]}},
                }
            return {"key": k, "value": {"stringValue": str(v)}}

        otlp_spans = []
        for s in spans:
            try:
                status_code = s.status.status_code.value
            except AttributeError:
                status_code = 0
            status_obj: dict = {"code": status_code}
            if s.status.description:
                status_obj["message"] = s.status.description

            try:
                kind = s.kind.value
            except AttributeError:
                kind = 0

            span_dict: dict = {
                "traceId": format(s.context.trace_id, "032x"),
                "spanId": format(s.context.span_id, "016x"),
                "name": s.name,
                "kind": kind,
                "startTimeUnixNano": str(s.start_time) if s.start_time else "0",
                "endTimeUnixNano": str(s.end_time) if s.end_time else "0",
                "attributes": [_typed_kv(k, v) for k, v in (s.attributes or {}).items()],
                "status": status_obj,
            }
            if s.parent:
                span_dict["parentSpanId"] = format(s.parent.span_id, "016x")
            if s.events:
                span_dict["events"] = [
                    {
                        "timeUnixNano": str(e.timestamp),
                        "name": e.name,
                        "attributes": [_typed_kv(k, v) for k, v in (e.attributes or {}).items()],
                    }
                    for e in s.events
                ]
            otlp_spans.append(span_dict)

        res_attrs = dict(self._tracer_provider.resource.attributes)
        res_attrs.setdefault("session_id", self._session_id)
        payload = {
            "resourceSpans": [
                {
                    "resource": {"attributes": [_typed_kv(k, v) for k, v in res_attrs.items()]},
                    "scopeSpans": [
                        {
                            "scope": {"name": "rhapsody"},
                            "spans": otlp_spans,
                        }
                    ],
                }
            ]
        }

        result = json.dumps(payload)

        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(result)

        return result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a plain-dict summary — no OTel knowledge required.

        Includes task counts, duration statistics, session duration, and latest resource utilization
        per node/backend.
        """
        self.read_metrics()
        spans = self.task_spans()

        task_counts = self.task_count()

        durations = [
            s["duration_seconds"]
            for s in spans
            if s["duration_seconds"] is not None and not s.get("incomplete_lifecycle", False)
        ]
        duration_stats: dict = {}
        if durations:
            duration_stats = {
                "total_seconds": round(sum(durations), 6),
                "mean_seconds": round(sum(durations) / len(durations), 6),
                "min_seconds": round(min(durations), 6),
                "max_seconds": round(max(durations), 6),
            }

        # Latest resource reading per (backend, node_id)
        resource_events = [e for e in self._event_log if e.event_type == "ResourceUpdate"]
        nodes: dict = {}
        for ev in resource_events:
            key = f"{ev.backend}/{ev.node_id}"
            nodes[key] = {
                "backend": ev.backend,
                "node_id": ev.node_id,
                "cpu_percent": ev.cpu_percent,
                "memory_percent": ev.memory_percent,
                "gpu_percent": ev.gpu_percent,
                "disk_read_bytes": ev.disk_read_bytes,
                "disk_write_bytes": ev.disk_write_bytes,
                "net_sent_bytes": ev.net_sent_bytes,
                "net_recv_bytes": ev.net_recv_bytes,
            }

        return {
            "session_id": self._session_id,
            "duration_seconds": self.session_duration(),
            "tasks": task_counts,
            "duration": duration_stats,
            "resources": nodes,
        }

    def task_count(self) -> dict:
        """Return task counts as a plain dict."""
        metrics = self.read_metrics()
        return {
            "submitted": int(_sum_metric(metrics, "tasks_submitted")),
            "completed": int(_sum_metric(metrics, "tasks_completed")),
            "failed": int(_sum_metric(metrics, "tasks_failed")),
            "canceled": int(_sum_metric(metrics, "tasks_canceled")),
            "running": int(_sum_metric(metrics, "tasks_running")),
        }

    def task_spans(self) -> list[dict]:
        """Return one plain dict per task span — no OTel knowledge required.

        Each dict contains: task_id, backend, executable, task_type, status,
        duration_seconds, error_type (failed tasks only), span_id, trace_id.
        """
        raw = self.read_traces()
        result = []
        for s in raw:
            if s.name != "task":
                continue
            attrs = dict(s.attributes or {})
            dur = (s.end_time - s.start_time) / 1e9 if s.end_time else None
            record: dict = {
                "task_id": attrs.get("task_id"),
                "backend": attrs.get("backend"),
                "executable": attrs.get("executable", ""),
                "task_type": attrs.get("task_type", ""),
                "status": attrs.get("status"),
                "duration_seconds": round(dur, 6) if dur is not None else None,
                "span_id": hex(s.context.span_id),
                "trace_id": hex(s.context.trace_id),
            }
            if "error_type" in attrs:
                record["error_type"] = attrs["error_type"]
            if attrs.get("incomplete_lifecycle"):
                record["incomplete_lifecycle"] = True
            result.append(record)
        return result

    def session_duration(self) -> float:
        """Return the session duration in seconds (0.0 if not yet stopped)."""
        ended = [e for e in self._event_log if e.event_type == "SessionEnded"]
        if ended:
            return round(ended[-1].duration_seconds, 6)
        if self._session_start_time:
            return round(time.time() - self._session_start_time, 6)
        return 0.0

    # ------------------------------------------------------------------
    # Internal: event bridge (called from TaskStateManager observer slot)
    # ------------------------------------------------------------------

    def _on_task_created(self, task: dict) -> None:
        """Called from Session.submit_tasks() when the future is created and the task is
        registered."""
        self.emit(
            make_event(
                TaskCreated,
                session_id=self._session_id,
                backend=task.get("backend", ""),
                task_id=task["uid"],
                attributes={
                    "executable": _task_executable(task),
                    "task_type": task.get("task_type", ""),
                },
            )
        )

    def _on_task_submitted(self, task: dict) -> None:
        """Called from Session.submit_tasks() after routing sets task['backend']."""
        self.emit(
            make_event(
                TaskSubmitted,
                session_id=self._session_id,
                backend=task.get("backend", ""),
                task_id=task["uid"],
                event_time=time.time(),
                attributes={
                    "executable": _task_executable(task),
                    "task_type": task.get("task_type", ""),
                },
            )
        )

    def _on_task_queued(self, task: dict) -> None:
        """Called from Session.submit_tasks() just before backend.submit_tasks()."""
        self.emit(
            make_event(
                TaskQueued,
                session_id=self._session_id,
                backend=task.get("backend", ""),
                task_id=task["uid"],
                attributes={
                    "executable": _task_executable(task),
                    "task_type": task.get("task_type", ""),
                },
            )
        )

    def _on_task_state_change(self, task: dict, state: str) -> None:
        """Called from TaskStateManager._update_task_impl — must be non-blocking."""
        now_wall = time.time()
        task_id = task["uid"]
        backend = task.get("backend", "")
        executable = _task_executable(task)
        task_type = task.get("task_type", "")

        if state == "RUNNING":
            # Record the wall-clock time we first see RUNNING so we can compute
            # duration in DONE/FAILED.
            self._task_start_times[task_id] = now_wall
            self.emit(
                make_event(
                    TaskStarted,
                    session_id=self._session_id,
                    backend=backend,
                    task_id=task_id,
                    event_time=now_wall,
                    attributes={
                        "executable": executable,
                        "task_type": task_type,
                    },
                )
            )

        elif state == "DONE":
            running_ts = self._task_start_times.pop(task_id, None)
            ended = now_wall
            if running_ts is not None:
                duration = max(ended - running_ts, 0.0)
                attrs: dict = {"executable": executable, "task_type": task_type}
            else:
                # NOTE: RUNNING was never seen for this task (backend did not emit
                # a RUNNING callback, or the task completed before telemetry started).
                # ended = now_wall is used as a best-effort event_time, but
                # duration_seconds = 0.0 and incomplete_lifecycle=True signal that
                # the true start time is unknown. If all tasks show incomplete_lifecycle,
                # the backend is likely not emitting a RUNNING callback — add one.
                duration = 0.0
                attrs = {
                    "executable": executable,
                    "task_type": task_type,
                    "incomplete_lifecycle": True,
                }
            self.emit(
                make_event(
                    TaskCompleted,
                    session_id=self._session_id,
                    backend=backend,
                    task_id=task_id,
                    event_time=ended,
                    duration_seconds=duration,
                    attributes=attrs,
                )
            )

        elif state == "FAILED":
            running_ts = self._task_start_times.pop(task_id, None)
            ended = now_wall
            exc = task.get("exception")
            error_type = type(exc).__name__ if exc is not None else "unknown"
            if running_ts is not None:
                duration = max(ended - running_ts, 0.0)
                attrs = {"executable": executable, "task_type": task_type, "error_type": error_type}
            else:
                # NOTE: RUNNING was never seen for this task — same situation as the
                # DONE branch above. ended = now_wall, duration_seconds = 0.0,
                # incomplete_lifecycle=True. Check whether the backend emits "RUNNING"
                # before reporting this as a data quality issue.
                duration = 0.0
                attrs = {
                    "executable": executable,
                    "task_type": task_type,
                    "error_type": error_type,
                    "incomplete_lifecycle": True,
                }
            self.emit(
                make_event(
                    TaskFailed,
                    session_id=self._session_id,
                    backend=backend,
                    task_id=task_id,
                    event_time=ended,
                    duration_seconds=duration,
                    error_type=error_type,
                    attributes=attrs,
                )
            )

        elif state == "CANCELED":
            running_ts = self._task_start_times.pop(task_id, None)
            ended = now_wall
            if running_ts is not None:
                duration = max(ended - running_ts, 0.0)
                attrs = {"executable": executable, "task_type": task_type}
            else:
                duration = 0.0
                attrs = {
                    "executable": executable,
                    "task_type": task_type,
                    "incomplete_lifecycle": True,
                }
            self.emit(
                make_event(
                    TaskCanceled,
                    session_id=self._session_id,
                    backend=backend,
                    task_id=task_id,
                    event_time=ended,
                    duration_seconds=duration,
                    attributes=attrs,
                )
            )

    # ------------------------------------------------------------------
    # Internal: span correlation
    # ------------------------------------------------------------------

    def _span_ids_for_event(self, event: BaseEvent) -> tuple[str | None, str | None]:
        """Return (trace_id_hex, span_id_hex) for the OTel span correlated to this event.

        Must be called AFTER _process_event() so that spans have been created/ended.

        Correlation rules:
        - TaskCreated/Submitted/Queued/Started/Completed/Failed → task span
        - SessionStarted/SessionEnded   → session span
        - ResourceUpdate                → session span
        """
        span_ctx = None

        if event.event_type in ("TaskCreated", "TaskStarted", "TaskSubmitted", "TaskQueued"):
            span = self._active_spans.get(event.task_id)
            if span:
                span_ctx = span.get_span_context()
            else:
                # Same-batch scenario: _process_event(TaskCompleted/Failed/Canceled) already
                # ran and moved the span_ctx to _completed_span_ctx. Use .get (not .pop) —
                # the terminal event's own _span_ids_for_event call will pop it.
                span_ctx = self._completed_span_ctx.get(event.task_id)
            if span_ctx is None and event.event_type in ("TaskSubmitted", "TaskQueued"):
                # Fallback: TaskCreated not yet processed — attach to session trace only.
                if self._session_span:
                    session_ctx = self._session_span.get_span_context()
                    if session_ctx.is_valid:
                        return hex(session_ctx.trace_id), None

        elif event.event_type in ("TaskCompleted", "TaskFailed", "TaskCanceled"):
            span_ctx = self._completed_span_ctx.pop(event.task_id, None)

        elif event.event_type in ("SessionStarted", "SessionEnded"):
            if self._session_span:
                span_ctx = self._session_span.get_span_context()
            elif event.event_type == "SessionEnded":
                # Same-batch: _process_event already ended _session_span; use snapshot.
                span_ctx = self._completed_session_ctx

        elif event.event_type == "ResourceUpdate":
            # Node/GPU metric belongs to the session scope — attach to the session
            # span so it is queryable as a child of the root in OTel tools.
            if self._session_span:
                span_ctx = self._session_span.get_span_context()

        else:
            # Generic fallback for custom/extension events (e.g.
            # any user-emitted event via telemetry.emit()).
            # Correlation by duck typing:
            #   - has task_id → belongs to the task span (active or recently completed)
            #   - no task_id  → belongs to the session span
            # Use .get not .pop — custom events are never terminal; they never own the
            # span lifecycle.
            task_id = getattr(event, "task_id", None)
            if task_id:
                span = self._active_spans.get(task_id)
                if span:
                    span_ctx = span.get_span_context()
                else:
                    span_ctx = self._completed_span_ctx.get(task_id)
            elif self._session_span:
                span_ctx = self._session_span.get_span_context()

        if span_ctx is None or not span_ctx.is_valid:
            return None, None
        return hex(span_ctx.trace_id), hex(span_ctx.span_id)

    # ------------------------------------------------------------------
    # Internal: OTel instruments
    # ------------------------------------------------------------------

    def _register_instruments(self) -> None:
        self._c_submitted = self._meter.create_counter(
            "tasks_submitted", description="Total tasks submitted"
        )
        self._c_started = self._meter.create_counter(
            "tasks_started", description="Total tasks started"
        )
        self._c_completed = self._meter.create_counter(
            "tasks_completed", description="Total tasks completed"
        )
        self._c_failed = self._meter.create_counter(
            "tasks_failed", description="Total tasks failed"
        )
        self._c_canceled = self._meter.create_counter(
            "tasks_canceled", description="Total tasks canceled"
        )
        self._g_running = self._meter.create_up_down_counter(
            "tasks_running", description="Tasks currently running"
        )
        self._h_duration = self._meter.create_histogram(
            "task_duration_seconds", description="Task execution duration in seconds"
        )
        self._g_cpu = self._meter.create_up_down_counter(
            "node_cpu_utilization", description="Node CPU utilization percent"
        )
        self._g_memory = self._meter.create_up_down_counter(
            "node_memory_utilization", description="Node memory utilization percent"
        )
        self._g_gpu = self._meter.create_up_down_counter(
            "node_gpu_utilization", description="Node GPU utilization percent"
        )

    # ------------------------------------------------------------------
    # Internal: dispatch loop
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """Background task: drain queue in batches, update OTel instruments, notify subscribers."""
        from opentelemetry import context as otel_context
        from opentelemetry import trace

        _batch_size = 500

        while True:
            # Fast path: drain up to _batch_size events without yielding to the event loop.
            # get_nowait() is a plain deque.popleft() — zero asyncio scheduling overhead.
            batch: list[BaseEvent] = []
            try:
                while len(batch) < _batch_size:
                    item = self._queue.get_nowait()
                    if item is _STOP_SENTINEL:
                        self._queue.task_done()
                        if batch:
                            self._flush_batch(batch, trace, otel_context)
                            for _ in batch:
                                self._queue.task_done()
                        return
                    batch.append(item)
            except asyncio.QueueEmpty:
                pass

            if not batch:
                # Slow path: queue was empty — wait without a timeout.
                # Plain queue.get(), no wait_for(), no timer handle, no Task allocation.
                item = await self._queue.get()
                if item is _STOP_SENTINEL:
                    self._queue.task_done()
                    return
                batch.append(item)

            self._flush_batch(batch, trace, otel_context)
            for _ in batch:
                self._queue.task_done()  # task_done AFTER processing (Fix 1)

            # Yield only when a full batch was drained — queue likely has more items.
            # Partial batch means queue is exhausted; await queue.get() yields naturally.
            if len(batch) == _batch_size:
                await asyncio.sleep(0)

    def _flush_batch(self, batch: list[BaseEvent], trace_mod: Any, otel_context: Any) -> None:
        """Process a collected batch: OTel instruments, JSONL write, subscribers."""
        for event in batch:
            self._event_log.append(event)
            try:
                self._process_event(event, trace_mod, otel_context)
            except Exception:
                logger.exception("Error processing telemetry event %s", event.event_type)

        if self._checkpoint_file is not None:
            self._write_batch(batch)

        if self._subscribers:
            for event in batch:
                for sub in self._subscribers:
                    try:
                        if asyncio.iscoroutinefunction(sub):
                            asyncio.create_task(sub(event))
                        else:
                            sub(event)
                    except Exception:
                        logger.debug("Subscriber raised exception", exc_info=True)

    def _write_batch(self, events: list[BaseEvent]) -> None:
        """Write a batch of events to JSONL in a single file.write() call."""
        lines = []
        for event in events:
            try:
                # fields() + getattr: shallow copy that captures init=False class-level
                # fields (e.g. event_type) which vars()/.__dict__ misses on frozen subclasses.
                d = {f.name: getattr(event, f.name) for f in dataclasses.fields(event)}
                trace_id, span_id = self._span_ids_for_event(event)
                d["trace_id"] = trace_id
                d["span_id"] = span_id
                d["name"] = event.event_type
                d["section"] = "event"
                lines.append(json.dumps(d, default=str))
            except Exception:
                logger.debug("Failed to serialize event for checkpoint", exc_info=True)
        if lines:
            try:
                self._checkpoint_file.write("\n".join(lines) + "\n")
            except OSError:
                logger.warning(
                    "Telemetry checkpoint write failed; events lost for this batch",
                    exc_info=True,
                )

    def _process_event(self, event: BaseEvent, trace_mod: Any, otel_context: Any) -> None:
        """Translate a RHAPSODY event into OTel instrument calls and span lifecycle."""
        etype = event.event_type
        attrs = {"session_id": event.session_id, "backend": event.backend or ""}

        if etype == "TaskCreated":
            executable = event.attributes.get("executable", "")
            task_type = event.attributes.get("task_type", "")
            a = {
                **attrs,
                "task_id": event.task_id,
                "executable": executable,
                "task_type": task_type,
            }
            for extractor in self._span_attribute_extractors:
                try:
                    a.update(extractor(event))
                except Exception:
                    logger.debug("span attribute extractor failed", exc_info=True)
            _ctx = self._task_contexts.pop(event.task_id, None)
            if _ctx is not None:
                _token = otel_context.attach(_ctx)
                try:
                    self._active_spans[event.task_id] = self._tracer.start_span(
                        "task", attributes=a
                    )
                finally:
                    otel_context.detach(_token)
            else:
                self._active_spans[event.task_id] = self._tracer.start_span("task", attributes=a)

        elif etype == "TaskSubmitted":
            self._c_submitted.add(1, {**attrs, "task_id": event.task_id})

        elif etype == "TaskStarted":
            executable = event.attributes.get("executable", "")
            task_type = event.attributes.get("task_type", "")
            a = {
                **attrs,
                "task_id": event.task_id,
                "executable": executable,
                "task_type": task_type,
            }
            self._c_started.add(1, a)
            self._g_running.add(1, a)
            if event.task_id not in self._active_spans:
                # Fallback: TaskCreated was never seen (incomplete lifecycle).
                # Use a_span copy so metric attributes (a) are not polluted with
                # workflow_id or other span-only extractor fields.
                a_span = dict(a)
                for extractor in self._span_attribute_extractors:
                    try:
                        a_span.update(extractor(event))
                    except Exception:
                        logger.debug("span attribute extractor failed", exc_info=True)
                _ctx = self._task_contexts.pop(event.task_id, None)
                if _ctx is not None:
                    _token = otel_context.attach(_ctx)
                    try:
                        self._active_spans[event.task_id] = self._tracer.start_span(
                            "task", attributes=a_span
                        )
                    finally:
                        otel_context.detach(_token)
                else:
                    self._active_spans[event.task_id] = self._tracer.start_span(
                        "task", attributes=a_span
                    )

        elif etype == "TaskCompleted":
            executable = event.attributes.get("executable", "")
            task_type = event.attributes.get("task_type", "")
            a = {
                **attrs,
                "task_id": event.task_id,
                "executable": executable,
                "task_type": task_type,
            }
            self._c_completed.add(1, a)
            self._h_duration.record(event.duration_seconds, a)
            span = self._active_spans.pop(event.task_id, None)
            if span is not None:
                self._g_running.add(-1, a)
            else:
                span = self._tracer.start_span("task", attributes=a)
            span.set_attribute("status", "completed")
            span.set_attribute("executable", executable)
            span.set_attribute("task_type", task_type)
            if event.attributes.get("incomplete_lifecycle"):
                span.set_attribute("incomplete_lifecycle", True)
            span.set_status(trace_mod.StatusCode.OK)
            span.end()
            self._completed_span_ctx[event.task_id] = span.get_span_context()

        elif etype == "TaskFailed":
            executable = event.attributes.get("executable", "")
            task_type = event.attributes.get("task_type", "")
            error_type = event.attributes.get("error_type", event.error_type)
            a = {
                **attrs,
                "task_id": event.task_id,
                "executable": executable,
                "task_type": task_type,
            }
            self._c_failed.add(1, a)
            self._h_duration.record(event.duration_seconds, a)
            span = self._active_spans.pop(event.task_id, None)
            if span is not None:
                self._g_running.add(-1, a)
            else:
                span = self._tracer.start_span("task", attributes=a)
            span.set_attribute("status", "failed")
            span.set_attribute("error_type", error_type)
            span.set_attribute("executable", executable)
            span.set_attribute("task_type", task_type)
            if event.attributes.get("incomplete_lifecycle"):
                span.set_attribute("incomplete_lifecycle", True)
            span.set_status(trace_mod.StatusCode.ERROR, error_type)
            span.end()
            self._completed_span_ctx[event.task_id] = span.get_span_context()

        elif etype == "TaskCanceled":
            executable = event.attributes.get("executable", "")
            task_type = event.attributes.get("task_type", "")
            a = {
                **attrs,
                "task_id": event.task_id,
                "executable": executable,
                "task_type": task_type,
            }
            self._c_canceled.add(1, a)
            self._h_duration.record(event.duration_seconds, a)
            span = self._active_spans.pop(event.task_id, None)
            if span is not None:
                self._g_running.add(-1, a)
            else:
                span = self._tracer.start_span("task", attributes=a)
            span.set_attribute("status", "canceled")
            span.set_attribute("executable", executable)
            span.set_attribute("task_type", task_type)
            if event.attributes.get("incomplete_lifecycle"):
                span.set_attribute("incomplete_lifecycle", True)
            span.set_status(trace_mod.StatusCode.ERROR)
            span.end()
            self._completed_span_ctx[event.task_id] = span.get_span_context()

        elif etype == "TaskQueued":
            span = self._active_spans.get(event.task_id)
            if span is not None and span.is_recording():
                span.add_event("TaskQueued", attributes={"task_id": event.task_id})

        elif etype == "SessionStarted":
            pass  # session span created synchronously in start() for immediate availability

        elif etype == "SessionEnded":
            if self._session_span:
                self._completed_session_ctx = self._session_span.get_span_context()
                self._session_span.end()
                self._session_span = None

        elif etype == "ResourceUpdate":
            ra = {**attrs, "node_id": event.node_id}
            for gauge, key, val in (
                (self._g_cpu, "cpu", event.cpu_percent),
                (self._g_memory, "mem", event.memory_percent),
                (self._g_gpu, "gpu", event.gpu_percent),
            ):
                if val is not None:
                    last = self._last[key].get(event.node_id, 0.0)
                    gauge.add(val - last, ra)
                    self._last[key][event.node_id] = val

    # ------------------------------------------------------------------
    # Internal: checkpoint file
    # ------------------------------------------------------------------

    def _open_checkpoint_file(self) -> None:
        path = Path(self._checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)
        ts = int(self._session_start_time or time.time())
        filename = f"{self._session_id}.{ts}.telemetry.jsonl"
        filepath = path / filename
        try:
            self._checkpoint_file = open(filepath, "w", buffering=131072)  # 128 KB block buffer
            logger.info("Telemetry checkpoint file: %s", filepath)
        except OSError:
            logger.exception("Could not open telemetry checkpoint file %s", filepath)

    def _write_line(self, section: str, data: dict) -> None:
        """Append one JSON line to the checkpoint file."""
        data["section"] = section
        self._checkpoint_file.write(json.dumps(data, default=str) + "\n")

    def _write_metric_snapshot(self) -> None:
        """Append a metric snapshot line to the checkpoint file."""
        if self._checkpoint_file is None:
            return
        metrics = self.read_metrics()
        snapshot: dict = {"checkpoint_time": time.time()}
        if metrics:
            for rm in metrics.resource_metrics:
                for sm in rm.scope_metrics:
                    for metric in sm.metrics:
                        dps = metric.data.data_points
                        if not dps:
                            continue
                        # Histogram data points expose .sum/.count, not .value
                        if hasattr(dps[0], "sum") and not hasattr(dps[0], "value"):
                            snapshot[metric.name + "_count"] = sum(dp.count for dp in dps)
                            snapshot[metric.name + "_sum"] = sum(dp.sum for dp in dps)
                        else:
                            snapshot[metric.name] = sum(dp.value for dp in dps)
        try:
            self._write_line("metric", snapshot)
        except Exception:
            logger.debug("Failed to write metric snapshot", exc_info=True)

    def _write_span_snapshot(self) -> None:
        """Append finished span records to the checkpoint file."""
        if self._checkpoint_file is None:
            return
        for s in self.read_traces():
            dur = (s.end_time - s.start_time) / 1e6 if s.end_time else None
            record = {
                "name": s.name,
                "span_id": hex(s.context.span_id),
                "trace_id": hex(s.context.trace_id),
                "parent_span_id": hex(s.parent.span_id) if s.parent else None,
                "start_ns": s.start_time,
                "end_ns": s.end_time,
                "start_time_s": s.start_time / 1e9 if s.start_time else None,
                "end_time_s": s.end_time / 1e9 if s.end_time else None,
                "duration_ms": round(dur, 3) if dur is not None else None,
                "status_code": s.status.status_code.name,
                "status_description": s.status.description or None,
                "attributes": dict(s.attributes or {}),
            }
            try:
                self._write_line("span", record)
            except Exception:
                logger.debug("Failed to write span record", exc_info=True)

    async def _checkpoint_loop(self) -> None:
        """Periodically flush metric snapshots and span records to the checkpoint file."""
        while self._running:
            await asyncio.sleep(self._checkpoint_interval)
            if self._checkpoint_file is not None:
                self._write_metric_snapshot()
                self._write_span_snapshot()
                self._checkpoint_file.flush()


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------


def _sum_metric(metrics_data: Any, name: str) -> float:
    total = 0.0
    if metrics_data is None:
        return total
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    for dp in metric.data.data_points:
                        total += dp.value
    return total


def _task_executable(task: dict) -> str:
    """Extract the executable/function identifier from a task dict."""
    # ComputeTask stores executable directly
    if "executable" in task:
        return str(task["executable"])
    # FunctionTask stores func reference
    func = task.get("func") or task.get("function")
    if func is not None:
        return getattr(func, "__name__", str(func))
    # AITask or others may have a model field
    model = task.get("model") or task.get("model_name")
    if model:
        return str(model)
    return ""
