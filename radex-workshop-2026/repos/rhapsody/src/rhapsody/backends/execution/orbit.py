"""Orbit execution backend for remote task execution via ORBIT.

This module provides a backend that submits tasks to a remote HPC node
through the ORBIT broker/endpoint infrastructure.  The Endpoint node
runs a Rhapsody plugin with a local backend (e.g. Dragon V3) that
actually executes the work.

Internally delegates to ``RhapsodyClient`` so all transport-level
optimizations (template compression, pipelined batching, event-based
wait, batch notifications) are inherited automatically.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import Callable

from ..base import BaseBackend
from ..constants import BackendMainStates
from ..constants import StateMapper

# ``radical.orbit`` is imported at module level so that tests can patch
# ``EndpointRuntime`` here.  When the package isn't installed we keep
# ``EndpointRuntime = None`` and remember the original ImportError; the
# actual chained re-raise happens in ``OrbitExecutionBackend.__init__``
# so the user sees the real cause (e.g. a downstream import failure
# inside ``radical.orbit`` itself, not just "package missing").
try:
    from radical.orbit import EndpointRuntime

    _radical_orbit_import_error = None
except ImportError as exc:
    EndpointRuntime = None
    _radical_orbit_import_error = exc

try:
    import radical.prof as rprof
except ImportError:
    rprof = None


class OrbitExecutionBackend(BaseBackend):
    """Execution backend that delegates to a remote ORBIT node.

    Uses ``radical.orbit.EndpointRuntime`` and ``RhapsodyClient`` for all
    communication — inheriting batching, template compression,
    pipelined submission, and event-based notifications.

    When tasks are submitted individually (one at a time), a batching
    layer collects them over a short time window (default 0.1 s) and
    flushes them as a single bulk request, dramatically reducing
    per-task HTTP round-trip overhead.

    Args:
        broker_url:    URL of the ORBIT broker
                       (e.g. ``"https://localhost:8000"``).  If omitted,
                       falls back to the ``RADICAL_ORBIT_BROKER_URL`` env var
                       or ``~/.radical/orbit/broker.url`` (resolved by
                       ``EndpointRuntime`` itself; the bearer token is
                       resolved the same way from
                       ``RADICAL_ORBIT_BROKER_TOKEN`` /
                       ``~/.radical/orbit/broker.token``).
        endpoint_name:     Name of the endpoint to target.  If omitted, the
                       backend auto-selects the first present endpoint whose
                       topology entry advertises a rhapsody plugin; the
                       broker and pure consumers (including this runtime
                       itself) are skipped.  Raises ``RuntimeError`` from
                       ``await backend`` if no candidate is found.
        backends:      Backend names to request on the remote session
                       (default ``["dragon_v3"]``).
        name:          Backend name for Rhapsody registration
                       (default ``"orbit"``).
        batch_window:  Seconds to collect tasks before flushing
                       (default 0.25).  Set to 0 to disable batching.
        batch_limit:   Max tasks per batch — triggers an immediate flush
                       when reached (default 1024).
        start_timeout: Seconds to wait for broker registration and the
                       first topology frame (default 30).
        init_timeout:  Seconds to wait for the remote session to become
                       ready after registration (default 120).  Together
                       with ``start_timeout`` this bounds the whole
                       ``await backend`` initialization.
    """

    _DEFAULT_BATCH_WINDOW = 0.25
    _PLUGIN_NAME = "rhapsody"
    _TERMINAL_STATES = frozenset({"DONE", "FAILED", "CANCELED", "COMPLETED"})

    def __init__(
        self,
        broker_url: str | None = None,
        endpoint_name: str | None = None,
        backends: list[str] | None = None,
        name: str = "orbit",
        plugin_name: str = _PLUGIN_NAME,
        batch_window: float | None = None,
        batch_limit: int = 1024,
        start_timeout: float = 30.0,
        init_timeout: float = 120.0,
    ):
        super().__init__(name=name)

        if EndpointRuntime is None:
            raise ImportError(
                f"OrbitExecutionBackend: cannot import radical.orbit: {_radical_orbit_import_error}"
            ) from _radical_orbit_import_error

        self.logger = logging.getLogger(__name__)
        self._broker_url = broker_url
        self._endpoint_name = endpoint_name
        self._plugin_name = plugin_name
        self._remote_backends = backends or ["dragon_v3"]
        self._start_timeout = start_timeout
        self._init_timeout = init_timeout
        self._endpoint_python_mm: tuple | None = None
        self._endpoint_python_lookup_done = False

        self._runtime = None  # EndpointRuntime
        self._rh = None  # RhapsodyClient (from get_rhapsody_handle)
        self._tasks: dict[str, dict] = {}
        # uids for which a terminal state callback has already fired, so the
        # notification and local-cancel paths don't double-fire the same
        # terminal state
        self._terminal_fired: set[str] = set()

        self._callback_func: Callable = lambda t, s: None
        self._initialized = False
        # Serializes _async_init so concurrent first submissions (which now
        # await inside init) don't run the initialization logic twice.
        self._init_lock = asyncio.Lock()
        self._backend_state = BackendMainStates.INITIALIZED
        self._loop: asyncio.AbstractEventLoop | None = None

        # -- submission batching --
        self._batch_window = (
            batch_window if batch_window is not None else self._DEFAULT_BATCH_WINDOW
        )
        self._batch_limit = batch_limit
        self._batch_buffer: list[dict] = []
        # ``_batch_lock`` guards ``_batch_buffer`` / ``_flush_handle`` and is
        # held only briefly (append, or detach-the-batch).  ``_send_lock``
        # serializes the actual remote sends, so concurrent flushes neither
        # overlap on the (not necessarily thread-safe) client nor reorder
        # batches — while leaving the batch lock free for other submitters to
        # keep buffering during a slow network flush.
        self._batch_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._flush_handle: asyncio.TimerHandle | None = None
        # Background (timer-triggered) flush tasks, awaited on shutdown so a
        # batch that is mid-send is never dropped.
        self._inflight_flushes: set[asyncio.Task] = set()

        # -- profiling --
        self._prof = rprof.Profiler("client.task", ns="radical.orbit") if rprof else None

    # ------------------------------------------------------------------
    # Async init
    # ------------------------------------------------------------------

    def __await__(self):
        return self._async_init().__await__()

    async def _async_init(self):
        if self._initialized:
            return self

        async with self._init_lock:
            # Re-check under the lock: a concurrent caller may have finished
            # initialization while we waited to acquire it.
            if self._initialized:
                return self

            return await self._do_async_init()

    async def _do_async_init(self):
        self._loop = asyncio.get_running_loop()

        # Register states
        StateMapper.register_backend_states_with_defaults(backend=self)
        StateMapper.register_backend_tasks_states_with_defaults(backend=self)

        # Runs blocking network I/O (EndpointRuntime.start, topology scan,
        # remote session registration) so keep it off the event loop.  Can
        # block up to ``start_timeout + init_timeout`` — all connection and
        # session waiting is paid here, never on the submit path.
        self._rh = await asyncio.to_thread(self._get_rhapsody_handle)

        # Register persistent notification callback for task completions
        self._rh.register_notification_callback(self._on_task_notification, topic="task_status")
        self._rh.register_notification_callback(
            self._on_task_notification, topic="task_status_batch"
        )

        self._initialized = True
        self.logger.info(
            "Orbit backend ready: %s/%s (session %s)",
            self._endpoint_name,
            self._plugin_name,
            self._rh.sid,
        )
        return self

    # ------------------------------------------------------------------
    # Notification handling
    # ------------------------------------------------------------------

    def _on_task_notification(self, endpoint, plugin, topic, data):
        """Notification callback — runs on the runtime's callback thread.

        Marshal the whole update onto the event loop so that every read/write of
        ``self._tasks`` and the task dicts happens single-threaded there, never
        racing loop-side writers (submit / cancel / flush-error).
        """
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._handle_notification, topic, data)

    def _handle_notification(self, topic, data):
        """Dispatch a notification on the event-loop thread."""
        if topic == "task_status_batch":
            for t in data.get("tasks", []):
                self._apply_task_update(t)
        else:
            self._apply_task_update(data)

    def _apply_task_update(self, body: dict):
        """Apply a single task status update from a notification."""
        if not isinstance(body, dict):
            return

        uid = body.get("uid")
        state = body.get("state", "")

        if uid not in self._tasks:
            return

        task = self._tasks[uid]

        # Decode base64-encoded return values (bytes results)
        if body.get("_return_value_encoding") == "base64":
            import base64

            body["return_value"] = base64.b64decode(body["return_value"])
            del body["_return_value_encoding"]

        # A remote failure may serialize its exception as a string or dict;
        # coerce it to a real exception so session.py raises it instead of
        # silently treating the failed task as successful.
        # ``.get(k, default)`` only falls back when the key is *absent*; a
        # present-but-null "exception" (common in JSON) must still fall back to
        # "error".
        exc = body.get("exception")
        if exc is None:
            exc = body.get("error")
        if exc is not None and not isinstance(exc, BaseException):
            body["exception"] = RuntimeError(str(exc))

        # Update local task dict with remote results
        for key in (
            "state",
            "stdout",
            "stderr",
            "exit_code",
            "return_value",
            "exception",
            "traceback",
            "error",
        ):
            if key in body:
                task[key] = body[key]

        # Profile task completion on client side
        if self._prof:
            self._prof.prof("task_complete", uid=uid, state=state)

        # Already on the event-loop thread (dispatched from
        # _on_task_notification via call_soon_threadsafe), so fire directly.
        self._fire_callback(task, state)

    def _fire_callback(self, task: dict, state: str) -> None:
        """Invoke the Rhapsody state callback, at most once per terminal state.

        Both the remote notification path and the local cancel path can report
        the same terminal state for a task (e.g. a cancel fires the callback
        locally *and* the endpoint emits a matching CANCELED notification).
        This guard ensures consumers that resolve a future on the terminal
        callback see it exactly once.

        Once a terminal state has fired, the task is dropped from ``_tasks``
        (and its uid from ``_terminal_fired``) to bound memory over a
        long-running session.  A later duplicate notification is then ignored
        by ``_apply_task_update`` because the uid is no longer tracked.

        Must run on the event loop thread — ``_terminal_fired`` is accessed
        without a lock, and every caller (notifications via
        ``call_soon_threadsafe``,
        cancel paths from coroutine bodies) already runs there.
        """
        uid = task.get("uid")
        terminal = bool(uid) and str(state).upper() in self._TERMINAL_STATES
        if terminal:
            if uid in self._terminal_fired:
                return
            self._terminal_fired.add(uid)
        self._callback_func(task, state)
        if terminal:
            self._tasks.pop(uid, None)
            self._terminal_fired.discard(uid)

    # ------------------------------------------------------------------
    # Endpoint auto-selection and Plugin retrieval
    # ------------------------------------------------------------------

    def _get_rhapsody_handle(self) -> Any:
        """Connect the EndpointRuntime, pick an endpoint, return a RhapsodyClient.

        The endpoint is either the named one, or auto-selected as the first
        present endpoint whose topology entry advertises a rhapsody plugin
        (the broker and pure consumers — including this runtime itself — are
        skipped).  Raises ``RuntimeError`` if no candidate is found.

        Runs blocking network I/O; always called via ``asyncio.to_thread``.
        """
        import uuid

        # A unique name suffix avoids the broker's name-in-use rejection when
        # several rhapsody clients connect (or one restarts within the
        # liveness grace window).
        rt = EndpointRuntime(
            broker_url=self._broker_url,
            name=f"rhapsody.{uuid.uuid4().hex[:8]}",
        )
        try:
            rt.start(wait=True, timeout=self._start_timeout)
            # start() returns silently on timeout — check registration
            # explicitly, or the failure would surface further down as a
            # misleading "no endpoint found".
            if not rt.wait_registered(timeout=0):
                raise RuntimeError(
                    f"failed to register with ORBIT broker {rt.broker_url!r} "
                    f"within {self._start_timeout}s"
                )
            self._broker_url = rt.broker_url

            # find a suitable endpoint from the (local) topology snapshot
            if not self._endpoint_name:
                for eid, info in rt.topology().items():
                    if eid == rt.name:
                        continue  # this runtime itself
                    if info.get("role") in ("broker", "consumer"):
                        continue
                    if info.get("liveness") not in (None, "present"):
                        continue  # suspect / lost
                    if self._plugin_name in (info.get("plugins") or {}):
                        self.logger.info(
                            "auto-selected endpoint %r (plugin %r)", eid, self._plugin_name
                        )
                        self._endpoint_name = eid
                        break

            if not self._endpoint_name:
                raise RuntimeError(
                    f"no endpoint advertises a {self._plugin_name!r} plugin "
                    f"on broker {self._broker_url}"
                )

            # get_plugin() registers the remote session and blocks until it
            # is ready (bounded by init_timeout).
            rh = rt.get_plugin(
                self._endpoint_name,
                self._plugin_name,
                backends=self._remote_backends,
                init_timeout=self._init_timeout,
            )
        except Exception:
            # Don't leak the runtime's daemon threads / WebSocket on a failed
            # init — callers that never reach shutdown() would keep it alive.
            rt.stop()
            raise

        self._runtime = rt
        return rh

    # ------------------------------------------------------------------
    # Python-version compatibility for cloudpickled function tasks
    # ------------------------------------------------------------------

    async def _check_python_compat(self, tasks: list) -> None:
        """Raise if any cloudpickled task in *tasks* would hit a Python version mismatch on the
        remote endpoint.  No check is performed if the batch contains only executable or import-path
        tasks, or if the endpoint's Python version could not be determined.

        Cloudpickle serializes function bytecode using ``CodeType``,
        whose tuple shape changed between Python 3.10 and 3.11 — any
        cross-minor-version skew fails deserialization on the endpoint.
        """
        import sys

        def needs_compat(t: dict) -> bool:
            fn = t.get("function")
            # A raw Python callable (a ComputeTask function not yet serialized)
            # takes the same cloudpickle path once submitted, so it must be
            # checked too — not just already-``cloudpickle::``-encoded strings.
            return (
                callable(fn)
                or (isinstance(fn, str) and fn.startswith("cloudpickle::"))
                or bool(t.get("_pickled_fields"))
            )

        if not any(needs_compat(t) for t in tasks):
            return

        if not self._endpoint_python_lookup_done:
            # Only the remote lookup is transient (network / plugin not yet
            # ready) — on failure we leave the check armed so a later
            # submission retries.  A successful lookup settles the check for
            # good, even if the reported version can't be parsed (that's
            # deterministic and retrying wouldn't help).
            info = None

            def _lookup():
                # get_plugin() auto-registers an ephemeral session even
                # though host_role() needs none — close the client so failed
                # retries don't accumulate sessions on the endpoint.
                si = self._runtime.get_plugin(self._endpoint_name, "sysinfo")
                try:
                    return si.host_role()
                finally:
                    si.close()

            try:
                # Blocking network I/O — run off the event loop.
                info = await asyncio.to_thread(_lookup)
            except Exception as exc:
                self.logger.debug(f"pickle compat lookup failed, will retry: {exc}")

            if info is not None:
                self._endpoint_python_lookup_done = True
                pyver = info.get("python_version") or ""
                parts = pyver.split(".")
                try:
                    if len(parts) >= 2:
                        self._endpoint_python_mm = (int(parts[0]), int(parts[1]))
                except ValueError:
                    pass
                if not self._endpoint_python_mm:
                    self.logger.debug(f"skip pickle compat check ({pyver})")

        if not self._endpoint_python_mm:
            return

        client_mm = (sys.version_info.major, sys.version_info.minor)
        if self._endpoint_python_mm != client_mm:
            raise RuntimeError(
                f"function tasks cannot be submitted to endpoint "
                f"{self._endpoint_name!r}: client Python "
                f"{client_mm[0]}.{client_mm[1]} != endpoint Python "
                f"{self._endpoint_python_mm[0]}.{self._endpoint_python_mm[1]}.  "
                f"cloudpickle is not portable across Python minor "
                f"versions — align the venvs, or use executable / "
                f"import-path tasks for this endpoint."
            )

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    async def submit_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Submit tasks to the remote endpoint for execution.

        When batching is enabled (batch_window > 0), tasks are collected
        in an internal buffer and flushed after the window expires or the
        buffer reaches ``batch_limit``.  This turns many small individual
        submissions into fewer bulk HTTP requests.

        When batching is disabled (batch_window == 0), delegates directly
        to ``RhapsodyClient.submit_tasks()``.
        """
        if not self._initialized:
            await self._async_init()

        if self._backend_state != BackendMainStates.RUNNING:
            self._backend_state = BackendMainStates.RUNNING

        # Fail fast on cross-Python cloudpickle skew *before* registering or
        # buffering, so the error always propagates to this caller.  When
        # batching defers the flush, the check would otherwise run inside an
        # orphaned ``ensure_future`` task and never reach the awaiter.
        await self._check_python_compat(tasks)

        # Assign UIDs, register locally, emit submit prof events — single pass
        import uuid

        prof = self._prof
        for task in tasks:
            task.setdefault("uid", f"task.{uuid.uuid4().hex[:8]}")
            self._tasks[task["uid"]] = task
            if prof:
                prof.prof("task_submit", uid=task["uid"])

        # No batching — submit immediately
        if self._batch_window <= 0:
            task_dicts = [dict(t) for t in tasks]
            await asyncio.to_thread(self._rh.submit_tasks, task_dicts)
            return

        # Batching — collect and, if the buffer is now full, flush inline.
        # The batch is detached *under* the lock but sent *outside* it, so a
        # slow network send never blocks other submitters from buffering.
        batch = None
        async with self._batch_lock:
            self._batch_buffer.extend(dict(t) for t in tasks)

            if len(self._batch_buffer) >= self._batch_limit:
                # Buffer full — take the batch now, send it below.
                batch = self._detach_batch()
            elif self._flush_handle is None:
                # Start the timer for the first task in this window
                loop = asyncio.get_running_loop()
                self._flush_handle = loop.call_later(self._batch_window, self._trigger_flush)

        if batch is not None:
            await self._send_batch(batch)

    def _trigger_flush(self):
        """Timer callback — schedule a background flush on the event loop.

        The task is tracked in ``_inflight_flushes`` so ``shutdown`` can await
        an in-progress send instead of dropping the batch.
        """
        task = asyncio.ensure_future(self._locked_flush())
        self._inflight_flushes.add(task)
        task.add_done_callback(self._on_flush_done)

    def _on_flush_done(self, task: asyncio.Task) -> None:
        """Untrack a finished background flush and retrieve its exception.

        ``_send_batch`` already logs the failure and marks tasks FAILED; we
        retrieve the exception here only so asyncio does not emit a spurious
        "Task exception was never retrieved" warning.
        """
        self._inflight_flushes.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                self.logger.debug("background batch flush failed: %s", exc)

    async def _locked_flush(self):
        """Detach the buffered batch under the batch lock, then send it unlocked."""
        async with self._batch_lock:
            batch = self._detach_batch()
        await self._send_batch(batch)

    def _detach_batch(self) -> list[dict]:
        """Swap out the current buffer and cancel the pending flush timer.

        Must be called while holding ``_batch_lock``.  Returns the detached
        batch (possibly empty).
        """
        batch = self._batch_buffer
        self._batch_buffer = []

        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None

        return batch

    async def _send_batch(self, batch: list[dict]) -> None:
        """Send an already-detached batch to the remote endpoint in one request.

        Runs *outside* ``_batch_lock`` (so other submitters can keep buffering
        during the network I/O) but *under* ``_send_lock`` (so concurrent
        flushes stay serialized and ordered, and never call the client
        concurrently).
        """
        if not batch:
            return

        # NOTE: compat is validated up front in submit_tasks(); buffered tasks
        # are already known-good by the time they reach the flush.

        prof = self._prof
        if prof:
            for t in batch:
                prof.prof("task_batch_flush", uid=t.get("uid", "?"))

        self.logger.debug("Flushing batch of %d tasks", len(batch))
        async with self._send_lock:
            try:
                await asyncio.to_thread(self._rh.submit_tasks, batch)
            except Exception as exc:
                # A timer-triggered flush runs as a detached background task; an
                # unhandled error here would leave the batch's tasks hung
                # forever.  Fail them explicitly so the failure reaches the
                # client (and re-raise for the inline buffer-full caller).
                self.logger.error("Failed to submit batch of %d tasks: %s", len(batch), exc)
                for t in batch:
                    task = self._tasks.get(t.get("uid"))
                    if task is not None:
                        task["exception"] = exc
                        task["state"] = "FAILED"
                        self._fire_callback(task, "FAILED")
                raise

    async def cancel_task(self, uid: str) -> bool:
        """Cancel a single task on the remote endpoint."""
        if not self._initialized:
            await self._async_init()

        if uid not in self._tasks:
            return False

        # If the task is still buffered locally (not yet flushed), the endpoint
        # never received it — drop it from the buffer and skip the remote
        # cancel (which would target an unknown uid).
        async with self._batch_lock:
            buffered = any(t.get("uid") == uid for t in self._batch_buffer)
            if buffered:
                self._batch_buffer = [t for t in self._batch_buffer if t.get("uid") != uid]

        if not buffered:
            await asyncio.to_thread(self._rh.cancel_task, uid)

        task = self._tasks[uid]
        task["state"] = "CANCELED"
        self._fire_callback(task, "CANCELED")
        return True

    async def cancel_all_tasks(self) -> int:
        """Cancel all non-terminal tasks on the remote endpoint.

        Mirrors ``cancel_task``: marks each non-terminal local task CANCELED
        and fires its state callback, so consumers that resolve futures via
        callbacks don't hang.  ``_fire_callback`` dedups against the matching
        CANCELED remote notification, so each task's terminal callback fires
        once.
        """
        if not self._initialized:
            await self._async_init()

        # Drop anything still buffered locally so it is not flushed and executed
        # after cancellation; the remote cancel_all only covers submitted tasks.
        async with self._batch_lock:
            self._batch_buffer = []
            if self._flush_handle is not None:
                self._flush_handle.cancel()
                self._flush_handle = None

        result = await asyncio.to_thread(self._rh.cancel_all_tasks)

        for task in list(self._tasks.values()):
            if str(task.get("state", "")).upper() not in self._TERMINAL_STATES:
                task["state"] = "CANCELED"
                self._fire_callback(task, "CANCELED")

        return result.get("canceled", 0)

    async def shutdown(self) -> None:
        """Flush pending tasks, close session and EndpointRuntime."""
        try:
            await self._locked_flush()
        except Exception as e:
            # Never let a flush failure skip client cleanup below.
            self.logger.warning("Failed to flush pending tasks during shutdown: %s", e)

        # Wait for any timer-triggered flush already in flight so its batch is
        # sent before we tear down the clients.
        if self._inflight_flushes:
            await asyncio.gather(*list(self._inflight_flushes), return_exceptions=True)

        self._backend_state = BackendMainStates.SHUTDOWN

        # close() calls perform blocking network I/O — run them off the loop.
        if self._rh:
            try:
                await asyncio.to_thread(self._rh.close)
            except Exception as e:
                self.logger.warning("Failed to close session: %s", e)
            self._rh = None

        # stop() joins the runtime's threads and does blocking teardown calls.
        if self._runtime:
            try:
                await asyncio.to_thread(self._runtime.stop)
            except Exception as e:
                self.logger.warning("Failed to stop endpoint runtime: %s", e)
            self._runtime = None

        if self._prof:
            self._prof.close()

        self.logger.info("Orbit execution backend shutdown complete")

    async def state(self) -> str:
        return self._backend_state.value

    def task_state_cb(self, task, state):
        pass

    def get_task_states_map(self):
        return StateMapper(backend=self)

    def build_task(self, uid, task_desc, task_specific_kwargs):
        pass

    def link_explicit_data_deps(self, src_task=None, dst_task=None, file_name=None, file_path=None):
        pass

    def link_implicit_data_deps(self, src_task, dst_task):
        pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self):
        if not self._initialized:
            await self._async_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
