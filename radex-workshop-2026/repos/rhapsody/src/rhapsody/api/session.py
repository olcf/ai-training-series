from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import TYPE_CHECKING
from typing import Callable

from rhapsody.api.errors import TaskExecutionError

if TYPE_CHECKING:
    from rhapsody.api.task import BaseTask
    from rhapsody.backends.base import BaseBackend
    from rhapsody.backends.data.base import DataBackend
    from rhapsody.telemetry.manager import TelemetryManager


logger = logging.getLogger(__name__)


class TaskStateManager:
    """Centralized manager for task state updates and monitoring.

    This class subscribes to backend callbacks and provides a synchronization mechanism for waiting
    on task completion.
    """

    def __init__(self):
        self._task_futures: dict[str, asyncio.Future] = {}
        self._terminal_states = set()  # Will be populated by backends
        self._loop: asyncio.AbstractEventLoop | None = None
        # Telemetry observer — set by Session.enable_telemetry(), None = zero cost
        self._telemetry_observer: Callable[[dict, str], None] | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def update_task(self, task: dict | BaseTask, state: str, **kwargs: object) -> None:
        """Update task state and notify waiters. Must be called from the event loop.

        Backends delivering results from a background thread must cross the thread boundary first
        via loop.call_soon_threadsafe(self.update_task, task, state).
        """
        uid = task["uid"]

        # Update the task object in-place (Single Source of Truth)
        task["state"] = state

        # Telemetry hook — O(1) None check, zero cost when not enabled
        if self._telemetry_observer is not None:
            self._telemetry_observer(task, state)

        # If terminal, notify waiters
        if state in self._terminal_states:
            if uid in self._task_futures:
                fut = self._task_futures.pop(uid)
                if not fut.done():
                    exc = task.get("exception")
                    exit_code = task.get("exit_code")
                    if isinstance(exc, BaseException):
                        fut.set_exception(exc)
                    elif exit_code is not None and exit_code != 0:
                        fut.set_exception(
                            TaskExecutionError(uid, task.get("stderr") or "", exit_code)
                        )
                    else:
                        fut.set_result(task)

    def get_wait_future(self, uid: str, task: dict | BaseTask) -> asyncio.Future:
        """Get or create a future to wait for a specific task."""
        if self._loop is None:
            try:
                self.bind_loop(asyncio.get_running_loop())
            except RuntimeError:
                pass

        if uid not in self._task_futures:
            # Create a future on the correct loop
            if self._loop:
                self._task_futures[uid] = self._loop.create_future()
            else:
                self._task_futures[uid] = asyncio.Future()

            # If already done before we started waiting, resolve immediately
            if task.get("state") in self._terminal_states:
                exc = task.get("exception")
                exit_code = task.get("exit_code")
                if isinstance(exc, BaseException):
                    self._task_futures[uid].set_exception(exc)
                elif exit_code is not None and exit_code != 0:
                    self._task_futures[uid].set_exception(
                        TaskExecutionError(uid, task.get("stderr") or "", exit_code)
                    )
                else:
                    self._task_futures[uid].set_result(task)

        return self._task_futures[uid]

    def set_terminal_states(self, states: set[str]) -> None:
        """Update the set of states considered terminal."""
        self._terminal_states = states


class Session:
    """Manages execution session, task submission, and monitoring.

    The Session acts as the central coordinator. It is initialized with a list of execution backends
    and manages the flow of tasks and their state updates.
    """

    def __init__(
        self,
        backends: list[BaseBackend | DataBackend] | None = None,
        uid: str | None = None,
        work_dir: str | None = None,
    ):
        """Initialize a new session.

        Args:
            backends: List of backends to use -- task-executing backends
                (ConcurrentExecutionBackend, DragonExecutionBackend, ...) and/or
                DataBackend instances (RedisDataBackend, DragonDataBackend, ...).
                If None, no backends are configured.
            uid: Optional unique identifier for the session.
            work_dir: working directory (default: cwd).
        """
        self.uid = uid or f"rhapsody.session.{uuid.uuid4().hex[:8]}"
        self.work_dir = work_dir or os.getcwd()
        self._tasks: dict[str, BaseTask | dict] = {}
        self._state_manager = TaskStateManager()
        self._telemetry: TelemetryManager | None = None
        self._resource_poll_interval: float = 5.0

        # Register callbacks with all provided backends
        backends_list = backends or []
        self.backends: dict[str, BaseBackend | DataBackend] = {}
        # Task-executing subset of self.backends, used for routing in
        # submit_tasks() -- kept separate so routing stays O(1) per task
        # regardless of how many DataBackend instances are also registered.
        self._exec_backends: dict[str, BaseBackend] = {}
        for backend in backends_list:
            self.add_backend(backend)

    def add_backend(self, backend: BaseBackend | DataBackend) -> None:
        """Add a backend to the session and register callbacks.

        Args:
            backend: The execution/inference backend, or DataBackend, to add.
        """
        self.backends[backend.name] = backend

        if not hasattr(backend, "submit_tasks"):
            # Infrastructure backend (e.g. DataBackend) -- no tasks, no
            # callbacks, no task-state map, and no Session-assigned
            # _work_dir: it resolves (and may already be using) its own
            # work_dir before ever reaching a Session, so stamping a fresh
            # rhapsody.session.<uid> directory here would just create an
            # empty, unused one. Registered only for inclusion in
            # Session.close()/telemetry.
            logger.debug(f"Registered data backend '{backend.name}' with Session '{self.uid}'")
            return

        backend._work_dir = os.path.join(self.work_dir, self.uid)
        os.makedirs(backend._work_dir, exist_ok=True)

        self._exec_backends[backend.name] = backend
        backend.is_attached = True
        backend.attached_to.append(self.uid)

        # Register state manager callback
        backend.register_callback(self._state_manager.update_task)

        logger.debug(f"Setting up backend callback for'{backend.name}' with Session '{self.uid}'")

        # Sync terminal states from backend
        if hasattr(backend, "get_task_states_map"):
            state_mapper = backend.get_task_states_map()
            self._state_manager._terminal_states.update(state_mapper.terminal_states)

        logger.debug(f"Registered backend '{backend.name}' with Session '{self.uid}'")

        # Register a telemetry adapter if telemetry is already enabled
        if self._telemetry is not None:
            self._attach_telemetry_adapter(backend)

    async def submit_tasks(self, tasks: list[dict | BaseTask]) -> list[asyncio.Future]:
        """Submit tasks to execution backends and return futures.

        Args:
            tasks: List of tasks to submit.

        Returns:
            List of asyncio.Future objects representing task lifecycles.
        """
        # Ensure we have a bound loop for callbacks
        if not self._state_manager._loop:
            try:
                self._state_manager.bind_loop(asyncio.get_running_loop())
            except RuntimeError:
                pass
        if not self._exec_backends:
            raise RuntimeError("No task-executing backend configured in Session")

        # Group tasks by their explicit backend target
        tasks_by_backend: dict[str, list] = {}
        futures = []
        for task in tasks:
            uid = task["uid"]
            self._tasks[uid] = task

            # Create and bind future
            fut = self._state_manager.get_wait_future(uid, task)
            if hasattr(task, "bind_future"):
                task.bind_future(fut)
            futures.append(fut)

            # Stamp task_type for telemetry event classification
            task["task_type"] = type(task).__name__

            # TaskCreated: task is registered and future is bound; backend not yet assigned.
            if self._telemetry is not None:
                self._telemetry._on_task_created(task)

            # Routing decision
            target_name = task.get("backend")
            if not target_name:
                # If no backend specified, use the first task-executing one as
                # default (DataBackend instances registered in the same
                # Session are never eligible here).
                target_name = next(iter(self._exec_backends))
                task["backend"] = target_name  # Ensure it's recorded

            # Emit TaskSubmitted AFTER routing so task["backend"] is always set.
            if self._telemetry is not None:
                self._telemetry._on_task_submitted(task)

            if target_name not in self._exec_backends:
                available = list(self._exec_backends.keys())
                raise ValueError(
                    f"Backend '{target_name}' requested by task {uid} not found in Session. "
                    f"Available backends: {available}"
                )

            tasks_by_backend.setdefault(target_name, []).append(task)

        # Submit each group to its respective backend concurrently
        submission_tasks = []
        for name, backend_tasks in tasks_by_backend.items():
            backend = self._exec_backends[name]
            # Emit TaskQueued at the backend boundary (after routing, before execution)
            if self._telemetry is not None:
                for task in backend_tasks:
                    self._telemetry._on_task_queued(task)
            submission_tasks.append(backend.submit_tasks(backend_tasks))

        if submission_tasks:
            await asyncio.gather(*submission_tasks)

        logger.info(f"Successfully submitted {len(tasks)} tasks")

        return futures

    async def wait_tasks(
        self,
        tasks: list[dict | BaseTask],
        timeout: float | None = None,
    ) -> list[dict | BaseTask]:
        """Wait for tasks to reach a terminal state.

        Args:
            tasks: List of tasks to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            The list of completed task objects.

        Raises:
            asyncio.TimeoutError: If timeout is reached.
        """
        if not tasks:
            return []

        futures = []
        for task in tasks:
            uid = task["uid"]
            futures.append(self._state_manager.get_wait_future(uid, task))

        # Wait for all futures; return_exceptions=True prevents task failures from
        # propagating — callers inspect task.state / task.exception directly.
        try:
            await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Check how many finished
            finished = sum(
                1 for t in tasks if t.get("state") in self._state_manager._terminal_states
            )
            raise asyncio.TimeoutError(
                f"Timeout after {timeout}s: {finished}/{len(tasks)} tasks completed"
            )

        return tasks

    async def start_telemetry(
        self,
        resource_poll_interval: float = 5.0,
        checkpoint_interval: float | None = None,
        checkpoint_path: str | None = None,
        span_processors: list | None = None,
        metric_readers: list | None = None,
        resource: object | None = None,
    ) -> TelemetryManager:
        """Enable and start telemetry collection for this session in one call.

        Creates a :class:`~rhapsody.telemetry.manager.TelemetryManager`, wires it
        into the task state manager, registers backend-specific adapters, and starts
        the async dispatch loop — all in a single ``await``.

        Telemetry stops automatically when :meth:`close` is called. There is no need
        to call ``stop()`` on the returned manager separately.

        Must be called **before** ``await session.submit_tasks()``.

        Args:
            resource_poll_interval: Seconds between resource metric polls (default: 5.0).
            checkpoint_interval:    Seconds between metric+span flushes to disk.
                                    None = no periodic flush (file still written at stop).
            checkpoint_path:        Directory for the JSONL checkpoint file.
                                    None = no file output.
            span_processors:        Optional list of
                                    ``opentelemetry.sdk.trace.SpanProcessor`` instances
                                    (e.g. ``BatchSpanProcessor(OTLPSpanExporter())``)
                                    added to RHAPSODY's TracerProvider alongside the
                                    internal SpanBuffer. Callers own exporter construction
                                    and configuration.
            metric_readers:         Optional list of
                                    ``opentelemetry.sdk.metrics.export.MetricReader``
                                    instances (e.g.
                                    ``PeriodicExportingMetricReader(OTLPMetricExporter())``)
                                    added to RHAPSODY's MeterProvider alongside the
                                    internal InMemoryMetricReader.
            resource:               Optional ``opentelemetry.sdk.resources.Resource``.
                                    When None (default), ``Resource.create()`` is called
                                    automatically and reads ``OTEL_SERVICE_NAME`` /
                                    ``OTEL_RESOURCE_ATTRIBUTES`` from the environment.

        Returns:
            The active :class:`~rhapsody.telemetry.manager.TelemetryManager`.
            Use it for optional advanced operations: ``subscribe()``, ``summary()``,
            ``task_spans()``, ``read_metrics()``, ``read_traces()``.
            Or retrieve it later via :meth:`get_telemetry`.
        """
        from rhapsody.telemetry.manager import TelemetryManager  # deferred import

        self._resource_poll_interval = resource_poll_interval
        self._telemetry = TelemetryManager(
            session_id=self.uid,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
            span_processors=span_processors,
            metric_readers=metric_readers,
            resource=resource,
        )
        self._state_manager._telemetry_observer = self._telemetry._on_task_state_change

        for backend in self.backends.values():
            self._telemetry.attach_backend(
                backend,
                session_id=self.uid,
                backend_name=backend.name,
                interval=resource_poll_interval,
            )

        await self._telemetry.start()
        return self._telemetry

    def get_telemetry(self) -> TelemetryManager:
        """Return the active TelemetryManager.

        Raises:
            RuntimeError: If telemetry has not been started via :meth:`start_telemetry`.
        """
        if self._telemetry is None:
            raise RuntimeError(
                "Telemetry not started. Call `await session.start_telemetry()` first."
            )
        return self._telemetry

    async def close(self) -> None:
        """Shutdown telemetry (if enabled) then all backends."""
        if self._telemetry is not None:
            await self._telemetry.stop()
            self._telemetry = None
        for backend in self.backends.values():
            await backend.shutdown()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
