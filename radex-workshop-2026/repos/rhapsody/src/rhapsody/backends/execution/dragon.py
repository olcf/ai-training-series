import asyncio
import logging
import os
import shutil
import threading
import time
from typing import Any
from typing import Callable
from typing import Optional

from ..base import BaseBackend
from ..constants import BackendMainStates
from ..constants import StateMapper

DRAGON_BATCH_INIT_ERROR = None

try:
    import dragon
    from dragon.native.process import ProcessTemplate
    from dragon.workflows.batch import Batch
    from dragon.workflows.batch import TaskCancelledError

except ImportError as e:  # pragma: no cover - environment without Dragon
    dragon = None
    ProcessTemplate = None
    Batch = None
    TaskCancelledError = None
    TaskNotReadyError = None
    DRAGON_BATCH_INIT_ERROR = e


def _get_logger() -> logging.Logger:
    """Get logger for dragon backend module.

    This function provides lazy logger evaluation, ensuring the logger is created after the user has
    configured logging, not at module import time.
    """
    return logging.getLogger(__name__)


# ============================================================================
# Backend Helper Classes
# ============================================================================


class TaskStateMapper:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    terminal_states = {DONE, FAILED, CANCELED}


# ============================================================================
# Main Backend Class — Dragon.Batch API (event-driven, HPC-scale)
# ============================================================================


class DragonExecutionBackend(BaseBackend):
    """Dragon Batch backend using the streaming pipeline model.

    Tasks submitted via batch.function()/process()/job() are auto-dispatched by the Batch
    background thread. A single monitor thread uses ``Batch.poll()`` for event-driven result
    delivery — zero busy-wait, zero private DDict access.

    Args:
        batch_kwargs: Forwarded verbatim to ``dragon.workflows.batch.Batch()``.

            Supported keys (Dragon 0.14.1+):

            - ``num_nodes`` *(int, optional)* — nodes to use; defaults to the full
              allocation.
            - ``disable_telem`` *(bool)* — disable Dragon's internal telemetry
              (default ``False``).
            - ``scheduler_workers`` *(int, optional)* — size of the scheduler's local
              worker pool; defaults to ``num_nodes``. Increase this when running many
              concurrent multi-node (MPI) jobs.
            - ``results_ddict_mem`` *(int, optional)* — bytes to allocate for the
              results DDict (default: 1 GiB × num_nodes). Increase for workloads that
              return large arrays or submit millions of tasks.
            - ``results_ddict_managers_per_pool`` *(int, optional)* — DDict shard count
              per worker pool (default: 4). At large scale (64K+ tasks), increasing this
              reduces DDict write contention. Valid range: ``[1, workers_per_node]``;
              Dragon clamps automatically.
            - ``pool_nodes`` — **no-op in Dragon 0.14.0+**, kept for API compatibility.

    Note on working directory:
        DragonExecutionBackend does not support a backend-level working directory.
        Set it per task via ``task_backend_specific_kwargs`` with ``process_template``
        (single process) or ``process_templates`` (MPI job)::

            ComputeTask(
                function=my_func,
                task_backend_specific_kwargs={
                    "process_template": {"cwd": "/path/to/dir"}
                },
            )

            # For MPI jobs:
            ComputeTask(
                function=my_func,
                task_backend_specific_kwargs={
                    "process_templates": [(nranks, {"cwd": "/path/to/dir"})]
                },
            )
    """

    # Hygiene backstop: max wall-clock seconds a tuid can stay pending before it's dropped.
    PENDING_TUID_TIMEOUT = 30.0

    def __init__(
        self,
        batch_kwargs: Optional[dict] = None,
        name: Optional[str] = "dragon",
    ):
        if not Batch:
            raise RuntimeError(DRAGON_BATCH_INIT_ERROR)

        super().__init__(name=name)

        self.logger = _get_logger()
        effective_kwargs = {"task_logs": True, **(batch_kwargs or {})}
        if not effective_kwargs.get("task_logs", True):
            self.logger.warning(
                "task_logs is disabled: task['stdout']/task['stderr'] will be empty "
                "for capture_stdio=False tasks; output goes directly to the console."
            )
        self.batch = Batch(**effective_kwargs)

        self._backend_state = BackendMainStates.INITIALIZED
        self._callback_func: Callable = lambda t, s: None
        self._task_registry: dict[str, Any] = {}
        self._task_states = TaskStateMapper()
        self._initialized = False
        self._cancelled_tasks: set[str] = set()
        self._monitored_batches = {}
        # Tuids polled before registration landed; retried each sweep. tuid -> first-seen time.
        self._pending_tuids: dict[str, float] = {}
        # Dragon tuids confirmed cancelled, so _monitor_loop discards them instead of retrying.
        self._cancelled_tuids: set[str] = set()
        self._batch_monitor_thread = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._shutdown_event = threading.Event()

        self.logger.info(
            f"DragonExecutionBackend: {self.batch.num_workers} workers, "
            f"{self.batch.num_managers} managers"
        )

    def __await__(self):
        return self._async_init().__await__()

    async def _async_init(self):
        """Unified async initialization with backend and task state registration.

        Pattern:
        1. Register backend states first
        2. Register task states
        3. Set backend state to INITIALIZED
        4. Initialize backend components (if needed)
        """
        if not self._initialized:
            try:
                self.logger.debug("Starting Dragon backend async initialization...")

                # Step 1: Register backend states
                self.logger.debug("Registering backend states...")
                StateMapper.register_backend_states_with_defaults(backend=self)

                # Step 2: Register task states
                self.logger.debug("Registering task states...")
                StateMapper.register_backend_tasks_states_with_defaults(backend=self)

                self._initialized = True
                self.logger.info("Dragon backend fully initialized and ready")

            except Exception as e:
                self.logger.exception(f"Dragon backend initialization failed: {e}")
                self._initialized = False
                raise
        return self

    def _monitor_loop(self) -> None:
        """Single thread to monitor all active tasks via event-driven Batch.poll().

        Blocks on poll() at the OS queue level — zero CPU when idle. Drains all simultaneously
        available completions before waking the asyncio event loop, preserving O(sweeps) cross-
        thread wakeups instead of O(tasks).
        """
        self.logger.debug("Starting Dragon batch monitor loop (event-driven)")

        while not self._shutdown_event.is_set() or self._monitored_batches or self._pending_tuids:
            try:
                tuid = self.batch.poll(timeout=0.05)

                tuids = []
                if tuid is not None:
                    tuids.append(tuid)
                    # Drain all already-queued completions in one pass
                    while True:
                        nxt = self.batch.poll(timeout=0)
                        if nxt is None:
                            break
                        tuids.append(nxt)

                # Retry tuids not yet registered when last seen.
                if self._pending_tuids:
                    tuids = list(self._pending_tuids.keys()) + tuids

                if not tuids:
                    continue

                completed = []
                for t in tuids:
                    entry = self._monitored_batches.pop(t, None)
                    if entry is None:
                        if t in self._cancelled_tuids:
                            self._cancelled_tuids.discard(t)
                            self._pending_tuids.pop(t, None)
                            continue
                        # Not registered yet; keep retrying instead of assuming cancellation.
                        first_seen = self._pending_tuids.setdefault(t, time.monotonic())
                        if time.monotonic() - first_seen > self.PENDING_TUID_TIMEOUT:
                            self.logger.warning(f"Dragon tuid {t} never registered; dropping")
                            self._pending_tuids.pop(t, None)
                        continue
                    self._pending_tuids.pop(t, None)
                    batch_task, uid = entry

                    try:
                        # Block until the result is fully committed to the DDict.
                        # poll() and the DDict write are not atomic; block=True here
                        # ensures get_stdout(block=False) is safe afterward.
                        result = batch_task.get(block=True)
                        raised, tb = False, None
                    except TaskCancelledError:
                        # Dragon-side timeout / upstream cancel (not a user cancel_task call)
                        task_info = self._task_registry.pop(uid, None)
                        if task_info:
                            self._callback_func(task_info["description"], "CANCELED")
                        continue
                    except Exception as exc:
                        result, raised, tb = exc, True, batch_task.traceback

                    task_reg = self._task_registry.get(uid)
                    is_native_function = (
                        task_reg.get("is_native_function", False) if task_reg else False
                    )
                    if not is_native_function:
                        # Dragon returns exit codes as int/list[int] for process/job tasks.
                        # Normalize non-zero to raised=True so Rhapsody delivers FAILED.
                        if isinstance(result, int) and result != 0:
                            result = SystemExit(result)
                            raised = True
                        elif isinstance(result, list) and any(
                            c != 0 for c in result if isinstance(c, int)
                        ):
                            result = SystemExit(result)
                            raised = True

                    capture_stdio = (
                        task_reg.get("description", {}).get("capture_stdio") if task_reg else False
                    )
                    if capture_stdio:
                        stdout = batch_task.stdout_path or ""
                        stderr = batch_task.stderr_path or ""
                    else:
                        try:
                            stdout = batch_task.get_stdout(block=False) or ""
                            stderr = batch_task.get_stderr(block=False) or ""
                        except OSError:
                            stdout, stderr = "", ""
                    completed.append((uid, result, tb, raised, stdout, stderr))

                if completed:
                    self._loop.call_soon_threadsafe(self._deliver_batch, completed)

            except Exception as e:
                self.logger.exception(f"Critical error in monitor loop: {e}")

        self.logger.debug("Dragon batch monitor loop stopped")

    def _deliver_batch(self, completions: list) -> None:
        """Deliver a batch of completed tasks. Runs on the asyncio event loop (via
        call_soon_threadsafe).

        Called once per monitor drain with all tasks that completed in that sweep, reducing cross-
        thread wakeups from O(tasks) to O(sweeps).
        """
        for uid, result, tb, raised, stdout, stderr in completions:
            task_info = self._task_registry.pop(uid, None)
            if not task_info:
                continue
            if uid in self._cancelled_tasks:
                self._cancelled_tasks.discard(uid)
                continue
            task_desc = task_info["description"]
            if raised:
                task_desc["exception"] = result
                task_desc["stderr"] = stderr if stderr else (tb if tb else str(result))
                task_desc["stdout"] = stdout
                self._callback_func(task_desc, "FAILED")
            else:
                task_desc["return_value"] = result
                task_desc["stdout"] = stdout
                task_desc["stderr"] = stderr
                self._callback_func(task_desc, "DONE")

    async def submit_tasks(self, tasks: list[dict]) -> None:
        """Submit tasks to the backend.

        :param tasks: List of task descriptions
        """
        if self._backend_state == BackendMainStates.SHUTDOWN:
            raise RuntimeError("Cannot submit during shutdown")

        # Set backend state to RUNNING when tasks are submitted
        if self._backend_state != BackendMainStates.RUNNING:
            self._backend_state = BackendMainStates.RUNNING
            self.logger.debug(f"Backend state set to: {self._backend_state.value}")

        # Capture event loop for cross-thread batch delivery
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        # Start monitor thread on first submission (lazy — avoids idle spin before tasks exist)
        if self._batch_monitor_thread is None or not self._batch_monitor_thread.is_alive():
            self._shutdown_event.clear()
            self._batch_monitor_thread = threading.Thread(
                target=self._monitor_loop, name="dragon_monitor_loop", daemon=True
            )
            self._batch_monitor_thread.start()

        # Build tasks
        batch_tasks_data = []
        for task in tasks:
            try:
                batch_task = await self.build_task(task)
                batch_tasks_data.append((task["uid"], batch_task))
                # This is the moment Dragon takes ownership was called
                # inside build_task) and start executing it.
                self._callback_func(task, "RUNNING")
            except Exception as e:
                self.logger.error(f"Failed to create task {task.get('uid')}: {e}", exc_info=True)
                task["exception"] = e
                self._callback_func(task, "FAILED")

        if not batch_tasks_data:
            return

        # Tasks are already in-flight and registered for result monitoring —
        # build_task() registers each one into _monitored_batches immediately
        # after Dragon dispatch, not in a second pass here (a fast task can
        # otherwise complete and be polled before this loop even finishes).
        self.logger.info(f"Submitted {len(batch_tasks_data)} tasks (streaming, auto-dispatched)")

    async def build_task(self, task: dict):
        """Translate AsyncFlow task to Dragon Batch task.

        Translation Priority (in order):
        1. If process_templates (list) provided → Job mode [function/executable]
        2. If process_template (single) provided → Process mode [function/executable]
        3. If is_function (no templates) → Function mode (native) [function only]
        4. If is_executable (no templates) → Process mode (auto-build) [executable only]

        Execution Modes:
        - Function Native: batch.function() - direct Python function call
        - Function Process: batch.process() - function wrapped in ProcessTemplate
        - Executable Process: batch.process() - single executable process
        - Executable Job: batch.job() - executable in MPI job via process_templates

        Setting cwd (working directory):
            Pass ``cwd`` inside ``process_template`` or each entry of ``process_templates``
            via ``task_backend_specific_kwargs``::

                ComputeTask(
                    function=my_func,
                    task_backend_specific_kwargs={
                        "process_template": {"cwd": "/path/to/dir"}
                    },
                )
        """
        # Fast path: extract everything upfront
        uid = task["uid"]
        is_function = bool(task.get("function"))
        target = task.get("function" if is_function else "executable")
        backend_kwargs = task.get("task_backend_specific_kwargs", {})
        name = task.get("name", uid)

        # For functions: use "args" and "kwargs"
        # For executables: use "arguments"
        if is_function:
            task_args = task.get("args", [])
            task_kwargs = task.get("kwargs", {})
        else:
            task_args = tuple(task.get("arguments", []))
            task_kwargs = None

        timeout = backend_kwargs.get("timeout", 1000000000.0)

        # Handle async functions
        if is_function and asyncio.iscoroutinefunction(target):
            original_target = target

            def target(*a, **kw):
                return asyncio.run(original_target(*a, **kw))

        # Get template configs once
        process_templates_config = backend_kwargs.get("process_templates")
        process_template_config = backend_kwargs.get("process_template")
        is_native_function = (
            is_function and process_templates_config is None and process_template_config is None
        )

        # Compute per-task stdio paths when capture is requested.
        # Dragon creates parent directories automatically.
        # None = forward output to console (no file capture).
        stdout_path = stderr_path = None
        if task.get("capture_stdio"):
            stdout_path = os.path.join(self._work_dir, f"{uid}.stdout")
            stderr_path = os.path.join(self._work_dir, f"{uid}.stderr")

        def _build_process_template_kwargs(template_cfg: dict[str, Any]) -> dict[str, Any]:
            """Build ProcessTemplate kwargs from user config."""
            return {**template_cfg, "args": task_args, "kwargs": task_kwargs}

        # Build task options once and reuse across all submission paths.
        # Dragon 0.14.1-rc requires metadata (name, timeout, stdio paths) to go through
        # Batch.options() rather than as direct kwargs to process/job/function.
        task_opts = self.batch.options(
            name=name,
            timeout=timeout,
            stdout=stdout_path,
            stderr=stderr_path,
        )

        # Single decision tree - no redundant checks
        if process_templates_config is not None:
            # Priority 1: Job with user templates
            process_templates = [
                (nranks, ProcessTemplate(target, **_build_process_template_kwargs(tc)))
                for nranks, tc in process_templates_config
            ]
            batch_task = task_opts.job(process_templates)

        elif process_template_config is not None:
            # Priority 2: Process with user template
            batch_task = task_opts.process(
                ProcessTemplate(target, **_build_process_template_kwargs(process_template_config))
            )

        elif is_function:
            # Priority 3: Function native (batch.function — returns Python value, not exit code)
            batch_task = task_opts.function(target, *task_args, **task_kwargs)

        else:
            # Priority 4: Executable process auto-build
            batch_task = task_opts.process(
                ProcessTemplate(target, **_build_process_template_kwargs({}))
            )

        # Register and return. Write _task_registry before _monitored_batches (order matters).
        self._task_registry[uid] = {
            "uid": uid,
            "description": task,
            "batch_task": batch_task,
            "is_native_function": is_native_function,
        }
        self._monitored_batches[batch_task.uid] = (batch_task, uid)

        self.logger.debug(
            f"Created {'function' if is_native_function else 'process/job'} task: {uid}"
        )

        return batch_task

    def link_implicit_data_deps(self, src_task, dst_task):
        pass

    async def state(self) -> str:
        """Get backend state.

        Returns:
            str: Current backend state (INITIALIZED, RUNNING, SHUTDOWN)
        """
        return self._backend_state.value

    def link_explicit_data_deps(self, src_task=None, dst_task=None, file_name=None, file_path=None):
        pass

    def task_state_cb(self, task: dict, state: str) -> None:
        self._callback_func(task, state)

    def get_task_states_map(self):
        return self._task_states

    async def cancel_task(self, uid: str) -> bool:
        if uid not in self._task_registry:
            raise ValueError(f"Task {uid} not found")

        batch_task = self._task_registry[uid]["batch_task"]
        loop = asyncio.get_running_loop()

        try:
            cancelled = await loop.run_in_executor(None, batch_task.cancel)
        except Exception as e:
            self.logger.warning(f"Dragon cancel failed for {uid}: {e}; falling back to soft-cancel")
            cancelled = True

        if cancelled:
            registry_entry = self._task_registry.pop(uid, None)
            if registry_entry is None:
                # Task completed naturally between the executor call and here — not cancelled.
                return False
            task_desc = registry_entry["description"]
            # Eagerly pop from _monitored_batches so when the cancelled tuid arrives from
            # poll(), the None entry is skipped without delivering a spurious result.
            # Mark cancelled so _monitor_loop drops it instead of retrying it forever.
            self._monitored_batches.pop(batch_task.uid, None)
            self._cancelled_tuids.add(batch_task.uid)
            self._callback_func(task_desc, "CANCELED")

        return cancelled

    async def shutdown(self) -> None:
        """Shutdown the backend and clean up resources."""
        if self._backend_state == BackendMainStates.SHUTDOWN:
            return

        # Set backend state to SHUTDOWN
        self._backend_state = BackendMainStates.SHUTDOWN
        self.logger.debug(f"Backend state set to: {self._backend_state.value}")
        self.logger.info("Shutting down Dragon backend")
        self._shutdown_event.set()

        # Wait for monitor thread to finish if it exists
        if self._batch_monitor_thread and self._batch_monitor_thread.is_alive():
            try:
                self.logger.debug("Waiting for batch monitor thread to stop...")
                self._batch_monitor_thread.join(timeout=5.0)
                if self._batch_monitor_thread.is_alive():
                    self.logger.warning("Batch monitor thread did not stop within timeout")
            except Exception as e:
                self.logger.exception(f"Error stopping monitor thread: {e}")

        # Close Batch
        if self.batch:
            try:
                self.logger.debug("Joining batch...")
                self.batch.join(timeout=10.0)
                self.logger.debug("Destroying batch...")
                self.batch.destroy(timeout=15.0)
                self.logger.debug("Batch destroyed successfully")
            except Exception as e:
                self.logger.warning(f"Error closing batch gracefully: {e}")
                try:
                    self.logger.debug("Attempting to terminate batch...")
                    self.batch.terminate()
                except Exception as te:
                    self.logger.warning(f"Error terminating batch: {te}")

            # Remove Dragon's task_logs directory
            try:
                shutil.rmtree(self.batch.log_dir(), ignore_errors=True)
            except RuntimeError:
                pass  # task_logs was explicitly disabled; nothing to remove
            except Exception as e:
                self.logger.warning(f"Failed to remove Dragon task_logs directory: {e}")

        self._task_registry.clear()
        self._pending_tuids.clear()
        self._cancelled_tuids.clear()
        self._state = "idle"
        self.logger.info("Dragon backend shutdown complete")

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for all submitted tasks to complete.

        Delegates to Dragon's Batch.fence(). Tasks submitted before this call
        will all complete before any tasks submitted after it are started.

        Args:
            timeout: Maximum seconds to wait. Uses Dragon's default if not set.
        """
        if timeout is not None:
            self.batch.fence(timeout=timeout)
        else:
            self.batch.fence()

    def create_ddict(self, *args, **kwargs):
        from dragon.data.ddict.ddict import DDict

        return DDict(*args, **kwargs)

    @classmethod
    async def create(
        cls,
        batch_kwargs: Optional[dict] = None,
    ):
        """Create and initialize a DragonExecutionBackend."""
        backend = cls(batch_kwargs=batch_kwargs)
        return await backend


def __getattr__(name: str):
    import warnings

    if name == "DragonExecutionBackendV3":
        warnings.warn(
            "DragonExecutionBackendV3 is deprecated and will be removed in a future release. "
            "Use DragonExecutionBackend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DragonExecutionBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
