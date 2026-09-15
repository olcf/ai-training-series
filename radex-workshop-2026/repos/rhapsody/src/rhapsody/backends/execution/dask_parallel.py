"""Dask distributed execution backend for parallel and distributed computing.

This module provides a backend that executes tasks on Dask clusters, supporting both local and
distributed execution environments.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from functools import wraps
from typing import Any
from typing import Callable

from ..base import BaseBackend
from ..constants import BackendMainStates
from ..constants import StateMapper

try:
    import dask.distributed as dask
except ImportError:
    dask = None


def _get_logger() -> logging.Logger:
    """Get logger for dask backend module.

    This function provides lazy logger evaluation, ensuring the logger is created after the user has
    configured logging, not at module import time.
    """
    return logging.getLogger(__name__)


def _run_executable(
    executable: str,
    arguments: list[str],
    cwd: str | None = None,
    env: dict | None = None,
    shell: bool = False,
    capture_stdio: bool = False,
    output_dir: str | None = None,
    uid: str | None = None,
) -> tuple[str, str, int]:
    """Run a subprocess inside a Dask worker.

    Must be defined at module level for Dask pickling compatibility.

    Args:
        executable: Path to the executable.
        arguments: List of command-line arguments.
        cwd: Working directory for the subprocess.
        env: Environment variables dict. None inherits the worker environment.
        shell: Whether to execute through the shell.
        capture_stdio: If True, redirect stdout/stderr to files in output_dir.
        output_dir: Directory for output files when capture_stdio is True.
        uid: Task uid used to name output files when capture_stdio is True.

    Returns:
        Tuple of (stdout, stderr, returncode). When capture_stdio is True,
        stdout and stderr are file paths instead of decoded content.
    """
    import subprocess

    cmd = [executable] + list(arguments)
    shell_cmd = " ".join(cmd) if shell else cmd

    if capture_stdio and output_dir and uid:
        import os as _os

        stdout_path = _os.path.join(output_dir, f"{uid}.stdout")
        stderr_path = _os.path.join(output_dir, f"{uid}.stderr")
        with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
            result = subprocess.run(
                shell_cmd, shell=shell, stdout=out_f, stderr=err_f, cwd=cwd, env=env
            )
        return stdout_path, stderr_path, result.returncode
    else:
        result = subprocess.run(shell_cmd, shell=shell, capture_output=True, cwd=cwd, env=env)
        return result.stdout.decode(), result.stderr.decode(), result.returncode


class DaskExecutionBackend(BaseBackend):
    """A Dask execution backend for distributed task execution.

    Handles task submission, cancellation, and proper async event loop handling
    for distributed task execution using Dask. Supports async functions, sync
    functions, and executable tasks.

    Usage:
        backend = await DaskExecutionBackend(resources)
        # or
        async with DaskExecutionBackend(resources) as backend:
            await backend.submit_tasks(tasks)
    """

    def __init__(
        self,
        resources: dict | None = None,
        name: str = "dask",
        cluster: Any | None = None,
        client: Any | None = None,
    ):
        """Initialize the Dask execution backend (non-async setup only).

        Args:
            resources: Dictionary of resource requirements for tasks. Contains
                configuration parameters for the Dask client initialization.
            name: Name of the backend.
            cluster: Optional preconfigured Dask Cluster object.
            client: Optional preconfigured Dask Client object.
        """

        if dask is None:
            raise ImportError("Dask is required for DaskExecutionBackend.")

        super().__init__(name=name)

        self.logger = _get_logger()
        self.tasks = {}
        self._client = None
        self._callback_func: Callable = lambda t, s: None
        self._resources = resources or {}
        self._cluster_provided = cluster
        self._client_provided = client
        self._initialized = False
        self._backend_state = BackendMainStates.INITIALIZED

    def __await__(self):
        """Make DaskExecutionBackend awaitable like Dask Client."""
        return self._async_init().__await__()

    async def _async_init(self):
        """Unified async initialization with backend and task state registration.

        Pattern:
        1. Register backend states first
        2. Register task states
        3. Set backend state to INITIALIZED
        4. Initialize backend components
        """
        if not self._initialized:
            try:
                # Step 1: Register backend states
                self.logger.debug("Registering backend states...")
                StateMapper.register_backend_states_with_defaults(backend=self)

                # Step 2: Register task states
                self.logger.debug("Registering task states...")
                StateMapper.register_backend_tasks_states_with_defaults(backend=self)

                # Step 3: Set backend state to INITIALIZED
                self._backend_state = BackendMainStates.INITIALIZED
                self.logger.debug(f"Backend state set to: {self._backend_state.value}")

                # Step 4: Initialize backend components
                await self._initialize()
                self._initialized = True

                self.logger.info("Dask backend fully initialized and ready")

            except Exception as e:
                self.logger.exception(f"Dask backend initialization failed: {e}")
                self._initialized = False
                raise
        return self

    async def _initialize(self) -> None:
        """Initialize the Dask client and set up worker environments.

        Raises:
            Exception: If Dask client initialization fails.
        """
        try:
            if self._client_provided:
                self._client = self._client_provided
            elif self._cluster_provided:
                self._client = await dask.Client(
                    self._cluster_provided, asynchronous=True, **self._resources
                )
            else:
                self._client = await dask.Client(asynchronous=True, **self._resources)

            dashboard_link = self._client.dashboard_link
            self.logger.info(f"Dask backend initialized with dashboard at {dashboard_link}")
        except Exception as e:
            self.logger.exception(f"Failed to initialize Dask client: {str(e)}")
            raise

    def register_callback(self, func: Callable) -> None:
        """Register a callback for task state changes.

        Args:
            func: Function to be called when task states change. Should accept
                task and state parameters.
        """
        self._callback_func = func

    def get_task_states_map(self) -> StateMapper:
        """Retrieve a mapping of task IDs to their current states.

        Returns:
            StateMapper: Object containing the mapping of task states for this backend.
        """
        return StateMapper(backend=self)

    async def cancel_task(self, uid: str) -> bool:
        """Cancel a task by its UID.

        Args:
            uid (str): The UID of the task to cancel.

        Returns:
            bool: True if the task was found and cancellation was attempted,
            False otherwise.
        """
        self._ensure_initialized()
        if uid in self.tasks:
            task = self.tasks[uid]
            future = task.get("future")
            if future:
                await future.cancel()
                return True
        return False

    async def submit_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Submit tasks to the Dask cluster.

        Dispatches each task to the appropriate submission method based on its type:
        executable tasks run via subprocess, async functions via an async wrapper,
        and sync functions are submitted directly to Dask workers.

        Args:
            tasks: List of task dictionaries containing:
                - uid: Unique task identifier
                - function: Callable to execute (sync or async)
                - args: Positional arguments
                - kwargs: Keyword arguments
                - executable: Path to executable (mutually exclusive with function)
                - arguments: CLI arguments for executable tasks
                - task_backend_specific_kwargs: Passed directly to client.submit()

        Note:
            Future objects are filtered out from args as they are not picklable.
        """
        self._ensure_initialized()

        if self._backend_state != BackendMainStates.RUNNING:
            self._backend_state = BackendMainStates.RUNNING
            self.logger.debug(f"Backend state set to: {self._backend_state.value}")

        for task in tasks:
            is_func_task = bool(task.get("function"))
            is_exec_task = bool(task.get("executable"))

            self.tasks[task["uid"]] = task

            # Filter out future objects as they are not picklable for Dask workers
            filtered_args = [
                arg for arg in task.get("args", ()) if not isinstance(arg, asyncio.Future)
            ]
            task["args"] = tuple(filtered_args)

            try:
                if is_exec_task:
                    await self._submit_executable(task)
                elif is_func_task and asyncio.iscoroutinefunction(task["function"]):
                    await self._submit_async_function(task)
                elif is_func_task:
                    await self._submit_sync_function(task)
                else:
                    raise ValueError("Task must specify either 'function' or 'executable'")
            except Exception as e:
                task["exception"] = e
                task["stdout"] = ""
                task["stderr"] = str(e)
                self._callback_func(task, "FAILED")

    async def _submit_to_dask(self, task: dict[str, Any], fn: Callable, *args) -> None:
        """Submit function to Dask and register completion callback.

        Submits the wrapped function to Dask client and registers a callback
        to handle task completion or failure.

        Args:
            task: Task dictionary containing task metadata and configuration.
            fn: The async function to submit to Dask.
            *args: Arguments to pass to the function.
        """

        async def on_done(f):
            task_uid = task["uid"]
            try:
                self._callback_func(task, "RUNNING")
                result = await f
                task["return_value"] = result
                task["stdout"] = ""
                task["stderr"] = ""
                self._callback_func(task, "DONE")
            except dask.client.FutureCancelledError:
                self._callback_func(task, "CANCELED")
            except Exception as e:
                task["exception"] = e
                task["stdout"] = ""
                task["stderr"] = str(e)
                self._callback_func(task, "FAILED")
            finally:
                # Clean up the future reference once task is complete
                if task_uid in self.tasks:
                    del self.tasks[task_uid]

        backend_kwargs = dict(task.get("task_backend_specific_kwargs", {}))
        dask_resources = backend_kwargs.get("resources", {})
        if dask_resources and not self._check_resources_satisfiable(dask_resources):
            task["exception"] = RuntimeError(
                f"No worker can satisfy resources {dask_resources}. "
                f"Workers must be started with matching --resources flags "
                f'(e.g. dask worker <scheduler> --resources "GPU=1").'
            )
            task["stdout"] = ""
            task["stderr"] = str(task["exception"])
            task["exit_code"] = 1
            self._callback_func(task, "FAILED")
            return

        dask_future = self._client.submit(fn, *args, **backend_kwargs)

        # Store the future for potential cancellation
        self.tasks[task["uid"]]["future"] = dask_future

        # Schedule the callback to run when future completes
        asyncio.create_task(on_done(dask_future))

    async def _submit_async_function(self, task: dict[str, Any]) -> None:
        """Submit async function to Dask.

        Creates an async wrapper that preserves the original function name
        for better visibility in the Dask dashboard.

        Args:
            task: Task dictionary containing the async function and its parameters.
        """

        # Preserve the real task name in dask dashboard
        @wraps(task["function"])
        async def async_wrapper():
            return await task["function"](*task["args"], **task["kwargs"])

        await self._submit_to_dask(task, async_wrapper)

    async def _submit_sync_function(self, task: dict[str, Any]) -> None:
        """Submit a sync (non-coroutine) function to Dask.

        Dask workers run sync functions natively — no async wrapper needed.

        Args:
            task: Task dictionary containing the sync function and its parameters.
        """
        fn = task["function"]
        if task.get("kwargs"):
            fn = partial(fn, **task["kwargs"])
        await self._submit_to_dask(task, fn, *task["args"])

    async def _submit_executable(self, task: dict[str, Any]) -> None:
        """Submit an executable task to run via subprocess inside a Dask worker.

        Args:
            task: Task dictionary containing executable path, arguments, and metadata.
        """
        bksp = task.get("task_backend_specific_kwargs", {})
        backend_kwargs = {k: v for k, v in bksp.items() if k not in ("cwd", "shell", "env")}
        dask_resources = backend_kwargs.get("resources", {})
        if dask_resources and not self._check_resources_satisfiable(dask_resources):
            msg = (
                f"No worker can satisfy resources {dask_resources}. "
                f"Workers must be started with matching --resources flags "
                f'(e.g. dask worker <scheduler> --resources "GPU=1").'
            )
            task["stderr"] = msg
            task["stdout"] = ""
            task["exit_code"] = 1
            self._callback_func(task, "FAILED")
            return

        dask_future = self._client.submit(
            _run_executable,
            task["executable"],
            task.get("arguments", []),
            cwd=bksp.get("cwd"),
            env=bksp.get("env"),
            shell=bksp.get("shell", False),
            capture_stdio=task.get("capture_stdio", False),
            output_dir=self._work_dir,
            uid=task["uid"],
            **backend_kwargs,
        )
        self.tasks[task["uid"]]["future"] = dask_future

        async def on_done(f):
            task_uid = task["uid"]
            try:
                self._callback_func(task, "RUNNING")
                stdout, stderr, returncode = await f
                task["stdout"] = stdout
                task["stderr"] = stderr
                task["exit_code"] = returncode
                state = "DONE" if returncode == 0 else "FAILED"
                self._callback_func(task, state)
            except dask.client.FutureCancelledError:
                self._callback_func(task, "CANCELED")
            except Exception as e:
                task["exception"] = e
                task["stdout"] = ""
                task["stderr"] = str(e)
                self._callback_func(task, "FAILED")
            finally:
                if task_uid in self.tasks:
                    del self.tasks[task_uid]

        asyncio.create_task(on_done(dask_future))

    def _check_resources_satisfiable(self, resources: dict) -> bool:
        """Return True if at least one connected worker can satisfy all resource constraints.

        Args:
            resources: Dict of resource requirements (e.g. {"GPU": 1}).

        Returns:
            True if a qualifying worker exists, False otherwise.
        """
        workers = self._client.scheduler_info().get("workers", {})
        return any(
            all(w.get("resources", {}).get(k, 0) >= v for k, v in resources.items())
            for w in workers.values()
        )

    async def cancel_all_tasks(self) -> int:
        """Cancel all currently running/pending tasks.

        Returns:
            Number of tasks that were successfully cancelled
        """
        self._ensure_initialized()
        cancelled_count = 0
        task_uids = list(self.tasks.keys())

        for task_uid in task_uids:
            if await self.cancel_task(task_uid):
                cancelled_count += 1

        return cancelled_count

    def link_explicit_data_deps(
        self,
        src_task: dict[str, Any] | None = None,
        dst_task: dict[str, Any] | None = None,
        file_name: str | None = None,
        file_path: str | None = None,
    ) -> None:
        """Handle explicit data dependencies between tasks.

        Args:
            src_task: The source task that produces the dependency.
            dst_task: The destination task that depends on the source.
            file_name: Name of the file that represents the dependency.
            file_path: Full path to the file that represents the dependency.
        """
        pass

    def link_implicit_data_deps(self, src_task: dict[str, Any], dst_task: dict[str, Any]) -> None:
        """Handle implicit data dependencies for a task.

        Args:
            src_task: The source task that produces data.
            dst_task: The destination task that depends on the source task's output.
        """
        pass

    async def state(self) -> str:
        """Get backend state.

        Returns:
            str: Current backend state (INITIALIZED, RUNNING, SHUTDOWN)
        """
        return self._backend_state.value

    async def task_state_cb(self, task: dict, state: str) -> None:
        """Callback function invoked when a task's state changes.

        Args:
            task: Dictionary containing task information and metadata.
            state: The new state of the task.
        """
        pass

    async def build_task(self, task: dict) -> None:
        """Build or prepare a task for execution.

        Args:
            task: Dictionary containing task definition, parameters, and metadata
                required for task construction.
        """
        pass

    async def shutdown(self) -> None:
        """Shutdown the Dask client and clean up resources.

        Closes the Dask client connection, clears task storage, and handles any cleanup exceptions
        gracefully.
        """
        # Set backend state to SHUTDOWN
        self._backend_state = BackendMainStates.SHUTDOWN
        self.logger.debug(f"Backend state set to: {self._backend_state.value}")

        if self._client is not None:
            try:
                # Cancel all running tasks first
                await self.cancel_all_tasks()

                # Close the client
                await self._client.close()
                self.logger.info("Dask client shutdown complete")
            except Exception as e:
                self.logger.exception(f"Error during shutdown: {str(e)}")
            finally:
                self._client = None
                self.logger.info("Dask execution backend shutdown complete")

        # Always clean up state regardless of client presence
        self.tasks.clear()
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure the backend has been properly initialized."""
        if not self._initialized:
            raise RuntimeError(
                "DaskExecutionBackend must be awaited before use. "
                "Use: backend = await DaskExecutionBackend(resources)"
            )

    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self._async_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    @classmethod
    async def create(
        cls,
        resources: dict | None = None,
        name: str = "dask",
        cluster: Any | None = None,
        client: Any | None = None,
    ) -> DaskExecutionBackend:
        """Alternative factory method for creating initialized backend.

        Args:
            resources: Configuration parameters for Dask client initialization.
            name: Name of the backend.
            cluster: Optional preconfigured Dask Cluster object.
            client: Optional preconfigured Dask Client object.

        Returns:
            Fully initialized DaskExecutionBackend instance.
        """
        backend = cls(resources=resources, name=name, cluster=cluster, client=client)
        return await backend
