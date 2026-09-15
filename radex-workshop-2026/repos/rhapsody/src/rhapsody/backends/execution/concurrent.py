"""Concurrent distributed execution backend for parallel and distributed computing.

This module provides a backend that executes tasks on local or single node HPC resources.
"""

import asyncio
import logging
import os
from concurrent.futures import Executor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Callable

from ..base import BaseBackend
from ..constants import BackendMainStates
from ..constants import StateMapper

try:
    import cloudpickle
except ImportError:
    cloudpickle = None


def _get_logger() -> logging.Logger:
    """Get logger for concurrent backend module.

    This function provides lazy logger evaluation, ensuring the logger is created after the user has
    configured logging, not at module import time.
    """
    return logging.getLogger(__name__)


class ConcurrentExecutionBackend(BaseBackend):
    """Simple async-only concurrent execution backend."""

    def __init__(self, executor: Executor = None, name: str = "concurrent"):
        super().__init__(name=name)

        self.logger = _get_logger()

        if not executor:
            executor = ThreadPoolExecutor()
            self.logger.info("No executor was provided. Falling back to default ThreadPoolExecutor")

        if not isinstance(executor, Executor):
            err = "Executor must be ThreadPoolExecutor or ProcessPoolExecutor"
            raise TypeError(err)

        if isinstance(executor, ProcessPoolExecutor) and cloudpickle is None:
            raise ImportError(
                "ProcessPoolExecutor requires 'cloudpickle'. "
                "Install it with: pip install cloudpickle"
            )

        self.executor = executor
        self.tasks: dict[str, asyncio.Task] = {}
        self._callback_func: Callable = lambda t, s: None
        self._initialized = False
        self._backend_state = BackendMainStates.INITIALIZED

    def __await__(self):
        """Make backend awaitable."""
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
                # Step 1: Register backend states
                self.logger.debug("Registering backend states...")
                StateMapper.register_backend_states_with_defaults(backend=self)

                # Step 2: Register task states
                self.logger.debug("Registering task states...")
                StateMapper.register_backend_tasks_states_with_defaults(backend=self)

                # Step 3: Set backend state to INITIALIZED
                self._backend_state = BackendMainStates.INITIALIZED
                self.logger.debug(f"Backend state set to: {self._backend_state.value}")

                # Step 4: Initialize backend components (already done in __init__)
                self._initialized = True

                executor_name = type(self.executor).__name__
                self.logger.info(f"{executor_name} execution backend started successfully")

            except Exception as e:
                self.logger.exception(f"Concurrent backend initialization failed: {e}")
                self._initialized = False
                raise
        return self

    def get_task_states_map(self):
        return StateMapper(backend=self)

    async def _execute_task(self, task: dict) -> tuple[dict, str]:
        """Execute a single task."""
        try:
            if "function" in task and task["function"]:
                return await self._execute_function(task)
            else:
                return await self._execute_command(task)
        except Exception as e:
            task.update(
                {
                    "stderr": str(e),
                    "stdout": "",
                    "exit_code": 1,
                    "exception": e,
                    "return_value": None,
                }
            )
            return task, "FAILED"

    @staticmethod
    def _run_in_process(func, args, kwargs):
        """Execute function (sync or async) in isolated executor process."""
        func = cloudpickle.loads(func)
        if asyncio.iscoroutinefunction(func):
            return asyncio.run(func(*args, **kwargs))
        return func(*args, **kwargs)

    @staticmethod
    def _run_in_thread(func, args, kwargs):
        """Execute function (sync or async) in isolated executor thread."""
        if asyncio.iscoroutinefunction(func):
            return asyncio.run(func(*args, **kwargs))
        return func(*args, **kwargs)

    async def _execute_function(self, task: dict) -> tuple[dict, str]:
        """Execute async function task in Process/Thread PoolExecutor."""
        func = task["function"]
        args = task.get("args", [])
        kwargs = task.get("kwargs", {})

        # Serialize the async function
        if isinstance(self.executor, ProcessPoolExecutor):
            func = cloudpickle.dumps(func)
            exec_wrapper = self._run_in_process
        else:
            exec_wrapper = self._run_in_thread

        loop = asyncio.get_running_loop()

        # Submit to the executor
        result = await loop.run_in_executor(
            self.executor,
            exec_wrapper,
            func,
            args,
            kwargs,
        )

        task.update({"return_value": result, "stdout": str(result), "exit_code": 0})
        return task, "DONE"

    async def _execute_command(self, task: dict) -> tuple[dict, str]:
        """Execute command task."""
        executable = task["executable"]
        arguments = task.get("arguments", [])
        backend_kwargs = task.get("task_backend_specific_kwargs", {})
        execute_in_shell = backend_kwargs.get("shell", False)
        cwd = backend_kwargs.get("cwd")
        env = backend_kwargs.get("env")  # None inherits the parent process environment

        redirect = task.get("capture_stdio")
        stdout_f = stderr_f = process = None
        stdout_val = stderr_val = None
        exit_code = 1

        if redirect:
            uid = task["uid"]
            stdout_path = os.path.join(self._work_dir, f"{uid}.stdout")
            stderr_path = os.path.join(self._work_dir, f"{uid}.stderr")
        else:
            stdout_path = stderr_path = None

        try:
            if redirect:
                stdout_f = open(stdout_path, "wb")
                stderr_f = open(stderr_path, "wb")
                stdout_arg = stdout_f
                stderr_arg = stderr_f
            else:
                stdout_arg = asyncio.subprocess.PIPE
                stderr_arg = asyncio.subprocess.PIPE

            if execute_in_shell:
                # Shell mode: join executable and arguments into single command string
                cmd = " ".join([executable] + arguments)
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=stdout_arg,
                    stderr=stderr_arg,
                    cwd=cwd,
                    env=env,
                )
            else:
                # Exec mode: pass executable and arguments separately (no shell)
                process = await asyncio.create_subprocess_exec(
                    executable,
                    *arguments,
                    stdout=stdout_arg,
                    stderr=stderr_arg,
                    cwd=cwd,
                    env=env,
                )

            if redirect:
                await process.wait()
                stdout_val = stdout_path
                stderr_val = stderr_path
            else:
                raw_out, raw_err = await process.communicate()
                stdout_val = raw_out.decode()
                stderr_val = raw_err.decode()

            exit_code = process.returncode
        finally:
            if stdout_f is not None:
                stdout_f.close()
            if stderr_f is not None:
                stderr_f.close()

        task.update(
            {
                "stdout": stdout_val,
                "stderr": stderr_val,
                "exit_code": exit_code,
            }
        )

        state = "DONE" if task["exit_code"] == 0 else "FAILED"
        return task, state

    async def _handle_task(self, task: dict) -> None:
        """Handle task execution with callback."""
        try:
            self._callback_func(task, "RUNNING")
            result_task, state = await self._execute_task(task)
            # Set state on the task object itself before callback
            self._callback_func(result_task, state)
        except Exception as e:
            self.logger.exception(f"Error handling task {task.get('uid')}: {e}")
            raise

    async def submit_tasks(self, tasks: list[dict[str, Any]]) -> list[asyncio.Task]:
        """Submit tasks for execution."""
        # Set backend state to RUNNING when tasks are submitted
        if self._backend_state != BackendMainStates.RUNNING:
            self._backend_state = BackendMainStates.RUNNING
            self.logger.debug(f"Backend state set to: {self._backend_state.value}")

        submitted_tasks = []

        for task in tasks:
            future = asyncio.create_task(self._handle_task(task))
            submitted_tasks.append(future)

            self.tasks[task["uid"]] = task
            self.tasks[task["uid"]]["future"] = future

        return submitted_tasks

    async def cancel_task(self, uid: str) -> bool:
        """Cancel a task by its UID.

        Args:
            uid (str): The UID of the task to cancel.

        Returns:
            bool: True if the task was found and cancellation was attempted,
                  False otherwise.
        """
        if uid in self.tasks:
            task = self.tasks[uid]
            future = task["future"]
            if future and future.cancel():
                # Set state on the task object itself before callback
                self._callback_func(task, "CANCELED")
                return True
        return False

    async def cancel_all_tasks(self) -> int:
        """Cancel all running tasks."""
        cancelled_count = 0
        for task in self.tasks.values():
            future = task["future"]
            future.cancel()
            cancelled_count += 1
        self.tasks.clear()
        return cancelled_count

    async def shutdown(self) -> None:
        """Shutdown the executor."""
        # Set backend state to SHUTDOWN
        self._backend_state = BackendMainStates.SHUTDOWN
        self.logger.debug(f"Backend state set to: {self._backend_state.value}")

        await self.cancel_all_tasks()
        self.executor.shutdown(wait=True)
        self.logger.info("Concurrent execution backend shutdown complete")

    def build_task(self, uid, task_desc, task_specific_kwargs):
        pass

    def link_explicit_data_deps(self, src_task=None, dst_task=None, file_name=None, file_path=None):
        pass

    def link_implicit_data_deps(self, src_task, dst_task):
        pass

    async def state(self) -> str:
        """Get backend state.

        Returns:
            str: Current backend state (INITIALIZED, RUNNING, SHUTDOWN)
        """
        return self._backend_state.value

    def task_state_cb(self):
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self._async_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    @classmethod
    async def create(cls, executor: Executor) -> "ConcurrentExecutionBackend":
        """Alternative factory method for creating initialized backend.

        Args:
            executor: A concurrent.Executor instance (ThreadPoolExecutor
                      or ProcessPoolExecutor).

        Returns:
            Fully initialized ConcurrentExecutionBackend instance.
        """
        backend = cls(executor)
        return await backend
