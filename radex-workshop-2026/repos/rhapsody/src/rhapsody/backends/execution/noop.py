"""No-op execution backend for performance benchmarking.

Tasks are immediately marked as DONE without executing anything.
"""

import asyncio
import logging
import uuid
from typing import Any
from typing import Callable

from ..base import BaseBackend
from ..constants import BackendMainStates
from ..constants import StateMapper


def _get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


class NoopExecutionBackend(BaseBackend):
    """Backend that completes every task instantly.

    Useful for measuring Orbit/broker/client overhead without any actual task execution cost.
    """

    def __init__(self, name: str = "noop"):
        super().__init__(name=name)
        self.logger = _get_logger()
        self.tasks: dict[str, dict] = {}
        self._futures: dict[str, asyncio.Task] = {}
        self._callback_func: Callable = lambda t, s: None
        self._initialized = False
        self._backend_state = BackendMainStates.INITIALIZED

    def __await__(self):
        return self._async_init().__await__()

    async def _async_init(self):
        if not self._initialized:
            StateMapper.register_backend_states_with_defaults(backend=self)
            StateMapper.register_backend_tasks_states_with_defaults(backend=self)
            self._backend_state = BackendMainStates.INITIALIZED
            self._initialized = True
            self.logger.info("Noop execution backend started")
        return self

    def get_task_states_map(self):
        return StateMapper(backend=self)

    async def submit_tasks(self, tasks: list[dict[str, Any]]) -> None:
        if self._backend_state != BackendMainStates.RUNNING:
            self._backend_state = BackendMainStates.RUNNING

        for task in tasks:
            task.update(
                {
                    "return_value": True,
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                }
            )
            uid = task.setdefault("uid", f"noop.{uuid.uuid4().hex[:8]}")
            self.tasks[uid] = task
            # Track the completion future so it can be cancelled via
            # cancel_task / cancel_all_tasks / shutdown.
            self._futures[uid] = asyncio.create_task(self._complete(task))

    async def _complete(self, task: dict) -> None:
        uid = task["uid"]
        try:
            task["state"] = "DONE"
            self._callback_func(task, "DONE")
        finally:
            # Drop terminal tasks so the backend does not retain every task it
            # ever ran (this backend is used to benchmark millions of them).
            self._futures.pop(uid, None)
            self.tasks.pop(uid, None)

    async def cancel_task(self, uid: str) -> bool:
        if uid not in self.tasks:
            return False

        future = self._futures.pop(uid, None)
        if future is not None and not future.done():
            future.cancel()

        # Remove the task (terminal); don't retain it.
        task = self.tasks.pop(uid)
        if task.get("state") not in ("DONE", "FAILED", "CANCELED"):
            task["state"] = "CANCELED"
            self._callback_func(task, "CANCELED")
        return True

    async def cancel_all_tasks(self) -> int:
        uids = list(self.tasks)
        for uid in uids:
            await self.cancel_task(uid)
        return len(uids)

    async def shutdown(self) -> None:
        await self.cancel_all_tasks()
        self._backend_state = BackendMainStates.SHUTDOWN
        self.tasks.clear()
        self._futures.clear()
        self.logger.info("Noop execution backend shutdown")

    def build_task(self, uid, task_desc, task_specific_kwargs):
        pass

    def link_explicit_data_deps(self, src_task=None, dst_task=None, file_name=None, file_path=None):
        pass

    def link_implicit_data_deps(self, src_task, dst_task):
        pass

    async def state(self) -> str:
        return self._backend_state.value

    def task_state_cb(self, task, state):
        pass

    async def __aenter__(self):
        if not self._initialized:
            await self._async_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    @classmethod
    async def create(cls, name: str = "noop") -> "NoopExecutionBackend":
        """Alternative factory that returns an initialized backend."""
        backend = cls(name=name)
        return await backend
