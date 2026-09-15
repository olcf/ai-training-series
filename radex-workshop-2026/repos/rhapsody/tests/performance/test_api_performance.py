import asyncio
import pickle
import time
from typing import Optional

import pytest

import rhapsody
from rhapsody import AITask
from rhapsody import ComputeTask
from rhapsody.api import Session
from rhapsody.backends.base import BaseBackend
from rhapsody.backends.constants import StateMapper

# Register a mock backend for performance testing
StateMapper.register_backend_tasks_states_with_defaults("mock")


class MockBackend(BaseBackend):
    """Minimal backend that does nothing but record submission."""

    def __init__(self, name: str = "mock"):
        super().__init__(name=name)
        self._callback_func = None

    async def initialize(self) -> None:
        pass

    async def submit_tasks(self, tasks: list[dict]) -> None:
        # Simulate immediate completion for performance testing of the session layer
        for task in tasks:
            if self._callback_func:
                self._callback_func(task, "DONE")

    async def shutdown(self) -> None:
        pass

    def state(self) -> str:
        return "running"

    def task_state_cb(self, task: dict, state: str) -> None:
        pass

    def get_task_states_map(self) -> StateMapper:
        return StateMapper("mock")

    def build_task(self, task: dict) -> None:
        pass

    def link_implicit_data_deps(self, src: dict, dst: dict) -> None:
        pass

    def link_explicit_data_deps(self, src=None, dst=None, file_name=None, file_path=None) -> None:
        pass

    async def cancel_task(self, uid: str) -> bool:
        return True


@pytest.mark.performance
class TestApiPerformance:
    """Performance benchmarks for Rhapsody API."""

    def test_task_creation_performance(self):
        """Benchmark creating 100,000 task objects."""
        n = 100_000
        start = time.time()
        tasks = [ComputeTask(executable="/bin/echo", uid=f"t.{i}") for i in range(n)]
        duration = time.time() - start

        print(f"\nTask Creation (100K): {duration:.4f}s ({duration / n * 1e6:.2f} μs/task)")
        assert duration < 1.0  # Should be well under 1s

    def test_serialization_performance(self):
        """Benchmark pickling 100,000 task objects."""
        n = 100_000
        tasks = [ComputeTask(executable="/bin/echo", uid=f"t.{i}") for i in range(n)]

        # Pickle
        start = time.time()
        p_data = pickle.dumps(tasks)
        p_duration = time.time() - start

        # Unpickle
        start = time.time()
        tasks2 = pickle.loads(p_data)
        u_duration = time.time() - start

        print(f"\nPickle (100K): {p_duration:.4f}s ({p_duration / n * 1e6:.2f} μs/task)")
        print(f"Unpickle (100K): {u_duration:.4f}s ({u_duration / n * 1e6:.2f} μs/task)")
        print(f"Payload Size: {len(p_data) / 1024 / 1024:.2f} MB")

        assert p_duration < 1.0
        assert u_duration < 1.0

    @pytest.mark.asyncio
    async def test_session_submission_throughput(self):
        """Benchmark Session.submit_tasks throughput with 10,000 tasks."""
        n = 10_000
        backend = MockBackend()
        session = Session(backends=[backend])
        tasks = [ComputeTask(executable="/bin/echo", uid=f"t.{i}") for i in range(n)]

        async with session:
            start = time.time()
            futures = await session.submit_tasks(tasks)
            duration = time.time() - start

            print(f"\nSession Submission (10K): {duration:.4f}s ({duration / n * 1e6:.2f} μs/task)")
            # 10K tasks should be submitted very quickly
            assert duration < 0.5

    @pytest.mark.asyncio
    async def test_state_resolution_performance(self):
        """Benchmark TaskStateManager resolving 10,000 futures."""
        n = 10_000
        backend = MockBackend()
        session = Session(backends=[backend])
        tasks = [ComputeTask(executable="/bin/echo", uid=f"t.{i}") for i in range(n)]

        async with session:
            futures = await session.submit_tasks(tasks)

            start = time.time()
            # In our MockBackend, submit_tasks calls update_task immediately.
            # So the futures might already be resolved or resolving.
            await asyncio.gather(*futures)
            duration = time.time() - start

            print(f"\nState Resolution (10K): {duration:.4f}s ({duration / n * 1e6:.2f} μs/task)")
            assert duration < 0.5
