import asyncio
import os

import pytest

from rhapsody import ComputeTask
from rhapsody.api import Session
from rhapsody.api import TaskStateManager
from rhapsody.api.errors import TaskExecutionError
from rhapsody.backends import ConcurrentExecutionBackend


def test_session_class_import():
    """Test that Session class can be imported."""
    assert Session is not None


def test_session_instantiation():
    """Test that Session class can be instantiated."""
    session = Session()
    assert session is not None
    assert session.backends == {}


def test_session_path_attribute():
    """Test that Session sets work_dir attribute to current working directory."""
    original_cwd = os.getcwd()
    session = Session()
    assert hasattr(session, "work_dir")
    assert session.work_dir == original_cwd


@pytest.mark.asyncio
async def test_session_submit_and_wait_flow():
    """Test complete flow: submit -> wait -> verify."""
    backend = await ConcurrentExecutionBackend()
    session = Session(backends=[backend])

    tasks = [
        ComputeTask(uid="task_1", executable="echo", arguments=["hello"]),
        ComputeTask(uid="task_2", executable="false"),  # Will fail
    ]

    async with session:
        await session.submit_tasks(tasks)

        # Tasks should have state updated eventually
        await session.wait_tasks(tasks, timeout=5.0)

        # Verify states
        task1 = next(t for t in tasks if t.uid == "task_1")
        task2 = next(t for t in tasks if t.uid == "task_2")

        assert task1.state == "DONE"
        assert task2.state == "FAILED"

    @pytest.mark.asyncio
    async def test_session_returns_futures(self):
        """Test that submit_tasks returns native futures."""
        backend = await ConcurrentExecutionBackend()
        session = Session(backends=[backend])
        task = ComputeTask(executable="echo", arguments=["hi"])

        async with session:
            futures = await session.submit_tasks([task])
            assert isinstance(futures, list)
            assert len(futures) == 1
            assert isinstance(futures[0], asyncio.Future)

            # Wait for completion via future
            result = await futures[0]
            assert result == task
            assert task.state == "DONE"

    @pytest.mark.asyncio
    async def test_session_direct_task_await(self):
        """Test that task objects can be awaited directly."""
        backend = await ConcurrentExecutionBackend()
        session = Session(backends=[backend])
        task = ComputeTask(executable="echo", arguments=["hi"])

        async with session:
            await session.submit_tasks([task])

            # Direct await on task
            result = await task
            assert result == task
            assert task.state == "DONE"

    @pytest.mark.asyncio
    async def test_session_gather_tasks(self):
        """Test that asyncio.gather works on tasks or futures."""
        backend = await ConcurrentExecutionBackend()
        session = Session(backends=[backend])
        tasks = [ComputeTask(uid=f"t{i}", executable="echo", arguments=[str(i)]) for i in range(3)]

        async with session:
            futures = await session.submit_tasks(tasks)

            # Gather futures
            results = await asyncio.gather(*futures)
            assert len(results) == 3
            assert all(t.state == "DONE" for t in tasks)

            # Reset tasks for another test
            tasks2 = [
                ComputeTask(uid=f"t2_{i}", executable="echo", arguments=[str(i)]) for i in range(3)
            ]
            await session.submit_tasks(tasks2)

            # Gather tasks directly
            results2 = await asyncio.gather(*tasks2)
            assert len(results2) == 3
            assert all(t.state == "DONE" for t in tasks2)


@pytest.mark.asyncio
async def test_session_wait_timeout():
    """Test that Session.wait_tasks respects timeout."""
    backend = await ConcurrentExecutionBackend()
    session = Session(backends=[backend])

    tasks = [
        ComputeTask(uid="slow_task", executable="sleep", arguments=["5"]),
    ]

    async with session:
        await session.submit_tasks(tasks)

        with pytest.raises(asyncio.TimeoutError):
            await session.wait_tasks(tasks, timeout=0.1)

        # Cancel to clean up
        await backend.cancel_task(tasks[0].uid)


@pytest.mark.asyncio
async def test_session_callbacks_registered():
    """Test that Session registers its state manager callback with backends."""
    backend = await ConcurrentExecutionBackend()
    session = Session(backends=[backend])

    # Check if callback is registered (implementation detail check)
    # The session registers self._state_manager.update_task
    assert backend._callback_func == session._state_manager.update_task
    await session.close()


@pytest.mark.asyncio
async def test_session_explicit_routing():
    """Test that Session routes tasks to the correct named backend."""
    # Create two backends with different names
    backend1 = await ConcurrentExecutionBackend(name="b1")
    backend2 = await ConcurrentExecutionBackend(name="b2")

    session = Session(backends=[backend1, backend2])

    # Task 1 goes to b1
    task1 = ComputeTask(uid="t1", executable="echo", arguments=["b1"], backend="b1")
    # Task 2 goes to b2
    task2 = ComputeTask(uid="t2", executable="echo", arguments=["b2"], backend="b2")
    # Task 3 has no backend, should go to first backend (b1)
    task3 = ComputeTask(uid="t3", executable="echo", arguments=["default"])

    async with session:
        await session.submit_tasks([task1, task2, task3])

        # Verify routing recorded in task object
        assert task1.backend == "b1"
        assert task2.backend == "b2"
        assert task3.backend == "b1"

        # Wait for all
        await asyncio.gather(task1, task2, task3)

        assert task1.state == "DONE"
        assert task2.state == "DONE"
        assert task3.state == "DONE"


@pytest.mark.asyncio
async def test_session_invalid_backend():
    """Test that Session raises ValueError for unknown backend names."""
    backend = await ConcurrentExecutionBackend(name="valid")
    session = Session(backends=[backend])
    task = ComputeTask(executable="echo", backend="invalid")

    async with session:
        with pytest.raises(ValueError, match="Backend 'invalid' requested by task .* not found"):
            await session.submit_tasks([task])


# ---------------------------------------------------------------------------
# Future exception propagation tests (Bug 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_future_set_result_on_successful_command_task():
    """Future resolves to the task object when a command task succeeds."""
    backend = await ConcurrentExecutionBackend()
    session = Session(backends=[backend])
    task = ComputeTask(uid="ok_task", executable="echo", arguments=["hello"])

    async with session:
        futures = await session.submit_tasks([task])
        result = await futures[0]

    assert result is task
    assert task.state == "DONE"


@pytest.mark.asyncio
async def test_session_future_raises_task_execution_error_on_failed_command_task():
    """Future raises TaskExecutionError when a command task fails (non-zero exit code)."""
    backend = await ConcurrentExecutionBackend()
    session = Session(backends=[backend])
    task = ComputeTask(uid="fail_task", executable="false")

    async with session:
        futures = await session.submit_tasks([task])
        with pytest.raises(TaskExecutionError) as exc_info:
            await futures[0]

    assert task.state == "FAILED"
    assert exc_info.value.task_uid == "fail_task"


@pytest.mark.asyncio
async def test_session_future_propagates_function_exception():
    """Future raises the original exception when a function task raises."""
    backend = await ConcurrentExecutionBackend()
    session = Session(backends=[backend])

    def boom():
        raise ValueError("intentional failure")

    task = ComputeTask(uid="exc_task", function=boom)

    async with session:
        futures = await session.submit_tasks([task])
        with pytest.raises(ValueError, match="intentional failure"):
            await futures[0]

    assert task.state == "FAILED"


@pytest.mark.asyncio
async def test_session_future_set_result_on_successful_function_task():
    """Future resolves to the task object when a function task succeeds."""
    backend = await ConcurrentExecutionBackend()
    session = Session(backends=[backend])

    def add(a, b):
        return a + b

    task = ComputeTask(uid="fn_ok_task", function=add, args=[1, 2])

    async with session:
        futures = await session.submit_tasks([task])
        result = await futures[0]

    assert result is task
    assert task.state == "DONE"
    assert task["return_value"] == 3
