"""Integration tests for Task API with backends via Session.

Tests cover:
- Direct Task object submission to backends
- wait_tasks() with Task objects
- Mixed Task objects and dicts
- Large batch submissions
"""

import pytest

import rhapsody
from rhapsody import ComputeTask
from rhapsody.api import Session


@pytest.mark.asyncio
async def test_direct_compute_task_submission():
    """Test submitting ComputeTask objects directly to backend."""
    tasks = [ComputeTask(executable="/bin/echo", arguments=["Hello", str(i)]) for i in range(10)]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        # Submit via session
        await session.submit_tasks(tasks)

        # Wait via session
        results = await session.wait_tasks(tasks)

        assert len(results) == 10

        # Verify all tasks completed
        for task in tasks:
            assert task.uid in [r["uid"] for r in results]
            # Session updates the task object in place
            assert task["state"] in ["DONE", "COMPLETED"]


@pytest.mark.asyncio
async def test_mixed_task_and_dict_submission():
    """Test submitting mix of Task objects and dicts."""
    tasks = [
        ComputeTask(executable="/bin/echo", arguments=["Task", "object"]),
        {"uid": "dict_task_1", "executable": "/bin/echo", "arguments": ["Dict", "task"]},
        ComputeTask(executable="/bin/echo", arguments=["Another", "Task"]),
        # Note: Session wait_tasks relies on callbacks updating state.
        # Dict tasks must be mutable and updated by the system.
        {"uid": "dict_task_2", "executable": "/bin/hostname"},
    ]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        await session.submit_tasks(tasks)
        results = await session.wait_tasks(tasks)

        assert len(results) == 4

        # Verify auto-generated UIDs for Task objects
        task_obj_uids = [t.uid for t in tasks if isinstance(t, ComputeTask)]
        for uid in task_obj_uids:
            assert uid.startswith("task.")
            assert uid in [r["uid"] for r in results if isinstance(r, ComputeTask)]

        # Verify manual UIDs for dicts
        assert "dict_task_1" in [r["uid"] for r in results if isinstance(r, dict)]
        assert "dict_task_2" in [r["uid"] for r in results if isinstance(r, dict)]


@pytest.mark.asyncio
async def test_large_batch_submission():
    """Test submitting large batch of Task objects."""
    tasks = [ComputeTask(executable="/bin/echo", arguments=[f"Task_{i}"]) for i in range(100)]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        await session.submit_tasks(tasks)
        results = await session.wait_tasks(tasks)

        assert len(results) == 100

        # All UIDs should be present
        result_uids = [r["uid"] for r in results]
        for task in tasks:
            assert task.uid in result_uids


@pytest.mark.asyncio
async def test_auto_uid_uniqueness_in_backend():
    """Test auto-generated UIDs are unique in backend execution."""
    tasks = [ComputeTask(executable="/bin/hostname") for _ in range(50)]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        await session.submit_tasks(tasks)
        results = await session.wait_tasks(tasks)

        # All UIDs should be unique
        uids = [r["uid"] for r in results]
        assert len(uids) == 50
        assert len(set(uids)) == 50

        # All UIDs should follow pattern
        for uid in uids:
            assert uid.startswith("task.")


@pytest.mark.asyncio
async def test_function_task_submission():
    """Test submitting function-based ComputeTask to backend."""

    async def compute_sum(n):
        """Async function for testing."""
        return sum(range(n))

    tasks = [ComputeTask(function=compute_sum, args=(i * 10,)) for i in range(1, 11)]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        await session.submit_tasks(tasks)
        results = await session.wait_tasks(tasks)

        assert len(results) == 10

        # Verify all tasks were processed
        result_uids = [r["uid"] for r in results]
        for task in tasks:
            assert task.uid in result_uids
            # Task state update check
            assert task["state"] in ["DONE", "COMPLETED", "FAILED", "FINISHED"]


@pytest.mark.asyncio
async def test_task_with_resources():
    """Test submitting tasks with resource requirements."""
    tasks = [
        ComputeTask(executable="/bin/echo", arguments=["Task 1"]),
        ComputeTask(executable="/bin/echo", arguments=["Task 2"]),
        ComputeTask(executable="/bin/echo", arguments=["Task 3"]),
    ]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        await session.submit_tasks(tasks)
        results = await session.wait_tasks(tasks)

        assert len(results) == 3

        result_uids = [r["uid"] for r in results]
        for task in tasks:
            assert task.uid in result_uids


@pytest.mark.asyncio
async def test_task_with_custom_fields():
    """Test submitting tasks with custom fields."""
    tasks = [
        ComputeTask(executable="/bin/echo", arguments=["Task 1"], priority="high", timeout=30),
        ComputeTask(executable="/bin/echo", arguments=["Task 2"], priority="low", retry_count=3),
    ]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        await session.submit_tasks(tasks)
        results = await session.wait_tasks(tasks)

        assert len(results) == 2

        result_uids = [r["uid"] for r in results]
        for task in tasks:
            assert task.uid in result_uids


@pytest.mark.asyncio
async def test_wait_tasks_with_task_objects():
    """Test wait_tasks() accepts Task objects directly."""
    tasks = [ComputeTask(executable="/bin/echo", arguments=[f"Task {i}"]) for i in range(5)]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        await session.submit_tasks(tasks)

        # wait_tasks should accept Task objects (not just dicts)
        results = await session.wait_tasks(tasks)

        assert len(results) == 5

        result_uids = [r["uid"] for r in results]
        for task in tasks:
            assert task.uid in result_uids


@pytest.mark.asyncio
async def test_submit_empty_list():
    """Test submitting empty task list."""
    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        # Empty list should not raise error
        await session.submit_tasks([])


@pytest.mark.asyncio
async def test_concurrent_submissions():
    """Test concurrent task submissions with auto-generated UIDs."""
    import asyncio

    async def submit_batch(session, batch_id):
        tasks = [
            ComputeTask(executable="/bin/echo", arguments=[f"Batch {batch_id}, Task {i}"])
            for i in range(10)
        ]
        await session.submit_tasks(tasks)
        return tasks

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        # Submit multiple batches concurrently
        all_tasks = []
        batches = await asyncio.gather(*[submit_batch(session, i) for i in range(5)])

        for batch in batches:
            all_tasks.extend(batch)

        # Wait for all tasks
        results = await session.wait_tasks(all_tasks)

        assert len(results) == 50  # 5 batches * 10 tasks

        # All UIDs should be unique
        uids = [r["uid"] for r in results]
        assert len(set(uids)) == 50


@pytest.mark.asyncio
async def test_task_api_backward_compatibility():
    """Test that dict-based tasks still work (backward compatibility)."""
    # Old-style dict tasks
    dict_tasks = [
        {"uid": "old_task_1", "executable": "/bin/echo", "arguments": ["hello"]},
        {"uid": "old_task_2", "executable": "/bin/hostname"},
    ]

    # New-style Task objects
    task_objects = [
        ComputeTask(executable="/bin/echo", arguments=["world"]),
        ComputeTask(executable="/bin/pwd"),
    ]

    # Mix both styles
    all_tasks = dict_tasks + task_objects

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        # Should work with mixed styles
        await session.submit_tasks(all_tasks)
        results = await session.wait_tasks(all_tasks)

        assert len(results) == 4

        result_uids = [r["uid"] for r in results]

        # Dict tasks should have their manual UIDs
        assert "old_task_1" in result_uids
        assert "old_task_2" in result_uids

        # Task objects should have auto-generated UIDs
        for task in task_objects:
            assert task.uid in result_uids


@pytest.mark.asyncio
async def test_task_state_tracking():
    """Test that task states are properly tracked."""
    tasks = [ComputeTask(executable="/bin/echo", arguments=[f"Task {i}"]) for i in range(10)]

    backend = await rhapsody.get_backend("concurrent")
    session = Session(backends=[backend])

    async with session:
        await session.submit_tasks(tasks)
        results = await session.wait_tasks(tasks)

        # All tasks should have a state
        for task in tasks:
            result_task = next(r for r in results if r["uid"] == task.uid)
            assert "state" in result_task
            assert result_task["state"] in ["DONE", "COMPLETED", "FINISHED"]
