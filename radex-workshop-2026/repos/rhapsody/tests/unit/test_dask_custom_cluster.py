import asyncio

import pytest
from dask.distributed import Client
from dask.distributed import LocalCluster

from rhapsody.api import ComputeTask
from rhapsody.api import Session
from rhapsody.backends.execution.dask_parallel import DaskExecutionBackend

# ---------------------------------------------------------------------------
# Cluster injection
# ---------------------------------------------------------------------------


async def test_dask_preconfigured_cluster():
    async with LocalCluster(n_workers=1, threads_per_worker=1, asynchronous=True) as cluster:
        backend = await DaskExecutionBackend(cluster=cluster)
        assert backend._client is not None
        assert backend._client.cluster is cluster
        await backend.shutdown()


async def test_dask_preconfigured_client():
    async with LocalCluster(n_workers=1, threads_per_worker=1, asynchronous=True) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            backend = await DaskExecutionBackend(client=client)
            assert backend._client is client
            await backend.shutdown()


# ---------------------------------------------------------------------------
# End-to-end task execution
# ---------------------------------------------------------------------------


async def test_dask_sync_function_e2e():
    """Sync function tasks complete with correct return_value."""

    def square(n):
        return n * n

    async with DaskExecutionBackend(resources={"n_workers": 1, "threads_per_worker": 1}) as backend:
        session = Session(backends=[backend])
        tasks = [ComputeTask(function=square, args=(i,)) for i in range(5)]
        async with session:
            await session.submit_tasks(tasks)
            await session.wait_tasks(tasks)

    assert all(t.state == "DONE" for t in tasks)
    assert [t.return_value for t in tasks] == [i * i for i in range(5)]


async def test_dask_async_function_e2e():
    """Async function tasks complete with correct return_value."""

    async def double(n):
        await asyncio.sleep(0)
        return n * 2

    async with DaskExecutionBackend(resources={"n_workers": 1, "threads_per_worker": 1}) as backend:
        session = Session(backends=[backend])
        tasks = [ComputeTask(function=double, args=(i,)) for i in range(5)]
        async with session:
            await session.submit_tasks(tasks)
            await session.wait_tasks(tasks)

    assert all(t.state == "DONE" for t in tasks)
    assert [t.return_value for t in tasks] == [i * 2 for i in range(5)]


async def test_dask_executable_e2e():
    """Executable tasks complete with correct stdout and exit_code."""
    async with DaskExecutionBackend(resources={"n_workers": 1, "threads_per_worker": 1}) as backend:
        session = Session(backends=[backend])
        tasks = [ComputeTask(executable="/bin/echo", arguments=[f"hello {i}"]) for i in range(3)]
        async with session:
            await session.submit_tasks(tasks)
            await session.wait_tasks(tasks)

    assert all(t.state == "DONE" for t in tasks)
    assert all(t.exit_code == 0 for t in tasks)
    for i, t in enumerate(tasks):
        assert f"hello {i}" in t.stdout


# ---------------------------------------------------------------------------
# Resource scheduling — fail fast on unmet constraints
# ---------------------------------------------------------------------------


async def test_dask_unmet_resources_fails_immediately():
    """Tasks with unsatisfiable resource constraints must FAIL, not hang."""
    async with DaskExecutionBackend(resources={"n_workers": 1, "threads_per_worker": 1}) as backend:
        session = Session(backends=[backend])
        # LocalCluster workers have no GPU resources — should fail immediately
        tasks = [
            ComputeTask(
                function=lambda: None,
                task_backend_specific_kwargs={"resources": {"GPU": 1}},
            )
        ]
        async with session:
            await session.submit_tasks(tasks)
            await session.wait_tasks(tasks)

    assert tasks[0].state == "FAILED"
    assert tasks[0].exception is not None


async def test_dask_unmet_resources_executable_sets_stderr():
    """Executable tasks with unsatisfiable resources must set stderr and FAIL."""
    async with DaskExecutionBackend(resources={"n_workers": 1, "threads_per_worker": 1}) as backend:
        session = Session(backends=[backend])
        tasks = [
            ComputeTask(
                executable="/bin/echo",
                arguments=["hi"],
                task_backend_specific_kwargs={"resources": {"GPU": 1}},
            )
        ]
        async with session:
            await session.submit_tasks(tasks)
            await session.wait_tasks(tasks)

    assert tasks[0].state == "FAILED"
    assert tasks[0].stderr  # must contain the error message
    assert tasks[0].exit_code == 1


if __name__ == "__main__":
    asyncio.run(test_dask_preconfigured_cluster())
    asyncio.run(test_dask_preconfigured_client())
