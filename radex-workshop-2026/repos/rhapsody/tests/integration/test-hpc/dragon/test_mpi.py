"""Category 5 — MPI integration tests (mpi4py).

Purpose:
    Validate that RHAPSODY can launch MPI jobs through Dragon's batch.job()
    API, that mpi4py communication patterns work correctly across ranks, and
    that multi-node MPI jobs distribute ranks across nodes as expected.

Execution strategy:
    Tasks use process_templates (plural) to specify (nranks, ProcessTemplate)
    tuples per node.  Each worker function uses mpi4py to perform standard
    communication patterns (gather, scatter, broadcast, allreduce).

    Results are written to a shared DDict or returned as the function's
    return value from rank 0.

Resource requirements:
    mpi mark — skipped when mpi4py is not importable.
    Multi-node MPI tests require multi_node mark.

Expected outcomes:
    - All MPI ranks start and finish cleanly.
    - Communication results match expected values.
    - No deadlocks or rank mismatches within the timeout.

Run:
    dragon python3 -m pytest test-hpc/test_mpi.py -v
"""

import pytest
from hpc_workers import mpi_allreduce_sum as _mpi_allreduce_sum
from hpc_workers import mpi_broadcast_check as _mpi_broadcast_check
from hpc_workers import mpi_gather_hostnames as _mpi_gather_hostnames
from hpc_workers import mpi_scatter_and_gather as _mpi_scatter_and_gather

# ---------------------------------------------------------------------------
# Single-node MPI tests
# ---------------------------------------------------------------------------


@pytest.mark.mpi
@pytest.mark.asyncio
async def test_mpi_gather_single_node(rhapsody_session, topology):
    """2-rank MPI job on one node: gather hostnames to rank 0."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    node = topology[0]
    ranks = 2

    task = ComputeTask(
        function=_mpi_gather_hostnames,
        task_backend_specific_kwargs={
            "process_templates": [
                (
                    ranks,
                    {
                        "policy": Policy(
                            placement=Policy.Placement.HOST_NAME,
                            host_name=node["hostname"],
                        )
                    },
                )
            ]
        },
    )

    await rhapsody_session.submit_tasks([task])
    await rhapsody_session.wait_tasks([task], timeout=120.0)

    assert task["state"] == "DONE"
    result = task["return_value"]
    assert result is not None, "rank 0 returned None"
    assert result["size"] == ranks
    assert len(result["hostnames"]) == ranks
    assert all(h == node["hostname"] for h in result["hostnames"]), (
        f"Expected all ranks on {node['hostname']!r}, got: {result['hostnames']}"
    )


@pytest.mark.mpi
@pytest.mark.asyncio
async def test_mpi_allreduce_single_node(rhapsody_session, topology):
    """4-rank MPI allreduce on one node: sum must equal ranks * input value."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    node = topology[0]
    ranks = 4
    input_val = 7

    task = ComputeTask(
        function=_mpi_allreduce_sum,
        args=(input_val,),
        task_backend_specific_kwargs={
            "process_templates": [
                (
                    ranks,
                    {
                        "policy": Policy(
                            placement=Policy.Placement.HOST_NAME,
                            host_name=node["hostname"],
                        )
                    },
                )
            ]
        },
    )

    await rhapsody_session.submit_tasks([task])
    await rhapsody_session.wait_tasks([task], timeout=120.0)

    assert task["state"] == "DONE"
    result = task["return_value"]
    assert result["total"] == result["expected"], (
        f"Allreduce sum {result['total']} != expected {result['expected']}"
    )


@pytest.mark.mpi
@pytest.mark.asyncio
async def test_mpi_broadcast_single_node(rhapsody_session, topology):
    """Broadcast from rank 0 to all ranks on one node; verify all received."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    node = topology[0]
    ranks = 4

    task = ComputeTask(
        function=_mpi_broadcast_check,
        args=(42,),
        task_backend_specific_kwargs={
            "process_templates": [
                (
                    ranks,
                    {
                        "policy": Policy(
                            placement=Policy.Placement.HOST_NAME,
                            host_name=node["hostname"],
                        )
                    },
                )
            ]
        },
    )

    await rhapsody_session.submit_tasks([task])
    await rhapsody_session.wait_tasks([task], timeout=120.0)

    assert task["state"] == "DONE"
    assert task["return_value"] is True, "Not all ranks received the broadcast value"


# ---------------------------------------------------------------------------
# Multi-node MPI tests
# ---------------------------------------------------------------------------


@pytest.mark.mpi
@pytest.mark.multi_node
@pytest.mark.asyncio
async def test_mpi_gather_across_nodes(rhapsody_session, topology):
    """1 rank per node, gather hostnames to rank 0; confirm each node is represented."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    ranks_per_node = 1
    expected_hosts = {node["hostname"] for node in topology}

    task = ComputeTask(
        function=_mpi_gather_hostnames,
        task_backend_specific_kwargs={
            "process_templates": [
                (
                    ranks_per_node,
                    {
                        "policy": Policy(
                            placement=Policy.Placement.HOST_NAME,
                            host_name=node["hostname"],
                        )
                    },
                )
                for node in topology
            ]
        },
    )

    await rhapsody_session.submit_tasks([task])
    await rhapsody_session.wait_tasks([task], timeout=180.0)

    assert task["state"] == "DONE"
    result = task["return_value"]
    assert result["size"] == len(topology) * ranks_per_node
    observed_hosts = set(result["hostnames"])
    assert observed_hosts == expected_hosts, (
        f"Missing nodes in MPI gather. Expected: {expected_hosts}, got: {observed_hosts}"
    )


@pytest.mark.mpi
@pytest.mark.multi_node
@pytest.mark.asyncio
async def test_mpi_allreduce_across_nodes(rhapsody_session, topology):
    """Multi-node allreduce: 2 ranks per node, verify global sum."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    ranks_per_node = 2
    input_val = 3
    total_ranks = len(topology) * ranks_per_node

    task = ComputeTask(
        function=_mpi_allreduce_sum,
        args=(input_val,),
        task_backend_specific_kwargs={
            "process_templates": [
                (
                    ranks_per_node,
                    {
                        "policy": Policy(
                            placement=Policy.Placement.HOST_NAME,
                            host_name=node["hostname"],
                        )
                    },
                )
                for node in topology
            ]
        },
    )

    await rhapsody_session.submit_tasks([task])
    await rhapsody_session.wait_tasks([task], timeout=180.0)

    assert task["state"] == "DONE"
    result = task["return_value"]
    assert result["total"] == input_val * total_ranks, (
        f"Expected allreduce sum {input_val * total_ranks}, got {result['total']}"
    )


@pytest.mark.mpi
@pytest.mark.multi_node
@pytest.mark.asyncio
async def test_multiple_independent_mpi_jobs(rhapsody_session, topology):
    """Submit multiple independent MPI jobs concurrently; all must complete."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    n_jobs = 4
    tasks = []

    for _ in range(n_jobs):
        task = ComputeTask(
            function=_mpi_gather_hostnames,
            task_backend_specific_kwargs={
                "process_templates": [
                    (
                        1,
                        {
                            "policy": Policy(
                                placement=Policy.Placement.HOST_NAME,
                                host_name=node["hostname"],
                            )
                        },
                    )
                    for node in topology
                ]
            },
        )
        tasks.append(task)

    await rhapsody_session.submit_tasks(tasks)
    await rhapsody_session.wait_tasks(tasks, timeout=300.0)

    failed = [t for t in tasks if t["state"] != "DONE"]
    assert not failed, f"{len(failed)}/{n_jobs} concurrent MPI jobs failed"
