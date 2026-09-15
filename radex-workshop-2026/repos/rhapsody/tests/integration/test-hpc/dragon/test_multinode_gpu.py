"""Category 1 — Multi-node, multi-GPU execution tests.

Purpose:
    Validate that RHAPSODY can distribute tasks across every node in the
    allocation, that each task lands on the correct node, and that GPU
    affinity is correctly enforced when GPUs are present.

Execution strategy:
    Pinning is done via HOST_NAME Policy inside process_template.
    Dragon's batch.process() returns the process exit code (0), not the
    Python function return value — so node identity is verified via stdout
    (e.g. /bin/hostname) rather than return_value.

Resource requirements:
    >= 2 nodes (multi_node mark), >= 1 GPU per node (gpu mark for GPU tests).

Expected outcomes:
    - Every task reaches DONE state.
    - stdout of each task contains the hostname it was pinned to.
    - Each GPU task has non-empty CUDA_VISIBLE_DEVICES in its stdout.

Run:
    dragon python3 -m pytest test-hpc/test_multinode_gpu.py -v
"""

import pytest

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.multi_node
@pytest.mark.asyncio
async def test_one_task_per_node(rhapsody_session, topology):
    """Each node receives one pinned executable task; stdout contains the hostname."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    tasks = [
        ComputeTask(
            executable="/bin/hostname",
            task_backend_specific_kwargs={
                "process_template": {
                    "policy": Policy(
                        placement=Policy.Placement.HOST_NAME,
                        host_name=node["hostname"],
                    )
                }
            },
        )
        for node in topology
    ]

    await rhapsody_session.submit_tasks(tasks)
    results = await rhapsody_session.wait_tasks(tasks, timeout=120.0)

    assert len(results) == len(topology)
    for task, node in zip(results, topology):
        assert task["state"] == "DONE", f"Task on {node['hostname']} did not reach DONE"
        stdout = (task.get("stdout") or "").strip()
        assert node["hostname"] in stdout, (
            f"Expected hostname {node['hostname']!r} in stdout, got {stdout!r}"
        )


@pytest.mark.multi_node
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_one_task_per_gpu_across_nodes(rhapsody_session, gpu_nodes):
    """One task per GPU node; stdout shows non-empty CUDA_VISIBLE_DEVICES."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    tasks = []
    expected = []

    for node in gpu_nodes:
        task = ComputeTask(
            executable="/bin/bash",
            arguments=["-c", "echo $CUDA_VISIBLE_DEVICES"],
            task_backend_specific_kwargs={
                "process_template": {
                    "policy": Policy(
                        placement=Policy.Placement.HOST_NAME,
                        host_name=node["hostname"],
                        gpu_affinity=node["gpus"],
                    )
                }
            },
        )
        tasks.append(task)
        expected.append(node)

    await rhapsody_session.submit_tasks(tasks)
    results = await rhapsody_session.wait_tasks(tasks, timeout=120.0)

    assert len(results) == len(gpu_nodes)
    for task, node in zip(results, expected):
        assert task["state"] == "DONE"
        stdout = (task.get("stdout") or "").strip()
        assert stdout not in ("", "NOT_SET"), (
            f"Node {node['hostname']}: CUDA_VISIBLE_DEVICES was empty in stdout — "
            f"GPU affinity not enforced"
        )


@pytest.mark.multi_node
@pytest.mark.asyncio
async def test_concurrent_submission_all_nodes(rhapsody_session, topology):
    """Submit tasks to all nodes in one call; all complete and each returns its hostname."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    tasks = [
        ComputeTask(
            executable="/bin/hostname",
            task_backend_specific_kwargs={
                "process_template": {
                    "policy": Policy(
                        placement=Policy.Placement.HOST_NAME,
                        host_name=node["hostname"],
                    )
                }
            },
        )
        for node in topology
    ]

    await rhapsody_session.submit_tasks(tasks)
    await rhapsody_session.wait_tasks(tasks, timeout=120.0)

    for task, node in zip(tasks, topology):
        assert task["state"] == "DONE"
        stdout = (task.get("stdout") or "").strip()
        assert node["hostname"] in stdout, (
            f"Pinning violated: expected {node['hostname']!r} in stdout, got {stdout!r}"
        )
