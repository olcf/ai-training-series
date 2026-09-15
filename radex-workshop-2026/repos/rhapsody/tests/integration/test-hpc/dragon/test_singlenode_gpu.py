"""Category 2 — Single-node, single-GPU tests.

Purpose:
    Validate correctness on the minimal execution configuration: one node,
    one GPU.

Execution strategy:
    Two complementary patterns are used:

    1. Pinned executable tasks (process_template + Policy):
       Dragon's batch.process() returns the process exit code, not a Python
       return value.  Node/GPU verification is done via stdout.

    2. Unpinned function tasks (no process_template):
       batch.function() captures the Python return value correctly.
       Used to validate argument passing and return value integrity.

Resource requirements:
    Always runnable (single node is always present).
    GPU sub-tests require >= 1 GPU (gpu mark).

Expected outcomes:
    - Pinned executable tasks: state == DONE, stdout contains expected string.
    - Unpinned function tasks: state == DONE, return_value matches expectation.
    - GPU tasks: non-empty CUDA_VISIBLE_DEVICES in stdout.

Run:
    dragon python3 -m pytest test-hpc/test_singlenode_gpu.py -v
"""

import pytest
from hpc_workers import add as _add

# ---------------------------------------------------------------------------
# Pinned executable tests — verify node placement via stdout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executable_pinned_to_first_node(rhapsody_session, topology):
    """/bin/hostname pinned to the first node; stdout contains its hostname."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    node = topology[0]
    task = ComputeTask(
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

    await rhapsody_session.submit_tasks([task])
    results = await rhapsody_session.wait_tasks([task], timeout=60.0)

    assert results[0]["state"] == "DONE"
    stdout = (results[0].get("stdout") or "").strip()
    assert node["hostname"] in stdout, f"Expected {node['hostname']!r} in stdout, got {stdout!r}"


@pytest.mark.asyncio
async def test_executable_echo_on_first_node(rhapsody_session, topology):
    """/bin/echo with arguments runs on first node and stdout is captured."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    node = topology[0]
    task = ComputeTask(
        executable="/bin/echo",
        arguments=["hello-rhapsody"],
        task_backend_specific_kwargs={
            "process_template": {
                "policy": Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=node["hostname"],
                )
            }
        },
    )

    await rhapsody_session.submit_tasks([task])
    results = await rhapsody_session.wait_tasks([task], timeout=60.0)

    assert results[0]["state"] == "DONE"
    assert "hello-rhapsody" in (results[0].get("stdout") or "")


# ---------------------------------------------------------------------------
# Unpinned function tests — verify return value capture via batch.function()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_function_return_value(rhapsody_session):
    """batch.function() correctly captures the Python return value."""
    from rhapsody.api import ComputeTask

    task = ComputeTask(function=_add, args=(21, 21))

    await rhapsody_session.submit_tasks([task])
    results = await rhapsody_session.wait_tasks([task], timeout=60.0)

    assert results[0]["state"] == "DONE"
    assert results[0]["return_value"] == 42


@pytest.mark.asyncio
async def test_multiple_function_tasks_return_correct_values(rhapsody_session):
    """Multiple concurrent function tasks all return correct independent values."""
    from rhapsody.api import ComputeTask

    pairs = [(i, i * 2) for i in range(10)]
    tasks = [ComputeTask(function=_add, args=(a, b)) for a, b in pairs]

    await rhapsody_session.submit_tasks(tasks)
    results = await rhapsody_session.wait_tasks(tasks, timeout=60.0)

    for task, (a, b) in zip(results, pairs):
        assert task["state"] == "DONE"
        assert task["return_value"] == a + b, (
            f"_add({a}, {b}) returned {task['return_value']!r}, expected {a + b}"
        )


# ---------------------------------------------------------------------------
# GPU affinity tests — verify via stdout
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_single_gpu_affinity_on_first_node(rhapsody_session, gpu_nodes):
    """Task pinned to the first GPU of the first GPU node; stdout shows CUDA_VISIBLE_DEVICES."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    node = gpu_nodes[0]
    first_gpu = [node["gpus"][0]]

    task = ComputeTask(
        executable="/bin/bash",
        arguments=["-c", "echo $CUDA_VISIBLE_DEVICES"],
        task_backend_specific_kwargs={
            "process_template": {
                "policy": Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=node["hostname"],
                    gpu_affinity=first_gpu,
                )
            }
        },
    )

    await rhapsody_session.submit_tasks([task])
    results = await rhapsody_session.wait_tasks([task], timeout=60.0)

    assert results[0]["state"] == "DONE"
    stdout = (results[0].get("stdout") or "").strip()
    assert stdout not in ("", "NOT_SET"), (
        f"CUDA_VISIBLE_DEVICES was {stdout!r} — GPU affinity not enforced"
    )


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_all_gpus_on_first_node(rhapsody_session, gpu_nodes):
    """One task per GPU on the first GPU node; each stdout shows its assigned GPU."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    node = gpu_nodes[0]
    tasks = [
        ComputeTask(
            executable="/bin/bash",
            arguments=["-c", "echo $CUDA_VISIBLE_DEVICES"],
            task_backend_specific_kwargs={
                "process_template": {
                    "policy": Policy(
                        placement=Policy.Placement.HOST_NAME,
                        host_name=node["hostname"],
                        gpu_affinity=[gpu_id],
                    )
                }
            },
        )
        for gpu_id in node["gpus"]
    ]

    await rhapsody_session.submit_tasks(tasks)
    results = await rhapsody_session.wait_tasks(tasks, timeout=120.0)

    for task in results:
        assert task["state"] == "DONE"
        stdout = (task.get("stdout") or "").strip()
        assert stdout not in ("", "NOT_SET"), (
            "At least one GPU task had empty CUDA_VISIBLE_DEVICES in stdout"
        )
