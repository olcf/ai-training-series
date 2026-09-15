"""Category 3 — Task pinning and affinity tests.

Purpose:
    Verify that placement constraints are strictly enforced by Dragon and
    respected by RHAPSODY's process_template forwarding.

Execution strategy:
    All pinning tests use executables (not Python functions) because
    Dragon's batch.process() returns the process exit code, not the
    Python function return value.  Node/GPU identity is verified via stdout.

    - Node pinning: /bin/hostname → stdout contains expected hostname
    - GPU pinning:  'echo $CUDA_VISIBLE_DEVICES' → stdout is non-empty
    - Combined:     'echo $(hostname):$CUDA_VISIBLE_DEVICES' → parsed from stdout

Resource requirements:
    Node pinning: single node minimum.
    Multi-node pinning: multi_node mark.
    GPU tests: gpu mark.

Expected outcomes:
    - Each task's stdout contains the hostname/GPU it was pinned to.
    - No task lands outside its declared placement constraint.

Run:
    dragon python3 -m pytest test-hpc/test_pinning.py -v
"""

import pytest

# ---------------------------------------------------------------------------
# Node pinning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pin_to_each_node_sequentially(rhapsody_session, topology):
    """One /bin/hostname task per node; stdout must match the pinned hostname."""
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


@pytest.mark.multi_node
@pytest.mark.asyncio
async def test_exclusive_node_pinning(rhapsody_session, topology):
    """Tasks pinned to node[0] and node[1] must land on different nodes."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    node_a = topology[0]
    node_b = topology[1]

    task_a = ComputeTask(
        executable="/bin/hostname",
        task_backend_specific_kwargs={
            "process_template": {
                "policy": Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=node_a["hostname"],
                )
            }
        },
    )
    task_b = ComputeTask(
        executable="/bin/hostname",
        task_backend_specific_kwargs={
            "process_template": {
                "policy": Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=node_b["hostname"],
                )
            }
        },
    )

    await rhapsody_session.submit_tasks([task_a, task_b])
    await rhapsody_session.wait_tasks([task_a, task_b], timeout=60.0)

    stdout_a = (task_a.get("stdout") or "").strip()
    stdout_b = (task_b.get("stdout") or "").strip()

    assert node_a["hostname"] in stdout_a, f"task_a landed on wrong node: {stdout_a!r}"
    assert node_b["hostname"] in stdout_b, f"task_b landed on wrong node: {stdout_b!r}"
    assert stdout_a != stdout_b, "Both tasks reported the same hostname — pinning had no effect"


# ---------------------------------------------------------------------------
# GPU affinity
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_pin_to_specific_gpu(rhapsody_session, gpu_nodes):
    """One task per GPU on the first GPU node; each stdout shows its GPU assignment."""
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
    await rhapsody_session.wait_tasks(tasks, timeout=120.0)

    for task in tasks:
        assert task["state"] == "DONE"
        stdout = (task.get("stdout") or "").strip()
        assert stdout not in ("", "NOT_SET"), (
            "GPU affinity not enforced: CUDA_VISIBLE_DEVICES is empty in stdout"
        )


@pytest.mark.multi_node
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_round_robin_gpu_across_nodes(rhapsody_session, gpu_nodes):
    """One task per (node, GPU) pair; stdout contains 'hostname:CUDA_VISIBLE_DEVICES'."""
    from dragon.infrastructure.policy import Policy

    from rhapsody.api import ComputeTask

    tasks = []
    expected_hosts = []

    for node in gpu_nodes:
        for gpu_id in node["gpus"]:
            task = ComputeTask(
                executable="/bin/bash",
                arguments=["-c", "echo $(hostname):$CUDA_VISIBLE_DEVICES"],
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
            tasks.append(task)
            expected_hosts.append(node["hostname"])

    await rhapsody_session.submit_tasks(tasks)
    await rhapsody_session.wait_tasks(tasks, timeout=180.0)

    for task, exp_host in zip(tasks, expected_hosts):
        assert task["state"] == "DONE"
        stdout = (task.get("stdout") or "").strip()
        # stdout format: "hostname:CUDA_VISIBLE_DEVICES"
        assert ":" in stdout, f"Unexpected stdout format: {stdout!r}"
        ret_host, ret_gpu = stdout.split(":", 1)
        assert exp_host in ret_host, f"Hostname mismatch: expected {exp_host!r}, got {ret_host!r}"
        assert ret_gpu not in ("", "NOT_SET"), (
            f"GPU affinity missing on node {exp_host!r}: CUDA_VISIBLE_DEVICES={ret_gpu!r}"
        )


# ---------------------------------------------------------------------------
# Executable pinning (baseline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executable_pinned_to_each_node(rhapsody_session, topology):
    """/bin/hostname pinned to each node; stdout contains the expected hostname."""
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
            f"Expected hostname {node['hostname']!r} in stdout, got {stdout!r}"
        )
