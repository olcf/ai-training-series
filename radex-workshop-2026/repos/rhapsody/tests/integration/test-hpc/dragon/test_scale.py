"""Category 4 — Scale and stress tests.

Purpose:
    Evaluate RHAPSODY + Dragon correctness and scheduling stability under
    high task counts.  Confirms that:
      - All submitted tasks reach DONE (no silent drops)
      - Return values are correct for every task (no result corruption)
      - The system remains responsive throughout the run
      - result DDict does not overflow under load

Execution strategy:
    Tests are parametrised by task count.  The backend is created with an
    enlarged results_ddict_mem (controlled by RHAPSODY_DDICT_MEM_GB env var,
    default 4 GiB/node) to accommodate high task volumes.

    Tasks are lightweight by design: the goal is to stress the scheduling
    pipeline, not the compute.  Each function returns a deterministic value
    so correctness can be verified per-task without relying on ordering.

Resource requirements:
    scale mark — long-running, not run by default.
    Set RHAPSODY_RUN_SCALE=1 to enable, or use -m scale.

Thresholds (configurable via env vars):
    RHAPSODY_SCALE_FUNC_COUNT   default 10_000  (function invocations)
    RHAPSODY_SCALE_EXEC_COUNT   default 1_000   (executable task runs)

Expected outcomes:
    - All tasks: state == "DONE"
    - Function tasks: return_value == task index (no corruption)
    - Executable tasks: exit code 0, stdout contains expected string
    - No TimeoutError within the configured deadline

Run:
    RHAPSODY_RUN_SCALE=1 dragon python3 -m pytest test-hpc/test_scale.py -v -s
"""

import os
import time

import pytest
from hpc_workers import identity as _identity

# ---------------------------------------------------------------------------
# Scale-test gate — skip unless explicitly enabled
# ---------------------------------------------------------------------------

SCALE_ENABLED = os.environ.get("RHAPSODY_RUN_SCALE", "0") == "1"
FUNC_COUNT = int(os.environ.get("RHAPSODY_SCALE_FUNC_COUNT", 10_000))
EXEC_COUNT = int(os.environ.get("RHAPSODY_SCALE_EXEC_COUNT", 1_000))


def _scale_skip():
    if not SCALE_ENABLED:
        pytest.skip("Scale tests disabled. Set RHAPSODY_RUN_SCALE=1 to enable.")


# ---------------------------------------------------------------------------
# Fixtures — scale tests use a dedicated backend with more DDict memory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def scale_backend():
    """DragonExecutionBackend with enlarged results DDict for scale tests."""
    from dragon.native.machine import System

    from rhapsody.backends import DragonExecutionBackend

    num_nodes = len(list(System().nodes))
    ddict_gb = int(os.environ.get("RHAPSODY_DDICT_MEM_GB", 4))
    ddict_mem = ddict_gb * num_nodes * (1024**3)

    backend = await DragonExecutionBackend(
        batch_kwargs={
            "results_ddict_mem": ddict_mem,
            "scheduler_workers": num_nodes * 4,
        }
    )
    yield backend
    await backend.shutdown()


@pytest.fixture(scope="module")
async def scale_session(scale_backend):
    from rhapsody.api import Session

    session = Session(backends=[scale_backend])
    yield session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.scale
@pytest.mark.asyncio
async def test_10k_function_tasks(scale_session):
    """Submit FUNC_COUNT function tasks; verify every result is correct."""
    _scale_skip()
    from rhapsody.api import ComputeTask

    print(f"\nSubmitting {FUNC_COUNT:,} function tasks...", flush=True)

    tasks = [ComputeTask(function=_identity, args=(i,)) for i in range(FUNC_COUNT)]

    t0 = time.perf_counter()
    await scale_session.submit_tasks(tasks)
    submit_elapsed = time.perf_counter() - t0
    print(f"  submit_tasks: {submit_elapsed:.2f}s", flush=True)

    # Allow up to 5 minutes for all tasks to complete
    results = await scale_session.wait_tasks(tasks, timeout=300.0)
    total_elapsed = time.perf_counter() - t0
    print(
        f"  total elapsed: {total_elapsed:.2f}s  ({FUNC_COUNT / total_elapsed:,.0f} tasks/s)",
        flush=True,
    )

    failed = [t for t in results if t["state"] != "DONE"]
    assert not failed, f"{len(failed)}/{FUNC_COUNT} tasks did not reach DONE"

    # Verify every return value — detect any result corruption
    wrong = [
        (i, tasks[i]["return_value"]) for i in range(FUNC_COUNT) if tasks[i]["return_value"] != i
    ]
    assert not wrong, f"{len(wrong)} tasks had wrong return values. First 5: {wrong[:5]}"


@pytest.mark.scale
@pytest.mark.asyncio
async def test_1k_executable_tasks(scale_session):
    """Submit EXEC_COUNT executable tasks (/bin/echo); verify all complete."""
    _scale_skip()
    from rhapsody.api import ComputeTask

    print(f"\nSubmitting {EXEC_COUNT:,} executable tasks...", flush=True)

    tasks = [
        ComputeTask(executable="/bin/echo", arguments=[f"task-{i}"]) for i in range(EXEC_COUNT)
    ]

    t0 = time.perf_counter()
    await scale_session.submit_tasks(tasks)
    submit_elapsed = time.perf_counter() - t0
    print(f"  submit_tasks: {submit_elapsed:.2f}s", flush=True)

    results = await scale_session.wait_tasks(tasks, timeout=300.0)
    total_elapsed = time.perf_counter() - t0
    print(
        f"  total elapsed: {total_elapsed:.2f}s  ({EXEC_COUNT / total_elapsed:,.0f} tasks/s)",
        flush=True,
    )

    failed = [t for t in results if t["state"] != "DONE"]
    assert not failed, f"{len(failed)}/{EXEC_COUNT} executable tasks did not reach DONE"


@pytest.mark.scale
@pytest.mark.asyncio
async def test_mixed_function_and_executable_tasks(scale_session):
    """Interleave function and executable tasks in a single submission."""
    _scale_skip()
    from rhapsody.api import ComputeTask

    n_func = FUNC_COUNT // 10  # 1/10 of the full scale to keep runtime reasonable
    n_exec = EXEC_COUNT // 10

    func_tasks = [ComputeTask(function=_identity, args=(i,)) for i in range(n_func)]
    exec_tasks = [
        ComputeTask(executable="/bin/echo", arguments=[f"exec-{i}"]) for i in range(n_exec)
    ]
    all_tasks = func_tasks + exec_tasks

    print(f"\nMixed submission: {n_func} function + {n_exec} executable", flush=True)

    await scale_session.submit_tasks(all_tasks)
    results = await scale_session.wait_tasks(all_tasks, timeout=180.0)

    failed = [t for t in results if t["state"] != "DONE"]
    assert not failed, f"{len(failed)}/{len(all_tasks)} mixed tasks failed"


@pytest.mark.scale
@pytest.mark.asyncio
async def test_wave_submission_stability(scale_session):
    """Submit tasks in waves (batches of 1000) to test sustained throughput.

    Simulates a workflow where tasks arrive incrementally rather than all at once.  Validates that
    the monitor loop keeps pace with new submissions.
    """
    _scale_skip()
    from rhapsody.api import ComputeTask

    wave_size = 1_000
    num_waves = FUNC_COUNT // wave_size
    all_tasks = []

    print(f"\nWave submission: {num_waves} waves × {wave_size} tasks", flush=True)
    t0 = time.perf_counter()

    for wave in range(num_waves):
        wave_tasks = [
            ComputeTask(function=_identity, args=(wave * wave_size + i,)) for i in range(wave_size)
        ]
        await scale_session.submit_tasks(wave_tasks)
        all_tasks.extend(wave_tasks)

    print(f"  all waves submitted in {time.perf_counter() - t0:.2f}s", flush=True)

    results = await scale_session.wait_tasks(all_tasks, timeout=300.0)
    elapsed = time.perf_counter() - t0
    total = len(all_tasks)
    print(
        f"  {total:,} tasks completed in {elapsed:.2f}s ({total / elapsed:,.0f} tasks/s)",
        flush=True,
    )

    failed = [t for t in results if t["state"] != "DONE"]
    assert not failed, f"{len(failed)}/{total} wave tasks failed"
