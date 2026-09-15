"""Tests for DragonExecutionBackend using Session.

Structure
---------
Shared behavior tests (1-15)
    Parametrized against dragon. Verify observable task execution semantics.

Dragon integration tests — process_template / process_templates routing
    Use a dedicated ``session_dragon`` fixture so only one V3 session is created.

Dragon unit tests — constructor and internal methods
    Verify the API surface with Batch mocked out. No Dragon cluster required.

Run with:
    dragon python -m pytest tests/unit/test_backend_execution_dragon.py -v
"""

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio

from rhapsody import ComputeTask
from rhapsody.api import Session
from rhapsody.backends.discovery import get_backend

# Skip the entire module when the Dragon runtime is not installed.
pytest.importorskip("dragon", reason="Dragon is required for Dragon backend tests")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module", params=["dragon"])
def backend_name(request):
    """Backend name for shared behavior tests (Dragon backend only)."""
    return request.param


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def session(backend_name):
    """Session with the parametrized Dragon backend, reused across shared tests."""
    backend_instance = await get_backend(backend_name)
    session_instance = Session(backends=[backend_instance])
    yield session_instance
    await session_instance.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def session_dragon():
    """Session backed exclusively by DragonExecutionBackend.

    Used by Dragon-specific tests so only one session is created instead of three.
    """
    backend_instance = await get_backend("dragon")
    session_instance = Session(backends=[backend_instance])
    yield session_instance
    await session_instance.close()


@pytest.fixture
def backend_dragon():
    """DragonExecutionBackend with Batch fully mocked — no Dragon cluster required.

    Suitable for unit tests that verify constructor wiring, internal callback helpers, and method
    delegation without running actual Dragon workers.
    """
    from rhapsody.backends.execution.dragon import DragonExecutionBackend

    mock_batch = MagicMock()
    mock_batch.num_workers = 16
    mock_batch.num_managers = 2

    with patch("rhapsody.backends.execution.dragon.Batch", return_value=mock_batch):
        backend = DragonExecutionBackend()

    backend._callback_func = MagicMock()
    backend._loop = asyncio.new_event_loop()
    return backend


def _make_task_dict(fn, args=(), kwargs=None, backend_specific=None):
    """Build a minimal task dict in the format expected by build_task."""
    import uuid

    return {
        "uid": f"task.test-{uuid.uuid4().hex[:8]}",
        "function": fn,
        "args": list(args),
        "kwargs": kwargs or {},
        "name": "test",
        "task_backend_specific_kwargs": backend_specific or {},
    }


# ============================================================================
# Test 1: Single Executable Task
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_single_executable(session):
    """Test executing a single shell command task."""
    task = ComputeTask(executable="echo", arguments=["Hello Dragon"])

    await session.submit_tasks([task])
    results = await session.wait_tasks([task])

    assert results[0].uid.startswith("task.")
    assert results[0].state == "DONE"
    assert "Hello Dragon" in results[0].get("stdout", "")


# ============================================================================
# Test 2: Single Function Task
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_single_function(session):
    """Test executing a single Python function task."""

    async def simple_function(x: int) -> int:
        return x * 2

    task = ComputeTask(function=simple_function, args=(21,))

    await session.submit_tasks([task])
    results = await session.wait_tasks([task])

    assert results[0].uid.startswith("task.")
    assert results[0].state == "DONE"
    assert results[0].get("return_value") == 42


# ============================================================================
# Test 3: Task with Arguments
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_task_with_args(session):
    """Test task execution with multiple arguments."""
    task = ComputeTask(executable="/bin/echo", arguments=["arg1", "arg2", "arg3"])

    await session.submit_tasks([task])
    results = await session.wait_tasks([task])

    assert results[0].state == "DONE"
    stdout = results[0].get("stdout", "")
    assert "arg1" in stdout and "arg2" in stdout and "arg3" in stdout


# ============================================================================
# Test 4: Task Failure Handling
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_task_failure(session):
    """Test that failed tasks are properly reported."""
    task = ComputeTask(executable="/bin/false")

    await session.submit_tasks([task])
    results = await session.wait_tasks([task])

    assert results[0].state == "FAILED"


# ============================================================================
# Test 5: Function with Exception
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_function_exception(session):
    """Test that function exceptions are properly handled."""

    async def failing_function():
        raise ValueError("Intentional test failure")

    task = ComputeTask(function=failing_function, args=())

    await session.submit_tasks([task])
    results = await session.wait_tasks([task])

    assert results[0].state == "FAILED"
    assert "exception" in results[0]


# ============================================================================
# Test 6: Two Independent Tasks (Parallel Execution)
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_two_independent_tasks(session):
    """Test executing two independent tasks in parallel."""
    tasks = [
        ComputeTask(executable="echo", arguments=["Task A"]),
        ComputeTask(executable="echo", arguments=["Task B"]),
    ]

    await session.submit_tasks(tasks)
    results = await session.wait_tasks(tasks)

    assert len(results) == 2
    for result in results:
        assert result.uid.startswith("task.")
        assert result.state == "DONE"

    outputs = [r.get("stdout", "") for r in results]
    assert any("Task A" in out for out in outputs)
    assert any("Task B" in out for out in outputs)


# ============================================================================
# Test 7: Mixed Success and Failure
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_mixed_success_failure(session):
    """Test handling tasks where some succeed and some fail."""
    tasks = [ComputeTask(executable="/bin/true"), ComputeTask(executable="/bin/false")]

    await session.submit_tasks(tasks)
    results = await session.wait_tasks(tasks)

    assert len(results) == 2
    states = [r.state for r in results]
    assert "DONE" in states
    assert "FAILED" in states


# ============================================================================
# Test 8: Function with Return Value
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_function_return_value(session):
    """Test that function return values are properly captured."""

    async def compute_function(a: int, b: int) -> dict:
        return {"sum": a + b, "product": a * b, "inputs": [a, b]}

    task = ComputeTask(function=compute_function, args=(5, 7))

    await session.submit_tasks([task])
    results = await session.wait_tasks([task])

    assert results[0].state == "DONE"
    return_value = results[0].get("return_value")
    assert return_value["sum"] == 12
    assert return_value["product"] == 35


# ============================================================================
# Test 9: Stdout Capture
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_stdout_capture(session):
    """Test that stdout is properly captured."""
    import sys

    task = ComputeTask(
        executable=sys.executable,
        arguments=["-c", "print('Line 1'); print('Line 2'); print('Line 3')"],
    )

    await session.submit_tasks([task])
    results = await session.wait_tasks([task])

    assert results[0].state == "DONE"
    stdout = results[0].get("stdout", "")
    assert "Line 1" in stdout
    assert "Line 2" in stdout
    assert "Line 3" in stdout


# ============================================================================
# Test 10: Task Cancellation
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_task_cancellation(session):
    """Test cancelling a task before completion."""
    task = ComputeTask(executable="/bin/sleep", arguments=["10"])

    await session.submit_tasks([task])
    await asyncio.sleep(0.5)

    backend = next(iter(session.backends.values()))
    cancelled = await backend.cancel_task(task.uid)
    assert cancelled is True


# ============================================================================
# Test 11: Backend State
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_backend_state(session):
    """Backend state is RUNNING while tasks are being processed and after they complete."""
    backend = next(iter(session.backends.values()))

    task = ComputeTask(executable="echo", arguments=["state-test"])
    await session.submit_tasks([task])

    # state() must return "RUNNING" from the moment the first task was submitted
    # (earlier tests in this module-scoped session have already triggered RUNNING)
    state_mid = await backend.state()
    assert state_mid == "RUNNING", f"Expected RUNNING during processing, got {state_mid!r}"

    await session.wait_tasks([task])

    state_after = await backend.state()
    assert state_after == "RUNNING", f"Expected RUNNING after completion, got {state_after!r}"


# ============================================================================
# Test 12: Multiple Submissions (Sequential)
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_sequential_submissions(session):
    """Test submitting tasks in multiple batches."""
    task1 = ComputeTask(executable="echo", arguments=["Batch 1"])

    await session.submit_tasks([task1])
    results1 = await session.wait_tasks([task1])
    assert results1[0].state == "DONE"
    assert "Batch 1" in results1[0].get("stdout", "")

    task2 = ComputeTask(executable="echo", arguments=["Batch 2"])

    await session.submit_tasks([task2])
    results2 = await session.wait_tasks([task2])
    assert results2[0].state == "DONE"
    assert "Batch 2" in results2[0].get("stdout", "")


# ============================================================================
# Test 13: Function with Kwargs
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_function_with_kwargs(session):
    """Test function execution with keyword arguments."""

    async def function_with_kwargs(x: int, y: int = 10, z: int = 20) -> int:
        return x + y + z

    task = ComputeTask(function=function_with_kwargs, args=(5,), kwargs={"y": 15, "z": 25})

    await session.submit_tasks([task])
    results = await session.wait_tasks([task])

    assert results[0].state == "DONE"
    assert results[0].get("return_value") == 45


# ============================================================================
# Test 14: Empty Task List
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_empty_task_list(session):
    """Test handling of empty task list."""
    await session.submit_tasks([])


# ============================================================================
# Test 15: Task UID Uniqueness
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_task_uid_uniqueness(session):
    """Test that auto-generated UIDs are unique."""
    tasks = [
        ComputeTask(executable="echo", arguments=["Task 1"]),
        ComputeTask(executable="echo", arguments=["Task 2"]),
    ]

    await session.submit_tasks(tasks)
    results = await session.wait_tasks(tasks)

    assert len(results) == 2
    uids = [r.uid for r in results]
    assert len(set(uids)) == 2
    assert all(uid.startswith("task.") for uid in uids)


# ============================================================================
# Dragon integration tests — per-task cwd and process_template routing
# ============================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_executable_with_cwd_via_process_template(session_dragon):
    """Test that cwd is honoured when set via process_template (Dragon backend only)."""
    import sys

    task = ComputeTask(
        executable=sys.executable,
        arguments=["-c", "import os; print(os.getcwd())"],
        task_backend_specific_kwargs={"process_template": {"cwd": "/tmp"}},
    )

    await session_dragon.submit_tasks([task])
    results = await session_dragon.wait_tasks([task])

    assert results[0].state == "DONE"
    assert "/tmp" in results[0].get("stdout", "")


@pytest.mark.asyncio
async def test_process_template_cwd_built_and_passed(backend_dragon):
    """Test A: process_template with cwd produces a ProcessTemplate with correct cwd.

    Uses backend_dragon (mocked Batch) — no Dragon cluster required.  The code calls
    batch.options(...).process(pt); we capture pt via a mock proxy on batch.options.
    """
    from dragon.native.process import ProcessTemplate

    captured_pt = []
    mock_proxy = MagicMock()
    mock_proxy.process.side_effect = lambda pt: captured_pt.append(pt) or MagicMock()

    task = _make_task_dict(lambda: None, backend_specific={"process_template": {"cwd": "/tmp"}})

    with patch.object(backend_dragon.batch, "options", return_value=mock_proxy):
        await backend_dragon.build_task(task)

    assert len(captured_pt) == 1
    pt = captured_pt[0]
    assert isinstance(pt, ProcessTemplate)
    assert pt.cwd == "/tmp"


@pytest.mark.asyncio
async def test_process_template_policy_gpu_affinity_built_and_passed(backend_dragon):
    """Test B: process_template with policy(gpu_affinity) produces ProcessTemplate with correct policy."""
    from dragon.infrastructure.policy import Policy
    from dragon.native.process import ProcessTemplate

    captured_pt = []
    mock_proxy = MagicMock()
    mock_proxy.process.side_effect = lambda pt: captured_pt.append(pt) or MagicMock()

    policy = Policy(gpu_affinity=[0, 1, 2, 3])
    task = _make_task_dict(lambda: None, backend_specific={"process_template": {"policy": policy}})

    with patch.object(backend_dragon.batch, "options", return_value=mock_proxy):
        await backend_dragon.build_task(task)

    assert len(captured_pt) == 1
    pt = captured_pt[0]
    assert isinstance(pt, ProcessTemplate)
    assert pt.policy is policy
    assert pt.policy.gpu_affinity == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_process_template_empty_dict_uses_process_mode(backend_dragon):
    """Test C: process_template={} still routes to proxy.process(), not proxy.function().

    Regression: a truthiness check on the dict falls through on an empty dict; the
    ``is not None`` guard in build_task prevents this.
    """
    process_calls = []
    function_calls = []
    mock_proxy = MagicMock()
    mock_proxy.process.side_effect = lambda pt: process_calls.append(pt) or MagicMock()
    mock_proxy.function.side_effect = lambda fn, *a, **kw: function_calls.append(fn) or MagicMock()

    task = _make_task_dict(lambda: None, backend_specific={"process_template": {}})

    with patch.object(backend_dragon.batch, "options", return_value=mock_proxy):
        await backend_dragon.build_task(task)

    assert len(process_calls) == 1, "options().process() should have been called (Priority 2)"
    assert len(function_calls) == 0, (
        "options().function() must NOT be called when process_template is provided"
    )


@pytest.mark.asyncio
async def test_process_templates_list_built_and_passed_to_job(backend_dragon):
    """Test D: process_templates list produces correct (nranks, ProcessTemplate) tuples for job()."""
    from dragon.native.process import ProcessTemplate

    captured_args = []
    mock_proxy = MagicMock()
    mock_proxy.job.side_effect = lambda templates: captured_args.append(templates) or MagicMock()

    task = _make_task_dict(
        lambda: None,
        backend_specific={"process_templates": [(2, {"cwd": "/tmp"})]},
    )

    with patch.object(backend_dragon.batch, "options", return_value=mock_proxy):
        await backend_dragon.build_task(task)

    assert len(captured_args) == 1
    templates = captured_args[0]
    assert len(templates) == 1
    nranks, pt = templates[0]
    assert nranks == 2
    assert isinstance(pt, ProcessTemplate)
    assert pt.cwd == "/tmp"


@pytest.mark.asyncio
async def test_process_template_combined_spec_policy_cwd_args(backend_dragon):
    """Test E: process_template with policy + cwd + args all land on ProcessTemplate correctly."""
    import cloudpickle
    from dragon.infrastructure.policy import Policy
    from dragon.native.process import ProcessTemplate

    captured_pt = []
    mock_proxy = MagicMock()
    mock_proxy.process.side_effect = lambda pt: captured_pt.append(pt) or MagicMock()

    policy = Policy(gpu_affinity=[0])
    task = _make_task_dict(
        lambda x: x,
        args=(42,),
        kwargs={"flag": True},
        backend_specific={"process_template": {"policy": policy, "cwd": "/tmp"}},
    )

    with patch.object(backend_dragon.batch, "options", return_value=mock_proxy):
        await backend_dragon.build_task(task)

    assert len(captured_pt) == 1
    pt = captured_pt[0]
    assert isinstance(pt, ProcessTemplate)
    assert pt.policy is policy
    assert pt.cwd == "/tmp"
    # Dragon serialises (target, args, kwargs) into pt.argdata via cloudpickle.
    _, stored_args, stored_kwargs = cloudpickle.loads(pt.argdata)
    assert list(stored_args) == [42]
    assert stored_kwargs == {"flag": True}


# ============================================================================
# Dragon unit tests — constructor and internal methods (no Dragon cluster required)
# ============================================================================


def test_constructor_batch_kwargs_forwarded_verbatim():
    """batch_kwargs contents are splatted into Batch() unchanged."""
    from rhapsody.backends.execution.dragon import DragonExecutionBackend

    mock_batch = MagicMock()
    mock_batch.num_workers = 8
    mock_batch.num_managers = 1

    kwargs = {"num_nodes": 4, "pool_nodes": 2, "disable_telem": True, "scheduler_workers": 2}

    with patch(
        "rhapsody.backends.execution.dragon.Batch", return_value=mock_batch
    ) as mock_batch_cls:
        backend = DragonExecutionBackend(batch_kwargs=kwargs)

    mock_batch_cls.assert_called_once_with(task_logs=True, **kwargs)
    assert backend.batch is mock_batch


def test_constructor_no_batch_kwargs_calls_batch_with_no_args():
    """DragonExecutionBackend() with no args calls Batch() with no args."""
    from rhapsody.backends.execution.dragon import DragonExecutionBackend

    mock_batch = MagicMock()
    mock_batch.num_workers = 4
    mock_batch.num_managers = 1

    with patch(
        "rhapsody.backends.execution.dragon.Batch", return_value=mock_batch
    ) as mock_batch_cls:
        DragonExecutionBackend()

    mock_batch_cls.assert_called_once_with(task_logs=True)


def test_constructor_rejects_bare_batch_params():
    """Batch params passed directly (not via batch_kwargs) raise TypeError.

    num_nodes, pool_nodes, disable_telemetry, and other old top-level params are no longer accepted
    as direct constructor arguments — they must go through batch_kwargs.
    """
    from rhapsody.backends.execution.dragon import DragonExecutionBackend

    mock_batch = MagicMock()
    mock_batch.num_workers = 8
    mock_batch.num_managers = 1

    with patch("rhapsody.backends.execution.dragon.Batch", return_value=mock_batch):
        with pytest.raises(TypeError):
            DragonExecutionBackend(num_nodes=4)
        with pytest.raises(TypeError):
            DragonExecutionBackend(pool_nodes=2)
        with pytest.raises(TypeError):
            DragonExecutionBackend(disable_telemetry=True)
        with pytest.raises(TypeError):
            DragonExecutionBackend(num_workers=4)
        with pytest.raises(TypeError):
            DragonExecutionBackend(disable_background_batching=True)
        with pytest.raises(TypeError):
            DragonExecutionBackend(disable_batch_submission=True)


@pytest.mark.asyncio
async def test_initial_state_is_initialized():
    """A freshly created backend reports INITIALIZED before any tasks are submitted."""
    from rhapsody.backends.execution.dragon import DragonExecutionBackend

    mock_batch = MagicMock()
    mock_batch.num_workers = 4
    mock_batch.num_managers = 1

    with patch("rhapsody.backends.execution.dragon.Batch", return_value=mock_batch):
        backend = DragonExecutionBackend()

    assert await backend.state() == "INITIALIZED"


def test_deliver_batch_success_stores_value_and_fires_done(backend_dragon):
    """_deliver_batch stores return_value on the task dict and fires the DONE callback."""
    uid = "task.unit-done"
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {"uid": uid, "description": task_desc}

    backend_dragon._deliver_batch([(uid, 42, None, False, "", "")])

    assert task_desc["return_value"] == 42
    assert task_desc["stdout"] == ""
    assert task_desc["stderr"] == ""
    backend_dragon._callback_func.assert_called_once_with(task_desc, "DONE")
    assert uid not in backend_dragon._task_registry


def test_deliver_batch_propagates_stdout_stderr(backend_dragon):
    """_deliver_batch stores stdout/stderr on task_desc when non-empty."""
    uid = "task.unit-done-out"
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {"uid": uid, "description": task_desc}

    backend_dragon._deliver_batch([(uid, "ok", None, False, "hello\n", "warn\n")])

    assert task_desc["stdout"] == "hello\n"
    assert task_desc["stderr"] == "warn\n"


def test_deliver_batch_failure_stores_exc_and_fires_failed(backend_dragon):
    """_deliver_batch stores the exception and stderr string, fires the FAILED callback."""
    uid = "task.unit-failed"
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {"uid": uid, "description": task_desc}
    exc = RuntimeError("something went wrong")

    # raised=True, tb=None: stderr falls back to str(exc)
    backend_dragon._deliver_batch([(uid, exc, None, True, "", "")])

    assert task_desc["exception"] is exc
    assert "something went wrong" in task_desc["stderr"]
    backend_dragon._callback_func.assert_called_once_with(task_desc, "FAILED")
    assert uid not in backend_dragon._task_registry


def test_deliver_batch_prefers_traceback_over_str_exc(backend_dragon):
    """_deliver_batch uses Dragon's traceback string when available."""
    uid = "task.unit-failed-tb"
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {"uid": uid, "description": task_desc}
    exc = RuntimeError("boom")
    tb = "Traceback (most recent call last):\n  File ...\nRuntimeError: boom"

    backend_dragon._deliver_batch([(uid, exc, tb, True, "", "")])

    assert task_desc["stderr"] == tb


def test_cancelled_task_skips_callback(backend_dragon):
    """_deliver_batch is a no-op for UIDs in _cancelled_tasks."""
    uid = "task.unit-cancelled"
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {"uid": uid, "description": task_desc}
    backend_dragon._cancelled_tasks.add(uid)

    backend_dragon._deliver_batch([(uid, 99, None, False, "", "")])

    backend_dragon._callback_func.assert_not_called()
    assert "return_value" not in task_desc
    assert uid not in backend_dragon._cancelled_tasks


def test_fence_delegates_to_batch(backend_dragon):
    """backend.wait() calls batch.fence() exactly once."""
    backend_dragon.wait()
    backend_dragon.batch.fence.assert_called_once()


def _run_monitor_one_cycle(
    backend, poll_tuid, monitored_entry=None, task_registry_entry=None, cancelled=False
):
    """Drive _monitor_loop for one completion cycle and return the captured mock_loop.

    - poll call 1 returns `poll_tuid` (the tuid to deliver)
    - poll call ≥2 sets _shutdown_event and returns None (draining + outer loop → exits)
    - backend._loop is replaced with a MagicMock so call_soon_threadsafe is capturable
    - If `monitored_entry` is given, it is inserted into _monitored_batches under poll_tuid
    - If `task_registry_entry` is given, it is inserted into _task_registry
    - If `cancelled` is True, `poll_tuid` is inserted into _cancelled_tuids so a miss is
      resolved immediately instead of being retried through the pending-tuid buffer
    """
    mock_loop = MagicMock()
    backend._loop = mock_loop

    if monitored_entry is not None:
        backend._monitored_batches[poll_tuid] = monitored_entry
    if task_registry_entry is not None:
        uid = task_registry_entry["uid"]
        backend._task_registry[uid] = task_registry_entry
    if cancelled:
        backend._cancelled_tuids.add(poll_tuid)

    calls = [0]

    def poll_side_effect(timeout=0):
        calls[0] += 1
        if calls[0] == 1:
            return poll_tuid
        backend._shutdown_event.set()
        return None

    backend.batch.poll.side_effect = poll_side_effect
    backend._monitor_loop()
    return mock_loop


def test_monitor_loop_get_called_after_poll(backend_dragon):
    """_monitor_loop calls batch_task.get(block=True) after poll() returns its tuid.

    block=True (not block=False) is required: poll() and the DDict write are not atomic, so blocking
    here ensures the result is fully committed before get_stdout(block=False) is called afterward
    (see the matching comment in dragon.py's _monitor_loop).
    """
    uid = "task.poll-get"
    tuid = "dragon-tuid-poll-get"
    mock_task = MagicMock()
    mock_task.get.return_value = "done-result"
    mock_task.traceback = None
    mock_task.stdout_path = None
    mock_task.stderr_path = None
    mock_task.get_stdout.return_value = None
    mock_task.get_stderr.return_value = None

    mock_loop = _run_monitor_one_cycle(
        backend_dragon,
        poll_tuid=tuid,
        monitored_entry=(mock_task, uid),
        task_registry_entry={"uid": uid, "description": {"uid": uid}, "is_native_function": True},
    )

    mock_task.get.assert_called_once_with(block=True)
    mock_loop.call_soon_threadsafe.assert_called_once()
    _, completions = mock_loop.call_soon_threadsafe.call_args.args
    assert completions == [(uid, "done-result", None, False, "", "")]


def test_monitor_loop_skips_cancelled_tuid(backend_dragon):
    """When poll() returns a tuid recorded in _cancelled_tuids, _deliver_batch is not called.

    This is the user-cancel path: cancel_task() eagerly pops from _monitored_batches and
    records the tuid in _cancelled_tuids before the monitor loop sees it via poll(). The loop
    must discard it immediately (single sweep) rather than retrying it through the
    pending-tuid buffer, which is reserved for tuids that are merely not registered yet.
    """
    tuid = "dragon-cancelled-tuid"
    # Intentionally not inserting into _monitored_batches — simulates cancel_task already popped
    # it — but cancelled=True records it in _cancelled_tuids, simulating cancel_task's own add().
    mock_loop = _run_monitor_one_cycle(backend_dragon, poll_tuid=tuid, cancelled=True)

    mock_loop.call_soon_threadsafe.assert_not_called()
    assert tuid not in backend_dragon._cancelled_tuids
    assert tuid not in backend_dragon._pending_tuids


def test_monitor_loop_buffers_unregistered_tuid_until_it_lands(backend_dragon):
    """A tuid polled before its _monitored_batches registration lands is retried, not dropped.

    Simulates build_task racing the monitor thread: poll() returns the tuid on sweep 1 while
    _monitored_batches is still empty for it (build_task hasn't written its registration yet).
    The registration lands as a side effect of the next poll() call — before sweep 2's
    tuid-processing loop runs, exactly as build_task's synchronous write would land before the
    monitor thread's next pass. The completion must be delivered on sweep 2, not silently
    dropped as "cancelled".
    """
    uid = "task.late-registration"
    tuid = "dragon-tuid-late-registration"
    mock_task = MagicMock()
    mock_task.get.return_value = "late-result"
    mock_task.traceback = None
    mock_task.stdout_path = None
    mock_task.stderr_path = None
    mock_task.get_stdout.return_value = None
    mock_task.get_stderr.return_value = None

    backend_dragon._task_registry[uid] = {
        "uid": uid,
        "description": {"uid": uid},
        "is_native_function": True,
    }

    mock_loop = MagicMock()
    backend_dragon._loop = mock_loop

    calls = [0]

    def poll_side_effect(timeout=0):
        calls[0] += 1
        if calls[0] == 1:
            # Sweep 1 outer poll: tuid arrives, but not yet registered.
            return tuid
        if calls[0] == 2:
            # Sweep 1 inner drain: nothing else queued.
            return None
        if calls[0] == 3:
            # Sweep 2 outer poll: the registration lands right here, before sweep 2's
            # processing loop runs — then request shutdown so the loop exits once this
            # sweep delivers the now-resolved completion.
            backend_dragon._monitored_batches[tuid] = (mock_task, uid)
            backend_dragon._shutdown_event.set()
            return None
        return None

    backend_dragon.batch.poll.side_effect = poll_side_effect

    backend_dragon._monitor_loop()

    mock_loop.call_soon_threadsafe.assert_called_once()
    _, completions = mock_loop.call_soon_threadsafe.call_args.args
    assert completions == [(uid, "late-result", None, False, "", "")]
    assert tuid not in backend_dragon._pending_tuids


def test_monitor_loop_reads_paths_from_batch_task(backend_dragon):
    """Stdout/stderr paths are taken from batch_task.stdout_path/.stderr_path after poll().

    The monitor loop must NOT look up paths from _task_registry — they live on the
    batch_task object as set at submission time via Batch.options(stdout=..., stderr=...).
    """
    uid = "task.poll-paths"
    tuid = "dragon-tuid-paths"
    mock_task = MagicMock()
    mock_task.get.return_value = "result"
    mock_task.traceback = None
    mock_task.stdout_path = "/work/uid.stdout"
    mock_task.stderr_path = "/work/uid.stderr"

    mock_loop = _run_monitor_one_cycle(
        backend_dragon,
        poll_tuid=tuid,
        monitored_entry=(mock_task, uid),
        task_registry_entry={
            "uid": uid,
            "description": {"uid": uid, "capture_stdio": True},
            "is_native_function": False,
        },
    )

    mock_loop.call_soon_threadsafe.assert_called_once()
    _, completions = mock_loop.call_soon_threadsafe.call_args.args
    assert len(completions) == 1
    assert completions[0][4] == "/work/uid.stdout"
    assert completions[0][5] == "/work/uid.stderr"


def test_monitor_loop_nonzero_exit_delivers_failed(backend_dragon):
    """Process task: get() returning int 1 must produce raised=True + SystemExit(1)."""
    uid, tuid = "task.exit-1", "tuid-exit-1"
    mock_task = MagicMock()
    mock_task.get.return_value = 1
    mock_task.traceback = None
    mock_task.get_stdout.return_value = None
    mock_task.get_stderr.return_value = None

    mock_loop = _run_monitor_one_cycle(
        backend_dragon,
        poll_tuid=tuid,
        monitored_entry=(mock_task, uid),
        task_registry_entry={"uid": uid, "description": {"uid": uid}, "is_native_function": False},
    )

    _, completions = mock_loop.call_soon_threadsafe.call_args.args
    uid_c, result_c, _tb, raised_c, *_ = completions[0]
    assert uid_c == uid
    assert raised_c is True
    assert isinstance(result_c, SystemExit)
    assert result_c.code == 1


def test_monitor_loop_function_int_return_delivers_done(backend_dragon):
    """Function task returning int 42 must deliver DONE — exit-code check must not fire."""
    uid, tuid = "task.func-42", "tuid-func-42"
    mock_task = MagicMock()
    mock_task.get.return_value = 42
    mock_task.traceback = None
    mock_task.get_stdout.return_value = None
    mock_task.get_stderr.return_value = None

    mock_loop = _run_monitor_one_cycle(
        backend_dragon,
        poll_tuid=tuid,
        monitored_entry=(mock_task, uid),
        task_registry_entry={"uid": uid, "description": {"uid": uid}, "is_native_function": True},
    )

    _, completions = mock_loop.call_soon_threadsafe.call_args.args
    uid_c, result_c, _tb, raised_c, *_ = completions[0]
    assert uid_c == uid
    assert raised_c is False
    assert result_c == 42


@pytest.mark.asyncio
async def test_cancel_task_calls_dragon_cancel(backend_dragon):
    """cancel_task delegates to batch_task.cancel() and fires CANCELED callback on success.

    The callback must be called synchronously — no asyncio.sleep(0) needed to observe it. This
    verifies that TaskStateManager.update_task (when bound) runs inline on the event loop.

    After cancellation the task must be removed from both _task_registry and _monitored_batches so
    the monitor loop stops polling its DDict entry (blocking IPC for a cancelled task would stall
    the monitor thread and delay result delivery for subsequently submitted tasks).
    """
    uid = "task.unit-cancel"
    mock_batch_task = MagicMock()
    mock_batch_task.cancel.return_value = True
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {
        "uid": uid,
        "description": task_desc,
        "batch_task": mock_batch_task,
    }
    backend_dragon._monitored_batches[mock_batch_task.uid] = (mock_batch_task, uid)

    result = await backend_dragon.cancel_task(uid)

    # Callback and state must be visible immediately — no yield required
    assert result is True
    mock_batch_task.cancel.assert_called_once()
    backend_dragon._callback_func.assert_called_once_with(task_desc, "CANCELED")
    # Task must be eagerly removed so the monitor loop stops polling
    assert uid not in backend_dragon._task_registry
    assert mock_batch_task.uid not in backend_dragon._monitored_batches
    # Recorded so _monitor_loop discards this tuid immediately if it later arrives via poll().
    assert mock_batch_task.uid in backend_dragon._cancelled_tuids


@pytest.mark.asyncio
async def test_cancel_task_returns_false_when_task_completed_before_cancel_lands(backend_dragon):
    """cancel_task returns False when the task was already delivered (registry entry gone).

    Race: the event loop yields inside run_in_executor; _deliver_batch could pop the task
    from _task_registry before cancel_task resumes. The guard must return False rather than
    raising KeyError, and must NOT fire the CANCELED callback.

    The test simulates this by patching run_in_executor to pop the registry entry as a
    side effect — exactly what _deliver_batch does while the executor is in flight.
    """
    uid = "task.unit-cancel-race"
    mock_batch_task = MagicMock()
    mock_batch_task.cancel.return_value = True
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {
        "uid": uid,
        "description": task_desc,
        "batch_task": mock_batch_task,
    }

    # Simulate _deliver_batch removing the registry entry while run_in_executor is awaited.
    original_cancel = mock_batch_task.cancel

    def cancel_and_deliver():
        result = original_cancel()
        backend_dragon._task_registry.pop(uid, None)
        return result

    mock_batch_task.cancel = cancel_and_deliver

    result = await backend_dragon.cancel_task(uid)

    assert result is False
    backend_dragon._callback_func.assert_not_called()
    assert uid not in backend_dragon._task_registry


@pytest.mark.asyncio
async def test_cancel_task_dragon_returns_false_no_callback(backend_dragon):
    """cancel_task does not fire callback when Dragon reports the task cannot be cancelled."""
    uid = "task.unit-cancel-false"
    mock_batch_task = MagicMock()
    mock_batch_task.cancel.return_value = False
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {
        "uid": uid,
        "description": task_desc,
        "batch_task": mock_batch_task,
    }

    result = await backend_dragon.cancel_task(uid)

    assert result is False
    backend_dragon._callback_func.assert_not_called()
    assert uid not in backend_dragon._cancelled_tasks


@pytest.mark.asyncio
async def test_cancel_task_dragon_raises_falls_back_to_soft_cancel(backend_dragon):
    """cancel_task falls back to soft-cancel (callback fired, cancelled=True) if Dragon raises."""
    uid = "task.unit-cancel-exc"
    mock_batch_task = MagicMock()
    mock_batch_task.cancel.side_effect = RuntimeError("dragon scheduler unreachable")
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {
        "uid": uid,
        "description": task_desc,
        "batch_task": mock_batch_task,
    }
    backend_dragon._monitored_batches[mock_batch_task.uid] = (mock_batch_task, uid)

    result = await backend_dragon.cancel_task(uid)

    assert result is True
    backend_dragon._callback_func.assert_called_once_with(task_desc, "CANCELED")
    assert uid not in backend_dragon._task_registry
    assert mock_batch_task.uid not in backend_dragon._monitored_batches


@pytest.mark.asyncio
async def test_shutdown_calls_join_and_destroy_not_close(backend_dragon):
    """Shutdown() calls batch.join() then batch.destroy(); batch.close() must NOT be called."""
    await backend_dragon.shutdown()

    backend_dragon.batch.join.assert_called_once()
    backend_dragon.batch.destroy.assert_called_once()
    backend_dragon.batch.close.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_removes_task_logs_directory(backend_dragon, tmp_path):
    """Shutdown() removes the directory returned by batch.log_dir()."""
    log_dir = tmp_path / "runinfo" / "run-1" / "client-0" / "task_logs"
    log_dir.mkdir(parents=True)
    (log_dir / "manifest.jsonl").write_text("{}\n")
    backend_dragon.batch.log_dir.return_value = log_dir

    await backend_dragon.shutdown()

    assert not log_dir.exists()


@pytest.mark.asyncio
async def test_shutdown_skips_cleanup_when_task_logs_disabled(backend_dragon):
    """Shutdown() tolerates batch.log_dir() raising RuntimeError (task_logs disabled)."""
    backend_dragon.batch.log_dir.side_effect = RuntimeError("task logging is disabled")

    await backend_dragon.shutdown()

    backend_dragon.batch.join.assert_called_once()
    backend_dragon.batch.destroy.assert_called_once()


# ============================================================================
# V3 capture_stdio tests — no Dragon cluster required
# ============================================================================


@pytest.mark.asyncio
async def test_capture_stdio_true_passes_paths_to_batch(backend_dragon, tmp_path):
    """capture_stdio=True passes stdout/stderr file paths to Batch.options() — no shell script.

    Dragon 0.14.1-rc requires metadata (name, timeout, stdio paths) in Batch.options(), not as
    direct kwargs to process()/function().  The old patch on batch.process is dead code. Assert via
    batch.options call_args instead.
    """
    backend_dragon._work_dir = str(tmp_path)

    task = {
        "uid": "task.capture-test-001",
        "executable": "/bin/echo",
        "arguments": ["hello"],
        "function": None,
        "args": [],
        "kwargs": {},
        "name": "capture-test",
        "task_backend_specific_kwargs": {"process_template": {}},
        "capture_stdio": True,
    }

    await backend_dragon.build_task(task)

    options_kwargs = backend_dragon.batch.options.call_args.kwargs
    assert options_kwargs["stdout"].endswith(".stdout")
    assert options_kwargs["stderr"].endswith(".stderr")
    # No shell script written — Dragon handles redirection natively
    assert not list(tmp_path.glob("*.sh"))
    # Registry no longer stores paths — they live on the batch_task object
    reg = backend_dragon._task_registry[task["uid"]]
    assert "script_path" not in reg


@pytest.mark.asyncio
async def test_capture_stdio_false_passes_none_to_batch(backend_dragon, tmp_path):
    """capture_stdio=False (default) passes stdout=None, stderr=None to Batch.options()."""
    backend_dragon._work_dir = str(tmp_path)

    task = {
        "uid": "task.capture-test-002",
        "executable": "/bin/echo",
        "arguments": ["hello"],
        "function": None,
        "args": [],
        "kwargs": {},
        "name": "no-capture-test",
        "task_backend_specific_kwargs": {"process_template": {}},
        "capture_stdio": False,
    }

    await backend_dragon.build_task(task)

    options_kwargs = backend_dragon.batch.options.call_args.kwargs
    assert options_kwargs["stdout"] is None
    assert options_kwargs["stderr"] is None
    assert not list(tmp_path.glob("*.sh"))


@pytest.mark.asyncio
async def test_capture_stdio_function_task_passes_paths_to_batch_function(backend_dragon, tmp_path):
    """capture_stdio=True on a function task passes stdout/stderr paths to Batch.options()."""
    backend_dragon._work_dir = str(tmp_path)

    task = {
        "uid": "task.capture-test-003",
        "executable": None,
        "function": lambda: None,
        "arguments": [],
        "args": [],
        "kwargs": {},
        "name": "func-capture-test",
        "task_backend_specific_kwargs": {},
        "capture_stdio": True,
    }

    await backend_dragon.build_task(task)

    options_kwargs = backend_dragon.batch.options.call_args.kwargs
    assert options_kwargs["stdout"].endswith(".stdout")
    assert options_kwargs["stderr"].endswith(".stderr")
    assert not list(tmp_path.glob("*.sh"))


# ============================================================================
# result_contract tests — stdout/stderr must be str after any DONE/FAILED
# ============================================================================


@pytest.mark.result_contract
def test_deliver_batch_empty_dragon_stdout_written_as_empty_string(backend_dragon):
    """Regression: stdout='' from Dragon must land as '' not None or absent key."""
    uid = "task.contract-empty-stdout"
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {"uid": uid, "description": task_desc}

    backend_dragon._deliver_batch([(uid, 0, None, False, "", "")])

    assert isinstance(task_desc["stdout"], str)
    assert isinstance(task_desc["stderr"], str)
    backend_dragon._callback_func.assert_called_once_with(task_desc, "DONE")


@pytest.mark.result_contract
def test_deliver_batch_path_from_tuple_stored_on_task(backend_dragon):
    """Stdout/stderr paths in the completion tuple are stored directly on task_desc.

    The monitor loop passes batch_task.stdout_path as the 5th element; _deliver_batch stores it
    verbatim — no registry lookup needed.
    """
    uid = "task.contract-redirect"
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {"uid": uid, "description": task_desc}

    backend_dragon._deliver_batch([(uid, 0, None, False, "/work/uid.stdout", "/work/uid.stderr")])

    assert task_desc["stdout"] == "/work/uid.stdout"
    assert task_desc["stderr"] == "/work/uid.stderr"


@pytest.mark.result_contract
def test_deliver_batch_failed_stdout_is_string(backend_dragon):
    """Stdout must be a str in the FAILED path."""
    uid = "task.contract-failed-stdout"
    task_desc = {"uid": uid}
    backend_dragon._task_registry[uid] = {"uid": uid, "description": task_desc}

    backend_dragon._deliver_batch([(uid, RuntimeError("boom"), None, True, "", "")])

    assert isinstance(task_desc["stdout"], str)
