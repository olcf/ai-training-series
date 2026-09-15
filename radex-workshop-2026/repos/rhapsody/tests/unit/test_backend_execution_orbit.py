"""Unit tests for OrbitExecutionBackend (refactored: delegates to RhapsodyClient)."""

import asyncio
import sys
import threading
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import rhapsody.backends.execution.orbit as orbit_mod
from rhapsody.backends.execution.orbit import OrbitExecutionBackend


@pytest.fixture(autouse=True)
def _stub_endpoint_runtime():
    """Allow the suite to run without ``radical.orbit`` installed (e.g. in CI).

    ``OrbitExecutionBackend.__init__`` raises ``ImportError`` when the optional
    ``radical.orbit`` package is missing (module-level ``EndpointRuntime`` is
    ``None``).  Every test mocks the runtime/rhapsody chain anyway, so when the
    real package is absent stub the symbol to a non-``None`` mock so the
    backend can be constructed.  When it is installed this is a no-op.
    """
    if orbit_mod.EndpointRuntime is not None:
        yield
        return
    with patch.object(orbit_mod, "EndpointRuntime", MagicMock()):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_rhapsody_client(sid="session.abc123"):
    """Return a mock RhapsodyClient (PluginClient)."""
    rh = MagicMock()
    rh.sid = sid
    rh.submit_tasks = MagicMock(return_value=[{"uid": "t.001", "state": "SUBMITTED"}])
    rh.cancel_task = MagicMock(return_value={"uid": "t.001", "status": "canceled"})
    rh.cancel_all_tasks = MagicMock(return_value={"canceled": 5})
    rh.close = MagicMock()
    rh.register_notification_callback = MagicMock()
    return rh


def _mock_runtime(rh=None, topology=None):
    """Return a mock EndpointRuntime whose ``get_plugin`` produces *rh*."""
    if rh is None:
        rh = _mock_rhapsody_client()
    rt = MagicMock()
    # NOTE: ``MagicMock(name=...)`` sets the mock's repr, not the attribute —
    # ``name`` must be assigned after construction.
    rt.name = "rhapsody.test0000"
    rt.broker_url = "http://localhost:8000"
    rt.start = MagicMock(return_value=rt)
    rt.wait_registered = MagicMock(return_value=True)
    rt.topology = MagicMock(return_value=topology or {})
    rt.get_plugin = MagicMock(return_value=rh)
    rt.stop = MagicMock()
    return rt, rh


def _make_backend(**kwargs):
    """Create an OrbitExecutionBackend (not yet initialised)."""
    defaults = {
        "broker_url": "http://localhost:8000",
        "endpoint_name": "test_endpoint",
    }
    defaults.update(kwargs)
    return OrbitExecutionBackend(**defaults)


async def _init_backend(**kwargs):
    """Create and initialise a backend with a mocked EndpointRuntime chain."""
    topology = kwargs.pop("topology", None)
    backend = _make_backend(**kwargs)
    rt, rh = _mock_runtime(topology=topology)
    with patch("rhapsody.backends.execution.orbit.EndpointRuntime", return_value=rt):
        await backend._async_init()
    # Expose mocks for assertions
    backend._mock_rt = rt
    backend._mock_rh = rh
    return backend


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_endpoint_backend_construction():
    backend = _make_backend()
    assert backend._broker_url == "http://localhost:8000"
    assert backend._endpoint_name == "test_endpoint"
    assert backend._plugin_name == "rhapsody"
    assert backend._remote_backends == ["dragon_v3"]
    assert backend._initialized is False


def test_endpoint_backend_custom_params():
    backend = _make_backend(
        plugin_name="my_rhapsody",
        backends=["dragon_v3"],
        name="my_endpoint",
    )
    assert backend._plugin_name == "my_rhapsody"
    assert backend._remote_backends == ["dragon_v3"]
    assert backend.name == "my_endpoint"


# ---------------------------------------------------------------------------
# Async init
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_init_creates_client_chain():
    backend = await _init_backend()

    assert backend._initialized is True
    assert backend._runtime is not None
    assert backend._rh is not None

    # runtime started and registration verified
    backend._mock_rt.start.assert_called_once_with(wait=True, timeout=30.0)
    # get_plugin called with endpoint + plugin name + session kwargs
    backend._mock_rt.get_plugin.assert_called_once_with(
        "test_endpoint", "rhapsody", backends=["dragon_v3"], init_timeout=120.0
    )

    # Notification callbacks registered
    calls = backend._mock_rh.register_notification_callback.call_args_list
    topics = [c[1]["topic"] for c in calls]
    assert "task_status" in topics
    assert "task_status_batch" in topics


@pytest.mark.asyncio
async def test_async_init_idempotent():
    backend = await _init_backend()
    rh = backend._mock_rh

    await backend._async_init()
    # register_notification_callback should NOT be called again
    assert rh.register_notification_callback.call_count == 2  # initial only


# ---------------------------------------------------------------------------
# submit_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_tasks_delegates_to_rhapsody_client():
    backend = await _init_backend(batch_window=0)

    tasks = [{"uid": "t.001", "executable": "/bin/echo", "arguments": ["hi"]}]
    await backend.submit_tasks(tasks)

    backend._mock_rh.submit_tasks.assert_called_once()
    submitted = backend._mock_rh.submit_tasks.call_args[0][0]
    assert len(submitted) == 1
    assert submitted[0]["uid"] == "t.001"

    # Task tracked locally
    assert "t.001" in backend._tasks


@pytest.mark.asyncio
async def test_submit_tasks_sets_running_state():
    backend = await _init_backend()
    assert await backend.state() == "INITIALIZED"

    await backend.submit_tasks([{"uid": "t.1", "executable": "/bin/true"}])
    assert await backend.state() == "RUNNING"


# ---------------------------------------------------------------------------
# cancel_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_task():
    backend = await _init_backend()
    backend._tasks["t.001"] = {"uid": "t.001", "state": "RUNNING"}
    cb = MagicMock()
    backend.register_callback(cb)

    result = await backend.cancel_task("t.001")
    assert result is True
    # terminal task is delivered via callback, then pruned from the registry
    task, state = cb.call_args[0]
    assert task["uid"] == "t.001"
    assert state == "CANCELED"
    assert task["state"] == "CANCELED"
    assert "t.001" not in backend._tasks
    backend._mock_rh.cancel_task.assert_called_once_with("t.001")


@pytest.mark.asyncio
async def test_cancel_unknown_task():
    backend = await _init_backend()
    result = await backend.cancel_task("no_such_task")
    assert result is False


# ---------------------------------------------------------------------------
# cancel_all_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_all_tasks():
    backend = await _init_backend()
    count = await backend.cancel_all_tasks()
    assert count == 5
    backend._mock_rh.cancel_all_tasks.assert_called_once()


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_closes_clients():
    backend = await _init_backend()
    await backend.shutdown()

    backend._mock_rh.close.assert_called_once()
    backend._mock_rt.stop.assert_called_once()
    assert backend._rh is None
    assert backend._runtime is None
    assert await backend.state() == "SHUTDOWN"


# ---------------------------------------------------------------------------
# Notification handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_task_notification_single():
    # _on_task_notification marshals to the event loop; drive one tick to let
    # the scheduled handler run before asserting.
    backend = await _init_backend()
    backend._tasks["t.001"] = {"uid": "t.001", "state": "SUBMITTED"}
    cb = MagicMock()
    backend.register_callback(cb)

    backend._on_task_notification(
        endpoint="hpc1",
        plugin="rhapsody",
        topic="task_status",
        data={"uid": "t.001", "state": "DONE", "stdout": "hello\n", "exit_code": 0},
    )
    await asyncio.sleep(0)

    task, state = cb.call_args[0]
    assert state == "DONE"
    assert task["state"] == "DONE"
    assert task["stdout"] == "hello\n"
    assert "t.001" not in backend._tasks  # terminal -> pruned


@pytest.mark.asyncio
async def test_on_task_notification_batch():
    backend = await _init_backend()
    backend._tasks["t.001"] = {"uid": "t.001", "state": "SUBMITTED"}
    backend._tasks["t.002"] = {"uid": "t.002", "state": "SUBMITTED"}
    cb = MagicMock()
    backend.register_callback(cb)

    backend._on_task_notification(
        endpoint="hpc1",
        plugin="rhapsody",
        topic="task_status_batch",
        data={
            "tasks": [
                {"uid": "t.001", "state": "DONE"},
                {"uid": "t.002", "state": "FAILED", "error": "boom"},
            ]
        },
    )
    await asyncio.sleep(0)

    fired = {c.args[0]["uid"]: c.args for c in cb.call_args_list}
    assert fired["t.001"][1] == "DONE"
    assert fired["t.002"][1] == "FAILED"
    assert fired["t.002"][0]["error"] == "boom"
    # both terminal -> pruned
    assert "t.001" not in backend._tasks
    assert "t.002" not in backend._tasks


@pytest.mark.asyncio
async def test_on_task_notification_ignores_unknown_task():
    backend = await _init_backend()
    # No tasks registered — should not crash
    backend._on_task_notification(
        endpoint="hpc1",
        plugin="rhapsody",
        topic="task_status",
        data={"uid": "unknown", "state": "DONE"},
    )
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_on_task_notification_coerces_string_exception():
    """A string exception serialized over the wire must become a real BaseException so session.py
    raises it instead of silently treating the failed task as successful."""
    backend = await _init_backend()
    backend._tasks["t.001"] = {"uid": "t.001", "state": "SUBMITTED"}
    cb = MagicMock()
    backend.register_callback(cb)

    backend._on_task_notification(
        endpoint="hpc1",
        plugin="rhapsody",
        topic="task_status",
        data={"uid": "t.001", "state": "FAILED", "exception": "remote boom"},
    )
    await asyncio.sleep(0)

    task, _ = cb.call_args[0]
    exc = task["exception"]
    assert isinstance(exc, BaseException)
    assert "remote boom" in str(exc)


# ---------------------------------------------------------------------------
# state / context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state():
    backend = await _init_backend()
    assert await backend.state() == "INITIALIZED"


@pytest.mark.asyncio
async def test_context_manager():
    backend = _make_backend()
    rt, rh = _mock_runtime()
    with patch("rhapsody.backends.execution.orbit.EndpointRuntime", return_value=rt):
        async with backend as b:
            assert b._initialized is True
        assert await b.state() == "SHUTDOWN"


# ---------------------------------------------------------------------------
# Endpoint auto-selection from the topology + registration failure
# ---------------------------------------------------------------------------


def _topo_entry(role, liveness="present", plugins=()):
    return {
        "role": role,
        "liveness": liveness,
        "plugins": {p: {"namespace": f"/{p}", "enabled": True} for p in plugins},
    }


@pytest.mark.asyncio
async def test_auto_select_endpoint_from_topology():
    """With no endpoint_name, the first present endpoint advertising a rhapsody plugin is picked."""
    topology = {
        "ep0": _topo_entry("endpoint", plugins=["sysinfo"]),  # no rhapsody
        "ep1": _topo_entry("endpoint", plugins=["rhapsody", "sysinfo"]),
        "ep2": _topo_entry("endpoint", plugins=["rhapsody"]),
    }
    backend = await _init_backend(endpoint_name=None, topology=topology)

    assert backend._endpoint_name == "ep1"
    backend._mock_rt.get_plugin.assert_called_once_with(
        "ep1", "rhapsody", backends=["dragon_v3"], init_timeout=120.0
    )


@pytest.mark.asyncio
async def test_auto_select_skips_broker_consumer_self_and_lost():
    """The broker (even if it hosts a rhapsody plugin), pure consumers, this runtime's own topology
    entry, and non-present endpoints must all be skipped."""
    topology = {
        "broker": _topo_entry("broker", plugins=["rhapsody"]),
        "rhapsody.test0000": _topo_entry("consumer", plugins=["rhapsody"]),  # self
        "watcher": _topo_entry("consumer", plugins=["rhapsody"]),
        "ep.lost": _topo_entry("endpoint", liveness="lost", plugins=["rhapsody"]),
        "ep.good": _topo_entry("endpoint", plugins=["rhapsody"]),
    }
    backend = await _init_backend(endpoint_name=None, topology=topology)

    assert backend._endpoint_name == "ep.good"


@pytest.mark.asyncio
async def test_auto_select_no_candidate_raises():
    """An empty/ineligible topology raises RuntimeError and stops the runtime."""
    topology = {
        "broker": _topo_entry("broker", plugins=["rhapsody"]),
        "ep0": _topo_entry("endpoint", plugins=["sysinfo"]),
    }
    backend = _make_backend(endpoint_name=None)
    rt, _ = _mock_runtime(topology=topology)
    with patch("rhapsody.backends.execution.orbit.EndpointRuntime", return_value=rt):
        with pytest.raises(RuntimeError, match="no endpoint advertises"):
            await backend._async_init()

    rt.stop.assert_called_once()  # failed init must not leak the runtime
    assert backend._initialized is False


@pytest.mark.asyncio
async def test_registration_timeout_raises():
    """A silent start() timeout must be detected as a missed registration, failing with a message
    naming the broker — not a misleading 'no endpoint found'."""
    backend = _make_backend()
    rt, _ = _mock_runtime()
    rt.wait_registered = MagicMock(return_value=False)
    with patch("rhapsody.backends.execution.orbit.EndpointRuntime", return_value=rt):
        with pytest.raises(RuntimeError, match="failed to register with ORBIT broker"):
            await backend._async_init()

    rt.stop.assert_called_once()
    assert backend._initialized is False


# ---------------------------------------------------------------------------
# Python-version compat for cloudpickled function tasks
#
# The check fires only for tasks that carry a cloudpickled function or
# pickled fields.  Executable / import-path function tasks bypass it.
# It queries sysinfo.host_role() once and caches the (major, minor)
# tuple on the backend instance.
# ---------------------------------------------------------------------------


def _runtime_with_sysinfo(rh, sysinfo_python_version):
    """Runtime mock whose ``get_plugin(eid, 'sysinfo')`` returns a sysinfo plugin reporting the
    given python_version, and whose ``get_plugin(eid, 'rhapsody', ...)`` returns *rh*."""
    sysinfo = MagicMock()
    sysinfo.host_role = MagicMock(
        return_value={
            "role": "compute",
            "scheduler": "slurm",
            "psij_executor": "slurm",
            "job_id": "12345",
            "python_version": sysinfo_python_version,
        }
    )
    sysinfo.close = MagicMock()

    rt, _ = _mock_runtime(rh=rh)

    def _get_plugin(eid, pname, **kwargs):
        return sysinfo if pname == "sysinfo" else rh

    rt.get_plugin = MagicMock(side_effect=_get_plugin)
    return rt, sysinfo


async def _init_backend_with_sysinfo(sysinfo_python_version, **kwargs):
    """Like _init_backend but with a sysinfo plugin returning the given python_version on
    host_role()."""
    kwargs.setdefault("batch_window", 0)  # immediate flush, no timer
    backend = _make_backend(**kwargs)
    rh = _mock_rhapsody_client()
    rt, si = _runtime_with_sysinfo(rh, sysinfo_python_version)
    with patch("rhapsody.backends.execution.orbit.EndpointRuntime", return_value=rt):
        await backend._async_init()
    backend._mock_rt = rt
    backend._mock_rh = rh
    backend._mock_sysinfo = si
    return backend


@pytest.mark.asyncio
async def test_python_compat_skipped_for_executable_tasks():
    """No sysinfo lookup, no exception when batch contains only executable / import-path tasks."""
    client_mm = f"{sys.version_info.major}.{sys.version_info.minor}.0"
    backend = await _init_backend_with_sysinfo(client_mm)

    await backend.submit_tasks(
        [
            {"uid": "t.1", "executable": "/bin/true"},
            {"uid": "t.2", "function": "mod:func"},  # import-path, not pickled
        ]
    )

    backend._mock_rh.submit_tasks.assert_called_once()
    backend._mock_sysinfo.host_role.assert_not_called()


@pytest.mark.asyncio
async def test_python_compat_passes_for_matching_versions():
    """Cloudpickled task + matching endpoint Python -> submission proceeds."""
    client_mm = f"{sys.version_info.major}.{sys.version_info.minor}.0"
    backend = await _init_backend_with_sysinfo(client_mm)

    await backend.submit_tasks(
        [
            {"uid": "t.1", "function": "cloudpickle::ABCDEF"},
        ]
    )

    backend._mock_rh.submit_tasks.assert_called_once()
    backend._mock_sysinfo.host_role.assert_called_once()


@pytest.mark.asyncio
async def test_python_compat_raises_on_mismatch():
    """Cloudpickled task + endpoint on a different Python minor -> RuntimeError;
    rhapsody.submit_tasks is NOT called."""
    # Pick a minor version guaranteed to differ from the test runner's.
    other_minor = sys.version_info.minor + 1
    endpoint_pyver = f"{sys.version_info.major}.{other_minor}.0"
    backend = await _init_backend_with_sysinfo(endpoint_pyver)

    with pytest.raises(RuntimeError, match="cloudpickle is not portable"):
        await backend.submit_tasks(
            [
                {"uid": "t.1", "function": "cloudpickle::ABCDEF"},
            ]
        )

    backend._mock_rh.submit_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_python_compat_caches_first_lookup():
    """Subsequent submits of cloudpickled tasks hit the sysinfo plugin only once (cached on the
    backend instance)."""
    client_mm = f"{sys.version_info.major}.{sys.version_info.minor}.0"
    backend = await _init_backend_with_sysinfo(client_mm)

    for _ in range(3):
        await backend.submit_tasks(
            [
                {"uid": f"t.{_}", "function": "cloudpickle::ABCDEF"},
            ]
        )

    assert backend._mock_rh.submit_tasks.call_count == 3
    backend._mock_sysinfo.host_role.assert_called_once()


@pytest.mark.asyncio
async def test_python_compat_pickled_fields_marker_triggers_check():
    """A task with ``_pickled_fields`` (no cloudpickle:: prefix) still
    triggers the check, because the deserialization path is the same."""
    other_minor = sys.version_info.minor + 1
    endpoint_pyver = f"{sys.version_info.major}.{other_minor}.0"
    backend = await _init_backend_with_sysinfo(endpoint_pyver)

    with pytest.raises(RuntimeError, match="cloudpickle is not portable"):
        await backend.submit_tasks(
            [
                {"uid": "t.1", "function": "mod:func", "_pickled_fields": ["args"]},
            ]
        )
    backend._mock_rh.submit_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_python_compat_fails_fast_under_default_batching():
    """The compat check must propagate to the submit_tasks() caller even when batching is enabled
    (the common case) — not vanish into a deferred flush.

    The offending task must not be registered locally either.
    """
    other_minor = sys.version_info.minor + 1
    endpoint_pyver = f"{sys.version_info.major}.{other_minor}.0"
    backend = await _init_backend_with_sysinfo(endpoint_pyver, batch_window=0.25)

    with pytest.raises(RuntimeError, match="cloudpickle is not portable"):
        await backend.submit_tasks(
            [
                {"uid": "t.1", "function": "cloudpickle::ABCDEF"},
            ]
        )

    backend._mock_rh.submit_tasks.assert_not_called()
    assert "t.1" not in backend._tasks


@pytest.mark.asyncio
async def test_sysinfo_client_closed_after_compat_lookup():
    """The throwaway sysinfo client is closed after the lookup so its auto-registered ephemeral
    session does not linger on the endpoint."""
    client_mm = f"{sys.version_info.major}.{sys.version_info.minor}.0"
    backend = await _init_backend_with_sysinfo(client_mm)

    await backend.submit_tasks([{"uid": "t.1", "function": "cloudpickle::ABCDEF"}])

    backend._mock_sysinfo.host_role.assert_called_once()
    backend._mock_sysinfo.close.assert_called_once()


@pytest.mark.asyncio
async def test_python_compat_retries_after_lookup_failure():
    """A transient sysinfo lookup failure must not permanently disable the check — the next
    submission retries the lookup."""
    client_mm = f"{sys.version_info.major}.{sys.version_info.minor}.0"
    backend = await _init_backend_with_sysinfo(client_mm)

    # First lookup blows up (e.g. plugin not ready); second succeeds.
    backend._mock_sysinfo.host_role.side_effect = [
        RuntimeError("endpoint not ready"),
        {"role": "compute", "python_version": client_mm},
    ]

    # First submit: lookup fails -> check skipped, submission still proceeds,
    # and the check stays armed for a retry.
    await backend.submit_tasks([{"uid": "t.1", "function": "cloudpickle::X"}])
    assert backend._endpoint_python_lookup_done is False

    # Second submit: lookup retried and settles.
    await backend.submit_tasks([{"uid": "t.2", "function": "cloudpickle::X"}])
    assert backend._endpoint_python_lookup_done is True
    assert backend._mock_sysinfo.host_role.call_count == 2
    assert backend._mock_rh.submit_tasks.call_count == 2


# ---------------------------------------------------------------------------
# cancel_all_tasks: local state + callbacks, and terminal-callback dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_all_updates_local_state_and_callbacks():
    """cancel_all_tasks marks every non-terminal local task CANCELED and fires its callback, leaving
    already-terminal tasks untouched."""
    backend = await _init_backend()
    backend._tasks = {
        "t.1": {"uid": "t.1", "state": "RUNNING"},
        "t.2": {"uid": "t.2", "state": "DONE"},  # terminal -> skip
        "t.3": {"uid": "t.3", "state": "SUBMITTED"},
    }
    cb = MagicMock()
    backend.register_callback(cb)

    await backend.cancel_all_tasks()

    # cancelled tasks fire CANCELED then are pruned; the already-terminal one
    # is left untouched.
    assert "t.1" not in backend._tasks
    assert "t.3" not in backend._tasks
    assert backend._tasks["t.2"]["state"] == "DONE"

    fired = {call.args[0]["uid"] for call in cb.call_args_list}
    assert fired == {"t.1", "t.3"}
    assert all(call.args[1] == "CANCELED" for call in cb.call_args_list)


@pytest.mark.asyncio
async def test_terminal_callback_fires_once_across_cancel_and_sse():
    """A local cancel and the matching CANCELED SSE notification must not double-fire the terminal
    callback."""
    backend = await _init_backend()
    backend._tasks["t.1"] = {"uid": "t.1", "state": "RUNNING"}
    cb = MagicMock()
    backend.register_callback(cb)

    await backend.cancel_task("t.1")  # local path fires once, prunes t.1
    # the matching CANCELED SSE notification arrives afterwards and is ignored
    # because t.1 is no longer tracked
    backend._on_task_notification(
        endpoint="hpc1",
        plugin="rhapsody",
        topic="task_status",
        data={"uid": "t.1", "state": "CANCELED"},
    )
    await asyncio.sleep(0)

    assert cb.call_count == 1
    assert "t.1" not in backend._tasks


# ---------------------------------------------------------------------------
# Batch flushing: lock is released during the network send, sends stay
# serialized/ordered, and buffered tasks survive shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_send_releases_lock_during_send():
    """A slow batch send must not hold the batch lock — other submitters can keep buffering while
    the network flush is in flight.

    With the send performed under the batch lock this would dead-time out.
    """
    backend = await _init_backend(batch_window=10, batch_limit=2)

    send_started = threading.Event()
    release_send = threading.Event()

    def slow_submit(batch):
        send_started.set()
        release_send.wait(timeout=5)
        return [{"uid": batch[0]["uid"], "state": "SUBMITTED"}]

    backend._mock_rh.submit_tasks = MagicMock(side_effect=slow_submit)

    # Fill the buffer (limit=2) → inline flush that blocks inside the send.
    first = asyncio.create_task(
        backend.submit_tasks(
            [
                {"uid": "t.1", "executable": "/bin/true"},
                {"uid": "t.2", "executable": "/bin/true"},
            ]
        )
    )

    # Wait until the send is actually in flight in the worker thread.
    assert await asyncio.to_thread(send_started.wait, 5)

    # While the send is blocked, another submit must still complete (buffer)
    # rather than hang on the batch lock.
    await asyncio.wait_for(
        backend.submit_tasks([{"uid": "t.3", "executable": "/bin/true"}]),
        timeout=1.0,
    )
    assert "t.3" in backend._tasks

    release_send.set()
    await asyncio.wait_for(first, timeout=5)
    await backend.shutdown()


@pytest.mark.asyncio
async def test_batch_sends_are_serialized():
    """Concurrent flushes must not call the client concurrently — the send lock serializes them, so
    at most one send runs at a time."""
    backend = await _init_backend(batch_window=10, batch_limit=1)

    counter_lock = threading.Lock()
    state = {"cur": 0, "max": 0}
    order = []

    def rec(batch):
        with counter_lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.02)
        with counter_lock:
            state["cur"] -= 1
        order.append(batch[0]["uid"])
        return [{"uid": batch[0]["uid"], "state": "SUBMITTED"}]

    backend._mock_rh.submit_tasks = MagicMock(side_effect=rec)

    # limit=1 → each submit flushes immediately; fire several concurrently.
    await asyncio.gather(
        *[backend.submit_tasks([{"uid": f"t.{i}", "executable": "/bin/true"}]) for i in range(5)]
    )

    assert len(order) == 5
    assert state["max"] == 1


@pytest.mark.asyncio
async def test_shutdown_flushes_buffered_tasks():
    """Tasks still buffered when shutdown is called are flushed, not dropped."""
    backend = await _init_backend(batch_window=10, batch_limit=100)

    await backend.submit_tasks([{"uid": "t.1", "executable": "/bin/true"}])
    backend._mock_rh.submit_tasks.assert_not_called()  # buffered, not sent yet

    await backend.shutdown()

    backend._mock_rh.submit_tasks.assert_called_once()
    sent = backend._mock_rh.submit_tasks.call_args[0][0]
    assert [t["uid"] for t in sent] == ["t.1"]


@pytest.mark.asyncio
async def test_sse_notification_applied_on_loop_thread():
    """A notification delivered from a non-loop (SSE) thread must be applied on the event-loop
    thread — never on the caller's thread — so it can't race loop-side writers of self._tasks."""
    backend = await _init_backend()
    backend._tasks["t.1"] = {"uid": "t.1", "state": "SUBMITTED"}

    applied = {}
    orig_apply = backend._apply_task_update

    def spy(body):
        applied["thread"] = threading.get_ident()
        return orig_apply(body)

    backend._apply_task_update = spy

    sse = {}

    def deliver():
        sse["thread"] = threading.get_ident()
        backend._on_task_notification(
            endpoint="e",
            plugin="rhapsody",
            topic="task_status",
            data={"uid": "t.1", "state": "RUNNING"},  # non-terminal: stays tracked
        )

    t = threading.Thread(target=deliver)
    t.start()
    t.join()

    # Scheduled on the loop, but not run yet (we haven't yielded).
    assert "thread" not in applied
    assert backend._tasks["t.1"]["state"] == "SUBMITTED"

    await asyncio.sleep(0)

    loop_thread = threading.get_ident()
    assert applied["thread"] == loop_thread  # applied on the loop thread
    assert applied["thread"] != sse["thread"]  # not the SSE (caller) thread
    assert backend._tasks["t.1"]["state"] == "RUNNING"


# ---------------------------------------------------------------------------
# Memory bounding + cancel of still-buffered tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_tasks_are_pruned():
    """Reaching a terminal state drops the task from _tasks and _terminal_fired so the backend does
    not retain every task it ever ran."""
    backend = await _init_backend(batch_window=0)
    await backend.submit_tasks([{"uid": "t.1", "executable": "/bin/true"}])
    assert "t.1" in backend._tasks

    backend._on_task_notification(
        endpoint="e",
        plugin="rhapsody",
        topic="task_status",
        data={"uid": "t.1", "state": "DONE"},
    )
    await asyncio.sleep(0)

    assert "t.1" not in backend._tasks
    assert "t.1" not in backend._terminal_fired


@pytest.mark.asyncio
async def test_cancel_buffered_task_skips_remote_and_is_not_flushed():
    """Cancelling a task still in the batch buffer removes it from the buffer, skips the remote
    cancel (endpoint never received it), and it is not sent on the next flush."""
    backend = await _init_backend(batch_window=10, batch_limit=100)  # stays buffered

    await backend.submit_tasks([{"uid": "t.1", "executable": "/bin/true"}])
    assert any(t.get("uid") == "t.1" for t in backend._batch_buffer)

    ok = await backend.cancel_task("t.1")
    assert ok is True
    backend._mock_rh.cancel_task.assert_not_called()  # not submitted yet
    assert not any(t.get("uid") == "t.1" for t in backend._batch_buffer)  # dropped

    # A subsequent shutdown flush must not send the cancelled task.
    await backend.shutdown()
    backend._mock_rh.submit_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_submitted_task_calls_remote():
    """A task already submitted (not buffered) still goes to the remote cancel."""
    backend = await _init_backend(batch_window=0)  # immediate submit, no buffering
    await backend.submit_tasks([{"uid": "t.1", "executable": "/bin/true"}])
    assert not backend._batch_buffer

    ok = await backend.cancel_task("t.1")
    assert ok is True
    backend._mock_rh.cancel_task.assert_called_once_with("t.1")


@pytest.mark.asyncio
async def test_cancel_all_drops_buffered_tasks():
    """cancel_all_tasks clears the local buffer so buffered tasks are not flushed after
    cancellation."""
    backend = await _init_backend(batch_window=10, batch_limit=100)
    await backend.submit_tasks([{"uid": "t.1", "executable": "/bin/true"}])
    assert backend._batch_buffer

    await backend.cancel_all_tasks()
    assert backend._batch_buffer == []

    await backend.shutdown()
    backend._mock_rh.submit_tasks.assert_not_called()
