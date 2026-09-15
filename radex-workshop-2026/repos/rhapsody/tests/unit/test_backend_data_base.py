"""Unit tests for the DataBackend/Endpoint lifecycle state machine.

No real infrastructure is needed -- these exercise rhapsody.data.base's state machine via an in-file
fake DataBackend subclass.
"""

import asyncio

import pytest

from rhapsody.backends.data.base import DataBackend
from rhapsody.backends.data.base import DataBackendNotReadyError
from rhapsody.backends.data.base import DataBackendStartupError
from rhapsody.backends.data.base import DataBackendState
from rhapsody.backends.data.base import DataBackendTerminatedError
from rhapsody.backends.data.base import Endpoint


class _FakeEndpoint(Endpoint):
    def __init__(self, tag: str = "fake"):
        self.tag = tag

    def serialize(self) -> str:
        return self.tag


class _FakeDataBackend(DataBackend):
    def __init__(self, fail_start: bool = False, fail_ready: bool = False):
        super().__init__()
        self.fail_start = fail_start
        self.fail_ready = fail_ready
        self.start_calls = 0
        self.shutdown_calls = 0
        self.shutdown_seen_states = []

    async def _do_start(self, wait: bool):
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("boom")
        return [_FakeEndpoint()]

    async def _do_shutdown(self) -> None:
        self.shutdown_calls += 1
        self.shutdown_seen_states.append(self.state)

    async def _do_ready(self) -> bool:
        return not self.fail_ready


def test_data_backend_is_abstract():
    with pytest.raises(TypeError):
        DataBackend()


async def test_endpoints_before_start_raises():
    backend = _FakeDataBackend()
    with pytest.raises(DataBackendNotReadyError):
        _ = backend.endpoints


async def test_start_then_endpoints():
    backend = _FakeDataBackend()
    await backend.start()
    assert backend.state is DataBackendState.READY
    eps = backend.endpoints
    assert len(eps) == 1
    assert eps[0].serialize() == "fake"


async def test_start_and_shutdown_return_self_for_fluent_usage():
    backend = await _FakeDataBackend().start()
    assert isinstance(backend, _FakeDataBackend)
    assert backend.state is DataBackendState.READY

    returned = await backend.shutdown()
    assert returned is backend
    assert backend.state is DataBackendState.SHUTDOWN


async def test_repeated_start_is_idempotent():
    backend = _FakeDataBackend()
    await backend.start()
    await backend.start()
    await backend.start()
    assert backend.start_calls == 1


async def test_repeated_shutdown_is_idempotent():
    backend = _FakeDataBackend()
    await backend.start()
    await backend.shutdown()
    await backend.shutdown()
    await backend.shutdown()
    assert backend.shutdown_calls == 1


async def test_shutdown_never_started_is_noop_but_calls_hook_once():
    backend = _FakeDataBackend()
    await backend.shutdown()
    assert backend.state is DataBackendState.SHUTDOWN
    assert backend.shutdown_calls == 1
    assert backend.shutdown_seen_states == [DataBackendState.CREATED]


async def test_start_after_shutdown_raises():
    backend = _FakeDataBackend()
    await backend.start()
    await backend.shutdown()
    with pytest.raises(DataBackendTerminatedError):
        await backend.start()


async def test_failing_start_sets_failed_and_raises_with_cause():
    backend = _FakeDataBackend(fail_start=True)
    with pytest.raises(DataBackendStartupError) as excinfo:
        await backend.start()
    assert backend.state is DataBackendState.FAILED
    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_start_after_failure_raises_terminated():
    backend = _FakeDataBackend(fail_start=True)
    with pytest.raises(DataBackendStartupError):
        await backend.start()
    with pytest.raises(DataBackendTerminatedError):
        await backend.start()


async def test_shutdown_after_failure_is_noop_and_invokes_hook_once():
    backend = _FakeDataBackend(fail_start=True)
    with pytest.raises(DataBackendStartupError):
        await backend.start()
    await backend.shutdown()
    assert backend.state is DataBackendState.SHUTDOWN
    assert backend.shutdown_calls == 1


async def test_ready_reflects_lifecycle():
    backend = _FakeDataBackend()
    assert await backend.ready() is False
    await backend.start()
    assert await backend.ready() is True
    await backend.shutdown()
    assert await backend.ready() is False


async def test_concurrent_start_calls_do_start_once():
    backend = _FakeDataBackend()
    await asyncio.gather(backend.start(), backend.start(), backend.start())
    assert backend.start_calls == 1
    assert backend.state is DataBackendState.READY
