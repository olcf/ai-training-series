"""Unit tests for RedisDataBackend.

Tests requiring a real `redis-server` binary are marked `redis` and skip
cleanly when it isn't on PATH.
"""

import asyncio
import shutil
import time

import pytest

from rhapsody.backends.data.base import DataBackendStartupError
from rhapsody.backends.data.base import DataBackendState
from rhapsody.backends.data.redis import RedisDataBackend


@pytest.fixture(scope="session")
def _requires_redis_server():
    if shutil.which("redis-server") is None:
        pytest.skip("redis-server binary not found on PATH")
    yield


@pytest.mark.redis
async def test_zero_arg_local_roundtrip(_requires_redis_server):
    backend = RedisDataBackend()
    await backend.start()
    try:
        assert backend.state is DataBackendState.READY
        eps = backend.endpoints
        assert len(eps) == 1
        assert eps[0].host == "localhost"
        assert await backend.ready() is True
    finally:
        await backend.shutdown()
    assert backend.state is DataBackendState.SHUTDOWN


@pytest.mark.redis
async def test_multi_node_concurrent_launch(_requires_redis_server):
    single = RedisDataBackend()
    t0 = time.monotonic()
    await single.start()
    single_elapsed = time.monotonic() - t0
    await single.shutdown()

    backend = RedisDataBackend(hosts=["localhost"] * 4)
    t0 = time.monotonic()
    await backend.start()
    multi_elapsed = time.monotonic() - t0
    try:
        eps = backend.endpoints
        assert len(eps) == 4
        assert len({ep.port for ep in eps}) == 4
        assert await backend.ready() is True
        # Loose bound: concurrent launch of 4 nodes should be nowhere near
        # 4x a single node's startup time.
        assert multi_elapsed < 3 * max(single_elapsed, 0.5)
    finally:
        await backend.shutdown()


@pytest.mark.redis
async def test_partial_failure_rolls_back_all_nodes(_requires_redis_server, monkeypatch):
    backend = RedisDataBackend(hosts=["localhost"] * 3, startup_timeout=1.0, poll_interval=0.05)
    seen: dict[tuple[str, int], int] = {}
    orig_ping = RedisDataBackend._ping

    async def fake_ping(self, host, port):
        key = (host, port)
        if key not in seen:
            seen[key] = len(seen)
        if seen[key] == 1:
            return False
        return await orig_ping(self, host, port)

    monkeypatch.setattr(RedisDataBackend, "_ping", fake_ping)

    with pytest.raises(DataBackendStartupError):
        await backend.start()

    assert backend.state is DataBackendState.FAILED
    assert backend._processes == {}


async def test_terminate_then_kill_after_grace_period():
    backend = RedisDataBackend(shutdown_grace_period=0.05)

    class _StubProc:
        def __init__(self):
            self.pid = 12345
            self.returncode = None
            self.terminate_called = False
            self.kill_called = False

        def terminate(self):
            self.terminate_called = True

        def kill(self):
            self.kill_called = True
            self.returncode = -9

        async def wait(self):
            if not self.kill_called:
                await asyncio.sleep(9999)
            return self.returncode

    proc = _StubProc()
    await backend._terminate_one(proc)
    assert proc.terminate_called is True
    assert proc.kill_called is True


async def test_terminate_one_skips_already_exited_process():
    backend = RedisDataBackend()

    class _ExitedProc:
        returncode = 0

        def terminate(self):
            raise AssertionError("should not be called")

        def kill(self):
            raise AssertionError("should not be called")

        async def wait(self):
            raise AssertionError("should not be called")

    await backend._terminate_one(_ExitedProc())
