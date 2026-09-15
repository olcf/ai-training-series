"""Tests for DragonDataBackend.

Run with:
    dragon python -m pytest tests/unit/test_backend_data_dragon.py -v
"""

import pytest

# Skip the entire module when the Dragon runtime is not installed.
pytest.importorskip("dragon", reason="Dragon is required for Dragon data backend tests")

from rhapsody.backends.data.base import DataBackendState  # noqa: E402
from rhapsody.backends.data.base import DataBackendTerminatedError  # noqa: E402
from rhapsody.backends.data.dragon import DragonDataBackend  # noqa: E402


def test_wait_for_keys_false_raises():
    with pytest.raises(ValueError, match=r"wait_for_keys"):
        DragonDataBackend(_wait_for_keys=False)


def test_wait_for_keys_via_kwargs_raises():
    with pytest.raises(ValueError, match=r"wait_for_keys"):
        DragonDataBackend(wait_for_keys=False)


def test_default_construction_does_not_raise():
    DragonDataBackend()


async def test_zero_arg_start_ready_shutdown_roundtrip():
    # Covers the working_set_size default-injection: DDict's own default
    # (1) is incompatible with the wait_for_keys=True this class always
    # forces, so a bare DragonDataBackend() must still construct cleanly.
    backend = DragonDataBackend()
    await backend.start()
    try:
        assert backend.state is DataBackendState.READY
        assert backend.endpoints[0].descriptor
        assert await backend.ready() is True
    finally:
        await backend.shutdown()
    assert backend.state is DataBackendState.SHUTDOWN


async def test_start_ready_shutdown_roundtrip():
    backend = DragonDataBackend(managers_per_node=1, n_nodes=1)
    await backend.start()
    try:
        assert backend.state is DataBackendState.READY
        eps = backend.endpoints
        assert len(eps) == 1
        assert eps[0].descriptor
        assert await backend.ready() is True
    finally:
        await backend.shutdown()
    assert backend.state is DataBackendState.SHUTDOWN


async def test_start_after_shutdown_raises():
    backend = DragonDataBackend(managers_per_node=1, n_nodes=1)
    await backend.start()
    await backend.shutdown()
    with pytest.raises(DataBackendTerminatedError):
        await backend.start()


async def test_wait_false_is_noop_for_dragon():
    backend = DragonDataBackend(managers_per_node=1, n_nodes=1)
    await backend.start(wait=False)
    try:
        assert backend.state is DataBackendState.READY
        assert backend.endpoints[0].descriptor
    finally:
        await backend.shutdown()
