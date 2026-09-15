"""Backend-independent lifecycle abstraction for RHAPSODY data infrastructure."""

from __future__ import annotations

import abc
import asyncio
import enum


class DataBackendState(enum.Enum):
    """Lifecycle state of a `DataBackend`."""

    CREATED = "CREATED"
    STARTING = "STARTING"
    READY = "READY"
    FAILED = "FAILED"
    SHUTDOWN = "SHUTDOWN"


_TERMINAL_STATES = frozenset({DataBackendState.FAILED, DataBackendState.SHUTDOWN})


class DataBackendError(Exception):
    """Base class for all rhapsody.data errors."""


class DataBackendStartupError(DataBackendError):
    """Raised when a DataBackend fails to reach READY during start()."""


class DataBackendStateError(DataBackendError):
    """Raised when a DataBackend method is invoked in an invalid lifecycle state."""


class DataBackendNotReadyError(DataBackendStateError):
    """Raised by `.endpoints` before a successful start()."""


class DataBackendTerminatedError(DataBackendStateError):
    """Raised by start() on a DataBackend that is already FAILED or SHUTDOWN."""


class Endpoint(abc.ABC):
    """Connection information for one DataBackend instance/node.

    Deliberately generic: a Redis node has a host/port, a Dragon DDict has an
    opaque serialized descriptor, and neither concept is assumed by callers
    that only depend on this interface. An `Endpoint` never constructs a
    client itself -- callers build whatever client they need from
    `serialize()`'s output.
    """

    @abc.abstractmethod
    def serialize(self) -> str:
        """Return a string a caller can pass through explicitly (env var, kwarg, task arg) to
        reconnect a client elsewhere."""


class DataBackend(abc.ABC):
    """Backend-independent lifecycle for a RHAPSODY data-infrastructure backend.

    State machine: CREATED -> STARTING -> {READY, FAILED};
    {CREATED, READY, FAILED} -> SHUTDOWN. FAILED and SHUTDOWN are terminal --
    a DataBackend cannot be restarted once it lands in either; construct a
    new instance instead.

    `start()`/`shutdown()` each run their body under a single per-instance
    lock, which is what makes repeated/concurrent calls idempotent and the
    terminal-state guard correct without any state duplicated in subclasses.
    """

    def __init__(self, *, name: str | None = None) -> None:
        self._name = name
        self._state: DataBackendState = DataBackendState.CREATED
        self._endpoints: list[Endpoint] = []
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name or type(self).__name__

    @property
    def state(self) -> DataBackendState:
        return self._state

    @property
    def endpoints(self) -> list[Endpoint]:
        if self._state is not DataBackendState.READY:
            raise DataBackendNotReadyError(
                f"{type(self).__name__} is not ready (state={self._state.name}); "
                "call `await backend.start()` first."
            )
        return list(self._endpoints)

    async def ready(self) -> bool:
        if self._state is not DataBackendState.READY:
            return False
        return await self._do_ready()

    def __await__(self):
        """Allow `await RedisDataBackend(...)` to start it and return self, matching BaseBackend's
        `await ConcurrentExecutionBackend(...)` convention."""
        return self.start().__await__()

    async def start(self, wait: bool = True) -> DataBackend:
        """Start the backend and return self, so both `await backend.start()` and `backend = await
        RedisDataBackend(...).start()` work."""
        async with self._lock:
            if self._state in _TERMINAL_STATES:
                raise DataBackendTerminatedError(
                    f"{type(self).__name__} is in terminal state "
                    f"{self._state.name} and cannot be started again; "
                    "construct a new DataBackend instance instead."
                )
            if self._state is DataBackendState.READY:
                return self
            self._state = DataBackendState.STARTING
            try:
                endpoints = await self._do_start(wait=wait)
            except BaseException as exc:
                self._state = DataBackendState.FAILED
                self._endpoints = []
                if isinstance(exc, DataBackendError) or not isinstance(exc, Exception):
                    # Already a DataBackendError, or a BaseException we must
                    # not mask (CancelledError, KeyboardInterrupt, SystemExit).
                    raise
                raise DataBackendStartupError(f"{type(self).__name__} failed to start") from exc
            else:
                self._endpoints = list(endpoints)
                self._state = DataBackendState.READY
                return self

    async def shutdown(self) -> DataBackend:
        """Tear down the backend and return self, for the same fluent usage as start()."""
        async with self._lock:
            if self._state is DataBackendState.SHUTDOWN:
                return self
            try:
                await self._do_shutdown()
            finally:
                self._state = DataBackendState.SHUTDOWN
                self._endpoints = []
            return self

    @abc.abstractmethod
    async def _do_start(self, wait: bool) -> list[Endpoint]:
        """Launch the backend and return its endpoint(s).

        Raise on failure after cleaning up any partially-started state.
        """

    @abc.abstractmethod
    async def _do_shutdown(self) -> None:
        """Tear down the backend.

        Must tolerate being called with empty or partial internal state (never-started, or a failed
        start).
        """

    @abc.abstractmethod
    async def _do_ready(self) -> bool:
        """Live readiness check, never cached."""
