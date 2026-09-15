"""DragonDataBackend: launches and owns the lifecycle of a Dragon DDict."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any

from rhapsody.backends.data.base import DataBackend
from rhapsody.backends.data.base import DataBackendStartupError
from rhapsody.backends.data.base import Endpoint

try:
    import dragon  # noqa: F401
    from dragon.data.ddict import DDict as _DDict
except ImportError:  # pragma: no cover - environment without Dragon
    dragon = None
    _DDict = None


_PROBE_KEY = "__rhapsody_data_backend_liveness_probe__"


def _get_logger() -> logging.Logger:
    """Get logger for the dragon data backend module.

    This function provides lazy logger evaluation, ensuring the logger is created after the user has
    configured logging, not at module import time.
    """
    return logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DragonEndpoint(Endpoint):
    """Connection information for a Dragon DDict-backed backend.

    Unlike Redis, there is no host/port here -- `serialize()` returns the
    opaque base64 descriptor produced by `DDict.serialize()`.
    `DragonEndpoint` never constructs a client itself -- build one directly
    from the descriptor, e.g. `radex.clients.core.DragonClient(descriptor=endpoint.serialize(), timeout=5)`
    or `DDict.attach(endpoint.serialize())`.
    """

    descriptor: str

    def serialize(self) -> str:
        return self.descriptor


class DragonDataBackend(DataBackend):
    """A DataBackend backed by a single Dragon DDict.

    Constructing a `dragon.data.ddict.DDict` is itself the blocking startup
    call -- it spins up the orchestrator and manager processes and blocks
    until ready, with no separate "start" step. This wraps that blocking
    call in `asyncio.to_thread`, matching Dragon's own established
    convention for calling blocking Dragon primitives from asyncio.

    `wait_for_keys` is hard-enforced to True. The constructor parameter is
    named `_wait_for_keys` (leading underscore) to make clear it isn't
    meant to be set by normal callers -- it exists so tests can exercise
    the rejection path.

    All other `DDict` construction arguments are accepted as arbitrary
    keyword arguments and forwarded to `DDict(...)` unchanged -- this
    class does not mirror or re-validate `DDict.__init__`'s own parameter
    list. If you pass something `DDict` doesn't like, `DDict` raises its
    own error; see `dragon.data.ddict.DDict` for the full set of accepted
    keyword arguments. The one adjustment made on your behalf: if you
    don't pass `working_set_size`, it defaults to `2` (`DDict`'s own
    default of `1` is incompatible with the `wait_for_keys=True` this
    class always forces) -- pass it explicitly to override.
    """

    def __init__(
        self, *, name: str = "dragon", _wait_for_keys: bool = True, **ddict_kwargs: Any
    ) -> None:
        """Initialize a DragonDataBackend.

        Args:
            name: Name this backend is registered under when attached to a
                Session.
            _wait_for_keys: Must be True (the default) -- exists only so
                tests can exercise the rejection path. Do not set this.
            **ddict_kwargs: Forwarded directly to `dragon.data.ddict.DDict`.
                See its docstring for the full set of accepted arguments.

        Raises:
            ImportError: If the `dragon` package is not importable.
            ValueError: If `_wait_for_keys` is not True, or `ddict_kwargs`
                tries to set `wait_for_keys`.
        """
        super().__init__(name=name)
        self.logger = _get_logger()
        if _DDict is None:
            raise ImportError(
                "The 'dragon' package is required to use DragonDataBackend. "
                "It is not pip-installable from PyPI; install it per your "
                "Dragon distribution/environment first."
            )
        if _wait_for_keys is not True:
            raise ValueError(
                "DragonDataBackend requires wait_for_keys=True: downstream "
                "clients (e.g. the compiled radex::drg::ddict::Client) "
                "refuse to attach to a DDict created with "
                "wait_for_keys=False."
            )
        if "wait_for_keys" in ddict_kwargs:
            raise ValueError(
                "'wait_for_keys' must not be passed as a keyword argument: "
                "it would silently override the enforced "
                "wait_for_keys=True above. Use the _wait_for_keys "
                "constructor parameter instead (and only for tests "
                "exercising the rejection path)."
            )

        self._ddict_kwargs: dict[str, Any] = dict(ddict_kwargs)
        self._ddict_kwargs["wait_for_keys"] = True
        self._ddict_kwargs.setdefault("working_set_size", 2)
        self._ddict: Any = None

    async def _do_start(self, wait: bool) -> list[Endpoint]:
        # `wait` is accepted for interface parity with DataBackend.start()
        # but has no effect: DDict.__init__ is already atomically
        # blocking-until-ready -- there is no separate "start" step to skip
        # waiting on.
        self.logger.info("Starting DragonDataBackend (constructing DDict)...")
        try:
            self._ddict = await asyncio.to_thread(_DDict, **self._ddict_kwargs)
        except Exception as exc:
            self._ddict = None
            self.logger.error("DragonDataBackend failed to construct DDict: %s", exc)
            raise DataBackendStartupError(
                f"DragonDataBackend failed to construct DDict: {exc}"
            ) from exc
        descriptor = self._ddict.serialize()
        self.logger.info("DragonDataBackend ready at %s...", descriptor[:32])
        return [DragonEndpoint(descriptor=descriptor)]

    async def _do_shutdown(self) -> None:
        self.logger.info("Shutting down DragonDataBackend...")
        if self._ddict is not None:
            await asyncio.to_thread(self._ddict.destroy)
        self._ddict = None
        self.logger.info("DragonDataBackend shutdown complete")

    async def _do_ready(self) -> bool:
        if self._ddict is None:
            return False

        # TODO: this probe-key check is a stand-in for a real health check.
        # Wire in a more direct Dragon-native liveness API here once one is
        # available, instead of a synthetic containment lookup.
        def _probe() -> bool:
            try:
                _ = _PROBE_KEY in self._ddict
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_probe)
