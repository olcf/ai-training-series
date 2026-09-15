"""Data infrastructure subsystem for Rhapsody.

This module provides backends that launch and own the lifecycle of data infrastructure (a Redis
server, a Dragon DDict) and hand back connection endpoints, mirroring how execution/inference
backends launch and own compute/inference infrastructure.
"""

from __future__ import annotations

from .base import DataBackend
from .base import DataBackendError
from .base import DataBackendNotReadyError
from .base import DataBackendStartupError
from .base import DataBackendState
from .base import DataBackendStateError
from .base import DataBackendTerminatedError
from .base import Endpoint
from .redis import RedisDataBackend
from .redis import RedisEndpoint

__all__ = [
    "DataBackend",
    "DataBackendError",
    "DataBackendNotReadyError",
    "DataBackendStartupError",
    "DataBackendState",
    "DataBackendStateError",
    "DataBackendTerminatedError",
    "Endpoint",
    "RedisDataBackend",
    "RedisEndpoint",
]

# Try to import the optional Dragon-backed data backend
try:
    from .dragon import DragonDataBackend  # noqa: F401
    from .dragon import DragonEndpoint  # noqa: F401

    __all__.append("DragonDataBackend")
    __all__.append("DragonEndpoint")
except ImportError:
    pass
