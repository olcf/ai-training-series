"""Backend subsystem for Rhapsody.

This module provides execution backends for running scientific workflows on various computing
infrastructures.
"""

from __future__ import annotations

from rhapsody.api import Session

from .base import BaseBackend
from .constants import StateMapper
from .constants import TasksMainStates
from .discovery import BackendRegistry
from .discovery import discover_backends
from .discovery import get_backend

# Import all execution backends for convenient access
from .execution import ConcurrentExecutionBackend

__all__ = [
    "BackendRegistry",
    "get_backend",
    "discover_backends",
    "BaseBackend",
    "Session",
    "TasksMainStates",
    "StateMapper",
    "ConcurrentExecutionBackend",
]

# Try to import optional execution backends
try:
    from .execution import DaskExecutionBackend  # noqa: F401

    __all__.append("DaskExecutionBackend")
except ImportError:
    pass

try:
    from .execution import RadicalExecutionBackend  # noqa: F401

    __all__.append("RadicalExecutionBackend")
except ImportError:
    pass

try:
    from .execution import DragonExecutionBackend  # noqa: F401

    __all__.append("DragonExecutionBackend")
except ImportError:
    pass

# Try to import optional AI (inference) backends
try:
    from .ai.vllm import DragonVllmInferenceBackend  # noqa: F401

    __all__.append("DragonVllmInferenceBackend")
except ImportError:
    pass
