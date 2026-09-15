"""
Rhapsody: Runtime system for executing heterogeneous HPC-AI workflows.

Rhapsody provides execution backends for scientific workflows, enabling execution
on various computing infrastructures including HPC clusters, local machines,
and distributed computing systems.
"""

from __future__ import annotations

from .api import AITask
from .api import BackendError
from .api import BaseTask
from .api import ComputeTask
from .api import ResourceError
from .api import RhapsodyError
from .api import Session
from .api import SessionError
from .api import TaskExecutionError
from .api import TaskValidationError
from .backends import BackendRegistry
from .backends import discover_backends
from .backends import get_backend
from .logger import enable_logging

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "get_backend",
    "discover_backends",
    "BackendRegistry",
    "enable_logging",
    # Task API
    "BaseTask",
    "ComputeTask",
    "AITask",
    "Session",
    # Errors
    "RhapsodyError",
    "BackendError",
    "TaskValidationError",
    "TaskExecutionError",
    "SessionError",
    "ResourceError",
]
