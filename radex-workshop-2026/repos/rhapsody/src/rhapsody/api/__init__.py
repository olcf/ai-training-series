"""RHAPSODY API module.

This module provides high-level API components for defining and managing computational and AI
inference tasks in RHAPSODY.
"""

from .errors import BackendError
from .errors import ResourceError
from .errors import RhapsodyError
from .errors import SessionError
from .errors import TaskExecutionError
from .errors import TaskValidationError
from .session import Session
from .session import TaskStateManager
from .task import AITask
from .task import BaseTask
from .task import ComputeTask

__all__ = [
    # Task classes
    "BaseTask",
    "ComputeTask",
    "AITask",
    "Session",
    "TaskStateManager",
    # Error classes
    "RhapsodyError",
    "BackendError",
    "TaskValidationError",
    "TaskExecutionError",
    "SessionError",
    "ResourceError",
]
