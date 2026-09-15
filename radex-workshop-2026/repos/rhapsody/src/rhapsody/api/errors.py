"""Exception classes for RHAPSODY API.

This module defines custom exceptions for different error scenarios in RHAPSODY, allowing fine-
grained error handling and differentiation between backend failures and task-level issues.
"""

from __future__ import annotations


class RhapsodyError(Exception):
    """Base exception class for all RHAPSODY errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class BackendError(RhapsodyError):
    """Exception raised when a backend itself fails.

    This indicates a failure in the backend infrastructure (e.g., connection failure,
    resource unavailability, backend crash) rather than a task execution failure.

    Examples:
        - Backend failed to initialize
        - Lost connection to execution infrastructure
        - Backend shutdown unexpectedly
        - Resource manager became unavailable
    """

    def __init__(self, backend_name: str, message: str):
        self.backend_name = backend_name
        super().__init__(f"Backend '{backend_name}' failed: {message}")


class TaskValidationError(RhapsodyError):
    """Exception raised when task validation fails.

    This indicates that a task definition is invalid or incomplete,
    such as missing required fields or conflicting parameters.

    Examples:
        - Missing required field (uid, executable/function, prompt)
        - Both executable and function specified
        - Invalid field types
        - Conflicting parameters
    """

    def __init__(self, task_uid: str | None, message: str):
        self.task_uid = task_uid
        if task_uid:
            super().__init__(f"Task '{task_uid}' validation failed: {message}")
        else:
            super().__init__(f"Task validation failed: {message}")


class TaskExecutionError(RhapsodyError):
    """Exception raised when a task execution fails.

    This indicates that a task failed during execution (not a backend failure),
    such as a non-zero exit code, runtime error, or timeout.

    Examples:
        - Executable returned non-zero exit code
        - Function raised an exception
        - Task timed out
        - Resource requirements not met
    """

    def __init__(self, task_uid: str, message: str, exit_code: int | None = None):
        self.task_uid = task_uid
        self.exit_code = exit_code
        if exit_code is not None:
            super().__init__(f"Task '{task_uid}' failed with exit code {exit_code}: {message}")
        else:
            super().__init__(f"Task '{task_uid}' failed: {message}")


class SessionError(RhapsodyError):
    """Exception raised when session operations fail.

    This indicates issues with session management, such as trying to use
    a closed session or invalid session configuration.

    Examples:
        - Attempting to use closed session
        - Invalid session configuration
        - Session already exists
    """

    pass


class ResourceError(RhapsodyError):
    """Exception raised when resource requirements cannot be met.

    This indicates that requested resources (memory, GPUs, CPUs) are
    unavailable or invalid.

    Examples:
        - Insufficient memory available
        - Requested GPUs not available
        - Invalid resource specification
    """

    def __init__(self, task_uid: str | None, resource_type: str, message: str):
        self.task_uid = task_uid
        self.resource_type = resource_type
        if task_uid:
            super().__init__(f"Task '{task_uid}' resource error ({resource_type}): {message}")
        else:
            super().__init__(f"Resource error ({resource_type}): {message}")
