"""Base classes and interfaces for Rhapsody execution backends.

This module defines the abstract base class and core interfaces that all execution backends must
implement.
"""

from __future__ import annotations

import os
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

if TYPE_CHECKING:
    pass


class BaseBackend(ABC):
    """Abstract base class for backends that manage task execution and state.

    This class defines the interface for backends that handle task submission, state management, and
    lifecycle in a distributed or parallel execution environment.

    Attributes:
        _work_dir: Output directory for capture_stdio files. Defaults to cwd;
            overwritten by Session.add_backend / WorkflowEngine._attach_backend.
        is_attached: True once this backend has been registered with a session
            or workflow engine.
        attached_to: Ordered list of session/engine UIDs this backend has been
            attached to (most recent last).
    """

    def __init__(self, name: str | None = None):
        """Initialize the backend."""
        self._name = name
        # Default output directory for capture_stdio tasks; overridden by Session.add_backend
        self._work_dir: str = os.getcwd()
        self.is_attached: bool = False
        self.attached_to: list[str] = []

    @property
    def name(self) -> str:
        """Name of the backend."""
        if self._name:
            return self._name
        return self.__class__.__name__

    @abstractmethod
    async def submit_tasks(self, tasks: list[dict]) -> None:
        """Submit a list of tasks for execution.

        Args:
            tasks: A list of task dictionaries containing task definitions.
                Task objects (ComputeTask, AITask) inherit from dict and can be
                passed directly. Each task should contain the necessary information
                for task execution.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the execution backend.

        This method should clean up resources, terminate running tasks if necessary, and prepare the
        backend for termination.
        """
        pass

    @abstractmethod
    def state(self) -> str:
        """Get the current state of the execution backend.

        Returns:
            A string representing the current state of the backend (e.g., 'running',
            'idle', 'shutting_down', 'error').
        """
        pass

    @abstractmethod
    def task_state_cb(self, task: dict, state: str) -> None:
        """Callback function invoked when a task's state changes.

        Args:
            task: Dictionary containing task information and metadata.
            state: The new state of the task (e.g., 'pending', 'running', 'completed',
                'failed').
        """
        pass

    def register_callback(self, func: Callable[[dict[str, Any], str], None]) -> None:
        """Register a callback function for task state changes.

        This chains the user's callback with the internal callback used by wait_tasks().
        Both callbacks will be invoked on every state change.

        Args:
            func: A callable that will be invoked when task states change.
                The function should accept task and state parameters.
        """
        self._callback_func = func

    @abstractmethod
    def get_task_states_map(self) -> Any:
        """Retrieve a mapping of task IDs to their current states.

        Returns:
            A dictionary mapping task identifiers to their current execution states.
        """
        pass

    @abstractmethod
    def build_task(self, task: dict) -> None:
        """Build or prepare a task for execution.

        Args:
            task: Dictionary containing task definition, parameters, and metadata
                required for task construction.
        """
        pass

    def link_implicit_data_deps(self, src_task: dict[str, Any], dst_task: dict[str, Any]) -> None:  # noqa: B027
        """Link implicit data dependencies between two tasks.

        Creates a dependency relationship where the destination task depends on
        data produced by the source task, with the dependency being inferred
        automatically.

        Args:
            src_task: The source task that produces data.
            dst_task: The destination task that depends on the source task's output.
        """
        pass

    def link_explicit_data_deps(  # noqa: B027
        self,
        src_task: dict[str, Any] | None = None,
        dst_task: dict[str, Any] | None = None,
        file_name: str | None = None,
        file_path: str | None = None,
    ) -> None:
        """Link explicit data dependencies between tasks or files.

        Creates explicit dependency relationships based on specified file names
        or paths, allowing for more precise control over task execution order.

        Args:
            src_task: The source task that produces the dependency.
            dst_task: The destination task that depends on the source.
            file_name: Name of the file that represents the dependency.
            file_path: Full path to the file that represents the dependency.
        """
        pass

    @abstractmethod
    async def cancel_task(self, uid: str) -> bool:
        """Cancel a task in the execution backend.

        Args:
            uid: Task identifier

        Raises:
            NotImplementedError: If the backend doesn't support cancellation
        """
        raise NotImplementedError("Not implemented in the base backend")
