"""Constants and state management for Rhapsody execution backends.

This module defines task states, state mapping utilities, and other constants used across different
execution backends.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class BackendMainStates(Enum):
    """Enumeration of standard backend states used across all backends.

    This enum defines the canonical backend states that are common to all
    execution backends, providing a unified interface for backend state management.

    Attributes:
        INITIALIZED: backend created, not yet executing
        RUNNING: backend actively executing work
        SHUTDOWN: backend fully stopped
    """

    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    SHUTDOWN = "SHUTDOWN"


class TasksMainStates(Enum):
    """Enumeration of standard task states used across all backends.

    This enum defines the canonical task states that are common to all
    execution backends, providing a unified interface for task state management.

    Attributes:
        DONE: Task completed successfully.
        FAILED: Task execution failed.
        CANCELED: Task was canceled before completion.
        RUNNING: Task is currently executing.
    """

    DONE = "DONE"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    RUNNING = "RUNNING"


class StateMapper:
    """Unified interface for mapping task states between main workflow and backend systems.

    StateMapper provides a centralized mechanism for translating task states between
    the main workflow system and various backend execution systems
    (e.g., 'radical', 'dask').
    It supports dynamic registration of backend-specific state mappings and
    bidirectional conversion between main states and backend-specific states.

    The class maintains a registry of backend state mappings and provides methods
    for state conversion, backend detection, and direct state access through
    attribute notation.

    Attributes:
        _backend_registry (dict[str, dict[TasksMainStates, Any]]): Class-level registry
            mapping backend identifiers to their state mappings.
        backend_name (str): Name of the current backend.
        backend_module (Optional[Any]): Reference to the backend module/object.
        _state_map (dict[TasksMainStates, Any]): Current backend's state mappings.
        _reverse_map (dict[Any, TasksMainStates]): Reverse mapping from backend states
            to main states.

    Args:
        backend (Union[str, Any]): The backend identifier, either as a string
            (e.g., 'radical', 'dask') or as a backend module/object instance.

    Raises:
        ValueError: If the specified backend is not registered or cannot be detected.

    Example:
        ::

            # Register a backend with custom states
            StateMapper.register_backend_tasks_states(
                backend='my_backend',
                done_state='COMPLETED',
                failed_state='ERROR',
                canceled_state='ABORTED',
                running_state='ACTIVE'
            )

            # Use the mapper
            mapper = StateMapper('my_backend')
            backend_state = mapper.DONE  # Returns 'COMPLETED'
            # Returns TasksMainStates.DONE
            main_state = mapper.to_main_state('COMPLETED')
    """

    _backend_registry: dict[str, dict[TasksMainStates, Any]] = {}

    def __init__(self, backend: str | Any):
        """Initialize StateMapper with a specific backend.

        Creates a StateMapper instance configured for the specified backend.
        The backend can be provided as either a string identifier or a
        module/object instance.

        Args:
            backend (Union[str, Any]): Backend identifier. Can be:
                - String: Backend name like 'radical', 'dask', etc.
                - Object: Backend module or instance from which name is detected.

        Raises:
            ValueError: If the backend is not registered in the registry or
                if backend name cannot be detected from the provided object.

        Note:
            The backend must be registered using register_backend_tasks_states() or
            register_backend_tasks_states_with_defaults() before initialization.
        """
        self.backend_name: str
        self.backend_module: Any | None = None

        if isinstance(backend, str):
            self.backend_name = backend.lower()
        else:
            self.backend_module = backend
            self.backend_name = self._detect_backend_name()

        if self.backend_name not in self._backend_registry:
            raise ValueError(
                f"Backend '{self.backend_name}' not registered. "
                f"Available backends: {list(self._backend_registry.keys())}"
            )

        self._state_map = self._backend_registry[self.backend_name]
        self._reverse_map = {v: k for k, v in self._state_map.items()}

    @classmethod
    def register_backend_states(
        cls,
        backend: str | Any,
        initialized_state: Any,
        running_state: Any,
        shutdown_state: Any,
        **additional_states: Any,
    ) -> None:
        """Register backend state mappings for a backend.

        Associates a backend identifier with its corresponding backend state values,
        mapping the standard backend states to backend-specific representations.

        Args:
            backend (Union[str, Any]): The identifier for the backend to register.
            initialized_state (Any): Backend's representation of INITIALIZED state.
            running_state (Any): Backend's representation of RUNNING state.
            shutdown_state (Any): Backend's representation of SHUTDOWN state.
            **additional_states: Additional state mappings beyond the core states.

        Returns:
            None

        Example:
            ::

                StateMapper.register_backend_states(
                    backend='my_backend',
                    initialized_state='INIT',
                    running_state='ACTIVE',
                    shutdown_state='TERMINATED'
                )
        """
        # Convert backend to string key for consistent lookup
        if isinstance(backend, str):
            backend_key = f"{backend.lower()}_backend_states"
        else:
            # Detect backend name from object
            if hasattr(backend, "__name__"):
                backend_key = f"{backend.__name__.lower()}_backend_states"
            elif hasattr(backend, "__class__"):
                backend_key = f"{backend.__class__.__name__.lower()}_backend_states"
            else:
                raise ValueError(f"Could not detect backend name from {backend}")

        additional_mapped = {BackendMainStates[k.upper()]: v for k, v in additional_states.items()}
        cls._backend_registry[backend_key] = {
            BackendMainStates.INITIALIZED: initialized_state,
            BackendMainStates.RUNNING: running_state,
            BackendMainStates.SHUTDOWN: shutdown_state,
            **additional_mapped,
        }

    @classmethod
    def register_backend_states_with_defaults(cls, backend: str | Any) -> None:
        """Register backend states using default main state values.

        Convenience method that registers backend states where the backend-specific
        states are identical to the main state values (i.e., the string values
        of the BackendMainStates enum).

        Args:
            backend (Union[str, Any]): The backend identifier to register.

        Returns:
            None

        Example:
            ::

                # Registers backend states as:
                # INITIALIZED -> "INITIALIZED", RUNNING -> "RUNNING", etc.
                StateMapper.register_backend_states_with_defaults('my_backend')
        """
        return cls.register_backend_states(
            backend,
            initialized_state=BackendMainStates.INITIALIZED.value,
            running_state=BackendMainStates.RUNNING.value,
            shutdown_state=BackendMainStates.SHUTDOWN.value,
        )

    @classmethod
    def register_backend_tasks_states(
        cls,
        backend: str | Any,
        done_state: Any,
        failed_state: Any,
        canceled_state: Any,
        running_state: Any,
        **additional_states: Any,
    ) -> None:
        """Register tasks state mappings for a new backend.

        Associates a backend identifier with its corresponding task state values,
        mapping the standard task states to backend-specific representations.
        Supports additional custom state mappings beyond the four core states.

        Args:
            backend (Union[str, Any]): The identifier for the backend to register. Can be
                a string, module, or any hashable object.
            done_state (Any): Backend's representation of the DONE state.
            failed_state (Any): Backend's representation of the FAILED state.
            canceled_state (Any): Backend's representation of the CANCELED state.
            running_state (Any): Backend's representation of the RUNNING state.
            **additional_states: Additional state mappings where the key is the
                state name (str) and the value is the backend's representation.
                Keys will be converted to uppercase and matched against TasksMainStates.

        Returns:
            None

        Example:
            ::

                StateMapper.register_backend_tasks_states(
                    backend='slurm',
                    done_state='COMPLETED',
                    failed_state='FAILED',
                    canceled_state='CANCELLED',
                    running_state='RUNNING',
                    pending='PENDING',
                    timeout='TIMEOUT'
                )
        """
        # Convert backend to string key for consistent lookup
        if isinstance(backend, str):
            backend_key = backend.lower()
        else:
            # Detect backend name from object for consistent key
            if hasattr(backend, "__name__"):
                backend_key = backend.__name__.lower()
            elif hasattr(backend, "__class__"):
                backend_key = backend.__class__.__name__.lower()
            else:
                raise ValueError(f"Could not detect backend name from {backend}")

        additional_mapped = {TasksMainStates[k.upper()]: v for k, v in additional_states.items()}
        cls._backend_registry[backend_key] = {
            TasksMainStates.DONE: done_state,
            TasksMainStates.FAILED: failed_state,
            TasksMainStates.CANCELED: canceled_state,
            TasksMainStates.RUNNING: running_state,
            **additional_mapped,
        }

    @classmethod
    def register_backend_tasks_states_with_defaults(cls, backend: str | Any) -> None:
        """Register a backend using default main state values.

        Convenience method that registers a backend where the backend-specific
        states are identical to the main state values (i.e., the string values
        of the TasksMainStates enum).

        Args:
            backend (Union[str, Any]): The backend identifier to register.

        Returns:
            The result of register_backend_tasks_states() with default values.

        Example:
            ::

                # This registers backend states as:
                # DONE -> "DONE", FAILED -> "FAILED", etc.
                StateMapper.register_backend_tasks_states_with_defaults('thread_backend')
        """
        return cls.register_backend_tasks_states(
            backend,
            done_state=TasksMainStates.DONE.value,
            failed_state=TasksMainStates.FAILED.value,
            canceled_state=TasksMainStates.CANCELED.value,
            running_state=TasksMainStates.RUNNING.value,
        )

    def _detect_backend_name(self) -> str:
        """Detect backend name from module/object.

        Attempts to extract a backend name from the provided module or object
        by examining its __name__ attribute or class name.

        Returns:
            str: The detected backend name in lowercase.

        Raises:
            ValueError: If backend name cannot be detected from the object.

        Detection Strategy:
            1. Check for __name__ attribute (for modules)
            2. Check for __class__.__name__ (for instances)
            3. Raise ValueError if neither is available
        """
        if hasattr(self.backend_module, "__name__"):
            module_name = self.backend_module.__name__.lower()
            return module_name

        # Try to detect from class name if it's an instance
        if hasattr(self.backend_module, "__class__"):
            class_name = self.backend_module.__class__.__name__.lower()
            return class_name

        raise ValueError(f"Could not detect backend from {self.backend_module}")

    def __getattr__(self, name: str) -> Any:
        """Access backend-specific states directly via attribute notation.

        Enables direct access to backend states using the main state names
        as attributes (e.g., mapper.DONE, mapper.FAILED).

        Args:
            name (str): The main state name to access (DONE, FAILED, CANCELED, RUNNING).

        Returns:
            Any: The backend-specific state value corresponding to the main state.

        Raises:
            AttributeError: If the specified state name is not valid.

        Example:
            ::

                mapper = StateMapper('my_backend')
                done_state = mapper.DONE  # Returns backend's DONE state
                running_state = mapper.RUNNING  # Returns backend's RUNNING state
        """
        try:
            main_state = TasksMainStates[name]
            return self._state_map[main_state]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' has no state '{name}'") from None

    def to_main_state(self, backend_state: Any) -> TasksMainStates:
        """Convert backend-specific state to main state.

        Translates a backend-specific state value back to the corresponding
        TasksMainStates enum value.

        Args:
            backend_state (Any): The backend-specific state value to convert.

        Returns:
            TasksMainStates: The corresponding main state enum value.

        Raises:
            ValueError: If the backend state is not recognized.

        Example:
            ::

                mapper = StateMapper('dragon')
                main_state = mapper.to_main_state('COMPLETED')  # TasksMainStates.DONE
        """
        try:
            return self._reverse_map[backend_state]
        except KeyError:
            raise ValueError(f"Unknown backend state: {backend_state}") from None

    def get_backend_state(self, main_state: TasksMainStates | str) -> Any:
        """Get backend-specific state for a main state.

        Retrieves the backend-specific state value that corresponds to the
        given main state. Accepts both TasksMainStates enum values and
        string representations.

        Args:
            main_state (Union[TasksMainStates, str]): The main state to convert.
                Can be a TasksMainStates enum value or its string representation.

        Returns:
            Any: The backend-specific state value.

        Raises:
            KeyError: If the main state is not found in the mapping.

        Example:
            ::

                mapper = StateMapper('my_backend')
                backend_state = mapper.get_backend_state(TasksMainStates.DONE)
                # Or using string
                backend_state = mapper.get_backend_state('DONE')
        """
        if isinstance(main_state, str):
            main_state = TasksMainStates[main_state]
        return self._state_map[main_state]

    @property
    def terminal_states(self) -> tuple:
        """Get all terminal states for the current backend.

        Returns a tuple containing the backend-specific representations of
        all terminal states (DONE, FAILED, CANCELED). These are states that
        indicate a task has finished execution and will not transition further.

        Returns:
            tuple: Backend-specific terminal state values (DONE, FAILED, CANCELED).

        Example:
            ::

                mapper = StateMapper('my_backend')
                terminals = mapper.terminal_states
                # Returns ('COMPLETED', 'ERROR', 'ABORTED') for example backend
        """
        return (self.DONE, self.FAILED, self.CANCELED)
