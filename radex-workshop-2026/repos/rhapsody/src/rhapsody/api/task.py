"""Task API for RHAPSODY.

This module provides lightweight, self-validating task classes for defining
computational and AI inference workloads in RHAPSODY.

Tasks inherit from dict for zero-overhead backend integration while providing
object-oriented interface for user convenience.

Design Goals:
    - Lightweight: Minimal overhead, fast instantiation
    - Self-validating: Validates fields at creation time
    - Extensible: Users can add custom fields via kwargs
    - Type-safe: Proper type hints and validation
    - Unified Interface: Works as both dict and object (task.uid or task['uid'])
    - Zero-Copy: Backends can update tasks in-place without conversion
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC
from abc import abstractmethod
from collections.abc import Generator
from typing import Any
from typing import Callable

from .errors import TaskValidationError


class BaseTask(dict, ABC):
    """Abstract base class for all RHAPSODY tasks.

    Inherits from dict to provide unified dict/object interface. Tasks can be
    accessed both as objects (task.uid) and as dicts (task['uid']), eliminating
    conversion overhead between API and backends.

    Attributes:
        uid: Unique identifier for the task (auto-generated if not provided)
        backend: Backend to use for task execution (optional)
    """

    # Global UID counter (thread-safe)
    _uid_counter = 0
    _uid_lock = threading.Lock()

    # Reserved keys that should not appear in dict iteration
    _RESERVED_ATTRS = {"_uid_counter", "_uid_lock", "_RESERVED_ATTRS", "_INTERNAL_ATTRS"}

    # Internal attributes stored on object, not in dict
    _INTERNAL_ATTRS = {"_future"}

    @classmethod
    def _generate_uid(cls) -> str:
        """Generate a unique task ID.

        Uses a thread-safe global counter to ensure unique IDs across all tasks.

        Returns:
            Unique task identifier string
        """
        with BaseTask._uid_lock:
            BaseTask._uid_counter += 1
            return f"task.{BaseTask._uid_counter:06d}"

    def __init__(
        self,
        uid: str | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ):
        """Initialize base task with common fields.

        Args:
            uid: Unique identifier for the task (auto-generated if not provided)
            backend: Backend to use for task execution (optional)
            **kwargs: Additional custom fields

        Raises:
            TaskValidationError: If uid is empty
        """
        # Initialize as dict first
        super().__init__()

        # Set all fields in dict (always initialize to None for consistent access)
        self["uid"] = uid if uid is not None else self._generate_uid()
        self["backend"] = backend

        # Initialize result fields (populated by backends after execution)
        self["state"] = None
        self["stdout"] = None
        self["stderr"] = None
        self["exit_code"] = None
        self["return_value"] = None  # Used by ComputeTask for function/executable output
        self["response"] = None  # Used by AITask for model responses
        self["exception"] = None

        # Add any extra fields
        self.update(kwargs)

        # Initialize internal future
        self._future: asyncio.Future | None = None

        # Validate base fields
        self._validate_base()

        # Validate task-specific fields (implemented by subclasses)
        self._validate_specific()

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to dict keys.

        This enables task.uid instead of requiring task['uid'].

        Args:
            name: Attribute name

        Returns:
            Value from dict

        Raises:
            AttributeError: If key not found in dict
        """
        # Avoid infinite recursion for special attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Setting attributes updates the dict.

        This enables task.state = 'DONE' to update the dict in-place.

        Args:
            name: Attribute name
            value: Value to set
        """
        # Internal/class attributes go to object, not dict
        if name in self._RESERVED_ATTRS or name in self._INTERNAL_ATTRS:
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def _validate_base(self) -> None:
        """Validate common base fields.

        Raises:
            TaskValidationError: If validation fails
        """
        uid = self.get("uid")
        if not uid or not isinstance(uid, str):
            raise TaskValidationError(None, "uid must be a non-empty string")

    @abstractmethod
    def _validate_specific(self) -> None:
        """Validate task-specific fields.

        Must be implemented by subclasses to validate their specific required fields.

        Raises:
            TaskValidationError: If validation fails
        """
        pass

    def validate(self) -> None:
        """Re-validate all task fields.

        Useful for validating after modifying task fields directly.

        Raises:
            TaskValidationError: If validation fails
        """
        self._validate_base()
        self._validate_specific()

    def bind_future(self, future: asyncio.Future) -> None:
        """Bind an asyncio.Future to the task for lifecycle management.

        Args:
            future: The future to bind to this task.
        """
        self._future = future

    def __await__(self) -> Generator[Any, None, Any]:
        """Allow the task object to be awaited directly.

        Delegates to the internal future.

        Returns:
            Generator[Any, None, Any]: An iterator for the task completion.

        Raises:
            RuntimeError: If future is not bound (task not submitted).
        """
        if self._future is None:
            raise RuntimeError(
                f"Task {self.uid} has no bound future. Has it been submitted to a Session?"
            )
        return self._future.__await__()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-serializable fields from pickling state."""
        state = self.__dict__.copy()
        # Remove future as it cannot be pickled (and belongs to submitting process)
        if "_future" in state:
            del state["_future"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Always re-initialize future to None in the new process
        self._future = None

    def __hash__(self) -> int:
        """Hash based on unique identifier.

        Allows tasks to be used in sets, as dictionary keys, and in asyncio.gather.
        """
        return hash(self.uid)

    def __eq__(self, other: Any) -> bool:
        """Equality based on unique identifier."""
        if not isinstance(other, BaseTask):
            return False
        return self.uid == other.uid

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseTask:
        """Create task from dictionary.

        Args:
            data: Dictionary containing task fields

        Returns:
            Task instance of appropriate type

        Raises:
            TaskValidationError: If required fields are missing or invalid
        """
        # Determine task type based on fields present.  Test for ``is not
        # None`` (not truthiness) so a present-but-falsy value — e.g. an empty
        # prompt list ``[]`` or empty string — still routes to the right class
        # instead of being treated as absent.
        if data.get("prompt") is not None:
            return AITask(**data)
        elif data.get("executable") is not None or data.get("function") is not None:
            return ComputeTask(**data)
        else:
            raise TaskValidationError(
                data.get("uid"),
                "Cannot determine task type: must have 'prompt' (AITask) or 'executable'/'function' (ComputeTask)",
            )

    def __repr__(self) -> str:
        """String representation of task showing all fields."""
        items = [f"{key}={repr(value)}" for key, value in self.items()]
        items_str = ", ".join(items)
        return f"{self.__class__.__name__}({items_str})"


class ComputeTask(BaseTask):
    """Task for HPC computational workloads.

    Supports both executable-based tasks (shell commands, binaries) and
    function-based tasks (Python callables).

    Result Access:
        ComputeTask results are accessed via task.return_value (function return value
        or executable output). For AITask, use task.response instead.

    Note:
        Either executable OR function must be specified, but not both.

    Attributes (accessible as task.attr or task['attr']):
        executable: Path to executable for executable tasks
        function: Python callable for function tasks
        arguments: Command-line arguments for executable tasks
        args: Positional arguments for function tasks
        kwargs: Keyword arguments for function tasks
        input_files: List of input file paths
        output_files: List of output file paths
        return_value: Task result (populated after execution)

    Note:
        Execution context (working directory, environment variables, shell mode, resource
        requirements) is backend-specific and must be passed via ``task_backend_specific_kwargs``.
        See the Resource Specification Guide for per-backend details.
    """

    def __init__(
        self,
        executable: str | None = None,
        function: Callable | None = None,
        arguments: list[str] | None = None,
        args: tuple | None = None,
        kwargs: dict | None = None,
        uid: str | None = None,
        input_files: list[str] | None = None,
        output_files: list[str] | None = None,
        capture_stdio: bool = False,
        **extra_kwargs: Any,
    ):
        """Initialize compute task.

        Args:
            executable: Path to executable (required if function not specified)
            function: Python callable (required if executable not specified)
            arguments: Command-line arguments for executable
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            uid: Unique identifier for the task (auto-generated if not provided)
            input_files: List of input file paths
            output_files: List of output file paths
            capture_stdio: If True, redirect stdout/stderr to files under the session
                work directory instead of returning decoded strings. Applies to
                executable tasks only; ignored for function tasks.
            **extra_kwargs: Additional custom fields (e.g. task_backend_specific_kwargs)

        Raises:
            TaskValidationError: If validation fails
        """
        # Store task-specific fields in dict (initialize all to None for consistent access)
        task_fields = {
            "executable": executable,
            "function": function,
            "arguments": arguments if arguments is not None else [],
            "args": args if args is not None else (),
            "kwargs": kwargs if kwargs is not None else {},
            "input_files": input_files,
            "output_files": output_files,
            "capture_stdio": capture_stdio,
        }

        # Initialize base with all fields
        super().__init__(
            uid=uid,
            **task_fields,
            **extra_kwargs,
        )

    def _validate_specific(self) -> None:
        """Validate compute task specific fields.

        Raises:
            TaskValidationError: If validation fails
        """
        uid = self.get("uid")
        executable = self.get("executable")
        function = self.get("function")

        # Exactly one of executable or function must be specified
        if executable is None and function is None:
            raise TaskValidationError(
                uid, "ComputeTask requires either 'executable' or 'function' to be specified"
            )

        if executable is not None and function is not None:
            raise TaskValidationError(
                uid, "ComputeTask cannot have both 'executable' and 'function' specified"
            )

        # Validate executable
        if executable is not None:
            if not isinstance(executable, str):
                raise TaskValidationError(
                    uid, f"executable must be a string, got {type(executable)}"
                )

            arguments = self.get("arguments")
            if arguments is not None and not isinstance(arguments, list):
                raise TaskValidationError(uid, f"arguments must be a list, got {type(arguments)}")

        # Validate function
        if function is not None:
            if not callable(function):
                raise TaskValidationError(uid, f"function must be callable, got {type(function)}")

            func_args = self.get("args")
            if func_args is not None and not isinstance(func_args, (tuple, list)):
                raise TaskValidationError(
                    uid, f"args must be a tuple or list, got {type(func_args)}"
                )

            func_kwargs = self.get("kwargs")
            if func_kwargs is not None and not isinstance(func_kwargs, dict):
                raise TaskValidationError(uid, f"kwargs must be a dict, got {type(func_kwargs)}")

    def __repr__(self) -> str:
        """String representation of compute task showing all fields."""
        items = [f"{key}={repr(value)}" for key, value in self.items()]
        items_str = ", ".join(items)
        return f"ComputeTask({items_str})"


class AITask(BaseTask):
    """Task for AI inference workloads.

    Supports various AI inference scenarios with configurable parameters
    for model selection, prompts, and inference settings.

    Result Access:
        Unlike ComputeTask which uses task.return_value, AITask results are accessed
        via task.response for semantic clarity (AI model responses).

    Attributes (accessible as task.attr or task['attr']):
        prompt: Input prompt for AI model (required)
        model: Model identifier or name (required if inference_endpoint not set)
        inference_endpoint: Alternative to model, direct endpoint URL
        system_prompt: System-level instructions for the model
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: List of sequences that stop generation
        response: AI model response (populated after execution)

    Note:
        Resource requirements (GPU count, CPU threads, memory, ranks) are backend-specific
        and must be specified via ``task_backend_specific_kwargs``. See each backend's
        documentation for supported resource keys.
    """

    def __init__(
        self,
        prompt: str,
        model: str | None = None,
        inference_endpoint: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        uid: str | None = None,
        backend: str | None = None,
        **extra_kwargs: Any,
    ):
        """Initialize AI task.

        Args:
            prompt: Input prompt for the model (required)
            model: Model identifier (required if inference_endpoint not set)
            inference_endpoint: Direct endpoint URL (alternative to model)
            system_prompt: System-level instructions
            temperature: Sampling temperature (typically 0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter
            stop_sequences: Sequences that stop generation
            uid: Unique identifier for the task (auto-generated if not provided)
            backend: Backend to use for task execution (optional)
            **extra_kwargs: Additional custom fields

        Raises:
            TaskValidationError: If validation fails
        """
        # Store AI-specific fields in dict (initialize all to None for consistent access)
        task_fields = {
            "prompt": prompt,
            "model": model,
            "inference_endpoint": inference_endpoint,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stop_sequences": stop_sequences,
        }

        # Initialize base with all fields
        super().__init__(
            uid=uid,
            backend=backend,
            **task_fields,
            **extra_kwargs,
        )

    def _validate_specific(self) -> None:
        """Validate AI task specific fields.

        Raises:
            TaskValidationError: If validation fails
        """
        uid = self.get("uid")
        prompt = self.get("prompt")

        # Prompt is required
        if not prompt:
            raise TaskValidationError(uid, "prompt is required")

        if not isinstance(prompt, (str, list)):
            raise TaskValidationError(
                uid, f"prompt must be a string or list of strings, got {type(prompt)}"
            )

        if isinstance(prompt, list):
            for i, p in enumerate(prompt):
                if not isinstance(p, str):
                    raise TaskValidationError(
                        uid, f"prompt item {i} must be a string, got {type(p)}"
                    )

        model = self.get("model")
        inference_endpoint = self.get("inference_endpoint")

        # Ensure a target is provided (model, endpoint, or named backend)
        if not any([self.get("model"), self.get("inference_endpoint"), self.get("backend")]):
            raise TaskValidationError(
                self["uid"],
                "AITask requires either 'model', 'inference_endpoint', or 'backend' to be specified",
            )

        # Validate model
        if model is not None and not isinstance(model, str):
            raise TaskValidationError(uid, f"model must be a string, got {type(model)}")

        # Validate inference_endpoint
        if inference_endpoint is not None and not isinstance(inference_endpoint, str):
            raise TaskValidationError(
                uid, f"inference_endpoint must be a string, got {type(inference_endpoint)}"
            )

        # Validate optional parameters
        temperature = self.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)) or temperature < 0:
                raise TaskValidationError(
                    uid, f"temperature must be a non-negative number, got {temperature}"
                )

        max_tokens = self.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1:
                raise TaskValidationError(
                    uid, f"max_tokens must be a positive integer, got {max_tokens}"
                )

        top_p = self.get("top_p")
        if top_p is not None:
            if not isinstance(top_p, (int, float)) or not (0 <= top_p <= 1):
                raise TaskValidationError(uid, f"top_p must be between 0 and 1, got {top_p}")

        top_k = self.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1:
                raise TaskValidationError(uid, f"top_k must be a positive integer, got {top_k}")

        stop_sequences = self.get("stop_sequences")
        if stop_sequences is not None and not isinstance(stop_sequences, list):
            raise TaskValidationError(
                uid, f"stop_sequences must be a list, got {type(stop_sequences)}"
            )

    def __repr__(self) -> str:
        """String representation of AI task showing all fields."""
        items = [f"{key}={repr(value)}" for key, value in self.items()]
        items_str = ", ".join(items)
        return f"AITask({items_str})"
