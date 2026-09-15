"""Shared pytest fixtures for Rhapsody tests.

This module provides common fixtures used across unit and integration tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

import rhapsody


@pytest.fixture
def session_instance():
    """Fixture providing a Session instance."""
    from rhapsody.api import Session

    return Session()


@pytest.fixture
def base_execution_backend_class():
    """Fixture providing the BaseBackend class."""
    from rhapsody.backends.base import BaseBackend

    return BaseBackend


@pytest.fixture
def tasks_main_states():
    """Fixture providing the TasksMainStates enum."""
    from rhapsody.backends.constants import TasksMainStates

    return TasksMainStates


@pytest.fixture
def state_mapper_class():
    """Fixture providing the StateMapper class."""
    from rhapsody.backends.constants import StateMapper

    return StateMapper


@pytest.fixture
def backend_registry():
    """Fixture providing the BackendRegistry."""
    return rhapsody.BackendRegistry


@pytest.fixture
def available_backends():
    """Fixture providing available backends information."""
    return rhapsody.discover_backends()


@pytest.fixture
def temp_working_directory():
    """Fixture providing a temporary working directory.

    Changes to the temp directory for the test duration and restores the original working directory
    afterward.
    """
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        yield Path(temp_dir)
        os.chdir(original_cwd)


class MockCallback:
    """Mock callback class for testing task state changes."""

    def __init__(self):
        self.calls = []

    def __call__(self, task: dict[str, Any], state: str) -> None:
        """Mock callback that stores calls for verification."""
        self.calls.append((task, state))

    def reset(self):
        """Reset the call history."""
        self.calls = []


@pytest.fixture
def mock_task_callback():
    """Fixture providing a mock task callback function."""
    return MockCallback()


@pytest.fixture
def sample_task():
    """Fixture providing a sample task dictionary."""
    return {
        "uid": "test_task_001",
        "executable": "echo",
        "arguments": ["Hello, World!"],
        "stage_on_error": False,
        "restartable": False,
        "tags": {"test": "fixture"},
        "metadata": {"source": "pytest_fixture"},
    }


@pytest.fixture
def sample_tasks(sample_task):
    """Fixture providing a list of sample tasks."""
    tasks = []
    for i in range(3):
        task = sample_task.copy()
        task["uid"] = f"test_task_{i:03d}"
        task["arguments"] = [f"Task {i}"]
        tasks.append(task)
    return tasks


@pytest.fixture(scope="session")
def backend_availability():
    """Session-scoped fixture providing backend availability status.

    This is cached for the entire test session since backend availability doesn't change during test
    execution.
    """
    return rhapsody.discover_backends()


@pytest.fixture
def skip_if_no_backends(backend_availability):
    """Fixture that skips test if no backends are available."""
    available = [name for name, avail in backend_availability.items() if avail]
    if not available:
        pytest.skip("No backends available for testing")
    return available


@pytest.fixture
def skip_if_no_dask(backend_availability):
    """Fixture that skips test if Dask backend is not available."""
    if not backend_availability.get("dask", False):
        pytest.skip("Dask backend not available")


@pytest.fixture
def skip_if_no_radical_pilot(backend_availability):
    """Fixture that skips test if RADICAL-Pilot backend is not available."""
    if not backend_availability.get("radical_pilot", False):
        pytest.skip("RADICAL-Pilot backend not available")


@pytest.fixture
async def test_backend(skip_if_no_backends, mock_task_callback):
    """Fixture providing a configured backend for testing.

    Automatically selects the first available backend, initializes it, and registers a callback.
    Handles proper cleanup.
    """
    backend_name = skip_if_no_backends[0]

    try:
        # Initialize backend
        if backend_name == "radical_pilot":
            backend = rhapsody.get_backend(backend_name, resources={})
        else:
            backend = rhapsody.get_backend(backend_name)

        # Handle async initialization
        if hasattr(backend, "__await__"):
            backend = await backend  # type: ignore[misc]

        # Register callback
        backend.register_callback(mock_task_callback)

        yield backend

        # Cleanup
        if hasattr(backend, "shutdown"):
            await backend.shutdown()

    except ImportError:
        pytest.skip(f"Backend '{backend_name}' not available")
