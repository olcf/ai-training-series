"""Unit tests for base backend interface.

This module tests the base backend abstract class and interface defined in rhapsody.backends.base.
"""

import inspect
import os

import pytest

from rhapsody import ComputeTask


def test_base_execution_backend_import():
    """Test that BaseBackend can be imported."""
    from rhapsody.backends.base import BaseBackend

    assert BaseBackend is not None


def test_base_execution_backend_is_abstract():
    """Test that BaseBackend is abstract and cannot be instantiated."""
    from rhapsody.backends.base import BaseBackend

    # Should raise TypeError when trying to instantiate abstract class
    with pytest.raises(TypeError):
        BaseBackend()


def test_base_execution_backend_abstract_methods():
    """Test that BaseBackend has all required abstract methods."""
    from rhapsody.backends.base import BaseBackend

    # Get all abstract methods
    abstract_methods = BaseBackend.__abstractmethods__

    # Expected abstract methods based on the source code
    expected_methods = {
        "submit_tasks",
        "shutdown",
        "state",
        "task_state_cb",
        "get_task_states_map",
        "build_task",
        "cancel_task",
    }

    assert abstract_methods == expected_methods


def test_base_execution_backend_method_signatures():
    """Test that BaseBackend abstract methods have correct signatures."""
    from rhapsody.backends.base import BaseBackend

    # Check submit_tasks signature
    submit_tasks_sig = inspect.signature(BaseBackend.submit_tasks)
    assert len(submit_tasks_sig.parameters) == 2  # self, tasks
    assert "tasks" in submit_tasks_sig.parameters

    # Check shutdown signature
    shutdown_sig = inspect.signature(BaseBackend.shutdown)
    assert len(shutdown_sig.parameters) == 1  # self only

    # Check state signature
    state_sig = inspect.signature(BaseBackend.state)
    assert len(state_sig.parameters) == 1  # self only

    # Check cancel_task signature
    cancel_task_sig = inspect.signature(BaseBackend.cancel_task)
    assert len(cancel_task_sig.parameters) == 2  # self, uid
    assert "uid" in cancel_task_sig.parameters
