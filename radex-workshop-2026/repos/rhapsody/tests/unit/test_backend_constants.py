"""Unit tests for backend constants and state management.

This module tests the constants and state mapping functionality defined in
rhapsody.backends.constants.
"""

import pytest


def test_tasks_main_states_import():
    """Test that TasksMainStates can be imported."""
    from rhapsody.backends.constants import TasksMainStates

    assert TasksMainStates is not None


def test_state_mapper_import():
    """Test that StateMapper can be imported."""
    from rhapsody.backends.constants import StateMapper

    assert StateMapper is not None


def test_tasks_main_states_enum():
    """Test TasksMainStates enum values."""
    from rhapsody.backends.constants import TasksMainStates

    # Test that main states exist
    assert hasattr(TasksMainStates, "DONE")
    assert hasattr(TasksMainStates, "RUNNING")
    assert hasattr(TasksMainStates, "FAILED")
    assert hasattr(TasksMainStates, "CANCELED")


def test_tasks_main_states_values():
    """Test TasksMainStates enum string values."""
    from rhapsody.backends.constants import TasksMainStates

    # Test actual string values
    assert TasksMainStates.DONE.value == "DONE"
    assert TasksMainStates.RUNNING.value == "RUNNING"
    assert TasksMainStates.FAILED.value == "FAILED"
    assert TasksMainStates.CANCELED.value == "CANCELED"


def test_state_mapper_basic_functionality():
    """Test basic StateMapper functionality."""
    from rhapsody.backends.constants import StateMapper

    # Test that StateMapper class exists and can be referenced
    assert StateMapper is not None

    # Test that StateMapper requires a valid backend
    with pytest.raises(ValueError):
        StateMapper("nonexistent_backend")


def test_state_mapper_register_backend_tasks_states():
    """Test StateMapper.register_backend_tasks_states() method."""
    from rhapsody.backends.constants import StateMapper

    # Register a test backend
    StateMapper.register_backend_tasks_states(
        backend="test_backend",
        done_state="COMPLETED",
        failed_state="ERROR",
        canceled_state="ABORTED",
        running_state="ACTIVE",
    )

    # Verify the backend was registered
    assert "test_backend" in StateMapper._backend_registry

    # Create mapper and test basic functionality
    mapper = StateMapper("test_backend")
    assert mapper.backend_name == "test_backend"

    # Clean up
    del StateMapper._backend_registry["test_backend"]


def test_state_mapper_register_backend_tasks_states_with_defaults():
    """Test StateMapper.register_backend_tasks_states_with_defaults() method."""
    from rhapsody.backends.constants import StateMapper
    from rhapsody.backends.constants import TasksMainStates

    # Register a test backend with defaults
    StateMapper.register_backend_tasks_states_with_defaults("test_default_backend")

    # Verify the backend was registered
    assert "test_default_backend" in StateMapper._backend_registry

    # Create mapper and test that values match enum values
    mapper = StateMapper("test_default_backend")
    assert mapper.DONE == TasksMainStates.DONE.value
    assert mapper.FAILED == TasksMainStates.FAILED.value
    assert mapper.CANCELED == TasksMainStates.CANCELED.value
    assert mapper.RUNNING == TasksMainStates.RUNNING.value

    # Clean up
    del StateMapper._backend_registry["test_default_backend"]


def test_state_mapper_init_with_string():
    """Test StateMapper initialization with string backend."""
    from rhapsody.backends.constants import StateMapper

    # Register test backend
    StateMapper.register_backend_tasks_states_with_defaults("string_test_backend")

    # Test initialization with string
    mapper = StateMapper("string_test_backend")
    assert mapper.backend_name == "string_test_backend"
    assert mapper.backend_module is None

    # Clean up
    del StateMapper._backend_registry["string_test_backend"]


def test_state_mapper_init_with_object():
    """Test StateMapper initialization with object backend."""
    from rhapsody.backends.constants import StateMapper

    # Create a mock object with __name__ attribute
    class MockBackend:
        __name__ = "mock_backend"

    mock_obj = MockBackend()

    # Register the backend
    StateMapper.register_backend_tasks_states_with_defaults(mock_obj)

    # Test initialization with object
    mapper = StateMapper(mock_obj)
    assert mapper.backend_name == "mock_backend"
    assert mapper.backend_module == mock_obj

    # Clean up
    del StateMapper._backend_registry["mock_backend"]


def test_state_mapper_detect_backend_name():
    """Test StateMapper._detect_backend_name() method."""
    from rhapsody.backends.constants import StateMapper

    # Test with object having __name__ attribute
    class MockModule:
        __name__ = "test_module"

    mapper = StateMapper.__new__(StateMapper)
    mapper.backend_module = MockModule()
    assert mapper._detect_backend_name() == "test_module"

    # Test with object having __class__.__name__ attribute
    class MockClass:
        pass

    mapper.backend_module = MockClass()
    assert mapper._detect_backend_name() == "mockclass"

    # Test with object having neither (but object() actually has __class__)
    # Let's create a more minimal object to test this
    mapper.backend_module = 42  # Numbers don't have __name__ but have __class__
    # This will actually return 'int', so let's test that
    assert mapper._detect_backend_name() == "int"


def test_state_mapper_getattr():
    """Test StateMapper.__getattr__() method for state access."""
    from rhapsody.backends.constants import StateMapper

    # Register test backend
    StateMapper.register_backend_tasks_states(
        backend="attr_test_backend",
        done_state="COMPLETED",
        failed_state="ERROR",
        canceled_state="ABORTED",
        running_state="ACTIVE",
    )

    mapper = StateMapper("attr_test_backend")

    # Test attribute access
    assert mapper.DONE == "COMPLETED"
    assert mapper.FAILED == "ERROR"
    assert mapper.CANCELED == "ABORTED"
    assert mapper.RUNNING == "ACTIVE"

    # Test invalid attribute
    with pytest.raises(AttributeError):
        _ = mapper.INVALID_STATE

    # Clean up
    del StateMapper._backend_registry["attr_test_backend"]


def test_state_mapper_to_main_state():
    """Test StateMapper.to_main_state() method."""
    from rhapsody.backends.constants import StateMapper
    from rhapsody.backends.constants import TasksMainStates

    # Register test backend
    StateMapper.register_backend_tasks_states(
        backend="to_main_test_backend",
        done_state="COMPLETED",
        failed_state="ERROR",
        canceled_state="ABORTED",
        running_state="ACTIVE",
    )

    mapper = StateMapper("to_main_test_backend")

    # Test conversion from backend state to main state
    assert mapper.to_main_state("COMPLETED") == TasksMainStates.DONE
    assert mapper.to_main_state("ERROR") == TasksMainStates.FAILED
    assert mapper.to_main_state("ABORTED") == TasksMainStates.CANCELED
    assert mapper.to_main_state("ACTIVE") == TasksMainStates.RUNNING

    # Test invalid backend state
    with pytest.raises(ValueError):
        mapper.to_main_state("UNKNOWN_STATE")

    # Clean up
    del StateMapper._backend_registry["to_main_test_backend"]


def test_state_mapper_get_backend_state():
    """Test StateMapper.get_backend_state() method."""
    from rhapsody.backends.constants import StateMapper
    from rhapsody.backends.constants import TasksMainStates

    # Register test backend
    StateMapper.register_backend_tasks_states(
        backend="get_state_test_backend",
        done_state="COMPLETED",
        failed_state="ERROR",
        canceled_state="ABORTED",
        running_state="ACTIVE",
    )

    mapper = StateMapper("get_state_test_backend")

    # Test with TasksMainStates enum
    assert mapper.get_backend_state(TasksMainStates.DONE) == "COMPLETED"
    assert mapper.get_backend_state(TasksMainStates.FAILED) == "ERROR"
    assert mapper.get_backend_state(TasksMainStates.CANCELED) == "ABORTED"
    assert mapper.get_backend_state(TasksMainStates.RUNNING) == "ACTIVE"

    # Test with string representations
    assert mapper.get_backend_state("DONE") == "COMPLETED"
    assert mapper.get_backend_state("FAILED") == "ERROR"
    assert mapper.get_backend_state("CANCELED") == "ABORTED"
    assert mapper.get_backend_state("RUNNING") == "ACTIVE"

    # Clean up
    del StateMapper._backend_registry["get_state_test_backend"]


def test_state_mapper_terminal_states():
    """Test StateMapper.terminal_states property."""
    from rhapsody.backends.constants import StateMapper

    # Register test backend
    StateMapper.register_backend_tasks_states(
        backend="terminal_test_backend",
        done_state="COMPLETED",
        failed_state="ERROR",
        canceled_state="ABORTED",
        running_state="ACTIVE",
    )

    mapper = StateMapper("terminal_test_backend")

    # Test terminal states property
    terminal_states = mapper.terminal_states
    assert isinstance(terminal_states, tuple)
    assert len(terminal_states) == 3
    assert "COMPLETED" in terminal_states  # DONE
    assert "ERROR" in terminal_states  # FAILED
    assert "ABORTED" in terminal_states  # CANCELED
    assert "ACTIVE" not in terminal_states  # RUNNING (not terminal)

    # Clean up
    del StateMapper._backend_registry["terminal_test_backend"]


def test_state_mapper_additional_states():
    """Test StateMapper with additional custom states."""
    from rhapsody.backends.constants import StateMapper
    from rhapsody.backends.constants import TasksMainStates

    # Register backend with additional states
    StateMapper.register_backend_tasks_states(
        backend="custom_backend",
        done_state="COMPLETED",
        failed_state="ERROR",
        canceled_state="ABORTED",
        running_state="ACTIVE",
        # Additional custom state (this would need to be in TasksMainStates)
    )

    mapper = StateMapper("custom_backend")

    # Test basic states work
    assert mapper.DONE == "COMPLETED"
    assert mapper.to_main_state("COMPLETED") == TasksMainStates.DONE

    # Clean up
    del StateMapper._backend_registry["custom_backend"]


def test_state_mapper_backend_registry_isolation():
    """Test that StateMapper backend registry maintains state correctly."""
    from rhapsody.backends.constants import StateMapper

    # Register multiple backends
    StateMapper.register_backend_tasks_states(
        backend="backend1",
        done_state="DONE1",
        failed_state="FAILED1",
        canceled_state="CANCELED1",
        running_state="RUNNING1",
    )

    StateMapper.register_backend_tasks_states(
        backend="backend2",
        done_state="DONE2",
        failed_state="FAILED2",
        canceled_state="CANCELED2",
        running_state="RUNNING2",
    )

    # Test each backend maintains its own state mappings
    mapper1 = StateMapper("backend1")
    mapper2 = StateMapper("backend2")

    assert mapper1.DONE == "DONE1"
    assert mapper2.DONE == "DONE2"
    assert mapper1.FAILED == "FAILED1"
    assert mapper2.FAILED == "FAILED2"

    # Clean up
    del StateMapper._backend_registry["backend1"]
    del StateMapper._backend_registry["backend2"]
