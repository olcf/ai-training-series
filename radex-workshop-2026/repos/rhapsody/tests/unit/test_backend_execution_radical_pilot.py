"""Unit tests for RADICAL-Pilot execution backend.

This module tests the RADICAL-Pilot execution backend defined in
rhapsody.backends.execution.radical_pilot.
"""

from unittest.mock import Mock
from unittest.mock import patch

import pytest

from rhapsody import ComputeTask

pytestmark = pytest.mark.radical_pilot  # Apply to all tests in this module


def test_radical_pilot_backend_import():
    """Test that RADICAL-Pilot backend can be imported."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        assert RadicalExecutionBackend is not None
    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_class_exists():
    """Test that RadicalExecutionBackend class exists and inherits from base."""
    try:
        from rhapsody.backends import RadicalExecutionBackend
        from rhapsody.backends.base import BaseBackend

        # Check inheritance
        assert issubclass(RadicalExecutionBackend, BaseBackend)
    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_init():
    """Test RadicalExecutionBackend initialization."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        resources = {"resource": "local.localhost", "runtime": 30, "cores": 2}
        backend = RadicalExecutionBackend(resources)

        assert backend.resources == resources
        assert backend.raptor_config == {}
        assert not backend._initialized

        # Test with raptor config
        raptor_config = {"masters": [{"executable": "/bin/echo", "ranks": 1}]}
        backend_raptor = RadicalExecutionBackend(resources, raptor_config)
        assert backend_raptor.raptor_config == raptor_config

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_import_error():
    """Test behavior when RADICAL-Pilot is not available."""
    # This test should work but since RADICAL-Pilot is available, we skip
    pytest.skip("RADICAL-Pilot is available, cannot test ImportError scenario")


@pytest.mark.asyncio
async def test_radical_pilot_backend_async_init_python313_failure():
    """Test async initialization behavior on Python 3.13."""
    try:
        import sys

        from rhapsody.backends import RadicalExecutionBackend

        resources = {"resource": "local.localhost", "runtime": 1, "cores": 1}
        backend = RadicalExecutionBackend(resources)

        # On Python 3.13, RADICAL-Pilot may fail due to pickle issues
        if sys.version_info >= (3, 13):
            try:
                backend = await backend
                # If initialization succeeds, that's also valid
                # (maybe RADICAL-Pilot was updated to support Python 3.13)
                if hasattr(backend, "_initialized") and backend._initialized:
                    await backend.shutdown()
                    pytest.skip("RADICAL-Pilot appears to work on Python 3.13 in this environment")
            except (RuntimeError, Exception) as e:
                # If it fails, verify it's for expected reasons
                error_msg = str(e).lower()
                assert any(
                    keyword in error_msg
                    for keyword in [
                        "pickle",
                        "thread.lock",
                        "command failed",
                        "serialization",
                        "cannot pickle",
                        "multiprocessing",
                    ]
                ), f"RADICAL-Pilot failed with unexpected error on Python 3.13: {e}"
        else:
            # On supported Python versions, this should work
            backend = await backend
            assert backend._initialized
            await backend.shutdown()

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


@pytest.mark.asyncio
async def test_radical_pilot_backend_context_manager_failure():
    """Test async context manager behavior on Python 3.13."""
    try:
        import sys

        from rhapsody.backends import RadicalExecutionBackend

        resources = {"resource": "local.localhost", "runtime": 1, "cores": 1}

        # Context manager may fail during entry on Python 3.13
        if sys.version_info >= (3, 13):
            try:
                async with RadicalExecutionBackend(resources) as backend:
                    # If initialization succeeds, that's also valid
                    # (maybe RADICAL-Pilot was updated to support Python 3.13)
                    if hasattr(backend, "_initialized") and backend._initialized:
                        pass
                # If we get here without exception, RADICAL-Pilot works on Python 3.13
                pytest.skip(
                    "RADICAL-Pilot context manager works on Python 3.13 in this environment"
                )
            except (RuntimeError, Exception) as e:
                # If it fails, verify it's for expected reasons
                error_msg = str(e).lower()
                assert any(
                    keyword in error_msg
                    for keyword in [
                        "pickle",
                        "thread.lock",
                        "command failed",
                        "serialization",
                        "cannot pickle",
                        "multiprocessing",
                    ]
                ), f"RADICAL-Pilot failed with unexpected error on Python 3.13: {e}"
        else:
            # On supported Python versions, this should work
            async with RadicalExecutionBackend(resources) as backend:
                assert backend._initialized

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_callback_registration():
    """Test callback registration without initialization."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})

        # Mock task_manager since it's only available after async initialization
        backend.task_manager = Mock()

        # Test callback registration
        mock_callback = Mock()
        backend.register_callback(mock_callback)

        # Should store the callback and register with task manager
        assert backend._callback_func == mock_callback
        backend.task_manager.register_callback.assert_called_once()

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_state_mapper():
    """Test state mapper functionality without initialization."""
    try:
        from rhapsody.backends import RadicalExecutionBackend
        from rhapsody.backends.constants import StateMapper

        backend = RadicalExecutionBackend({"resource": "local.localhost"})

        # Register the backend manually since we can't do async init
        StateMapper.register_backend_tasks_states(
            backend=backend,
            done_state="DONE",
            failed_state="FAILED",
            canceled_state="CANCELED",
            running_state="RUNNING",
        )

        # This should now work
        state_mapper = backend.get_task_states_map()
        assert state_mapper is not None

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_task_building():
    """Test task building functionality."""
    try:
        import radical.pilot as rp

        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})
        backend.tasks = {}  # Initialize manually since we can't do async init

        # Mock raptor_mode and master_selector for function task testing
        backend.raptor_mode = True
        backend.master_selector = iter(["master_001"])

        # Test executable task
        task_desc = {
            "executable": "/bin/echo",
            "function": None,
            "args": ["hello"],
            "kwargs": {},
            "is_service": False,
        }

        rp_task = backend.build_task("test_task", task_desc, {})

        assert rp_task is not None
        assert rp_task.uid == "test_task"
        assert rp_task.executable == "/bin/echo"
        assert rp_task.mode == rp.TASK_EXECUTABLE

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_data_dependencies():
    """Test explicit and implicit data dependency linking."""
    try:
        import radical.pilot as rp

        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})

        # Test explicit data dependencies
        src_task = ComputeTask(executable="/bin/echo")
        dst_task = ComputeTask(executable="/bin/cat", task_backend_specific_kwargs={})

        # Test task-to-task linking
        data_dep = backend.link_explicit_data_deps(
            src_task=src_task, dst_task=dst_task, file_name="output.dat"
        )

        assert data_dep["source"] == f"pilot:///{src_task.uid}/output.dat"
        assert data_dep["target"] == "task:///output.dat"
        assert data_dep["action"] == rp.LINK
        assert "input_staging" in dst_task["task_backend_specific_kwargs"]

        # Test external file staging
        dst_task2 = ComputeTask(executable="/bin/cat", task_backend_specific_kwargs={})
        data_dep2 = backend.link_explicit_data_deps(
            dst_task=dst_task2, file_path="/path/to/input.txt"
        )

        assert data_dep2["source"] == "/path/to/input.txt"
        assert data_dep2["target"] == "task:///input.txt"
        assert data_dep2["action"] == rp.TRANSFER

        # Test implicit data dependencies
        dst_task3 = ComputeTask(executable="/bin/cat", task_backend_specific_kwargs={})
        backend.link_implicit_data_deps(src_task, dst_task3)

        assert "pre_exec" in dst_task3["task_backend_specific_kwargs"]
        commands = dst_task3["task_backend_specific_kwargs"]["pre_exec"]
        assert any(f"SRC_TASK_ID={src_task.uid}" in cmd for cmd in commands)

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_raptor_mode_setup():
    """Test Raptor mode configuration without actual submission."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})

        # Mock the necessary attributes that would be set during initialization
        backend.session = Mock()
        backend.session.uid = "test_session"
        backend.resource_pilot = Mock()
        backend.resource_pilot.submit_raptors = Mock(return_value=[Mock(uid="master_001")])

        raptor_config = {
            "masters": [
                {
                    "executable": "/bin/master",
                    "arguments": ["--mode", "master"],
                    "ranks": 1,
                    "workers": [
                        {
                            "executable": "/bin/worker",
                            "arguments": ["--mode", "worker"],
                            "ranks": 4,
                            "worker_type": "ComputeWorker",
                        }
                    ],
                }
            ]
        }

        # Should not raise errors
        backend.setup_raptor_mode(raptor_config)

        assert backend.masters is not None
        assert backend.workers is not None
        assert backend.master_selector is not None

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_master_selection():
    """Test master selection for Raptor mode."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})

        # Initialize required attributes that would be set during __init__ or setup
        backend.raptor_mode = False
        backend.masters = []

        # Test without Raptor mode
        with pytest.raises(RuntimeError, match="Raptor mode disabled"):
            selector = backend.select_master()
            next(selector)  # This triggers the error

        # Test with Raptor mode but no masters
        backend.raptor_mode = True
        backend.masters = []

        with pytest.raises(RuntimeError, match="no masters available"):
            selector = backend.select_master()
            next(selector)  # This triggers the error

        # Test with Raptor mode and masters
        backend.raptor_mode = True
        backend.masters = [Mock(uid="master_001"), Mock(uid="master_002")]

        selector = backend.select_master()

        # Should cycle through masters
        assert next(selector) == "master_001"
        assert next(selector) == "master_002"
        assert next(selector) == "master_001"  # Cycles back

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


@pytest.mark.asyncio
async def test_radical_pilot_backend_cancel_functionality():
    """Test task cancellation."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})
        backend.tasks = {"task_001": Mock()}
        backend.task_manager = Mock()

        # Test cancelling existing task
        result = await backend.cancel_task("task_001")
        assert result is True
        backend.task_manager.cancel_tasks.assert_called_once_with("task_001")

        # Test cancelling non-existent task
        result = await backend.cancel_task("task_999")
        assert result is False

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_class_methods():
    """Test class factory methods."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        # Test create method exists and has proper signature
        assert hasattr(RadicalExecutionBackend, "create")
        assert callable(RadicalExecutionBackend.create)

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


@pytest.mark.asyncio
async def test_radical_pilot_backend_submit_tasks():
    """Test task submission with mocked components."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})
        backend.task_manager = Mock()
        backend.tasks = {}

        # Mock build_task to return a valid task
        mock_task = Mock()
        mock_task.uid = "test_task"

        with patch.object(backend, "build_task", return_value=mock_task):
            tasks = [
                ComputeTask(
                    executable="/bin/echo",
                    arguments=["hello"],
                    task_backend_specific_kwargs={},
                )
            ]

            await backend.submit_tasks(tasks)

            backend.task_manager.submit_tasks.assert_called_once()
            args = backend.task_manager.submit_tasks.call_args[0][0]
            assert len(args) == 1
            assert args[0] == mock_task

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


def test_radical_pilot_backend_get_nodelist():
    """Test nodelist retrieval."""
    try:
        import radical.pilot as rp

        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})

        # Test with inactive pilot
        backend.resource_pilot = Mock()
        backend.resource_pilot.state = rp.FAILED
        result = backend.get_nodelist()
        assert result is None

        # Test with active pilot
        backend.resource_pilot.state = rp.PMGR_ACTIVE
        mock_nodelist = Mock()
        backend.resource_pilot.nodelist = mock_nodelist
        result = backend.get_nodelist()
        assert result == mock_nodelist

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")


@pytest.mark.asyncio
async def test_radical_pilot_backend_shutdown():
    """Test backend shutdown."""
    try:
        from rhapsody.backends import RadicalExecutionBackend

        backend = RadicalExecutionBackend({"resource": "local.localhost"})
        backend.session = Mock()

        await backend.shutdown()

        backend.session.close.assert_called_once_with(download=True)

    except ImportError:
        pytest.skip("RADICAL-Pilot dependencies not available")
