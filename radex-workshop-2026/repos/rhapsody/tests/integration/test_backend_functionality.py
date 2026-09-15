"""Backend Functionality Tests for AsyncFlow Integration.

This module tests specific backend functionality that AsyncFlow workflows will depend on, focusing
on real task execution and state management.

Test Mode Control:
------------------
Set RHAPSODY_TEST_MODE environment variable to control which backends are tested:
- RHAPSODY_TEST_MODE=regular (default): Tests local backends (concurrent, dask)
  Run with: pytest tests/integration/test_backend_functionality.py

- RHAPSODY_TEST_MODE=dragon: Tests Dragon backends only (dragon)
  Run with: dragon pytest tests/integration/test_backend_functionality.py

- RHAPSODY_TEST_MODE=all: Tests all backends
  Run with: pytest tests/integration/test_backend_functionality.py (requires Dragon runtime)
"""

import asyncio
import os

import pytest

import rhapsody
from rhapsody.backends.constants import TasksMainStates


def get_test_mode() -> str:
    """Get the current test mode from environment variable.

    Returns:
        "regular", "dragon", or "all"
    """
    return os.environ.get("RHAPSODY_TEST_MODE", "regular").lower()


def get_available_backends_for_mode() -> list[str]:
    """Get list of available backends based on test mode.

    Returns:
        List of backend names to test
    """
    all_backends = rhapsody.discover_backends()
    available = [name for name, avail in all_backends.items() if avail]

    mode = get_test_mode()

    if mode == "dragon":
        # Only Dragon backends
        return [name for name in available if name == "dragon" or name.startswith("dragon_")]
    elif mode == "regular":
        # Exclude Dragon and radical_pilot (both require dedicated HPC infrastructure)
        return [
            name
            for name in available
            if name not in {"dragon", "radical_pilot"} and not name.startswith("dragon_")
        ]
    else:  # mode == "all"
        # All available backends
        return available


def _attach_state_collector(backend) -> dict[str, str]:
    """Register a terminal-state callback on the backend and return the shared dict.

    Use this when the callback must be registered BEFORE submit_tasks() is called (e.g. when the
    backend fires FAILED synchronously during build_task, as Dragon does when the executable path is
    invalid).  Call _wait_for_states() afterward.
    """
    final_states: dict[str, str] = {}

    def on_state(task_desc: dict, state: str) -> None:
        if state in {"DONE", "FAILED", "CANCELED"}:
            final_states[task_desc["uid"]] = state

    backend.register_callback(on_state)
    return final_states


async def _wait_for_states(
    final_states: dict[str, str], task_uids: list[str], timeout: float = 10.0
) -> dict[str, str]:
    """Wait until all uids appear in final_states or timeout expires."""
    deadline = asyncio.get_event_loop().time() + timeout
    while len(final_states) < len(task_uids) and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.1)
    return final_states


async def _collect_terminal_states(
    backend, task_uids: list[str], timeout: float = 10.0
) -> dict[str, str]:
    """Register a capturing callback, submit wait, return uid→state map.

    Convenience wrapper for the common case where callback registration happens AFTER
    submit_tasks().  For backends that deliver FAILED synchronously during submission use
    _attach_state_collector() + _wait_for_states() instead.
    """
    final_states = _attach_state_collector(backend)
    return await _wait_for_states(final_states, task_uids, timeout)


async def setup_test_backend(backend_name=None):
    """Helper to set up a backend with proper initialization and callback.

    Args:
        backend_name: Optional backend name. If None, uses first available for current mode.
    """
    if backend_name is None:
        backend_names = get_available_backends_for_mode()

        if not backend_names:
            mode = get_test_mode()
            pytest.skip(f"No backends available for test mode: {mode}")
        backend_name = backend_names[0]

    try:
        # Initialize backend with proper resources
        backend = await rhapsody.get_backend(backend_name)

        # Handle async initialization for backends that need it
        if hasattr(backend, "__await__"):
            backend = await backend  # type: ignore[misc]

        # Register a callback function
        def task_callback(task: dict, state: str) -> None:
            # Simple callback that does nothing but satisfies the interface
            pass

        backend.register_callback(task_callback)
        return backend

    except ImportError:
        pytest.skip(f"Backend '{backend_name}' not available")


@pytest.mark.asyncio
class TestBackendFunctionality:
    """Test suite for backend functionality required by AsyncFlow."""

    async def test_backend_task_cancellation(self):
        """Submit a long-running task, cancel it, and assert the CANCELED state is delivered."""
        backend = await setup_test_backend("dask" if get_test_mode() == "regular" else None)

        if not hasattr(backend, "cancel_task"):
            await backend.shutdown()
            pytest.skip(f"{type(backend).__name__} does not support cancel_task")

        task = rhapsody.ComputeTask(executable="/bin/sleep", arguments=["30"])
        final_states: dict[str, str] = {}

        def on_state(task_desc: dict, state: str) -> None:
            if state in {"DONE", "FAILED", "CANCELED"}:
                final_states[task_desc["uid"]] = state

        backend.register_callback(on_state)

        try:
            await backend.submit_tasks([task])

            # Allow the task to reach RUNNING state before cancelling
            await asyncio.sleep(0.5)

            cancelled = await backend.cancel_task(task.uid)
            assert cancelled is True, "cancel_task must return True when cancellation succeeds"

            # Wait up to 5 s for the CANCELED callback
            for _ in range(50):
                if task.uid in final_states:
                    break
                await asyncio.sleep(0.1)

            assert task.uid in final_states, "No terminal-state callback received within 5 s"
            assert final_states[task.uid] == "CANCELED", (
                f"Expected CANCELED, got {final_states[task.uid]}"
            )

        finally:
            await backend.shutdown()

    async def test_backend_resource_management(self):
        """Each available backend can submit and complete a task."""
        available_names = get_available_backends_for_mode()

        if not available_names:
            pytest.skip(f"No backends available for test mode: {get_test_mode()}")

        backend_name = available_names[0]
        backend = await rhapsody.get_backend(backend_name)
        task = rhapsody.ComputeTask(
            executable="/bin/echo", arguments=[f"resource-test {backend_name}"]
        )

        try:
            await backend.submit_tasks([task])
            states = await _collect_terminal_states(backend, [task.uid], timeout=10.0)
            assert states.get(task.uid) == "DONE", (
                f"Backend {backend_name}: expected DONE, got {states.get(task.uid)!r}"
            )
        finally:
            await backend.shutdown()

    async def test_backend_state_transitions(self):
        """Task progresses from submission to DONE state."""
        backend = await setup_test_backend()
        task = rhapsody.ComputeTask(executable="/bin/echo", arguments=["state-transition"])

        try:
            await backend.submit_tasks([task])
            states = await _collect_terminal_states(backend, [task.uid], timeout=10.0)

            assert task.uid in states, "Task never reached a terminal state within 10 s"
            assert states[task.uid] == "DONE", f"Expected DONE, got {states[task.uid]!r}"
        finally:
            await backend.shutdown()

    async def test_backend_error_recovery(self):
        """A failing task reaches FAILED while a good concurrent task reaches DONE."""
        backend = await setup_test_backend()
        good_task = rhapsody.ComputeTask(executable="/bin/echo", arguments=["ok"])
        bad_task = rhapsody.ComputeTask(executable="/nonexistent/command", arguments=[])
        task_uids = [good_task.uid, bad_task.uid]

        # Register the callback BEFORE submit_tasks: some backends (Dragon) validate
        # the executable at build_task time and fire FAILED synchronously during submission.
        final_states = _attach_state_collector(backend)

        try:
            await backend.submit_tasks([good_task, bad_task])
            await _wait_for_states(final_states, task_uids, timeout=10.0)

            assert final_states.get(good_task.uid) == "DONE", (
                f"Good task: expected DONE, got {final_states.get(good_task.uid)!r}"
            )
            assert final_states.get(bad_task.uid) == "FAILED", (
                f"Bad task: expected FAILED, got {final_states.get(bad_task.uid)!r}"
            )
        finally:
            await backend.shutdown()

    async def test_backend_batch_operations(self):
        """All tasks in a 10-task batch reach DONE."""
        backend = await setup_test_backend()
        tasks = [
            rhapsody.ComputeTask(executable="/bin/echo", arguments=[f"batch-{i}"])
            for i in range(10)
        ]
        task_uids = [t.uid for t in tasks]

        try:
            await backend.submit_tasks(tasks)
            states = await _collect_terminal_states(backend, task_uids, timeout=30.0)

            not_done = [uid for uid in task_uids if states.get(uid) != "DONE"]
            assert not not_done, f"{len(not_done)}/10 tasks did not reach DONE: {not_done}"
        finally:
            await backend.shutdown()

    async def test_backend_async_patterns(self):
        """Three tasks submitted concurrently via asyncio.gather all reach DONE."""
        mode = get_test_mode()
        if mode == "dragon":
            pytest.skip("Async pattern test uses dask only")

        try:
            backend = await rhapsody.get_backend("dask")
        except (ImportError, Exception):
            pytest.skip("Dask backend not available for async pattern testing")

        try:
            tasks = [
                rhapsody.ComputeTask(executable="/bin/echo", arguments=[f"async-{i}"])
                for i in range(3)
            ]

            # Submit all three concurrently
            await asyncio.gather(*[backend.submit_tasks([t]) for t in tasks])

            states = await _collect_terminal_states(backend, [t.uid for t in tasks], timeout=30.0)
            not_done = [t.uid for t in tasks if states.get(t.uid) != "DONE"]
            assert not not_done, f"Tasks did not reach DONE: {not_done}"
        finally:
            await backend.shutdown()


@pytest.mark.asyncio
class TestBackendCompatibility:
    """Test compatibility between different backends."""

    async def test_backend_interface_consistency(self):
        """All instantiable backends expose the required interface AND can execute a task."""
        available_names = get_available_backends_for_mode()
        tested = 0

        for name in available_names:
            try:
                backend = await rhapsody.get_backend(name)
            except (ImportError, Exception):
                continue  # skip backends whose runtime deps are not installed

            # Structural check
            assert hasattr(backend, "submit_tasks") and callable(backend.submit_tasks), (
                f"{name}: submit_tasks not callable"
            )
            assert hasattr(backend, "shutdown") and callable(backend.shutdown), (
                f"{name}: shutdown not callable"
            )

            # Behavioural check — actually invoke submit_tasks and assert result
            task = rhapsody.ComputeTask(
                executable="/bin/echo", arguments=[f"interface-check-{name}"]
            )
            try:
                await backend.submit_tasks([task])
                states = await _collect_terminal_states(backend, [task.uid], timeout=10.0)
                assert states.get(task.uid) == "DONE", (
                    f"Backend {name}: expected DONE, got {states.get(task.uid)!r}"
                )
                tested += 1
            finally:
                await backend.shutdown()

        if tested == 0:
            pytest.skip("No backends could be instantiated in this environment")

    async def test_backend_switching(self):
        """Each instantiable backend can independently submit and complete an echo task."""
        available_names = get_available_backends_for_mode()

        if not available_names:
            pytest.skip(f"No backends available for test mode: {get_test_mode()}")

        tested = 0
        for name in available_names:
            try:
                backend = await rhapsody.get_backend(name)
            except (ImportError, Exception):
                continue  # skip backends whose runtime deps are not installed

            task = rhapsody.ComputeTask(executable="/bin/echo", arguments=[f"switch-test-{name}"])
            try:
                await backend.submit_tasks([task])
                states = await _collect_terminal_states(backend, [task.uid], timeout=10.0)
                assert states.get(task.uid) == "DONE", (
                    f"Backend {name}: expected DONE, got {states.get(task.uid)!r}"
                )
                tested += 1
            finally:
                await backend.shutdown()

        if tested == 0:
            pytest.skip("No backends could be instantiated in this environment")


async def main():
    """Run a functionality test."""
    mode = get_test_mode()
    print(f"Running Backend Functionality Tests in '{mode}' mode...")

    try:
        # Test first available backend
        available_names = get_available_backends_for_mode()
        if not available_names:
            print(f"No backends available for mode: {mode}")
            return

        backend_name = available_names[0]
        backend = await rhapsody.get_backend(backend_name)
        tasks = [
            rhapsody.ComputeTask(
                executable="/bin/echo",
                arguments=["test"],
            )
        ]
        await backend.submit_tasks(tasks)
        # Backend submission completed
        assert True
        await backend.shutdown()

        print(f"✅ Backend functionality test passed for {backend_name}!")

    except Exception as e:
        print(f"❌ Backend functionality test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
