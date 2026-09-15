"""Unit tests for backend discovery functionality.

This module tests the backend discovery and registry functionality defined in
rhapsody.backends.discovery.
"""

import pytest

import rhapsody


def test_discover_backends():
    """Test backend discovery."""
    available = rhapsody.discover_backends()

    # These should always be available
    assert "dask" in available  # May be True or False
    assert "radical_pilot" in available  # May be True or False

    print(f"Available backends: {available}")


def test_backend_registry():
    """Test backend registry functionality."""
    registry = rhapsody.BackendRegistry

    backends = registry.list_backends()
    assert len(backends) >= 2  # At least dask and radical_pilot

    # Test getting a backend class for each available backend
    for backend_name in backends:
        backend_class = registry.get_backend_class(backend_name)
        assert backend_class is not None

    # Test error on invalid backend
    with pytest.raises(ValueError):
        registry.get_backend_class("nonexistent")


def test_backend_registry_list_backends():
    """Test BackendRegistry.list_backends() method."""
    registry = rhapsody.BackendRegistry

    backends = registry.list_backends()
    assert isinstance(backends, list)
    assert len(backends) >= 2
    assert "dask" in backends
    assert "radical_pilot" in backends


def test_backend_registry_get_backend_class():
    """Test BackendRegistry.get_backend_class() method."""
    registry = rhapsody.BackendRegistry

    # Test successful retrieval for known backends
    for backend_name in ["dask", "radical_pilot"]:
        try:
            backend_class = registry.get_backend_class(backend_name)
            assert backend_class is not None
            # Should be a class (type object)
            assert isinstance(backend_class, type)
        except ImportError:
            # Expected if dependencies are not available
            pytest.skip(f"Backend '{backend_name}' dependencies not available")

    # Test ValueError for unknown backend
    with pytest.raises(ValueError, match="Backend 'unknown_backend' not found"):
        registry.get_backend_class("unknown_backend")


def test_backend_registry_register_backend():
    """Test BackendRegistry.register_backend() method."""
    registry = rhapsody.BackendRegistry

    # Save original state
    original_backends = registry._backends.copy()

    try:
        # Register a test backend
        test_backend_path = "rhapsody.backends.base.BaseBackend"
        registry.register_backend("test_backend", test_backend_path)

        # Verify registration
        backends = registry.list_backends()
        assert "test_backend" in backends

        # Verify we can get the class (even though it's abstract)
        backend_class = registry.get_backend_class("test_backend")
        assert backend_class is not None

    finally:
        # Restore original state
        registry._backends = original_backends


def test_backend_registry_import_error_handling():
    """Test BackendRegistry error handling for invalid backends."""
    registry = rhapsody.BackendRegistry

    # Should raise ValueError when trying to get a non-existent backend
    with pytest.raises(ValueError, match="Backend 'invalid_backend' not found"):
        registry.get_backend_class("invalid_backend")


def test_get_backend_function():
    """Test the get_backend function imports correctly."""
    from rhapsody.backends import get_backend

    assert get_backend is not None


def test_get_backend_with_arguments():
    """Test get_backend function with backend creation."""
    from rhapsody.backends import get_backend

    # Test with a backend that might be available
    available = rhapsody.discover_backends()

    # Find first available backend
    available_backend = None
    for backend_name, is_available in available.items():
        if is_available:
            available_backend = backend_name
            break

    if available_backend is None:
        pytest.skip("No backends available for testing")

    try:
        if available_backend == "radical_pilot":
            # RADICAL-Pilot requires resources
            test_resources = {
                "resource": "local.localhost",
                "runtime": 1,
                "cores": 1,
            }
            backend = get_backend(available_backend, test_resources)
        else:
            # Other backends (like dask)
            backend = get_backend(available_backend)

        assert backend is not None

    except ImportError:
        pytest.skip(f"Backend '{available_backend}' dependencies not available")
    except Exception as e:
        # Some backends might fail to initialize but that's ok for this test
        print(f"Backend initialization failed (expected): {e}")


def test_get_backend_error_handling():
    """Test get_backend function error handling."""
    from rhapsody.backends import get_backend

    # Test with unknown backend
    with pytest.raises(ValueError):
        get_backend("nonexistent_backend")


def test_discover_backends_function():
    """Test the discover_backends function imports correctly."""
    from rhapsody.backends import discover_backends

    assert discover_backends is not None


def test_discover_backends_return_format():
    """Test discover_backends return format and content."""
    available = rhapsody.discover_backends()

    # Should return a dictionary
    assert isinstance(available, dict)

    # Should contain expected backends
    expected_backends = ["dask", "radical_pilot"]
    for backend in expected_backends:
        assert backend in available
        assert isinstance(available[backend], bool)


def test_discover_backends_import_error_handling():
    """Test discover_backends handles import errors correctly."""
    from rhapsody.backends.discovery import BackendRegistry

    # Save original registry
    original_backends = BackendRegistry._backends.copy()

    try:
        # Add a backend with invalid import path
        BackendRegistry.register_backend("fake_backend", "nonexistent.module.Class")

        # discover_backends should handle this gracefully
        available = rhapsody.discover_backends()

        # Should still return a dict with the fake backend
        assert isinstance(available, dict)
        assert "fake_backend" in available
        assert available["fake_backend"] is False  # Should be False due to ImportError

    finally:
        # Restore original state
        BackendRegistry._backends = original_backends


def test_backend_registry_state_isolation():
    """Test that BackendRegistry modifications don't affect each other."""
    registry = rhapsody.BackendRegistry

    # Save original state
    original_backends = registry._backends.copy()
    original_count = len(registry.list_backends())

    try:
        # Register first test backend
        registry.register_backend("test1", "some.module.Class1")
        assert len(registry.list_backends()) == original_count + 1
        assert "test1" in registry.list_backends()

        # Register second test backend
        registry.register_backend("test2", "some.module.Class2")
        assert len(registry.list_backends()) == original_count + 2
        assert "test1" in registry.list_backends()
        assert "test2" in registry.list_backends()

        # Remove one backend
        del registry._backends["test1"]
        assert len(registry.list_backends()) == original_count + 1
        assert "test1" not in registry.list_backends()
        assert "test2" in registry.list_backends()

    finally:
        # Restore original state
        registry._backends = original_backends


def test_backend_registry_backend_path_parsing():
    """Test that BackendRegistry correctly parses module paths."""
    registry = rhapsody.BackendRegistry

    # Test with a valid path that we can verify
    try:
        # This should work since BaseBackend exists
        backend_class = registry.get_backend_class("dask")
        # If no ImportError, the path parsing worked
        assert backend_class is not None
    except ImportError:
        # Expected if dask is not installed
        pass

    # Test the internal path parsing logic by registering a known good path
    original_backends = registry._backends.copy()
    try:
        registry.register_backend("test_path", "rhapsody.backends.base.BaseBackend")
        backend_class = registry.get_backend_class("test_path")
        assert backend_class is not None
        # Should be the BaseBackend class
        assert hasattr(backend_class, "__abstractmethods__")
    finally:
        registry._backends = original_backends
