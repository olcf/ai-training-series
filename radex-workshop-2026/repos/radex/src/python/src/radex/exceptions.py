"""Python mirror of the C++ exception hierarchy in ``radex/exceptions.hpp``.
"""

__all__ = [
    "RadexError",
    "KeyNotFoundError",
    "TimeoutError",
    "TypeMismatchError",
    "RankMismatchError",
    "DTypeMismatchError",
    "MetadataError",
    "BackendUnavailableError",
]


class RadexError(RuntimeError):
    """Base class for every error raised by radex."""


class KeyNotFoundError(RadexError):
    """A key was requested that is not present in the store."""


class TimeoutError(RadexError):
    """A key did not appear in the store before the timeout elapsed."""


class TypeMismatchError(RadexError):
    """The stored item does not match the description it was requested with."""


class RankMismatchError(TypeMismatchError):
    """A scalar was requested at a tensor key, or a tensor at a scalar key."""


class DTypeMismatchError(TypeMismatchError):
    """The stored element type differs from the requested one."""


class MetadataError(RadexError):
    """An item's metadata record could not be decoded."""


class BackendUnavailableError(RadexError):
    """A client was requested for a backend that was disabled at build time."""
