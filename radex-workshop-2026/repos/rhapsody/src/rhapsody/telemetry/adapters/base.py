"""Base class for RHAPSODY telemetry adapters."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TelemetryAdapter(ABC):
    """Abstract base for backend telemetry adapters.

    Each adapter is responsible for collecting resource metrics from a specific
    execution backend and emitting :class:`~rhapsody.telemetry.events.ResourceUpdate`
    events via ``manager.emit()``.

    Rules:
    - Must emit normalized events only — never create metrics or spans directly.
    - Must include both ``event_time`` and ``emit_time`` on every event.
    - Must be safe to call ``stop()`` even if ``start()`` was a no-op.
    """

    @abstractmethod
    def start(self) -> None:
        """Start resource collection.

        ``self._manager`` is guaranteed to be set before this is called —
        :meth:`~rhapsody.telemetry.manager.TelemetryManager.register_adapter`
        injects the back-reference at registration time.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop resource collection cleanly."""
