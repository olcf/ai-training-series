"""Push (subscribe) API for RHAPSODY telemetry events."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from rhapsody.telemetry.events import BaseEvent

if TYPE_CHECKING:
    from rhapsody.telemetry.manager import TelemetryManager


class TelemetrySubscriber:
    """Push interface for receiving RHAPSODY telemetry events.

    Callbacks receive typed :class:`BaseEvent` subclass instances — not OTel
    primitives — so subscribers always work with RHAPSODY semantics.

    Usage::

        sub = TelemetrySubscriber(manager)
        sub.subscribe(lambda e: print(e.event_type, e.task_id))

        # Async callbacks are supported:
        async def handler(event):
            await my_queue.put(event)
        sub.subscribe(handler)
    """

    def __init__(self, manager: TelemetryManager) -> None:
        self._manager = manager

    def subscribe(self, callback: Callable[[BaseEvent], Any]) -> None:
        """Register a callback to receive every telemetry event.

        Args:
            callback: Sync or async callable accepting a single BaseEvent argument.
                      Exceptions inside the callback are caught and logged — they
                      never propagate to the dispatch loop.
        """
        self._manager.subscribe(callback)
