"""Pull API for RHAPSODY telemetry data."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from rhapsody.telemetry.manager import TelemetryManager


class TelemetryReader:
    """Pull interface for metrics and traces.

    Usage::

        reader = TelemetryReader(manager)
        metrics = reader.read_metrics()   # OTel MetricsData
        spans   = reader.read_traces()    # tuple[ReadableSpan, ...]
    """

    def __init__(self, manager: TelemetryManager) -> None:
        self._manager = manager

    def read_metrics(self) -> Any:
        """Return current OTel MetricsData snapshot.

        Navigate via::

            for rm in metrics.resource_metrics:
                for sm in rm.scope_metrics:
                    for metric in sm.metrics:
                        for dp in metric.data.data_points:
                            print(metric.name, dp.value)
        """
        return self._manager.read_metrics()

    def read_traces(
        self,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> tuple:
        """Return finished OTel ReadableSpan objects, optionally filtered.

        Args:
            session_id: If set, return only spans whose session_id attribute matches.
            task_id:    If set, return only spans whose task_id attribute matches.

        Returns:
            Tuple of ReadableSpan objects.
        """
        spans = self._manager.read_traces()
        if session_id is not None:
            spans = tuple(s for s in spans if s.attributes.get("session_id") == session_id)
        if task_id is not None:
            spans = tuple(s for s in spans if s.attributes.get("task_id") == task_id)
        return spans
