"""Session wrapper with OpenFOAM-friendly default PMI injection."""

from __future__ import annotations

from typing import Any

from rhapsody.api import Session

from dragon.infrastructure.facts import PMIBackend


class OFSession(Session):
    """Session wrapper that can inject a default Dragon PMI backend.

    Default PMI is injected only for parallel compute tasks and only when
    a task does not already provide ``task_backend_specific_kwargs['pmi']``.
    """

    def __init__(self, *args, default_pmi: PMIBackend | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_pmi: PMIBackend | None = None
        if default_pmi is not None:
            self.set_default_pmi(default_pmi)

    def set_default_pmi(self, pmi: PMIBackend) -> OFSession:
        """Set Dragon PMI backend to auto-apply on parallel compute tasks.

        Only ``dragon.infrastructure.facts.PMIBackend`` enum values are accepted.
        """

        if not isinstance(pmi, PMIBackend):
            allowed = ", ".join(member.name for member in PMIBackend)
            raise TypeError(
                "set_default_pmi expects a dragon.infrastructure.facts.PMIBackend "
                f"enum value. Allowed values: {allowed}."
            )

        self._default_pmi = pmi
        return self

    @staticmethod
    def _is_parallel_task(task: dict) -> bool:
        """Return true when the task is configured as parallel/MPI."""

        backend_kwargs = task.get("task_backend_specific_kwargs") or {}

        templates = backend_kwargs.get("process_templates")
        if isinstance(templates, list):
            for template in templates:
                if not isinstance(template, (tuple, list)) or not template:
                    continue
                try:
                    if int(template[0]) > 1:
                        return True
                except (TypeError, ValueError):
                    continue

        if backend_kwargs.get("type") == "mpi":
            try:
                return int(backend_kwargs.get("ranks", 1)) > 1
            except (TypeError, ValueError):
                return True

        arguments = task.get("arguments") or []
        return "-parallel" in arguments or "--parallel" in arguments

    def _inject_default_pmi(self, tasks: list[dict]) -> None:
        """Inject default PMI into eligible tasks in-place."""

        if self._default_pmi is None:
            return

        for task in tasks:
            if not (task.get("executable") or task.get("function")):
                continue

            if not self._is_parallel_task(task):
                continue

            backend_kwargs = dict(task.get("task_backend_specific_kwargs") or {})
            if "pmi" not in backend_kwargs:
                backend_kwargs["pmi"] = self._default_pmi
                task["task_backend_specific_kwargs"] = backend_kwargs

    async def submit_tasks(self, tasks: list[dict]) -> list[Any]:
        """Submit tasks after applying default PMI injection when eligible."""

        self._inject_default_pmi(tasks)
        return await super().submit_tasks(tasks)

    async def submit(self, tasks: list[dict]) -> list[Any]:
        """Alias for ``submit_tasks`` to support submit-style call sites."""

        return await self.submit_tasks(tasks)
