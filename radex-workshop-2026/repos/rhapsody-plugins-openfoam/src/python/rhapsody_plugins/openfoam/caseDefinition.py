"""Case-level staged execution helpers for OpenFOAM workflows."""

from __future__ import annotations

import shutil
from collections import OrderedDict
from dataclasses import dataclass, field, InitVar
from pathlib import Path

from .ofTask import OFTask, OFStage


@dataclass
class CaseDefinition:
    """Holds all the runtime information for defining where and how an OpenFOAM case should be run

    Attributes:
        source_directory: Path to the case to be run
        run_directory: Path where the source case should be copied to and run
        stages: Each stage (e.g. preprocess, solve) which holds tasks that execute the case
        clean: If True (default), clean the run_directory if it already exists
    """

    source_directory: InitVar[str | Path]
    run_directory: InitVar[str | Path]

    _run_path: Path = field(init=False)
    _source_path: Path = field(init=False)
    stages: dict[str, OFStage] = field(default_factory=OrderedDict)
    clean: bool = True

    @property
    def run_path(self):
        """Return the resolved run directory for this case instance."""
        return self._run_path

    @property
    def source_path(self):
        """Return the resolved source case directory copied into ``run_path``."""
        return self._source_path

    def __post_init__(self, source_directory, run_directory) -> None:

        # Ensure that directories are Path objects
        self._run_path = Path(run_directory).expanduser().resolve()
        self._source_path = Path(source_directory).expanduser().resolve()
        self._prepare_run_directory()

    def _prepare_run_directory(self) -> None:
        """Clean (optional) and create the run directory"""

        if self.run_path.exists():
            if self.clean:
              shutil.rmtree(self.run_path)
            else:
                raise FileExistsError(
                    "OpenFOAM case path already exists. Delete manually or re-run with `clean=True`"
                )
        self.run_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self._source_path, self.run_path, dirs_exist_ok=True)

    def add_stage(self, stage_name: str, tasks: OFTask | list[OFTask] | OFStage) -> None:
        """Add a named stage of ordered tasks.

        Args:
            stage_name: The name of the stage used for logging
            tasks: A task, list of tasks, or pre-built stage for this stage
        """

        if stage_name in self.stages:
            raise ValueError(f"Stage already exists: {stage_name}")

        if isinstance(tasks, OFStage):
            stage = tasks
        elif isinstance(tasks, OFTask):
            stage = OFStage(tasks=[tasks])
        else:
            stage = OFStage(tasks=tasks)

        for task in stage.tasks:
            self._inject_run_directory(task)

        self.stages[stage_name] = stage

    def _inject_run_directory(self, task: OFTask) -> None:
        """Ensure backend kwargs contain a cwd for the case run location."""

        backend_kwargs = dict(task.task_backend_specific_kwargs)
        backend_kwargs.setdefault("cwd", str(self._run_path))

        if "process_templates" in backend_kwargs:
            templates = []
            for nranks, template_cfg in backend_kwargs["process_templates"]:
                cfg = dict(template_cfg or {})
                cfg.setdefault("cwd", str(self._run_path))
                templates.append((nranks, cfg))
            backend_kwargs["process_templates"] = templates
        elif "process_template" in backend_kwargs:
            cfg = dict(backend_kwargs.get("process_template") or {})
            cfg.setdefault("cwd", str(self._run_path))
            backend_kwargs["process_template"] = cfg
        else:
            backend_kwargs["process_template"] = {"cwd": str(self._run_path)}

        task.task_backend_specific_kwargs = backend_kwargs
