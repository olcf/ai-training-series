"""Base wrapper types for emitting RHAPSODY ComputeTasks."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rhapsody.api import ComputeTask, Session
from dragon.data.ddict import DDict
from asyncio import Future


def resolve_runfunctions_path(runfunctions_path: str | Path | None = None) -> str:
    """Resolve the OpenFOAM ``RunFunctions`` path.
    """

    if runfunctions_path is not None:
        candidate = Path(runfunctions_path).expanduser().resolve()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"RunFunctions does not exist: {runfunctions_path}")

    value = os.environ.get("WM_PROJECT_DIR")

    if value:
        candidate = Path(value).expanduser().resolve() / "bin" / "tools" / "RunFunctions"
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Unable to locate OpenFOAM RunFunctions. Set WM_PROJECT_DIR or pass runfunctions_path."
    )


@dataclass
class OFTask:
    """Wrap an OpenFOAM command as a RHAPSODY ``ComputeTask``.

    Attributes:
        executable: Path or name of the executable to launch.
        args: Command-line arguments passed to ``executable``.
        options: Command-line options as name/value pairs.
        num_ranks: Number of MPI ranks/processes to request for this task.
        task_backend_specific_kwargs: Backend-specific options forwarded to
            ``ComputeTask.task_backend_specific_kwargs``.

    Notes:
        If ``num_ranks > 1``, ``-parallel`` is added to ``args`` when missing,
        and ``process_templates`` is synthesized when not explicitly provided.
    """

    executable: str = ""
    args: list[str | Path] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    num_ranks: int = 1
    execute_async: bool = False
    radex_store: str | DDict | None = None

    task_backend_specific_kwargs: dict[str, Any] = field(default_factory=dict)
    _options: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._options = {}
        self.add_options(self.options)
        if isinstance(self.radex_store, DDict):
            self.radex_store = self.radex_store.serialize()

    @staticmethod
    def _is_none_like(value: Any) -> bool:
        """Return true when an option should be emitted as a standalone flag."""

        if value is None:
            return True
        return isinstance(value, str) and value.strip().lower() in {"none", "null", "nil"}

    @staticmethod
    def _stringify_cli_value(value: Any) -> str:
        """Convert CLI values to strings, normalizing paths with as_posix."""

        if isinstance(value, Path):
            return value.as_posix()

        return str(value)

    def add_options(self, options: dict[str, Any] | None) -> None:
        """Add dictionary options after normalizing names and values.

        ``None``-like values (``None``, ``"null"``, etc.) emit standalone flags.
        """

        if options is None:
            return

        if not isinstance(options, dict):
            raise TypeError("options must be a dict[str, Any]")

        for raw_name, value in options.items():
            normalized = self._normalize_option_name(raw_name)

            if normalized in self._options and self._options[normalized] != value:
                raise ValueError(
                    f"option '{normalized}' already set to {self._options[normalized]!r}"
                )

            self._options[normalized] = value

        self.options = dict(self._options)

    @staticmethod
    def _normalize_option_name(raw_name: Any) -> str:
        """Normalize and validate option names for safe CLI emission."""

        normalized = str(raw_name).lstrip("-")
        if not normalized:
            raise ValueError("option name cannot be empty")
        if any(ch.isspace() for ch in normalized):
            raise ValueError(f"option name cannot contain whitespace: {raw_name!r}")
        return normalized

    def _render_option_arguments(self) -> list[str]:
        """Render normalized options as command-line arguments."""

        rendered: list[str] = []
        for name, value in self._options.items():
            option = f"-{name}"
            if self._is_none_like(value):
                rendered.append(option)
            else:
                rendered.extend([option, self._stringify_cli_value(value)])
        return rendered

    def _build_arguments(self) -> list[str]:
        """Build full command arguments from args, options, and parallel settings."""

        task_args = [self._stringify_cli_value(arg) for arg in self.args]
        task_args.extend(self._render_option_arguments())

        if (
            self.num_ranks > 1
            and "-parallel" not in task_args
        ):
            task_args.append("-parallel")

        return task_args

    def to_command_string(self) -> str:
        """Return a shell-escaped command string for this task."""

        parts = [self.executable, *self._build_arguments()]
        if not self.executable:
            parts = self._build_arguments()
        return shlex.join(parts)

    def to_compute_task(self) -> ComputeTask:
        """Build a RHAPSODY ComputeTask from this wrapper."""

        if self.num_ranks < 1:
            raise ValueError("num_ranks must be >= 1")

        backend_kwargs = dict(self.task_backend_specific_kwargs)
        task_args = self._build_arguments()

        if self.radex_store is not None:
            process_template = dict(backend_kwargs.get("process_template", {}) or {})
            env = dict(process_template.get("env", {}) or {})
            if os.name == "posix" and os.uname().sysname == "Darwin":
                foam_user_libbin = os.environ.get("FOAM_USER_LIBBIN")
                if foam_user_libbin:
                    fallback_paths = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
                    env["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(
                        path
                        for path in (foam_user_libbin, fallback_paths)
                        if path
                    )
            elif library_path := os.environ.get("LD_LIBRARY_PATH"):
                env.setdefault("LD_LIBRARY_PATH", library_path)
            env["RADEX_STORE"] = self._stringify_cli_value(self.radex_store)
            process_template["env"] = env
            backend_kwargs["process_template"] = process_template

        if self.num_ranks > 1 and "process_templates" not in backend_kwargs:
            template_cfg = dict(backend_kwargs.pop("process_template", {}) or {})
            backend_kwargs["process_templates"] = [(self.num_ranks, template_cfg)]

        return ComputeTask(
            executable=self.executable,
            arguments=task_args,
            task_backend_specific_kwargs=backend_kwargs,
            capture_stdio=True
        )

    def run_local(
        self,
        check: bool = False,
        timeout: float | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str]:
        """Execute this task locally and return ``(stdout, stderr)``.

        Args:
            check: When true, raise ``CalledProcessError`` on non-zero exit status.
            timeout: Optional timeout in seconds for process completion.
            cwd: Optional working directory to run the command in.
        """

        command = [self.executable, *self._build_arguments()]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
            cwd=None if cwd is None else str(cwd),
        )
        return completed.stdout, completed.stderr


class OFRunFunction(OFTask):
    """Run a command after sourcing OpenFOAM ``RunFunctions``.

    Example:
        ``reg.RunFunctions("restore0Dir", options={"parallel": None})``
    """

    def __init__(
        self,
        command: str,
        options: dict[str, Any] | None = None,
        *,
        runfunctions_path: str | Path | None = None,
        num_ranks: int = 1,
        task_backend_specific_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")

        self.command = command
        self.runfunctions_path = resolve_runfunctions_path(runfunctions_path)

        command_fragment = self._quote_command(command)
        option_fragment = self._render_cli_options(options)
        command_with_options = " ".join(
            part for part in (command_fragment, option_fragment) if part
        )
        shell_payload = f". {shlex.quote(self.runfunctions_path)} && {command_with_options}"
        super().__init__(
            executable="bash",
            args=["-lc", shell_payload],
            options={},
            num_ranks=num_ranks,
            task_backend_specific_kwargs=dict(task_backend_specific_kwargs or {}),
        )

    @staticmethod
    def _quote_command(command: str) -> str:
        """Return a shell-safe command string from a user-provided command."""

        tokens = shlex.split(command)
        if not tokens:
            raise ValueError("command must contain at least one token")
        return " ".join(shlex.quote(token) for token in tokens)

    @staticmethod
    def _render_cli_options(options: dict[str, Any] | None) -> str:
        """Render option input to a shell-safe argument fragment."""

        if options is None:
            return ""

        if not isinstance(options, dict):
            raise TypeError("options must be a dict[str, Any]")

        tokens: list[str] = []
        for name, value in options.items():
            normalized = OFTask._normalize_option_name(name)
            tokens.append(f"-{normalized}")
            if not OFTask._is_none_like(value):
                tokens.append(OFTask._stringify_cli_value(value))

        return " ".join(shlex.quote(token) for token in tokens)

    def _build_arguments(self) -> list[str]:
        """Return shell arguments without OFTask option expansion.

        For RunFunctions tasks, options are already rendered into the shell payload.
        """

        return [self._stringify_cli_value(arg) for arg in self.args]


@dataclass
class OFStage:
    """An ordered collection of ``OFTask`` objects for one pipeline stage.

    Attributes:
        tasks: Tasks to execute in sequence for this stage.
    """

    tasks: list[OFTask] = field(default_factory=list)

    def add_tasks(self, task: OFTask | list[OFTask]) -> None:
        """Add one task or a list of tasks to the stage."""

        if isinstance(task, OFTask):
            self.tasks.append(task)
            return

        if isinstance(task, list):
            if not all(isinstance(item, OFTask) for item in task):
                raise TypeError("all items in task list must be OFTask instances")
            self.tasks.extend(task)
            return

        raise TypeError("task must be an OFTask or list[OFTask]")

    def to_tasks(self) -> list[ComputeTask]:
        """Return a list of ComputeTasks by calling to_compute_task on each task."""
        return [task.to_compute_task() for task in self.tasks]

    def run_local(
        self,
        check: bool = True,
        timeout: float | None = None,
        cwd: str | Path | None = None,
    ) -> list[tuple[str, str]]:
        """Execute each stage task locally and return their ``(stdout, stderr)`` outputs."""
        return [task.run_local(check=check, timeout=timeout, cwd=cwd) for task in self.tasks]

    async def execute(self, session: Session) -> list[Future]:
        """Execute sync tasks in order while async tasks run in background.

        Args:
            session: the RHAPSODY session used to submit tasks
        """

        async_tasks: list[Any] = []
        sync_tasks: list[Any] = []
        for task in self.tasks:
            compute_task = task.to_compute_task()
            if task.execute_async:
                async_tasks.append(compute_task)
            else:
                sync_tasks.append(compute_task)

        async_futures = []
        if async_tasks:
            async_futures = await session.submit_tasks(async_tasks)

        for task in sync_tasks:
            futures = await session.submit_tasks([task])
            await futures[0]

        return async_futures
