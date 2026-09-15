"""Discovery helpers for OpenFOAM executable wrappers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .ofTask import OFRunFunction, OFTask, resolve_runfunctions_path


def _sanitize_identifier(name: str) -> str:
    """Return a valid attribute name from an executable name."""

    sanitized = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not sanitized:
        return "_"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _task_class_name(attribute_name: str) -> str:
    """Build a descriptive class name for a generated OFTask subclass."""

    if attribute_name.endswith("Task"):
        return attribute_name
    return f"{attribute_name}Task"


def _create_task_subclass(class_name: str, executable_path: str) -> type[OFTask]:
    """Create a dataclass that defaults `executable` to `executable_path`."""

    cls = type(
        class_name,
        (OFTask,),
        {
            "__module__": __name__,
            "__doc__": f"Auto-generated OFTask wrapper for '{Path(executable_path).name}'.",
            "__annotations__": {"executable": str},
            "executable": executable_path,
        },
    )
    return dataclass(cls)


@dataclass
class OFExecutableRegistry:
    """Container exposing generated OFTask subclasses as attributes."""

    directories: str | Path | Iterable[str | Path] | None = None
    recursive: bool = True
    include_default_locations: bool = True
    _classes: dict[str, type[OFTask]] = field(default_factory=dict, init=False, repr=False)
    _paths: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._classes = {}
        self._paths = {}

        self._register_runfunctions_task()

        if self.include_default_locations:
            for directory in _default_executable_directories():
                self.add_directory(directory, recursive=False, missing_ok=True)

        if self.directories is None:
            return

        if isinstance(self.directories, (str, Path)):
            directory_paths = [Path(self.directories)]
        else:
            directory_paths = [Path(path) for path in self.directories]

        for directory in directory_paths:
            self.add_directory(directory, recursive=self.recursive)

    def _register_runfunctions_task(self) -> None:
        """Attach the OpenFOAM ``RunFunctions`` helper task."""

        attr_name = "RunFunctions"
        self._classes[attr_name] = OFRunFunction

        try:
            self._paths[attr_name] = resolve_runfunctions_path()
        except FileNotFoundError:
            self._paths[attr_name] = "$WM_PROJECT_DIR/bin/tools/RunFunctions"

        setattr(self, attr_name, OFRunFunction)

    def available(self) -> list[str]:
        """Return attribute names that can be used to create tasks."""

        return sorted(self._classes)

    def executable_path(self, name: str) -> str:
        """Return the executable path used by a generated class."""

        return self._paths[name]

    def get(self, name: str) -> type[OFTask]:
        """Return a generated OFTask subclass by attribute name."""

        return self._classes[name]

    def __getattr__(self, name: str) -> type[OFTask]:
        """Resolve dynamically generated task classes for static analyzers."""

        try:
            return self._classes[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def add_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        missing_ok: bool = False,
    ) -> None:
        """Scan and register executables from one additional directory."""

        before = set(self._classes)
        _register_from_directory(
            task_classes=self._classes,
            executable_paths=self._paths,
            directory=directory,
            recursive=recursive,
            missing_ok=missing_ok,
        )
        for attr_name in set(self._classes) - before:
            setattr(self, attr_name, self._classes[attr_name])


def _iter_executables(directory: Path, recursive: bool) -> Iterable[Path]:
    """Yield executable files from `directory`."""

    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for entry in iterator:
        if entry.is_file() and entry.stat().st_mode & 0o111:
            yield entry


def _register_from_directory(
    task_classes: dict[str, type[OFTask]],
    executable_paths: dict[str, str],
    directory: str | Path,
    recursive: bool,
    missing_ok: bool = False,
) -> None:
    """Scan one directory and merge discovered executables into registries."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        if missing_ok:
            return
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    for executable in _iter_executables(root, recursive=recursive):
        executable_name = executable.name
        attr_name = _sanitize_identifier(executable_name)

        # Preserve first match order when duplicate executable names appear.
        if attr_name in task_classes:
            continue

        task_class = _create_task_subclass(
            class_name=_task_class_name(attr_name),
            executable_path=str(executable),
        )
        task_classes[attr_name] = task_class
        executable_paths[attr_name] = str(executable)


def _default_executable_directories() -> list[Path]:
    """Return existing executable directories derived from OpenFOAM env vars."""

    candidates: list[Path] = []

    for env_var in ("FOAM_APPBIN", "FOAM_USER_APPBIN"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())

    wm_project_dir = os.environ.get("WM_PROJECT_DIR")
    wm_options = os.environ.get("WM_OPTIONS")
    if wm_project_dir:
        root = Path(wm_project_dir).expanduser()
        candidates.append(root / "bin")
        if wm_options:
            candidates.append(root / "platforms" / wm_options / "bin")

    existing: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        resolved_str = str(resolved)
        if resolved_str in seen:
            continue
        seen.add(resolved_str)
        if resolved.is_dir():
            existing.append(resolved)

    return existing

