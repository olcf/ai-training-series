from __future__ import annotations

import abc
import dataclasses
import importlib.metadata
import os
import pathlib
import re
import subprocess as sp
import sys

import pytest

HERE = pathlib.Path(__file__).absolute().parent
ROOT = HERE.parent.parent
EXAMPLES_DIR = ROOT / "example"


@dataclasses.dataclass(frozen=True)
class Example(abc.ABC):
    directory: pathlib.Path
    marks: list[pytest.MarkDecorator] = dataclasses.field(
        default_factory=list, kw_only=True
    )

    def __post_init__(self):
        self.marks.append(pytest.mark.example)

    @abc.abstractmethod
    def _run(self, cwd, out, err) -> int: ...

    @property
    def test_id(self) -> str:
        return os.fspath(self.directory.relative_to(EXAMPLES_DIR))

    @property
    def driver(self) -> pathlib.Path:
        return self.directory / "driver.py"

    @property
    def requirements_file(self) -> pathlib.Path:
        return self.directory / "requirements.txt"

    @property
    def expected_stdout(self) -> pathlib.Path:
        return self.directory / "result.txt"

    def check_for_requirements(self) -> None:
        if not self.requirements_file.exists():
            return
        pattern = re.compile(r"^\s*([\w\.\-]+)")
        with self.requirements_file.open("r", encoding="utf-8") as reqs:
            for req in reqs:
                match = pattern.match(req)
                if match is None:
                    continue
                pkg_name = match.group(1)
                try:
                    importlib.metadata.version(pkg_name)
                except importlib.metadata.PackageNotFoundError:
                    pytest.skip(
                        f"Failed to find requirement `{pkg_name}`. "
                        "Try running "
                        f"`pip install -r {os.fspath(self.requirements_file)}`"
                    )

    def run(self, *, where) -> tuple[int, pathlib.Path, pathlib.Path]:
        self.check_for_requirements()
        out_file = where / "example.out"
        err_file = where / "example.err"
        with (
            open(out_file, "w", encoding="utf-8") as out,
            open(err_file, "w", encoding="utf-8") as err,
        ):
            return self._run(self.directory, out, err), out_file, err_file


class LocalExample(Example):
    def _run(self, cwd, out, err) -> int:
        return sp.run(
            [sys.executable, os.fspath(self.driver)], cwd=cwd, stdout=out, stderr=err
        ).returncode


class DragonExample(Example):
    def __init__(
        self,
        directory: pathlib.Path,
        num_nodes: int | None,
        *,
        marks: list[pytest.MarkDecorator] | None = None,
    ) -> None:
        marks = marks or []
        dragon_args = []
        if num_nodes is not None and num_nodes <= 0:
            raise ValueError(
                "Dragon examples must either be runnable without an allocation "
                "or on a posative number of nodes"
            )

        try:
            import dragon
            from dragon.globalservices.api_setup import get_gs_ret_cuid
            from dragon.native.machine import System as DrgSystem
        except ImportError:
            libs_path = None
            marks.append(pytest.mark.skip(reason="This example requires dragon"))
        else:
            NOT_ENOUGH_NODES = pytest.mark.skip(
                reason=f"Example requires an allocation of {num_nodes} node(s)"
            )
            try:
                get_gs_ret_cuid()
            except Exception:
                # Test suite was not run through dragon
                dragon_args.append("-s")
                if num_nodes is not None:
                    marks.append(NOT_ENOUGH_NODES)
            else:
                # Test suite was run through dragon (including `dragon -s ...`)
                drg_system = DrgSystem()
                if num_nodes is not None and num_nodes < drg_system.nnodes:
                    marks.append(NOT_ENOUGH_NODES)

                # FIXME: We should allow for running multinode examples. This
                #        is a good enough starting point to wire in a few
                #        single node examples for now. We should probably move
                #        this over to launching a proper
                #        `dragon.native.process.Process` or equivalent.
                if num_nodes is None:
                    dragon_args.append("-s")
                else:
                    dragon_args.extend(["-N", str(num_nodes)])
                    marks.append(
                        pytest.mark.xfail(
                            strict=False,
                            reason=(
                                "Examples that cannot be run with `dragon -s ...` "
                                "are not fully supported in the test suite. See "
                                "https://github.com/radical-cybertools/radex/issues/25 "
                                "for more info"
                            ),
                        )
                    )

        super().__init__(directory, marks=marks)
        self._dragon_args = dragon_args

    def _run(self, cwd, out, err) -> int:
        return sp.run(
            [
                sys.executable,
                "-m",
                "dragon",
                "--",
                *self._dragon_args,
                os.fspath(self.driver),
            ],
            cwd=cwd,
            stdout=out,
            stderr=err,
        ).returncode


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(example, id=example.test_id, marks=example.marks)
        for example in [
            LocalExample(EXAMPLES_DIR / "cpp-exchange/in-mem"),
            LocalExample(EXAMPLES_DIR / "cpp-exchange/redis"),
            DragonExample(EXAMPLES_DIR / "cpp-exchange/dragon", num_nodes=None),
            DragonExample(
                EXAMPLES_DIR / "py-cpp-exchange/dragon",
                num_nodes=None,
                marks=[pytest.mark.slow],
            ),
        ]
    ],
)
def example(request):
    yield request.param
