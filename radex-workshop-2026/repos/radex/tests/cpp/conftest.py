from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cpp-bin-dir",
        action="store",
        default=None,
        help=(
            "Directory containing C++ test executables. "
            "Defaults to {radex_path}/install/bin/tests."
        ),
    )


@pytest.fixture
def resolve_bin_path(pytestconfig):
    def _resolve(name):
        cpp_bin_dir = pytestconfig.getoption("--cpp-bin-dir")
        if cpp_bin_dir:
            cpp_bin = Path(cpp_bin_dir).expanduser() / name
        else:
            repo_root = Path(__file__).resolve().parents[2]
            cpp_bin = repo_root / "install" / "bin" / "tests" / name

        if not cpp_bin.is_file():
            raise pytest.UsageError(
                f"C++ test executable not found at {cpp_bin}. "
                "Build and install first or set --cpp-bin-dir."
            )
        return cpp_bin

    return _resolve
