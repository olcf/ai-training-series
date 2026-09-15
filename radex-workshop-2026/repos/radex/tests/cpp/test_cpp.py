import subprocess

import pytest

CPP_TEST_BINARIES = [
    "test-local-client",
]


@pytest.mark.parametrize("binary_name", CPP_TEST_BINARIES)
def test_cpp(binary_name, resolve_bin_path):
    exe_path = resolve_bin_path(binary_name)
    result = subprocess.run([str(exe_path)], capture_output=True, text=True)

    assert result.returncode == 0, (
        f"C++ binary failed: {exe_path}\n"
        f"exit code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
