import dataclasses
import functools
import operator
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

_SUPPORTED_NP_DTYPES = {
    np.int32: "std::int32_t",
    np.int64: "std::int64_t",
    np.float32: "float",
    np.float64: "double",
}


@pytest.fixture(scope="session")
def _radex_cpp():
    try:
        import radex.cpp
    except ImportError:
        pytest.fail(
            "The `radex_cpp` package is not importable. Install the RaDex "
            "Python package to make the C++ headers and libraries available."
        )
    yield radex.cpp


@pytest.fixture(scope="session")
def _radex_lib_dir(_radex_cpp):
    lib_dir = pathlib.Path(_radex_cpp.get_lib_dir())
    os_to_libname = {"linux": "libradex.so", "darwin": "libradex.dylib"}
    libname = os_to_libname[sys.platform]
    if not (lib_dir / libname).is_file():
        raise FileNotFoundError(f"Cannot find {libname} at: {lib_dir}")
    yield lib_dir


@pytest.fixture(scope="session")
def _radex_include_dir(_radex_cpp):
    yield pathlib.Path(_radex_cpp.get_include())


@pytest.fixture(scope="session")
def _radex_lib_name(_radex_cpp):
    yield _radex_cpp.get_lib_name()


@pytest.fixture(scope="function")
def map_np_dtypes_to_cpp_types():
    yield _SUPPORTED_NP_DTYPES.copy()


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(dtype, id=f"dtype-{dtype.__name__}")
        for dtype in _SUPPORTED_NP_DTYPES
    ],
)
def np_dtype(request):
    yield request.param


@pytest.fixture(scope="function")
def random_np_value(np_dtype):
    rng = np.random.default_rng()
    if np.issubdtype(np_dtype, np.integer):
        value = rng.integers(0, 100, dtype=np_dtype)
    else:
        value = np_dtype(rng.random(dtype=np_dtype))
    yield value


def _fmt_shape(shape):
    return "x".join(str(dim) for dim in shape)


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(shape, id=f"shape-{_fmt_shape(shape)}")
        for shape in [
            (10,),
            (5, 2),
            (2, 5),
            (24,),
            (6, 4),
            (3, 8),
            (12, 2),
            (4, 3, 2),
            (2, 4, 3),
            (2, 2, 3, 2),
            (9, 2, 2),
            (3, 2, 2, 3),
        ]
    ],
)
def random_np_tensor(np_dtype, request):
    shape = request.param
    size = functools.reduce(operator.mul, shape, 1)
    yield np.arange(size, dtype=np_dtype).reshape(shape)


@dataclasses.dataclass(frozen=True)
class MyPickleable:
    some_str: str
    some_int: int


@dataclasses.dataclass(frozen=True)
class MyPickleable2:
    some_float: float


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(picklable, id=f"picklable-{i}")
        for i, picklable in enumerate(
            [
                MyPickleable("spam", 123),
                MyPickleable("eggs", 0),
                MyPickleable("ham", -72),
                MyPickleable2(1.23),
                MyPickleable2(-98.7),
            ]
        )
    ],
)
def random_picklable(request):
    yield request.param


@pytest.fixture
def _cpp_compiler_path():
    cpp_path = shutil.which(os.environ.get("CXX", "g++"))
    if cpp_path is None:
        pytest.fail("Could not find C++ compiler")
    yield cpp_path


@pytest.fixture
def cpp_type_name(np_dtype, map_np_dtypes_to_cpp_types):
    if (type_ := map_np_dtypes_to_cpp_types.get(np_dtype, None)) is None:
        pytest.fail(f"No C++ type provided for dtype {np_dtype.__name__}")
    return type_


# Hacky but effective-ish way to apply a `compiled` mark to any test that uses
# this fixture
# For more contex: https://github.com/pytest-dev/pytest/issues/1368
@pytest.fixture(
    params=[pytest.param(..., id="compiler-CXX", marks=pytest.mark.compiled)]
)
def cpp_compile(
    tmp_path, _cpp_compiler_path, _radex_include_dir, _radex_lib_dir, _radex_lib_name
):
    def _compile(src, src_name="src.cpp", bin_name="a.out", extra_compile_args=()):
        src_file = tmp_path / src_name
        src_file.touch()
        src_file.write_text(src)
        bin_path = tmp_path / bin_name
        subprocess.check_call(
            [
                os.fspath(_cpp_compiler_path),
                "--std=c++17",
                f"-I{os.fspath(_radex_include_dir)}",
                os.fspath(src_file),
                *extra_compile_args,
                f"-L{os.fspath(_radex_lib_dir)}",
                f"-Wl,-rpath,{os.fspath(_radex_lib_dir)}",
                f"-l{_radex_lib_name}",
                "-o",
                os.fspath(bin_path),
            ]
        )
        return bin_path

    yield _compile
