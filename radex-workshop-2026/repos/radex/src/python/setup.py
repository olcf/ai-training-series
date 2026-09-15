import os
import pathlib
import re
import shutil

import numpy
from Cython.Build import build_ext, cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py


HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
PY_SRC = HERE / "src"

RADEX_INSTALL_DIR = pathlib.Path(
    os.environ.get("RADEX_INSTALL_DIR", os.fspath(ROOT / "install"))
)
RADEX_INCLUDE_DIR = pathlib.Path(
    os.environ.get(
        "RADEX_INCLUDE_DIR", os.fspath(RADEX_INSTALL_DIR / "include")
    )
)
RADEX_LIB_DIR = pathlib.Path(
    os.environ.get("RADEX_LIB_DIR", os.fspath(next(RADEX_INSTALL_DIR.glob("lib*"))))
)
BUILD_CONFIG = RADEX_INCLUDE_DIR / "radex" / "build_config.hpp"


def _backend_enabled(macro):
    if not BUILD_CONFIG.is_file():
        raise FileNotFoundError(
            f"Could not find {BUILD_CONFIG}. Build and install the RaDex C++ "
            "library first, or point RADEX_INCLUDE_DIR at its headers."
        )
    pattern = rf"#if\s+1\s*\n\s*#define\s+{macro}\b"
    return re.search(pattern, BUILD_CONFIG.read_text()) is not None


def _required_env(name, backend):
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError(
            f"{name} must be set: the installed radex library was built with "
            f"{backend} support."
        ) from None


def make_extensions():
    include_dirs = [
        os.fspath(RADEX_INCLUDE_DIR),
        os.fspath(PY_SRC),
        numpy.get_include(),
    ]
    library_dirs = [os.fspath(RADEX_LIB_DIR)]
    libraries = ["radex"]
    runtime_library_dirs = [os.fspath(RADEX_LIB_DIR)]

    if _backend_enabled("RADEX_HAS_DRAGON"):
        include_dirs.append(_required_env("DRAGON_INCLUDE_DIR", "Dragon"))
        library_dirs.append(_required_env("DRAGON_LIB_DIR", "Dragon"))
        libraries.append("dragon")
        runtime_library_dirs.append(_required_env("DRAGON_LIB_DIR", "Dragon"))

    if _backend_enabled("RADEX_HAS_SMARTREDIS"):
        include_dirs.append(_required_env("SMARTREDIS_INCLUDE_DIR", "SmartRedis"))
        library_dirs.append(_required_env("SMARTREDIS_LIB_DIR", "SmartRedis"))
        libraries.append("smartredis")
        runtime_library_dirs.append(
            _required_env("SMARTREDIS_LIB_DIR", "SmartRedis")
        )

    common = {
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "libraries": libraries,
        "runtime_library_dirs": runtime_library_dirs,
        "language": "c++",
        "extra_compile_args": ["-std=c++17"],
    }

    def source_path(path):
        return os.fspath(path.relative_to(HERE))

    return [
        Extension(
            "radex.clients.core",
            sources=[source_path(PY_SRC / "radex/clients/core.pyx")],
            **common,
        ),
        Extension(
            "radex.handles.handles",
            sources=[source_path(PY_SRC / "radex/handles/handles.pyx")],
            **common,
        ),
    ]


class build_py_with_cpp_artifacts(build_py):
    """Vendor the installed RaDex tree into the Python distribution."""

    def run(self):
        super().run()
        if not RADEX_INSTALL_DIR.is_dir():
            raise FileNotFoundError(
                f"Could not find the installed RaDex tree at {RADEX_INSTALL_DIR}. "
                "Build and install the RaDex C++ library first, or set "
                "RADEX_INSTALL_DIR to its install prefix."
            )
        pkg_dir = pathlib.Path(self.build_lib) / "radex/cpp"
        shutil.copytree(RADEX_INSTALL_DIR, pkg_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_ext": build_ext,
            "build_py": build_py_with_cpp_artifacts,
        },
        packages=find_packages("src"),
        package_dir={"": "src"},
        package_data={
            "radex.cpp": [
                "bin/**/*",
                "include/**/*",
                "lib/**/*",
                "share/**/*",
            ]
        },
        ext_modules=cythonize(make_extensions()),
    )
