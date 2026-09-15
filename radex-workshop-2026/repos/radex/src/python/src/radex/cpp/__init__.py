"""Locations of the RaDex C++ headers and libraries bundled with the package."""

import os
import pathlib

_HERE = pathlib.Path(__file__).parent.absolute()

__all__ = ["get_include", "get_lib_dir", "get_lib_name"]


def get_include() -> pathlib.Path:
	"""Return the directory containing the bundled RaDex headers."""
	return _HERE / "include"


def get_lib_dir() -> pathlib.Path:
	"""Return the directory containing the bundled RaDex libraries."""
	return next(_HERE.glob("lib*"))


def get_lib_name() -> str:
	"""Return the RaDex library name without the linker prefix or suffix."""
	return "radex"
