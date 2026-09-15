# API Reference

radex ships a C++ core with two language bindings: a Cython-based Python client and a native C++ client.

## Python

The **Python** section in the nav is generated automatically from the docstrings in the `radex` package (`src/python/src/radex`). It's built when `radex` is importable in the environment running `mkdocs build` — see [Installation](../getting-started/installation.md) for how to build and install it.

## C++

The **C++** section is generated from the headers under `include/radex` via [Doxygen](https://www.doxygen.nl/) and [mkdoxy](https://mkdoxy.kubaandrysek.cz/), starting at [radexCpp/annotated.md](../radexCpp/annotated.md).
