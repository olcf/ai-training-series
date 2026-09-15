# Developer's Guide

RaDex's development tooling — linting, tests, and documentation — lives under [`dev-resources/`](https://github.com/radical-cybertools/RaDex/tree/main/dev-resources), driven by [`dev-resources/Makefile`](https://github.com/radical-cybertools/RaDex/blob/main/dev-resources/Makefile) and two requirements files:

- [`dev-resources/requirements-dev.txt`](https://github.com/radical-cybertools/RaDex/blob/main/dev-resources/requirements-dev.txt) — `pytest`, `black`, `isort` (linting and testing).
- [`dev-resources/requirements-doc.txt`](https://github.com/radical-cybertools/RaDex/blob/main/dev-resources/requirements-doc.txt) — `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`, `mkdoxy`, and related plugins (building this documentation site).

All `make` targets below are run with `-f dev-resources/Makefile` from the repository root (or `cd dev-resources && make ...`).

## Linting

```bash
pip install -r dev-resources/requirements-dev.txt
make -f dev-resources/Makefile format       # black + isort, auto-fix
make -f dev-resources/Makefile lint         # black-check + isort-check, no changes
```

## Running Tests

```bash
pip install -r dev-resources/requirements-dev.txt
make -f dev-resources/Makefile test
```

This runs `pytest` (via `dragon -s --`) against `tests/`, configured by [`dev-resources/radex-config.toml`](https://github.com/radical-cybertools/RaDex/blob/main/dev-resources/radex-config.toml). Pass extra `pytest` flags with `PYTEST_ARGS`, e.g.:

```bash
make -f dev-resources/Makefile test PYTEST_ARGS="-vvv -s"
```

The C++ library must already be built and installed (see [Building From Source](building-from-source.md)) and, for backend-specific tests, the [Dragon/SmartRedis environments fetched](fetching-backends.md).

## Building the Documentation

```bash
make -f dev-resources/Makefile docs         # installs requirements-doc.txt, then `mkdocs build --clean`
make -f dev-resources/Makefile docs-serve   # same, then `mkdocs serve` with live-reload on 0.0.0.0:8001
```

Both targets install [`dev-resources/requirements-doc.txt`](https://github.com/radical-cybertools/RaDex/blob/main/dev-resources/requirements-doc.txt) with `PYTHON` (defaults to `python3`; override with `PYTHON=/path/to/python`) before invoking `mkdocs`. The C++ API reference additionally requires `doxygen` on `PATH`, and the Python API reference is only populated if `radex` is built/installed into that same Python environment (see [API Reference](../api/index.md)).
