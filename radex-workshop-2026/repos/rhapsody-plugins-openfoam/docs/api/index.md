# API Reference

`rhapsody-plugins-openfoam` exposes a single Python API used to define and run OpenFOAM cases as RHAPSODY tasks.

- **Python API** — see "API Reference > Python" in the navigation, generated from the `rhapsody_plugins.openfoam` package under `src/python/rhapsody_plugins/openfoam` (modules: `caseDefinition`, `executableRegistry`, `ofTask`, `ofSession`, `utils`). The package and its `rhapsody`/`dragonhpc` dependencies must be importable in the documentation build environment for these pages to be populated. See [Installation](../getting-started/installation.md).
- **OpenFOAM Function Objects** — the `radexWrite` and `radexRead` function objects are documented alongside their build instructions in [Building the OpenFOAM Function Objects](../getting-started/building-the-function-object.md).

## Core Types

- `CaseDefinition` — copies a source case into a run directory and holds the case's named `stages`.
- `OFStage` — an ordered group of `OFTask`s executed as a unit (e.g. `mesh-generation`, `solve`).
- `OFTask` — a RHAPSODY-compatible task wrapping a single OpenFOAM utility or solver invocation.
- `OFExecutableRegistry` — discovers OpenFOAM executables on `PATH` and exposes them as `OFTask` subclasses (e.g. `reg.blockMesh()`, `reg.simpleFoam(num_ranks=8)`).
- `OFSession` — a RHAPSODY `Session` subclass used as the async context manager that executes a case's stages against one or more backends.
