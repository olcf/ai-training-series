# RHAPSODY Plugins: OpenFOAM

**Overview**

`rhapsody-plugins-openfoam` connects [RHAPSODY](https://radical-cybertools.github.io/rhapsody) workflows to [OpenFOAM](https://www.openfoam.com), letting you drive multi-stage CFD cases (meshing, solving, post-processing) as RHAPSODY tasks, and stream field data out of a running solver in real time via a [radex](https://github.com/radical-cybertools/radex)-backed OpenFOAM function object.

The project has two parts that can be used independently or together:

- **Python plugin (`rhapsody_plugins.openfoam`)** — case and task abstractions (`CaseDefinition`, `OFStage`, `OFTask`, `OFExecutableRegistry`, `OFSession`) that wrap OpenFOAM utilities and solvers as RHAPSODY tasks, so a case's stages can be scheduled and executed across RHAPSODY execution backends (e.g. Dragon).
- **OpenFOAM function object (`radexWrite`)** — a compiled `libso` function object that exports `volScalarField`/`volVectorField` data (and other scalar results) from a running solver into a radex-backed key-value store at write intervals, so external Python code (training loops, monitors, optimizers) can consume simulation data without touching the filesystem.

## Key Features

- **Staged case execution**: Group OpenFOAM utilities and solvers into named stages (`preprocessing`, `solve`, `postprocessing`, ...) on a `CaseDefinition` and run them as RHAPSODY tasks.
- **Executable discovery**: `OFExecutableRegistry` exposes OpenFOAM utilities/solvers (`blockMesh`, `decomposePar`, `simpleFoam`, ...) as ready-to-use `OFTask` subclasses.
- **In-memory field export**: The `radexWrite` function object writes solver fields into a Dragon `DDict` or Redis store keyed by `<fieldName>_<timeStep>_<subdomainId>`, for zero-copy consumption by Python.
- **Composable with RHAPSODY backends**: Cases run through the same `Session`/backend model as any other RHAPSODY workload (e.g. `DragonExecutionBackend`).

## Getting Started

Ready to dive in? Check out our [Installation Guide](getting-started/installation.md), learn how to [build the `radexWrite` function object](getting-started/building-the-function-object.md), or jump straight into the [Quick Start](getting-started/quick-start.md) and [Examples](examples/index.md).
