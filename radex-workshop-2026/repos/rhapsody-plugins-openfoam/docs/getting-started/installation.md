# Installation

## Prerequisites

- Python >= 3.9
- A sourced [OpenFOAM](https://www.openfoam.com) installation (required to run cases; only needed to build the [`radexWrite` function object](building-the-function-object.md) if you want live field export)
- [Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html) (`dragonhpc`), used as the default RHAPSODY execution backend in the examples
- [RHAPSODY](https://radical-cybertools.github.io/rhapsody) (`rhapsody-py`), used as the runtime for execution workflows
- [radex](https://radical-cybertools.github.io/radex/), used to exchange data in-memory from OpenFOAM to other workflow component

## Installing the Python Plugin

The Python package lives under [`src/python`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/src/python) and is installed with `pip`/`setuptools` from the repository root:

```bash
git clone https://github.com/CrayLabs/rhapsody-plugins-openfoam.git
cd rhapsody-plugins-openfoam
pip install -e .
```

This installs the `rhapsody_plugins.openfoam` package along with its declared dependencies (`dragonhpc`, `rhapsody-py`).

!!! note
    The package is imported as `rhapsody_plugins.openfoam`, e.g. `import rhapsody_plugins.openfoam as rof`.

## Installing radex
Currently radex is not available via PyPI. Please follow these [installation directions](https://radical-cybertools.github.io/radex/getting-started/installation/)

## Verifying the Install

```bash
python -c "import rhapsody_plugins.openfoam as rof; print(rof.CaseDefinition)"
python -c "from radex.clients.core import DragonClient as Client"
```

## Next Steps

- [Build the function objects](building-the-function-object.md) if you want to stream live field data out of a running solver.
- Jump into the [Quick Start](quick-start.md) to run your first case.
