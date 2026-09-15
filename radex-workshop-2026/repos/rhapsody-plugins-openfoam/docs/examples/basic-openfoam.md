# Basic OpenFOAM Case

**Location:** [`examples/basic-openfoam`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/examples/basic-openfoam)

Runs the `airFoil2D` case (under [`examples/openfoam-cases`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/examples/openfoam-cases)) through the three canonical OpenFOAM stages — decompose, solve, reconstruct — using the `rhapsody_plugins.openfoam` API described in the [Quick Start](../getting-started/quick-start.md).

```python
--8<-- "examples/basic-openfoam/driver.py"
```

- `OFExecutableRegistry` discovers the `decomposePar`, `simpleFoam`, and `reconstructPar` executables and exposes them as callables that build `OFTask`s.
- `simpleFoam(num_ranks=8)` runs the solver across 8 MPI ranks.
- Each stage is executed and awaited in order against the same `OFSession`/`DragonExecutionBackend`.

## Running

```bash
cd examples/basic-openfoam
python driver.py
```
