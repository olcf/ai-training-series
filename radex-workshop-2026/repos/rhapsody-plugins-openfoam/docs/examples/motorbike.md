# motorBike

**Location:** [`examples/motorBike`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/examples/motorBike)

A more realistic case built on the standard OpenFOAM `motorBike` tutorial, showing multi-step mesh generation feeding into initialization, solving, and reconstruction — all defined as named `OFStage`s on a single `CaseDefinition`.

```python
--8<-- "examples/motorBike/driver.py"
```

## Stages

| Stage | Tasks |
| --- | --- |
| `mesh-generation-1` | `surfaceFeatureExtract`, `blockMesh`, `decomposePar` |
| `mesh-generation-2` | `snappyHexMesh` (parallel), `topoSet` (parallel) |
| `initialization` | `restore0Dir`, `patchSummary`, `potentialFoam`, `checkMesh` (all parallel) |
| `Solver` | `simpleFoam` (parallel) |
| `reconstruct` | `reconstructParMesh`, `reconstructPar` |

The number of subdomains for parallel stages is read directly out of the case's `decomposeParDict` using `reg.foamDictionary(...).run_local(...)`, so the workflow adapts to however the case was already decomposed.

## Running

```bash
cd examples/motorBike
python driver.py
```

The driver copies the case's geometry (`motorBike.obj.gz`) into `constant/triSurface` of the run directory before executing any stages.
