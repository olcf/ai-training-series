# Quick Start

This walks through the smallest useful case: decompose, solve, and reconstruct an OpenFOAM case as a three-stage RHAPSODY workflow, using the same pattern as the [Basic OpenFOAM Case example](../examples/basic-openfoam.md).

## 1. Define the Case and Its Stages

```python
import rhapsody_plugins.openfoam as rof

def create_openfoam_case():
    case = rof.CaseDefinition("path/to/case", "./run", clean=True)
    reg = rof.OFExecutableRegistry()

    case.add_stage("preprocessing", reg.decomposePar())
    case.add_stage("solve", reg.simpleFoam(num_ranks=8))
    case.add_stage("postprocessing", reg.reconstructPar())

    return case
```

- `CaseDefinition` copies `path/to/case` into `./run` (cleaning it first, since `clean=True`).
- `OFExecutableRegistry` exposes discovered OpenFOAM executables as `OFTask` subclasses; calling one (e.g. `reg.simpleFoam(num_ranks=8)`) builds a task for that executable.
- `case.add_stage(name, tasks)` accepts a single task, a list of tasks, or a pre-built `OFStage`.

## 2. Run the Stages Through a RHAPSODY Session

```python
import asyncio
import multiprocessing as mp

from rhapsody.backends import DragonExecutionBackend

async def main():
    mp.set_start_method("dragon", force=True)

    backend = await DragonExecutionBackend()
    case = create_openfoam_case()

    async with rof.OFSession(backends=[backend]) as session:
        for stage_name, stage in case.stages.items():
            print(f"Running stage: {stage_name}")
            await stage.execute(session)

asyncio.run(main())
```

`OFSession` is a RHAPSODY `Session` used as an async context manager. Each `OFStage.execute(session)` submits its tasks and awaits their completion before moving to the next stage.

## Next Steps

- See the full [Examples](../examples/index.md) for multi-stage meshing (motorBike), live field streaming (Online Training & Inference), and ensemble optimization (pitzDaily).
- Check the [API Reference](../api/index.md) for `CaseDefinition`, `OFStage`, `OFTask`, and `OFExecutableRegistry` details.
