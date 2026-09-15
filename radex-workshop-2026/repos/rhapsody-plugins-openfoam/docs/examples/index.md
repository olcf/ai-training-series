# Examples

The [`examples/`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/examples) directory contains runnable drivers that show the OpenFOAM plugin being used for a range of workflows, from a minimal MPI "hello world" through full CFD cases and ensemble optimization.

| Example | Backend | Description |
| --- | --- | --- |
| [Hello World](hello-world.md) | Dragon | Runs a compiled MPI executable as a RHAPSODY `ComputeTask`, without any OpenFOAM-specific abstractions. |
| [Basic OpenFOAM Case](basic-openfoam.md) | Dragon | Decompose, solve, and reconstruct a simple case (`airFoil2D`) as three staged tasks. |
| [motorBike](motorbike.md) | Dragon | A realistic multi-stage case: mesh generation, initialization, solving, and reconstruction. |
| [Online Training & Inference](online-training-inference.md) | Dragon | Pipelines an MPI preprocessing chain with a persistent GPU inference service running alongside it. |
| [pitzDaily Optimization](pitzdaily-optimize.md) | Dragon | Runs an ensemble of `pitzDaily` cases per iteration, exchanging results with a Bayesian optimizer (`skopt`) over a radex-backed `DDict`. |

Each example's `driver.py` is run directly with `python`; drivers that use the Dragon backend call `multiprocessing.set_start_method("dragon", force=True)` before creating a `DragonExecutionBackend` (or `DragonExecutionBackendV3`).
