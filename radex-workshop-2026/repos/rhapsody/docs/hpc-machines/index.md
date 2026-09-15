# RHAPSODY on NSF and DOE Machines

This section provides machine-specific setup guides and verified examples for running RHAPSODY on NSF and DOE leadership computing facilities. These machines are periodically tested and verified against RHAPSODY and its pre-release changes.

## Backend compatibility

**`ConcurrentExecutionBackend (designed for debug and tests single node only)`** works out of the box on every machine listed here — no machine-specific configuration is required. Install RHAPSODY and run.

**`DragonExecutionBackend (designed for large scale multi-node)`** requires machine-specific setup: each HPC system has its own software stack, MPI library, interconnect, and job launcher. The exact steps — module loading, virtual environment creation, transport configuration, and launch command — differ per machine and are documented in each section below.

Each guide covers environment setup, module loading, job allocation, and working examples.

## Supported Machines

| Machine | Facility | Stack | Status |
|---------|----------|-------|--------|
| [NCSA Delta](ncsa-delta.md) | NSF / NCSA | Cray MPICH (PMIx) | Verified |
| [NERSC Perlmutter](perlmutter.md) | DOE / NERSC | Cray libfabric + CUDA | Verified |
| [PSC Bridges-2](bridges2.md) | NSF / PSC | Standard SLURM, TCP | Verified |
| [Purdue Anvil](anvil.md) | NSF / ACCESS / Purdue | Cray MPICH (PMIx) | Verified |

More machines will be added as they are tested and validated.
