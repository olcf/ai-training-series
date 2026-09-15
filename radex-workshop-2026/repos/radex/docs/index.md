# The Radical Data Exchanger (radex)

**Overview**

radex is a lightweight, high-performance data exchange layer for moving scalars and tensors between the heterogeneous components of an HPC-AI workflow — MPI simulations and Python functions without going through a filesystem.

The library is based around a C++ core (with a Cython-based Python client) for putting and getting named values into shared backends such as Dragon's Distributed Dictionary (DDict) or Redis, so producers and consumers written in different languages can exchange data directly.

## Key Features

- **Typed put/get API**: Exchange scalars (`int32`, `int64`, `float`, `double`) and n-dimensional tensors by name (`OutgoingHandle` / `IncomingHandle`).
- **Multiple backends**: Use the same API but different clients for interacting with different stores: [Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html) DDict and Redis
- **C++ and Python clients**: C++ and Cython-based Python client (`radex`) built against the same compiled core, so both languages can read/write the same keys.

## Getting Started

Ready to dive in? Check out our [Installation Guide](getting-started/installation.md), learn how to [build from source](getting-started/building-from-source.md), or jump straight into the [Quick Start](getting-started/quick-start.md) and [Examples](examples/index.md).
