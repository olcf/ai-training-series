# RHAPSODY

**Runtime for Heterogeneous Applications, Service Orchestration and Dynamism**

RHAPSODY is a high-performance runtime system designed for orchestrating complex, heterogeneous workflows that combine traditional HPC tasks with modern AI/ML inference services.

!!! note "Core Objective"
    RHAPSODY aims to provide a unified, asynchronous and scalable runtime capabilities for managing AI-HPC workloads and workflows across diverse computing infrastructures, from local workstations to large-scale HPC machines.

## Key Features

- **Heterogeneous Task Support**: Seamlessly mix and scale `ComputeTask` (HPC/Binary) and `AITask` (Inference) in a single workflow.
- **Asynchronous API**: Built on Python's `asyncio`, allowing for high-throughput task submission and non-blocking state monitoring.
- **Extensible Backends**: Scale to large number of nodes with multiple execution backends including Dragon, Dask, RADICAL-Pilot, remote execution via ORBIT, and local concurrent workers.
- **Integrations**: Integrates with with highly scalable workflow systems such `radical.asyncflow` and agentic frameworks such as `flowgentic`

## Quick Example

```python
import asyncio
from rhapsody import Session, ComputeTask
from rhapsody.backends import DragonExecutionBackend

async def main():
    # 1. Initialize session with a backend
    backend = await DragonExecutionBackend()
    async with Session(backends=[backend]) as session:

        # 2. Define a task
        task = ComputeTask(executable="/bin/echo", arguments=["Hello, Rhapsody!"])

        # 3. Submit and wait
        await session.submit_tasks([task])
        result = await task

        print(f"Task finished with state: {task.state}")
        print(f"Output: {task.stdout}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Getting Started

Ready to dive in? Check out our [Installation Guide](getting-started/installation.md) or jump straight into the [Quick Start](getting-started/quick-start.md).
