# Quick Start

This guide will help you run your first heterogeneous workflow using RHAPSODY in just a few minutes.

## Core Concepts

Before we start, keep these three key objects in mind:

1.  **Backend**: Where your tasks actually run (e.g., your local CPU, a Dask cluster, or an HPC queue).
2.  **Task**: A unit of work. `ComputeTask` for binaries/scripts, or `AITask` for inference.
3.  **Session**: The manager that connects backends and tasks together.

## Basic Workflow

Here is a complete script demonstrating a simple compute workflow.

```python
import asyncio
import rhapsody
from rhapsody.api import Session, ComputeTask
from rhapsody.backends import ConcurrentExecutionBackend

async def run_example():
    # 1. Choose an execution backend (Concurrent uses local threads/processes)
    backend = await ConcurrentExecutionBackend(name="local")

    # 2. Open a session
    async with Session(backends=[backend]) as session:

        # 3. Define tasks
        tasks = [
            ComputeTask(executable="/bin/echo", arguments=[f"Task {i}"])
            for i in range(1024)
        ]

        # 4. Submit tasks to the session
        await session.submit_tasks(tasks)

        # 5. Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # 6. Inspect results
        for t in tasks:
            print(f"{t.uid}: {t.state} (Output: {t.stdout})")

if __name__ == "__main__":
    asyncio.run(run_example())
```

!!! success "Output"
    If everything is set up correctly, you should see:
    ```text
    task.000001: DONE (Output: Task 0)
    task.000002: DONE (Output: Task 1)
    task.000003: DONE (Output: Task 2)
    task.000004: DONE (Output: Task 3)
    task.000005: DONE (Output: Task 4)
    task.000006: DONE (Output: Task 5)
    task.000007: DONE (Output: Task 6)
    task.000008: DONE (Output: Task 7)
    task.000009: DONE (Output: Task 8)
    task.000010: DONE (Output: Task 9)
    task.000011: DONE (Output: Task 10)
    task.000012: DONE (Output: Task 11)
    ......
    task.001021: DONE (Output: Task 1020)
    task.001022: DONE (Output: Task 1021)
    task.001023: DONE (Output: Task 1022)
    task.001024: DONE (Output: Task 1023)
    real    0m2.669s
    user    0m2.384s
    sys     0m2.071s
    ```

## What's Next?

- Explore **[Advanced Usage](advanced-usage.md)** for multi-backend and AI mixed workloads.
- Check the **[Configuration Guide](configuration.md)** for backend-specific tuning.
