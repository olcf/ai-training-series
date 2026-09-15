"""Example: DaskExecutionBackend with sync functions, async functions, and executables.

Runs a LocalCluster automatically. Pass cluster= or client= to target SLURM, Kubernetes,
or any other Dask-compatible cluster — no other code changes needed.
"""

import asyncio
import logging

import rhapsody
from rhapsody.api import ComputeTask
from rhapsody.api import Session
from rhapsody.backends import DaskExecutionBackend

rhapsody.enable_logging(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def compute_square_sync(n):
    """Sync function — dispatched natively to Dask workers."""
    import time

    time.sleep(0.1)
    return {"input": n, "result": n * n}


async def compute_square_async(n):
    """Async function — wrapped transparently, name visible in Dask dashboard."""
    import asyncio

    await asyncio.sleep(0.1)
    return {"input": n, "result": n * n}


async def main():
    async with DaskExecutionBackend() as backend:
        session = Session(backends=[backend])

        tasks = []
        tasks += [ComputeTask(function=compute_square_sync, args=(i,)) for i in range(10)]
        tasks += [ComputeTask(function=compute_square_async, args=(i,)) for i in range(10)]
        tasks += [ComputeTask(executable="/bin/echo", arguments=[f"task {i}"]) for i in range(10)]

        async with session:
            await session.submit_tasks(tasks)
            await session.wait_tasks(tasks)

    done = sum(1 for t in tasks if t.state == "DONE")
    failed = sum(1 for t in tasks if t.state == "FAILED")
    print(f"Done: {done}, Failed: {failed}")
    for t in tasks:
        if t.function:
            # function task: result in return_value, error in exception
            output = t.return_value if t.state == "DONE" else t.exception
        else:
            # executable task: output in stdout, error in stderr
            output = t.stdout.strip() if t.state == "DONE" else t.stderr
        print(f"{t.uid} -> {output}")


if __name__ == "__main__":
    asyncio.run(main())
