import asyncio
import logging
import multiprocessing as mp

import rhapsody
from rhapsody.api import ComputeTask
from rhapsody.api import Session
from rhapsody.backends import ConcurrentExecutionBackend

rhapsody.enable_logging(level=logging.DEBUG)


async def main():
    # Get a backend (concurrent backend by default)
    backend = await ConcurrentExecutionBackend()
    session = Session([backend])

    # Define tasks (UIDs auto-generated!)
    tasks = [
        ComputeTask(
            executable="/bin/bash",
            capture_stdio=True,
            arguments=["-c", "echo Hello from task 1 on $HOSTNAME"],
        ),
        ComputeTask(
            executable="/bin/bash",
            capture_stdio=True,
            arguments=["-c", "echo Hello from task 2 on $HOSTNAME"],
        ),
    ]

    # Submit tasks
    await session.submit_tasks(tasks)

    # Wait for all tasks to complete (no manual callback needed!)
    await session.wait_tasks(tasks)

    # Access task results - tasks are updated in-place
    for task in tasks:
        print(f"Task {task.uid} in {task.state} state.")
        print(f"Output: {task.stdout.strip() or task.stderr}")

    # Cleanup
    await backend.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
