import asyncio
import json
import logging
import multiprocessing as mp

import rhapsody
from rhapsody.api import ComputeTask
from rhapsody.api import Session
from rhapsody.backends import DragonExecutionBackend

rhapsody.enable_logging(level=logging.DEBUG)


def func_task():
    print("Hello from function task")


async def main():
    # Initialize session with Concurrent backend
    backend = await DragonExecutionBackend()
    session = Session(backends=[backend])

    print("--- Submitting Tasks ---")
    tasks = [ComputeTask(function=func_task) for i in range(1024)]

    async with session:
        # returns futures
        futures = await session.submit_tasks(tasks)

        print(f"Submitted {len(tasks)} tasks. Received {len(futures)} futures.")

        await asyncio.gather(*futures)  # or tasks both works

        for t in tasks:
            print(f"Task {t.uid}: {t.state} (output: {t.stdout.strip() if t.stdout else 'N/A'})")


if __name__ == "__main__":
    asyncio.run(main())
