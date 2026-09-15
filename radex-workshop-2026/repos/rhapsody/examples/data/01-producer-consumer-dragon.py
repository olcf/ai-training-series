"""Dragon counterpart of 00-workload-native-api.py.

Run with:
    dragon -s -- python3 00-workload-native-api-dragon.py
"""

import asyncio
import logging

import rhapsody

from rhapsody.api import ComputeTask
from rhapsody.api import Session
from rhapsody.backends import DragonExecutionBackend
from rhapsody.backends.data import DragonDataBackend

rhapsody.enable_logging(level=logging.INFO)


# NOTE: task functions below are each fully self-contained. A task may be
# invoked in a completely separate process/node with no knowledge of this
# module or anything else defined here -- every import and every bit of
# setup a task needs must live inside that task's own function body, never
# factored into a shared helper or relying on driver-scope state. The only
# input a task gets is whatever is explicitly passed as an argument
# (`descriptor`, here) -- DragonDataBackend hands that back from
# `.start()`, it never constructs a client itself. Unlike Redis, the
# Dragon client takes the descriptor directly as a constructor argument --
# no environment variables involved.


# func1 (producer) and func2 (consumer) are submitted together, with no
# ordering guarantee between them -- func2 uses wait_for_*, not get_*, so
# it correctly blocks until func1's data actually lands instead of racing
# it.


def func1(descriptor):
    import numpy as np

    from radex.clients.core import DragonClient
    from radex.handles.handles import OutgoingHandle

    client = DragonClient(descriptor=descriptor, timeout=5)

    samples = np.arange(10, dtype=np.float64) ** 2  # [0, 1, 4, 9, ..., 81]
    client.put_tensor(OutgoingHandle("samples"), samples)
    client.put_scalar(OutgoingHandle("sample-count"), len(samples))
    return len(samples)


def func2(descriptor):
    from radex.clients.core import DragonClient
    from radex.handles.handles import IncomingHandle

    client = DragonClient(descriptor=descriptor, timeout=5)

    samples = client.wait_for_tensor(IncomingHandle("samples"), 10)
    count = client.wait_for_scalar(IncomingHandle("sample-count"), 10)
    return {"count": int(count), "sum": float(samples.sum()), "mean": float(samples.mean())}


async def main():
    # RHAPSODY owns launching the Dragon DDict; RADEX only ever sees the
    # resulting endpoint, never the launch mechanism.
    data_backend = await DragonDataBackend(managers_per_node=1, n_nodes=1)
    exec_backend = await DragonExecutionBackend()

    session = Session([exec_backend, data_backend])

    descriptor = data_backend.endpoints[0].serialize()

    # Define tasks (UIDs auto-generated!)
    tasks = [
        ComputeTask(function=func1, args=(descriptor,)),
        ComputeTask(function=func2, args=(descriptor,)),
    ]

    # Submit tasks
    futures = await session.submit_tasks(tasks)

    # Wait for all tasks to complete (no manual callback needed!)
    results = await asyncio.gather(*futures)

    # Access task results - tasks are updated in-place
    for task in tasks:
        print(f"Task {task.uid} in {task.state} state.")
        print(f"Output: {task.return_value}")

    # Cleanup
    await data_backend.shutdown()
    await exec_backend.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
