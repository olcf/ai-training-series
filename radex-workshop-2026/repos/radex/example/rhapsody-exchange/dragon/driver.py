"""RHAPSODY-launched Dragon DDict exchange, executed via RHAPSODY tasks.

Run with:
    dragon -s -- driver.py

RHAPSODY's DragonDataBackend constructs and owns the `dragon.data.ddict.DDict`
-- this driver never constructs the DDict itself, unlike
../../cpp-exchange/dragon or ../../py-cpp-exchange/dragon. Once the backend
hands back a serialized endpoint, radex takes over exactly as it would
against any other DDict: DragonClient attaches directly from the descriptor.

The producer/consumer exchange itself runs as RHAPSODY ComputeTasks
dispatched through a DragonExecutionBackend, not inline in this driver
process -- each task function below is fully self-contained (its own
imports, its own radex client, built only from the endpoint passed as an
argument) since a task may execute in a completely separate process with
no knowledge of this module or anything else defined here.
"""

import asyncio

from rhapsody.api import ComputeTask, Session
from rhapsody.backends import DragonExecutionBackend
from rhapsody.backends.data import DragonDataBackend


def produce(descriptor):
    import numpy as np

    from radex.clients.core import DragonClient
    from radex.handles.handles import OutgoingHandle

    client = DragonClient(descriptor=descriptor, timeout=5)

    samples = np.arange(10, dtype=np.float64) ** 2
    client.put_tensor(OutgoingHandle("samples"), samples)
    client.put_scalar(OutgoingHandle("sample-count"), len(samples))
    return len(samples)


def consume(descriptor):
    from radex.clients.core import DragonClient
    from radex.handles.handles import IncomingHandle

    client = DragonClient(descriptor=descriptor, timeout=5)

    samples = client.wait_for_tensor(IncomingHandle("samples"), 10)
    count = client.wait_for_scalar(IncomingHandle("sample-count"), 10)
    return {
        "count": int(count),
        "sum": float(samples.sum()),
        "mean": float(samples.mean()),
    }


async def main() -> int:
    print("Driver: Starting Backends", flush=True)
    session = Session()
    exec_backend = await DragonExecutionBackend()
    data_backend = await DragonDataBackend(managers_per_node=1, n_nodes=1)

    session.add_backend(exec_backend)
    session.add_backend(data_backend)

    descriptor = data_backend.endpoints[0].serialize()

    tasks = [
        ComputeTask(function=produce, args=(descriptor,)),
        ComputeTask(function=consume, args=(descriptor,)),
    ]

    print("Driver: Submitting tasks", flush=True)
    futures = await session.submit_tasks(tasks)
    await asyncio.gather(*futures)

    for task in tasks:
        print(f"Driver: Task {task.uid} in {task.state} state.", flush=True)
        print(f"Driver: Output: {task.return_value}", flush=True)

    print("Driver: Shutting down", flush=True)
    await data_backend.shutdown()
    await exec_backend.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
