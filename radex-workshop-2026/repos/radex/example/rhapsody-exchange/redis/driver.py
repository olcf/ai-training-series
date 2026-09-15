"""RHAPSODY-launched Redis store exchange, executed via RHAPSODY tasks.

RHAPSODY's RedisDataBackend launches and owns the `redis-server` process --
this driver never spawns it directly, unlike ../../cpp-exchange/redis
(SmartSim) or a raw `dragon.data.ddict.DDict(...)` construction. Once the
backend hands back a serialized endpoint, radex takes over exactly as it
would against any other Redis deployment: RedisClient reads its connection
info from RADEX_STORE/RADEX_STORE_OPTS.

The producer/consumer exchange itself runs as RHAPSODY ComputeTasks
dispatched through a ConcurrentExecutionBackend, not inline in this driver
process -- each task function below is fully self-contained (its own
imports, its own radex client, built only from the endpoint passed as an
argument) since a task may execute in a completely separate process with
no knowledge of this module or anything else defined here.
"""

import asyncio
import os
import time

from rhapsody.api import ComputeTask, Session
from rhapsody.backends import ConcurrentExecutionBackend
from rhapsody.backends.data import RedisDataBackend


def produce(descriptor):
    import os

    import numpy as np

    from radex.clients.core import RedisClient
    from radex.handles.handles import OutgoingHandle

    os.environ["RADEX_STORE"] = descriptor
    os.environ["RADEX_STORE_OPTS"] = "Standalone"
    client = RedisClient()

    samples = np.arange(10, dtype=np.float64) ** 2
    client.put_tensor(OutgoingHandle("samples"), samples)
    client.put_scalar(OutgoingHandle("sample-count"), len(samples))
    return len(samples)


def consume(descriptor):
    import os

    from radex.clients.core import RedisClient
    from radex.handles.handles import IncomingHandle

    os.environ["RADEX_STORE"] = descriptor
    os.environ["RADEX_STORE_OPTS"] = "Standalone"
    client = RedisClient()

    time.sleep(2)
    samples = client.wait_for_tensor(IncomingHandle("samples"), 10)
    count = client.wait_for_scalar(IncomingHandle("sample-count"), 10)
    return {
        "count": int(count),
        "sum": float(samples.sum()),
        "mean": float(samples.mean()),
    }


async def main() -> int:
    # Session is constructed first so its work_dir/uid exist before
    # RedisDataBackend launches redis-server -- that lets the server's log
    # file land inside the session's own directory.

    print("Driver: Starting Backends", flush=True)
    session = Session(uid="radex.session.0000")

    data_backend = await RedisDataBackend(
        work_dir=os.path.join(session.work_dir, session.uid)
    )

    exec_backend = await ConcurrentExecutionBackend()

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
    await session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
