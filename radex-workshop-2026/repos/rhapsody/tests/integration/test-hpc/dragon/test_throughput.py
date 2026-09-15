import argparse
import asyncio
import time

import rhapsody
from rhapsody.api import ComputeTask
from rhapsody.api import Session
from rhapsody.backends import DragonExecutionBackend

width_num_tasks = 10
width_throughput = 20


def no_op():
    pass


async def run_tasks(session: Session, num_tasks: int, warmup: bool = False, backend=None) -> None:
    start_time = time.time()

    tasks = [ComputeTask(function=no_op) for _ in range(num_tasks)]
    futures = await session.submit_tasks(tasks)

    backend.wait()  # high performance wait loop

    end_time = time.time()

    if not warmup:
        runtime = end_time - start_time
        throughput = num_tasks / runtime
        print(
            f"{num_tasks:<{width_num_tasks}} {throughput:<{width_throughput}}",
            flush=True,
        )

    backend.batch.clear_results()


async def run_bench(session: Session, min_tasks: int, max_tasks: int, backend=None) -> None:
    print("task throughput benchmark", flush=True)
    print("-------------------------", flush=True)

    await run_tasks(session, min_tasks, warmup=True, backend=backend)

    num_tasks_str = "num tasks"
    throughput_str = "throughput [tasks/s]"
    print(
        f"{num_tasks_str.ljust(width_num_tasks)} {throughput_str.ljust(width_throughput)}",
        flush=True,
    )

    num_tasks = min_tasks

    while num_tasks <= max_tasks:
        await run_tasks(session, num_tasks, backend=backend)
        num_tasks *= 2

    print("", flush=True)


async def main(min_tasks: int, max_tasks: int) -> None:
    backend = await DragonExecutionBackend(batch_kwargs={"task_logs": False})
    async with Session(backends=[backend]) as session:
        await run_bench(session, min_tasks, max_tasks, backend=backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task throughput benchmark")
    parser.add_argument("--min_tasks", type=int, default=4, help="minimum number of tasks to run")
    parser.add_argument("--max_tasks", type=int, default=128, help="maximum number of tasks to run")
    args = parser.parse_args()

    if args.max_tasks < args.min_tasks:
        args.max_tasks = args.min_tasks

    asyncio.run(main(args.min_tasks, args.max_tasks))
