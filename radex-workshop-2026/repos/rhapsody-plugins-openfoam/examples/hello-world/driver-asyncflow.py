import asyncio
import logging
import time
import multiprocessing as mp

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import DragonExecutionBackend

from radical.asyncflow.logging import init_default_logger

import rhapsody
rhapsody.enable_logging(level=logging.DEBUG)

NUM_RANKS=8

async def main():
    mp.set_start_method("dragon", force=True)
    backend = await DragonExecutionBackend()
    flow = await WorkflowEngine.create(backend)

    @flow.executable_task
    async def mpi_hello_world(task_description={"process_templates": [(NUM_RANKS, {})]}):
        return "./hello_world"

    await mpi_hello_world()

    await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())