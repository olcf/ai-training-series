import asyncio
import multiprocessing as mp
import os

from rhapsody import Session
from rhapsody.backends import DragonExecutionBackend
from rhapsody.api import ComputeTask

NUM_RANKS=8

async def main():

    mp.set_start_method("dragon", force=True)

    backend = await DragonExecutionBackend()

    async with Session(backends=[backend]) as session:

      task = ComputeTask(
         executable="./hello_world",
         task_backend_specific_kwargs={
            "process_templates": [(NUM_RANKS, {"env": {**os.environ}})]
          }
      )
      await session.submit_tasks([task])
      await task

if __name__ == "__main__":
    asyncio.run(main())
