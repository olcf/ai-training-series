import asyncio
import logging
import multiprocessing as mp
import os

import rhapsody
from rhapsody.api import AITask
from rhapsody.api import ComputeTask
from rhapsody.api import Session
from rhapsody.backends import DragonExecutionBackend
from rhapsody.backends import DragonVllmInferenceBackend
from rhapsody.backends.ai.config import HardwareConfig
from rhapsody.backends.ai.config import ModelConfig

rhapsody.enable_logging(level=logging.DEBUG)

logger = logging.getLogger(__name__)


async def main():
    mp.set_start_method("dragon")

    execution_backend = await DragonExecutionBackend()

    inference_backend = await DragonVllmInferenceBackend(
        model=ModelConfig(
            tp_size=1,
            vllm_log_level="debug",
            gpu_memory_utilization=0.85,
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hf_token=os.environ.get("HF_TOKEN", ""),
        ),
        hardware=HardwareConfig(
            num_nodes=2,  # total number of nodes requested to be occupied by the engine
            num_gpus=2,  # this corresponds to --gpus-per-node and not total gpus
            node_offset=0,  # Change this to control the number of nodes each inference pipeline takes
        ),
    )

    # Define multiple tasks with single or multiple prompts
    # Note: Explicit backend mapping by user
    tasks = [
        AITask(prompt="What is the capital of France?", backend=inference_backend.name),
        AITask(prompt=["Tell me a joke", "What is 2+2?"], backend=inference_backend.name),
        ComputeTask(
            executable="/usr/bin/echo",
            arguments=["Hello from Dragon backend!"],
            backend=execution_backend.name,
        ),
    ]

    session = Session([execution_backend, inference_backend])

    # Submit all tasks at once via session - they will be routed correctly!
    print(f"Submitting {len(tasks)} mixed tasks via Session...")
    await session.submit_tasks(tasks)

    # Gather results using standard asyncio
    results = await asyncio.gather(*tasks)

    for i, task in enumerate(results):
        backend_name = task.get("backend")
        if isinstance(task, AITask):
            # AITask: model response is in task.response
            print(f"Task {i + 1} [AI] ({backend_name}): {task.response}", flush=True)
        elif task.get("function"):
            # ComputeTask (function): return value is in task.return_value
            print(f"Task {i + 1} [Compute/fn] ({backend_name}): {task.return_value}", flush=True)
        else:
            # ComputeTask (executable): output is in task.stdout / task.stderr
            print(
                f"Task {i + 1} [Compute/exec] ({backend_name}): "
                f"stdout={task.stdout.strip()!r}  exit_code={task.exit_code}",
                flush=True,
            )

    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
