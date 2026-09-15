"""Two independent vLLM inference services, driven both directly (AITask) and over HTTP (ComputeTask
+ aiohttp).

Where 03-workload-ai-hpc.py submits AITask objects to a single
DragonVllmInferenceBackend, this example runs *two* services at once and
exercises both ways of talking to them:

* AITask — submitted straight to a DragonVllmInferenceBackend by name;
  RHAPSODY routes it internally, no HTTP involved (same as 03).
* ComputeTask(function=run_inference) — a plain compute function that POSTs a
  batch of N prompts to a service's /generate endpoint over aiohttp. This is
  the shape you'd use to drive vLLM from code that doesn't have direct access
  to the Dragon queues (e.g. a separate client process), or to fan work out
  across more than one inference service in the same allocation.

Requires: GPU access, rhapsody-py[ai], and a locally
downloaded model directory (see the migration notes on HF_HUB_OFFLINE for why
a local path is used instead of a bare HF hub ID).
"""

import asyncio
import logging
import multiprocessing as mp
import os
import time

import aiohttp

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


async def run_inference(prompts, endpoint):
    """ComputeTask function: POST a batch of prompts to a vLLM service's /generate endpoint."""
    start = time.time()
    print(f"START {endpoint} ({len(prompts)} prompts) at {start}", flush=True)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{endpoint}/generate",
            json={"prompts": prompts, "timeout": 300},
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            data = await resp.json()

    duration = time.time() - start
    print(f"END {endpoint} at {time.time()}, duration={duration:.2f}s", flush=True)

    if data.get("status") != "success":
        raise RuntimeError(f"Service error from {endpoint}: {data.get('message', 'Unknown error')}")

    return data["results"]


async def _make_vllm_backend(name: str, port: int, node_offset: int) -> DragonVllmInferenceBackend:
    """Build and initialize one single-node, single-GPU vLLM service at a disjoint node slice.

    node_offset gives each service its own node within the allocation — see
    the Dragon AI inference cookbook, "Example 5 - Two Services in One
    Allocation".
    """
    return await DragonVllmInferenceBackend(
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",  # or the snapshot path on disk
            hf_token="",
            tp_size=1,
            gpu_memory_utilization=0.65,
            vllm_log_level="debug",
        ),
        hardware=HardwareConfig(num_nodes=1, num_gpus=1, node_offset=node_offset),
        port=port,
        use_service=True,
        name=name,
    )


async def main():
    mp.set_start_method("dragon")

    execution_backend = await DragonExecutionBackend()

    logger.info("Initializing 2 vLLM services concurrently...")
    inference_backend_1, inference_backend_2 = await asyncio.gather(
        _make_vllm_backend("vllm-1", port=8001, node_offset=0),
        _make_vllm_backend("vllm-2", port=8002, node_offset=1),
    )

    endpoint_1 = inference_backend_1.get_endpoint()
    endpoint_2 = inference_backend_2.get_endpoint()
    print(f"Service 1 endpoint: {endpoint_1}", flush=True)
    print(f"Service 2 endpoint: {endpoint_2}", flush=True)

    prompts_batch_1 = [f"What is 1 + {i}?" for i in range(1, 513)]
    prompts_batch_2 = [f"What is 1 * {i}?" for i in range(1, 513)]

    tasks = [
        # Direct submission: RHAPSODY routes these to each backend internally,
        # no HTTP involved — same pattern as 03-workload-ai-hpc.py.
        AITask(prompt="What is the capital of France?", backend=inference_backend_1.name),
        AITask(prompt=["Tell me a joke", "What is 2+2?"], backend=inference_backend_2.name),
        # Service/HTTP submission: a compute function drives each backend's
        # /generate endpoint over aiohttp, one N-prompt batch per service.
        ComputeTask(
            function=run_inference,
            args=(prompts_batch_1, endpoint_1),
            backend=execution_backend.name,
        ),
        ComputeTask(
            function=run_inference,
            args=(prompts_batch_2, endpoint_2),
            backend=execution_backend.name,
        ),
        ComputeTask(
            executable="/usr/bin/echo",
            arguments=["Hello from Dragon backend!"],
            backend=execution_backend.name,
        ),
    ]

    session = Session([execution_backend, inference_backend_1, inference_backend_2])

    print(f"Submitting {len(tasks)} tasks via Session...", flush=True)
    await session.submit_tasks(tasks)

    results = await asyncio.gather(*tasks)

    for i, task in enumerate(results):
        backend_name = task.get("backend")
        if isinstance(task, AITask):
            print(f"Task {i + 1} [AI] ({backend_name}): {task.response}", flush=True)
        elif task.get("function"):
            return_value = task.return_value
            summary = (
                f"{len(return_value)} results" if isinstance(return_value, list) else return_value
            )
            print(f"Task {i + 1} [Compute/fn] ({backend_name}): {summary}", flush=True)
        else:
            print(
                f"Task {i + 1} [Compute/exec] ({backend_name}): "
                f"stdout={task.stdout.strip()!r}  exit_code={task.exit_code}",
                flush=True,
            )

    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
