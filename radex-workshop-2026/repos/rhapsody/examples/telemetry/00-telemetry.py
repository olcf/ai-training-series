"""Telemetry example — everything the RHAPSODY telemetry system can produce.

Demonstrates:
  - session.enable_telemetry()
  - Real-time event stream via subscribe()
  - OTel metrics snapshot via read_metrics()
  - OTel distributed traces via read_traces()
  - ResourceUpdate events from the ConcurrentTelemetryAdapter (psutil)

Run:
    python examples/06-telemetry.py
"""

import json
import asyncio
import logging


import rhapsody
from rhapsody.api.session import Session
from rhapsody.api.task import ComputeTask
from rhapsody.backends import ConcurrentExecutionBackend

rhapsody.enable_logging(level=logging.DEBUG)

# ── main ───────────────────────────────────────────────────────────────────────


async def main():
    print("=" * 70)
    print("RHAPSODY Telemetry — full output demo")
    print("=" * 70)

    # 1. Build session with telemetry enabled
    backend = await ConcurrentExecutionBackend()
    session = Session(backends=[backend])

    telemetry = await session.start_telemetry(
        resource_poll_interval=0.1, checkpoint_path=f"telemetry-output"
    )

    # 3. Submit workload
    tasks = [ComputeTask(executable="./gpu-cuda-stress") for i in range(10)]

    await session.submit_tasks(tasks)

    await asyncio.gather(*tasks)

    print("=" * 70, flush=True)
    print("RHAPSODY Telemetry — Summary")
    print("=" * 70, flush=True)

    print(json.dumps(telemetry.summary(), indent=4), flush=True)

    print("=" * 70, flush=True)
    print("=" * 70, flush=True)

    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
