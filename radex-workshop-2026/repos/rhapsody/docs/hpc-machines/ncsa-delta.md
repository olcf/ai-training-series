# NCSA Delta

[Delta](https://www.ncsa.illinois.edu/research/project-highlights/delta/) is an NSF-funded GPU and CPU cluster at the National Center for Supercomputing Applications (NCSA). It runs a Cray PE software stack with Cray MPICH as the system MPI library.

## Environment Setup

### 1. Allocate compute nodes

```bash
salloc --nodes=2 --exclusive --account=<your_account> --partition=cpu --tasks-per-node=32
```

### 2. Create a virtual environment using the system Python

Delta provides a system Python through the Cray PE. Use it with `--system-site-packages` so that Cray-optimized libraries (libfabric, PMIx, `mpi4py`, etc.) are inherited inside the venv — no need to rebuild `mpi4py` manually:

```bash
/opt/cray/pe/python/3.11.7/bin/python3 -m venv --system-site-packages ~/ve/rhapsody-cray
```

### 3. Load the Cray MPICH ABI module

```bash
module load cray-mpich-abi
```

!!! note
    `cray-mpich-abi` provides the ABI-compatible MPI libraries required at runtime. Load it before activating the venv.

### 4. Activate the environment

```bash
source ~/ve/rhapsody-cray/bin/activate
```

### 5. Install RHAPSODY with Dragon backend

```bash
pip install "rhapsody-py[dragon]"
```

### 6. Configure Dragon for fast interconnect

This step is required for Dragon to use Cray's high-speed libfabric instead of the default transport:

```bash
dragon-config add --ofi-build-lib=/opt/cray/libfabric/1.22.0/lib64
dragon-config add --ofi-include=/opt/cray/libfabric/1.22.0/include
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
```

!!! warning
    Skipping this step will result in Dragon falling back to a slower transport. Always run it before launching jobs on Delta.

### Run

```bash
dragon <script.py>
```

---

## Example

This example runs 32 MPI jobs concurrently using RHAPSODY's `Session` API with the `DragonExecutionBackend`. Each job gets a random number of ranks (between 2 and 32), all scheduled across the 2 allocated nodes. Worker count is determined automatically by Dragon from the allocation.

```python title="mpi-rhapsody.py"
import asyncio
import logging
import random
import time

import rhapsody
from rhapsody.api import ComputeTask, Session
from rhapsody.backends import DragonExecutionBackend

rhapsody.enable_logging(level=logging.DEBUG)


def mpi_f(i, secs):
    import mpi4py
    mpi4py.rc.initialize = False

    from mpi4py import MPI

    MPI.Init()

    time.sleep(secs)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == (size - 1):
        print(f"Job {i}: Rank {rank} of {size} says: Hello from MPI!", flush=True)


async def main():
    njobs = 32
    sleepsecs = 2
    maxranks = 32

    backend = await DragonExecutionBackend()
    session = Session(backends=[backend])

    print("--- Submitting Tasks ---")

    tasks = []
    for i in range(njobs):
        nranks = random.randint(2, maxranks)
        tasks.append(ComputeTask(
            function=mpi_f,
            args=[i, sleepsecs],
            task_backend_specific_kwargs={'process_templates': [(nranks, {})]}
        ))

    async with session:
        futures = await session.submit_tasks(tasks)
        print(f"Submitted {len(tasks)} tasks. Received {len(futures)} futures.")

        await asyncio.gather(*futures)

        for t in tasks:
            print(f"Task {t.uid}: {t.state} (output: {t.stdout.strip() if t.stdout else 'N/A'})")


if __name__ == "__main__":
    asyncio.run(main())
```

Run with:

```bash
dragon mpi-rhapsody.py
```

!!! success "Output"
    ```text
    2026-03-25 23:07:36,936 | DEBUG    | [asyncio] | Using selector: EpollSelector
    2026-03-25 23:07:36,936 | INFO     | [api_setup] | connecting to infrastructure from 1712232
    2026-03-25 23:07:36,938 | DEBUG    | [api_setup] | waiting for handshake
    2026-03-25 23:07:36,938 | DEBUG    | [api_setup] | Got response GSPingProc: 7
    2026-03-25 23:07:36,938 | DEBUG    | [api_setup] | got handshake
    2026-03-25 23:07:36,938 | INFO     | [api_setup] | debug entry hooked
    2026-03-25 23:07:36,942 | DEBUG    | [dragon.native.queue] | Created queue {self!r}
    2026-03-25 23:07:37,648 | INFO     | [rhapsody.backends.execution.dragon] | DragonExecutionBackend: 128 workers, 2 managers
    2026-03-25 23:07:37,648 | DEBUG    | [rhapsody.backends.execution.dragon] | Starting Dragon backend V3 async initialization...
    2026-03-25 23:07:37,648 | DEBUG    | [rhapsody.backends.execution.dragon] | Registering backend states...
    2026-03-25 23:07:37,649 | DEBUG    | [rhapsody.backends.execution.dragon] | Registering task states...
    2026-03-25 23:07:37,650 | INFO     | [rhapsody.backends.execution.dragon] | Dragon backend V3 fully initialized and ready
    2026-03-25 23:07:37,650 | DEBUG    | [rhapsody.api.session] | Setting up backend callback for'dragon' with Session 'session.0000'
    2026-03-25 23:07:37,650 | DEBUG    | [rhapsody.api.session] | Registered backend 'dragon' with Session 'session.0000'
    2026-03-25 23:07:37,651 | DEBUG    | [rhapsody.backends.execution.dragon] | Backend state set to: RUNNING
    2026-03-25 23:07:37,658 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000001
    2026-03-25 23:07:37,658 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000002
    2026-03-25 23:07:37,658 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000003
    2026-03-25 23:07:37,658 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000004
    2026-03-25 23:07:37,659 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000005
    2026-03-25 23:07:37,659 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000006
    2026-03-25 23:07:37,659 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000007
    2026-03-25 23:07:37,659 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000008
    2026-03-25 23:07:37,660 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000009
    2026-03-25 23:07:37,660 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000010
    2026-03-25 23:07:37,660 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000011
    2026-03-25 23:07:37,660 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000012
    2026-03-25 23:07:37,660 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000013
    2026-03-25 23:07:37,660 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000014
    2026-03-25 23:07:37,660 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000015
    2026-03-25 23:07:37,661 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000016
    2026-03-25 23:07:37,661 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000017
    2026-03-25 23:07:37,661 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000018
    2026-03-25 23:07:37,661 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000019
    2026-03-25 23:07:37,661 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000020
    2026-03-25 23:07:37,661 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000021
    2026-03-25 23:07:37,661 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000022
    2026-03-25 23:07:37,661 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000023
    2026-03-25 23:07:37,662 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000024
    2026-03-25 23:07:37,662 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000025
    2026-03-25 23:07:37,662 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000026
    2026-03-25 23:07:37,662 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000027
    2026-03-25 23:07:37,662 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000028
    2026-03-25 23:07:37,662 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000029
    2026-03-25 23:07:37,662 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000030
    2026-03-25 23:07:37,663 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000031
    2026-03-25 23:07:37,663 | DEBUG    | [rhapsody.backends.execution.dragon] | Created job task: task.000032
    2026-03-25 23:07:37,665 | INFO     | [rhapsody.backends.execution.dragon] | Submitted 32 tasks in a single batch
    2026-03-25 23:07:37,665 | INFO     | [rhapsody.api.session] | Successfully submitted 32 tasks
    2026-03-25 23:07:57,844 | DEBUG    | [rhapsody.backends.execution.dragon] | Batch 53523da5-28c9-11f1-94af-b47af154fc64 complete, processing results
    2026-03-25 23:07:57,851 | DEBUG    | [rhapsody.backends.execution.dragon] | Backend state set to: SHUTDOWN
    2026-03-25 23:07:57,852 | INFO     | [rhapsody.backends.execution.dragon] | Shutting down V3 backend
    2026-03-25 23:07:57,852 | DEBUG    | [rhapsody.backends.execution.dragon] | Waiting for batch monitor thread to stop...
    2026-03-25 23:07:57,856 | DEBUG    | [rhapsody.backends.execution.dragon] | Dragon batch monitor loop stopped
    2026-03-25 23:07:57,856 | DEBUG    | [rhapsody.backends.execution.dragon] | Closing batch...
    2026-03-25 23:07:58,889 | DEBUG    | [rhapsody.backends.execution.dragon] | Batch closed successfully
    2026-03-25 23:07:58,889 | INFO     | [rhapsody.backends.execution.dragon] | Shutdown V3 complete
    --- Submitting Tasks ---
    Submitted 32 tasks. Received 32 futures.
    Job 27: Rank 5 of 6 says: Hello from MPI!
    Job 21: Rank 22 of 23 says: Hello from MPI!
    Job 22: Rank 11 of 12 says: Hello from MPI!
    Job 4: Rank 19 of 20 says: Hello from MPI!
    Job 16: Rank 6 of 7 says: Hello from MPI!
    Job 0: Rank 4 of 5 says: Hello from MPI!
    Job 13: Rank 18 of 19 says: Hello from MPI!
    Job 24: Rank 18 of 19 says: Hello from MPI!
    Job 30: Rank 11 of 12 says: Hello from MPI!
    Job 9: Rank 1 of 2 says: Hello from MPI!
    Job 7: Rank 16 of 17 says: Hello from MPI!
    Job 14: Rank 7 of 8 says: Hello from MPI!
    Job 28: Rank 2 of 3 says: Hello from MPI!
    Job 11: Rank 16 of 17 says: Hello from MPI!
    Job 25: Rank 8 of 9 says: Hello from MPI!
    Job 29: Rank 18 of 19 says: Hello from MPI!
    Job 5: Rank 25 of 26 says: Hello from MPI!
    Job 19: Rank 8 of 9 says: Hello from MPI!
    Job 20: Rank 13 of 14 says: Hello from MPI!
    Job 17: Rank 22 of 23 says: Hello from MPI!
    Job 23: Rank 20 of 21 says: Hello from MPI!
    Job 31: Rank 10 of 11 says: Hello from MPI!
    Job 1: Rank 30 of 31 says: Hello from MPI!
    Job 6: Rank 12 of 13 says: Hello from MPI!
    Job 10: Rank 23 of 24 says: Hello from MPI!
    Job 2: Rank 3 of 4 says: Hello from MPI!
    Job 18: Rank 14 of 15 says: Hello from MPI!
    Job 3: Rank 13 of 14 says: Hello from MPI!
    Job 8: Rank 5 of 6 says: Hello from MPI!
    Job 26: Rank 23 of 24 says: Hello from MPI!
    Job 12: Rank 22 of 23 says: Hello from MPI!
    Job 15: Rank 1 of 2 says: Hello from MPI!
    Task task.000001: DONE (output: Job 0: Rank 4 of 5 says: Hello from MPI!)
    Task task.000002: DONE (output: Job 1: Rank 30 of 31 says: Hello from MPI!)
    Task task.000003: DONE (output: Job 2: Rank 3 of 4 says: Hello from MPI!)
    Task task.000004: DONE (output: Job 3: Rank 13 of 14 says: Hello from MPI!)
    Task task.000005: DONE (output: Job 4: Rank 19 of 20 says: Hello from MPI!)
    Task task.000006: DONE (output: Job 5: Rank 25 of 26 says: Hello from MPI!)
    Task task.000007: DONE (output: Job 6: Rank 12 of 13 says: Hello from MPI!)
    Task task.000008: DONE (output: Job 7: Rank 16 of 17 says: Hello from MPI!)
    Task task.000009: DONE (output: Job 8: Rank 5 of 6 says: Hello from MPI!)
    Task task.000010: DONE (output: Job 9: Rank 1 of 2 says: Hello from MPI!)
    Task task.000011: DONE (output: Job 10: Rank 23 of 24 says: Hello from MPI!)
    Task task.000012: DONE (output: Job 11: Rank 16 of 17 says: Hello from MPI!)
    Task task.000013: DONE (output: Job 12: Rank 22 of 23 says: Hello from MPI!)
    Task task.000014: DONE (output: Job 13: Rank 18 of 19 says: Hello from MPI!)
    Task task.000015: DONE (output: Job 14: Rank 7 of 8 says: Hello from MPI!)
    Task task.000016: DONE (output: Job 15: Rank 1 of 2 says: Hello from MPI!)
    Task task.000017: DONE (output: Job 16: Rank 6 of 7 says: Hello from MPI!)
    Task task.000018: DONE (output: Job 17: Rank 22 of 23 says: Hello from MPI!)
    Task task.000019: DONE (output: Job 18: Rank 14 of 15 says: Hello from MPI!)
    Task task.000020: DONE (output: Job 19: Rank 8 of 9 says: Hello from MPI!)
    Task task.000021: DONE (output: Job 20: Rank 13 of 14 says: Hello from MPI!)
    Task task.000022: DONE (output: Job 21: Rank 22 of 23 says: Hello from MPI!)
    Task task.000023: DONE (output: Job 22: Rank 11 of 12 says: Hello from MPI!)
    Task task.000024: DONE (output: Job 23: Rank 20 of 21 says: Hello from MPI!)
    Task task.000025: DONE (output: Job 24: Rank 18 of 19 says: Hello from MPI!)
    Task task.000026: DONE (output: Job 25: Rank 8 of 9 says: Hello from MPI!)
    Task task.000027: DONE (output: Job 26: Rank 23 of 24 says: Hello from MPI!)
    Task task.000028: DONE (output: Job 27: Rank 5 of 6 says: Hello from MPI!)
    Task task.000029: DONE (output: Job 28: Rank 2 of 3 says: Hello from MPI!)
    Task task.000030: DONE (output: Job 29: Rank 18 of 19 says: Hello from MPI!)
    Task task.000031: DONE (output: Job 30: Rank 11 of 12 says: Hello from MPI!)
    Task task.000032: DONE (output: Job 31: Rank 10 of 11 says: Hello from MPI!)
    ```

    All 32 MPI jobs complete in ~20 seconds wall-clock time across 2 nodes.

!!! note "Key observation"
    Jobs complete out-of-order because each job gets a random rank count — RHAPSODY's Dragon backend schedules them dynamically across available node resources without any manual placement.

---

## Contributors

<div class="grid cards" markdown>

-   :fontawesome-solid-user-tie: **Aymen Alsaadi**

    ---

    RADICAL Lab, Rutgers University

    [:fontawesome-solid-arrow-up-right-from-square: Profile](https://radical.rutgers.edu/people/aymen-alsaadi)

-   :fontawesome-solid-user-tie: **Andrew Park**

    ---

    RADICAL Lab, Rutgers University

    [:fontawesome-solid-arrow-up-right-from-square: Profile](https://radical.rutgers.edu/people/andrew_park)

</div>
