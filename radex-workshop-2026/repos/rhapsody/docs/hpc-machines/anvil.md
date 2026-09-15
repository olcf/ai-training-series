# Purdue Anvil

[Anvil](https://www.rcac.purdue.edu/compute/anvil) is an NSF-funded supercomputer at Purdue University, available through the ACCESS program. It features a Cray PE software stack with Cray MPICH as the system MPI library, making the setup nearly identical to NCSA Delta.

## Environment Setup

### 1. Allocate compute nodes

```bash
salloc --nodes=2 --exclusive --account=<your_account> --partition=cpu --tasks-per-node=32
```

### 2. Create a virtual environment using the system Python

Anvil provides a system Python through the Cray PE. Use it with `--system-site-packages` so that Cray-optimized libraries (libfabric, PMIx, `mpi4py`, etc.) are inherited inside the venv — no need to rebuild `mpi4py` manually:

```bash
/opt/cray/pe/python/3.11.7/bin/python3 -m venv --system-site-packages rhapsody_env
```

!!! note
    Any Python version available under `/opt/cray/pe/python/` can be used. Replace `3.11.7` with the version installed on your allocation.

### 3. Load the Cray MPICH ABI module

```bash
module load cray-mpich-abi
```

!!! note
    `cray-mpich-abi` provides the ABI-compatible MPI libraries required at runtime. Load it before activating the venv.

### 4. Activate the environment

```bash
source rhapsody_env/bin/activate
```

### 5. Install RHAPSODY with Dragon backend

```bash
pip install "rhapsody-py[dragon]"
```

### 6. Configure Dragon for fast interconnect

This step is required for Dragon to use Cray's high-speed libfabric instead of the default transport:

```bash
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
```

!!! warning
    Skipping this step will result in Dragon falling back to a slower transport and significantly reduced multi-node performance. Always run it once after creating a new environment.

### Run

```bash
dragon <script.py>
```

---

## Example

This example runs 32 MPI jobs concurrently using RHAPSODY's `Session` API with `DragonExecutionBackend`. Each job gets a random number of ranks (between 2 and 32), all scheduled across the 2 allocated nodes. Worker count is determined automatically by Dragon from the allocation.

```python title="rhapsody-mpi.py"
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
dragon rhapsody-mpi.py
```

---

## Contributors

<div class="grid cards" markdown>

-   :fontawesome-solid-user-tie: **Aymen Alsaadi**

    ---

    RADICAL Lab, Rutgers University

    [:fontawesome-solid-arrow-up-right-from-square: Profile](https://radical.rutgers.edu/people/aymen-alsaadi)

</div>
