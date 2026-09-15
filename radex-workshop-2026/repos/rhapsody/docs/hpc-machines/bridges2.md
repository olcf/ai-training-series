# PSC Bridges-2

[Bridges-2](https://www.psc.edu/resources/bridges-2/) is an NSF-funded supercomputer at the Pittsburgh Supercomputing Center (PSC). It features GPU and CPU partitions with NVIDIA V100 and A100 GPUs. Unlike Cray-based systems, Bridges-2 runs a standard SLURM scheduler without Cray PE, and Dragon's native SLURM launcher is **not** supported. Instead, Dragon must be launched using the SSH-based network launcher backed by a SLURM-generated host file.

## Environment Setup

### 1. Allocate compute nodes

```bash
salloc --nodes=2 --ntasks-per-node=28 --partition=RM --time=01:00:00 --account=<your_account>
```

For GPU nodes use the `GPU` partition:

```bash
salloc --nodes=2 --ntasks-per-node=8 --gpus-per-node=8 --partition=GPU --time=01:00:00 --account=<your_account>
```

### 2. Create a virtual environment

Bridges-2 provides system Python via modules. Load it first:

```bash
module load python/3.11.5
python3 -m venv ~/ve/rhapsody
source ~/ve/rhapsody/bin/activate
```

### 3. Install RHAPSODY with Dragon backend

```bash
pip install "rhapsody-py[dragon]"
```

### 4. Generate the Dragon network configuration from SLURM

Because Dragon's native SLURM launcher is not supported on Bridges-2, generate
a network configuration YAML from the SLURM-assigned node list instead. Run
this **inside your active SLURM allocation**:

```bash
dragon-network-config --output-to-yaml
```

This reads `$SLURM_NODELIST` and writes a `slurm.yaml` file in the current
directory describing the network topology Dragon will use.

!!! note
    `dragon-network-config` must be run inside a SLURM allocation (after `salloc`
    or inside an `sbatch` script) so that `$SLURM_NODELIST` is populated. Running
    it on the login node produces an empty or single-node configuration.

!!! warning
    Re-run `dragon-network-config --output-to-yaml` each time you request a new
    allocation — SLURM assigns a different set of nodes and the previous
    `slurm.yaml` will reference stale hostnames.

### 5. Run with the SSH-based launcher

Use the `--network-config` flag to pass the generated YAML and `-t tcp` to
select the TCP transport (required on Bridges-2 since Cray libfabric is not
available):

```bash
dragon -w ssh --network-config slurm.yaml -t tcp <script.py>
```

---

## Run

Full launch sequence from inside a SLURM allocation:

```bash
# Step 1 — generate network config from the current allocation
dragon-network-config --output-to-yaml

# Step 2 — run your RHAPSODY script
dragon -w ssh --network-config slurm.yaml -t tcp 01-workload-async-gather.py
```

---

## Example

This example submits a heterogeneous workload mixing function tasks and
executable tasks using RHAPSODY's `Session` API with `DragonExecutionBackend`.

```python title="01-workload-async-gather.py"
import asyncio
import logging

import rhapsody
from rhapsody.api import ComputeTask, Session
from rhapsody.backends import DragonExecutionBackend

rhapsody.enable_logging(level=logging.DEBUG)


def compute(x: int) -> int:
    return x * x


async def main():
    backend = await DragonExecutionBackend()
    session = Session(backends=[backend])

    tasks = [ComputeTask(function=compute, args=(i,)) for i in range(16)]

    async with session:
        futures = await session.submit_tasks(tasks)
        print(f"Submitted {len(tasks)} tasks. Received {len(futures)} futures.")

        await asyncio.gather(*futures)

        for t in tasks:
            print(f"{t.uid}: state={t.state}  return_value={t.return_value}")


if __name__ == "__main__":
    asyncio.run(main())
```

Launch:

```bash
dragon-network-config --output-to-yaml
dragon -w ssh --network-config slurm.yaml -t tcp 01-workload-async-gather.py
```

---

## Notes

### Why SSH launcher instead of SLURM launcher

Dragon supports two multi-node launchers: `slurm` (native SLURM integration,
used on Delta and Perlmutter) and `ssh` (SSH-based, generic). Bridges-2 does
not expose the PMI/PMIx hooks that Dragon's SLURM launcher depends on, so
`-w ssh` is required. The `--network-config` flag bridges the gap by giving
Dragon the node list that it would otherwise obtain from the scheduler.

### TCP transport (`-t tcp`)

Bridges-2 does not have Cray libfabric or a compatible OFI provider. Use
`-t tcp` to select Dragon's TCP transport. Performance will be lower than on
Cray libfabric systems (Delta, Perlmutter), but correctness is unaffected.

### Re-using `slurm.yaml` across runs

`slurm.yaml` is tied to the node names in a specific SLURM allocation. It is
safe to run multiple scripts in the same session without regenerating it, but
it must be regenerated whenever a new `salloc` or `sbatch` is issued.

---

## Contributors

<div class="grid cards" markdown>

-   :fontawesome-solid-user-tie: **Aymen Alsaadi**

    ---

    RADICAL Lab, Rutgers University

    [:fontawesome-solid-arrow-up-right-from-square: Profile](https://radical.rutgers.edu/people/aymen-alsaadi)

</div>
