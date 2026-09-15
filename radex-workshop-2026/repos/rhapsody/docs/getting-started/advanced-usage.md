# Advanced Usage

RHAPSODY excels at orchestrating complex, multi-backend workloads. This guide covers wait modes, sync/async callable tasks, explicit task routing, and mixing AI inference with traditional compute.

---

## Wait Modes

After calling `submit_tasks`, RHAPSODY gives you three ways to wait for results.
They differ in whether task failures raise exceptions or are left for you to inspect.

### `await session.wait_tasks(tasks)` — Collect all, inspect states

The session-level helper. It waits for **all** listed tasks to reach a terminal
state without raising on task failures. Inspect `task.state` afterwards.

```python
async with Session(backends=[backend]) as session:
    await session.submit_tasks(tasks)
    await session.wait_tasks(tasks, timeout=30.0)  # (1)

    for t in tasks:
        if t.state == "DONE":
            print(t.return_value or t.stdout)
        else:
            print(f"{t.uid} failed: {t.exception or t.stderr}")
```

1. Raises `asyncio.TimeoutError` if the deadline is exceeded; never raises on task failure.

### `await asyncio.gather(*futures)` — Concurrent, raises on first failure

Use the futures returned by `submit_tasks`. By default `asyncio.gather` re-raises
the first exception it encounters. Use `return_exceptions=True` to collect all
outcomes without raising.

```python
from rhapsody.api.errors import TaskExecutionError

async with Session(backends=[backend]) as session:
    futures = await session.submit_tasks(tasks)

    # Fail-fast: raises on the first failure
    try:
        await asyncio.gather(*futures)
    except TaskExecutionError as e:          # command task (executable)
        print(f"Command failed: {e}")
    except Exception as e:                   # function task raised
        print(f"Function raised: {e}")

    # --- OR --- collect all, decide per-result
    results = await asyncio.gather(*futures, return_exceptions=True)
    for t, r in zip(tasks, results):
        if isinstance(r, Exception):
            print(f"{t.uid} → error: {r}")
        else:
            print(f"{t.uid} → {t.return_value}")
```

### `await future` / `await task` — Wait for one specific task

Both the `asyncio.Future` objects returned by `submit_tasks` and the task
objects themselves are awaitable. Awaiting raises if that task failed.

```python
async with Session(backends=[backend]) as session:
    futures = await session.submit_tasks([task_a, task_b])

    result_a = await futures[0]   # raises if task_a failed
    result_b = await task_b       # task objects are awaitable too
```

This is ideal for sequential pipelines where the output of one task is the
input of the next.

### Quick reference

| Mode | Raises on task failure | Raises on timeout | Best for |
|---|---|---|---|
| `await session.wait_tasks(tasks)` | No — check `task.state` | Yes | Batch / fire-and-inspect |
| `await asyncio.gather(*futures)` | Yes (first exception) | No | Fail-fast / concurrent |
| `await future` / `await task` | Yes | No | Sequential pipelines |

### Exception types

| Situation | Exception raised |
|---|---|
| Executable task fails (non-zero exit code) | `TaskExecutionError(uid, stderr, exit_code)` |
| Function task raises | Original exception propagated as-is |
| Task not yet submitted | `RuntimeError` — no bound future |
| Timeout in `wait_tasks` | `asyncio.TimeoutError` |

All RHAPSODY-specific exceptions inherit from `RhapsodyError` and can be
imported from `rhapsody.api.errors`:

```
RhapsodyError
├── BackendError        — backend infrastructure failure
├── TaskValidationError — invalid task definition
├── TaskExecutionError  — task ran but failed (command or function)
├── SessionError        — session-level misuse
└── ResourceError       — unsatisfiable resource requirements
```

---

## Sync and Async Callable Tasks

`ComputeTask` accepts **any Python callable** — both regular (synchronous) and
`async def` functions are dispatched transparently. RHAPSODY detects the
function type at runtime; no changes to your task definition are needed.

```python
# Synchronous function
def compute_pi(n_samples):
    import random
    hits = sum(1 for _ in range(n_samples)
               if random.random()**2 + random.random()**2 < 1)
    return 4 * hits / n_samples

# Async function
async def fetch_result(url):
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            return await r.text()

async with Session(backends=[backend]) as session:
    tasks = [
        ComputeTask(function=compute_pi, args=(1_000_000,)),
        ComputeTask(function=fetch_result, args=("https://example.com",)),
    ]
    futures = await session.submit_tasks(tasks)
    await asyncio.gather(*futures)

    print(tasks[0].return_value)   # ~3.1415...
    print(tasks[1].return_value)   # HTML string
```

Both functions run inside the executor (thread or process pool). Async
functions are driven by an isolated `asyncio.run(...)` call inside the worker
so they have their own event loop and do not share the session's loop.

### Executor dispatch

| Backend | Executor | Sync function | Async function |
|---|---|---|---|
| `ConcurrentExecutionBackend` (default) | `ThreadPoolExecutor` | called directly | run via `asyncio.run` |
| `ConcurrentExecutionBackend` | `ProcessPoolExecutor` | called directly | run via `asyncio.run` |
| `DaskExecutionBackend` | Dask workers | submitted natively | wrapped transparently |
| `OrbitExecutionBackend` | remote ORBIT endpoint | shipped via `cloudpickle` | shipped via `cloudpickle` |

!!! note "ProcessPoolExecutor requires cloudpickle"
    Functions are serialised with `cloudpickle` before being sent to the worker
    process. Both sync and async callables are supported:
    ```bash
    pip install cloudpickle
    ```

### Result access

After a task completes, results are stored on the task object:

| Task type | Field | Description |
|---|---|---|
| Function task (`function=`) | `task.return_value` | the callable's return value |
| Function task | `task.stdout` | `str(return_value)` |
| Executable task (`executable=`) | `task.stdout` | captured standard output |
| Executable task | `task.stderr` | captured standard error |
| Executable task | `task.exit_code` | process exit code |
| AI task (`AITask`) | `task.response` | model response string |

---

## Capturing stdout/stderr to Files

By default, output from executable tasks is collected in-memory as strings on
`task.stdout` and `task.stderr`. For long-running tasks or large outputs, use
`capture_stdio=True` to redirect output directly to files — zero in-memory
buffering.

```python
from rhapsody.api import ComputeTask, Session
from rhapsody.backends import ConcurrentExecutionBackend

async with ConcurrentExecutionBackend() as backend:
    session = Session(backends=[backend], work_dir="/tmp/my-run")

    tasks = [
        ComputeTask(
            executable="/bin/bash",
            arguments=["-c", "echo hello; echo err >&2"],
            capture_stdio=True,
        )
        for _ in range(10)
    ]

    async with session:
        await session.submit_tasks(tasks)
        await session.wait_tasks(tasks)

    for t in tasks:
        print(t.stdout)   # path: /tmp/my-run/session.0000/task.000001.stdout
        print(open(t.stdout).read())
```

When `capture_stdio=True`, `task.stdout` and `task.stderr` hold **file paths**
instead of strings. The files are written to `{work_dir}/{session_uid}/` and
named after the task UID (`task.000001.stdout`, `task.000001.stderr`).

| Behaviour | `capture_stdio=False` (default) | `capture_stdio=True` |
|---|---|---|
| `task.stdout` | decoded string | file path (`*.stdout`) |
| `task.stderr` | decoded string | file path (`*.stderr`) |
| Memory usage | O(output size) | O(1) |
| Log persistence | lost after process | file on disk |

!!! note "Supported backends"
    `capture_stdio` is supported by `ConcurrentExecutionBackend`,
    `DaskExecutionBackend`, and `DragonExecutionBackend`. On
    `ConcurrentExecutionBackend` and `DaskExecutionBackend`, function tasks
    (`function=`) are unaffected by this flag. On `DragonExecutionBackend`,
    `capture_stdio` applies to **all** task types including function tasks — see
    [Dragon: Task Output and File Paths](#dragon-task-output-and-file-paths).

!!! note "Session work_dir"
    Files are placed in `{session.work_dir}/{session.uid}/`. Pass `work_dir=`
    when creating the session to control where logs land.

---

## Multiple Execution Backends

You can register multiple backends in a single session and explicitly route tasks to them using the `backend` keyword.


!!! note "Running Command"
    Using Dragon backend requires starting the entire example with `dragon` instead of `python`


```python
import asyncio

import rhapsody
from rhapsody.api import Session
from rhapsody.api import ComputeTask
from rhapsody.backends import ConcurrentExecutionBackend, DragonExecutionBackend


async def compute_function():
    import platform
    print(f"Running on host: {platform.node()}")

async def multi_backend_demo():
    # Initialize two different backends
    be_local = ConcurrentExecutionBackend(name="local_cpu")
    be_remote = DragonExecutionBackend(name="hpc_gpu")
    await asyncio.gather(be_local, be_remote)


    async with Session(backends=[be_local, be_remote]) as session:
        # Route tasks explicitly by backend name
        task1 = ComputeTask(
            function=compute_function, backend=be_local.name)
        task2 = ComputeTask(
            function=compute_function, backend=be_remote.name)

        await session.submit_tasks([task1, task2])
        await asyncio.gather(task1, task2)

        print(f"Task 1 ran on: {task1.backend}")
        print(f"Task 2 ran on: {task2.backend}")


if __name__ == "__main__":
    asyncio.run(multi_backend_demo())
```


!!! success "Output"
    If everything is set up correctly, you should see:
    ```text
    Running on host: alienware
    Running on host: alienware
    Task 1 ran on: local_cpu
    Task 2 ran on: hpc_gpu
    +++ head proc exited, code 0

    real    0m5.729s
    user    0m9.789s
    sys     0m3.921s
    ```



!!! tip "Implicit Routing"
    If a task does not specify a `backend`, RHAPSODY will automatically route it to the first compatible backend registered in the session.



## HPC Workloads with Dragon

For large-scale HPC deployments, the Dragon backend provides native integration with the Dragon runtime, optimized for high-performance process management and communication.

```python
import asyncio
import rhapsody
from rhapsody.api import Session, ComputeTask
from rhapsody.backends import DragonExecutionBackend

async def dragon_demo():
    # Initialize the Dragon backend (optimized for HPC)
    # V3 uses native Dragon Batch for high-performance wait/callbacks
    dragon_be = await DragonExecutionBackend(
        name="dragon_hpc"
    )

    async with Session(backends=[dragon_be]) as session:
        # Define tasks for the Dragon cluster
        tasks = [
            ComputeTask(
                executable="hostname", backend=dragon_be.name)
            for _ in range(16)
        ]

        await session.submit_tasks(tasks)
        results = await asyncio.gather(*tasks)

        for t in tasks:
            print(f"Task {t.uid} finished on Dragon")
if __name__ == "__main__":
    asyncio.run(dragon_demo())
```

!!! important "Dragon Environment"
    The Dragon backend requires a running Dragon runtime. Typically, you would start your script using the `dragon` launcher:
    ```bash
    dragon my_workflow.py
    ```

## Dragon Process Templates

When using `DragonExecutionBackend`, you can pass Dragon-native process configuration to `ComputeTask` via the `task_backend_specific_kwargs` parameter. This exposes the `process_template` and `process_templates` options, which map directly to Dragon's [ProcessTemplate](https://dragonhpc.github.io/dragon/doc/_build/html/ref/native/dragon.native.process.html#dragon.native.process.ProcessTemplate).

The `ProcessTemplate` accepts the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `cwd` | `str`, optional | Current working directory, defaults to `"."` |
| `env` | `dict`, optional | Environment variables to pass to the process |
| `stdin` | `int`, optional | Standard input file handling (`PIPE`, `None`) |
| `stdout` | `int`, optional | Standard output file handling (`PIPE`, `STDOUT`, `None`) |
| `stderr` | `int`, optional | Standard error file handling (`PIPE`, `STDOUT`, `None`) |
| `policy` | `Policy`, optional | Determines the placement and resources of the process |
| `options` | `ProcessOptions`, optional | Process options, such as allowing the process to connect to the infrastructure |

!!! warning "Excluded Parameters"
    `DragonExecutionBackend` manages `target` (the binary or Python callable), `args`, and `kwargs` internally through the `ComputeTask` interface. These three parameters are **not** available in `process_template` / `process_templates` — use `ComputeTask.executable`, `ComputeTask.arguments`, and `ComputeTask.function` instead.

### Single-Process Template

Use `process_template` (singular) to apply a Dragon process configuration to a single task:

```python
# Single-process executable with a process template
ComputeTask(
    executable='/bin/bash',
    arguments=['-c', 'echo $HOSTNAME'],
    task_backend_specific_kwargs={
        "process_template": {}
    },
)

# Single-process function with a process template
ComputeTask(
    function=my_function,
    task_backend_specific_kwargs={
        "process_template": {}
    },
)
```

### Multi-Process (Parallel Job) Templates

Use `process_templates` (plural) to launch a task as a parallel job composed of multiple process groups. Each entry is a tuple of `(num_processes, template_dict)`:

```python
# Parallel job: 2 groups of 2 processes each (4 total)
ComputeTask(
    executable='/bin/bash',
    arguments=['-c', 'echo $HOSTNAME'],
    task_backend_specific_kwargs={
        "process_templates": [(2, {}), (2, {})]
    },
)

# Parallel function: 2 groups of 2 processes each
ComputeTask(
    function=parallel_function,
    task_backend_specific_kwargs={
        "process_templates": [(2, {}), (2, {})]
    },
)
```

### GPU Affinity with Policies

You can use Dragon's `Policy` to control process placement and GPU affinity. This is useful for pinning each worker to a specific GPU across multiple nodes in a round-robin fashion:

```python
import asyncio
from dragon.infrastructure.policy import Policy

import rhapsody
from rhapsody.api import ComputeTask, Session
from rhapsody.backends import DragonExecutionBackend


from dragon.native.machine import System, Node


def find_gpus():

    all_gpus = []
    # loop through all nodes Dragon is running on
    for huid in System().nodes:
        node = Node(huid)
        # loop through however many GPUs it may have
        for gpu_id in node.gpus:
            all_gpus.append((node.hostname, gpu_id))
    return all_gpus


def make_policies(all_gpus, nprocs=32):
    """Create per-process policies with round-robin GPU assignment."""
    policies = []
    i = 0
    for worker in range(nprocs):
        policies.append(
            Policy(
                placement=Policy.Placement.HOST_NAME,
                host_name=all_gpus[i][0],
                gpu_affinity=[all_gpus[i][1]],
            )
        )
        i += 1
        if i == len(all_gpus):
            i = 0
    return policies


async def main():
    backend = await DragonExecutionBackend()
    session = Session(backends=[backend])

    all_gpus = find_gpus()  # e.g. [("node-0", 0), ("node-0", 1), ("node-1", 0), ...]
    policies = make_policies(all_gpus, nprocs=4)

    async def gpu_work():
        import socket
        return socket.gethostname()

    # Single-process task pinned to a specific GPU
    task_single = ComputeTask(
        function=gpu_work,
        task_backend_specific_kwargs={
            "process_template": {"policy": policies[0]}
        },
    )

    # Parallel job: 4 processes in two groups, each group with its own GPU policy
    task_parallel = ComputeTask(
        function=gpu_work,
        task_backend_specific_kwargs={
            "process_templates": [
                (2, {"policy": policies[0]}),
                (2, {"policy": policies[1]}),
            ]
        },
    )

    async with session:
        await session.submit_tasks([task_single, task_parallel])
        await asyncio.gather(task_single, task_parallel)

        for t in [task_single, task_parallel]:
            print(f"Task {t.uid}: {t.state} (output: {t.return_value})")


if __name__ == "__main__":
    asyncio.run(main())
```

In the example above, `make_policies` assigns GPUs round-robin across all discovered nodes. Each policy is then passed into the `process_template` (or individual entries in `process_templates`) via the `policy` key, giving you fine-grained control over where each process runs and which GPU it uses.

### Full Heterogeneous Workload Example

The following example demonstrates mixing native functions, single-process tasks, and parallel jobs in a single session:

```python
import asyncio
import rhapsody
from rhapsody.api import ComputeTask, Session
from rhapsody.backends import DragonExecutionBackend


async def main():
    backend = await DragonExecutionBackend()
    session = Session(backends=[backend])

    async def single_function():
        import socket
        return socket.gethostname()

    async def parallel_function():
        import socket
        return socket.gethostname()

    async def native_function():
        import socket
        return socket.gethostname()

    tasks = [
        # Native function (no backend-specific config)
        ComputeTask(function=native_function),

        # Single-process executable
        ComputeTask(
            executable='/bin/bash',
            arguments=['-c', 'echo $HOSTNAME'],
            task_backend_specific_kwargs={
                "process_template": {}
            },
        ),

        # Parallel job executable (2+2 processes)
        ComputeTask(
            executable='/bin/bash',
            arguments=['-c', 'echo $HOSTNAME'],
            task_backend_specific_kwargs={
                "process_templates": [(2, {}), (2, {})]
            },
        ),

        # Single-process function
        ComputeTask(
            function=single_function,
            task_backend_specific_kwargs={
                "process_template": {}
            },
        ),

        # Parallel job function (2+2 processes)
        ComputeTask(
            function=parallel_function,
            task_backend_specific_kwargs={
                "process_templates": [(2, {}), (2, {})]
            },
        ),
    ]

    async with session:
        futures = await session.submit_tasks(tasks)
        await asyncio.gather(*futures)

        for t in tasks:
            result = t.return_value if t.function else t.stdout
            print(f"Task {t.uid}: {t.state} (output: {result})")

if __name__ == "__main__":
    asyncio.run(main())
```

!!! note "Running Command"
    This example requires the Dragon runtime:
    ```bash
    dragon my_heterogeneous_workload.py
    ```

## Dragon: Task Output and File Paths

After a task completes, `task.stdout` and `task.stderr` work as follows:

- **`capture_stdio=False` (default)** — content is an inline string (whatever the task printed). Read it directly: `task.stdout`.
- **`capture_stdio=True`** — content is a file path under the Rhapsody session directory (`rhapsody.session.{uid}/task.000001.stdout`). Open it to read: `open(task.stdout).read()`.

!!! note "Performance at scale"
    At 100K+ tasks on shared filesystems, the default file-read overhead (~100–400 s on Lustre) can be avoided by passing `batch_kwargs={"task_logs": False}` to `DragonExecutionBackend`. With this option, `task.stdout` and `task.stderr` will be **empty** for all `capture_stdio=False` tasks — output goes to the console only. A warning is logged when this option is active.

!!! warning "Functions launched via process_template or process_templates"
    If you use `function=` together with `process_template` or `process_templates`,
    Dragon runs the callable as a subprocess — **not** in Dragon's native function pool.
    In this mode `task.return_value` is the **exit code** (`0` on success), not the
    Python return value of your function. Use a bare `ComputeTask(function=my_fn)` (no
    templates) when you need the actual return value.

---

## Mixed HPC and AI Workloads

One of RHAPSODY's most powerful features is the ability to orchestrate traditional binaries alongside AI inference services (like vLLM).

```python
import asyncio
import rhapsody
from rhapsody.api import Session, ComputeTask, AITask
from rhapsody.backends import ConcurrentExecutionBackend
from rhapsody.backends import DragonVllmInferenceBackend
from rhapsody.backends.ai.config import ModelConfig

async def mixed_workload():
    # 1. Setup Compute (HPC) and Inference (AI) backends
    hpc_backend = await ConcurrentExecutionBackend(name="hpc_cluster")
    ai_backend = await DragonVllmInferenceBackend(
        name="vllm_service",
        model=ModelConfig(model_name="llama-3", hf_token="...", tp_size=1),
    )

    async with Session(backends=[hpc_backend, ai_backend]) as session:

        # 2. Define a ComputeTask (e.g., simulation)
        sim_task = ComputeTask(executable="./simulate.sh", backend="hpc_cluster")

        # 3. Define an AITask (e.g., summarize results)
        summary_task = AITask(
            prompt="Summarize the simulation findings: ...",
            backend="vllm_service"
        )

        # 4. Submit both!
        await session.submit_tasks([sim_task, summary_task])

        # Wait for simulation to finish first
        await sim_task
        print(f"Simulation Done: {sim_task.stdout}")  # executable ComputeTask: use .stdout

        # Then summarize
        await summary_task
        print(f"AI Summary: {summary_task.response}")  # AITask: use .response

asyncio.run(mixed_workload())
```

## Dask Distributed Backend

`DaskExecutionBackend` runs tasks on any Dask cluster and is **cluster-agnostic** — the same task code works across `LocalCluster`, `SLURMCluster`, `KubeCluster`, and any other Dask-compatible cluster.

### Supported task types

All three `ComputeTask` types are supported:

```python
import asyncio
from rhapsody.api import ComputeTask, Session
from rhapsody.backends import DaskExecutionBackend

# Sync function — dispatched natively to Dask workers
def compute_square(n):
    return n * n

# Async function — wrapped transparently, name visible in Dask dashboard
async def fetch_data(n):
    await asyncio.sleep(0.01)
    return n * 2

async def run():
    async with DaskExecutionBackend() as backend:
        session = Session(backends=[backend])
        tasks = [
            ComputeTask(function=compute_square, args=(i,)),       # sync fn
            ComputeTask(function=fetch_data, args=(i,)),           # async fn
            ComputeTask(executable="/bin/echo", arguments=[f"{i}"]), # executable
        ]
        async with session:
            await session.submit_tasks(tasks)
            await session.wait_tasks(tasks)

        for t in tasks:
            result = t.return_value if t.function else t.stdout
            print(f"{t.uid}: {t.state} -> {result}")
```

Executable task results are available via `task.stdout`, `task.stderr`, and `task.exit_code`.

### Cluster injection

Pass a pre-configured cluster (or client) via the `cluster=` / `client=` parameters.
The task submission code is unchanged:

```python
# SLURM (requires dask-jobqueue)
from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(cores=4, memory="8GB", walltime="01:00:00")
cluster.scale(jobs=4)
backend = await DaskExecutionBackend(cluster=cluster)

# Kubernetes (requires dask-kubernetes)
from dask_kubernetes.operator import KubeCluster

cluster = KubeCluster(name="rhapsody-workers", n_workers=8)
backend = await DaskExecutionBackend(cluster=cluster)

# Pre-existing Client
from dask.distributed import Client
client = await Client("tcp://scheduler:8786", asynchronous=True)
backend = await DaskExecutionBackend(client=client)
```

!!! tip "Default cluster"
    If `cluster` and `client` are both omitted, Rhapsody creates a `LocalCluster` using the `resources` dict (e.g. `{"n_workers": 4}`).

### GPU and CPU resource scheduling

Pass Dask resource constraints via `task_backend_specific_kwargs={"resources": {...}}`.
Workers must be started with `--resources "GPU=1"` (or equivalent in the cluster config).

```python
ComputeTask(
    function=my_gpu_fn, args=(x,),
    task_backend_specific_kwargs={"resources": {"GPU": 1}},  # requires a GPU worker
)
ComputeTask(
    executable="/bin/sim", arguments=["--n", "4"],
    task_backend_specific_kwargs={"resources": {"CPU": 4}, "shell": True},
)
```

---

## Session API Quick Reference

```python
from rhapsody.api import Session

session = Session(
    backends=[backend],   # list of execution / inference backends
    uid="my-session",     # optional identifier (default: "session.0000")
    work_dir="/path",     # optional working directory (default: cwd)
)

# Add a backend after construction
session.add_backend(another_backend)

# Submit tasks → returns list[asyncio.Future]
futures = await session.submit_tasks(tasks)

# Wait for all tasks (never raises on task failure, raises TimeoutError)
await session.wait_tasks(tasks, timeout=30.0)

# Per-task / batch await (raises on failure)
result  = await futures[0]
results = await asyncio.gather(*futures)

# Session-wide statistics
stats = session.get_statistics()
# stats["counts"]          Counter of terminal states (DONE, FAILED, …)
# stats["summary"]         avg_total / avg_queue / avg_execution latencies (seconds)

# Shutdown (called automatically when using `async with`)
await session.close()
```
