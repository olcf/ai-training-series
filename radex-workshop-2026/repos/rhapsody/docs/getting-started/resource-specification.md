# Resource Specification Guide

RHAPSODY uses a **per-backend** model for resource requirements.
Resource requirements are expressed through `task_backend_specific_kwargs`, which is forwarded directly to the backend that executes the task during the creation and submission of the task.

This design keeps the task API thin and makes resource semantics explicit per backend.

---

## Quick Reference

| Backend | Resource key | cwd | env | shell |
|---------|-------------|-----|-----|-------|
| `ConcurrentExecutionBackend` | not supported | `task_backend_specific_kwargs={"cwd": "..."}` | `task_backend_specific_kwargs={"env": {...}}` | `task_backend_specific_kwargs={"shell": True}` |
| `DaskExecutionBackend` | `task_backend_specific_kwargs={"resources": {"GPU": 1}}` | `task_backend_specific_kwargs={"cwd": "..."}` | `task_backend_specific_kwargs={"env": {...}}` | `task_backend_specific_kwargs={"shell": True}` |
| `DragonExecutionBackend` | `task_backend_specific_kwargs={"process_template": {"policy": ...}}` | `task_backend_specific_kwargs={"process_template": {"cwd": "..."}}` | `task_backend_specific_kwargs={"process_template": {"env": {...}}}` | not applicable |

---

## Concurrent Backend

`ConcurrentExecutionBackend` runs tasks in a local `ThreadPoolExecutor` or `ProcessPoolExecutor`.
There is no slot or resource scheduler — all processes share the same host.

**Supported `task_backend_specific_kwargs`:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cwd` | `str` | `None` (inherits process cwd) | Working directory for the subprocess |
| `env` | `dict` | `None` (inherits parent environment) | Environment variables |
| `shell` | `bool` | `False` | Execute via shell (`create_subprocess_shell`) |

```python
from rhapsody.api import ComputeTask

# Working directory
task = ComputeTask(
    executable="/bin/pwd",
    task_backend_specific_kwargs={"cwd": "/data"},
)

# Custom environment
task = ComputeTask(
    executable="/bin/printenv",
    task_backend_specific_kwargs={
        "env": {"MY_VAR": "hello", "CUDA_VISIBLE_DEVICES": "0"},
    },
)

# Shell execution
task = ComputeTask(
    executable="echo $HOSTNAME && date",
    task_backend_specific_kwargs={"shell": True, "cwd": "/tmp"},
)
```

!!! note "CPU / GPU pinning"
    The Concurrent backend has no resource scheduler. To pin threads or GPUs, do it
    inside your task function — for example, call `torch.cuda.set_device(gpu_id)` at
    the start of your callable, or use `os.sched_setaffinity` for CPU pinning.

---

## Dask Backend

`DaskExecutionBackend` submits tasks to a Dask cluster. Resource scheduling uses Dask's
native [resource system](https://distributed.dask.org/en/stable/resources.html).

**Supported `task_backend_specific_kwargs`:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `resources` | `dict` | `{}` | Dask resource constraints forwarded to `client.submit(..., resources=...)` |
| `cwd` | `str` | `None` | Working directory for executable tasks |
| `env` | `dict` | `None` | Environment variables for executable tasks |
| `shell` | `bool` | `False` | Execute executable via shell |

`cwd` and `env` apply to executable tasks only (they are passed to `subprocess.run` inside the Dask worker). For function tasks, set the working directory or environment from within the function itself.

```python
from rhapsody.api import ComputeTask

# GPU resource constraint — worker must be started with --resources "GPU=1"
task = ComputeTask(
    function=my_gpu_fn,
    args=(data,),
    task_backend_specific_kwargs={
        "resources": {"GPU": 1},
    },
)

# Multi-resource constraint
task = ComputeTask(
    function=large_model_fn,
    args=(data,),
    task_backend_specific_kwargs={
        "resources": {"GPU": 2, "memory": 32e9},
    },
)

# Executable with cwd, env, and shell
task = ComputeTask(
    executable="./simulate.sh",
    arguments=["--config", "run.yaml"],
    task_backend_specific_kwargs={
        "cwd": "/scratch/run-001",
        "env": {"OMP_NUM_THREADS": "4"},
        "shell": True,
    },
)
```

Workers must be started advertising the resources you request:

```bash
# Start a worker with GPU and custom memory resources
dask worker tcp://scheduler:8786 --resources "GPU=1,memory=32e9"
```

!!! warning "Unsatisfiable resources"
    If no worker can satisfy the requested resources, the task fails immediately with
    a `TaskValidationError`-like message instead of hanging indefinitely. This is caught
    by `DaskExecutionBackend._check_resources_satisfiable()` at submit time.

### Cluster injection

The `resources=` init parameter configures `LocalCluster`. For any other cluster,
pass it via `cluster=` or `client=` — the task code is unchanged:

```python
from dask_jobqueue import SLURMCluster
from rhapsody.backends import DaskExecutionBackend

cluster = SLURMCluster(cores=4, memory="8GB", walltime="01:00:00")
cluster.scale(jobs=4)
backend = await DaskExecutionBackend(cluster=cluster)
```

---

## Dragon Backend (V3)

`DragonExecutionBackend` submits tasks to the Dragon runtime. All per-task resource
and placement settings flow through Dragon's [ProcessTemplate](https://dragonhpc.github.io/dragon/doc/_build/html/ref/native/dragon.native.process.html#dragon.native.process.ProcessTemplate).

There is **no backend-level `working_directory`** in V3. Every process
configuration must be set per-task via `task_backend_specific_kwargs`.

### Backend-level configuration (`batch_kwargs`)

Pass a `batch_kwargs` dict when constructing the backend to tune the underlying
`dragon.workflows.batch.Batch` instance:

```python
backend = await DragonExecutionBackend(batch_kwargs={
    "num_nodes": 8,
    "results_ddict_mem": 4 * 1024**3,  # 4 GiB results DDict
    "scheduler_workers": 16,
})
```

All accepted keys and their defaults are documented in the
[Dragon Batch API](https://dragonhpc.github.io/dragon/doc/_build/html/ref/workflows/dragon.workflows.batch.Batch.html#dragon.workflows.batch.Batch).

!!! note "Dragon 0.14.0"
    `pool_nodes` is accepted for backwards compatibility but has no effect in
    Dragon 0.14.0 — it is silently forced to `1` internally.

**Supported `task_backend_specific_kwargs`:**

| Key | Type | Description |
|-----|------|-------------|
| `process_template` | `dict` | `ProcessTemplate` kwargs for a single-process task |
| `process_templates` | `list[tuple[int, dict]]` | List of `(n_procs, template_dict)` for multi-rank jobs |

**`ProcessTemplate` kwargs** (passed inside `process_template` or each entry of `process_templates`):

| Parameter | Type | Description |
|-----------|------|-------------|
| `cwd` | `str` | Working directory for the process (default: `"."`) |
| `env` | `dict` | Environment variables |
| `stdout` | `int` | stdout handling (`PIPE`, `STDOUT`, `None`) |
| `stderr` | `int` | stderr handling (`PIPE`, `STDOUT`, `None`) |
| `policy` | `Policy` | Placement and GPU/NUMA affinity |
| `options` | `ProcessOptions` | Dragon process options |

!!! warning "Managed parameters"
    `target`, `args`, and `kwargs` are managed by RHAPSODY internally via
    `ComputeTask.executable`, `ComputeTask.arguments`, and `ComputeTask.function`.
    Do not set them inside `process_template`.

    `stdout` defaults to `Popen.PIPE` for all process tasks (Dragon 0.14.0
    requires this to be set explicitly for output capture). You can override
    it by passing `stdout=None` in your template if you want to suppress
    capture, or `stdout=Popen.STDOUT` to merge with stderr.

### Single-process task

```python
from rhapsody.api import ComputeTask

# Executable with cwd and env
task = ComputeTask(
    executable="/bin/bash",
    arguments=["-c", "echo $HOSTNAME"],
    task_backend_specific_kwargs={
        "process_template": {
            "cwd": "/scratch/run-001",
            "env": {"OMP_NUM_THREADS": "8"},
        }
    },
)

# Function task with cwd
task = ComputeTask(
    function=my_simulation,
    args=(config,),
    task_backend_specific_kwargs={
        "process_template": {"cwd": "/scratch/run-001"},
    },
)
```

### Multi-rank (parallel) job

Use `process_templates` (plural) to launch a task as a parallel job. Each entry is
`(n_procs, template_dict)`:

```python
# 2 groups × 2 processes = 4 processes total
task = ComputeTask(
    executable="/bin/bash",
    arguments=["-c", "echo $HOSTNAME"],
    task_backend_specific_kwargs={
        "process_templates": [
            (2, {"cwd": "/data", "env": {"RANK": "0"}}),
            (2, {"cwd": "/data", "env": {"RANK": "2"}}),
        ]
    },
)
```

### GPU affinity with `Policy`

Use Dragon's `Policy` to pin processes to specific GPUs across nodes:

```python
from dragon.infrastructure.policy import Policy
from dragon.native.machine import System, Node
from rhapsody.api import ComputeTask

def make_gpu_policies(nprocs: int):
    """Round-robin GPU assignment across all Dragon nodes."""
    all_gpus = [
        (node.hostname, gpu_id)
        for huid in System().nodes
        for node in [Node(huid)]
        for gpu_id in node.gpus
    ]
    return [
        Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=all_gpus[i % len(all_gpus)][0],
            gpu_affinity=[all_gpus[i % len(all_gpus)][1]],
        )
        for i in range(nprocs)
    ]

policies = make_gpu_policies(nprocs=4)

# Single process pinned to GPU 0
task = ComputeTask(
    function=gpu_work,
    task_backend_specific_kwargs={
        "process_template": {"policy": policies[0]},
    },
)

# Parallel job: 2 groups of 2 processes each, each group on its own GPU
task = ComputeTask(
    function=gpu_work,
    task_backend_specific_kwargs={
        "process_templates": [
            (2, {"policy": policies[0]}),
            (2, {"policy": policies[1]}),
        ]
    },
)
```
