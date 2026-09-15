# RHAPSODY Examples

This directory contains example scripts demonstrating RHAPSODY's capabilities, from simple task submission to AI-HPC workflows and MPI integration.

## How to Run

Different examples require different launchers depending on the backend they use:

| Backend | Launcher | Command |
|---------|----------|---------|
| `ConcurrentExecutionBackend` | Standard Python | `python example.py` |
| `DragonExecutionBackend` | Dragon runtime | `dragon example.py` |
| `DragonVllmInferenceBackend` | Dragon runtime + GPU | `dragon example.py` |

> **Why `dragon` instead of `python`?**
> The Dragon backend requires its own managed runtime environment for distributed process management. The `dragon` launcher bootstraps this runtime before executing your script. Running a Dragon-based example with plain `python` will fail.

---

## Examples

### 00 — Native API (Concurrent Backend)

```bash
python 00-workload-native-api.py
```

The simplest possible RHAPSODY example. Uses `ConcurrentExecutionBackend` (Python's built-in thread/process pool) to run two shell commands. Demonstrates the core workflow: create tasks, submit via a session, wait for completion, read results. No Dragon or HPC infrastructure required.

**What you'll learn:** `ComputeTask` with executables, `Session.submit_tasks()`, `Session.wait_tasks()`, in-place result access.

---

### 01 — Async Gather (Dragon Backend)

```bash
dragon 01-workload-async-gather.py
```

Submits **1024 function tasks** to `DragonExecutionBackend` and awaits all of them with `asyncio.gather()`. Demonstrates RHAPSODY's batch submission throughput and the `async with session` context manager pattern.

**What you'll learn:** Large-scale batch submission, `asyncio.gather(*futures)` for collecting results, Dragon backend initialization.

---

### 02 — Heterogeneous Workloads (Dragon Backend)

```bash
dragon 02-workload-heterogeneous.py
```

Runs **five different execution modes** in a single session:

| Task | Mode | Configuration |
|------|------|---------------|
| Native function | Function Native | No backend kwargs |
| Shell command (single process) | Executable Process | `process_template: {}` |
| Shell command (2+2 parallel) | Executable Job | `process_templates: [(2, {}), (2, {})]` |
| Python function (single process) | Function Process | `process_template: {}` |
| Python function (2+2 parallel) | Function Job | `process_templates: [(2, {}), (2, {})]` |

**What you'll learn:** `task_backend_specific_kwargs`, `process_template` vs `process_templates`, mixing task types in one submission.

---

### 03 — AI-HPC Workflow (Dragon + vLLM)

```bash
dragon 03-workload-ai-hpc.py
```

Combines `DragonExecutionBackend` (HPC compute) with `DragonVllmInferenceBackend` (LLM inference) in a single session. Submits a mix of `AITask` and `ComputeTask` objects — RHAPSODY routes each to the correct backend automatically.

**Requires:** GPU access and AI package installed (`pip install rhapsody-py[ai]`) Configuration is passed directly as `ModelConfig`/`HardwareConfig` objects — no YAML file needed.

**What you'll learn:** Multi-backend sessions, `AITask` with single and batched prompts, task routing via the `backend` field.

---

### 04 — AsyncFlow Integration (Dragon Backend)

```bash
dragon 04-integration-asyncflow.py
```

Integrates RHAPSODY with [RADICAL AsyncFlow](https://github.com/radical-cybertools) to run **1024 concurrent workflows**, each with a three-task dependency chain (generate → process → aggregate). Demonstrates how RHAPSODY backends plug into higher-level workflow engines.

**Requires:** `radical.asyncflow` installed.

**What you'll learn:** `WorkflowEngine`, `@flow.function_task` decorator, task dependencies, concurrent workflow execution.

---

### 06 — Multi-Service AI-HPC Workflow (Dragon + vLLM)

```bash
dragon 06-workload-ai-hpc-multi-service.py
```

Runs **two independent `DragonVllmInferenceBackend` services** at once (one per node, via `HardwareConfig(node_offset=...)`) and drives each of them two ways in the same session: `AITask` submitted directly (no HTTP, same as 03), and a `ComputeTask(function=...)` that POSTs a batch of N prompts to each service's `/generate` endpoint over `aiohttp`.

**Requires:** GPU access (2 GPUs), vLLM installed (`dragonhpc[ai]`), and a locally downloaded model directory (`HF_HUB_OFFLINE=1` recommended — see the model download notes for this backend).

**What you'll learn:** Running multiple inference services in one allocation, `use_service=True` HTTP endpoints, `get_endpoint()`, mixing direct (`AITask`) and service-style (`ComputeTask` + `aiohttp`) access to the same backend type.
