# RHAPSODY HPC Integration Tests — Dragon Backend

End-to-end integration tests for **RHAPSODY + DragonHPC** on multi-node,
GPU-enabled HPC systems. These tests exercise the full stack: RHAPSODY's
session and backend layer, Dragon's Batch scheduler, DDict result transport,
process template placement, and MPI job orchestration.

This directory is part of a broader `test-hpc/` structure designed to support
multiple HPC execution backends side by side:

```
tests/integration/test-hpc/
├── conftest.py      ← CI gate: entire subtree skipped unless RHAPSODY_HPC=1
├── dragon/          ← this directory (DragonHPC backend)
└── flux/            ← planned (Flux backend, same categories)
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.10 – 3.12 (Dragon constraint) |
| DragonHPC | 0.14.0+ installed in the active environment |
| RHAPSODY | installed from source (`pip install -e .`) |
| mpi4py | required only for MPI tests (`pip install mpi4py`) |
| Active allocation | Dragon global services must be running (`dragon` launcher) |

---

## CI behaviour

These tests are **never collected or run in GitHub Actions**. A parent-level
`conftest.py` at `tests/integration/test-hpc/conftest.py` uses the
`pytest_ignore_collect` hook to skip the entire subtree unless the environment
variable `RHAPSODY_HPC=1` is set. No Makefile or CI YAML changes are needed.

```
make test-ci        # RHAPSODY_HPC unset → test-hpc/ silently skipped
make test-regular   # same
```

---

## Running on an HPC cluster

All commands must be run from the **repository root** with the Dragon launcher.
Set `RHAPSODY_HPC=1` to unlock collection.

### Step 1 — activate your environment

```bash
# Example: Cray/NERSC system
module load python
source /path/to/your/venv/bin/activate
```

### Step 2 — obtain an allocation and start Dragon

```bash
# Example: SLURM allocation with 4 nodes, 4 GPUs each
salloc -N 4 --gpus-per-node=4 -t 01:00:00
```

### Step 3 — run the tests

```bash
# All standard tests (scale excluded by default)
RHAPSODY_HPC=1 dragon python3 -m pytest tests/integration/test-hpc/dragon/ -v

# Single category
RHAPSODY_HPC=1 dragon python3 -m pytest tests/integration/test-hpc/dragon/test_pinning.py -v

# Filter by resource mark
RHAPSODY_HPC=1 dragon python3 -m pytest tests/integration/test-hpc/dragon/ -m gpu -v
RHAPSODY_HPC=1 dragon python3 -m pytest tests/integration/test-hpc/dragon/ -m multi_node -v
RHAPSODY_HPC=1 dragon python3 -m pytest tests/integration/test-hpc/dragon/ -m "not scale" -v

# Scale / stress tests (long-running, off by default)
RHAPSODY_HPC=1 RHAPSODY_RUN_SCALE=1 dragon python3 -m pytest \
    tests/integration/test-hpc/dragon/test_scale.py -v -s

# MPI tests only
RHAPSODY_HPC=1 dragon python3 -m pytest tests/integration/test-hpc/dragon/test_mpi.py -v
```

---

## File structure

```
dragon/
├── conftest.py            # topology detection, shared fixtures, auto-skip logic
├── hpc_workers.py         # worker functions importable by Dragon remote processes
├── pytest.ini             # asyncio mode, marker registration, pythonpath = .
├── test_multinode_gpu.py  # Category 1 — multi-node, multi-GPU execution
├── test_singlenode_gpu.py # Category 2 — single-node, single-GPU execution
├── test_pinning.py        # Category 3 — task pinning and GPU affinity
├── test_scale.py          # Category 4 — scale and stress (10K+ tasks)
└── test_mpi.py            # Category 5 — MPI integration (mpi4py)
```

---

## Topology-aware skip logic

Tests auto-skip when the allocation cannot satisfy their resource requirements.
No manual configuration is needed — topology is detected at runtime from
`dragon.native.machine.System`.

| Mark | Skipped when |
|------|-------------|
| `multi_node` | fewer than 2 nodes detected in the Dragon allocation |
| `gpu` | no GPU-equipped nodes detected |
| `mpi` | `mpi4py` not importable |
| `scale` | `RHAPSODY_RUN_SCALE != 1` |

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RHAPSODY_HPC` | unset | **Must be `1`** for pytest to collect these tests at all |
| `RHAPSODY_RUN_SCALE` | `0` | Set to `1` to enable scale/stress tests |
| `RHAPSODY_SCALE_FUNC_COUNT` | `10000` | Function task count for scale tests |
| `RHAPSODY_SCALE_EXEC_COUNT` | `1000` | Executable task count for scale tests |
| `RHAPSODY_DDICT_MEM_GB` | `2` (standard) / `4` (scale) | GiB per node for Dragon's results DDict |

---

## Design decisions

### `hpc_workers.py` — why a separate module

Dragon serialises Python functions by **module reference**
(`hpc_workers.identity`, etc.). When a remote worker on another node
deserialises the function, it imports `hpc_workers` by name. Functions defined
directly in test files (e.g. `test_scale.py`) would require `test_scale` to be
importable on every worker node, which fails on multi-node clusters where the
test directory is not on `sys.path`. `pytest.ini` sets `pythonpath = .` so
`hpc_workers.py` (in the same directory) is always importable.

### Pinning tests use executables, not Python functions

Dragon's `batch.process()` — used whenever `process_template` is specified —
stores the **process exit code** (integer `0`) in the DDict result, not the
Python function's return value. Node and GPU identity is therefore verified via
`stdout` (`/bin/hostname`, `echo $CUDA_VISIBLE_DEVICES`). Unpinned function
tasks use `batch.function()` which correctly captures the Python return value.

### `asyncio_default_test_loop_scope = module`

Module-scoped fixtures (`dragon_backend`, `rhapsody_session`) bind asyncio
futures to the module event loop. Without setting the test loop scope to
`module` as well, pytest-asyncio runs each test on its own function-scoped
loop, causing "future belongs to a different loop" errors.

---

## Full Test Coverage Matrix

**25 tests total** — 21 standard + 4 scale-gated (★)

| # | Test | File | Task Type | Node Scope | GPU | MPI | Scale | What is verified |
|---|------|------|-----------|-----------|-----|-----|-------|-----------------|
| 1 | `test_one_task_per_node` | multinode_gpu | Executable (`/bin/hostname`) | All nodes | — | — | — | 1 pinned task/node; stdout = hostname |
| 2 | `test_one_task_per_gpu_across_nodes` | multinode_gpu | Executable (`/bin/bash`) | All GPU nodes | All GPUs/node | — | — | 1 task/GPU; stdout = `CUDA_VISIBLE_DEVICES` |
| 3 | `test_concurrent_submission_all_nodes` | multinode_gpu | Executable (`/bin/hostname`) | All nodes | — | — | — | Single `submit_tasks` spans all nodes simultaneously |
| 4 | `test_executable_pinned_to_first_node` | singlenode_gpu | Executable (`/bin/hostname`) | 1 node | — | — | — | Minimal pinning path; stdout = hostname |
| 5 | `test_executable_echo_on_first_node` | singlenode_gpu | Executable (`/bin/echo`) | 1 node | — | — | — | Stdout capture; CLI argument forwarding |
| 6 | `test_function_return_value` | singlenode_gpu | Python function (`add`) | Any | — | — | — | `batch.function()` return value capture |
| 7 | `test_multiple_function_tasks_return_correct_values` | singlenode_gpu | Python function (`add`) | Any | — | — | — | 10 concurrent tasks; no result corruption |
| 8 | `test_single_gpu_affinity_on_first_node` | singlenode_gpu | Executable (`/bin/bash`) | 1 GPU node | 1 GPU | — | — | Single GPU affinity; stdout = device ID |
| 9 | `test_all_gpus_on_first_node` | singlenode_gpu | Executable (`/bin/bash`) | 1 GPU node | All GPUs | — | — | Per-GPU task; each task sees only its GPU |
| 10 | `test_pin_to_each_node_sequentially` | pinning | Executable (`/bin/hostname`) | All nodes | — | — | — | Sequential pinning correctness per node |
| 11 | `test_exclusive_node_pinning` | pinning | Executable (`/bin/hostname`) | 2 nodes | — | — | — | Tasks on A ≠ tasks on B; cross-node isolation |
| 12 | `test_pin_to_specific_gpu` | pinning | Executable (`/bin/bash`) | 1 GPU node | Per GPU | — | — | Each task sees exactly its assigned GPU |
| 13 | `test_round_robin_gpu_across_nodes` | pinning | Executable (`/bin/bash`) | All GPU nodes | All GPUs/node | — | — | (node, GPU) pair enforced; parsed from stdout |
| 14 | `test_executable_pinned_to_each_node` | pinning | Executable (`/bin/hostname`) | All nodes | — | — | — | Baseline executable pinning across all nodes |
| 15 | `test_mpi_gather_single_node` | mpi | Python function (`mpi_gather_hostnames`) | 1 node | — | 2 ranks | — | Gather: all hostnames collected at rank 0 |
| 16 | `test_mpi_allreduce_single_node` | mpi | Python function (`mpi_allreduce_sum`) | 1 node | — | 4 ranks | — | Allreduce: global sum = n × ranks |
| 17 | `test_mpi_broadcast_single_node` | mpi | Python function (`mpi_broadcast_check`) | 1 node | — | 4 ranks | — | Broadcast: all ranks receive the correct value |
| 18 | `test_mpi_gather_across_nodes` | mpi | Python function (`mpi_gather_hostnames`) | All nodes | — | 1 rank/node | — | Each node hostname present in gather result |
| 19 | `test_mpi_allreduce_across_nodes` | mpi | Python function (`mpi_allreduce_sum`) | All nodes | — | 2 ranks/node | — | Multi-node allreduce sum correctness |
| 20 | `test_multiple_independent_mpi_jobs` | mpi | Python function (`mpi_gather_hostnames`) | All nodes | — | 1 rank/node | — | 4 concurrent MPI jobs; all complete without interference |
| 21★ | `test_10k_function_tasks` | scale | Python function (`identity`) | All nodes | — | — | 10,000 | All DONE; `return_value == index` (no result corruption) |
| 22★ | `test_1k_executable_tasks` | scale | Executable (`/bin/echo`) | All nodes | — | — | 1,000 | All DONE; throughput measured and reported |
| 23★ | `test_mixed_function_and_executable_tasks` | scale | Mixed | All nodes | — | — | 1,100 | Mixed task types in one submission; all complete |
| 24★ | `test_wave_submission_stability` | scale | Python function (`identity`) | All nodes | — | — | 10,000 | 10 waves × 1,000 tasks; monitor loop keeps pace with arrivals |

★ scale-gated — requires `RHAPSODY_RUN_SCALE=1`

---

## Coverage Summary

| Dimension | What is covered |
|-----------|----------------|
| **Task types** | Python function (`batch.function()`), Executable (`batch.process()`), MPI job (`batch.job()`) |
| **Node scope** | 1 node · 2 nodes (exclusive isolation) · all nodes (sequential and concurrent) |
| **GPU scope** | No GPU · 1 GPU · all GPUs on one node · all GPUs across all nodes (round-robin) |
| **MPI patterns** | gather · allreduce · broadcast · concurrent independent jobs |
| **MPI ranks** | 2 ranks (single-node) · 4 ranks (single-node) · 1 rank/node · 2 ranks/node (multi-node) |
| **Scale** | 1 task · 10 tasks · 100 tasks · 1 K executables · 10 K functions · 10 K in waves |
| **Return value paths** | Exit code (executable) · Python return value (`batch.function()`) · MPI collective result at rank 0 |
| **Submission patterns** | Sequential · concurrent (single call) · wave (incremental batches) |
| **Verification methods** | `state == DONE` · stdout content · return value equality · MPI collective correctness |

---

## Known Gaps

| Gap | Description | Priority |
|-----|-------------|----------|
| **GPU + MPI combined** | No test pins MPI ranks to specific GPUs (e.g. 1 rank/GPU with `gpu_affinity`) | High |
| **Task failure propagation** | No test submits an intentionally failing task to verify `FAILED` state, exception capture, and non-blocking delivery of other tasks | High |
| **Task cancellation** | Cancellation under multi-node and GPU workloads not covered here; see `test_cancel.py` | Medium |
| **Return value from pinned processes** | `batch.process()` with `process_template` only returns the exit code; Python return values from pinned tasks require a separate communication channel | Medium |
| **DDict overflow / back-pressure** | No test fills the results DDict past capacity to verify graceful failure | Medium |
| **Heterogeneous node topologies** | All tests assume homogeneous GPU count per node | Low |
| **Flux backend parity** | `flux/` sibling directory does not yet exist; the same 25 tests should be ported when the Flux backend matures | Low |
