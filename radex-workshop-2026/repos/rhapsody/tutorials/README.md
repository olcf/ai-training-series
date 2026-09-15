# RHAPSODY Tutorials

This directory contains Jupyter notebook tutorials for learning RHAPSODY.

## Launching the Notebooks

### Without Dragon (basic-usage.ipynb)

`basic-usage.ipynb` uses `ConcurrentExecutionBackend` and runs anywhere — no HPC or Dragon required:

```bash
jupyter notebook
```

### With Dragon Backend (dragon-resource-placement.ipynb, rhapsody-tutorial.ipynb)

Tutorials that use `DragonExecutionBackend` require the Dragon runtime. You **must** launch the notebook server using `dragon-jupyter`:

```bash
dragon-jupyter
```

This starts a Jupyter server within the Dragon-managed runtime, giving your notebook access to Dragon's process management, distributed execution, and Batch API. Without it, any cell that initializes a Dragon backend will fail.

> **Why not regular `jupyter notebook`?**
> Dragon requires its own managed runtime environment. The `dragon-jupyter` command bootstraps this environment before starting the Jupyter server, so all kernels inherit the necessary Dragon context.

### Quick Start

```bash
# 1. Activate your environment
source /path/to/your/venv/bin/activate

# 2a. For basic-usage.ipynb (no Dragon needed)
jupyter notebook

# 2b. For Dragon tutorials
dragon-jupyter
```

## Contents

| Notebook | Backend | Description |
|----------|---------|-------------|
| `basic-usage.ipynb` | Concurrent (ProcessPool) | Run 100 function tasks and 100 executable tasks, process stdout/stderr, handle errors and resubmit failed tasks. No HPC required. |
| `dragon-resource-placement.ipynb` | Dragon (V3) | Control task placement with Dragon Policy: node targeting, CPU affinity, GPU affinity, and combined resource control. Requires Dragon runtime. |
| `rhapsody-tutorial.ipynb` | Dragon (V3) | Progressive tutorial covering basic usage, heterogeneous workloads, and AI-HPC workflows with vLLM. Requires Dragon runtime. |
