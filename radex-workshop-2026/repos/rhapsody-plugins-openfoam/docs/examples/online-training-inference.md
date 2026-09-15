# Online Training & Inference

**Location:** [`examples/online-training-inference`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/examples/online-training-inference)

Shows RHAPSODY orchestrating a heterogeneous pipeline alongside a persistent service: a GPU-backed inference service is started once and left running for the duration of the workflow, while an MPI preprocessing pipeline (mesh generation, partitioning, solver-input generation) runs to completion, feeding data to it.

```python
--8<-- "examples/online-training-inference/driver.py"
```

Key points:

- The GPU service is submitted but never awaited directly — it runs in the background for the life of the session, communicating over a Redis-backed feature/prediction queue.
- Each preprocessing task uses `task_backend_specific_kwargs={"partition": ...}` to route work to a named `DragonExecutionBackendV3` partition (e.g. `gpu_partition` vs. `preprocess_partition`).
- Preprocessing tasks are awaited sequentially, with `task.state` checked after each `await` to fail fast if a stage doesn't complete (`"DONE"`).

## Running

```bash
cd examples/online-training-inference
python driver.py
```
