# Hello World

**Location:** [`examples/hello-world`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/examples/hello-world)

The simplest possible example: it doesn't use any of the OpenFOAM plugin's abstractions, and instead shows the underlying RHAPSODY `ComputeTask` API used to run a compiled MPI executable ([`hello_world.c`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/blob/main/examples/hello-world/hello_world.c)) across multiple ranks.

```python
--8<-- "examples/hello-world/driver.py"
```

`task_backend_specific_kwargs["process_templates"]` passes a list of `(num_ranks, options)` tuples to the Dragon backend, here launching 8 ranks that inherit the driver's environment.

## Running

```bash
cd examples/hello-world
python driver.py
```

An `driver-asyncflow.py` variant is also provided, showing the same task run through [AsyncFlow](https://radical-cybertools.github.io/rhapsody/integrations).
