# pitzDaily Optimization

**Location:** [`examples/pitzDaily-optimize`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/examples/pitzDaily-optimize)

The most involved example: a Bayesian-optimization loop (via [`scikit-optimize`](https://scikit-optimize.github.io/stable/)) proposes k-epsilon turbulence model parameters, an ensemble of `pitzDaily` cases is run in parallel for each iteration, and the resulting loss is fed back into the optimizer — all while the running solvers stream results back through a radex-backed Dragon `DDict` rather than reading case output from disk.

```python
--8<-- "examples/pitzDaily-optimize/driver-staged.py"
```

## How It Works

1. `initialize_optimizer()` sets up a Gaussian-process-based `skopt.Optimizer` over bounds derived from the default k-epsilon parameters.
2. Each iteration, `optimizer.ask(n_points=ensemble_size)` proposes a batch of parameter sets; `create_case(...)` builds a `pitzDailyCase` per set, passing the serialized radex `DDict` (`radex_store.serialize()`) so each case's `radexWrite` function object can stream fields directly to it.
3. All ensemble members' `solve` stages are collected into one task list and submitted together to the same `OFSession`.
4. `update_optimizer(...)` calls `optimizer.tell(...)` with each converged case's loss, closing the loop.

A `driver.py` variant without the staged-batch optimizer is also provided for comparison.

## Running

```bash
cd examples/pitzDaily-optimize
python driver-staged.py
```

!!! note
    Requires `scikit-optimize` (`skopt`) in addition to the core plugin dependencies.
