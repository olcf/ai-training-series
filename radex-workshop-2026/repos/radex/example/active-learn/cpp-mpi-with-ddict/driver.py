"""
MPI simulation + Dragon DDict + SequentialActiveLearner — single-file example.

Architecture
────────────
  Dragon DDict (shared key-value store, visible to all Dragon-managed processes)

    sim_meta_iter_{i}       ← rank-0 sentinel; used by every task to self-detect
                               the current iteration without capturing acl or ddict
    sim_rank_{r}_iter_{i}   ← per-rank sample data written by each MPI rank
    model_iter_{i}          ← GP surrogate written by training()
    mse_iter_{i}            ← scalar MSE written by training()
    query_points_iter_{i}   ← high-uncertainty query points written by active_learn()
                              stored as Dragon Serializables for both key and values

  Argument-passing discipline
  ───────────────────────────
  DragonExecutionBackend serialises (cloudpickle) every function task before
  submitting it to a Dragon process. Objects that are not safely serialisable
  (live DDict connections, SequentialActiveLearner with asyncio internals, ...)
  must never appear in a task function's closure.

  The MPI-parallel "simulation" code reads in the DDict Descriptor via an
  environment variable ROSE_DDICT_DESCRIPTOR and communicates to/from the
  DDict using the cross C++/Python Serializables to maintain exchange data
  between the C++ simulation component and the other Python-based processes.

  Rule: pass ``ddict_descriptor`` into every task through
  ``acl.start(initial_config=...)`` as task kwargs. Each task opens its own
  DDict connection with ``DDict.attach(ddict_descriptor)`` and derives the
  current iteration by counting ``sim_meta_iter_*`` sentinel keys.

Run with:
    dragon driver.py
"""

import asyncio
import os
import sys
from pathlib import Path

import dragon  # noqa: F401  – must precede dragon.data imports
import numpy as np
import rhapsody
from dragon.data.ddict import DDict
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import DragonExecutionBackend
from rose.al.active_learner import SequentialActiveLearner
from rose.learner import LearnerConfig, TaskConfig
from rose.metrics import MEAN_SQUARED_ERROR_MSE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from radex.clients.core import DragonClient as Client
from radex.handles.handles import IncomingHandle, OutgoingHandle

rhapsody.enable_logging()

# ── Configuration ──────────────────────────────────────────────────────────────
N_MPI_RANKS: int = 4  # MPI ranks per simulation launch
N_SAMPLES_PER_RANK: int = 16  # few pts per rank → sparse start, AL drives exploration
N_QUERY: int = 8  # query points selected per AL step
MSE_THRESHOLD: float = 0.1  # convergence target
MAX_ITER: int = 10  # hard cap on iterations

# ── Consts ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
EXAMPLES_BIN_DIR = ROOT / "install" / "bin" / "examples"


async def rose_mpi_ddict() -> None:

    # ── 1. Create the shared Dragon DDict ─────────────────────────────────────
    ddict = DDict(
        managers_per_node=1,
        n_nodes=1,
        total_mem=512 * 1024 * 1024,  # 512 MB – scales with ranks × samples × iters
        wait_for_keys=True,
        working_set_size=MAX_ITER + 2,
    )
    ddict_descriptor: str = ddict.serialize()
    print(f"[ROSE] DDict ready  (descriptor prefix: {ddict_descriptor[:32]}…)")
    client = Client(ddict_descriptor, 5)

    # ── 2. Set up ROSE engine and learner ─────────────────────────────────────
    engine = await DragonExecutionBackend()
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    # ── 3. Register all tasks ─────────────────────────────────────────────────
    #
    # Iteration is derived from sentinel keys already in DDict.
    # ddict_descriptor is injected via initial_config task kwargs.

    @acl.simulation_task(as_executable=True)
    async def simulation(
        *args,
        ddict_descriptor: str,
        task_description={
            "process_templates": [
                (
                    N_MPI_RANKS,
                    {"env": {"ROSE_DDICT_DESCRIPTOR": ddict_descriptor}},
                )
            ]
        },
    ):
        path = EXAMPLES_BIN_DIR / "simulation"
        if not path.exists():
            raise FileNotFoundError(
                f"`{os.fspath(path)}` binary does not exist. "
                "Try building the project examples"
            )
        return os.fspath(path)

    @acl.training_task(as_executable=False)
    async def training(*args, ddict_descriptor: str):
        """Train a GP surrogate on simulation data from DDict.

        Iteration: count sim_meta_iter_* sentinels, take the last completed one.
        ddict_descriptor is passed as a task kwarg.
        """

        print("Starting trainer", file=sys.stderr)
        client = Client(ddict_descriptor, 5)

        iteration = (
            client.get_scalar(IncomingHandle("sim_meta_iter_count"))
            if client.contains("sim_meta_iter_count")
            else 0
        )
        iteration -= 1  # latest completed simulation

        X_parts, y_parts = [], []

        for rank in range(N_MPI_RANKS):
            X_parts.append(
                client.get_tensor(IncomingHandle(f"sim_rank_{rank}_iter_{iteration}_X"))
            )
            y_parts.append(
                client.get_tensor(IncomingHandle(f"sim_rank_{rank}_iter_{iteration}_y"))
            )

        X_train = np.vstack(X_parts).ravel().reshape(-1, 1)
        y_train = np.vstack(y_parts).ravel()

        kernel = RBF(length_scale=0.3, length_scale_bounds=(0.01, 5.0)) + WhiteKernel(
            noise_level=1e-2
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, normalize_y=True
        )
        gp.fit(X_train, y_train)

        X_test = np.linspace(0.0, 2.0 * np.pi, 300).reshape(-1, 1)
        y_pred = gp.predict(X_test)
        y_true = (np.sin(X_test) * np.sin(5 * X_test)).ravel()
        mse = float(mean_squared_error(y_true, y_pred))

        client.put_scalar(OutgoingHandle("model_iter_count"), iteration + 1)
        client.put_picklable(f"model_iter_{iteration}", gp)
        client.put_scalar(OutgoingHandle(f"mse_iter_{iteration}"), mse)

        print(
            f"[train]    iter={iteration} | n_train={len(X_train)} | MSE={mse:.6f}",
            flush=True,
        )
        return {}

    @acl.active_learn_task(as_executable=False)
    async def active_learn(*args, ddict_descriptor: str):
        """Max-variance query strategy; writes query points to DDict.

        Iteration: latest entry with both model_iter_{i} and mse_iter_{i}.
        ddict_descriptor is passed as a task kwarg.
        """
        client = Client(ddict_descriptor, 5)

        iteration = (
            client.get_scalar(IncomingHandle("model_iter_count"))
            if client.contains("model_iter_count")
            else 0
        )
        iteration -= 1  # latest trained model

        gp: GaussianProcessRegressor = client.get_picklable(f"model_iter_{iteration}")

        X_candidates = np.linspace(0.0, 2.0 * np.pi, 500).reshape(-1, 1)
        _, std = gp.predict(X_candidates, return_std=True)

        top_idx = np.argsort(std)[-N_QUERY:]

        client.put_tensor(
            OutgoingHandle(f"query_points_iter_{iteration}"),
            X_candidates[top_idx].ravel(),
        )

        mean_unc = float(std.mean())
        max_unc = float(std.max())
        print(
            f"[active]   iter={iteration} | mean_unc={mean_unc:.4f} | "
            f"max_unc={max_unc:.4f} | n_query={N_QUERY}",
            flush=True,
        )
        client.put_scalar(OutgoingHandle("model_iter_count"), iteration + 1)

        mean_unc = max_unc = 0
        return {"mean_uncertainty": mean_unc, "max_uncertainty": max_unc}

    @acl.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE, threshold=MSE_THRESHOLD, as_executable=False
    )
    async def check_mse(*args, ddict_descriptor: str) -> float:
        """Read the scalar MSE for the latest trained model from DDict.

        ddict_descriptor is passed as a task kwarg.
        """
        client = Client(ddict_descriptor, 5)

        iteration = (
            client.get_scalar(IncomingHandle("model_iter_count"))
            if client.contains("model_iter_count")
            else 0
        )
        iteration -= 1  # latest computed MSE

        mse = client.get_scalar(IncomingHandle(f"mse_iter_{iteration}"))
        print(
            f"[check]    iter={iteration} | MSE={mse:.6f} (threshold < {MSE_THRESHOLD})",
            flush=True,
        )
        client.put_scalar(OutgoingHandle("mse_iter_count"), iteration + 1)
        return mse

    # ── 4. Run the active-learning loop ───────────────────────────────────────

    print("\n[ROSE] Starting active-learning loop\n" + "─" * 60)
    common_kwargs = {"ddict_descriptor": ddict_descriptor}
    initial_config = LearnerConfig(
        simulation=TaskConfig(kwargs=common_kwargs),
        training=TaskConfig(kwargs=common_kwargs),
        active_learn=TaskConfig(kwargs=common_kwargs),
        criterion=TaskConfig(kwargs=common_kwargs),
    )

    final_state = None
    async for state in acl.start(max_iter=MAX_ITER, initial_config=initial_config):
        final_state = state
        print(
            f"\n[ROSE]  ── iter={state.iteration:2d}"
            f" | MSE={state.metric_value:.6f}"
            f" | mean_unc={state.mean_uncertainty}"
            f" | should_stop={state.should_stop}\n",
            flush=True,
        )
        if state.should_stop:
            break

    # ── 5. Convergence summary ────────────────────────────────────────────────
    last_iter = final_state.iteration if final_state else 0
    print("\n── Convergence Summary ──────────────────────────────────────────────")
    for i in range(last_iter + 1):
        key = f"mse_iter_{i}"
        if client.contains(key):
            print(
                f"  iter {i:2d} "
                f"│ MSE = {client.get_scalar(IncomingHandle(key)):.6f}"
            )

    # ── 6. Cleanup ────────────────────────────────────────────────────────────
    del client  # FIXME: would really like this to instead be `client.disconnect()`
    ddict.destroy()
    await acl.shutdown()


if __name__ == "__main__":
    asyncio.run(rose_mpi_ddict())
