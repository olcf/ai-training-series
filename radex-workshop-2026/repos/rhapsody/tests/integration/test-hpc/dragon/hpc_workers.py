"""Worker functions that execute inside Dragon's remote worker processes.

All functions used as the `function=` argument in ComputeTask must live here
(not in test files) because Dragon serialises them by module reference.
When PYTHONPATH includes the test-hpc directory, every worker node can import
this module and deserialise the functions successfully.

Rules:
  - No imports at module level that are unavailable on worker nodes.
  - Heavy imports (mpi4py, torch, cupy …) go inside the function body.
  - Functions must be picklable by cloudpickle (all closures captured, no
    references to pytest fixtures or test-file globals).
"""


def add(a, b):
    """Return a + b."""
    return a + b


def identity(n: int) -> int:
    """Return n unchanged — minimal payload for scheduler stress tests."""
    return n


# ---------------------------------------------------------------------------
# MPI worker functions
# ---------------------------------------------------------------------------


def mpi_gather_hostnames() -> dict:
    """Gather all rank hostnames to rank 0 and return them."""
    import socket

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    host = socket.gethostname()

    all_hosts = comm.gather(host, root=0)
    if rank == 0:
        return {"size": size, "hostnames": all_hosts}
    return None


def mpi_allreduce_sum(n: int) -> dict:
    """Each rank contributes n; allreduce sum returned from rank 0."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    total = comm.allreduce(n, op=MPI.SUM)
    if rank == 0:
        return {"size": size, "total": total, "expected": n * size}
    return None


def mpi_broadcast_check(value: int) -> bool:
    """Rank 0 broadcasts value; all ranks verify receipt; rank 0 returns result."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data = value if rank == 0 else None
    data = comm.bcast(data, root=0)
    correct = data == value

    all_correct = comm.gather(correct, root=0)
    if rank == 0:
        return all(all_correct)
    return None


def mpi_scatter_and_gather(data_per_rank: list) -> list:
    """Scatter data_per_rank to all ranks; double each element; gather back."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    scattered = comm.scatter(data_per_rank if rank == 0 else None, root=0)
    processed = scattered * 2

    gathered = comm.gather(processed, root=0)
    if rank == 0:
        return gathered
    return None
