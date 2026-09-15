"""Shared fixtures and markers for the RHAPSODY + Dragon HPC test suite.

Run with:
    dragon python3 -m pytest test-hpc/ -v

All topology detection happens here.  Tests never hardcode node counts,
GPU counts, or hostnames — they skip automatically when the required
resources are not present in the current allocation.
"""

import os

import pytest

# ---------------------------------------------------------------------------
# Custom marks — registered so pytest does not warn about unknown marks
# ---------------------------------------------------------------------------
# Usage in test files:
#   @pytest.mark.multi_node   – skipped when allocation has < 2 nodes
#   @pytest.mark.gpu          – skipped when no GPUs are detected
#   @pytest.mark.mpi          – skipped when mpi4py is not importable
#   @pytest.mark.scale        – long-running scale/stress tests


def pytest_configure(config):
    config.addinivalue_line("markers", "multi_node: require >= 2 nodes in the Dragon allocation")
    config.addinivalue_line("markers", "gpu: require >= 1 GPU per node")
    config.addinivalue_line("markers", "mpi: require mpi4py")
    config.addinivalue_line("markers", "scale: long-running scale and stress tests")


# ---------------------------------------------------------------------------
# Topology detection  (session-scoped — runs once per pytest session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def topology():
    """Detect nodes and GPUs from the live Dragon allocation.

    Returns a list of dicts, one per node:
        [{"huid": int, "hostname": str, "gpus": [int, ...], "num_gpus": int}, ...]

    Raises RuntimeError if Dragon is not initialised.
    """
    from dragon.native.machine import Node
    from dragon.native.machine import System

    nodes = []
    for huid in System().nodes:
        node = Node(huid)
        gpu_ids = list(node.gpus)
        nodes.append(
            {
                "huid": huid,
                "hostname": node.hostname,
                "gpus": gpu_ids,
                "num_gpus": len(gpu_ids),
            }
        )
    assert nodes, "Dragon allocation is empty — no nodes detected"
    return nodes


@pytest.fixture(scope="session")
def num_nodes(topology):
    return len(topology)


@pytest.fixture(scope="session")
def gpu_nodes(topology):
    """Subset of topology entries that have at least one GPU."""
    return [n for n in topology if n["num_gpus"] > 0]


@pytest.fixture(scope="session")
def total_gpus(gpu_nodes):
    return sum(n["num_gpus"] for n in gpu_nodes)


# ---------------------------------------------------------------------------
# Automatic skip logic — applied via autouse fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _apply_topology_skips(request, topology, gpu_nodes):
    """Skip tests whose resource marks cannot be satisfied by the allocation."""
    if request.node.get_closest_marker("multi_node") and len(topology) < 2:
        pytest.skip(f"multi_node test requires >= 2 nodes; allocation has {len(topology)}")

    if request.node.get_closest_marker("gpu") and not gpu_nodes:
        pytest.skip("gpu test requires at least one GPU-equipped node; none detected")

    if request.node.get_closest_marker("mpi"):
        try:
            import mpi4py  # noqa: F401
        except ImportError:
            pytest.skip("mpi test requires mpi4py; package not available")


# ---------------------------------------------------------------------------
# Backend and session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def dragon_backend(request):
    """Initialise a DragonExecutionBackend for the test module.

    Scale tests can override batch_kwargs via indirect parametrisation or by setting the
    RHAPSODY_DDICT_MEM_GB env var (default: 2 GiB/node).
    """
    from dragon.native.machine import System

    from rhapsody.backends import DragonExecutionBackend

    num_nodes = len(list(System().nodes))
    ddict_mem_gb = int(os.environ.get("RHAPSODY_DDICT_MEM_GB", 2))
    ddict_mem = ddict_mem_gb * num_nodes * (1024**3)

    backend = await DragonExecutionBackend(batch_kwargs={"results_ddict_mem": ddict_mem})
    yield backend
    await backend.shutdown()


@pytest.fixture(scope="module")
async def rhapsody_session(dragon_backend):
    """RHAPSODY Session wrapping the module-scoped Dragon backend."""
    from rhapsody.api import Session

    session = Session(backends=[dragon_backend])
    yield session
    # Session is closed by the backend shutdown in dragon_backend fixture
