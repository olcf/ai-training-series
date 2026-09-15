import os


def pytest_ignore_collect(collection_path, config):
    """Skip all HPC tests unless RHAPSODY_HPC=1 is explicitly set.

    This prevents the test-hpc/ subtree from being collected during regular
    CI runs or local development runs.  The tests require a live Dragon
    allocation on an HPC cluster and cannot run in GitHub Actions.

    To run on a cluster:
        RHAPSODY_HPC=1 dragon python3 -m pytest tests/integration/test-hpc/dragon/ -v
    """
    if os.environ.get("RHAPSODY_HPC") != "1":
        return True  # tell pytest to skip this path entirely
    return None  # normal collection
