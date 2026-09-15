import os

import dragon
import pytest
from dragon.data.ddict import DDict
from dragon.globalservices.api_setup import get_gs_ret_cuid
from dragon.infrastructure.facts import DRAGON_BASE_DIR

from radex.clients.core import DragonClient as Client


@pytest.fixture(scope="session")
def _requires_dragon_runtime():
    try:
        get_gs_ret_cuid()
    except Exception:
        in_dragon_session = False
    else:
        in_dragon_session = True

    if not in_dragon_session:
        pytest.skip(
            "Dragon session not detected!" "Try running with `dragon -s -- -m pytest`?"
        )
    yield


@pytest.fixture(scope="session")
def _dragon_include_dir():
    yield os.path.join(DRAGON_BASE_DIR, "include")


@pytest.fixture(scope="session")
def _dragon_lib_dir():
    yield os.path.join(DRAGON_BASE_DIR, "lib")


@pytest.fixture(scope="session")
def _dragon_compile_args(_dragon_include_dir, _dragon_lib_dir):
    yield [
        "-fPIC",
        f"-I{os.fspath(_dragon_include_dir)}",
        f"-L{os.fspath(_dragon_lib_dir)}",
        f"-Wl,-rpath,{os.fspath(_dragon_lib_dir)}",
        "-ldragon",
    ]


@pytest.fixture
def cpp_dragon_compile(cpp_compile, _dragon_compile_args):
    def _compile(*args, extra_compile_args=(), **kwargs):
        return cpp_compile(
            *args,
            **kwargs,
            extra_compile_args=[
                *_dragon_compile_args,
                *extra_compile_args,
            ],
        )

    yield _compile


@pytest.fixture(scope="session")
def _persistent_ddict_client(_requires_dragon_runtime):
    ddict = DDict(
        managers_per_node=1, n_nodes=1, wait_for_keys=True, working_set_size=2
    )
    yield ddict


@pytest.fixture(scope="function")
def ddict(_persistent_ddict_client):
    yield _persistent_ddict_client
    _persistent_ddict_client.clear()


@pytest.fixture(scope="function")
def client(ddict):
    yield Client(descriptor=ddict.serialize(), timeout=5)
