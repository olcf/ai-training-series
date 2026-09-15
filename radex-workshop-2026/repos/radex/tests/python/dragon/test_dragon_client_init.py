import os

import pytest
from dragon.data.ddict import DDict

import radex.clients.core
from radex.handles.handles import IncomingHandle, OutgoingHandle


def test_client_is_importable():
    from radex.clients.core import DragonClient


def test_init_raises_if_wait_for_keys_is_disabled(_requires_dragon_runtime):
    ddict = DDict(managers_per_node=1, n_nodes=1, wait_for_keys=False)
    with pytest.raises(RuntimeError, match=r"wait_for_keys"):
        radex.clients.core.DragonClient(descriptor=ddict.serialize(), timeout=1)


def test_init_from_env(ddict, monkeypatch):
    monkeypatch.setenv("RADEX_STORE", ddict.serialize())
    client = radex.clients.core.DragonClient()
    key = "some-value"
    some_value = 0.1
    client.put_scalar(OutgoingHandle(key), some_value)
    ret_val = client.get_scalar(IncomingHandle(key))
    assert some_value == ret_val
