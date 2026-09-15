import pytest

from radex.handles.handles import IncomingHandle, OutgoingHandle


def test_delete_typed_item(client, random_np_value):
    key = "some-value-to-delete"
    client.put_scalar(OutgoingHandle(key), random_np_value)
    assert client.contains(key)

    client.delete_item(OutgoingHandle(key))

    assert not client.contains(key)
    assert not client.contains(f"meta::{key}")


def test_delete_raw_key(client, random_picklable):
    key = "some-object-to-delete"
    client.put_picklable(key, random_picklable)
    assert client.contains(key)

    client.delete_item(OutgoingHandle(key))

    assert not client.contains(key)
