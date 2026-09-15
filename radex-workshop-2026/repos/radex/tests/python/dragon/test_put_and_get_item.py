import math

import numpy as np
import pytest

import radex.exceptions as ex
from radex.handles.handles import IncomingHandle, OutgoingHandle


def test_put_and_get_scalar(client, np_dtype, random_np_value):
    key = "some-value"
    assert not client.contains(key)

    client.put_scalar(OutgoingHandle(key), random_np_value)
    assert client.contains(key)

    ret_val = client.get_scalar(IncomingHandle(key))
    assert random_np_value.dtype == ret_val.dtype == np_dtype
    assert random_np_value == ret_val


def test_put_and_get_tensor(client, random_np_tensor, np_dtype):
    key = "some-tensor"
    assert not client.contains(key)

    client.put_tensor(OutgoingHandle(key), random_np_tensor)
    assert client.contains(key)
    ret_tensor = client.get_tensor(IncomingHandle(key))

    assert random_np_tensor.dtype == ret_tensor.dtype == np_dtype
    assert random_np_tensor.shape == ret_tensor.shape
    assert (random_np_tensor == ret_tensor).all()


def test_get_missing_key_raises(client):
    # Regression: a get on an absent key used to block on the key instead
    key = "no-such-key"
    assert not client.contains(key)

    with pytest.raises(ex.KeyNotFoundError):
        client.get_scalar(IncomingHandle(key))

    with pytest.raises(ex.KeyNotFoundError):
        client.get_tensor(IncomingHandle(key))


def test_reading_a_tensor_as_a_scalar_raises(client):
    key = "a-tensor"
    client.put_tensor(OutgoingHandle(key), np.arange(4, dtype=np.float64))

    with pytest.raises(ex.RankMismatchError):
        client.get_scalar(IncomingHandle(key))


def test_radex_errors_remain_runtime_errors(client):
    # Callers written before the exception hierarchy caught RuntimeError
    with pytest.raises(RuntimeError):
        client.get_scalar(IncomingHandle("still-no-such-key"))


@pytest.mark.parametrize("size", [pytest.param(int(1e6), id="size-1MB")])
@pytest.mark.parametrize(
    "n_dims", [pytest.param(n, id=f"shape-{n}D") for n in range(1, 5)]
)
def test_put_and_get_tensor_of_size(client, np_dtype, size, n_dims):
    min_n_elements = math.ceil(size / np.dtype(np_dtype).itemsize)
    n_elements_per_dim = math.ceil(math.pow(min_n_elements, 1 / n_dims))
    shape = tuple([n_elements_per_dim] * n_dims)
    n_elements = math.prod(shape)
    assert n_elements >= min_n_elements, f"Tensor is smaller than {size} bytes"
    assert len(shape) == n_dims, f"Got {len(shape)} dims but expected {n_dims}"

    key = "spam-eggs"
    tensor = np.arange(n_elements, dtype=np_dtype).reshape(shape)

    client.put_tensor(OutgoingHandle(key), tensor)
    assert client.contains(key)
    ret_tensor = client.get_tensor(IncomingHandle(key))

    assert tensor.dtype == ret_tensor.dtype == np_dtype
    assert tensor.shape == ret_tensor.shape == shape
    assert (tensor == ret_tensor).all()


@pytest.mark.parametrize(
    "value, expected_dtype",
    [
        pytest.param(value, dtype, id=f"value-{value}-dtype-{dtype.__name__}")
        for value, dtype in [
            (-123, np.int32),
            (0, np.int32),
            (123, np.int32),
            (-10.0, np.float64),
            (-22.5, np.float64),
            (0.0, np.float64),
            (0.75, np.float64),
        ]
    ],
)
def test_put_and_get_native_py_scalars(client, value, expected_dtype):
    key = "some-value"
    client.put_scalar(OutgoingHandle(key), value)
    ret = client.get_scalar(IncomingHandle(key))
    assert ret.dtype == expected_dtype
    assert ret == value


def test_put_and_get_pickleable(client, random_picklable):
    key = "some-obj"
    assert not client.contains(key)
    client.put_picklable(key, random_picklable)
    assert client.contains(key)

    ret_val = client.get_picklable(key)
    assert type(random_picklable) == type(ret_val)
    assert random_picklable == ret_val
