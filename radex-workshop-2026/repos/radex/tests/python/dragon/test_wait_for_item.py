import contextlib
import os
import sys
import time

import pytest

from radex.clients.core import DragonClient
from radex.handles.handles import IncomingHandle, OutgoingHandle

# >>> Test Utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# FIXME: For some reason these tests fail in the Github Actions MacOS runner via
# a timeout. These do pass on local MacOS environments.
skip_macos_on_github_actions = (sys.platform == "darwin") and (
    os.environ.get("GITHUB_ACTIONS", "false") == "true"
)
pytestmark = pytest.mark.skipif(
    skip_macos_on_github_actions,
    reason="Tests currently fail on Github Actions under MacOS",
)

_TRIVIAL_WAIT_TIME_LIMIT = 0.1  # seconds
# FIXME: We should expose the poll rate through the `radex` namespace
#        rather than looking for magic env vars in the tests
_POLL_INTERVAL = os.environ.get("RADEX_POLL_INTERVAL", 100) / 1000  # seconds


@pytest.fixture(
    params=[
        pytest.param(0.05, id="set-immediately"),
        pytest.param(
            ((_POLL_INTERVAL * 4) - (_POLL_INTERVAL / 5)),
            id="set-before-interval-end",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            ((_POLL_INTERVAL * 4) + (_POLL_INTERVAL / 5)),
            id="set-after-interval-end",
            marks=[pytest.mark.slow],
        ),
    ]
)
def wait_time_delay(request):
    yield request.param


@contextlib.contextmanager
def _is_acceptable_wait_for_item_time(expected_delay):
    start_t = time.perf_counter()
    yield
    run_t = time.perf_counter() - start_t

    # FIXME: We should expose the poll rate through the `radex` namespace
    #        rather than looking for magic env vars in the tests
    poll_rate = os.environ.get("RADEX_POLL_INTERVAL", 100)  # milliseconds
    process_overhead = 0.1
    min_t = expected_delay
    max_t = expected_delay + poll_rate + process_overhead

    assert min_t < run_t, "Runtime too low! Are you sure the delay was observed?"
    assert run_t < max_t, "Wait for item time appears slower than polling"


@pytest.fixture
def is_acceptable_wait_for_item_time():
    yield _is_acceptable_wait_for_item_time


# <<< End Test Utils <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def test_wait_for_present_scalar(client, np_dtype, random_np_value):
    key = "some-value"
    assert not client.contains(key)

    client.put_scalar(OutgoingHandle(key), random_np_value)
    assert client.contains(key)

    start_t = time.perf_counter()
    ret_val = client.wait_for_scalar(IncomingHandle(key), timeout=10)
    end_t = time.perf_counter()
    assert end_t - start_t < _TRIVIAL_WAIT_TIME_LIMIT
    assert random_np_value.dtype == ret_val.dtype == np_dtype
    assert random_np_value == ret_val


def test_wait_for_present_tensor(client, np_dtype, random_np_tensor):
    key = "some-tensor"
    assert not client.contains(key)

    client.put_tensor(OutgoingHandle(key), random_np_tensor)
    assert client.contains(key)

    start_t = time.perf_counter()
    ret_val = client.wait_for_tensor(IncomingHandle(key), timeout=10)
    end_t = time.perf_counter()
    assert end_t - start_t < _TRIVIAL_WAIT_TIME_LIMIT
    assert random_np_tensor.dtype == ret_val.dtype == np_dtype
    assert (random_np_tensor == ret_val).all()


def test_wait_for_present_picklable(client, random_picklable):
    key = "some-obj"
    assert not client.contains(key)

    client.put_picklable(key, random_picklable)
    assert client.contains(key)

    start_t = time.perf_counter()
    ret_val = client.wait_for_picklable(key, timeout=10)
    end_t = time.perf_counter()
    assert end_t - start_t < _TRIVIAL_WAIT_TIME_LIMIT
    assert type(random_picklable) == type(ret_val)
    assert random_picklable == ret_val


def test_put_and_wait_for_scalar(
    ddict,
    client,
    np_dtype,
    random_np_value,
    wait_time_delay,
    is_acceptable_wait_for_item_time,
):
    process = pytest.importorskip("dragon.native.process")

    key = "some-value"
    assert not client.contains(key)

    def put_key_after(dd, key, value, delay):
        start_t = time.perf_counter()
        client = DragonClient(descriptor=dd, timeout=1)
        time.sleep(max(delay - (time.perf_counter() - start_t), 0))
        client.put_scalar(OutgoingHandle(key), value)

    proc = process.Process(
        target=put_key_after,
        args=(ddict.serialize(), key, random_np_value, wait_time_delay),
    )
    try:
        with is_acceptable_wait_for_item_time(wait_time_delay):
            proc.start()
            ret_val = client.wait_for_scalar(IncomingHandle(key), timeout=10)
    finally:
        proc.join()

    assert random_np_value.dtype == ret_val.dtype == np_dtype
    assert random_np_value == ret_val


def test_put_and_wait_for_tensor(
    ddict,
    client,
    np_dtype,
    random_np_tensor,
    wait_time_delay,
    is_acceptable_wait_for_item_time,
):
    process = pytest.importorskip("dragon.native.process")

    key = "some-value"
    assert not client.contains(key)

    def put_key_after(dd, key, value, delay):
        start_t = time.perf_counter()
        client = DragonClient(descriptor=dd, timeout=1)
        time.sleep(max(delay - (time.perf_counter() - start_t), 0))
        client.put_tensor(OutgoingHandle(key), value)

    proc = process.Process(
        target=put_key_after,
        args=(ddict.serialize(), key, random_np_tensor, wait_time_delay),
    )
    try:
        with is_acceptable_wait_for_item_time(wait_time_delay):
            proc.start()
            ret_val = client.wait_for_tensor(IncomingHandle(key), timeout=10)
    finally:
        proc.join()

    assert random_np_tensor.dtype == ret_val.dtype == np_dtype
    assert (random_np_tensor == ret_val).all()


def test_put_and_wait_for_picklable(
    ddict, client, random_picklable, wait_time_delay, is_acceptable_wait_for_item_time
):
    process = pytest.importorskip("dragon.native.process")

    key = "some-obj"
    assert not client.contains(key)

    def put_key_after(dd, key, value, delay):
        start_t = time.perf_counter()
        client = DragonClient(descriptor=dd, timeout=1)
        time.sleep(max(delay - (time.perf_counter() - start_t), 0))
        client.put_picklable(key, value)

    proc = process.Process(
        target=put_key_after,
        args=(ddict.serialize(), key, random_picklable, wait_time_delay),
    )
    try:
        with is_acceptable_wait_for_item_time(wait_time_delay):
            proc.start()
            ret_val = client.wait_for_picklable(key, timeout=10)
    finally:
        proc.join()

    assert type(random_picklable) == type(ret_val)
    assert random_picklable == ret_val
