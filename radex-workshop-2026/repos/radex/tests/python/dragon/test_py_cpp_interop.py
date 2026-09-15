import os
import textwrap
import time
from collections.abc import Iterable

import dragon
import numpy as np
import pytest
from dragon.native.process import Popen

from radex.handles.handles import IncomingHandle, OutgoingHandle


def _comma_seperate_ints(ints: Iterable[int]) -> str:
    return ", ".join(str(i) for i in ints)


def test_get_cpp_scalar(cpp_dragon_compile, ddict, client, np_dtype, cpp_type_name):
    key = "some-scalar"
    bin_ = cpp_dragon_compile(textwrap.dedent(f"""\
        #include "radex/dragon.hpp"
        #include "radex/handles.hpp"
        #include <cstdint>

        int main(void) {{
            timespec timeout {{5, 0}};
            radex::drg::ddict::Client client {{"{ddict.serialize()}", &timeout}};
            {cpp_type_name} value = 123;
            client.put_scalar(radex::data::OutgoingHandle("{key}"), value);
            return 0;
        }}
        """))

    assert not client.contains(key)
    proc = Popen(executable=os.fspath(bin_), args=[], env=os.environ)
    proc.wait()
    assert proc.returncode == 0

    assert client.contains(key)
    value = client.get_scalar(IncomingHandle(key))
    assert value.dtype == np_dtype
    assert value == np_dtype(123)


def test_get_cpp_tensor(cpp_dragon_compile, ddict, client, np_dtype, cpp_type_name):
    n_elements = 12
    tensors = {
        "tensor-1d": (12,),
        "tensor-2d": (6, 2),
        "tensor-3d": (2, 3, 2),
    }

    put_tensors = "\n".join(textwrap.dedent(f"""\
            client.put_tensor(radex::data::OutgoingHandle("{key}"),
                              {{{_comma_seperate_ints(shape)}}},
                              tensor);""") for key, shape in tensors.items())
    bin_ = cpp_dragon_compile(textwrap.dedent(f"""\
        #include "radex/dragon.hpp"
        #include "radex/handles.hpp"
        #include <vector>
        #include <numeric>
        #include <cstdint>

        int main(void) {{
            timespec timeout {{5, 0}};
            radex::drg::ddict::Client client {{"{ddict.serialize()}", &timeout}};
            std::vector<{cpp_type_name}> tensor({n_elements});
            std::iota(tensor.begin(), tensor.end(), 0);
            {put_tensors}
            return 0;
        }}
        """))

    assert not any(client.contains(tensor) for tensor in tensors)
    proc = Popen(executable=os.fspath(bin_), args=[], env=os.environ)
    proc.wait()
    assert proc.returncode == 0

    expected = np.arange(n_elements, dtype=np_dtype)
    for key, shape in tensors.items():
        assert client.contains(key)
        ret = client.get_tensor(IncomingHandle(key))
        assert ret.dtype == np_dtype
        assert ret.shape == shape
        assert (ret == expected.reshape(shape)).all()


def test_put_py_scalar(cpp_dragon_compile, ddict, client, np_dtype, cpp_type_name):
    key = "some-scalar"
    bin_ = cpp_dragon_compile(textwrap.dedent(f"""\
        #include "radex/dragon.hpp"
        #include "radex/handles.hpp"
        #include <cstdint>

        int main(void) {{
            timespec timeout {{5, 0}};
            radex::drg::ddict::Client client {{"{ddict.serialize()}", &timeout}};
            auto x = client.get_scalar<{cpp_type_name}>(
                radex::data::IncomingHandle("{key}"));
            return x == 34 ? 0 : 1;
        }}
        """))

    assert not client.contains(key)
    client.put_scalar(OutgoingHandle(key), np_dtype(34))
    assert client.contains(key)

    proc = Popen(executable=os.fspath(bin_), args=[], env=os.environ)
    proc.wait()
    assert proc.returncode == 0


def test_put_py_tensor(cpp_dragon_compile, ddict, client, np_dtype, cpp_type_name):
    n_elements = 16
    tensors = {
        "tensor-1d": (16,),
        "tensor-2d": (4, 4),
        "tensor-4d": (2, 2, 2, 2),
    }

    eval_tensors = "\n".join(
        f'eval_tensor(client, "{key}", {{ {_comma_seperate_ints(shape)} }}, data);\n'
        for key, shape in tensors.items()
    )

    bin_ = cpp_dragon_compile(textwrap.dedent(f"""\
        #include "radex/dragon.hpp"
        #include "radex/handles.hpp"
        #include <cstdint>
        #include <vector>
        #include <numeric>
        #include <exception>

        template <typename T>
        void eval_tensor(radex::IClient& client,
                         const std::string& key,
                         const std::vector<radex::detail::MetaInt>& expected_dims,
                         const std::vector<T>& expected_data) {{
            auto [dims, data] = client.get_tensor<T>(radex::data::IncomingHandle(key));
            if (expected_dims != dims)
                throw std::logic_error(key + ": Dims do not match");
            if (expected_data != data)
                throw std::logic_error(key + ": Data does not match");
        }}

        int main(void) {{
            timespec timeout {{5, 0}};
            radex::drg::ddict::Client client {{"{ddict.serialize()}", &timeout}};

            std::vector<{cpp_type_name}> data({n_elements});
            std::iota(data.begin(), data.end(), 0);

            {eval_tensors}

            return 0;
        }}
        """))

    assert not any(client.contains(tensor) for tensor in tensors)
    for key, shape in tensors.items():
        tensor = np.arange(n_elements, dtype=np_dtype).reshape(shape)
        client.put_tensor(OutgoingHandle(key), tensor)
    assert all(client.contains(tensor) for tensor in tensors)

    proc = Popen(executable=os.fspath(bin_), args=[], env=os.environ)
    proc.wait()
    assert proc.returncode == 0


@pytest.mark.slow
def test_wait_for_scalars(
    cpp_dragon_compile,
    ddict,
    client,
    np_dtype,
    cpp_type_name,
):
    py_key = "some-py-val"
    cpp_key = "some-cpp-val"

    py_put_delay = 3  # seconds
    cpp_put_delay = 3  # seconds

    bin_ = cpp_dragon_compile(textwrap.dedent(f"""\
        #include "radex/dragon.hpp"
        #include "radex/handles.hpp"
        #include <iostream>
        #include <chrono>
        #include <thread>

        int main(void) {{
            timespec timeout {{5, 0}};
            radex::drg::ddict::Client client {{"{ddict.serialize()}", &timeout}};
            {cpp_type_name} value = 123;
            std::this_thread::sleep_for(std::chrono::seconds({cpp_put_delay}));
            client.put_scalar(radex::data::OutgoingHandle("{cpp_key}"), value);

            auto x = client.wait_for_scalar<{cpp_type_name}>(
                radex::data::IncomingHandle("{py_key}"), std::chrono::milliseconds({10000}));
            return x == 12 ? 0 : 1;
        }}
        """))

    assert not client.contains(py_key)
    assert not client.contains(cpp_key)

    try:
        proc = Popen(executable=os.fspath(bin_), args=[])
        start_t = time.perf_counter()
        value = client.wait_for_scalar(IncomingHandle(cpp_key), 10)
        py_wait_time = time.perf_counter() - start_t

        time.sleep(py_put_delay)
        client.put_scalar(OutgoingHandle(py_key), np_dtype(12))
    except Exception:
        proc.kill()
    finally:
        proc.wait()

    wait_interval = 2.0
    assert cpp_put_delay - wait_interval < py_wait_time < cpp_put_delay + wait_interval
    assert value.dtype == np_dtype
    assert value == np_dtype(123)
    assert proc.returncode == 0


@pytest.mark.slow
def test_wait_for_tensors(
    cpp_dragon_compile,
    ddict,
    client,
    np_dtype,
    cpp_type_name,
):
    py_key = "some-py-val"
    cpp_key = "some-cpp-val"

    py_put_delay = 2  # seconds
    cpp_put_delay = 2  # seconds

    cpp_tensor_size = 36
    cpp_tensor_shape = (4, 3, 3)
    expected_cpp_tensor = np.arange(cpp_tensor_size, dtype=np_dtype).reshape(
        cpp_tensor_shape
    )

    py_tensor_data = [12, 34, 56, 78, 90, 0]
    py_tenosr_shape = (2, 3)
    py_tensor = np.array(py_tensor_data, dtype=np_dtype).reshape(py_tenosr_shape)

    bin_ = cpp_dragon_compile(textwrap.dedent(f"""\
        #include "radex/dragon.hpp"
        #include "radex/handles.hpp"
        #include <chrono>
        #include <numeric>
        #include <thread>
        #include <vector>

        int main(void) {{
            timespec timeout {{5, 0}};
            radex::drg::ddict::Client client {{"{ddict.serialize()}", &timeout}};

            std::vector<{cpp_type_name}> cpp_tensor({cpp_tensor_size});
            std::iota(cpp_tensor.begin(), cpp_tensor.end(), 0);

            std::vector<{cpp_type_name}> expected_py_data
                {{{_comma_seperate_ints(py_tensor_data)}}};
            std::vector<radex::detail::MetaInt> expected_py_dims
                {{{_comma_seperate_ints(py_tenosr_shape)}}};

            std::this_thread::sleep_for(std::chrono::seconds({cpp_put_delay}));
            client.put_tensor(
                radex::data::OutgoingHandle("{cpp_key}"),
                {{{_comma_seperate_ints(cpp_tensor_shape)}}},
                cpp_tensor);

            auto [dims, data] = client.wait_for_tensor<{cpp_type_name}>(
                radex::data::IncomingHandle("{py_key}"), std::chrono::milliseconds({10000}));
            if (dims != expected_py_dims)
                throw std::logic_error("Dims did not match");
            if (data != expected_py_data)
                throw std::logic_error("Data did not match");
            return 0;
        }}
        """))
    assert not client.contains(py_key)
    assert not client.contains(cpp_key)

    try:
        proc = Popen(executable=os.fspath(bin_), args=[])
        start_t = time.perf_counter()
        cpp_tensor = client.wait_for_tensor(IncomingHandle(cpp_key), 10)
        py_wait_time = time.perf_counter() - start_t

        time.sleep(py_put_delay)
        client.put_tensor(OutgoingHandle(py_key), py_tensor)
    except Exception:
        proc.kill()
    finally:
        proc.wait()
    wait_interval = 2.0
    assert cpp_put_delay - wait_interval < py_wait_time < cpp_put_delay + wait_interval
    assert cpp_tensor.dtype == expected_cpp_tensor.dtype == np_dtype
    assert (cpp_tensor == expected_cpp_tensor).all()
    assert proc.returncode == 0
