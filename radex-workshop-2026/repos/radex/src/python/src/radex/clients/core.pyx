import cython

from libc.stdint cimport uint64_t, int32_t, int64_t
from libc.time cimport timespec
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from radex.clients.core cimport IClient
from radex.clients.dragon cimport Client as _CXXDragonClient
from radex.clients.redis cimport Client as _CXXSRClient
from radex.utils.data cimport (
    BytesBuffer,
    DType,
    ItemInfo,
    MetaInt as size_t,
    SupportedType,
    coerce_py_objects_to_np_numbers,
    construct_scalar,
    construct_tensor,
)
from radex.utils.utils cimport EncodedStr, encode_str
from radex.utils.libcpp_chrono cimport milliseconds
from radex.handles.handles cimport IncomingHandle, OutgoingHandle

import cloudpickle

cimport numpy as np
import numpy as np
np.import_array()


cdef class PyClient:
    cdef IClient *_client

    def __dealloc__(self):
        # TODO: The clients really should be using a unique_ptr instead
        if self._client != NULL:
            del self._client
            self._client = NULL

    def contains(self, str key):
        """Check whether a key is present in the backing store.

        Args:
            key (str): The key name to look up.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return self._client.contains(encode_str(key).c_str())

    def put_scalar(self, OutgoingHandle handle, object value not None):
        """Store a scalar value under the given handle.

        Args:
            handle (OutgoingHandle): The handle naming the key to write to.
            value (int | float | numpy.int32 | numpy.int64 | numpy.float32 |
                numpy.float64): A scalar value coercible to `numpy.int32`,
                `numpy.int64`, `numpy.float32`, or `numpy.float64`.

        Raises:
            TypeError: If `value` cannot be coerced to a supported numpy
                scalar type.
        """
        cdef np.number val = coerce_py_objects_to_np_numbers(value)
        return self._put_scalar(handle, val)

    def delete_item(self, OutgoingHandle handle):
        """Delete a typed value and its associated metadata."""
        self._client.delete_item(handle.unwrap()[0])

    def _put_scalar(self, OutgoingHandle handle, np.number value not None):
        # FIXME: Get rid of this ugly swith statment. Ideally we could used the
        #        fused `SupportedType` type, but there seems to be a known
        #        issue with how it dispatches
        # https://github.com/cython/cython/issues/4932
        if isinstance(value, np.int32):
            return self._client.put_scalar[int32_t](handle.unwrap()[0], value)
        if isinstance(value, np.int64):
            return self._client.put_scalar[int64_t](handle.unwrap()[0], value)
        if isinstance(value, np.float32):
            return self._client.put_scalar[np.float32_t](handle.unwrap()[0], value)
        if isinstance(value, np.float64):
            return self._client.put_scalar[np.float64_t](handle.unwrap()[0], value)
        raise TypeError(f"Unsupported data type: {type(value)}")

    def get_scalar(self, IncomingHandle handle):
        """Read a scalar value previously written under the given handle.

        Args:
            handle (IncomingHandle): The handle naming the key to read.

        Returns:
            numpy.int32 | numpy.int64 | numpy.float32 | numpy.float64: The
                scalar value.
        """
        cdef unique_ptr[ItemInfo] info = self._client.get_item_info_ptr(
                handle.unwrap()[0])
        return construct_scalar(info.get()[0])

    def wait_for_scalar(self, IncomingHandle handle, float timeout):
        """Block until a scalar value is available under the given handle.

        Args:
            handle (IncomingHandle): The handle naming the key to read.
            timeout (float): Maximum time to wait, in seconds.

        Returns:
            numpy.int32 | numpy.int64 | numpy.float32 | numpy.float64: The
                scalar value, once it becomes available.
        """
        cdef milliseconds timeout_ = milliseconds(<int64_t>(timeout * 1000))
        cdef unique_ptr[ItemInfo] info = self._client.wait_for_item_info_ptr(
                handle.unwrap()[0], timeout_)
        return construct_scalar(info.get()[0])

    def put_tensor(self, OutgoingHandle handle, np.ndarray tensor not None):
        """Store an n-dimensional tensor under the given handle.

        Args:
            handle (OutgoingHandle): The handle naming the key to write to.
            tensor (numpy.ndarray): An array with a supported dtype
                (`int32`, `int64`, `float32`, or `float64`). Its shape is
                preserved.
        """
        cdef np.ndarray[np.uintp_t, ndim=1] dims_arr = np.asarray(
                (<object>tensor).shape, dtype=np.uintp)
        self._put_tensor(handle, dims_arr, tensor.ravel())

    def _put_tensor(
        self,
        OutgoingHandle handle,
        const size_t[:] dims not None,
        const SupportedType[:] data not None,
    ):
        return self._client.put_tensor(handle.unwrap()[0],
                                       &dims[0], dims.size,
                                       &data[0], data.size)

    def get_tensor(self, IncomingHandle handle):
        """Read a tensor previously written under the given handle.

        Args:
            handle (IncomingHandle): The handle naming the key to read.

        Returns:
            numpy.ndarray: An array with the shape and dtype it was written
                with.
        """
        cdef unique_ptr[ItemInfo] info = self._client.get_item_info_ptr(
                handle.unwrap()[0])
        return construct_tensor(info.get()[0])

    def wait_for_tensor(self, IncomingHandle handle, float timeout):
        """Block until a tensor is available under the given handle.

        Args:
            handle (IncomingHandle): The handle naming the key to read.
            timeout (float): Maximum time to wait, in seconds.

        Returns:
            numpy.ndarray: An array with the shape and dtype it was written
                with, once it becomes available.
        """
        cdef milliseconds timeout_ = milliseconds(<int64_t>(timeout * 1000))
        cdef unique_ptr[ItemInfo] info = self._client.wait_for_item_info_ptr(
                handle.unwrap()[0], timeout_)
        return construct_tensor(info.get()[0])

    def put_picklable(self, str key, object picklable):
        cdef string key_ = encode_str(key)
        cdef bytes bytes_ = cloudpickle.dumps(picklable)
        cdef void* ptr = <void*><char*>bytes_
        cdef size_t len_ = len(bytes_)
        self._client.put_bytes(key_.c_str(), ptr, len_)

    def get_picklable(self, str key):
        cdef string key_ = encode_str(key)
        cdef BytesBuffer buf = self._client.get_bytes(key_.c_str())

        cdef size_t len_ = buf.get_length()
        cdef np.uint8_t *ptr = <np.uint8_t*>buf.get_ptr()

        cdef bytes bytes_ = <bytes>ptr[:len_]
        return cloudpickle.loads(bytes_)

    def wait_for_picklable(self, str key, float timeout):
        cdef string key_ = encode_str(key)
        cdef milliseconds timeout_ = milliseconds(<int64_t>(timeout * 1000))
        cdef BytesBuffer buf = self._client.wait_for_bytes(key_.c_str(), timeout_)

        cdef size_t len_ = buf.get_length()
        cdef np.uint8_t *ptr = <np.uint8_t*>buf.get_ptr()

        cdef bytes bytes_ = <bytes>ptr[:len_]
        return cloudpickle.loads(bytes_)


cdef class DragonClient(PyClient):
    """Create a client that attaches to a Dragon DDict, either from arguments or the environment.

    Args:
        descriptor (str | None): The serialized DDict descriptor to attach
            to. If omitted (along with `timeout`), the descriptor and
            timeout are instead read from the environment.
        timeout (int | None): Timeout, in seconds, for attaching to the
            DDict. Must be set together with `descriptor`, or omitted
            together with it.

    Raises:
        ValueError: If exactly one of `descriptor`/`timeout` is set, or
            if `descriptor` is empty or `timeout` is negative.
    """
    def __cinit__(
        self, descriptor: str | None = None, timeout: int | None = None
    ):
        if (descriptor is None) and (timeout is None):
            self._init_from_env()
        elif (descriptor is not None) and (timeout is not None):
            self._init_from_args(descriptor, timeout)
        else:
            raise ValueError(
                "Both descriptor and timeout must be set or both be None"
            )

    cdef void _init_from_env(self):
        self._client = new _CXXDragonClient()

    cdef void _init_from_args(self, str descriptor, int timeout):
        if len(descriptor) == 0:
            raise ValueError("DDict descriptor cannot be an empty string")
        if timeout < 0:
            raise ValueError("timeout must be positive")
        cdef EncodedStr desc = encode_str(descriptor)
        cdef timespec spec = timespec(timeout, 0)
        self._client = new _CXXDragonClient(desc.c_str(), &spec)


cdef class RedisClient(PyClient):
    def __cinit__(self, logger_name: str | None = None):
        cdef EncodedStr name

        if logger_name is None:
            self._client = new _CXXSRClient()
        elif isinstance(logger_name, str):
            name = encode_str(logger_name)
            self._client = new _CXXSRClient(name.c_str())
        else:
            raise TypeError(f"Unexpected logger name type: {type(logger_name)}")
