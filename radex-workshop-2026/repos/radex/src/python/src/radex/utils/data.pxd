from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport unique_ptr

cimport numpy as np
import numpy as np
np.import_array()

ctypedef fused SupportedType:
    int32_t
    int64_t
    float
    double


cdef extern from "radex/client.hpp" namespace "radex::detail":
    ctypedef size_t MetaInt

    cdef cppclass BytesBuffer:
        const void* get_ptr() except +
        size_t get_length() except +
        unique_ptr[np.uint8_t[]] release() except +

    cdef cppclass MetaData:
        size_t n_dims() except +
        const size_t *dims_ptr() except +
        size_t n_elements() except +
        DType type() except +

    cdef cppclass ItemInfo:
        const MetaData &metadata() except +
        const void *data() except +


cdef extern from "radex/client.hpp" namespace "radex::data":
    cdef enum class DType(MetaInt):
        INT32,
        INT64,
        FLOAT32,
        FLOAT64


# TODO: Don't like this method name
cdef inline make_ndarray(DType dtype, const void* data, MetaInt size):
    if dtype == DType.INT32:
        return np.copy(<np.int32_t[:size]> data)
    if dtype == DType.INT64:
        return np.copy(<np.int64_t[:size]> data)
    if dtype == DType.FLOAT32:
        return np.copy(<np.float32_t[:size]> data)
    if dtype == DType.FLOAT64:
        return np.copy(<np.float64_t[:size]> data)
    raise TypeError("Unknown type encountered")


cdef inline np.number coerce_py_objects_to_np_numbers(object value):
    _DEFAULT_FIXED_WIDTH_INT = np.int32
    _DEFAULT_FIXED_WIDTH_FLOAT = np.float64

    if isinstance(value, int):
        value = _DEFAULT_FIXED_WIDTH_INT(value)
    if isinstance(value, float):
        value = _DEFAULT_FIXED_WIDTH_FLOAT(value)
    if not isinstance(value, np.number):
        raise TypeError(
            f"Could not figure out how to coerce {type(value)} to a numpy.number"
        )
    return value


cdef inline construct_scalar(const ItemInfo &info):
    if info.metadata().n_dims() != 0:
        from radex.exceptions import RankMismatchError
        raise RankMismatchError(
            "Attempted to retrieve scalar at a key with a vector")

    cdef DType type_ = info.metadata().type()
    return make_ndarray(type_, info.data(), 1)[0]


cdef inline construct_tensor(const ItemInfo &info):
    cdef MetaInt n_dims = info.metadata().n_dims()
    if n_dims == 0:
        from radex.exceptions import RankMismatchError
        raise RankMismatchError(
            "Attempted to retrieve vector at a key with a scalar")

    cdef const MetaInt[:] dims = <const MetaInt[:n_dims]> info.metadata().dims_ptr()
    cdef n_elements = info.metadata().n_elements()
    cdef DType type_ = info.metadata().type()

    data = make_ndarray(type_, info.data(), n_elements)
    return data.reshape(dims)
