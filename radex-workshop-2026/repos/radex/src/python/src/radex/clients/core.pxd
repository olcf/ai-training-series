from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.string_view cimport string_view

from radex.utils.libcpp_chrono cimport milliseconds
from radex.utils.data cimport (
    BytesBuffer,
    DType,
    ItemInfo,
    MetaInt as size_t,
)
from radex.handles.handles cimport (
    CXXIncomingHandle as IncomingHandle,
    CXXOutgoingHandle as OutgoingHandle,
)
from radex.utils.exceptions cimport raise_py_error

cimport numpy as np
np.import_array()

cdef extern from "radex/client.hpp" namespace "radex":
    cdef cppclass IClient:
        # >>> Start Virtual Methods >>>
        bint contains(string_view) except +raise_py_error
        void put_bytes(string_view, const void*, size_t) except +raise_py_error
        BytesBuffer get_bytes(string_view) except +raise_py_error
        BytesBuffer wait_for_bytes(string_view, milliseconds) except +raise_py_error
        void delete_key(string_view) except +
        # <<< End Virtual Methods <<<

        void put_scalar[T](const OutgoingHandle&, T) except +raise_py_error
        void put_tensor[T](const OutgoingHandle&,
                           const size_t*, size_t,
                           const T*, size_t) except +raise_py_error

        unique_ptr[ItemInfo] get_item_info_ptr(const IncomingHandle&) except +raise_py_error
        unique_ptr[ItemInfo] wait_for_item_info_ptr(
                const IncomingHandle&, milliseconds) except +raise_py_error
        void delete_item(const OutgoingHandle&) except +raise_py_error
