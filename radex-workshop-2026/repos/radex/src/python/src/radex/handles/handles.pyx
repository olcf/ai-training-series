import cython

from libcpp.memory cimport unique_ptr, make_unique

from radex.utils.utils cimport encode_str, EncodedStr
from radex.handles.handles cimport CXXIncomingHandle, CXXOutgoingHandle


cdef class IncomingHandle:
    """A named handle for reading a value from the store.

    Args:
        handle_id (str): The key name to read from.
    """

    def __cinit__(self, str handle_id):
        cdef EncodedStr id_ = encode_str(handle_id)
        self._handle = make_unique[CXXIncomingHandle](id_.c_str())

    cdef CXXIncomingHandle* unwrap(self):
        return self._handle.get()


cdef class OutgoingHandle:
    """A named handle for writing a value to the store.

    Args:
        handle_id (str): The key name to write to.
    """

    def __cinit__(self, str handle_id):
        cdef EncodedStr id_ = encode_str(handle_id)
        self._handle = make_unique[CXXOutgoingHandle](id_.c_str())

    cdef CXXOutgoingHandle* unwrap(self):
        return self._handle.get()
