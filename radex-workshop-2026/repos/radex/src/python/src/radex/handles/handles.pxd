from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.string_view cimport string_view

cdef extern from "radex/handles.hpp" namespace "radex::data":
    cdef cppclass IHandle:
        string key() except +
        string metadata_key() except +

    cdef cppclass CXXIncomingHandle "radex::data::IncomingHandle"(IHandle):
        CXXIncomingHandle(string_view)

    cdef cppclass CXXOutgoingHandle "radex::data::OutgoingHandle"(IHandle):
        CXXOutgoingHandle(string_view)


cdef class IncomingHandle:
    cdef unique_ptr[CXXIncomingHandle] _handle
    cdef CXXIncomingHandle* unwrap(self)


cdef class OutgoingHandle:
    cdef unique_ptr[CXXOutgoingHandle] _handle
    cdef CXXOutgoingHandle* unwrap(self)
