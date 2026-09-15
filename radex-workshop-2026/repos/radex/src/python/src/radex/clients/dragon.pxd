from libc.time cimport timespec

from radex.clients.core cimport IClient
from radex.utils.exceptions cimport raise_py_error


cdef extern from "radex/dragon.hpp" namespace "radex::drg::ddict":
    cdef cppclass Client(IClient):
        Client() except +raise_py_error
        Client(const char*, const timespec*) except +raise_py_error
