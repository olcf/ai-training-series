from libcpp.string_view cimport string_view

from radex.clients.core cimport IClient
from radex.utils.exceptions cimport raise_py_error


cdef extern from "radex/smartredis.hpp" namespace "radex::redis::smartredis":
    cdef cppclass Client(IClient):
        Client() except +raise_py_error
        Client(string_view) except +raise_py_error
