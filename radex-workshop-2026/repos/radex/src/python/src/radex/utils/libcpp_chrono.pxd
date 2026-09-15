from libc.stdint cimport int64_t

cdef extern from "<chrono>" namespace "std::chrono" nogil:
    cdef cppclass milliseconds:
        milliseconds() except +
        milliseconds(int64_t) except +
        int64_t count()
