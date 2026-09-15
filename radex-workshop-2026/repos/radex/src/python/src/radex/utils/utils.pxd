from libcpp.string cimport string

ctypedef string EncodedStr


cdef inline EncodedStr encode_str(str s):
    return s.encode("utf-8")
