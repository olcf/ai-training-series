#ifndef __RADEX_PY_EXCEPTION_TRANSLATION_HPP__
#define __RADEX_PY_EXCEPTION_TRANSLATION_HPP__

#include <Python.h>

#include <new>
#include <stdexcept>

#include "radex/exceptions.hpp"

namespace radex_py {

namespace detail {

inline void set_error(const char *name, const char *what) {
    PyObject *module = PyImport_ImportModule("radex.exceptions");
    if (module == nullptr) {
        return; // ImportError is already set
    }

    PyObject *exc_type = PyObject_GetAttrString(module, name);
    if (exc_type != nullptr) {
        PyErr_SetString(exc_type, what);
        Py_DECREF(exc_type);
    }

    Py_DECREF(module);
}

} // namespace detail

/// Translate the in-flight C++ exception into the matching Python one.
///
/// Only valid inside a catch block, which is where Cython's
/// `except +raise_py_error` invokes it. Derived types must be caught before
/// their bases or the subclass information is lost.
inline void raise_py_error() {
    try {
        throw;
    } catch (const radex::KeyNotFoundError &e) {
        detail::set_error("KeyNotFoundError", e.what());
    } catch (const radex::TimeoutError &e) {
        detail::set_error("TimeoutError", e.what());
    } catch (const radex::RankMismatchError &e) {
        detail::set_error("RankMismatchError", e.what());
    } catch (const radex::DTypeMismatchError &e) {
        detail::set_error("DTypeMismatchError", e.what());
    } catch (const radex::TypeMismatchError &e) {
        detail::set_error("TypeMismatchError", e.what());
    } catch (const radex::MetadataError &e) {
        detail::set_error("MetadataError", e.what());
    } catch (const radex::BackendUnavailableError &e) {
        detail::set_error("BackendUnavailableError", e.what());
    } catch (const radex::Error &e) {
        detail::set_error("RadexError", e.what());
    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::out_of_range &e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const std::bad_alloc &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception");
    }
}

} // namespace radex_py

#endif // __RADEX_PY_EXCEPTION_TRANSLATION_HPP__
