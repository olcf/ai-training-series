#ifndef __RADEX_EXCEPTIONS_HPP__
#define __RADEX_EXCEPTIONS_HPP__

#include <stdexcept>

namespace radex {

/// Base class for every error raised by radex.
class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// A key was requested that is not present in the store.
class KeyNotFoundError : public Error {
  public:
    using Error::Error;
};

/// A key did not appear in the store before timing out.
class TimeoutError : public Error {
  public:
    using Error::Error;
};

/// The stored item does not match the item type requested.
class TypeMismatchError : public Error {
  public:
    using Error::Error;
};

/// A scalar was requested at a tensor key, or a tensor at a scalar key.
class RankMismatchError : public TypeMismatchError {
  public:
    using TypeMismatchError::TypeMismatchError;
};

/// The stored element type differs from the requested one. Distinct from
/// `RankMismatchError` so that callers can retry with another element type
/// without also retrying a request that asked for the wrong shape entirely.
class DTypeMismatchError : public TypeMismatchError {
  public:
    using TypeMismatchError::TypeMismatchError;
};

/// An item's metadata record could not be decoded.
class MetadataError : public Error {
  public:
    using Error::Error;
};

/// A client was requested for a backend that was disabled at build time.
class BackendUnavailableError : public Error {
  public:
    using Error::Error;
};

} // namespace radex

#endif // __RADEX_EXCEPTIONS_HPP__
