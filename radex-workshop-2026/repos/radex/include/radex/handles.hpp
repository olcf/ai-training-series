#ifndef __RADEX_HANDLES_HPP__
#define __RADEX_HANDLES_HPP__

#include <string>
#include <string_view>

namespace radex::data {

/// Base interface for a named reference to a value in a RaDex backend.
class IHandle {
  public:
    /// The key under which the value's raw bytes are stored.
    virtual auto key() const -> std::string = 0;
    /// The key under which the value's metadata (type, dimensions) is stored.
    virtual auto metadata_key() const -> std::string = 0;
    virtual ~IHandle() {}
};

/// A handle for reading a value from the store.
class IncomingHandle : public IHandle {
  private:
    const std::string name;

  public:
    /// @param name The key name to read from.
    IncomingHandle(std::string_view name) : name{name} {}
    auto key() const -> std::string override;
    auto metadata_key() const -> std::string override;
};

/// A handle for writing a value to the store.
class OutgoingHandle : public IHandle {
  private:
    const std::string name;

  public:
    /// @param name The key name to write to.
    OutgoingHandle(std::string_view name) : name{name} {}
    auto key() const -> std::string override;
    auto metadata_key() const -> std::string override;
};

} // namespace radex::data

#endif
