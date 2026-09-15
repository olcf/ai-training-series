#ifndef __RADEX_DRAGON_HPP__
#define __RADEX_DRAGON_HPP__

#include "radex/client_base.hpp"
#include "radex/build_config.hpp"

#ifdef RADEX_HAS_DRAGON
#include <dragon/dictionary.hpp>
#include <dragon/serializable.hpp>
#endif
#include <ctime>
#include <string_view>

namespace radex::drg::ddict {

#ifdef RADEX_HAS_DRAGON
/// RaDex client backed by a Dragon Distributed Dictionary (DDict).
class Client : public IClient {
  private:
    dragon::DDict<dragon::Serializable, dragon::Serializable> ddict;

    Client(dragon::DDict<dragon::Serializable, dragon::Serializable> ddict);

    void delete_key(std::string_view key) override;

    bool dormant_timeout_warning_triggered = false;

  public:
    /// Attach using a descriptor/timeout read from the environment.
    Client();
    /// @param descriptor A serialized DDict descriptor to attach to.
    /// @param timeout Timeout for attaching to the DDict.
    Client(const char *descriptor, const timespec *timeout);

    Client(const Client &other) = delete;
    Client(Client &&other) = default;
    Client &operator=(const Client &other) = delete;
    Client &operator=(Client &&other) = default;
    ~Client() = default;

    bool contains(std::string_view key) override;
    detail::BytesBuffer wait_for_bytes(std::string_view key,
                                       std::chrono::milliseconds timeout) override;
    void put_bytes(std::string_view key, const void *bytes,
                   detail::MetaInt length) override;
    radex::detail::BytesBuffer get_bytes(std::string_view key) override;
};
#else
/// Placeholder used when RaDex was built without Dragon support (`BUILD_DRAGON=OFF`).
class Client : public radex::unsupported_backend::Client {
  public:
    Client();
    Client(const char *descriptor, const timespec *timeout);
};
#endif

} // namespace radex::drg::ddict

#endif
