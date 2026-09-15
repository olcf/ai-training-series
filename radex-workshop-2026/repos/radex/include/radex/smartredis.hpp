#ifndef __RADEX_SMARTREDIS_HPP__
#define __RADEX_SMARTREDIS_HPP__

#include "radex/client_base.hpp"
#include "radex/build_config.hpp"

#ifdef RADEX_HAS_SMARTREDIS
#include <client.h>
#endif

#include <string_view>

namespace radex::redis::smartredis {

#ifdef RADEX_HAS_SMARTREDIS
/// RaDex client backed by a SmartRedis (Redis) connection.
class Client : public IClient {
  private:
    SmartRedis::Client client;
    Client(std::unique_ptr<SmartRedis::ConfigOptions> options,
           std::string_view logger_name);

    void delete_key(std::string_view key) override;

  public:
    /// Connect using configuration read from the environment.
    Client();
    /// @param logger_name Name used to tag log messages from this client.
    Client(std::string_view logger_name);

    Client(const Client &other) = delete;
    Client(Client &&other) = default;
    Client &operator=(const Client &other) = delete;
    Client &operator=(Client &&other) = default;
    ~Client() = default;

    bool contains(std::string_view key) override;
    void put_bytes(std::string_view key, const void *bytes,
                   detail::MetaInt length) override;
    radex::detail::BytesBuffer get_bytes(std::string_view key) override;
    radex::detail::BytesBuffer
    wait_for_bytes(std::string_view key,
             std::chrono::milliseconds timeout) override;
};
#else
/// Placeholder used when RaDex was built without SmartRedis support (`BUILD_SMARTREDIS=OFF`).
class Client : public radex::unsupported_backend::Client {
  public:
    Client();
    Client(std::string_view logger_name);
};
#endif

} // namespace radex::redis::smartredis

#endif
