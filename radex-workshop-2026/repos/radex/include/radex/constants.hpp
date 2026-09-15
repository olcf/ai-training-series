#ifndef __RADEX_CONSTANTS_HPP__
#define __RADEX_CONSTANTS_HPP__

#include <string>

/// Environment variable naming which backend to use (e.g. "dragon", "smartredis").
inline const std::string RADEX_STORE_VAR = "RADEX_STORE";
/// Environment variable holding backend-specific connection options.
inline const std::string RADEX_STORE_OPTS_VAR = "RADEX_STORE_OPTS";
/// Environment variable overriding the default backend connection timeout.
inline const std::string RADEX_CONNECTION_TIMEOUT_VAR = "RADEX_CONNECTION_TIMEOUT";
/// Default backend connection timeout, in seconds, if `RADEX_CONNECTION_TIMEOUT` is unset.
constexpr unsigned int RADEX_DEFAULT_CONNECTION_TIMEOUT_SECONDS = 5;

/// Environment variable overriding the default `wait_for_*` poll interval.
inline const std::string RADEX_POLL_INTERVAL_VAR = "RADEX_POLL_INTERVAL";
/// Default `wait_for_*` poll interval, in milliseconds, if `RADEX_POLL_INTERVAL` is unset.
constexpr unsigned int RADEX_DEFAULT_POLL_INTERVAL_MILLISECONDS = 100;

#endif
