#ifndef __RADEX_BUILD_CONFIG_HPP__
#define __RADEX_BUILD_CONFIG_HPP__

// Defaults — overridden by the CMake-generated build_config.hpp.
/// Defined (to 1) if RaDex was built with Dragon backend support (`BUILD_DRAGON=ON`).
#undef RADEX_HAS_DRAGON
/// Defined (to 1) if RaDex was built with SmartRedis backend support (`BUILD_SMARTREDIS=ON`).
#undef RADEX_HAS_SMARTREDIS

#endif
