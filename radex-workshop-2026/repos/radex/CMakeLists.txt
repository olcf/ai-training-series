cmake_minimum_required(VERSION 3.13)

# Policy for VERSION in project()
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/radex-version.cmake)
project(radex VERSION ${RADEX_VERSION} LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/radex-standalone.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/radex-options.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/radex-backends.cmake)

radex_define_root_options()
radex_language_support()
find_package(Threads REQUIRED)
radex_find_backends()

radex_setup_standalone_build()

# ============================================================================
# Options
# ============================================================================
if(ENABLE_CXX AND (BUILD_SHARED_LIBS OR BUILD_STATIC_LIBS))
    add_subdirectory(src)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests/cpp)
endif()