cmake_minimum_required(VERSION 3.13)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/radex-standalone.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/radex-options.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/radex-backends.cmake)

radex_define_example_options()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/radex-version.cmake)
  project(RADEX-Tests
    VERSION ${RADEX_VERSION}
    LANGUAGES CXX
  )
  find_package(radex
    REQUIRED
    HINTS ${CMAKE_CURRENT_LIST_DIR}/../../install
  )
  radex_find_backends()
endif()

radex_setup_standalone_build()

# Catch2 =====================================================================
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch2)

# Compile test executable
add_executable(test-local-client test_local_client.cpp)
radex_set_target_install_rpath(
  test-local-client
  smartredis
  dragon
  Catch2::Catch2
  Catch2::Catch2Main
)
target_link_libraries(test-local-client PRIVATE radex::radex)
target_link_libraries(test-local-client PRIVATE Catch2::Catch2WithMain)
install(
    TARGETS test-local-client
    DESTINATION ${CMAKE_INSTALL_BINDIR}/tests
)
