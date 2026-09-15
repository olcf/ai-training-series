cmake_minimum_required(VERSION 3.13)

# Policy for VERSION in project()
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/radex-standalone.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/radex-options.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/radex-backends.cmake)

radex_define_library_options()
radex_language_support()

# Standalone build support
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/radex-version.cmake)
    project(RADEX-Library
        VERSION ${RADEX_VERSION}
        LANGUAGES CXX
    )
    find_package(Threads REQUIRED)
    radex_find_backends()
endif()

radex_setup_standalone_build()

if(ENABLE_CXX)
    # ============================================================================
    # RaDex Library
    # ============================================================================
    set(RADEX_BUILD_HAS_DRAGON 0)
    if(BUILD_DRAGON)
        set(RADEX_BUILD_HAS_DRAGON 1)
    endif()

    set(RADEX_BUILD_HAS_SMARTREDIS 0)
    if(BUILD_SMARTREDIS)
        set(RADEX_BUILD_HAS_SMARTREDIS 1)
    endif()

    set(RADEX_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/include)
    set(RADEX_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)

    file(GLOB_RECURSE RADEX_PUBLIC_HEADERS
        RELATIVE ${RADEX_PUBLIC_INCLUDE_DIR}
        ${RADEX_PUBLIC_INCLUDE_DIR}/*.hpp
    )
    foreach(header IN LISTS RADEX_PUBLIC_HEADERS)
        configure_file(
            ${RADEX_PUBLIC_INCLUDE_DIR}/${header}
            ${RADEX_GENERATED_INCLUDE_DIR}/${header}
            COPYONLY
        )
    endforeach()

    # Overwrite the build_config.hpp
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/radex-build-config.hpp.in
        ${RADEX_GENERATED_INCLUDE_DIR}/radex/build_config.hpp
        @ONLY
    )

    set(RADEX_SOURCES)
    list(APPEND RADEX_SOURCES cpp/client.cpp)
    list(APPEND RADEX_SOURCES cpp/handles.cpp)
    list(APPEND RADEX_SOURCES cpp/smartredis.cpp)
    list(APPEND RADEX_SOURCES cpp/dragon.cpp)

    add_library(objlib OBJECT ${RADEX_SOURCES})
    target_compile_options(objlib PRIVATE -Wpedantic -Wall)

    set(LIBRARY_TARGETS objlib)
    if(BUILD_SHARED_LIBS)
        set_target_properties(objlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
        add_library(radex_shared SHARED $<TARGET_OBJECTS:objlib>)
        list(APPEND LIBRARY_TARGETS radex_shared)
    endif()

    if(BUILD_STATIC_LIBS)
        add_library(radex_static STATIC $<TARGET_OBJECTS:objlib>)
        list(APPEND LIBRARY_TARGETS radex_static)
    endif()

    # Canonical target for consumers: shared when available, otherwise static.
    if(TARGET radex_shared)
        set(RADEX_PRIMARY_TARGET radex_shared)
    elseif(TARGET radex_static)
        set(RADEX_PRIMARY_TARGET radex_static)
    else()
        message(FATAL_ERROR "No RaDex library target was created. Enable BUILD_SHARED_LIBS and/or BUILD_STATIC_LIBS.")
    endif()

    add_library(radex INTERFACE)
    add_library(radex::radex ALIAS radex)
    target_link_libraries(radex INTERFACE ${RADEX_PRIMARY_TARGET})
    target_include_directories(radex INTERFACE
        $<BUILD_INTERFACE:${RADEX_GENERATED_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # Apply include and properties to each library being built
    foreach(target IN LISTS LIBRARY_TARGETS)
        target_include_directories(${target} PUBLIC
            $<BUILD_INTERFACE:${RADEX_GENERATED_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
        radex_set_target_install_rpath(${target} smartredis dragon)
        set_target_properties(${target} PROPERTIES OUTPUT_NAME radex)
        if(BUILD_SMARTREDIS)
            target_link_libraries(${target} PUBLIC smartredis)
        endif()
        if(BUILD_DRAGON)
            target_link_libraries(${target} PUBLIC dragon)
        endif()
    endforeach()


    # ============================================================================
    # Install library and headers
    # ============================================================================
    list(REMOVE_ITEM LIBRARY_TARGETS objlib)
    install(
        TARGETS ${LIBRARY_TARGETS}
        EXPORT radex-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
        TARGETS radex
        EXPORT radex-targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY ${RADEX_GENERATED_INCLUDE_DIR}/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(EXPORT radex-targets
        FILE radex-targets.cmake
        NAMESPACE radex::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/radex
    )

    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/radex-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/radex-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/radex
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/radex-config-version.cmake
        VERSION ${CMAKE_PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/radex-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/radex-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/radex
    )
endif()

# ============================================================================
# Install the Python client
# ============================================================================
if(ENABLE_PYTHON)
  if(NOT ALLOW_SYSTEM_PYTHON)
      if(NOT (DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX}))
          message(FATAL_ERROR
              "Python virtual environment not detected. To force installation in the system Python set the ALLOW_SYSTEM_PYTHON option"
          )
      endif()
      set(Python3_FIND_VIRTUALENV ONLY)
  endif()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  message(STATUS "Python package will be installed")
  message(STATUS "\tPython Executable: ${Python3_EXECUTABLE}")
  message(STATUS "\tPython Version: ${Python3_VERSION}")

  #
  install(CODE "
      message(STATUS \"Building and installing Python package\")
      execute_process(
          COMMAND ${CMAKE_COMMAND} -E env
              RADEX_INSTALL_DIR=${CMAKE_INSTALL_PREFIX}
              DRAGON_INCLUDE_DIR=${DRAGON_INCLUDE_DIR}
              DRAGON_LIB_DIR=${DRAGON_LIB_DIR}
              SMARTREDIS_INCLUDE_DIR=${SMARTREDIS_INCLUDE_DIR}
              SMARTREDIS_LIB_DIR=${SMARTREDIS_LIB_DIR}
              ${Python3_EXECUTABLE} -m pip install \"${CMAKE_CURRENT_SOURCE_DIR}/python\"
          COMMAND_ERROR_IS_FATAL ANY
      )
  ")
endif()
