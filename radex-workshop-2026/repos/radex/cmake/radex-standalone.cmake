set(RADEX_STANDALONE_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(radex_set_target_install_rpath target_name)
    if(NOT DEFINED RADEX_INSTALL_RPATH)
        include(GNUInstallDirs)
        if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
            set(RADEX_INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}")
        else()
            set(RADEX_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
        endif()
    endif()

    set(_target_install_rpath "${RADEX_INSTALL_RPATH}")
    foreach(dep_target IN LISTS ARGN)
        if(TARGET ${dep_target})
            list(APPEND _target_install_rpath "$<TARGET_FILE_DIR:${dep_target}>")
        endif()
    endforeach()

    set_target_properties(${target_name} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_target_install_rpath}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endfunction()

function(radex_setup_standalone_build)
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        return()
    endif()

    set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
    set(CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)
    set(CMAKE_CXX_VISIBILITY_PRESET default PARENT_SCOPE)

    if(NOT DEFINED CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release")
    endif()

    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install PARENT_SCOPE)
    endif()

    include(GNUInstallDirs)

    # Ensure the paths are absolute for the radex library
    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(RADEX_INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}")
    else()
        set(RADEX_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    endif()

    set(RADEX_INSTALL_RPATH "${RADEX_INSTALL_RPATH}" PARENT_SCOPE)

endfunction()