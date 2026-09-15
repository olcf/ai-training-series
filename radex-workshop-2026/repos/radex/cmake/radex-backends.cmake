include_guard(GLOBAL)

function(radex_find_backends)
    if(BUILD_SMARTREDIS)
        find_package(smartredis REQUIRED)
        message(STATUS "SmartRedis found: ${smartredis_DIR}")

        get_target_property(SMARTREDIS_LIBRARY smartredis LOCATION)
        get_filename_component(SMARTREDIS_LIB_DIR ${SMARTREDIS_LIBRARY} DIRECTORY)
        get_filename_component(SMARTREDIS_ROOT ${SMARTREDIS_LIB_DIR} DIRECTORY)
        set(SMARTREDIS_LIB_DIR ${SMARTREDIS_LIB_DIR} CACHE PATH "SmartRedis library directory")
        set(SMARTREDIS_INCLUDE_DIR ${SMARTREDIS_ROOT}/include CACHE PATH "SmartRedis include directory")
    endif()

    if(BUILD_DRAGON)

        # If Dragon variables are not present in the environment try to parse the
        # information from dragon-config
        if(NOT (DEFINED ENV{dragon_DIR} OR DEFINED ENV{DRAGON_BASE_DIR}))
            find_program(DRAGON_CONFIG dragon-config)
            if(DRAGON_CONFIG)
                execute_process(
                    COMMAND dragon-config -l
                    OUTPUT_VARIABLE DRAGON_LINKER_FLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )
                string(REGEX MATCH "-L[ \t]*([^ \t]+)" TEMP "${DRAGON_LINKER_FLAGS}")
                get_filename_component(dragon_DIR ${CMAKE_MATCH_1} DIRECTORY)
                message(STATUS "Dragon found via dragon-config: ${dragon_DIR}")
            else()
                message(FATAL_ERROR "Dragon requested by dragon-config could not be found and dragon_DIR was not defined")
            endif()
        endif()
        find_path(DRAGON_INCLUDE_DIR
            NAMES dragon/ddict.h
            HINTS ENV DRAGON_BASE_DIR
            HINTS ${dragon_DIR}
            PATH_SUFFIXES include
        )
        find_library(DRAGON_LIBRARY
            NAMES dragon
            HINTS ENV DRAGON_BASE_DIR
            HINTS ${dragon_DIR}
            HINTS
            PATH_SUFFIXES lib
        )
        get_filename_component(DRAGON_LIB_DIR ${DRAGON_LIBRARY} DIRECTORY)
        set(DRAGON_LIB_DIR ${DRAGON_LIB_DIR} CACHE PATH "Dragon library directory")

        if(NOT TARGET dragon)
            add_library(dragon UNKNOWN IMPORTED)
            set_target_properties(dragon PROPERTIES
                IMPORTED_LOCATION ${DRAGON_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES "${DRAGON_INCLUDE_DIR}"
            )
        endif()

        message(STATUS "Dragon found:")
        message(STATUS "\tLibrary: ${DRAGON_LIB_DIR}")
        message(STATUS "\tHeaders: ${DRAGON_INCLUDE_DIR}")
    endif()
endfunction()