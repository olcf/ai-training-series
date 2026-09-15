if [ -z "${RADEX_DEMO_INSTALL_DIR:-}" ]; then
    printf '`RADEX_DEMO_INSTALL_DIR` is unset or empty!\n' >&2
    printf '    Hint: Set this environment variable to modify your ' >&2
    printf 'environment to run the examples\n' >&2
else
    source "${RADEX_DEMO_INSTALL_DIR}/venv/bin/activate"

    # SmartRedis install directories
    export SMARTREDIS_INSTALL_DIR="${RADEX_DEMO_INSTALL_DIR}/smartredis/install"
    export SMARTREDIS_DIR="${SMARTREDIS_INSTALL_DIR}"

    # Add `redis-server` to path
    export PATH="${PATH}:$(dirname $(smart dbcli))"

    # Radex install directories
    export RADEX_INSTALL_DIR="${RADEX_DEMO_INSTALL_DIR}/radex/install"
    export RADEX_DIR="${RADEX_INSTALL_DIR}"

    # Source the OpenFOAM bashrc
    # NOTE: If the `rhapsody-plugins-openfoam` examples were built by a
    #       different user than the one running this script, then the shared
    #       libraries created by running its `Allwmake` will NOT be available
    #       on a user's LD_LIBRARY_PATH. The easiest way to address this is to
    #       either:
    #
    #       1) Have the new user (the one running the examples) re-build their
    #          own copy of the libraries by running the
    #          `rhapsody-plugins/openfoam/Allwmake` script
    #
    #       OR
    #
    #       2) Add the `FOAM_USER_LIBBIN` path of the original user (the one
    #          who built the libraries) path to the `LD_LIBRARY_PATH` of the new
    #          user. This does require that the `FOAM_USER_LIBBIN` of the original
    #          user is readable to the new user which may not be the case if
    #          the libraries were originally built under the original user's
    #          home directory
    source "${RADEX_DEMO_INSTALL_DIR}/OpenFoam-2506/etc/bashrc"

    printf "RHAPSODY/Radex demo environment successfully configured!\n"
fi
