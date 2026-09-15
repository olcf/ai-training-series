# Building From Source

radex uses CMake for the C++ core and examples, and a standard `setuptools` + `Cython` build for the Python client.

## CMake Options

Defined in [`cmake/radex-options.cmake`](https://github.com/radical-cybertools/radex/blob/main/cmake/radex-options.cmake):

| Option | Default | Description |
| --- | --- | --- |
| `BUILD_EXAMPLES` | `ON` | Build the example applications under `example/`. |
| `BUILD_TESTS` | `ON` | Build the C++ test suite under `tests/cpp`. |
| `BUILD_SHARED_LIBS` | `ON` | Build the shared `radex` library. |
| `BUILD_STATIC_LIBS` | `ON` | Build the static `radex` library. |
| `BUILD_SMARTREDIS` | `ON` | Build with [SmartRedis](https://github.com/CrayLabs/SmartRedis) backend support. |
| `BUILD_DRAGON` | `ON` | Build with [Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html) backend support. |
| `ENABLE_CXX` | `ON` | Build the C++ libraries. |
| `ENABLE_PYTHON` | `ON` | Build and install the Python client. |
| `ALLOW_SYSTEM_PYTHON` | `OFF` | Allow the Python client to be installed system-wide. |

## Configuring Backends

Both backends below must already be built/installed before configuring radex — see [Fetching the Dragon and SmartRedis Backends](fetching-backends.md) for step-by-step instructions. Take particular note of the installation path of the the SmartRedis library.

### Dragon

If `dragon_DIR`/`DRAGON_BASE_DIR` are not set in the environment, the build looks for `dragon-config` on `PATH` and uses `dragon-config -l` to locate the library. Either ensure `dragon-config` is available, or set `dragon_DIR`/`DRAGON_BASE_DIR` explicitly:

```bash
export DRAGON_BASE_DIR=/path/to/dragon
```

### SmartRedis

SmartRedis is located via `find_package(smartredis REQUIRED)`, so `smartredis_DIR` (or a location CMake can discover) must point at an installed SmartRedis CMake package.

To disable a backend you don't need:

```bash
cmake -S . -B build -DBUILD_DRAGON=OFF -DBUILD_SMARTREDIS=ON
```

## Building the C++ Library, Examples, and Tests

```bash
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_TESTS=ON
cmake --build build -j
cmake --install build
```

This installs headers, libraries, and example binaries under the configured `CMAKE_INSTALL_PREFIX` (`install/` by default), including `install/include/radex/build_config.hpp`, which records which backends were enabled.

## Building the Python Client

The Python client (`src/python`) is a Cython extension that links against the *installed* C++ library, so build the C++ side first.

```bash
export RADEX_INSTALL_DIR="$(pwd)/install"   # defaults to <repo>/install
python3 -m pip install ./src/python
```

`src/python/setup.py` reads `install/include/radex/build_config.hpp` to determine which backends were enabled, and requires the corresponding `DRAGON_INCLUDE_DIR`/`DRAGON_LIB_DIR` and/or `SMARTREDIS_INCLUDE_DIR`/`SMARTREDIS_LIB_DIR` environment variables to be set when that backend is active, so it can compile and link the `radex.clients.core` and `radex.handles.handles` extension modules.

## Running the Tests

All tests use `pytest` as the testing harness which itself is configured in [`dev-resources/radex-config.toml`](https://github.com/radical-cybertools/radex/blob/main/dev-resources/radex-config.toml):

```bash
pip install -r dev-resources/requirements-dev.txt
make -f dev-resources/Makefile test
```
