# Install Instructions


## Prepping the Environment

Begin by loading modules such that the installers have

  - Cmake >= 3.13
  - A C++17 compiler (preferably gcc, but any should work)
  - Python 3.12
  - git-lfs (Needed for building Redis with SmartSim)

Next create a virtual env and activate it.

```sh
$ python3.12 -m venv venv && source ./venv/bin/activate
```


## Installing Dragon

There are two ways to install DragonHPC.

1. The Easy Way (from PyPI)
2. The Hard Way (from source)


### Dragon: The Easy Way (from PyPI)

Simply install `dragonhpc` from  PyPI. You will need to make sure that it is
the latest release and that it comes with the `telemetry` optional
dependencies.

```sh
$ pip install dragonhpc\[telemetry\]==0.14.2
```


### Dragon: The Hard Way (From Source)

It's highly encouraged to avoid this way if possible, but it does tend to be a
bit more configurable than installing directly from PyPI.

Navigate to the [public dragon repository](https://github.com/DragonHPC/dragon)
and clone the repository into a `dragon` directory.

```sh
$ git clone https://github.com/DragonHPC/dragon.git dragon
```

Next, open the file `dragon/devtools/VARIABLES` and comment out the
`PYTHONPATH` and the `dragon-config` exports. Similarly you _can_ comment out
the MacOS specific `DYLIB_*` exports, but they should not have an ill effect on
install process if left alone. Then source the file.

```sh
$ source ./dragon/devtools/VARIABLES
```

Next we need to build the compiled portion of dragon. We can do this by simply
calling the `build` target using the `dragon/src/Makefile`. The `dragon/src`
directory has been handily exported as `DRAGON_BASE_DIR` from the `VARIABLES`
file.

```sh
$ make -C "${DRAGON_BASE_DIR}" build
```

Finally, the previous command will install the `dragonhpc` library in the
virtual environment by building from source.  It's also worth installing this
with the `telemetry` optional dependencies.

```sh
$ pip install "${DRAGON_BASE_DIR}"\[telemetry\]
```


## Installing SmartRedis

Navigate to the [SmartRedis repository](https://github.com/CrayLabs/SmartRedis)
and clone into a `smartredis` directory.

```sh
$ git clone https://github.com/CrayLabs/SmartRedis.git smartredis
```

Next we're going to pop into that directory and build the C and C++ backends
from source.

```sh
$ pushd smartredis
$ mkdir build install
$ export SMARTREDIS_INSTALL_DIR="$(pwd)/install"
$ cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX="${SMARTREDIS_INSTALL_DIR}"
$ make install -j
$ popd
```

Finally, some of the demos we  will be presenting require a SmartRedis install.
Since we already cloned the repository, we can install it from source.

```sh
$ pip install ./smartredis
```

Lastly, we need a way to actually set up a Redis database in order to use the
SmartRedis clients. This is typically done via its companion library, SmartSim.
This library has a two step install process. First is to pip install it from
its repository.

```sh
$ pip install git+https://github.com/CrayLabs/SmartSim.git@1e0ce3a
```

And the second step is to build its Redis backends. This can be done using the
`smart` command line tool that comes with the package. Since we do not need any
of its AI dependencies we can turn most of the extra options off.

```sh
$ smart build --device cpu \
              --skip-torch \
              --skip-tensorflow \
              --skip-onnx \
              --skip-python-packages
```


## Installing Radex

Navigate to the [Radex repository](https://github.com/radical-cybertools/radex)
and clone it into a `radex` directory.

```sh
$ git clone https://github.com/radical-cybertools/radex.git radex
```

We are once again going to build both the C++ and python clients. Both of which
are installable through CMake using the following commands.

```sh
$ pushd radex
$ export RADEX_INSTALL_DIR="$(pwd)/install"
$ cmake -S . \
        -B build \
        -DCMAKE_INSTALL_PREFIX="${RADEX_INSTALL_DIR}" \
        -DBUID_EXAMPLES=ON \
        -DBUID_TESTS=ON \
        -DBUID_DRAGON=ON \
        -DBUILD_SMARTREDIS=ON \
        -Dsmartredis_DIR="${SMARTREDIS_INSTALL_DIR}/share/cmake/smartredis"
$ cmake --build build
$ cmake --install build
$ popd
```

Once that completes, the last thing we need to do is to check that the install
is working as expected. We can do this by simply installing the development
resources and running the test suite.

```sh
$ pushd radex
$ pip install -r dev-resources/requirements-dev.txt
$ make -f dev-resources/Makefile test PYTEST_ARGS="-vvv"
$ popd
```

At this point in time, if the test suite passed, you should have everything
installed everything up to this point correctly!


## Installing RHAPSDOY

Finally, some of our simple examples are intended to show how radex can be used
in the larger RHAPSODY ecosystem. To do this we need to install a custom
version of RHAPSODY that is capable of setting up a key-value store for radex
to connect to. This can be done with simple pip command

```sh
pip install git+https://github.com/radical-cybertools/rhapsody.git@feature/data-backends
```


## Running the basic examples

The basic examples are intended to show the raw functionality of radex and what
the API might look like for users attempting to send complex tensor data across
ranks of a distributed program. Additionally, it is intended to show that radex
is agnostic to the type of key value store used. Finally, there are a few
examples that how how radex could integrate into a larger, more complex
rhapsody workflow. These examples are in the directories listed below:

  - `radex/example/cpp-exchange/in-mem`
    (Using radex with a local key values store)
  - `radex/example/cpp-exchange/dragon` 
    (Using radex to share data among C++ apps using dragon)
  - `radex/example/cpp-exchange/redis`
    (Using radex to share data among C++ apps using SmartRedis)
  - `radex/example/py-cpp-exchange/dragon`
    (Using radex to share data across a C++ app and a python app using dragon)
  - `radex/example/rhapsody-exchange/dragon`
    (Shows a RHAPSODY integration with radex using a dragon key-value store)
  - `radex/example/rhapsody-exchange/redis`
    (Shows a RHAPSODY integration with radex using a redis key-value store
    NOTE: This example require `redis-server` be on your `PATH`. We installed
    `redis-server` through SmartSim and it can be added to you `PATH` with
    `export PATH="${PATH}:$(dirname $(smart dbcli))"`)

Users are encouraged to look through the source code to get a feel for the
ergonomics of the library, as well a try running the examples for themselves.
This should be a simple as activating the python environment that we installed
all of the dependencies into and simple running the `driver.py` file for each
example.

```sh
$ pushd "radex/example/${EXAMPLE_YOU_WOULD_LIKE_TO_RUN}"
$ python driver.py       # if the example does not require the dragon runtime
$                        # -- OR --
$ dragon -s ./driver.py  # if the example does require the dragon runtime
$ popd
```
