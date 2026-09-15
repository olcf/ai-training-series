# Installation

radex is currently distributed as source only: a C++ core library and an optional Cython-based Python client that binds to it. There is no published binary package yet, so installation means building from source (see [Building From Source](building-from-source.md) for full details).

## Prerequisites

- **CMake** >= 3.13
- **C++17** compiler
- A backend library, matching what you enable at configure time:
    - [Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html) (`BUILD_DRAGON`, on by default)
    - [SmartRedis](https://github.com/CrayLabs/SmartRedis) (`BUILD_SMARTREDIS`, on by default)
- **Python** 3.12, `Cython`, and `numpy` if building the Python client (`ENABLE_PYTHON`, on by default)

## Summary

```bash
git clone https://github.com/radical-cybertools/radex.git
cd radex
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install
cmake --build build --target install
```

## Verification

After installation, verify the C++ library is present:

```bash
ls install/lib
```

And, if you built the Python client:

```bash
python3 -c "import radex; from radex.clients.core import DragonClient"
```

See [Building From Source](building-from-source.md) for backend-specific configuration (environment variables, `dragon-config`, disabling backends, etc.).
