# Fetching the Dragon and SmartRedis Backends

radex's `BUILD_DRAGON` and `BUILD_SMARTREDIS` CMake options (see [Building From Source](building-from-source.md)) link against pre-built Dragon and SmartRedis installations — CMake does not fetch or build either backend for you.

## System Dependencies

**Ubuntu:**

```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ make cmake mpich libnuma-dev
```

**macOS:**

```bash
brew update
brew install cmake mpich autoconf automake libtool
```

## Dragon

Clone and install [DragonHPC](https://github.com/DragonHPC/dragon) into your active Python environment:

```bash
git clone https://github.com/DragonHPC/dragon.git
cd dragon
git checkout main
cd devtools && source VARIABLES && cd ..
pip install -e src/
```

`source VARIABLES` sets the environment variables (including `DRAGON_BASE_DIR`) that radex's CMake config uses to locate Dragon's headers and library — see [Building From Source](building-from-source.md#dragon). Keep this environment sourced (or re-source it) in any shell you use to configure/build radex.

## SmartRedis

Clone and build [SmartRedis](https://github.com/CrayLabs/SmartRedis):

```bash
git clone -b develop --single-branch https://github.com/CrayLabs/SmartRedis.git
cd SmartRedis
make lib
```

Then point CMake at the resulting package config when configuring radex:

```bash
cmake -S . -B build -Dsmartredis_DIR=SmartRedis/install/share/cmake/smartredis
```

(equivalent to setting `smartredis_DIR` in the environment).

See [Building From Source](building-from-source.md) for the full set of CMake options, including how to disable a backend you don't need.
