# radical-openfoam

The RHAPSODY OpenFOAM plugin lets you define and run OpenFOAM cases as RHAPSODY workflows.

### Prerequisites

- Python >= 3.9
- A sourced [OpenFOAM](https://www.openfoam.com) installation
- [Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html) (`dragonhpc`), the default execution backend used in the examples
- [RHAPSODY](https://radical-cybertools.github.io/rhapsody) (`rhapsody-py`), used to execute workflows
- [radex](https://radical-cybertools.github.io/radex/), used to exchange in-memory data between OpenFOAM and other workflow components

### Install the Python Plugin

```bash
git clone https://github.com/CrayLabs/rhapsody-plugins-openfoam.git
cd rhapsody-plugins-openfoam
pip install -e .
```

This installs `rhapsody_plugins.openfoam` and its declared dependencies, including `dragonhpc` and `rhapsody-py`. Install radex separately by following its [installation directions](https://radical-cybertools.github.io/radex/getting-started/installation/).

Verify the installation:

```bash
python -c "import rhapsody_plugins.openfoam as rof; print(rof.CaseDefinition)"
python -c "from radex.clients.core import DragonClient as Client"
```

### Build the OpenFOAM Function Objects

Build the `radexWrite` and `radexRead` function objects to stream fields from or into a running solver. This requires `wmake` on `PATH`, a writable `$FOAM_USER_LIBBIN`, radex installed at `$RADEX_DIR`, and Dragon libraries and includes available through `dragon-config`.

```bash
export RADEX_DIR=/path/to/radex/install
export DRAGON_LIBS=$(dragon-config -l)
export DRAGON_INCLUDES=$(dragon-config -o)
# Optional, only needed for the Redis backend
export SMARTREDIS_DIR=/path/to/SmartRedis/install

./Allwmake
```

`./Allwmake` builds and installs the OpenFOAM libraries and installs the Python package. Run `./Allwclean` to remove compiled artefacts, installed libraries, and Python build directories.

### Run and understand a simple case

The basic `airFoil2D` OpenFOAM tutorial demonstrates a complete RHAPSODY workflow: decompose the case, run `simpleFoam` across eight MPI ranks, and reconstruct the results. See the [Basic OpenFOAM Case documentation](docs/examples/basic-openfoam.md) for the walkthrough and run instructions.
