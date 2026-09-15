# Building the OpenFOAM Function Objects

The [`radexWrite`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/src/openFOAM/functionObjects/radexWrite) function object exports OpenFOAM fields to a [radex](https://github.com/radical-cybertools/radex) key-value store at write intervals, so external Python code can read simulation data directly out of memory instead of parsing case output on disk.

The [`radexRead`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/main/src/openFOAM/functionObjects/radexRead) function object is its counterpart: it imports fields from a radex key-value store into the running solver at every time step, so external Python code can drive or perturb a simulation in place.

## Prerequisites

- A sourced OpenFOAM installation (`wmake` on `PATH`, `$FOAM_USER_LIBBIN` writable)
- [radex](https://github.com/radical-cybertools/radex) built and installed (`$RADEX_DIR`)
- [Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html) libraries/includes available via `dragon-config`
- Optionally, [SmartRedis](https://github.com/CrayLabs/SmartRedis) if you plan to use the Redis backend

## Building

The [`Allwmake`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/blob/main/Allwmake) script wires up the required environment variables, compiles the function objects with `wmake`, and installs the Python package:

```bash
export RADEX_DIR=/path/to/radex/install
export DRAGON_LIBS=$(dragon-config -l)
export DRAGON_INCLUDES=$(dragon-config -o)
# Optional, only needed for the Redis backend
export SMARTREDIS_DIR=/path/to/SmartRedis/install

./Allwmake
```

This builds `libradexIO`, `libradexBase`, `libradexWrite` and `libradexRead` and installs them into `$FOAM_USER_LIBBIN`, creating the directory first if it doesn't already exist, then installs the `rhapsody_plugins.openfoam` Python package.

`./Allwclean` reverses the build: it removes the compiled artefacts, deletes the installed libraries from `$FOAM_USER_LIBBIN` so a stale one cannot be picked up by a case, and clears the Python build directories.

## `radexWrite` Function Object

Once built, enable the function object in a case's `controlDict`:

```cpp
radexFields
{
    type            radexWrite;
    libs            (radexWrite);

    fields          (U p nut nuTilda);
    scalars         (someAverage);
}
```

### Key Naming

Each field is written through a radex [`OutgoingHandle`](https://github.com/radical-cybertools/radex/blob/main/include/radex/handles.hpp) named:

```
<fieldName>_<timeStep>_<subdomainId>
```

or, if an `identifier` is configured:

```
<identifier>_<fieldName>_<timeStep>_<subdomainId>
```

The handle's name is the key under which the field's raw bytes are stored; radex derives the associated metadata key (dtype/shape) from it automatically.

### Supported Field Types

| Source | OpenFOAM type | Exported as |
| --- | --- | --- |
| `fields` | `volScalarField` | tensor of shape `[nCells]` |
| `fields` | `volVectorField` | tensor of shape `[nCells, 3]` |
| `scalars` | `uniformDimensionedScalarField` | single scalar |
| `scalars` | result of another function object (e.g. `surfaceFieldValue`) | single scalar |

On finalization, the master rank writes the final timestep once under `<identifier>_final_step` (or `final_step` if no identifier is set).

### Backend Selection

The backend is selected at runtime via the `backend` dictionary entry:

```cpp
backend   dragon;  // default
backend   redis;
```

See [`radexWrite.H`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/blob/main/src/openFOAM/functionObjects/radexWrite/radexWrite.H) for the full set of supported options.

## `radexRead` Function Object

`radexRead` mirrors `radexWrite`, but imports fields from the store instead of exporting them:

```cpp
radexFields
{
    type            radexRead;
    libs            (radexRead);

    fields          (U p nut nuTilda);
    scalars         (someAverage);

    // Optional; defaults shown
    wait            true;
    timeout         30;
}
```

Each configured field/scalar is read through a radex [`IncomingHandle`](https://github.com/radical-cybertools/radex/blob/main/include/radex/handles.hpp) named the same way as `radexWrite`'s keys (`<fieldName>_<timeStep>_<subdomainId>`, optionally prefixed by `identifier`), and is fetched at every time step rather than only at write intervals.

### Wait vs. Get Semantics

Each `radexRead` instance chooses independently, via the `wait` entry, whether reads block for the value to appear or fail immediately if it's absent:

| `wait` | Semantics | Behavior |
| --- | --- | --- |
| `true` (default) | `wait_for_scalar` / `wait_for_tensor` | Blocks up to `timeout` seconds for the value to appear in the store |
| `false` | `get_scalar` / `get_tensor` | Fetches immediately; logs a warning and skips the field if the value isn't yet present |

`timeout` (seconds) is only used when `wait` is `true`.

See [`radexRead.H`](https://github.com/CrayLabs/rhapsody-plugins-openfoam/blob/main/src/openFOAM/functionObjects/radexRead/radexRead.H) for the full set of supported options.
