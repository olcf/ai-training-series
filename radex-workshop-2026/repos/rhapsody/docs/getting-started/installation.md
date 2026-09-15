# Installation

RHAPSODY can be installed from source or via pip. We recommend using a virtual environment.

## Prerequisites

- **Python**: 3.8 or higher.
- **asyncio**: Included in the Python standard library.

## Standard Installation

To install the core RHAPSODY package:

```bash
pip install rhapsody-py
```

## Complete Installation (Recommended)

To install RHAPSODY with all optional backends and dependencies (Dragon, Dask, RADICAL-Pilot, etc.):

```bash
pip install "rhapsody-py[all]"
```

!!! warning "Backend Dependencies"
    Some backends require specific system-level libraries or environments. For example, the Dragon backend requires the Dragon runtime to be installed and configured on your HPC system.

## Optional Extras

You can also install specific backend supports individually:

- **Dask**: `pip install "rhapsody-py[dask]"`
- **RADICAL-Pilot**: `pip install "rhapsody-py[radical_pilot]"`
- **Dragon**: `pip install "rhapsody-py[dragon]"`
- **Orbit**: `pip install "rhapsody-py[orbit]"` (requires Python >= 3.10)
- **Dragon AI (vLLM) Inference**: `pip install "rhapsody-py[ai]"`

## Verification

After installation, you can verify that RHAPSODY is working correctly by running:

```bash
python3 -c "import rhapsody; print(rhapsody.__version__)"
```
