# Python/C++ Data Exchange

**Location:** [`example/py-cpp-exchange/dragon`](https://github.com/radical-cybertools/radex/tree/main/example/py-cpp-exchange/dragon)

This example shows a Python driver launching a compiled C++ application and exchanging data with it through a shared Dragon `DDict`, using the radex Python client on one side and the C++ client on the other.

## How It Works

[`driver.py`](https://github.com/radical-cybertools/radex/blob/main/example/py-cpp-exchange/dragon/driver.py):

1. Creates a Dragon `DDict` and serializes it.
2. Launches the compiled example binary (`dragon-cpp-with-py`, from `install/bin/examples`) as a Dragon `Process`, passing the serialized `DDict` through the `SERIALIZED_DDICT` environment variable.
3. Creates a radex `DragonClient` attached to the same `DDict`.
4. Writes values from Python that the C++ application reads, and vice versa.

```python
from radex.clients.core import DragonClient as Client
from radex.handles.handles import IncomingHandle, OutgoingHandle

client = Client(serial_dd, 5)
client.put_scalar(OutgoingHandle("py-int"), 123)
```

The C++ side (built from `app.cpp`) attaches its own client to the same serialized `DDict` descriptor and reads/writes using the equivalent C++ API (see [C++ Data Exchange](cpp-exchange.md)).

## Running

```bash
cd example/py-cpp-exchange/dragon
dragon driver.py
```
