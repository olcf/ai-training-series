# Quick Start

This guide walks through exchanging a value between two processes using radex's Dragon DDict backend.

## 1. Start a DDict and a client (Python)

```python
from dragon.data.ddict import DDict
from radex.clients.core import DragonClient as Client
from radex.handles.handles import IncomingHandle, OutgoingHandle

dd = DDict(managers_per_node=1, n_nodes=1)
serialized_dd = dd.serialize()

client = Client(serialized_dd, 5)  # 5 second timeout
client.put_scalar(OutgoingHandle("greeting-count"), 1)
```

## 2. Read the same value from C++

Pass the serialized descriptor to a separate process (for example, via an environment variable), then attach a C++ client to the same DDict:

```cpp
#include "radex/dragon.hpp"
#include "radex/handles.hpp"

char *serialized_dd = getenv("SERIALIZED_DDICT");
timespec timeout{5, 0};
radex::drg::ddict::Client client{serialized_dd, &timeout};

int32_t value = client.get_scalar<int32_t>(
    radex::data::IncomingHandle{"greeting-count"});
```

## 3. Exchange tensors

Both clients support n-dimensional tensors in addition to scalars:

```python
import numpy as np
client.put_tensor(OutgoingHandle("weights"), np.zeros((4, 4), dtype=np.float64))
```

```cpp
client.put_tensor<double>(radex::data::OutgoingHandle{"weights"},
                          {4, 4}, std::vector<double>(16, 0.0));
```

## Next Steps

- Walk through the full [Examples](../examples/index.md) for realistic driver/worker setups (MPI + Dragon DDict, C++-to-C++, and Python-to-C++ exchange).
- See the [API Reference](../api/index.md) for the complete set of client methods and handle types.
