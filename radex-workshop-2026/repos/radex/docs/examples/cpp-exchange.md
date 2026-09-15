# C++ Data Exchange

**Location:** [`example/cpp-exchange`](https://github.com/radical-cybertools/radex/tree/main/example/cpp-exchange)

This example is a pure C++ producer/consumer pair that exchange scalars and tensors through a radex backend. It is the simplest way to see the C++ client API in isolation.

## Variants

- **`dragon/`** — producer and consumer processes exchange data through a Dragon `DDict`. The `DDict` descriptor is passed via the `SERIALIZED_DDICT` environment variable.
- **`in-mem/`** — an in-process backend, useful for exercising the client API without any external service.
- **`redis/`** — exchange through a SmartRedis-backed Redis instance.

## Dragon Variant Walkthrough

[`producer.cpp`](https://github.com/radical-cybertools/radex/blob/main/example/cpp-exchange/dragon/producer.cpp) attaches a client to the shared `DDict` and writes a mix of scalars and tensors:

```cpp
--8<-- "example/cpp-exchange/dragon/producer.cpp:docs-example"
```

[`consumer.cpp`](https://github.com/radical-cybertools/radex/blob/main/example/cpp-exchange/dragon/consumer.cpp) attaches to the same `DDict` and reads back the values written by the producer using the matching `get_scalar`/`get_tensor` calls.

## Running

Binaries are installed under `install/bin/examples` when built with `BUILD_EXAMPLES=ON`. See [`driver.py`](https://github.com/radical-cybertools/radex/blob/main/example/cpp-exchange/dragon/driver.py) in the `dragon/` variant for how a Python script can start the producer/consumer pair against a shared `DDict`.
