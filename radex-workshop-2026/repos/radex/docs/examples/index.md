# Examples

The [`example/`](https://github.com/radical-cybertools/radex/tree/main/example) directory contains runnable programs that show radex being used across language and process boundaries. All examples are built as part of the CMake build when `BUILD_EXAMPLES=ON` (the default); Python drivers additionally require the [Python client](../getting-started/building-from-source.md#building-the-python-client) to be installed.

| Example | Languages | Backend | Description |
| --- | --- | --- | --- |
| [Active Learning](active-learn.md) | C++ (MPI) + Python | Dragon DDict | An MPI simulation exchanges data with a Python-driven active-learning loop. |
| [C++ Data Exchange](cpp-exchange.md) | C++ | Dragon DDict, in-memory, Redis | A producer/consumer pair exchange scalars and tensors entirely in C++. |
| [Python/C++ Data Exchange](py-cpp-exchange.md) | C++ + Python | Dragon DDict | A Python driver launches and exchanges data with a compiled C++ application. |

Binaries built from these examples are installed under `install/bin/examples`.
