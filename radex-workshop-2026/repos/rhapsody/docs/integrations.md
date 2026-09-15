# Integrations

RHAPSODY provides seamless integrations with several high-performance computing and AI infrastructure tools. This guide covers the available integrations and how to use them effectively.

## Dragon-VLLM Inference Backend

The Dragon-VLLM integration provides a high-performance inference backend that combines Dragon's distributed computing capabilities with vLLM's efficient LLM serving.

### Overview

The `DragonVllmInferenceBackend` offers:

- **Request Batching**: Automatically accumulates individual requests into efficient batches
- **Server Mode**: Optional HTTP server with OpenAI-compatible API endpoints
- **Engine Mode**: Direct Python API for programmatic access
- **Async Operations**: Non-blocking inference with asyncio integration
- **Multi-Node Support**: Scale inference across multiple GPU nodes

### Installation

Install RHAPSODY with Dragon AI (vLLM) support — this pulls in `dragonhpc[ai]` (which bundles vLLM and the Dragon vLLM compatibility plugin) automatically:

```bash
pip install "rhapsody-py[ai]"
```

!!! warning "Prerequisites"
    The Dragon-VLLM backend requires:

    - Python >= 3.10
    - GPU nodes with CUDA support

### Basic Usage

Here's a complete example of using the Dragon-VLLM backend for AI inference:

```python
import asyncio
import multiprocessing as mp

from rhapsody import Session
from rhapsody.api import AITask, ComputeTask
from rhapsody.backends import DragonExecutionBackend, DragonVllmInferenceBackend
from rhapsody.backends.ai.config import HardwareConfig, ModelConfig

async def main():
    # Set Dragon as the multiprocessing start method
    mp.set_start_method("dragon")

    # Initialize execution backend for compute tasks
    execution_backend = await DragonExecutionBackend()

    # Initialize inference backend for AI tasks
    inference_backend = await DragonVllmInferenceBackend(
        model=ModelConfig(
            model_name="Qwen2.5-0.5B-Instruct",  # or a local snapshot path
            hf_token="...",
            tp_size=1,
        ),
        hardware=HardwareConfig(num_nodes=1, num_gpus=1, node_offset=0),
        port=8001,
    )

    # Create a session with both backends
    session = Session([execution_backend, inference_backend])

    # Define mixed workload: AI and compute tasks
    tasks = [
        AITask(
            prompt="What is the capital of France?",
            backend=inference_backend.name
        ),
        AITask(
            prompt=["Tell me a joke", "What is 2+2?"],
            backend=inference_backend.name
        ),
        ComputeTask(
            executable="/usr/bin/echo",
            arguments=["Hello from Dragon!"],
            backend=execution_backend.name,
        ),
    ]

    # Submit all tasks at once
    await session.submit_tasks(tasks)

    # Wait for results
    results = await asyncio.gather(*tasks)

    # Process results
    for i, task in enumerate(results):
        if "prompt" in task:
            # AITask: use .response for model output
            print(f"Task {i + 1} [AI]: {task.response}")
        else:
            # ComputeTask (executable): use .stdout for output
            print(f"Task {i + 1} [Compute]: {task.stdout}")

    await session.close()

if __name__ == "__main__":
    asyncio.run(main())
```

!!! important "Running with Dragon"
    Scripts using Dragon backends must be launched with the `dragon` command:
    ```bash
    dragon -m my_script.py
    ```

### Configuration

`DragonVllmInferenceBackend` takes typed configuration objects — re-exported as-is from `dragon.ai.inference` via `rhapsody.backends.ai.config` — instead of a YAML file. `model` is required; `hardware`, `batching`, `guardrails`, and `dynamic_worker` are optional and default to sensible values (single node/GPU, request batching enabled, guardrails and dynamic worker scaling off) when omitted.

```python
from rhapsody.backends.ai.config import (
    BatchingConfig,
    DynamicWorkerConfig,
    GuardrailsConfig,
    HardwareConfig,
    ModelConfig,
)

inference_backend = await DragonVllmInferenceBackend(
    model=ModelConfig(
        model_name="Qwen2.5-0.5B-Instruct",  # or a local snapshot path
        hf_token="...",
        tp_size=1,
        max_tokens=256,
    ),
    hardware=HardwareConfig(num_nodes=1, num_gpus=1, node_offset=0),
    batching=BatchingConfig(enabled=True, batch_type="pre-batch"),  # only "pre-batch" is supported today
    guardrails=GuardrailsConfig(enabled=False),
    dynamic_worker=DynamicWorkerConfig(enabled=False),
    port=8001,
)
```

### Configuration Options

`DragonVllmInferenceBackend`'s own constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ModelConfig` | Required | Model name/path, HF token, tensor-parallel size, sampling defaults |
| `hardware` | `HardwareConfig` | `HardwareConfig()` | Node/GPU allocation, node offset for multi-service deployments |
| `batching` | `BatchingConfig` | pre-batch enabled | Request batching mode (`batch_type="dynamic"` is not yet supported by this backend) |
| `guardrails` | `GuardrailsConfig` | disabled | Optional PromptGuard-based prompt filtering |
| `dynamic_worker` | `DynamicWorkerConfig` | disabled | Optional automatic GPU worker spin-up/spin-down |
| `port` | int | 8000 | HTTP server port (if `use_service=True`) |
| `use_service` | bool | True | Enable HTTP server with OpenAI-compatible API |
| `max_batch_size` | int | 1024 | Maximum requests RHAPSODY accumulates client-side per pipeline call |
| `max_batch_wait_ms` | int | 500 | Maximum time to wait for client-side batch accumulation |

`ModelConfig`/`HardwareConfig`/`BatchingConfig`/`GuardrailsConfig`/`DynamicWorkerConfig` are `dragon.ai.inference` dataclasses — see the [Dragon AI inference API reference](https://dragonhpc.github.io/dragon/doc/_build/html/ref/ai/inference/index.html) for the full field list on each.

### Server Mode vs Engine Mode

#### Server Mode (HTTP API)

When `use_service=True`, the backend starts an HTTP server with OpenAI-compatible endpoints:

```python
inference_backend = await DragonVllmInferenceBackend(
    model=ModelConfig(model_name="llama-3-8b", hf_token="...", tp_size=1),
    use_service=True,
    port=8000
)

# Access via HTTP
# GET  http://<hostname>:8000/health
# POST http://<hostname>:8000/generate
# POST http://<hostname>:8000/v1/chat/completions  (OpenAI-compatible)
# GET  http://<hostname>:8000/v1/models
```

#### Engine Mode (Direct API)

When `use_service=False`, use the Python API directly:

```python
inference_backend = await DragonVllmInferenceBackend(
    model=ModelConfig(model_name="llama-3-8b", hf_token="...", tp_size=1),
    use_service=False
)

# Direct inference
results = await inference_backend.generate(
    prompts=["What is AI?", "Explain machine learning"],
    timeout=300
)
```

### Batching Strategy

The backend automatically batches requests for optimal throughput:

1. Accumulates requests for up to `max_batch_wait_ms` milliseconds
2. Processes immediately when batch reaches `max_batch_size`
3. Submits combined batch to vLLM pipeline
4. Distributes responses back to individual requests

This is significantly more efficient than processing requests individually, especially for high-throughput workloads.

!!! tip "Performance Tuning"
    - Increase `max_batch_size` for higher throughput with more memory
    - Reduce `max_batch_wait_ms` for lower latency
    - Use `ModelConfig(tp_size=...)` greater than 1 for large models that don't fit on a single GPU

## RADICAL AsyncFlow Integration

RHAPSODY integrates seamlessly with [RADICAL AsyncFlow](https://github.com/radical-cybertools/radical.asyncflow), a high-performance workflow engine for dynamic, asynchronous task graphs.

### Overview

RADICAL AsyncFlow provides:

- **Dynamic Task Graphs**: Create workflows with dependencies at runtime
- **Async/Await Syntax**: Natural Python async programming model
- **Decorator-Based API**: Simple function-to-task conversion
- **Backend Flexibility**: Use any RHAPSODY backend as execution engine

### Installation

Install RHAPSODY with AsyncFlow support:

```bash
pip install radical.asyncflow
pip install "rhapsody-py[dragon]"  # or your preferred backend
```

### Basic Usage

Here's a complete workflow example using AsyncFlow with RHAPSODY's Dragon backend:

```python
import asyncio
import multiprocessing as mp

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import DragonExecutionBackend

async def main():
    # Set Dragon as the multiprocessing start method
    mp.set_start_method("dragon")

    # Initialize RHAPSODY backend
    backend = await DragonExecutionBackend()

    # Create AsyncFlow workflow engine with RHAPSODY backend
    flow = await WorkflowEngine.create(backend=backend)

    # Define tasks using decorators
    @flow.function_task
    async def task1(*args):
        """Data generation task"""
        print("Task 1: Generating data")
        data = list(range(1000))
        return sum(data)

    @flow.function_task
    async def task2(*args):
        """Data processing task"""
        input_data = args[0]
        print(f"Task 2: Processing data, input sum: {input_data}")
        return [x for x in range(1000) if x % 2 == 0]

    @flow.function_task
    async def task3(*args):
        """Data aggregation task"""
        sum_data, even_numbers = args
        print(f"Task 3: Aggregating results")
        return {
            "total_sum": sum_data,
            "even_count": len(even_numbers)
        }

    # Define workflow with dependencies
    async def run_workflow(wf_id):
        print(f"Starting workflow {wf_id}")

        # Create task graph: task3 depends on task1 and task2
        # task2 depends on task1
        t1 = task1()
        t2 = task2(t1)  # task2 waits for task1
        t3 = task3(t1, t2)  # task3 waits for both task1 and task2

        result = await t3  # Await final task
        print(f"Workflow {wf_id} completed, result: {result}")
        return result

    # Run multiple workflows concurrently
    results = await asyncio.gather(*[run_workflow(i) for i in range(10)])

    print(f"Completed {len(results)} workflows")

    # Shutdown the workflow engine
    await flow.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

!!! important "Running with Dragon"
    When using Dragon backend with AsyncFlow, launch with the `dragon` command:
    ```bash
    dragon -m workflow.py
    ```

### Key Features

#### 1. Automatic Dependency Management

AsyncFlow automatically tracks dependencies between tasks based on function arguments:

```python
@flow.function_task
async def step1():
    return "data"

@flow.function_task
async def step2(input_data):
    return f"processed_{input_data}"

# AsyncFlow automatically creates dependency: step2 waits for step1
result1 = step1()
result2 = step2(result1)
await result2
```

#### 2. Concurrent Workflow Execution

Run multiple independent workflows in parallel:

```python
# Each workflow has its own task graph
workflows = [run_workflow(i) for i in range(1000)]

# Execute all workflows concurrently
results = await asyncio.gather(*workflows)
```

#### 3. Backend Interoperability

AsyncFlow works with any RHAPSODY backend:

```python
# Local execution
from rhapsody.backends import ConcurrentExecutionBackend
backend = await ConcurrentExecutionBackend()

# Dask cluster
from rhapsody.backends import DaskExecutionBackend
backend = await DaskExecutionBackend()

# Dragon HPC
from rhapsody.backends import DragonExecutionBackend
backend = await DragonExecutionBackend()

# Create workflow with chosen backend
flow = await WorkflowEngine.create(backend=backend)
```

### Performance Considerations

!!! tip "Scaling Guidelines"
    - Use Dragon backend for HPC-scale workflows (1000+ concurrent tasks)
    - Use Dask backend for distributed cluster computing
    - Use Concurrent backend for local development and testing

!!! note "Task Granularity"
    - AsyncFlow excels at dynamic, fine-grained task graphs
    - For coarse-grained tasks, consider using RHAPSODY's Session API directly
    - All RHAPSODY API capabilities are exposed to AsyncFlow to launch workloads and workflows.


!!! warning "Dual API Usage"
    It is highly recommended not to combine RHAPSODY
    API with AsyncFlow API due to the possibility of
    `asyncio.loop` blocking.

For more information on specific backends, see the [Advanced Usage](getting-started/advanced-usage.md) guide.
