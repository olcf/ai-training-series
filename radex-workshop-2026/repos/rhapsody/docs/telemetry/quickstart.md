# Quick Start

Telemetry is enabled with a single method call on the session. No external collector, database, or configuration file is needed.

## Installation

```bash
pip install 'rhapsody-py[telemetry]'
```

This pulls in `opentelemetry-sdk`. All other telemetry dependencies (`psutil`, `pynvml`) are optional and enabled when present.

---

## Minimal example

!!! note
    **Requires:** `pip install rhapsody-py[telemetry]`


```python
import asyncio
from rhapsody import Session, ComputeTask
from rhapsody.backends import ConcurrentExecutionBackend

async def main():
    backend = ConcurrentExecutionBackend(num_workers=4)
    session = Session(backends=[backend])

    # 1. Start telemetry — single await, returns TelemetryManager
    telemetry = await session.start_telemetry(
        resource_poll_interval=2.0,      # collect node metrics every 2 s
        checkpoint_path="./telemetry/",  # write a JSONL file here
    )

    # 2. Submit your workload
    tasks = [ComputeTask(executable="/bin/sleep", arguments=["0.1"]) for _ in range(20)]
    async with session:
        await session.submit_tasks(tasks)
        await session.wait_tasks(tasks)
    # session.close() (called by async with) stops telemetry automatically

    # 3. Inspect results — no OTel knowledge required
    print(telemetry.summary())

asyncio.run(main())
```

### Sample output

```python
{
    "session_id": "session.0000",
    "duration_seconds": 3.42,
    "tasks": {
        "submitted": 20,
        "completed": 20,
        "failed":     0,
        "canceled":   0,
        "running":    0,
    },
    "duration": {
        "total_seconds": 2.18,
        "mean_seconds":  0.109,
        "min_seconds":   0.101,
        "max_seconds":   0.134,
    },
    "resources": {
        "concurrent/myhost": {
            "backend":           "concurrent",
            "node_id":           "myhost",
            "cpu_percent":       41.2,
            "memory_percent":    28.7,
            "gpu_percent":       None,
            "disk_read_bytes":   4096.0,
            "disk_write_bytes":  0.0,
            "net_sent_bytes":    512.0,
            "net_recv_bytes":   1024.0,
        }
    }
}
```

---

## Reading metrics and spans

```python
# Read raw OTel MetricsData (after telemetry.stop())
metrics = telemetry.read_metrics()
for rm in metrics.resource_metrics:
    for sm in rm.scope_metrics:
        for metric in sm.metrics:
            for dp in metric.data.data_points:
                print(metric.name, dp.value)

# Read per-task spans as plain dicts
for span in telemetry.task_spans():
    print(
        span["task_id"],
        span["status"],
        span["duration_seconds"],
        span["trace_id"],
        span["span_id"],
    )
```

---

## Subscribing to live events

```python
def on_event(event):
    if event.event_type == "TaskCompleted":
        print(f"Task {event.task_id} done in {event.duration_seconds:.3f}s")
    elif event.event_type == "ResourceUpdate" and event.resource_scope == "per_gpu":
        print(f"GPU {event.gpu_id} on {event.node_id}: {event.gpu_percent:.1f}%")

telemetry.subscribe(on_event)
```

Async callbacks are also supported:

```python
async def async_handler(event):
    await my_metrics_queue.put(event)

telemetry.subscribe(async_handler)
```

Exceptions inside a subscriber are caught and logged — they never crash the dispatch loop.

---

## Emitting custom events

Application code can define and emit its own event types through the same telemetry bus without modifying RHAPSODY source:

```python
import time
from rhapsody.telemetry import define_event
from rhapsody.telemetry.events import make_event

# Define once at module level — name must be namespaced (contain a dot)
LineTimer = define_event("myapp.LineTimer", label=str, duration_ms=float)

# Emit from application code
t0 = time.time()
# ... work ...
telemetry.emit(
    make_event(
        LineTimer,
        session_id=telemetry.session_id,
        backend="app",
        label="preprocessing",
        duration_ms=(time.time() - t0) * 1000,
    )
)
```

`telemetry.emit()` is synchronous. Custom events flow through the same subscriber callbacks and appear in the JSONL checkpoint file.

See [Events & Metrics Reference — Custom events](reference.md#custom-events-define_event) for the full API.

---

## Using `TelemetrySubscriber` and `TelemetryReader`

For larger applications RHAPSODY provides typed wrapper classes:

```python
from rhapsody.telemetry import TelemetrySubscriber, TelemetryReader

sub    = TelemetrySubscriber(telemetry)
reader = TelemetryReader(telemetry)

# Push API
sub.subscribe(lambda e: print(e.event_type))

# Pull API — filter by session or task
task_spans = reader.read_traces(task_id="task.0042")
```

---

## Checkpoint file

When `checkpoint_path` is set, RHAPSODY writes a JSONL file:

```
telemetry/
└── rhapsody.session.session.0000.1775068460.telemetry.jsonl
```

The file is written **line-buffered** during the session — you can `tail -f` it in another terminal to watch events stream in real time.

### Plotting the checkpoint

```bash
python examples/plot.py telemetry/rhapsody.session.session.0000.1775068460.telemetry.jsonl
```

This produces a multi-panel PNG (shown in [Integrations — Real Run Visualization](integrations.md#real-run-visualization)) with CPU, memory, GPU (per-device), disk I/O, network I/O, task throughput, and task lifetime timeline.

---

## `start_telemetry` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `resource_poll_interval` | `float` | `5.0` | Seconds between resource metric polls |
| `checkpoint_interval` | `float \| None` | `None` | Seconds between periodic metric+span flushes. `None` = flush only at session end |
| `checkpoint_path` | `str \| None` | `None` | Directory for the JSONL file. `None` = no file output |
| `span_processors` | `list \| None` | `None` | OTel `SpanProcessor` instances (e.g. `BatchSpanProcessor(OTLPSpanExporter())`) added alongside RHAPSODY's internal `SpanBuffer`. Callers own exporter construction. |
| `metric_readers` | `list \| None` | `None` | OTel `MetricReader` instances (e.g. `PeriodicExportingMetricReader(OTLPMetricExporter())`) added alongside RHAPSODY's internal `InMemoryMetricReader`. |
| `resource` | `Resource \| None` | `None` | OTel `Resource` override. `None` = `Resource.create()` reads `OTEL_SERVICE_NAME` / `OTEL_RESOURCE_ATTRIBUTES` from the environment automatically. |

To access the manager after the initial call use `session.get_telemetry()`.

---

## What gets collected automatically

Once `await session.start_telemetry()` is called, RHAPSODY hooks into the task state manager and registers the appropriate backend adapter. You do not need to instrument individual tasks.

| Source | Automatic? | Notes |
|---|---|---|
| Task lifecycle events | ✅ | All state transitions are captured via the session's observer slot |
| Session start / end | ✅ | Emitted in `start()` / `stop()` |
| Node CPU & memory | ✅ | Polled by the backend adapter on the `resource_poll_interval` |
| Disk & network I/O | ✅ on Concurrent + Dragon | Not exposed by Dask's `scheduler_info()` for network |
| Per-GPU utilization | ✅ if `pynvml` installed (Concurrent) or Dragon runtime active | `resource_scope="per_gpu"` events with `gpu_id=N` |
