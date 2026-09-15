# Telemetry FAQ

---

## GPU utilization is not showing up — what should I check?

GPU metrics require both a supported GPU and the correct Python library. Work through the checklist below.

### 1. Is nvidia-ml-py installed? (NVIDIA GPUs)

RHAPSODY collects per-device GPU utilization via `nvidia-ml-py` (the NVIDIA Management Library Python bindings). It is an optional dependency — if it is absent, GPU metrics are silently skipped.

```bash
pip show nvidia-ml-py
```

If not installed:

```bash
pip install nvidia-ml-py
```

Or install RHAPSODY with the telemetry extra, which includes nvidia-ml-py:

```bash
pip install 'rhapsody-py[telemetry]'
```

### 2. Is the NVIDIA driver loaded?

`nvidia-ml-py` requires the NVIDIA kernel driver to be loaded on the node where RHAPSODY runs. If the driver is present but not active, `pynvml.nvmlInit()` raises an exception. RHAPSODY logs this as a `WARNING` once at startup — check your logs:

```
WARNING  rhapsody.telemetry.adapters.concurrent — nvidia-ml-py initialization failed on myhost — GPU metrics disabled: ...
```

Common causes and fixes:

| Symptom | Cause | Fix |
|---|---|---|
| `NVMLError: Driver Not Loaded` | NVIDIA kernel module not active | Load the driver: `sudo modprobe nvidia` |
| `NVMLError: Insufficient Permissions` | Running without access to `/dev/nvidia*` | Run as root, or add user to the `video` group |
| `NVMLError: No devices found` | No NVIDIA GPU detected | Confirm with `nvidia-smi` |

### 3. Is your GPU supported by nvidia-ml-py?

nvidia-ml-py wraps NVIDIA's NVML library, which supports NVIDIA GPUs from the **Fermi architecture (2010) onwards**. Very old cards (pre-Fermi) are not supported and will cause `nvmlInit` to fail.

Check your GPU generation:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
```

If your card is Kepler (2012) or newer, it is fully supported. If you see errors about unsupported GPUs, the card is likely too old for NVML.

### 4. Are you using a backend that exposes GPU metrics?

Not all RHAPSODY backends collect GPU utilization:

| Backend | GPU support | Mechanism |
|---|---|---|
| **Concurrent** | ✅ NVIDIA only | nvidia-ml-py (install required) |
| **Dragon** | ✅ NVIDIA, AMD, Intel | Dragon's built-in GPU collector |
| **Dask** | ❌ | `scheduler_info()` does not expose GPU metrics |

If you are using the Dask backend, GPU utilization will always be `null` — this is by design. Switch to the Concurrent or Dragon backend for GPU observability or use Dask Dashboard url (appears once the program starts) to visualize resource metrics in realtime.

### 5. Verifying GPU metrics are flowing

After fixing the issue above, confirm GPU events are being emitted:

```python
from rhapsody.telemetry.events import ResourceUpdate

def check_gpu(event):
    if isinstance(event, ResourceUpdate) and event.resource_scope == "per_gpu":
        print(f"GPU {event.gpu_id} on {event.node_id}: {event.gpu_percent:.1f}%")

telemetry.subscribe(check_gpu)
```

Or inspect the JSONL file:

```bash
jq 'select(.resource_scope == "per_gpu")' run.jsonl | head
```

If you see no `per_gpu` lines, GPU collection is still disabled — revisit steps 1–4 above.

---

## Why does my ResourceUpdate have `cpu_percent=null`?

This is expected for `resource_scope="per_gpu"` events. Per-GPU events carry only `gpu_percent` and `gpu_id`. To get CPU/memory for the same node at the same time, read the corresponding `resource_scope="per_node"` event and join on `(session_id, node_id)`.

---

## How do I tell node-level and per-GPU events apart?

Use the `resource_scope` field — it is always set and is the canonical discriminator:

```python
if event.resource_scope == "per_node":
    print(event.cpu_percent, event.memory_percent)
elif event.resource_scope == "per_gpu":
    print(event.gpu_id, event.gpu_percent)
```

Do not rely on `event.gpu_id is None` — while currently equivalent, `resource_scope` is the stable, documented API.

---

## Does telemetry add overhead to my workload?

Telemetry is designed to be low-overhead:

- The event bus is an `asyncio.Queue` with `put_nowait` — O(1), non-blocking from any context.
- Resource polling runs in a **daemon thread** (Concurrent) or a separate **Dragon worker process** (Dragon), so it never competes with the asyncio event loop.
- Default poll interval is 5 seconds. For dashboards, 1–2 seconds is fine. Sub-second polling is not recommended in production.
- The JSONL file is line-buffered — no fsync per event.

At default settings (5 s poll, checkpoint on stop), telemetry overhead is negligible. For a 6-hour run with 10 nodes, expect a JSONL file of roughly 10–50 MB depending on task throughput.

---

## Can I use telemetry without writing a checkpoint file?

Yes. Omit `checkpoint_path`:

```python
telemetry = await session.start_telemetry(resource_poll_interval=5.0)
```

All in-memory data (metrics, spans, event log) remains available via `telemetry.summary()`, `telemetry.read_metrics()`, and `telemetry.read_traces()` until `session.close()` is called.

---

## Can I send telemetry to Grafana / Jaeger / Prometheus?

Yes — see [Integrations & Extensions](integrations.md) for step-by-step setup with Prometheus+Grafana (via the OTel Prometheus exporter) and any OTLP-compatible backend (Jaeger, Grafana Tempo, Dynatrace).
