# Telemetry

RHAPSODY ships a built-in, zero-configuration telemetry layer (the **Telemetry Abstraction Layer**, or TAL) that turns every session run into structured, queryable observability data — without requiring an external collector, database, or agent.

## What you get out of the box

| Layer | What is collected |
|---|---|
| **Session** | Start / end timestamps, total duration, session span (root of the OTel trace) |
| **Tasks** | Full lifecycle events (Created → Submitted → Queued → Started → Completed / Failed) with wall-clock timestamps, per-task duration, backend, executable identity |
| **Resources** | Per-node CPU %, memory %, disk I/O bytes/interval, network I/O bytes/interval, per-GPU utilization % with device index |
| **OTel SDK** | Counters, gauges, histograms, and traces stored in-memory and written to a JSONL checkpoint file |

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RHAPSODY Session                            │
│                                                                     │
│  ┌───────────┐   events   ┌──────────────────────────────────────┐  │
│  │  Session  │ ─────────► │         TelemetryManager             │  │
│  │  Task SM  │            │  ┌─────────────┐  ┌───────────────┐  │  │
│  └───────────┘            │  │ asyncio     │  │  OTel SDK     │  │  │
│                           │  │ EventBus    │  │  MeterProvider│  │  │
│  ┌───────────────────┐    │  │  (Queue)    │  │  Tracer       │  │  │
│  │ Backend Adapters  │    │  └──────┬──────┘  └───────┬───────┘  │  │
│  │  ┌─────────────┐  │    │         │                 │          │  │
│  │  │ Concurrent  │  │    │  ┌──────▼─────────────────▼───────┐  │  │
│  │  │  (psutil +  │  │    │  │         Dispatch Loop          │  │  │
│  │  │   pynvml)   │  │    │  │  • Update OTel instruments     │  │  │
│  │  ├─────────────┤  │    │  │  • Open / close task spans     │  │  │
│  │  │  Dask       │  │    │  │  • Attach trace_id / span_id   │  │  │
│  │  │ (scheduler  │  │    │  │  • Fan-out to subscribers      │  │  │
│  │  │   _info)    │  │    │  │  • Write JSONL checkpoint      │  │  │
│  │  ├─────────────┤  │    │  └────────────────────────────────┘  │  │
│  │  │  Dragon     │  │    └────────────────────────────────────-─┘  │
│  │  │ (dps queue) │  │                                              │
│  │  └─────────────┘  │    ┌─────────────┐  ┌──────────────────────┐ │
│  └───────────────────┘    │  Subscribers│  │  JSONL Checkpoint    │ │
│                           │  (push API) │  │  .telemetry.jsonl    │ │
│                           └─────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Documentation pages

<div class="grid cards" markdown>

-   :material-chart-timeline: **[Events & Metrics Reference](reference.md)**

    Complete catalog of every event type, field, OTel instrument, and span emitted by RHAPSODY.

-   :material-rocket-launch: **[Quick Start](quickstart.md)**

    Enable telemetry in three lines; read metrics and spans with zero OTel knowledge.

-   :material-puzzle: **[Integrations & Extensions](integrations.md)**

    Build dashboards, alerting, and custom analytics on top of RHAPSODY telemetry using Prometheus, Grafana, Jaeger, or your own subscriber.

-   :material-help-circle: **[FAQ](faq.md)**

    Troubleshooting GPU metrics, understanding `resource_scope`, overhead estimates, and other common questions.

</div>

## How telemetry reuses the backend runtime

RHAPSODY backends already have native runtime monitoring — Dragon exposes a `dps` metric queue, Dask exposes `scheduler_info()`, the concurrent backend runs in a single process with `psutil`. Rather than adding a separate monitoring sidecar, RHAPSODY **wraps each backend's native source** with a thin adapter:

```
Backend runtime            Adapter                 TelemetryManager
─────────────────          ────────────────────     ──────────────────────
Dragon dps queue    →  DragonTelemetryAdapter  →   emit(ResourceUpdate)
Dask scheduler_info →  DaskTelemetryAdapter    →   emit(ResourceUpdate)
psutil + pynvml     →  ConcurrentTelemetry     →   emit(ResourceUpdate)
                        Adapter
```

This means:

- **No extra dependencies** are added to a backend that does not already use them.
- **Polling intervals** are configurable and low-overhead (default 5 s, can be 0.1 s for dashboards).
- **GPU data is per-device**: on a 4-GPU node the adapter emits one aggregate `ResourceUpdate` (with `gpu_id=None`) plus one per-GPU `ResourceUpdate` (with `gpu_id=0..3`), matching the OTel hierarchy.

## OpenTelemetry compatibility

All RHAPSODY telemetry data is produced through the **OpenTelemetry Python SDK**. RHAPSODY creates its own private `TracerProvider` and `MeterProvider` with in-memory storage. To forward data to any OTel-compatible backend (Jaeger, Tempo, Prometheus, Dynatrace, Honeycomb), pass pre-built `SpanProcessor` and/or `MetricReader` instances to `start_telemetry()` — no other RHAPSODY code changes required. See [Integrations & Extensions](integrations.md#connecting-an-otlp-exporter-jaeger-tempo-dynatrace) for examples.

The JSONL checkpoint file is a portable, OTel-aligned export that can be replayed, plotted, or ingested into any time-series tool.

!!! tip "Install"
    ```bash
    pip install 'rhapsody-py[telemetry]'
    ```
