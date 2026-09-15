# Changelog

## [Unreleased]

### Added

- **Dragon 0.14.1 public API migration completed** for `DragonExecutionBackend`
  — task metadata (`name`, `timeout`, `stdout`/`stderr`) now goes through
  `Batch.options()` rather than direct kwargs to `process()`/`job()`/`function()`,
  matching the 0.14.1 API. `V1`/`V2` backends and the `BatchError`/`Policy`
  imports they required are removed; `DragonExecutionBackendV3` is kept only as
  a deprecated alias.
- **`DragonVllmInferenceBackend` migrated to `dragon.ai.inference`** — moved from
  `rhapsody.backends.inference` to `rhapsody.backends.ai`, and from a YAML
  `config_file=` argument to typed config objects (`ModelConfig`,
  `HardwareConfig`, `BatchingConfig`, `GuardrailsConfig`, `DynamicWorkerConfig`
  in the new `rhapsody.backends.ai.config`), re-exported as-is from
  `dragon.ai.inference`. **Breaking change** for existing callers still passing
  `config_file="config.yaml"` — see `docs/integrations.md` for the new
  constructor shape.
- `DragonVllmInferenceBackend` now supports `await DragonVllmInferenceBackend(...)`
  as a single step, consistent with every other backend (`DragonExecutionBackend`,
  `ConcurrentExecutionBackend`, `DaskExecutionBackend`, `RadicalExecutionBackend`).
  The existing two-step `backend = DragonVllmInferenceBackend(...); await backend.initialize()`
  pattern still works unchanged. Docs, README, examples, and the tutorial
  notebook updated to the new pattern.

  - **`OrbitExecutionBackend`** — submits tasks to a remote HPC node through the
  ORBIT broker/endpoint infrastructure (`radical.orbit`); the endpoint node
  runs a RHAPSODY plugin with a local backend (e.g. Dragon) that actually
  executes the work. Delegates to `RhapsodyClient` internally, inheriting
  template compression, pipelined batching, and event-based wait/batch
  notifications.
- **`NoopExecutionBackend`** — tasks are marked `DONE` immediately without
  executing anything; for benchmarking RHAPSODY's own submission/orchestration
  overhead independent of any real backend.

### Fixed

- **Task completions silently dropped under load** — `submit_tasks()` used to
  register each task into `_monitored_batches` only after building the *entire*
  batch, so a fast task could complete and be polled by the monitor thread
  before its own registration landed; the monitor loop then treated the
  unmatched tuid as "already delivered as cancelled" and dropped the result
  permanently. `build_task()` now registers each task immediately after Dragon
  dispatch, and the monitor loop buffers (`_pending_tuids`) and retries any
  tuid that arrives before registration instead of assuming cancellation.
  `cancel_task()` now marks confirmed cancellations in `_cancelled_tuids` so
  genuinely cancelled tuids are still discarded immediately.
- **Dragon exit codes not surfacing as `FAILED`** — Dragon 0.14.1's
  `process()`/`job()` tasks return a non-zero exit code as the task *result*
  instead of raising, so a failed executable was silently reported `DONE`.
  Non-zero `int`/`list[int]` results from process/job tasks are now normalized
  to a raised `SystemExit` so RHAPSODY delivers `FAILED` as expected.
- **`runinfo/` directory left behind after every run** — Dragon's `task_logs`
  feature (needed so `capture_stdio=False` tasks get real stdout/stderr content
  back via `get_stdout()`/`get_stderr()`) creates a `runinfo/<run-id>/` tree and
  `manifest.jsonl` in the working directory. `shutdown()` now removes it via
  `Batch.log_dir()` after `join()`/`destroy()` complete.
- **CI silently reporting broken Dragon installs as "skipped"** — `make test-ci`
  treated *any* `test-dragon-only` failure (dragon launcher genuinely missing,
  or dragon present but broken) as "Dragon not available on this Python
  version" and reported success either way. It now only skips when the
  `dragon` launcher itself isn't on `PATH`; a real test failure now fails the
  build.
- **Missing `flask`/`gunicorn` dependency** — `dragon.workflows.batch` imports
  `dragon.telemetry.telemetry` unconditionally, which requires `flask`/
  `gunicorn` even when telemetry is never used; `dragonhpc`'s own package
  metadata doesn't declare them. Added to the `dragon` extra in
  `pyproject.toml`, and CI now installs via `pip install -e ".[dragon]"`
  instead of a bare `pip install dragonhpc` so the extra actually takes effect.
- Stale test assertion in `test_monitor_loop_get_called_after_poll` expecting
  `get(block=False)`; the code correctly calls `get(block=True)` (poll() and
  the DDict write aren't atomic) since an earlier fix — the test just never
  caught up.

## [0.4.0] - 2026-06-11

### Added

- **Dragon 0.14.0 full migration** — `DragonExecutionBackendV3` is fully ported
  to the DragonHPC 0.14.0 API. Key changes:
  - **Sharded DDict** — task results are now read from the correct DDict shard
    using `results_ddict.manager(shard_idx)[tuid]`, matching how Dragon 0.14.0
    writes results per `manager_idx`. The previous unsharded access caused the
    monitor loop to block indefinitely.
  - **Real task cancellation** — `cancel_task()` now calls Dragon's
    `batch_task.cancel()` instead of a soft/fake cancel. Cancelled tasks are
    eagerly removed from both `_task_registry` and `_monitored_batches`,
    preventing the monitor thread from stalling on DDict polls for tasks that
    will never deliver results.
  - **Stdout capture restored** — Dragon 0.14.0 removed the unconditional
    `stdout=Popen.PIPE` that Dragon 0.13.2 applied automatically. RHAPSODY now
    explicitly injects it across all five `ProcessTemplate` construction paths so
    stdout is captured for every process and executable task by default. Users can
    override it via `process_template`.
  - **Python 3.13 support** — Dragon 0.14.0 extends support to Python 3.13; the
    CI matrix and `pyproject.toml` extras comment are updated accordingly.

- **HPC integration test suite** (`tests/integration/test-hpc/dragon/`) — 25
  end-to-end tests across five categories: multi-node GPU execution, single-node
  GPU, task pinning and affinity, scale/stress (10 K+ tasks), and MPI
  (mpi4py gather/allreduce/broadcast). Topology is detected dynamically from
  `dragon.native.machine.System`; tests auto-skip when the required resources
  are absent. The entire suite is excluded from CI via a `pytest_ignore_collect`
  hook and only runs on a live cluster with `RHAPSODY_HPC=1`.

- **HPC machine guides** — new setup guides for PSC Bridges-2 (SSH-based Dragon
  launcher with TCP transport) and Purdue Anvil (Cray PE stack, identical to
  Delta). The HPC machines index now clarifies that `ConcurrentExecutionBackend`
  works out of the box on every machine, while `DragonExecutionBackendV3`
  requires machine-specific configuration documented per page.

- **Team and contributor documentation** — `docs/project/team.md` and
  `README.md` updated to credit both the RADICAL Research Group (Rutgers) and
  the HPE team.

### Changed

- **`TaskStateManager.update_task`** — removed per-call `asyncio.get_running_loop()`
  / `try-except RuntimeError` on-loop detection. All backends already call this
  from the event loop; the check was firing 128 K+ times per 64 K-task workload
  and always taking the inline branch. The method now calls the implementation
  directly. Unused `asyncio.Lock` also removed.

- **`_build_process_template_kwargs` helper** — ProcessTemplate construction
  across all five task priorities is consolidated into a single helper, adopted
  from the `ashao/dragon-executor-hanging-bugfix` branch and adapted for
  RHAPSODY's `stdout_pipe` semantics. Co-Authored-By: Andrew Shao (HPE).

### Fixed

- **Monitor loop race condition** — the two-step `if tuid not in dict` / `dict[tuid]`
  access in `_monitor_loop` was a TOCTOU race with `cancel_task` running on the
  event loop. Replaced with a single atomic `dict.get(tuid)` call.

- **Python 3.13 `asyncio.get_event_loop()` error** — the `backend_v3` test
  fixture called `asyncio.get_event_loop()` in a sync context. Python 3.13
  tightened this from a `DeprecationWarning` to a hard `RuntimeError`. Fixed
  by using `asyncio.new_event_loop()` instead.

- **`long_running` task fixture unresponsive to cancellation** — the integration
  test helper used `time.sleep(60)`, which ignores Dragon's `kill_by_exception`
  mechanism until the sleep finishes (up to 60 seconds). Changed to a
  `while True: time.sleep(0.1)` loop so the Dragon worker is freed within
  ~100 ms of cancellation.

## [0.3.1] - 2026-05-19

### Fixed

- Pinned `dragonhpc==0.13.2` in the `dragon` optional dependency. Dragon 0.14.0
  changed DDict write routing in `Batch` such that `DragonExecutionBackendV3`'s
  monitor loop blocks indefinitely on every task result. Users on Dragon 0.14.0
  should upgrade to the forthcoming release that includes the full migration fix.

## [0.3.0] - 2026-05-15

### Added

- **`SpanBuffer`** — new public, thread-safe span collector that replaces
  `InMemorySpanExporter` as RHAPSODY's internal span store. Uses `list.extend()`
  for O(1) appends (vs. the exporter's O(n²) `tuple +=` per flush). `shutdown()`
  is a no-op so spans remain accessible after `tracer_provider.shutdown()`.
  `get_finished_spans()`, `clear()`, and `force_flush()` match the
  `InMemorySpanExporter` API.

- **Provider-agnostic OTel injection** — `TelemetryManager.__init__()` and
  `Session.start_telemetry()` accept three new optional parameters:
  - `span_processors: list | None` — OTel `SpanProcessor` instances
    (e.g. `BatchSpanProcessor(OTLPSpanExporter())`) added to RHAPSODY's private
    `TracerProvider` alongside the internal `SpanBuffer`. Callers own exporter
    construction; RHAPSODY never imports specific exporter classes.
  - `metric_readers: list | None` — OTel `MetricReader` instances
    (e.g. `PeriodicExportingMetricReader(OTLPMetricExporter())`) added to
    RHAPSODY's private `MeterProvider` alongside `InMemoryMetricReader`.
  - `resource: Resource | None` — optional OTel `Resource` override.
    `None` (default) calls `Resource.create()` which reads `OTEL_SERVICE_NAME`
    and `OTEL_RESOURCE_ATTRIBUTES` from the environment automatically.

- **`TelemetryManager.annotate_current_span(event)`** — records a `BaseEvent`
  as an OTel span event (with serialized attributes) on the currently active
  span. No-op when no valid span is active or telemetry is stopped.

- **`TelemetryManager.register_span_enricher(func)`** — registers a callback
  invoked at `TaskCreated` time that returns extra `dict` attributes to stamp
  onto the task's OTel span. Multiple enrichers are applied in registration
  order. Used by orchestration layers (e.g. AsyncFlow) to inject workflow-level
  grouping keys.

- **`TelemetryManager.export_as_otlp(path=None)`** — serializes all finished
  spans to a standards-compliant OTLP JSON payload (the POST `/v1/traces` wire
  format: 32/16-char hex IDs, nanosecond timestamps as strings, typed KV
  attributes). Returns the JSON string and optionally writes to `path`.

- **`TelemetryManager._task_contexts`** — internal dict that captures the OTel
  context (`context.get_current()`) at `TaskCreated` emit time and restores it
  via `attach/detach` in `_process_event()`. This is the standard in-process OTel
  propagation pattern across asyncio task boundaries.

- **`TelemetryManager._session_ctx_token`** — stores the attach token for the
  session span context, detached cleanly in `stop()`.

### Performance

- **Batch-drain dispatch loop** — replaced the per-event
  `asyncio.wait_for(queue.get(), timeout=0.05)` pattern in `_dispatch_loop`
  with a batch-drain approach that reduces asyncio event-loop interactions from
  O(N) to O(N/500) under burst load.

  Key changes:
  - Fast path: drain up to 500 events per iteration with `queue.get_nowait()`
    (plain `deque.popleft()` — zero asyncio scheduling overhead, no timer handle,
    no `Task` allocation per event). Fall back to a blocking `await queue.get()`
    only when the queue is empty.
  - `asyncio.sleep(0)` yielded once per *full* batch, not once per event.
    Partial batches block naturally on the next `await queue.get()`.
  - `_flush_batch()` — new helper that processes OTel instruments, writes the
    JSONL checkpoint, and dispatches to subscribers for an entire batch at once.
    Subscriber dispatch is guarded by `if self._subscribers` to skip the inner
    loop when no subscribers are registered.
  - `_write_batch()` — new helper that serializes a batch of events and issues a
    **single `file.write()` call** per batch. Uses
    `dataclasses.fields() + getattr` instead of `dataclasses.asdict()`: avoids
    the deep-recursive copy, and correctly captures `init=False` class-level
    fields (e.g. `event_type`) that `vars()` / `__dict__` miss on frozen
    dataclass subclasses.
  - Checkpoint file opened with `buffering=131072` (128 KB block buffer) instead
    of `buffering=1` (line-buffered). The previous setting issued one `write()`
    syscall per event — up to 80 000 syscalls for a 40 K-task session.
  - **Sentinel-based shutdown** — `stop()` sets `_running=False`, awaits
    `queue.join()` with a 5 s timeout to guarantee full drain, then pushes
    `_STOP_SENTINEL` to unblock the dispatch loop cleanly. `emit()` is now a
    no-op after `stop()` to prevent late emissions racing with the sentinel.

  Measured at 40 000 tasks (`DragonExecutionBackend`):

  | | Telemetry overhead |
  |---|---|
  | Before (per-event `wait_for`) | +7 s (3.66× slower than no-telemetry) |
  | After (batch-drain) | +2 s (1.73× slower than no-telemetry) |

  **71% reduction in telemetry overhead** at 40 K tasks.

### Fixed

- **Task callback contract (`stdout`/`stderr`)** — `task["stdout"]` and `task["stderr"]` are now guaranteed to be `str` (never absent, never `None`) after any DONE or FAILED callback across all backends. `DragonExecutionBackendV3`, `ConcurrentExecutionBackend`, and `DaskExecutionBackend` each had paths that left the key missing or set it to `None`, causing `KeyError: 'stdout'` in consumers using bracket notation. A new `pytest.mark.result_contract` marker groups seven cross-backend regression tests that enforce this invariant.

- **`TaskCreated` as task span head** — task spans were previously opened at
  `TaskSubmitted`, leaving `TaskCreated` (the earliest lifecycle event) with
  only a `trace_id` and no `span_id` in the JSONL. The span is now opened at
  `TaskCreated` so all lifecycle events carry both `trace_id` and `span_id`,
  making the JSONL 100% OTel-aligned from the first event.

- **Same-batch span ordering hazard** — when `TaskCreated` and
  `TaskCompleted` / `TaskFailed` land in the same 500-event batch,
  `_process_event` closes the span before `_span_ids_for_event` can look it up
  for intermediate events in the batch. Fixed by:
  - `_completed_span_ctx` dict: stores the `SpanContext` snapshot when a span
    is ended so intermediate events in the same batch can still retrieve it
    via `.get()` (not `.pop()`).
  - Terminal events (Completed/Failed/Canceled) `.pop()` the entry, ensuring
    no unbounded growth.
  - Same fix applied to the session span: `_completed_session_ctx` snapshot is
    taken at `SessionEnded` so JSONL serialization for same-batch session events
    still finds a valid context.

- **`ResourceUpdate` JSONL correlation** — `ResourceUpdate` events now carry
  the session span's `trace_id` and `span_id` (previously `(None, None)`),
  making node and GPU metrics queryable within the session trace in Jaeger /
  Tempo.

- **`TaskSubmitted` / `TaskQueued` fallback** — if `TaskCreated` was never
  seen (incomplete lifecycle, e.g. task registered before telemetry started),
  `TaskSubmitted` opens the task span instead, preserving best-effort
  correlation.

- **External trace/span injection** — components outside RHAPSODY (e.g. an
  orchestration layer or application code) can now emit events that carry an
  explicit `trace_id` / `span_id` and have them written to the JSONL with the
  correct OTel correlation IDs, enabling cross-component trace stitching.

- **Memory leak in `ConcurrentExecutionBackend._run_executable`** — file
  handles (`stdout_f`, `stderr_f`) opened for `capture_stdio=True` tasks were
  not closed on exception paths (subprocess launch failure, timeout, etc.).
  Refactored to initialize both handles and `exit_code` before the `try` block
  and close them in `finally`, eliminating the leak.

- **OTel span parent hierarchy** — task spans previously had no parent span and
  scattered across multiple unrelated traces. Root cause: `_dispatch_loop` runs
  in a separate asyncio `Task` and does not inherit the calling coroutine's OTel
  context. Fix: capture `context.get_current()` at `TaskCreated` emit time,
  store in `_task_contexts`, restore via `attach/detach` in `_process_event()`.
  All task spans now correctly nest under the session root span in the trace
  hierarchy.

- **`span_scope()` session span fallback** — when called from a context with no
  valid active span (e.g. inside a block body spawned before `start_telemetry()`),
  `span_scope()` now explicitly uses `self._session_span` as the parent context
  instead of creating a root span. Replaces custom `_null_context()` with stdlib
  `nullcontext`.


### Changed

- **`TelemetryManager.export_as_otlp()`** — renamed from
  `export_traces_otlp_json()` for consistency with the OTel ecosystem naming
  convention.

- **`TelemetryManager.start()` OTel provider setup** — always uses
  `Resource.create({"service.name": "rhapsody", "session_id": ...})` as the
  default resource (reads `OTEL_SERVICE_NAME` from env automatically as
  fallback). Caller-supplied `span_processors` are added via
  `tracer_provider.add_span_processor()` after the internal `SpanBuffer` processor.
  Caller-supplied `metric_readers` are appended to the `MeterProvider` reader
  list alongside the internal `InMemoryMetricReader`.

### Docs

- **`docs/telemetry/integrations.md`**:
  - Fixed "Connecting an OTLP exporter" section: removed the broken
    `set_tracer_provider()` pattern (RHAPSODY creates private providers and
    ignores globals) and replaced with the correct `span_processors=` injection.
    Added tip block showing env-var configuration for Honeycomb/Grafana Cloud.
  - Fixed "Grafana + Prometheus Step 3": removed `set_meter_provider()` pattern;
    replaced with `metric_readers=[PrometheusMetricReader()]`.
  - Updated "What RHAPSODY produces" table: `InMemorySpanExporter` → `SpanBuffer`.

- **`docs/telemetry/quickstart.md`**: added three missing rows to the
  `start_telemetry` parameters table (`span_processors`, `metric_readers`,
  `resource`).

- **`docs/telemetry/reference.md`**: added "Span annotation and enrichment API"
  section documenting `span_scope()`, `annotate_current_span()`,
  `register_span_enricher()`, and `export_as_otlp()`.

- **`docs/telemetry/index.md`**: updated "OpenTelemetry compatibility" paragraph
  to accurately describe the private-provider model and direct users to the
  `span_processors`/`metric_readers` injection API.

- `ComputeTask`: new `capture_stdio: bool = False` parameter. When `True`, stdout and stderr from executable tasks are redirected directly to files (`{work_dir}/{session_uid}/{task_uid}.stdout` / `.stderr`) with zero in-memory buffering. `task.stdout` and `task.stderr` then hold file paths instead of decoded strings. Function tasks are unaffected regardless of the flag value.
- `ConcurrentExecutionBackend`: honours `capture_stdio` by passing open file handles to `asyncio.create_subprocess_exec` / `asyncio.create_subprocess_shell`; the kernel writes directly, no Python-level buffering.
- `DaskExecutionBackend`: honours `capture_stdio` in the module-level `_run_executable` function; file handles are passed to `subprocess.run` inside the worker process.
- `DragonExecutionBackendV3`: honours `capture_stdio` by wrapping the command in `bash -c "cmd 1>out 2>err"`, bypassing Dragon's internal PIPE infrastructure entirely.
- `BaseBackend`: added `_work_dir: str` (defaults to `os.getcwd()`) and `is_attached: bool = False`. `Session.add_backend()` is now the single authority that sets `_work_dir = {session.work_dir}/{session.uid}` and flips `is_attached = True`.
- `Session.add_backend()`: raises `RuntimeError` if the same backend instance is added to more than one session, preventing silent sharing bugs.
- Unit tests for `capture_stdio` across `ConcurrentExecutionBackend`, `DaskExecutionBackend`, and `DragonExecutionBackendV3`.
- Getting-started guide: new **Capturing stdout/stderr to Files** section in `docs/getting-started/advanced-usage.md`.

## [0.2.0] - 2026-04-04

### Added

- **Telemetry Abstraction Layer (TAL)**: new `rhapsody.telemetry` package providing event-driven, backend-agnostic observability built on the OpenTelemetry Python SDK.
    - `TelemetryManager`: central orchestrator owning the asyncio event bus, OTel `MeterProvider` + `TracerProvider` (in-memory), JSONL checkpoint writer, and subscriber fan-out.
    - `Session.enable_telemetry(resource_poll_interval, checkpoint_interval, checkpoint_path)`: one-call activation; returns a `TelemetryManager` wired into the task state manager and all registered backends.
    - **Event types**: `SessionStarted`, `SessionEnded`, `TaskSubmitted`, `TaskQueued`, `TaskStarted`, `TaskCompleted`, `TaskFailed`, `ResourceUpdate` — all frozen dataclasses with consistent `event_id`, `event_time`, `emit_time`, `session_id`, `backend` fields.
    - **OTel trace hierarchy**: session span (root) → task spans (children, opened on `TaskStarted`, closed on `TaskCompleted`/`TaskFailed`) → `ResourceUpdate` events anchored to the session span. Every JSONL line carries `trace_id` and `span_id`.
    - **`ResourceUpdate.resource_scope`**: computed discriminator field (`"per_node"` or `"per_gpu"`) set automatically from `gpu_id`. Replaces ad-hoc `gpu_id is None` checks throughout consumers.
    - **Per-GPU telemetry**: `ResourceUpdate` events with `resource_scope="per_gpu"` carry only `gpu_percent` and `gpu_id`; `cpu_percent`, `memory_percent`, and all I/O fields are `None` on per-GPU events. Node-level aggregates use `resource_scope="per_node"`.
    - **Backend adapters**:
        - `ConcurrentTelemetryAdapter`: psutil for CPU/memory/disk/net; pynvml (NVIDIA only) for per-device GPU, initialized once at thread start. `ImportError` on missing pynvml is silent; driver failures (e.g., no NVIDIA kernel module loaded) log a one-time `WARNING`.
        - `DaskTelemetryAdapter`: polls `scheduler_info()` per worker; emits disk I/O deltas from cumulative Dask counters; net I/O and GPU not exposed by Dask.
        - `DragonTelemetryAdapter`: reads Dragon's `dps` metric queue; emits per-device GPU events tagged with `device_id`; supports NVIDIA, AMD, and Intel GPU vendors.
    - **Adapter registration**: failures during `_attach_telemetry_adapter` now log at `WARNING` with full traceback instead of silently passing.
    - `TelemetryReader` and `TelemetrySubscriber`: typed pull/push interfaces over `TelemetryManager`.
    - `TelemetryManager.summary()`, `.task_spans()`, `.task_count()`, `.session_duration()`: convenience methods returning plain dicts; no OTel knowledge required.
    - JSONL checkpoint file (`rhapsody.session.<id>.<ts>.telemetry.jsonl`): line-buffered, readable during a live run; contains `event`, `metric`, and `span` sections.
    - OTel contract test suite: `tests/unit/telemetry/conftest.py` (`AdapterCapabilities`, `assert_resource_update_contract`) + `tests/unit/telemetry/test_otel_contract.py` (parametrized across Concurrent, Dask, Dragon).
    - Full telemetry documentation: `docs/telemetry/` (overview, events & metrics reference, quick start, integrations).
- `DragonExecutionBackendV3`: migrated to the Dragon batch.py streaming pipeline. Tasks submitted via `session.submit_tasks()` are dispatched individually by a continuously running background thread — there is no compile or start step.
- `DragonExecutionBackendV3`: accepts two new constructor parameters: `num_nodes` (total nodes) and `pool_nodes` (nodes per worker pool), forwarded directly to `Batch()`.
- `DragonExecutionBackendV3.fence()`: new method that delegates to `batch.fence()`, allowing callers to wait for all in-flight tasks submitted by this client to complete.
- `DragonExecutionBackendV3.create_ddict()`: new helper that creates a Dragon `DDict` instance directly from the backend.
- `DragonExecutionBackendV3._deliver_batch()`: replaces `_deliver_result` / `_deliver_failure`. All tasks that complete within a single monitor sweep are collected into a list and delivered to the asyncio event loop in **one** `call_soon_threadsafe` call instead of one per task — reducing cross-thread wakeups from O(tasks) to O(sweeps).
- `DragonExecutionBackendV3` result monitoring reads directly from `Batch.results_ddict` (the same distributed dict Dragon workers write to) instead of going through the `Task` object. The membership check and value read are now a single `try/except KeyError` operation, eliminating the redundant `__contains__` network round-trip (was two DDict RTTs per ready task, now one).
- `DragonExecutionBackendV3`: monitor thread now starts lazily on first `submit_tasks()` call instead of at `_async_init` time, eliminating the idle-spin phase while tasks are being built and registered.
- `DragonExecutionBackendV3._cancelled_tasks` converted from `list` to `set`: O(1) membership checks and automatic deduplication. Cancelled UIDs are now removed from the set once the monitor loop processes the task, preventing unbounded growth.
- `DragonExecutionBackendV3._task_registry` entries are now removed atomically (via `dict.pop`) on result or failure delivery, eliminating unbounded registry growth for long-running sessions.

### Changed

- `ResourceUpdate.cpu_percent` and `ResourceUpdate.memory_percent`: type changed from `float` (default `0.0`) to `float | None` (default `None`). These fields are `None` on `resource_scope="per_gpu"` events and non-null floats on `resource_scope="per_node"` events. Consumers filtering node-level events should use `event.resource_scope == "per_node"` instead of `event.gpu_id is None`.

### Fixed

- `ConcurrentTelemetryAdapter`: replaced nvidia-smi subprocess (one blocking call per poll, no per-device breakdown) with pynvml initialized once at thread start. Per-device GPU utilization is now collected with device index matching Dragon's pattern.
- `ConcurrentTelemetryAdapter`: pynvml driver failure (e.g., driver not loaded, permission denied) now logs a one-time `WARNING` instead of silently disabling GPU metrics with no indication.
- `ConcurrentExecutionBackend`: regular (synchronous) functions are now executed correctly in both `ThreadPoolExecutor` and `ProcessPoolExecutor`. Previously, all function tasks were dispatched via `asyncio.run(func(...))`, which raised `ValueError` for non-coroutine callables. The executor now detects `asyncio.iscoroutinefunction` and calls sync functions directly.
- `Session` / `TaskStateManager`: task futures now propagate exceptions on failure. Previously all futures were resolved with `set_result(task)` regardless of outcome. Failures now resolve as:
    - Function task raises → original exception propagated via `fut.set_exception(exc)`.
    - Executable task returns non-zero exit code → `fut.set_exception(TaskExecutionError(uid, stderr, exit_code))`.
    - Successful tasks → `fut.set_result(task)` (unchanged).
- `Session.wait_tasks`: replaced bare `except Exception` (which silently swallowed all errors) with `asyncio.gather(*futures, return_exceptions=True)`. Task failures no longer propagate through `wait_tasks`; callers inspect `task.state` / `task.exception` / `task.stderr` directly.
- API documentation: new **Wait Modes** reference in `docs/api/index.md` with a comparison table of `wait_tasks`, `asyncio.gather(*futures)`, and `await future / await task`, including exception types and when to use each.
- API documentation: new **Callable Task API** section documenting sync and async function support, executor behaviour, and result access fields.
- Getting-started guide: new **Wait Modes** and **Sync and Async Callable Tasks** sections added to `docs/getting-started/advanced-usage.md` with runnable code examples.
- `ConcurrentExecutionBackend`: executable tasks now honour per-task working directory via `task.cwd` (task-level) or `task_backend_specific_kwargs={"cwd": "..."}` (overrides task-level); passed as `cwd=` to `asyncio.create_subprocess_exec` / `asyncio.create_subprocess_shell`.
- `DaskExecutionBackend`: executable tasks now honour `cwd` from `task_backend_specific_kwargs` (overrides task-level `task.cwd`); forwarded to `_run_executable` which passes it as `cwd=` to `subprocess.run`.
- `DragonExecutionBackendV3`: documented that per-task working directory must be set via `task_backend_specific_kwargs={"process_template": {"cwd": "..."}}` (or `"process_templates"` for MPI jobs); this is the only supported mechanism in V3.
- `ConcurrentExecutionBackend`: executable tasks now honour `env` from `task_backend_specific_kwargs={"env": {...}}`; passed as `env=` to `asyncio.create_subprocess_exec` / `asyncio.create_subprocess_shell`. `None` (default) inherits the parent process environment.
- Unit tests for `cwd` and `env` behaviour across `ConcurrentExecutionBackend`, `DaskExecutionBackend`, and `DragonExecutionBackendV3`.
- `DaskExecutionBackend` now supports all three `ComputeTask` types: sync functions, async functions, and executable tasks (via subprocess inside a Dask worker).
- Cluster-agnostic design: pass `cluster=` or `client=` to target any Dask-compatible cluster (SLURM, Kubernetes, LocalCUDACluster, etc.) without changing task code.
- Resource scheduling via `task_backend_specific_kwargs={"resources": {"GPU": 1}}` routes tasks to workers advertising matching Dask resources.
- `shell=True` for executable tasks via `task_backend_specific_kwargs={"shell": True}`.
- `DaskExecutionBackend._check_resources_satisfiable()`: pre-submit check that immediately fails tasks with unsatisfiable resource constraints instead of hanging indefinitely. Function tasks set `task.exception`; executable tasks set `task.stderr` and `task.exit_code = 1`.
- Integration tests for Dask backend: end-to-end sync/async/executable task execution, cluster injection (cluster and client), and resource constraint failure behavior.

### Performance

- `DragonExecutionBackendV3._monitor_loop`: eliminated redundant `Batch.results_ddict.__contains__` check — each ready task previously required two distributed DDict network round-trips (membership test + value read); now a single `try/except KeyError` read is used, saving ~570µs per task on HPC interconnects (~5.7s for 10K tasks).
- `DragonExecutionBackendV3._monitor_loop`: monitor thread now starts lazily at first `submit_tasks()` call instead of at backend initialisation, eliminating ~1.2s of idle spinning while tasks are being built.
- `DragonExecutionBackendV3._deliver_batch`: cross-thread wakeups reduced from O(tasks) to O(sweeps) — all completions found in a single sweep are batched into one `call_soon_threadsafe` call (~0.78s saved for 10K tasks).
- `DragonExecutionBackendV3`: removed per-task `logger.debug` calls from the monitor hot path, eliminating 10K f-string allocations per run (29ms at 10K tasks, scales linearly).
- `TaskStateManager`: removed `_task_states` shadow dict — task state is now read directly from `task["state"]` (single source of truth), eliminating one dict write per task completion.
- `TaskStateManager._update_task_impl`: `_task_futures` entries are now removed via `dict.pop` on resolution, preventing unbounded memory growth at scale.
- `Session`: removed history recording (`task["history"]` timestamps) — two `time.time()` syscalls and two dict writes per task eliminated.
- `Session.get_statistics`: method removed entirely; it depended on history timestamps and iterated all tasks on every call.

Measured on 2 nodes / 128 workers (HPC), 10K function tasks:

| | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| Dragon batch only | 9.82s | 10.92s | 9.99s | 10.08s |
| RHAPSODY + Dragon batch | 10.50s | 12.17s | 9.92s | 11.18s |

RHAPSODY overhead reduced from ~7.8s (before) to ≤1.3s (after) — within normal run-to-run variance of the Dragon batch itself.

### Fixed

- `ConcurrentExecutionBackend._run_in_thread`: calling a regular (sync) function no longer raises `ValueError: a coroutine was expected`.
- `ConcurrentExecutionBackend._run_in_process`: same fix — sync functions are called directly after cloudpickle deserialization.
- `TaskStateManager._update_task_impl`: futures for failed tasks are now resolved with `set_exception` instead of `set_result`, so `await future` and `await asyncio.gather(*futures)` correctly raise on task failure.
- `TaskStateManager.get_wait_future`: same fix applied to the race-condition path (task already completed before `get_wait_future` was called).
- `Session.wait_tasks`: removed `except Exception: logger.error(...)` which was silently eating all non-timeout exceptions including programming errors.

### Changed

- `ComputeTask`: renamed `working_directory` parameter and dict key to `cwd` to distinguish task-level working directory from backend-level working directory (used by Dragon V1/V2 via `resources["working_dir"]`). Update call sites: `ComputeTask(cwd="/path")` and `task["cwd"]`.
- `DragonExecutionBackendV3`: removed the silently-ignored `working_directory` parameter from `__init__` and `create()` — it was never stored or used (unlike V1/V2 which honour `resources["working_dir"]`). Use `task_backend_specific_kwargs={"process_template": {"cwd": "..."}}` instead.
- `DaskExecutionBackend`: resource constraints are now specified exclusively via `task_backend_specific_kwargs={"resources": {...}}` (previously via `gpu=` / `cpu_threads=` on `ComputeTask`).
- `DaskExecutionBackend._build_dask_resources()`: reads from `task_backend_specific_kwargs["resources"]` instead of task-level `gpu` / `cpu_threads` fields.
- `DaskExecutionBackend._submit_executable()`: `shell=` now reads from `task_backend_specific_kwargs` for consistency.
- Optimized Dragon backend tests by using module-scoped fixtures, reusing a single backend instance per Dragon version (V1, V2, V3) across all tests instead of creating/destroying one per test.

### Breaking Changes

- **`DragonExecutionBackendV3`**: removed `num_workers`, `disable_background_batching`, and `disable_batch_submission` constructor parameters. These were incompatible with the new streaming pipeline — the Dragon batch always runs in streaming mode and worker counts are controlled via `num_nodes` / `pool_nodes`. Migration:
    - `DragonExecutionBackendV3(num_workers=16)` → `DragonExecutionBackendV3(num_nodes=4, pool_nodes=2)` (or simply omit both to let Dragon decide)
    - `DragonExecutionBackendV3(disable_batch_submission=True)` → remove (streaming is now always on)
    - `DragonExecutionBackendV3(disable_background_batching=True)` → remove
- **`ComputeTask` and `AITask`**: removed `ranks`, `memory`, `gpu`, `cpu_threads`, and `environment` parameters. These fields were never consumed by any execution backend — they were silently ignored, creating a misleading API. Resource requirements are now backend-specific and must be passed via `task_backend_specific_kwargs`. Migration:
  - `ComputeTask(executable=..., gpu=2)` → `ComputeTask(executable=..., task_backend_specific_kwargs={"resources": {"GPU": 2}})` (Dask) or `task_backend_specific_kwargs={"process_template": {"gpu": 2}}` (Dragon V3)
  - `ComputeTask(executable=..., environment={"K": "V"})` → `ComputeTask(executable=..., task_backend_specific_kwargs={"env": {"K": "V"}})` (Concurrent/Dask)
- **`ComputeTask`**: removed `cwd` and `shell` as named parameters. All execution context is now exclusively specified via `task_backend_specific_kwargs` for consistency. Migration:
  - `ComputeTask(executable=..., cwd="/path")` → `ComputeTask(executable=..., task_backend_specific_kwargs={"cwd": "/path"})`
  - `ComputeTask(executable=..., shell=True)` → `ComputeTask(executable=..., task_backend_specific_kwargs={"shell": True})`

### Fixed

- `_run_executable`: replaced `stdout=subprocess.PIPE, stderr=subprocess.PIPE` with `capture_output=True` (ruff UP022).

## [0.1.2] - 2026-02-16

### Fixed

- Fixed project URLs in `pyproject.toml` to point to the correct GitHub repository.
- Removed Python 3.8 classifier (package requires Python >= 3.9).

## [0.1.1] - 2026-02-16

### Fixed

- Fixed `ImportError` when installing `rhapsody-py` without optional backends (Dragon, Dask, RADICAL-Pilot). Optional backend imports in `backends/__init__.py` are now wrapped in `try/except`.

## [0.1.0] - 2025-02-16

### Added

- Initial PyPI release as `rhapsody-py` (import name: `rhapsody`).
- Execution backends: Concurrent (built-in), Dask, RADICAL-Pilot, Dragon (V1, V2, V3).
- Inference backend: Dragon-VLLM for LLM serving on HPC.
- Task API: `ComputeTask` for executables/functions, `AITask` for LLM inference.
- Session API for task submission, monitoring, and lifecycle management.
- Backend discovery and registry system.
- Removed ThreadPoolExecutor wait approach (`_wait_executor`), streamlining concurrency management.
- Added `disable_batch_submission` parameter to `DragonExecutionBackendV3` for configurable task submission strategy.
- Introduced polling-based batch monitoring with `_monitor_loop` and `_process_batch_results`.
