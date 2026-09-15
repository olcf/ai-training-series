"""Dragon VLLM Inference Service with Request Batching Accumulates incoming requests and submits as
batches to pipeline for efficiency."""

import asyncio
import logging
import socket
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional

from aiohttp import web

try:
    import dragon
    from dragon.native.queue import Queue as DragonQueue
except ImportError:
    dragon = None
    DragonQueue = None

from rhapsody.backends.ai.config import BatchingConfig
from rhapsody.backends.ai.config import DynamicWorkerConfig
from rhapsody.backends.ai.config import GuardrailsConfig
from rhapsody.backends.ai.config import HardwareConfig
from rhapsody.backends.ai.config import Inference
from rhapsody.backends.ai.config import InferenceConfig
from rhapsody.backends.ai.config import ModelConfig
from rhapsody.backends.base import BaseBackend
from rhapsody.backends.constants import StateMapper

logger = logging.getLogger(__name__)

# Global registry for queues (to avoid serialization)
_QUEUE_REGISTRY = {}


def _unwrap_assistant_response(response: Any) -> Any:
    """Extract generated text from an ``Inference.query()`` response.

    ``Inference.query()`` responses are dicts keyed by ``"assistant"`` plus
    latency/throughput metrics; the batch processor's own timeout/error
    sentinel strings pass through unchanged.
    """
    if isinstance(response, dict):
        return response.get("assistant", response)
    return response


@dataclass
class PendingRequest:
    """Represents a pending inference request."""

    prompts: list[str]
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)
    task_uid: Optional[str] = None  # UID of the AITask if applicable


class DragonVllmInferenceBackend(BaseBackend):
    """Server-side batching VLLM inference backend.

    Key Features:
    - Accumulates individual requests into batches
    - Submits batches to pipeline (efficient for VLLM)
    - Async polling (non-blocking)
    - Serialization-safe (lazy queue initialization)

    Batching Strategy:
    - Collects requests for max_batch_wait_ms milliseconds
    - OR until batch reaches max_batch_size requests
    - Then submits entire batch to pipeline
    - Distributes responses back to individual waiting requests

    This is MUCH more efficient than sending 1000 individual requests to pipeline!
    """

    def __init__(
        self,
        model: ModelConfig,
        hardware: Optional[HardwareConfig] = None,
        batching: Optional[BatchingConfig] = None,
        guardrails: Optional[GuardrailsConfig] = None,
        dynamic_worker: Optional[DynamicWorkerConfig] = None,
        port: int = 8000,
        use_service: bool = True,
        max_batch_size: int = 1024,
        max_batch_wait_ms: int = 500,
        name: Optional[str] = "vllm",
    ):
        if dragon is None:
            raise ImportError("Dragon is required for DragonVllmInferenceBackend.")

        if Inference is None:
            raise ImportError("dragon.ai.inference is required for DragonVllmInferenceBackend.")

        super().__init__(name=name)

        # Register states in StateMapper
        StateMapper.register_backend_tasks_states_with_defaults(self.name)

        # Dragon AI config objects, re-imported as-is (see rhapsody.backends.ai.config)
        self.model = model
        self.model_name = model.model_name  # convenience field for HTTP handlers
        self.hardware = hardware or HardwareConfig()
        self.batching = batching or BatchingConfig(enabled=True, batch_type="pre-batch")
        self.guardrails = guardrails or GuardrailsConfig(enabled=False)
        self.dynamic_worker = dynamic_worker or DynamicWorkerConfig(enabled=False)

        if self.batching.batch_type != "pre-batch":
            raise NotImplementedError(
                f"DragonVllmInferenceBackend only supports batching.batch_type='pre-batch' "
                f"today, got {self.batching.batch_type!r}. 'dynamic' batch_type requires "
                f"per-prompt query submission and is not yet supported by this backend."
            )

        self.port = port
        self.use_service = use_service

        # Batching parameters (RHAPSODY's own client-side accumulator, distinct from
        # self.batching.max_batch_size which controls vLLM's internal max_num_seqs)
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = max_batch_wait_ms

        # Generate unique ID for this instance
        self._instance_id = uuid.uuid4().hex

        # These will be lazily initialized
        self.inference_pipeline = None
        self.is_initialized = False
        self.hostname = socket.gethostname()

        # HTTP components (only used if use_service=True)
        self.app = None
        self.runner = None
        self.site = None

        # Request batching
        self._pending_requests: list[PendingRequest] = []
        self._batch_lock = asyncio.Lock()
        self._batch_processor_task = None
        self._new_request_event = asyncio.Event()  # Signal when requests arrive

        # Callbacks and state tracking
        self._callback_func = None
        self._tasks_in_flight = {}  # UID -> AITask

    def __await__(self):
        """Enable `backend = await DragonVllmInferenceBackend(...)`, consistent with the other
        execution backends.

        Delegates to the same initialize() used by __aenter__ and the explicit two-step construction
        pattern — no behavior change for existing callers.
        """
        return self.initialize().__await__()

    def _build_inference_config(self) -> InferenceConfig:
        """Assemble the Dragon AI InferenceConfig from the stored config objects."""
        return InferenceConfig(
            model=self.model,
            hardware=self.hardware,
            batching=self.batching,
            guardrails=self.guardrails,
            dynamic_worker=self.dynamic_worker,
        )

    def _get_or_create_queues(self):
        """Get queues from registry or create new ones."""
        if self._instance_id not in _QUEUE_REGISTRY:
            _QUEUE_REGISTRY[self._instance_id] = {
                "input": DragonQueue(),
                "response": DragonQueue(),
            }
        return _QUEUE_REGISTRY[self._instance_id]

    def _get_input_queue(self):
        """Get input queue (lazy)"""
        return self._get_or_create_queues()["input"]

    def _get_response_queue(self):
        """Get response queue (lazy)"""
        return self._get_or_create_queues()["response"]

    def register_callback(self, func: Callable[[dict, str], None]) -> None:
        """Register a callback for task state changes."""
        self._callback_func = func

    async def submit_tasks(self, tasks: list[dict]) -> None:
        """Submit AITask objects for inference.

        Each task is decomposed into prompts and added to the batching queue.
        """
        if not self.is_initialized:
            await self.initialize()

        for task in tasks:
            uid = task["uid"]
            self._tasks_in_flight[uid] = task

            # Report RUNNING state
            if self._callback_func:
                self._callback_func(task, "RUNNING")

            # Extract prompts (can be single str or list[str])
            prompt = task.get("prompt")
            prompts = [prompt] if isinstance(prompt, str) else prompt

            # Create a future to track this specific task's completion in batch processor
            completion_future = asyncio.Future()
            request = PendingRequest(prompts=prompts, future=completion_future, task_uid=uid)

            # Add to pending queue
            async with self._batch_lock:
                self._pending_requests.append(request)

            # Signal processor
            self._new_request_event.set()

            # We don't await here; the batch processor will update task state and results

    async def initialize(self):
        """Initialize the VLLM pipeline and optionally HTTP server Blocks until service is ready."""
        if self.is_initialized:
            logger.warning("Service already initialized")
            return self

        mode = "service" if self.use_service else "engine"
        logger.info(f"Initializing VLLM {mode} on {self.hostname}...")
        logger.info(
            f"Batching: max_size={self.max_batch_size}, max_wait={self.max_batch_wait_ms}ms"
        )

        config = self._build_inference_config()

        # Get queues from registry (lazy initialization)
        input_queue = self._get_input_queue()

        # Initialize pipeline
        logger.info("Initializing VLLM pipeline...")

        def _init_pipeline():
            self.inference_pipeline = Inference(config, input_queue)
            self.inference_pipeline.initialize()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _init_pipeline)

        logger.info("VLLM pipeline initialized!")

        # Start batch processor
        self._batch_processor_task = asyncio.create_task(self._batch_processor())

        # Start HTTP server only if use_service=True
        if self.use_service:
            await self._start_http_server()
            logger.info(f"Service ready at http://{self.hostname}:{self.port}")
            logger.info(f"GET  http://{self.hostname}:{self.port}/health")
            logger.info(f"POST http://{self.hostname}:{self.port}/generate")
        else:
            logger.info("Engine ready (no HTTP server)")
            logger.info("Use engine.generate() for direct pipeline access")

        self.is_initialized = True
        return self

    async def _batch_processor(self):
        """Background task that processes batches of requests.

        Strategy:
        1. Wait for at least one request to arrive
        2. Accumulate more requests for up to max_batch_wait_ms
        3. OR process immediately if batch reaches max_batch_size
        4. Submit combined batch to pipeline
        5. Distribute responses back to individual futures
        """
        response_queue = self._get_response_queue()

        while True:
            try:
                # Wait for at least one request to arrive
                await self._new_request_event.wait()
                self._new_request_event.clear()

                # Now accumulate requests for a short time
                accumulation_start = asyncio.get_event_loop().time()
                accumulation_deadline = accumulation_start + (self.max_batch_wait_ms / 1000.0)

                while asyncio.get_event_loop().time() < accumulation_deadline:
                    async with self._batch_lock:
                        # If we hit max batch size, stop accumulating
                        if len(self._pending_requests) >= self.max_batch_size:
                            break
                    # Short sleep to allow more requests to arrive
                    await asyncio.sleep(0.001)

                # Collect pending requests
                async with self._batch_lock:
                    if not self._pending_requests:
                        continue  # Race condition: all requests cancelled

                    # Take up to max_batch_size requests
                    batch = self._pending_requests[: self.max_batch_size]
                    self._pending_requests = self._pending_requests[self.max_batch_size :]

                    # If there are still more pending, signal to process next batch
                    if self._pending_requests:
                        self._new_request_event.set()

                # Combine all prompts
                all_prompts = []
                request_sizes = []
                for req in batch:
                    all_prompts.extend(req.prompts)
                    request_sizes.append(len(req.prompts))

                if not all_prompts:
                    continue

                logger.info(
                    f"Processing batch: {len(batch)} requests, {len(all_prompts)} total prompts"
                )

                # Submit to pipeline
                logger.debug("Submitting batch to Inference pipeline...")

                self.inference_pipeline.query((all_prompts, response_queue))

                logger.debug("Batch submitted, waiting for responses...")
                # Collect responses
                all_results = []
                deadline = asyncio.get_event_loop().time() + 300  # 5min timeout

                for i in range(len(all_prompts)):
                    logger.debug(f"Waiting for response {i + 1}/{len(all_prompts)}...")
                    response_received = False
                    poll_count = 0

                    while True:
                        poll_count += 1
                        if not response_queue.empty():
                            response = _unwrap_assistant_response(response_queue.get_nowait())
                            all_results.append(response)
                            logger.debug(
                                f"Received response {i + 1}/{len(all_prompts)} after {poll_count} polls"
                            )
                            response_received = True
                            break

                        if asyncio.get_event_loop().time() > deadline:
                            error_msg = f"Timeout waiting for response {i + 1}/{len(all_prompts)} after 300s ({poll_count} polls)"
                            logger.error(error_msg)
                            all_results.append(f"ERROR: {error_msg}")
                            break

                        # Log every 10 seconds
                        if poll_count % 10000 == 0:
                            elapsed = asyncio.get_event_loop().time() - (deadline - 300)
                            logger.debug(
                                f"Still waiting for response {i + 1}/{len(all_prompts)} after {elapsed:.1f}s ({poll_count} polls)"
                            )

                        await asyncio.sleep(0.001)  # Async polling

                    if not response_received and asyncio.get_event_loop().time() > deadline:
                        logger.error(
                            f"Batch processing timed out at response {i + 1}/{len(all_prompts)}"
                        )
                        # Fill remaining responses with errors
                        for j in range(i + 1, len(all_prompts)):
                            all_results.append(
                                f"ERROR: Batch timeout, response {j + 1} not received"
                            )
                        break

                # Distribute results back to individual requests
                offset = 0
                for req, size in zip(batch, request_sizes):
                    req_results = all_results[offset : offset + size]

                    # Store results in future (for direct generate() calls)
                    if not req.future.done():
                        req.future.set_result(req_results)

                    # If this was part of an AITask, update task state and trigger callback
                    if req.task_uid and req.task_uid in self._tasks_in_flight:
                        task = self._tasks_in_flight.pop(req.task_uid)
                        task["response"] = req_results  # AITask uses .response for model output
                        if self._callback_func:
                            self._callback_func(task, "DONE")

                    offset += size

                logger.info(f"Batch complete: {len(all_prompts)} prompts processed")

            except Exception as e:
                logger.error(f"Batch processor error: {e}", exc_info=True)

                # Fail requests in the current batch being processed
                if "batch" in locals():
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(e)

                        if req.task_uid and req.task_uid in self._tasks_in_flight:
                            task = self._tasks_in_flight.pop(req.task_uid)
                            task["exception"] = str(e)
                            if self._callback_func:
                                self._callback_func(task, "FAILED")

                # Fail pending requests still in queue
                async with self._batch_lock:
                    for req in self._pending_requests:
                        if not req.future.done():
                            req.future.set_exception(e)

                        if req.task_uid and req.task_uid in self._tasks_in_flight:
                            task = self._tasks_in_flight.pop(req.task_uid)
                            task["exception"] = str(e)
                            if self._callback_func:
                                self._callback_func(task, "FAILED")

                    self._pending_requests.clear()

    async def _start_http_server(self):
        """Start the HTTP server (only called if use_service=True)"""
        self.app = web.Application()

        # Add routes
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_post("/generate", self._handle_generate)

        # Ad OpenAI Client endpoints
        self.app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        self.app.router.add_get("/v1/models", self._handle_models)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.hostname, self.port)
        await self.site.start()

        logger.info(f"HTTP server started on {self.hostname}:{self.port}")

        logger.info("Available endpoints:")
        logger.info("GET  /health")
        logger.info("POST /generate")
        logger.info("POST /v1/chat/completions (OpenAI compatible)")
        logger.info("GET  /v1/models (OpenAI compatible)")

    async def generate(self, prompts: list[str], timeout: int = 300) -> list[str]:
        """Generate responses for given prompts.

        Queues request for batching instead of immediate submission.
        """
        if not self.is_initialized:
            raise RuntimeError("Not initialized. Call await initialize() first.")

        # Create future for this request
        future = asyncio.Future()
        request = PendingRequest(prompts=prompts, future=future)

        # Add to pending queue
        async with self._batch_lock:
            self._pending_requests.append(request)
            logger.debug(
                f"Queued request with {len(prompts)} prompts. Queue size: {len(self._pending_requests)}"
            )

        # Signal batch processor that new request arrived
        self._new_request_event.set()

        # Wait for result
        try:
            results = await asyncio.wait_for(future, timeout=timeout)
            return results
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out after {timeout}s")

    async def shutdown(self):
        """Shutdown the service/engine."""
        if not self.is_initialized:
            return

        mode = "service" if self.use_service else "engine"
        logger.info(f"Shutting down {mode}...")

        # Cancel batch processor
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        # Stop HTTP server (only if use_service=True)
        if self.use_service:
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()

        # Shutdown pipeline. Inference.destroy() already closes the input queue
        # internally, so only the response queue needs cleaning up below.
        if self.inference_pipeline:
            self.inference_pipeline.destroy()

        # Clean up queues from registry
        if self._instance_id in _QUEUE_REGISTRY:
            queues = _QUEUE_REGISTRY[self._instance_id]
            queues["response"].close()
            del _QUEUE_REGISTRY[self._instance_id]

        self.is_initialized = False
        logger.info(f"{mode.capitalize()} shutdown complete")

    async def _handle_chat_completions(self, request):
        """OpenAI-compatible chat completions endpoint.

        Converts OpenAI format to VLLM /generate format.
        """
        try:
            data = await request.json()

            # Extract OpenAI-style parameters
            messages = data.get("messages", [])

            # FIXME pass to self.generate() if Inference supports these params
            data.get("max_tokens", 1000)
            data.get("temperature", 0.7)
            model = data.get("model", self.model_name)

            # Convert messages to single prompt
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            # Combine into single prompt
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

            # Call internal generate method
            time.time()
            results = await self.generate([prompt], timeout=300)
            time.time()

            response_text = results[0] if results else ""

            # FIX: Normalize response_text to handle dict or string
            def normalize_response(text):
                """Handle both string and dict responses from vLLM."""
                if isinstance(text, dict):
                    # Try common dict keys ("assistant" is what Inference.query() uses)
                    return text.get(
                        "assistant",
                        text.get(
                            "text", text.get("content", text.get("generated_text", str(text)))
                        ),
                    )
                return str(text) if text else ""

            # Normalize the response
            response_text_normalized = normalize_response(response_text)

            # Return in OpenAI format to support integration with agentic frameworks
            return web.json_response(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text_normalized,  # Use normalized version
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(
                            response_text_normalized.split()
                        ),  # Use normalized version
                        "total_tokens": len(prompt.split())
                        + len(response_text_normalized.split()),  # Use normalized version
                    },
                }
            )

        except Exception as e:
            logger.error(f"Chat completions error: {e}", exc_info=True)
            return web.json_response(
                {"error": {"message": str(e), "type": "internal_error", "code": "internal_error"}},
                status=500,
            )

    async def _handle_models(self, request):
        """OpenAI-compatible models list endpoint."""
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "organization-owner",
                        "permission": [],
                        "root": self.model_name,
                        "parent": None,
                    }
                ],
            }
        )

    # HTTP Handlers (only used if use_service=True)
    async def _handle_health(self, request):
        """Health check endpoint."""
        async with self._batch_lock:
            queue_size = len(self._pending_requests)

        return web.json_response(
            {
                "status": "healthy",
                "hostname": self.hostname,
                "initialized": self.is_initialized,
                "endpoint": f"http://{self.hostname}:{self.port}",
                "mode": "service",
                "batch_config": {
                    "max_batch_size": self.max_batch_size,
                    "max_batch_wait_ms": self.max_batch_wait_ms,
                    "pending_requests": queue_size,
                },
            }
        )

    async def _handle_generate(self, request):
        """Generate responses via HTTP."""
        try:
            data = await request.json()
            prompts = data.get("prompts", [])
            timeout = data.get("timeout", 300)

            if not prompts:
                return web.json_response(
                    {"status": "error", "message": "No prompts provided"}, status=400
                )

            start_time = time.time()
            results = await self.generate(prompts, timeout)
            end_time = time.time()

            return web.json_response(
                {
                    "status": "success",
                    "hostname": self.hostname,
                    "results": results,
                    "num_prompts": len(prompts),
                    "total_time": end_time - start_time,
                    "avg_time_per_prompt": (end_time - start_time) / len(prompts),
                }
            )

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return web.json_response(
                {"status": "error", "message": str(e), "hostname": self.hostname}, status=500
            )

    def get_endpoint(self):
        """Get the service endpoint URL Returns None if use_service=False."""
        if not self.use_service:
            logger.warning("Engine mode: No HTTP endpoint available")
            return None
        return f"http://{self.hostname}:{self.port}"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    # Serialization support
    def __getstate__(self):
        """Control pickling - exclude non-serializable objects"""
        state = self.__dict__.copy()
        # Remove non-serializable items
        state["inference_pipeline"] = None
        state["app"] = None
        state["runner"] = None
        state["site"] = None
        state["_batch_processor_task"] = None
        state["_pending_requests"] = []
        state["_batch_lock"] = None
        state["_new_request_event"] = None
        return state

    def __setstate__(self, state):
        """Control unpickling - restore state"""
        self.__dict__.update(state)
        # These will be lazily re-initialized if needed
        self._batch_lock = asyncio.Lock()
        self._new_request_event = asyncio.Event()

    # BaseBackend abstract methods
    def state(self) -> str:
        return "RUNNING" if self.is_initialized else "INITIALIZED"

    def task_state_cb(self, task: dict, state: str) -> None:
        if self._callback_func:
            self._callback_func(task, state)

    def get_task_states_map(self) -> Any:
        # Return a mapper that includes terminal states
        return StateMapper("vllm")

    def build_task(self, task: dict) -> None:
        pass

    async def cancel_task(self, uid: str) -> bool:
        # vLLM/Inference doesn't easily support cancellation of single queries
        return False
