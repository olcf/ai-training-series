"""Unit tests for DragonVllmInferenceBackend (rhapsody.backends.ai.vllm).

These tests cover the config-building/translation layer and the response
unwrapping logic added by the migration to dragon.ai.inference. They do not
require a running Dragon allocation or GPU — Inference/DragonQueue internals
are never invoked; only the constructor and a couple of pure helper functions
are exercised.

Run with:
    dragon python -m pytest tests/unit/test_backend_ai_vllm.py -v
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

# Skip the entire module when dragon.ai.inference (dragonhpc[ai]) is not installed.
# Bare `dragon` can import fine while `dragon.ai.inference` still fails — its
# dependency closure (vllm, torch, etc.) is only pulled in by the [ai] extra,
# which isn't available on every Python version (e.g. 3.9/3.10 in some envs).
pytest.importorskip(
    "dragon.ai.inference",
    reason="dragon.ai.inference (dragonhpc[ai]) is required for DragonVllmInferenceBackend tests",
)

from rhapsody.backends.ai.config import BatchingConfig
from rhapsody.backends.ai.config import DynamicWorkerConfig
from rhapsody.backends.ai.config import GuardrailsConfig
from rhapsody.backends.ai.config import HardwareConfig
from rhapsody.backends.ai.config import ModelConfig
from rhapsody.backends.ai.vllm import DragonVllmInferenceBackend
from rhapsody.backends.ai.vllm import _unwrap_assistant_response


def _model_config(**overrides):
    kwargs = {"model_name": "test-model", "hf_token": "hf-test-token", "tp_size": 1}
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


# ============================================================================
# Config translation
# ============================================================================


def test_build_inference_config_passes_objects_through_verbatim():
    """_build_inference_config() forwards the stored config objects unchanged."""
    model = _model_config()
    hardware = HardwareConfig(num_nodes=2, num_gpus=4)
    batching = BatchingConfig(enabled=True, batch_type="pre-batch")
    guardrails = GuardrailsConfig(enabled=True)
    dynamic_worker = DynamicWorkerConfig(enabled=True)

    backend = DragonVllmInferenceBackend(
        model=model,
        hardware=hardware,
        batching=batching,
        guardrails=guardrails,
        dynamic_worker=dynamic_worker,
        use_service=False,
    )

    config = backend._build_inference_config()

    assert config.model is model
    assert config.hardware is hardware
    assert config.batching is batching
    assert config.guardrails is guardrails
    assert config.dynamic_worker is dynamic_worker


def test_defaults_match_current_rhapsody_behavior():
    """Omitting hardware/batching/guardrails/dynamic_worker preserves today's forced defaults:

    pre-batch batching enabled, guardrails off, dynamic workers off.
    """
    backend = DragonVllmInferenceBackend(model=_model_config(), use_service=False)

    assert backend.batching.enabled is True
    assert backend.batching.batch_type == "pre-batch"
    assert backend.guardrails.enabled is False
    assert backend.dynamic_worker.enabled is False
    assert isinstance(backend.hardware, HardwareConfig)


def test_model_name_convenience_field():
    """self.model_name mirrors model.model_name for the HTTP handlers."""
    backend = DragonVllmInferenceBackend(
        model=_model_config(model_name="my-model"), use_service=False
    )

    assert backend.model_name == "my-model"


def test_dynamic_batch_type_rejected():
    """batch_type='dynamic' is out of scope for this backend and must fail fast."""
    with pytest.raises(NotImplementedError):
        DragonVllmInferenceBackend(
            model=_model_config(),
            batching=BatchingConfig(enabled=True, batch_type="dynamic"),
            use_service=False,
        )


@pytest.mark.asyncio
async def test_await_backend_delegates_to_initialize():
    """Await backend (and thus await DragonVllmInferenceBackend(...)) delegates to initialize() and
    returns the backend itself, matching the other execution backends' __await__ contract."""
    backend = DragonVllmInferenceBackend(model=_model_config(), use_service=False)
    backend.initialize = AsyncMock(return_value=backend)

    result = await backend

    backend.initialize.assert_called_once()
    assert result is backend


# ============================================================================
# Response unwrapping
# ============================================================================


def test_unwrap_assistant_response_extracts_assistant_key():
    """Inference.query() dict responses are unwrapped to their 'assistant' text."""
    response = {
        "assistant": "hello world",
        "model_inference_latency": 0.42,
        "requests_per_second": 3.1,
    }

    assert _unwrap_assistant_response(response) == "hello world"


def test_unwrap_assistant_response_dict_without_assistant_key_passes_through():
    """A dict missing the 'assistant' key is returned as-is (defensive fallback)."""
    response = {"unexpected": "shape"}

    assert _unwrap_assistant_response(response) == response


def test_unwrap_assistant_response_string_passthrough():
    """Plain strings (e.g. the batch processor's own timeout/error sentinels) pass through."""
    assert _unwrap_assistant_response("ERROR: timeout") == "ERROR: timeout"
    assert _unwrap_assistant_response("plain text") == "plain text"


# ============================================================================
# HTTP handler: /v1/chat/completions response normalization
# ============================================================================


@pytest.mark.asyncio
async def test_chat_completions_normalizes_assistant_dict_response():
    """_handle_chat_completions extracts 'assistant' from a dict response.

    Regression test for the response-shape change: Inference.query() now returns a
    metrics dict instead of a bare string, and the OpenAI-compatible endpoint must
    still surface the generated text, not the whole dict.
    """
    backend = DragonVllmInferenceBackend(model=_model_config(), use_service=True)
    backend.is_initialized = True
    backend.generate = AsyncMock(
        return_value=[{"assistant": "generated text", "model_inference_latency": 0.1}]
    )

    request = MagicMock()
    request.json = AsyncMock(
        return_value={"messages": [{"role": "user", "content": "hi"}], "model": "test-model"}
    )

    response = await backend._handle_chat_completions(request)

    import json

    body = json.loads(response.body)
    assert body["choices"][0]["message"]["content"] == "generated text"
