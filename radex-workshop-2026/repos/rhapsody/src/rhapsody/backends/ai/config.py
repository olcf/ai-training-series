"""Re-export of dragon.ai.inference types used by RHAPSODY's AI backends.

Single import boundary: RHAPSODY code depends on rhapsody.backends.ai.config,
never on dragon.ai.inference directly. The optional-dependency guard lives
here once instead of being duplicated per consumer.
"""

from __future__ import annotations

try:
    from dragon.ai.inference import BatchingConfig
    from dragon.ai.inference import DynamicWorkerConfig
    from dragon.ai.inference import GuardrailsConfig
    from dragon.ai.inference import HardwareConfig
    from dragon.ai.inference import Inference
    from dragon.ai.inference import InferenceConfig
    from dragon.ai.inference import ModelConfig
except ImportError:
    BatchingConfig = None
    DynamicWorkerConfig = None
    GuardrailsConfig = None
    HardwareConfig = None
    Inference = None
    InferenceConfig = None
    ModelConfig = None

__all__ = [
    "BatchingConfig",
    "DynamicWorkerConfig",
    "GuardrailsConfig",
    "HardwareConfig",
    "Inference",
    "InferenceConfig",
    "ModelConfig",
]
