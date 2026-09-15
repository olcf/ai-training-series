"""Execution backends for Rhapsody.

This module contains concrete implementations of inference backends for different computing
environments.
"""

from __future__ import annotations

__all__ = []


try:
    from .vllm import DragonVllmInferenceBackend  # noqa: F401

    __all__.append("DragonVllmInferenceBackend")
except ImportError:
    pass
