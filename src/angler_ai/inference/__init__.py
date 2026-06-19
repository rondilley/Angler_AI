"""Inference Layer (IL) - llama-cpp-python embedded runtime + OpenAI-compatible server.

See FR-2, design 5.2, research/06.
"""

from angler_ai.inference.runtime import InferenceRuntime, LoadedModel

__all__ = ["InferenceRuntime", "LoadedModel"]
