"""Engine interface.

`app.py` owns the turboquant-compatible HTTP surface and is engine-agnostic;
every backend implements this interface. There are two production engines:

    ProxyEngine  — forwards to an OpenAI-compatible upstream (vLLM or HF TGI).
                   This is the "optimized GPU loader" path: vLLM/TGI load HF
                   models directly and bring PagedAttention / continuous
                   batching / fused kernels.
    OnnxEngine   — in-process Optimum + ONNX Runtime. This is the optimized
                   CPU path that replaces the original service's plain
                   transformers.generate() fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class InferenceEngine(ABC):
    """Common interface for all inference backends."""

    #: short identifier surfaced in /health ("vllm", "tgi", "onnx")
    name: str = "base"

    #: True if weights can be freed/reloaded at runtime (in-process engines).
    #: Proxy engines front an external server that owns its own VRAM → False.
    unloadable: bool = False

    def uses_gpu(self) -> bool:
        """Whether this engine occupies discrete GPU memory (for the lease)."""
        return False

    # ── weight residency (only meaningful when ``unloadable``) ─────────

    def weights_resident(self) -> bool:
        """True when model weights currently occupy memory."""
        return True

    async def load_weights(self) -> None:
        """Load weights into memory. No-op unless ``unloadable``."""

    async def unload_weights(self) -> None:
        """Free weights from memory. No-op unless ``unloadable``."""

    @abstractmethod
    async def startup(self) -> None:
        """Prepare to serve (cheap). Heavy weights load lazily on first use."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources."""

    @abstractmethod
    async def is_ready(self) -> bool:
        """True when a model is loaded and able to serve requests."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Engine-specific health fields.

        `app.py` merges these into the turboquant-compatible /health payload.
        Should include at least ``active_model`` (str | None), ``model``
        (str | None) and ``active_model_loaded`` (bool).
        """

    @abstractmethod
    async def list_models(self) -> dict[str, Any]:
        """OpenAI-style model listing for GET /v1/models."""

    @abstractmethod
    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat completion. Returns an OpenAI chat.completion."""

    @abstractmethod
    def chat_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream a chat completion as raw Server-Sent-Event bytes.

        Yields ``data: {chunk}\\n\\n`` frames terminated by
        ``data: [DONE]\\n\\n`` — byte-for-byte compatible with the OpenAI
        streaming format the ai-backend client already parses.
        """
