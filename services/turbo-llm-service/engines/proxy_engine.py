"""Proxy engine — forwards inference to an OpenAI-compatible upstream.

This is the optimized GPU path. The upstream is a mature, HuggingFace-loading
inference server:

    vLLM   →  `vllm/vllm-openai` image. PagedAttention, continuous batching,
              FP8 KV cache, AWQ/GPTQ/bitsandbytes weight quant. Loads any HF
              causal LM by id (`--model org/name`).
    HF TGI →  `ghcr.io/huggingface/text-generation-inference`. HF's own
              optimized server; FlashAttention, tensor parallelism, quant.

Both already speak `/v1/chat/completions`, so this engine is a thin, correct
adapter: it forwards chat requests (streaming and non-streaming) verbatim and
synthesises the bespoke turboquant `/health` fields the ai-backend expects.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import InferenceEngine

# Re-probe upstream readiness at most this often (seconds).
_READY_TTL = 10.0


class ProxyEngine(InferenceEngine):
    def __init__(
        self,
        upstream_url: str,
        kind: str = "vllm",
        configured_model: str | None = None,
        request_timeout: float = 300.0,
    ) -> None:
        self.upstream = upstream_url.rstrip("/")
        self.name = kind
        self._configured_model = configured_model
        self._model_id: str | None = configured_model
        self._ready = False
        self._last_probe = 0.0
        self._timeout = request_timeout

    async def startup(self) -> None:
        await self._probe(force=True)

    async def shutdown(self) -> None:  # nothing to release — upstream is external
        return None

    async def _probe(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_probe) < _READY_TTL:
            return
        self._last_probe = now
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # vLLM / TGI (>=1.4) both expose /v1/models.
                resp = await client.get(f"{self.upstream}/v1/models")
                if resp.status_code == 200:
                    data = resp.json().get("data") or []
                    if data:
                        self._model_id = data[0].get("id", self._configured_model)
                        self._ready = True
                        return
                # TGI fallback: /info carries the model_id.
                info = await client.get(f"{self.upstream}/info")
                if info.status_code == 200:
                    self._model_id = info.json().get("model_id", self._configured_model)
                    self._ready = True
                    return
                # Last resort: plain liveness probe.
                health = await client.get(f"{self.upstream}/health")
                self._ready = health.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            self._ready = False

    async def is_ready(self) -> bool:
        await self._probe()
        return self._ready

    async def health(self) -> dict[str, Any]:
        await self._probe()
        return {
            "model": self._model_id,
            "active_model": self._model_id if self._ready else None,
            "active_model_loaded": self._ready,
            "upstream": self.upstream,
            # The upstream owns its own (FP8/quant) KV cache; we don't measure a
            # per-request ratio here.
            "compression_ratio_last": None,
        }

    async def list_models(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.upstream}/v1/models")
                if resp.status_code == 200:
                    payload = resp.json()
                    payload["active_model"] = self._model_id
                    return payload
        except httpx.HTTPError:
            pass
        return {
            "object": "list",
            "data": [{"id": self._model_id or "unknown", "object": "model"}],
            "active_model": self._model_id,
        }

    def _prepare(self, body: dict[str, Any]) -> dict[str, Any]:
        out = dict(body)
        # Upstream requires the served model name; fill it if the caller omitted it.
        if not out.get("model"):
            out["model"] = self._model_id or self._configured_model
        return out

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._prepare(body)
        payload["stream"] = False
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self.upstream}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            return resp.json()

    async def chat_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        payload = self._prepare(body)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", f"{self.upstream}/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                # Relay the upstream SSE frames byte-for-byte (incl. [DONE]).
                async for raw in resp.aiter_raw():
                    if raw:
                        yield raw
