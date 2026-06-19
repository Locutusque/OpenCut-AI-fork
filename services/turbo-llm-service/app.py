"""turbo-llm-service — a drop-in, optimized replacement for turboquant-service.

Speaks the exact HTTP contract the OpenCut-AI ai-backend expects from the LLM
inference service on port 8430:

    GET  /health                 turboquant-shaped status (active_model_loaded, …)
    GET  /v1/models              OpenAI model listing
    POST /v1/chat/completions    OpenAI chat (streaming + non-streaming)

…so the ai-backend's `model_backend.LLMBackend` and `turboquant_service` work
unchanged — point OPENCUTAI_TURBOQUANT_SERVICE_URL at this service (or keep the
container named `turboquant-service`) and flip AI_LLM_BACKEND=auto.

Unlike the original it (a) actually supports streaming and (b) delegates
inference to an optimized HuggingFace loader — vLLM/TGI on GPU (ProxyEngine) or
Optimum/ONNX Runtime on CPU (OnnxEngine) — instead of the plain
transformers.generate() CPU fallback.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from engines import create_engine
from engines.base import InferenceEngine
from gpu_lifecycle import ModelGuard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (mirrors turboquant-service env so it's a literal drop-in)
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
KV_CACHE_BITS = int(os.getenv("KV_CACHE_BITS", "4"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
DEVICE = os.getenv("DEVICE", "auto")

_cors_raw = os.getenv("CORS_ORIGINS", "*").strip()
CORS_ORIGINS = ["*"] if _cors_raw == "*" else [o.strip() for o in _cors_raw.split(",") if o.strip()]

# Model lifecycle (lazy load + idle eviction + single-GPU lease).
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("OPENCUTAI_REDIS_URL")
MODEL_IDLE_TTL = float(os.getenv("MODEL_IDLE_TTL", "120"))
GPU_HOLDER_ID = os.getenv("GPU_HOLDER_ID", "llm")
GPU_LEASE_TTL = int(os.getenv("GPU_LEASE_TTL", "60"))

engine: InferenceEngine | None = None
guard: ModelGuard | None = None


def _detect_device() -> str:
    if DEVICE in {"cpu", "cuda", "mps"}:
        return DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _memory_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_used_mb"] = round(vm.used / 1e6)
        info["ram_total_mb"] = round(vm.total / 1e6)
        info["ram_percent"] = vm.percent
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6)
            info["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6)
    except Exception:
        pass
    return info


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, guard
    engine = create_engine()
    try:
        await engine.startup()
    except Exception:
        # Don't crash the container — /health will report not-ready and the
        # ai-backend falls back to Ollama until the model finishes loading.
        logger.exception("Engine startup failed; service will report not-ready")

    # Unloadable engines (ONNX) get a guard: weights load lazily on first
    # request, unload when idle, and yield the GPU to other services on demand.
    if engine.unloadable:
        guard = await ModelGuard.create(
            GPU_HOLDER_ID,
            redis_url=REDIS_URL,
            lease_ttl=GPU_LEASE_TTL,
            load=engine.load_weights,
            unload=engine.unload_weights,
            is_loaded=engine.weights_resident,
            uses_gpu=engine.uses_gpu,
            idle_ttl=MODEL_IDLE_TTL,
        )
        await guard.start()
        logger.info(
            "ModelGuard active (holder=%s, idle_ttl=%ss, gpu=%s)",
            GPU_HOLDER_ID, MODEL_IDLE_TTL, engine.uses_gpu(),
        )

    yield

    if guard:
        await guard.stop()
    if engine:
        await engine.shutdown()


app = FastAPI(title="turbo-llm-service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models (permissive — extra OpenAI fields are forwarded)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 2048
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    stream: bool | None = False


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    eh = await engine.health() if engine else {}
    compute_mode = _detect_device()
    gpu = compute_mode == "cuda"
    ready = bool(eh.get("active_model_loaded"))
    return {
        "status": "ok",
        "engine": getattr(engine, "name", "unknown"),
        # Both keys are populated on purpose: model_backend reads `active_model`
        # while turboquant_service.get_service_status reads `model`.
        "model": eh.get("model") or MODEL_NAME,
        "active_model": eh.get("active_model"),
        "active_model_loaded": ready,
        "kv_cache_bits": KV_CACHE_BITS,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "gpu_available": gpu,
        "backend": "gpu" if gpu else "cpu",
        "compute_mode": compute_mode,
        "device": compute_mode,
        # Drop-in parity: signals an accelerated engine is serving requests.
        "turboquant_available": True,
        "turboquant_engine_available": ready,
        "compression_ratio_last": eh.get("compression_ratio_last"),
        "memory_usage": _memory_info(),
        # Lifecycle: are weights currently resident, and the guard's view.
        "weights_resident": eh.get("weights_resident", ready),
        "lifecycle": guard.status() if guard else None,
        **{k: eh[k] for k in ("upstream", "provider") if k in eh},
    }


# ---------------------------------------------------------------------------
# Model management — the engine owns the model lifecycle (loaded at startup),
# so these expose status and degrade gracefully instead of 404-ing the UI.
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    if not engine:
        raise HTTPException(503, "Engine not initialised")
    return await engine.list_models()


@app.get("/v1/models/catalog")
async def model_catalog() -> dict[str, Any]:
    models = await engine.list_models() if engine else {"data": []}
    return {
        "catalog": models.get("data", []),
        "device": _detect_device(),
        "gpu_available": _detect_device() == "cuda",
        "memory": _memory_info(),
        "note": "Model is configured via MODEL_NAME and loaded at startup.",
    }


@app.get("/v1/models/downloaded")
async def downloaded_models() -> dict[str, Any]:
    models = await engine.list_models() if engine else {"data": []}
    return {"models": models.get("data", []), "total_size_mb": 0}


@app.post("/v1/models/load")
async def load_model() -> dict[str, Any]:
    # Switching models at runtime means restarting the upstream (vLLM) or
    # re-exporting (ONNX); not supported live. Report current state instead.
    ready = await engine.is_ready() if engine else False
    return {
        "status": "managed_by_engine",
        "active_model_loaded": ready,
        "detail": "This service loads MODEL_NAME at startup. Set MODEL_NAME and restart to switch.",
    }


@app.post("/v1/models/unload")
async def unload_model() -> dict[str, Any]:
    if guard:
        await guard.unload_now()
        return {
            "status": "unloaded",
            "detail": "Weights freed; they reload automatically on the next request.",
        }
    return {"status": "not_supported", "detail": "Model lifecycle is managed by the engine."}


# ---------------------------------------------------------------------------
# Inference — OpenAI-compatible
# ---------------------------------------------------------------------------


async def _guarded_stream(body: dict[str, Any]):
    """Hold the GPU lease + keep weights resident for the whole stream."""
    if guard:
        async with guard.session():
            async for chunk in engine.chat_stream(body):
                yield chunk
    else:
        async for chunk in engine.chat_stream(body):
            yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not engine or not await engine.is_ready():
        raise HTTPException(503, "No model loaded. The engine is not ready yet.")

    body = request.model_dump(exclude_none=True)

    if request.stream:
        return StreamingResponse(
            _guarded_stream(body), media_type="text/event-stream"
        )

    try:
        if guard:
            # Lazy-loads weights, holds the single-GPU lease for the request.
            async with guard.session():
                return await engine.chat(body)
        return await engine.chat(body)
    except TimeoutError as exc:
        raise HTTPException(503, str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat completion failed")
        raise HTTPException(500, "Generation failed")


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "turbo-llm-service", "docs": "/docs", "health": "/health"}
