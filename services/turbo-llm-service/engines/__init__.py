"""Engine factory.

Selects the inference backend from environment:

    ENGINE        auto | vllm | tgi | onnx     (default: auto)
    DEVICE        auto | cuda | cpu | mps       (default: auto)
    MODEL_NAME    HuggingFace model id          (default: Qwen/Qwen2.5-3B-Instruct)
    UPSTREAM_URL  OpenAI-compatible upstream    (default: http://vllm:8000 for proxy)
    ONNX_PROVIDER ONNX Runtime execution provider (default: CPUExecutionProvider)

`auto` resolves to a proxy engine (vLLM) when a GPU is present or an
UPSTREAM_URL is configured, otherwise to the in-process ONNX CPU engine.
"""

from __future__ import annotations

import logging
import os

from .base import InferenceEngine

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_UPSTREAM = "http://vllm:8000"


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def create_engine() -> InferenceEngine:
    engine = os.getenv("ENGINE", "auto").lower()
    device = os.getenv("DEVICE", "auto").lower()
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    upstream = os.getenv("UPSTREAM_URL", "").rstrip("/")
    max_ctx = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
    trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "false").lower() in {
        "1",
        "true",
        "yes",
    }

    if engine == "auto":
        if upstream:
            engine = "vllm"
        elif device in {"cuda", "gpu"} or (device == "auto" and _gpu_available()):
            engine = "vllm"
        else:
            engine = "onnx"

    logger.info("Selected engine=%s device=%s model=%s", engine, device, model)

    if engine in {"vllm", "tgi", "proxy"}:
        from .proxy_engine import ProxyEngine

        return ProxyEngine(
            upstream_url=upstream or DEFAULT_UPSTREAM,
            kind="tgi" if engine == "tgi" else "vllm",
            configured_model=model,
        )

    if engine == "onnx":
        from .onnx_engine import OnnxEngine

        return OnnxEngine(
            model_name=model,
            max_context=max_ctx,
            provider=os.getenv("ONNX_PROVIDER", "CPUExecutionProvider"),
            trust_remote_code=trust_remote_code,
        )

    raise ValueError(f"Unknown ENGINE={engine!r} (expected auto|vllm|tgi|onnx)")
