"""ONNX engine — in-process optimized CPU/GPU inference via HuggingFace Optimum.

Replaces the original turboquant-service CPU path (which fell back to plain
``transformers.generate()``). Models load through Optimum's ONNX Runtime
integration (`optimum.onnxruntime.ORTModelForCausalLM`), which loads any HF
causal LM by id, exports to ONNX once (cached), and runs on the ONNX Runtime
graph — faster on CPU than eager PyTorch, and GPU-capable via
CUDAExecutionProvider.

This engine is **unloadable**: only the tokenizer loads at startup (cheap, so
the service reports ready and the ai-backend routes to it); the weights load
lazily on first request and are freed by the ModelGuard when idle or when
another service needs the GPU. Generation runs in a worker thread; a
process-wide lock serialises it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import AsyncIterator
from typing import Any

from . import openai_proto as proto
from .base import InferenceEngine

logger = logging.getLogger(__name__)

_GPU_PROVIDERS = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}


class OnnxEngine(InferenceEngine):
    name = "onnx"
    unloadable = True

    def __init__(
        self,
        model_name: str,
        max_context: int = 8192,
        provider: str = "CPUExecutionProvider",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_context = max_context
        self.provider = provider
        self.trust_remote_code = trust_remote_code
        self._tok: Any = None
        self._model: Any = None
        self._can_serve = False
        # Serialises generation + load/unload — one model on one device.
        self._lock = threading.Lock()

    def uses_gpu(self) -> bool:
        return self.provider in _GPU_PROVIDERS

    # ── lifecycle ─────────────────────────────────────────────────────

    async def startup(self) -> None:
        # Cheap: load only the tokenizer so we can report "ready" without
        # pinning any weights. Heavy weights load lazily via load_weights().
        await asyncio.to_thread(self._load_tokenizer)

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        self._can_serve = True
        logger.info("Tokenizer ready for %s (weights load on first use)", self.model_name)

    def _load_model_sync(self) -> None:
        from optimum.onnxruntime import ORTModelForCausalLM

        logger.info("Loading %s weights via ONNX Runtime (%s)", self.model_name, self.provider)
        # export=True converts HF weights → ONNX on first load, then caches.
        self._model = ORTModelForCausalLM.from_pretrained(
            self.model_name,
            export=True,
            provider=self.provider,
            trust_remote_code=self.trust_remote_code,
        )
        logger.info("Weights resident for %s", self.model_name)

    def _unload_model_sync(self) -> None:
        import gc

        with self._lock:
            self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Weights freed for %s", self.model_name)

    async def load_weights(self) -> None:
        await asyncio.to_thread(self._load_model_sync)

    async def unload_weights(self) -> None:
        await asyncio.to_thread(self._unload_model_sync)

    def weights_resident(self) -> bool:
        return self._model is not None

    async def shutdown(self) -> None:
        self._can_serve = False
        self._model = None
        self._tok = None

    async def is_ready(self) -> bool:
        # Ready to serve as soon as the tokenizer is loaded — weights are lazy.
        return self._can_serve

    async def health(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "active_model": self.model_name if self._can_serve else None,
            "active_model_loaded": self._can_serve,
            "weights_resident": self.weights_resident(),
            "provider": self.provider,
            "compression_ratio_last": None,
        }

    async def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": self.model_name, "object": "model"}],
            "active_model": self.model_name if self._can_serve else None,
        }

    # ── prompt + sampling helpers ─────────────────────────────────────

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        tok = self._tok
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return f"{body}\nassistant:"

    def _gen_kwargs(self, body: dict[str, Any]) -> dict[str, Any]:
        temperature = float(body.get("temperature", 0.7) or 0.0)
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(body.get("max_tokens") or 512),
            "do_sample": temperature > 0,
            "pad_token_id": self._tok.pad_token_id or self._tok.eos_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(body.get("top_p", 1.0) or 1.0)
        return kwargs

    def _ensure_model(self) -> None:
        # Defensive: the ModelGuard loads weights before calling us, but if an
        # engine is used without a guard, load on demand here.
        if self._model is None:
            self._load_model_sync()

    # ── inference ─────────────────────────────────────────────────────

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages", [])
        text, prompt_tokens, completion_tokens = await asyncio.to_thread(
            self._generate_sync, messages, body
        )
        return proto.chat_completion(
            self.model_name, text, prompt_tokens, completion_tokens
        )

    def _generate_sync(
        self, messages: list[dict[str, str]], body: dict[str, Any]
    ) -> tuple[str, int, int]:
        self._ensure_model()
        prompt = self._render_prompt(messages)
        inputs = self._tok(prompt, return_tensors="pt")
        prompt_len = int(inputs["input_ids"].shape[1])
        with self._lock:
            output = self._model.generate(**inputs, **self._gen_kwargs(body))
        generated = output[0][prompt_len:]
        text = self._tok.decode(generated, skip_special_tokens=True)
        return text, prompt_len, int(generated.shape[0])

    async def chat_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        messages = body.get("messages", [])
        completion_id = proto.new_id()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        yield proto.sse(proto.chunk(self.model_name, completion_id, role="assistant"))

        def run() -> None:
            from transformers import TextIteratorStreamer

            try:
                self._ensure_model()
                prompt = self._render_prompt(messages)
                inputs = self._tok(prompt, return_tensors="pt")
                streamer = TextIteratorStreamer(
                    self._tok, skip_prompt=True, skip_special_tokens=True
                )
                kwargs = {**inputs, **self._gen_kwargs(body), "streamer": streamer}
                with self._lock:
                    worker = threading.Thread(
                        target=self._model.generate, kwargs=kwargs, daemon=True
                    )
                    worker.start()
                    for token in streamer:
                        if token:
                            loop.call_soon_threadsafe(queue.put_nowait, token)
                    worker.join()
            except Exception as exc:
                logger.exception("ONNX stream failed")
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        future = loop.run_in_executor(None, run)
        try:
            while True:
                item = await queue.get()
                if item is None or isinstance(item, Exception):
                    break
                yield proto.sse(proto.chunk(self.model_name, completion_id, delta=item))
        finally:
            yield proto.sse(
                proto.chunk(self.model_name, completion_id, finish_reason="stop")
            )
            yield proto.sse_done()
            await future
