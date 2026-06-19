"""Helpers for building OpenAI-compatible response payloads.

Used by engines that generate locally (OnnxEngine). Proxy engines pass the
upstream's already-OpenAI payloads through untouched.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def new_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def chat_completion(
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a non-streaming chat.completion object."""
    now = int(time.time())
    return {
        "id": new_id(),
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chunk(
    model: str,
    completion_id: str,
    *,
    role: str | None = None,
    delta: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build a single streaming chat.completion.chunk object."""
    d: dict[str, Any] = {}
    if role is not None:
        d["role"] = role
    if delta is not None:
        d["content"] = delta
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": d, "finish_reason": finish_reason}],
    }


def sse(payload: dict[str, Any]) -> bytes:
    """Encode a payload as a Server-Sent-Event ``data:`` frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"
