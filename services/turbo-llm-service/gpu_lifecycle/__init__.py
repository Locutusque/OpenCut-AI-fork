"""Single-GPU model lifecycle toolkit for the OpenCut-AI services.

Lazy load + idle eviction + a Redis-backed cross-service GPU lease, so a stack
of model-holding services (LLM, Whisper, TTS, image, speaker) keeps only what's
actually in use resident on one GPU.

Source of truth: opencut-ai-fork/shared/gpu_lifecycle. Each service vendors a
copy (Docker build contexts are per-service); keep copies in sync.
"""

from .arbiter import Arbiter, NullArbiter, RedisArbiter, make_arbiter
from .guard import ModelGuard

__all__ = ["Arbiter", "NullArbiter", "RedisArbiter", "make_arbiter", "ModelGuard"]
