"""ModelGuard — lazy load, idle eviction, and single-GPU arbitration.

Wrap any model holder that exposes load / unload / is-loaded callables. The
guard:

* loads the model on first use (lazy — nothing sits in VRAM until needed),
* keeps it warm between requests, then unloads it after ``idle_ttl`` seconds,
* (for GPU models) holds a cross-service :class:`Arbiter` lease while resident
  and yields it — unloading — when another service requests the GPU.

Usage::

    guard = await ModelGuard.create(
        "image", redis_url=os.getenv("REDIS_URL"),
        load=svc.load_async, unload=svc.unload_async,
        is_loaded=lambda: svc.loaded, uses_gpu=lambda: svc.device == "cuda",
    )
    await guard.start()
    ...
    async with guard.session():
        result = svc.run(...)
    ...
    await guard.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from .arbiter import Arbiter, NullArbiter, make_arbiter

logger = logging.getLogger(__name__)


class ModelGuard:
    def __init__(
        self,
        holder_id: str,
        *,
        load: Callable[[], Any],
        unload: Callable[[], Any],
        is_loaded: Callable[[], bool],
        uses_gpu: Callable[[], bool] | bool = True,
        idle_ttl: float = 120.0,
        acquire_timeout: float = 300.0,
        reap_interval: float = 5.0,
        arbiter: Arbiter | None = None,
    ) -> None:
        self.holder_id = holder_id
        self._load = load
        self._unload = unload
        self._is_loaded = is_loaded
        self._uses_gpu = uses_gpu if callable(uses_gpu) else (lambda: bool(uses_gpu))
        self.idle_ttl = idle_ttl
        self.acquire_timeout = acquire_timeout
        self.reap_interval = reap_interval
        self._arbiter = arbiter or NullArbiter()

        self._inflight = 0
        self._last_used = time.monotonic()
        self._load_lock = asyncio.Lock()
        self._reaper_task: asyncio.Task | None = None
        self._holds_lease = False
        self._closing = False

    @classmethod
    async def create(
        cls, holder_id: str, *, redis_url: str | None = None, lease_ttl: int = 60, **kwargs: Any
    ) -> "ModelGuard":
        arbiter = await make_arbiter(redis_url, lease_ttl=lease_ttl)
        return cls(holder_id, arbiter=arbiter, **kwargs)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    async def _call(fn: Callable[[], Any]) -> Any:
        result = fn()
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _ensure_loaded(self) -> None:
        if await self._call(self._is_loaded):
            return
        async with self._load_lock:
            if await self._call(self._is_loaded):
                return
            logger.info("[%s] loading model", self.holder_id)
            await self._call(self._load)

    async def _ensure_unloaded(self) -> None:
        async with self._load_lock:
            if await self._call(self._is_loaded):
                await self._call(self._unload)

    async def _acquire_lease(self) -> None:
        deadline = time.monotonic() + self.acquire_timeout
        requested = False
        while True:
            if await self._arbiter.acquire(self.holder_id):
                self._holds_lease = True
                return
            if not requested:
                # Ask the current holder to yield the GPU when it goes idle.
                await self._arbiter.request_preempt(self.holder_id)
                requested = True
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"GPU busy: '{self.holder_id}' could not acquire the lease "
                    f"within {self.acquire_timeout:.0f}s"
                )
            await asyncio.sleep(0.5)

    # ── public surface ────────────────────────────────────────────────

    async def start(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reaper())

    async def stop(self) -> None:
        self._closing = True
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        if self._holds_lease:
            await self._arbiter.release(self.holder_id)
            self._holds_lease = False

    @asynccontextmanager
    async def session(self):
        """Hold the GPU + ensure the model is loaded for the duration."""
        if self._uses_gpu():
            await self._acquire_lease()
        self._inflight += 1
        self._last_used = time.monotonic()
        try:
            await self._ensure_loaded()
            yield
        finally:
            self._inflight -= 1
            self._last_used = time.monotonic()

    async def unload_now(self) -> None:
        """Force an immediate unload + lease release (manual /unload)."""
        await self._ensure_unloaded()
        if self._holds_lease:
            await self._arbiter.release(self.holder_id)
            self._holds_lease = False

    def status(self) -> dict[str, Any]:
        return {
            "holder_id": self.holder_id,
            "in_flight": self._inflight,
            "idle_seconds": round(time.monotonic() - self._last_used, 1),
            "idle_ttl": self.idle_ttl,
            "holds_gpu_lease": self._holds_lease,
            "uses_gpu": self._uses_gpu(),
        }

    # ── background eviction ───────────────────────────────────────────

    async def _reaper(self) -> None:
        while not self._closing:
            try:
                await asyncio.sleep(self.reap_interval)
                resident = await self._call(self._is_loaded)
                if not resident:
                    continue

                # Actively serving → keep the model + renew the lease.
                if self._inflight > 0:
                    if self._holds_lease:
                        await self._arbiter.renew(self.holder_id)
                    continue

                idle = time.monotonic() - self._last_used
                preempted = self._holds_lease and await self._arbiter.preempt_requested(
                    self.holder_id
                )
                # Re-check in-flight AFTER the await above, with no further await
                # before the decision — closes the unload-mid-request window.
                if self._inflight == 0 and (preempted or idle >= self.idle_ttl):
                    reason = "preempted by another service" if preempted else (
                        f"idle {idle:.0f}s >= {self.idle_ttl:.0f}s"
                    )
                    logger.info("[%s] unloading model (%s)", self.holder_id, reason)
                    await self._ensure_unloaded()
                    if self._holds_lease:
                        await self._arbiter.release(self.holder_id)
                        self._holds_lease = False
                elif self._holds_lease:
                    # Idle but still warm — keep renewing so nobody else grabs
                    # the GPU while our weights are resident.
                    await self._arbiter.renew(self.holder_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[%s] reaper iteration failed", self.holder_id)
