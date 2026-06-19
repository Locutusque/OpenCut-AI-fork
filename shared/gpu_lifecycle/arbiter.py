"""Cross-process single-GPU arbiter.

Coordinates which service may keep a model resident on a shared GPU. Because
the OpenCut-AI services are separate containers, an in-process lock is not
enough — coordination goes through Redis (already in the stack).

Protocol (cooperative + preemptible):

* A single ``tenant`` key names the service currently allowed to hold the GPU.
  It carries a short TTL and is renewed by the holder while its model is
  resident, so a crashed holder's claim self-expires.
* A waiter that can't claim the tenant sets a ``preempt`` key naming itself.
  The current holder, once it has no in-flight work, sees the preempt request,
  unloads its model and releases — then the waiter claims the slot.

If Redis is unavailable the arbiter degrades to :class:`NullArbiter` (always
grants), so a service still runs — it just loses cross-service coordination.
Idle eviction (see :mod:`guard`) keeps working regardless.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

TENANT_KEY = "opencut:gpu:tenant"
PREEMPT_KEY = "opencut:gpu:preempt"


@runtime_checkable
class Arbiter(Protocol):
    async def acquire(self, holder_id: str) -> bool: ...
    async def renew(self, holder_id: str) -> bool: ...
    async def release(self, holder_id: str) -> None: ...
    async def preempt_requested(self, holder_id: str) -> bool: ...
    async def request_preempt(self, holder_id: str) -> None: ...


class NullArbiter:
    """No-op arbiter — always grants. Used when Redis is unavailable."""

    async def acquire(self, holder_id: str) -> bool:
        return True

    async def renew(self, holder_id: str) -> bool:
        return True

    async def release(self, holder_id: str) -> None:
        return None

    async def preempt_requested(self, holder_id: str) -> bool:
        return False

    async def request_preempt(self, holder_id: str) -> None:
        return None


class RedisArbiter:
    """Redis-backed single-tenant GPU lease."""

    def __init__(self, client, lease_ttl: int = 60, preempt_ttl: int = 15) -> None:
        self._r = client
        self._ttl = lease_ttl
        self._preempt_ttl = preempt_ttl

    async def acquire(self, holder_id: str) -> bool:
        # Already ours → just renew and clear any preempt request.
        current = await self._r.get(TENANT_KEY)
        if current == holder_id:
            await self._r.expire(TENANT_KEY, self._ttl)
            await self._r.delete(PREEMPT_KEY)
            return True
        # Try to claim the empty slot.
        claimed = await self._r.set(TENANT_KEY, holder_id, nx=True, ex=self._ttl)
        if claimed:
            await self._r.delete(PREEMPT_KEY)
            return True
        return False

    async def renew(self, holder_id: str) -> bool:
        if await self._r.get(TENANT_KEY) == holder_id:
            await self._r.expire(TENANT_KEY, self._ttl)
            return True
        return False

    async def release(self, holder_id: str) -> None:
        # Only release if we still own it (best-effort check-and-delete).
        if await self._r.get(TENANT_KEY) == holder_id:
            await self._r.delete(TENANT_KEY)

    async def preempt_requested(self, holder_id: str) -> bool:
        waiter = await self._r.get(PREEMPT_KEY)
        return bool(waiter) and waiter != holder_id

    async def request_preempt(self, holder_id: str) -> None:
        await self._r.set(PREEMPT_KEY, holder_id, ex=self._preempt_ttl)


async def make_arbiter(redis_url: str | None, lease_ttl: int = 60) -> Arbiter:
    """Build a Redis arbiter, falling back to NullArbiter on any failure."""
    if not redis_url:
        logger.warning(
            "No REDIS_URL set — GPU arbiter disabled (idle eviction still active, "
            "but no cross-service single-GPU coordination)."
        )
        return NullArbiter()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(redis_url, decode_responses=True)
        await client.ping()
        logger.info("GPU arbiter using Redis at %s", redis_url)
        return RedisArbiter(client, lease_ttl=lease_ttl)
    except Exception as exc:  # redis missing or unreachable
        logger.warning("Redis unavailable (%s) — GPU arbiter disabled.", exc)
        return NullArbiter()
