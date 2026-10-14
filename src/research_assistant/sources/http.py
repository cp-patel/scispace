"""One HTTP client for outside APIs: rate limiting, retries with backoff, timeouts, an optional disk cache, and cost tracking."""

import asyncio
import hashlib
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from research_assistant.observability import EXTERNAL_REQUESTS

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class HttpError(Exception):
    def __init__(self, status: int | None, url: str, message: str):
        super().__init__(f"{status} {url}: {message}")
        self.status = status
        self.url = url


class RateLimiter:
    """Spaces requests at least 1 / per_second seconds apart, across all coroutines sharing it."""

    def __init__(self, per_second: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.interval = 1.0 / per_second
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._clock = clock
        self._sleep = sleep

    async def acquire(self) -> None:
        async with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            await self._sleep(start - now)


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0, rng: random.Random | None = None) -> float:
    """Exponential backoff with full jitter: a random wait between 0 and min(cap, base * 2^attempt).
    The randomness keeps many clients from retrying in lockstep."""
    return (rng or random).uniform(0.0, min(cap, base * 2**attempt))


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return min(float(value), 60.0) if value is not None else None
    except ValueError:
        return None


class HttpClient:
    def __init__(
        self,
        service: str,
        *,
        rate_per_second: float,
        headers: dict[str, str] | None = None,
        cache_dir: str | None = None,
        timeout_s: float = 30.0,
        max_tries: int = 5,
        max_connections: int = 10,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.service = service
        self.limiter = RateLimiter(rate_per_second, clock=clock, sleep=sleep)
        self.cache_dir = Path(cache_dir) / service if cache_dir else None
        self.max_tries = max_tries
        self.cost_usd = 0.0
        self._sleep = sleep
        self.client = httpx.AsyncClient(
            headers=headers or {}, timeout=timeout_s, transport=transport, follow_redirects=True,
            limits=httpx.Limits(max_connections=max_connections),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def get_json(self, url: str, params: dict | None = None) -> dict:
        key = self._cache_key(url, params)
        cached = self._cache_read(key)
        if cached is not None:
            return cached
        data = (await self._request(url, params)).json()
        self._cache_write(key, data)
        return data

    async def get_text(self, url: str, params: dict | None = None) -> str:
        key = self._cache_key(url, params)
        cached = self._cache_read(key)
        if cached is not None:
            return cached["text"]
        text = (await self._request(url, params)).text
        self._cache_write(key, {"text": text})
        return text

    async def get_bytes(self, url: str, max_bytes: int) -> tuple[bytes, str]:
        """Download a body of at most max_bytes. Returns (body, content type). Never cached."""
        last: Exception | None = None
        for attempt in range(self.max_tries):
            await self.limiter.acquire()
            try:
                async with self.client.stream("GET", url) as response:
                    if response.status_code in RETRY_STATUS:
                        last = HttpError(response.status_code, url, "retryable status")
                        EXTERNAL_REQUESTS.labels(self.service, "retry").inc()
                        await self._sleep(_retry_after(response) or backoff_delay(attempt))
                        continue
                    if response.status_code >= 400:
                        EXTERNAL_REQUESTS.labels(self.service, "error").inc()
                        raise HttpError(response.status_code, url, "request failed")
                    parts, size = [], 0
                    async for part in response.aiter_bytes():
                        size += len(part)
                        if size > max_bytes:
                            EXTERNAL_REQUESTS.labels(self.service, "error").inc()
                            raise HttpError(response.status_code, url, f"body is larger than {max_bytes} bytes")
                        parts.append(part)
                    EXTERNAL_REQUESTS.labels(self.service, "ok").inc()
                    return b"".join(parts), response.headers.get("content-type", "")
            except httpx.TransportError as exc:
                last = exc
                EXTERNAL_REQUESTS.labels(self.service, "error").inc()
                await self._sleep(backoff_delay(attempt))
        raise HttpError(getattr(last, "status", None), url, f"gave up after {self.max_tries} tries: {last}")

    async def _request(self, url: str, params: dict | None) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.max_tries):
            await self.limiter.acquire()
            try:
                response = await self.client.get(url, params=params)
            except httpx.TransportError as exc:
                last = exc
                EXTERNAL_REQUESTS.labels(self.service, "error").inc()
                log.warning("transport error, retrying", extra={"service": self.service, "attempt": attempt})
                await self._sleep(backoff_delay(attempt))
                continue
            self.cost_usd += float(response.headers.get("x-ratelimit-cost-usd") or 0.0)
            if response.status_code in RETRY_STATUS:
                last = HttpError(response.status_code, url, response.text[:200])
                EXTERNAL_REQUESTS.labels(self.service, "retry").inc()
                log.warning("retryable status", extra={"service": self.service, "status": response.status_code, "attempt": attempt})
                await self._sleep(_retry_after(response) or backoff_delay(attempt))
                continue
            if response.status_code >= 400:
                EXTERNAL_REQUESTS.labels(self.service, "error").inc()
                raise HttpError(response.status_code, url, response.text[:200])
            EXTERNAL_REQUESTS.labels(self.service, "ok").inc()
            return response
        raise HttpError(getattr(last, "status", None), url, f"gave up after {self.max_tries} tries: {last}")

    def _cache_key(self, url: str, params: dict | None) -> str:
        raw = url + "?" + json.dumps(params or {}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_read(self, key: str) -> dict | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _cache_write(self, key: str, data: dict) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(data))
