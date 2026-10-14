import random

import httpx
import pytest

from research_assistant.sources.http import HttpClient, HttpError, RateLimiter, backoff_delay


async def no_sleep(_seconds: float) -> None:
    return None


def client(handler, **kwargs) -> HttpClient:
    return HttpClient("test", rate_per_second=1000, transport=httpx.MockTransport(handler), sleep=no_sleep, **kwargs)


async def test_rate_limiter_spaces_calls():
    now = [0.0]
    waits = []

    async def fake_sleep(s):
        waits.append(s)

    limiter = RateLimiter(2.0, clock=lambda: now[0], sleep=fake_sleep)
    for _ in range(3):
        await limiter.acquire()
    assert waits == [0.5, 1.0]


def test_backoff_delay_is_bounded():

    rng = random.Random(0)
    assert all(0 <= backoff_delay(a, rng=rng) <= min(30, 2**a) for a in range(10))


async def test_retries_429_then_succeeds_and_tracks_cost():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "1"})
        return httpx.Response(200, json={"ok": True}, headers={"x-ratelimit-cost-usd": "0.001"})

    http = client(handler)
    assert await http.get_json("https://api.example.com/x", {"a": 1}) == {"ok": True}
    assert len(calls) == 2 and http.cost_usd == pytest.approx(0.001)


async def test_gives_up_after_max_tries_and_does_not_retry_404():
    http = client(lambda r: httpx.Response(503), max_tries=3)
    with pytest.raises(HttpError):
        await http.get_json("https://api.example.com/x")
    calls = []
    http = client(lambda r: calls.append(r) or httpx.Response(404))
    with pytest.raises(HttpError) as err:
        await http.get_json("https://api.example.com/x")
    assert err.value.status == 404 and len(calls) == 1


async def test_disk_cache_serves_repeat_calls(tmp_path):
    calls = []
    http = client(lambda r: calls.append(r) or httpx.Response(200, json={"n": 1}), cache_dir=str(tmp_path))
    await http.get_json("https://api.example.com/x", {"q": "a"})
    await http.get_json("https://api.example.com/x", {"q": "a"})
    assert len(calls) == 1


async def test_get_bytes_enforces_size_limit():
    http = client(lambda r: httpx.Response(200, content=b"x" * 100, headers={"content-type": "application/pdf"}))
    body, kind = await http.get_bytes("https://arxiv.org/pdf/1", max_bytes=200)
    assert len(body) == 100 and kind == "application/pdf"
    with pytest.raises(HttpError):
        await http.get_bytes("https://arxiv.org/pdf/1", max_bytes=50)
