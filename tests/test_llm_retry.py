"""Tests for the unified LLM retry kernel in common/llm.py + common/metric.py.

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_llm_retry.py

Each test_* function is run in isolation; failures print a traceback and the
runner exits non-zero if any test fails.

Fake-client injection: tests monkeypatch `common.llm._get_async_client` so no
real network calls happen. Retry delays are zeroed by monkeypatching
`common.llm._retry_delay`.
"""
import asyncio
import os
import sys
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The suite never talks to the network (fake client injected), but
# _get_async_client() constructs a real AsyncOpenAI, which raises without a
# key. Keep the suite self-contained (e.g. inside the eval container).
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

import httpx
import openai

import common.llm as llm_mod
from common.llm import Agent, Embedding


# ---------------- helpers ----------------

def _fake_response(content="hello", finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )],
        usage=None,
    )


def _fake_embedding_response(n):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in range(n)],
        usage=None,
    )


def _make_status_error(cls, status_code, message="err", headers=None):
    """Build an openai APIStatusError subclass with a real httpx response."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request, headers=headers or {})
    return cls(message, response=response, body=None)


def _rate_limit_error(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return _make_status_error(openai.RateLimitError, 429, "rate limited", headers)


def _bad_request_error():
    return _make_status_error(openai.BadRequestError, 400, "bad request")


def _timeout_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APITimeoutError(request=request)


class FakeAsyncClient:
    """Scripted fake for AsyncOpenAI. `script` is a list of either Exceptions
    (raised) or response objects (returned); the last entry repeats forever.
    Records every create() kwargs in .calls and tracks peak concurrency."""

    def __init__(self, script, delay=0.0):
        self._script = list(script)
        self._delay = delay
        self.calls = []
        self.in_flight = 0
        self.peak_in_flight = 0
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                return await outer._create(kwargs)

        class _Embeddings:
            async def create(self, **kwargs):
                return await outer._create(kwargs)

        self.chat = SimpleNamespace(completions=_Completions())
        self.embeddings = _Embeddings()

    async def _create(self, kwargs):
        self.calls.append(kwargs)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            action = self._script.pop(0) if len(self._script) > 1 else self._script[0]
            if isinstance(action, Exception):
                raise action
            return action
        finally:
            self.in_flight -= 1


class _Patched:
    """Context manager: patch common.llm._get_async_client to return `fake`
    and zero out retry delays."""

    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self._orig_get = llm_mod._get_async_client
        self._orig_delay = llm_mod._retry_delay
        llm_mod._get_async_client = lambda *a, **k: self.fake
        llm_mod._retry_delay = lambda *a, **k: 0.0
        return self.fake

    def __exit__(self, *exc):
        llm_mod._get_async_client = self._orig_get
        llm_mod._retry_delay = self._orig_delay
        return False


# ---------------- Agent: retry classification ----------------

def test_agent_retries_rate_limit():
    """429 must be retried (it previously raised immediately)."""
    fake = FakeAsyncClient([_rate_limit_error(), _fake_response("ok")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1")
        answer = asyncio.run(agent.ask("hi"))
    assert answer == "ok", f"expected 'ok', got {answer!r}"
    assert len(fake.calls) == 2, f"expected 2 attempts, got {len(fake.calls)}"


def test_agent_retries_timeout():
    """Timeouts stay retryable (regression guard)."""
    fake = FakeAsyncClient([_timeout_error(), _timeout_error(), _fake_response("ok")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1")
        answer = asyncio.run(agent.ask("hi"))
    assert answer == "ok"
    assert len(fake.calls) == 3


def test_agent_does_not_retry_400():
    """Deterministic 4xx (except 429) must fail fast."""
    fake = FakeAsyncClient([_bad_request_error(), _fake_response("ok")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1")
        try:
            asyncio.run(agent.ask("hi"))
            assert False, "expected BadRequestError to raise"
        except openai.BadRequestError:
            pass
    assert len(fake.calls) == 1, f"400 was retried: {len(fake.calls)} attempts"


def test_agent_retries_empty_response():
    """Empty content is a transient upstream glitch -> retry, and the fake
    'Error occur when generating answer.' string must be gone."""
    fake = FakeAsyncClient([_fake_response(content=""), _fake_response("real answer")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1")
        answer = asyncio.run(agent.ask("hi"))
    assert answer == "real answer", f"got {answer!r}"
    assert len(fake.calls) == 2


def test_agent_length_limit_not_retried():
    """finish_reason=length with empty content is deterministic -> no retry,
    clear error."""
    fake = FakeAsyncClient([
        _fake_response(content="", finish_reason="length"),
        _fake_response("ok"),
    ])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1")
        try:
            asyncio.run(agent.ask("hi"))
            assert False, "expected LengthLimitError"
        except llm_mod.LengthLimitError:
            pass
    assert len(fake.calls) == 1, f"length-limit was retried: {len(fake.calls)}"


def test_agent_exhausts_retries_and_raises():
    fake = FakeAsyncClient([_timeout_error()])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1", max_retries=2)
        try:
            asyncio.run(agent.ask("hi"))
            assert False, "expected APITimeoutError after exhaustion"
        except openai.APITimeoutError:
            pass
    assert len(fake.calls) == 3, f"expected 3 attempts (1+2 retries), got {len(fake.calls)}"


# ---------------- Agent: request shaping ----------------

def test_agent_model_suffix_parsing():
    """Agent('gpt-5-mini/low') -> model=gpt-5-mini, reasoning_effort=low."""
    fake = FakeAsyncClient([_fake_response("ok")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-5-mini/low")
        asyncio.run(agent.ask("hi"))
    kwargs = fake.calls[0]
    assert kwargs["model"] == "gpt-5-mini", f"model={kwargs['model']!r}"
    assert kwargs.get("reasoning_effort") == "low", f"reasoning_effort={kwargs.get('reasoning_effort')!r}"


def test_agent_explicit_effort_beats_suffix():
    fake = FakeAsyncClient([_fake_response("ok")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-5-mini/low")
        asyncio.run(agent.ask("hi", reasoning_effort="high"))
    assert fake.calls[0].get("reasoning_effort") == "high"


def test_agent_suffix_dropped_for_non_reasoning_model():
    """gpt-4.1/low: strip the suffix but never send reasoning_effort."""
    fake = FakeAsyncClient([_fake_response("ok")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1/low")
        asyncio.run(agent.ask("hi"))
    kwargs = fake.calls[0]
    assert kwargs["model"] == "gpt-4.1"
    assert "reasoning_effort" not in kwargs


def test_agent_max_completion_tokens_default():
    fake = FakeAsyncClient([_fake_response("ok")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1")
        asyncio.run(agent.ask("hi"))
    assert fake.calls[0].get("max_completion_tokens") == 16384, (
        f"max_completion_tokens={fake.calls[0].get('max_completion_tokens')!r}"
    )


def test_agent_max_completion_tokens_none_omits():
    fake = FakeAsyncClient([_fake_response("ok")])
    with _Patched(fake):
        agent = Agent(system_prompt="s", model="gpt-4.1", max_completion_tokens=None)
        asyncio.run(agent.ask("hi"))
    assert "max_completion_tokens" not in fake.calls[0]


# ---------------- global concurrency + shared client ----------------

def test_global_semaphore_caps_concurrency():
    os.environ["MEMEVOL_LLM_MAX_CONCURRENT"] = "2"
    try:
        fake = FakeAsyncClient([_fake_response("ok")], delay=0.05)
        with _Patched(fake):
            async def run():
                agents = [Agent(system_prompt="s", model="gpt-4.1") for _ in range(6)]
                await asyncio.gather(*[a.ask("hi") for a in agents])
            asyncio.run(run())
        assert fake.peak_in_flight <= 2, f"peak concurrency {fake.peak_in_flight} > 2"
        assert len(fake.calls) == 6
    finally:
        del os.environ["MEMEVOL_LLM_MAX_CONCURRENT"]


def test_shared_client_per_loop():
    """Two lookups in the same running loop return the same client object."""
    async def run():
        c1 = llm_mod._get_async_client()
        c2 = llm_mod._get_async_client()
        return c1 is c2
    assert asyncio.run(run()), "clients differ within one event loop"


# ---------------- retry delay math ----------------

def test_retry_delay_jitter_bounds():
    exc = _timeout_error()
    for attempt in range(4):
        expected = min(60.0, 2.0 * (2 ** (attempt + 1)))
        for _ in range(50):
            d = llm_mod._retry_delay(exc, attempt)
            assert expected * 0.5 - 1e-9 <= d <= expected * 1.5 + 1e-9, (
                f"attempt {attempt}: delay {d} outside [{expected*0.5}, {expected*1.5}]"
            )


def test_retry_delay_respects_retry_after():
    exc = _rate_limit_error(retry_after=45)
    for _ in range(20):
        d = llm_mod._retry_delay(exc, 0)
        assert d >= 45.0, f"delay {d} ignored Retry-After: 45"


# ---------------- Embedding ----------------

def test_embedding_does_not_retry_400():
    fake = FakeAsyncClient([_bad_request_error(), _fake_embedding_response(1)])
    with _Patched(fake):
        emb = Embedding()
        try:
            asyncio.run(emb.get_batch_embeddings(["hello"]))
            assert False, "expected failure on 400"
        except (openai.BadRequestError, RuntimeError):
            pass
    assert len(fake.calls) == 1, f"400 was retried: {len(fake.calls)} attempts"


def test_embedding_retries_timeout():
    fake = FakeAsyncClient([_timeout_error(), _fake_embedding_response(1)])
    with _Patched(fake):
        emb = Embedding()
        result = asyncio.run(emb.get_batch_embeddings(["hello"]))
    assert len(result) == 1 and result[0] == [0.1, 0.2]
    assert len(fake.calls) == 2


# ---------------- Judge ----------------

def test_judge_retries_rate_limit():
    from common.metric import Judge
    fake = FakeAsyncClient([
        _rate_limit_error(),
        _fake_response('{"reason": "good", "score": 8}'),
    ])
    with _Patched(fake):
        judge = Judge(model="gpt-4.1")
        score, reason = asyncio.run(judge.score("q", "p", "r"))
    assert score == 8, f"score={score}, reason={reason}"
    assert len(fake.calls) == 2


def test_judge_retries_bad_json_once():
    from common.metric import Judge
    fake = FakeAsyncClient([
        _fake_response("not json at all"),
        _fake_response('{"reason": "good", "score": 7}'),
    ])
    with _Patched(fake):
        judge = Judge(model="gpt-4.1")
        score, reason = asyncio.run(judge.score("q", "p", "r"))
    assert score == 7, f"bad JSON not retried: score={score}, reason={reason}"
    assert len(fake.calls) == 2


def test_judge_bad_json_twice_returns_min():
    from common.metric import Judge
    fake = FakeAsyncClient([_fake_response("still not json")])
    with _Patched(fake):
        judge = Judge(model="gpt-4.1")
        score, reason = asyncio.run(judge.score("q", "p", "r"))
    assert score == 0
    assert len(fake.calls) == 2, f"expected exactly 2 parse attempts, got {len(fake.calls)}"


def test_judge_never_raises_on_exhaustion():
    from common.metric import Judge
    fake = FakeAsyncClient([_timeout_error()])
    with _Patched(fake):
        judge = Judge(model="gpt-4.1", max_retries=2)
        score, reason = asyncio.run(judge.score("q", "p", "r"))
    assert score == 0
    assert "error" in reason.lower() or "timed" in reason.lower(), f"reason={reason!r}"


def test_judge_strips_effort_suffix():
    """Audit M8: 'gpt-5-mini/low' sent verbatim would 404 every judge call
    and silently zero the whole benchmark. The suffix must be parsed like
    Agent does, with the explicit effort reaching the payload."""
    from common.metric import Judge
    fake = FakeAsyncClient([_fake_response('{"reason": "ok", "score": 9}')])
    with _Patched(fake):
        judge = Judge(model="gpt-5-mini/high")
        score, _ = asyncio.run(judge.score("q", "p", "r"))
    assert score == 9
    assert fake.calls[0]["model"] == "gpt-5-mini", fake.calls[0]["model"]
    assert fake.calls[0]["reasoning_effort"] == "high"


def test_judge_default_low_effort_without_suffix():
    from common.metric import Judge
    fake = FakeAsyncClient([_fake_response('{"reason": "ok", "score": 5}')])
    with _Patched(fake):
        judge = Judge(model="gpt-5-mini")
        asyncio.run(judge.score("q", "p", "r"))
    assert fake.calls[0]["model"] == "gpt-5-mini"
    assert fake.calls[0]["reasoning_effort"] == "low"


def test_judge_coerces_string_float_score():
    """A '8.5' score string must coerce (int(float(...))) instead of falling
    into the broad except and skipping the parse retry."""
    from common.metric import Judge
    fake = FakeAsyncClient([_fake_response('{"reason": "ok", "score": "8.5"}')])
    with _Patched(fake):
        judge = Judge(model="gpt-4.1")
        score, reason = asyncio.run(judge.score("q", "p", "r"))
    assert score == 8, f"score={score}, reason={reason}"
    assert len(fake.calls) == 1


def test_judge_non_dict_json_retries_then_min():
    from common.metric import Judge
    fake = FakeAsyncClient([_fake_response('["valid json", "but not a dict"]')])
    with _Patched(fake):
        judge = Judge(model="gpt-4.1")
        score, reason = asyncio.run(judge.score("q", "p", "r"))
    assert score == 0
    assert len(fake.calls) == 2, f"expected parse retry, got {len(fake.calls)} calls"


def test_judge_bad_template_degrades_not_raises():
    from common.metric import Judge
    fake = FakeAsyncClient([_fake_response('{"reason": "ok", "score": 1}')])
    with _Patched(fake):
        judge = Judge(model="gpt-4.1",
                      prompt_template="Broken {placeholder} {query}")
        score, reason = asyncio.run(judge.score("q", "p", "r"))
    assert score == 0
    assert "template" in reason.lower(), reason
    assert len(fake.calls) == 0, "must fail before any API call"


# ---------------- runner ----------------

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
