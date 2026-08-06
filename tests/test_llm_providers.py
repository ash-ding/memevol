"""Tests for the multi-provider chat kernel (OpenAI | Anthropic API | Vertex).

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_llm_providers.py          # repo-root uv project

Covers common/llm.py:
  - _provider_for_model routing
  - anthropic payload shape (system split, required max_tokens, effort via
    extra_body gated by the model allowlist, no temperature/response_format)
  - retry classification for anthropic exceptions (429/5xx retried; 400 and
    refusal fast-fail; Retry-After floor honoured)
  - stop_reason mapping (refusal → RefusalError, max_tokens+no text →
    LengthLimitError, empty text → EmptyResponseError → retried)
  - usage normalization (input/output_tokens → prompt/completion/total)
  - transport selection (_build_anthropic_client: api vs vertex, actionable
    errors on missing vertex env)
  - Embedding rejects claude-* models
and common/judge.py:
  - Judge on a claude-* model routes through the kernel; never-raises holds.
"""
import asyncio
import contextlib
import os
import sys
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

import httpx  # noqa: E402

from common import llm as llm_mod  # noqa: E402
from common.llm import (  # noqa: E402
    Agent,
    Embedding,
    EmptyResponseError,
    LengthLimitError,
    RefusalError,
    _chat_completion,
    _provider_for_model,
    _split_system_messages,
    _supports_anthropic_effort,
)

try:
    import anthropic
except ImportError:
    anthropic = None

HAS_ANTHROPIC = anthropic is not None


# ---------------- fakes (mirroring tests/test_llm_retry.py style) ----------------

def _a_response(text="ok", stop_reason="end_turn", in_tok=10, out_tok=5, blocks=None):
    content = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def _a_status_error(cls, status_code, message="err", headers=None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request, headers=headers or {})
    return cls(message, response=response, body=None)


class FakeAnthropicClient:
    """Scripted fake exposing `.messages.create`. Same contract as the
    FakeAsyncClient in test_llm_retry.py: script entries are exceptions
    (raised) or responses (returned); the last entry repeats."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        outer = self

        class _Messages:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                action = outer._script.pop(0) if len(outer._script) > 1 else outer._script[0]
                if isinstance(action, Exception):
                    raise action
                return action

        self.messages = _Messages()


class _Patched:
    """Patch common.llm._get_async_client → fake; zero retry delays."""

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


@contextlib.contextmanager
def _patched_env(**env):
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _chat(model="claude-opus-4-6", **kw):
    kw.setdefault("messages", [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hi"},
    ])
    kw.setdefault("timeout", 5.0)
    kw.setdefault("max_retries", 2)
    return asyncio.run(_chat_completion(model=model, what="Agent", **kw))


# ---------------- routing ----------------

def test_provider_routing():
    assert _provider_for_model("claude-opus-4-6") == "anthropic"
    assert _provider_for_model("claude-sonnet-5") == "anthropic"
    assert _provider_for_model("gpt-5-mini") == "openai"
    assert _provider_for_model("o3") == "openai"
    assert _provider_for_model("text-embedding-3-small") == "openai"
    assert _provider_for_model("") == "openai"


def test_effort_allowlist():
    assert _supports_anthropic_effort("claude-opus-4-6")
    assert _supports_anthropic_effort("claude-sonnet-5")
    assert not _supports_anthropic_effort("claude-haiku-4-5")


def test_split_system_messages():
    sys_txt, rest = _split_system_messages([
        {"role": "system", "content": "a"},
        {"role": "user", "content": "u1"},
        {"role": "system", "content": "b"},
        {"role": "assistant", "content": "x"},
    ])
    assert sys_txt == "a\n\nb"
    assert [m["role"] for m in rest] == ["user", "assistant"]


# ---------------- anthropic payload shape ----------------

def test_anthropic_payload_shape():
    if not HAS_ANTHROPIC:
        return
    fake = FakeAnthropicClient([_a_response("hello")])
    with _Patched(fake):
        text, usage = _chat(effort="low", max_tokens=4096)
    assert text == "hello"
    call = fake.calls[0]
    assert call["system"] == "sys prompt"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["max_tokens"] == 4096
    assert call["extra_body"] == {"output_config": {"effort": "low"}}
    assert "temperature" not in call and "response_format" not in call


def test_anthropic_max_tokens_required_default():
    if not HAS_ANTHROPIC:
        return
    fake = FakeAnthropicClient([_a_response()])
    with _Patched(fake):
        _chat(max_tokens=None)   # caller disabled the cap → API still needs one
    assert fake.calls[0]["max_tokens"] == 16384


def test_anthropic_effort_gated_by_allowlist():
    if not HAS_ANTHROPIC:
        return
    fake = FakeAnthropicClient([_a_response()])
    with _Patched(fake):
        _chat(model="claude-haiku-4-5", effort="low")
    assert "extra_body" not in fake.calls[0]


def test_anthropic_multi_block_text_join():
    if not HAS_ANTHROPIC:
        return
    blocks = [
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text="part1 "),
        SimpleNamespace(type="text", text="part2"),
    ]
    fake = FakeAnthropicClient([_a_response(blocks=blocks)])
    with _Patched(fake):
        text, _ = _chat()
    assert text == "part1 part2"


# ---------------- retry classification ----------------

def test_anthropic_rate_limit_retried():
    if not HAS_ANTHROPIC:
        return
    err = _a_status_error(anthropic.RateLimitError, 429, "slow down")
    fake = FakeAnthropicClient([err, _a_response("recovered")])
    with _Patched(fake):
        text, _ = _chat()
    assert text == "recovered" and len(fake.calls) == 2


def test_anthropic_5xx_retried():
    if not HAS_ANTHROPIC:
        return
    err = _a_status_error(anthropic.InternalServerError, 529, "overloaded")
    fake = FakeAnthropicClient([err, _a_response("recovered")])
    with _Patched(fake):
        text, _ = _chat()
    assert text == "recovered" and len(fake.calls) == 2


def test_anthropic_400_fast_fails():
    if not HAS_ANTHROPIC:
        return
    err = _a_status_error(anthropic.BadRequestError, 400, "bad params")
    fake = FakeAnthropicClient([err])
    with _Patched(fake):
        try:
            _chat()
        except anthropic.BadRequestError:
            pass
        else:
            raise AssertionError("expected BadRequestError")
    assert len(fake.calls) == 1  # no retries on deterministic 4xx


def test_anthropic_retry_after_floor():
    if not HAS_ANTHROPIC:
        return
    err = _a_status_error(anthropic.RateLimitError, 429, "x", {"retry-after": "7"})
    delay = llm_mod._retry_delay(err, attempt=0, base_delay=0.001, cap=60.0)
    assert delay >= 7.0


# ---------------- stop_reason mapping ----------------

def test_refusal_not_retried():
    if not HAS_ANTHROPIC:
        return
    fake = FakeAnthropicClient([_a_response("", stop_reason="refusal", blocks=[])])
    with _Patched(fake):
        try:
            _chat()
        except RefusalError:
            pass
        else:
            raise AssertionError("expected RefusalError")
    assert len(fake.calls) == 1


def test_max_tokens_no_text_is_length_error():
    if not HAS_ANTHROPIC:
        return
    fake = FakeAnthropicClient([_a_response("", stop_reason="max_tokens", blocks=[])])
    with _Patched(fake):
        try:
            _chat()
        except LengthLimitError:
            pass
        else:
            raise AssertionError("expected LengthLimitError")
    assert len(fake.calls) == 1


def test_empty_text_retried():
    if not HAS_ANTHROPIC:
        return
    fake = FakeAnthropicClient([_a_response("  ", blocks=[SimpleNamespace(type="text", text="  ")]),
                                _a_response("fine")])
    with _Patched(fake):
        text, _ = _chat()
    assert text == "fine" and len(fake.calls) == 2


# ---------------- usage normalization ----------------

def test_usage_normalized():
    if not HAS_ANTHROPIC:
        return
    fake = FakeAnthropicClient([_a_response(in_tok=100, out_tok=30)])
    with _Patched(fake):
        _, usage = _chat()
    assert usage == {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}


def test_agent_ask_claude_tracks_tokens():
    if not HAS_ANTHROPIC:
        return
    from common.tokens import TokenTracker
    import common.tokens as tokens_mod
    fake = FakeAnthropicClient([_a_response("answer", in_tok=11, out_tok=7)])
    tracker = TokenTracker()
    old = tokens_mod.GLOBAL_TOKEN_TRACKER
    tokens_mod.GLOBAL_TOKEN_TRACKER = tracker
    try:
        with _Patched(fake):
            agent = Agent(system_prompt="s", model="claude-opus-4-6/low")
            answer = asyncio.run(agent.ask("q"))
    finally:
        tokens_mod.GLOBAL_TOKEN_TRACKER = old
    assert answer == "answer"
    stats = tracker.summary()["claude-opus-4-6"]
    assert stats["prompt_tokens"] == 11 and stats["completion_tokens"] == 7
    # "/low" suffix parsed off the model and delivered as effort via extra_body
    assert fake.calls[0]["extra_body"] == {"output_config": {"effort": "low"}}


# ---------------- transport selection ----------------

def test_build_client_api_transport():
    if not HAS_ANTHROPIC:
        return
    with _patched_env(MEMEVOL_ANTHROPIC_TRANSPORT=None, ANTHROPIC_API_KEY="sk-x"):
        client = llm_mod._build_anthropic_client()
    assert isinstance(client, anthropic.AsyncAnthropic)


def test_build_client_vertex_transport():
    if not HAS_ANTHROPIC:
        return
    captured = {}

    class FakeVertex:
        def __init__(self, **kw):
            captured.update(kw)

    orig = anthropic.AsyncAnthropicVertex
    anthropic.AsyncAnthropicVertex = FakeVertex
    try:
        with _patched_env(MEMEVOL_ANTHROPIC_TRANSPORT="vertex",
                          ANTHROPIC_VERTEX_PROJECT_ID="itpc-gcp-ai-eng-claude",
                          CLOUD_ML_REGION="us-east5"):
            client = llm_mod._build_anthropic_client()
    finally:
        anthropic.AsyncAnthropicVertex = orig
    assert isinstance(client, FakeVertex)
    assert captured["project_id"] == "itpc-gcp-ai-eng-claude"
    assert captured["region"] == "us-east5"
    assert captured["max_retries"] == 0


def test_build_client_vertex_missing_project_actionable():
    if not HAS_ANTHROPIC:
        return
    with _patched_env(MEMEVOL_ANTHROPIC_TRANSPORT="vertex",
                      ANTHROPIC_VERTEX_PROJECT_ID=None, CLOUD_ML_REGION=None):
        try:
            llm_mod._build_anthropic_client()
        except RuntimeError as exc:
            assert "ANTHROPIC_VERTEX_PROJECT_ID" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


def test_bad_transport_value_rejected():
    with _patched_env(MEMEVOL_ANTHROPIC_TRANSPORT="bedrock"):
        try:
            llm_mod._anthropic_transport()
        except RuntimeError as exc:
            assert "vertex" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


# ---------------- Embedding guard ----------------

def test_embedding_rejects_claude():
    try:
        Embedding(model="claude-opus-4-6")
    except ValueError as exc:
        assert "embedding" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


# ---------------- Judge on a claude model ----------------

def test_judge_claude_path():
    if not HAS_ANTHROPIC:
        return
    from common.judge import Judge
    fake = FakeAnthropicClient([_a_response('{"score": 8, "reason": "good"}')])
    with _Patched(fake):
        judge = Judge(model="claude-opus-4-6")
        score, reason = asyncio.run(judge.score("q", "pred", "ref"))
    assert (score, reason) == (8, "good")
    call = fake.calls[0]
    assert "response_format" not in call          # anthropic branch
    # judge default effort "low" flows through the anthropic allowlist
    assert call["extra_body"] == {"output_config": {"effort": "low"}}


def test_judge_claude_never_raises_on_transport_death():
    if not HAS_ANTHROPIC:
        return
    from common.judge import Judge
    err = _a_status_error(anthropic.InternalServerError, 500, "boom")
    fake = FakeAnthropicClient([err])
    with _Patched(fake):
        judge = Judge(model="claude-opus-4-6", max_retries=1)
        score, reason = asyncio.run(judge.score("q", "pred", "ref"))
    assert score == judge.score_min and "Judge error" in reason


# ---------------- guarded-import behavior ----------------

def test_missing_sdk_actionable_error():
    """With the SDK absent, claude routing must fail with install guidance."""
    orig = llm_mod.anthropic
    llm_mod.anthropic = None
    try:
        try:
            _chat()
        except RuntimeError as exc:
            assert "anthropic[vertex]" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        llm_mod.anthropic = orig


# ---------------- runner ----------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed.append(name)
    if not HAS_ANTHROPIC:
        print("NOTE: anthropic SDK not installed — anthropic-path tests were no-ops")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
