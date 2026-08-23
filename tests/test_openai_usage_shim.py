"""The SDK-boundary shim that captures the vendored baselines' own OpenAI calls.

    uv run python tests/test_openai_usage_shim.py

Drives the REAL openai SDK over a mocked HTTP transport, so what is verified is
that the patch point is actually on the path a vendored caller takes — not that
a stub we wrote calls a stub we wrote.
"""
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx           # noqa: E402
import openai          # noqa: E402

from common import openai_usage    # noqa: E402
from common import tokens as T     # noqa: E402


CHAT_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
    # Deliberately different from the requested model: the server's echoed
    # name is what gets recorded (it resolves aliases / deployment names).
    "model": "gpt-4o-mini-2024-07-18",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128},
}

EMBED_BODY = {
    "object": "list", "model": "text-embedding-3-small",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "usage": {"prompt_tokens": 17, "total_tokens": 17},
}


def _transport(body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)
    return httpx.MockTransport(handler)


def _client(body) -> "openai.OpenAI":
    return openai.OpenAI(api_key="test",
                         http_client=httpx.Client(transport=_transport(body)))


def _aclient(body) -> "openai.AsyncOpenAI":
    return openai.AsyncOpenAI(api_key="test",
                              http_client=httpx.AsyncClient(transport=_transport(body)))


def _fresh_tracker() -> T.TokenTracker:
    tracker = T.TokenTracker()
    T.GLOBAL_TOKEN_TRACKER = tracker
    return tracker


def setup_module(_module=None):
    openai_usage.install()


# ---------------------------------------------------------------------------

def test_a_vendored_style_sync_call_is_captured():
    """This is how amem / simplemem / memoryos / lightmem call the API:
    a plain synchronous client built inside vendored code."""
    setup_module()
    tracker = _fresh_tracker()
    with T.phase(T.BUILD):
        _client(CHAT_BODY).chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])

    usage = tracker.summary()
    entry = usage["by_model_phase"]["gpt-4o-mini-2024-07-18"]["build"]
    assert entry["calls"] == 1
    assert entry["prompt_tokens"] == 120 and entry["completion_tokens"] == 8
    assert entry["total_tokens"] == 128
    assert usage["by_phase"]["build"]["calls"] == 1


def test_a_vendored_style_async_call_is_captured():
    """zep/graphiti uses AsyncOpenAI."""
    setup_module()
    tracker = _fresh_tracker()

    async def go():
        with T.phase(T.RETRIEVE):
            await _aclient(CHAT_BODY).chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])

    asyncio.run(go())
    assert tracker.summary()["by_phase"]["retrieve"]["calls"] == 1


def test_embedding_calls_are_captured_too():
    """hipporag2 and mem0 use OpenAI embedders, which were equally invisible."""
    setup_module()
    tracker = _fresh_tracker()
    with T.phase(T.BUILD):
        _client(EMBED_BODY).embeddings.create(
            model="text-embedding-3-small", input="hello")
    entry = tracker.summary()["by_model_phase"]["text-embedding-3-small"]["build"]
    assert entry["calls"] == 1 and entry["prompt_tokens"] == 17


def test_our_own_client_is_not_double_counted():
    """common.llm reports its own usage. If the shim also counted it, every
    QA and judge call would be doubled."""
    setup_module()
    tracker = _fresh_tracker()
    client = _client(CHAT_BODY)
    setattr(client, openai_usage.OWNED_CLIENT_ATTR, True)
    with T.phase(T.ANSWER):
        client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    assert tracker.summary()["totals"]["calls"] == 0, "shim double-counted"


def test_common_llm_tags_its_shared_client():
    """The tag has to actually be set on the client common.llm builds — a
    silent rename here would reintroduce double counting."""
    from common import llm as llm_mod

    async def go():
        return llm_mod._get_async_client("openai")

    client = asyncio.run(go())
    assert getattr(client, openai_usage.OWNED_CLIENT_ATTR, False) is True


def test_install_is_idempotent():
    """memo.py files call install() unconditionally; a second call must not
    wrap the wrapper (which would count every call twice)."""
    setup_module()
    assert openai_usage.install() is False        # already installed
    tracker = _fresh_tracker()
    with T.phase(T.BUILD):
        _client(CHAT_BODY).chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    assert tracker.summary()["totals"]["calls"] == 1


def test_phase_attribution_flows_into_vendored_calls():
    """The whole point: the same vendored client, called from two different
    workflow phases, lands in two different buckets."""
    setup_module()
    tracker = _fresh_tracker()
    client = _client(CHAT_BODY)
    with T.phase(T.BUILD):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "a"}])
    with T.phase(T.RETRIEVE):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "b"}])
    by_phase = tracker.summary()["by_phase"]
    assert by_phase["build"]["calls"] == 1 and by_phase["retrieve"]["calls"] == 1


def test_recording_failure_cannot_break_the_caller():
    """Accounting is telemetry: if it throws, the baseline's eval must not die."""
    setup_module()

    class _Exploding:
        def update(self, **kwargs):
            raise RuntimeError("tracker is broken")

    old = T.GLOBAL_TOKEN_TRACKER
    T.GLOBAL_TOKEN_TRACKER = _Exploding()
    try:
        resp = _client(CHAT_BODY).chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
        assert resp.choices[0].message.content == "hi"
    finally:
        T.GLOBAL_TOKEN_TRACKER = old


def test_no_tracker_installed_is_harmless():
    setup_module()
    old = T.GLOBAL_TOKEN_TRACKER
    T.GLOBAL_TOKEN_TRACKER = None
    try:
        resp = _client(CHAT_BODY).chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
        assert resp.usage.total_tokens == 128
    finally:
        T.GLOBAL_TOKEN_TRACKER = old


# ---------------------------------------------------------------------------
# The invariant this whole design exists to satisfy
# ---------------------------------------------------------------------------

BASELINES = ("amem", "hipporag2", "lightmem", "mem0", "memoryos", "simplemem", "zep")


def test_every_baseline_installs_the_shim():
    """A baseline that forgets this reports 0 build tokens and looks free."""
    missing = []
    for name in BASELINES:
        memo = PROJECT_ROOT / "baselines" / "harness" / name / "memo.py"
        src = memo.read_text(encoding="utf-8")
        if "_install_openai_usage()" not in src:
            missing.append(name)
    assert not missing, f"memo.py does not install the usage shim: {missing}"


def test_no_vendored_source_was_edited_to_report_usage():
    """Vendored `src/` is byte-identical to upstream and each README ships a
    `diff -r` asserting it. Usage tracking must never be wired in by editing
    those files — if this fails, someone took the shortcut the shim exists to
    avoid."""
    offenders = []
    for name in BASELINES:
        src_dir = PROJECT_ROOT / "baselines" / "harness" / name / "src"
        if not src_dir.is_dir():
            continue           # hipporag2 / mem0 use venv-installed packages
        for path in src_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "common.tokens" in text or "common.openai_usage" in text:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, (
        "vendored source references memevol's accounting modules, breaking "
        f"byte-identity: {offenders}"
    )


# ---------------------------------------------------------------------------

def _main():
    failures = 0
    setup_module()
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:
            failures += 1
            import traceback
            print(f"  FAIL {name}: {exc!r}")
            traceback.print_exc()
    print("ALL PASS" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
