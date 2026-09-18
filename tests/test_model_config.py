"""Tests for the shared model-configuration layer (baselines/harness/model_config.py).

Covers the two levers issue #26 rests on — the embedder factory and the
OpenAI-SDK param normalisation — plus the config surface every baseline now
exposes. Zero-dependency runner (no pytest); needs only the ROOT project env
(the openai SDK; sentence-transformers is faked, never imported for real):

    uv run python tests/test_model_config.py
"""
import ast
import sys
import traceback
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.harness import model_config as mc  # noqa: E402

HARNESS_DIR = PROJECT_ROOT / "baselines" / "harness"


# ---------------- device resolution ----------------
#
# lightmem (x2) and zep (x1) used to default to a hardcoded "cuda" and crashed
# outright on a CPU-only box.

def test_resolve_device_passes_explicit_values_through():
    for pinned in ("cpu", "cuda", "cuda:1", "mps"):
        assert mc.resolve_device(pinned) == pinned


def test_resolve_device_auto_forms_pick_an_available_device():
    for auto in (None, "", "auto"):
        assert mc.resolve_device(auto) in ("cpu", "cuda")


# ---------------- API vs local embedding models ----------------

def test_is_api_embedding_model():
    assert mc.is_api_embedding_model("text-embedding-3-small")
    assert mc.is_api_embedding_model("text-embedding-3-large")
    assert not mc.is_api_embedding_model("all-MiniLM-L6-v2")
    assert not mc.is_api_embedding_model("Qwen/Qwen3-Embedding-0.6B")
    assert not mc.is_api_embedding_model("BAAI/bge-m3")
    assert not mc.is_api_embedding_model(None)


# ---------------- param normalisation ----------------
#
# amem/lightmem/simplemem/memoryos hardcode temperature+max_tokens and zep's
# graphiti still sends max_tokens — all rejected by the gpt-5 family. Without
# this rewrite the unified arm cannot run at all on 5 of the 7 baselines.

def test_normalise_leaves_4_series_untouched():
    """Nothing model-specific is rewritten for a 4-series call. The stall
    bound is added for every model, so it is the one permitted addition."""
    given = {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 1000}
    out = mc.normalise_chat_params(given)
    assert {k: out[k] for k in given} == given
    assert set(out) - set(given) == {"timeout"}


def test_normalise_drops_sampling_params_for_reasoning_models():
    out = mc.normalise_chat_params({
        "model": "gpt-5-mini", "temperature": 0.1, "top_p": 0.9,
        "presence_penalty": 0.5, "frequency_penalty": 0.5,
    })
    for dropped in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        assert dropped not in out, dropped
    assert out["model"] == "gpt-5-mini"


def test_normalise_bounds_a_call_that_set_no_timeout():
    """The vendored clients build their own OpenAI() with no timeout, so they
    inherit the SDK's 600 s read timeout and 2 retries: one request the server
    never answers costs 1800 s and fails the user. Measured twice, exactly."""
    out = mc.normalise_chat_params({"model": "gpt-5-mini", "messages": []})
    assert out["timeout"] == mc.VENDORED_REQUEST_TIMEOUT_SECONDS
    # Not a reasoning-model-only rewrite — a stall is model-agnostic.
    out = mc.normalise_chat_params({"model": "gpt-4.1", "messages": []})
    assert out["timeout"] == mc.VENDORED_REQUEST_TIMEOUT_SECONDS


def test_normalise_leaves_a_caller_supplied_timeout_alone():
    """A caller that thought about its own deadline keeps it."""
    out = mc.normalise_chat_params({"model": "gpt-5-mini", "messages": [], "timeout": 30})
    assert out["timeout"] == 30


def test_normalise_drops_max_tokens_for_reasoning_models():
    """It used to be renamed to max_completion_tokens. On a reasoning model
    that cap covers reasoning AND output, so a value chosen for a 4-series
    model (mem0 sends 2000) leaves nothing for the answer: measured against
    gpt-5-mini with a 30k-char json_object prompt, the capped request never
    returned (300 s), the uncapped one answered in 4.2 s."""
    out = mc.normalise_chat_params({"model": "gpt-5-mini", "max_tokens": 1000})
    assert "max_tokens" not in out
    assert "max_completion_tokens" not in out, "a 4-series cap is not a reasoning cap"


def test_normalise_raises_a_starving_completion_cap():
    """mem0 maps its own max_tokens=2000 to max_completion_tokens for the GPT-5
    family. That budget covers REASONING too: measured on gpt-5-mini, an
    extraction call spends all 2000 thinking and returns empty content, so the
    memory system stores nothing and every answer is a guess."""
    out = mc.normalise_chat_params(
        {"model": "o3", "max_tokens": 10, "max_completion_tokens": 99})
    assert out["max_completion_tokens"] == mc.REASONING_MIN_COMPLETION_TOKENS
    assert "max_tokens" not in out


def test_normalise_keeps_a_cap_that_already_has_room():
    out = mc.normalise_chat_params({"model": "gpt-5-mini", "max_completion_tokens": 40000})
    assert out["max_completion_tokens"] == 40000, "a caller asking for more keeps it"


def test_normalise_adds_no_cap_where_there_was_none():
    out = mc.normalise_chat_params({"model": "gpt-5-mini", "messages": []})
    assert "max_completion_tokens" not in out


def test_normalise_leaves_the_four_series_alone():
    out = mc.normalise_chat_params(
        {"model": "gpt-4o-mini", "max_tokens": 1000, "temperature": 0.7})
    assert out["max_tokens"] == 1000 and out["temperature"] == 0.7


def test_normalise_splits_the_effort_suffix():
    # "model/effort" is a repo-wide convention (see common/llm.py); vendored
    # clients pass the configured string straight through and would 400 on it.
    out = mc.normalise_chat_params({"model": "gpt-5-mini/low", "temperature": 0.3})
    assert out["model"] == "gpt-5-mini"
    assert out["reasoning_effort"] == "low"
    assert "temperature" not in out


def test_normalise_effort_suffix_on_a_non_reasoning_model_degrades_gracefully():
    out = mc.normalise_chat_params({"model": "gpt-4.1/low", "temperature": 0.3})
    assert out["model"] == "gpt-4.1"
    assert "reasoning_effort" not in out   # 4-series would reject it
    assert out["temperature"] == 0.3       # and still accepts temperature


def test_normalise_returns_a_copy():
    given = {"model": "gpt-5-mini", "temperature": 0.1}
    mc.normalise_chat_params(given)
    assert given["temperature"] == 0.1


def test_sdk_patch_rewrites_a_real_create_call():
    """End-to-end through the genuine SDK method, with only the transport faked.

    This is the assertion that matters: patching `normalise_chat_params` alone
    proves nothing if the interception point is wrong.
    """
    mc.install_openai_param_normalisation()
    from openai.resources.chat.completions import Completions

    sent = {}

    class _FakeResource:
        def _post(self, _path, *, body, **_kw):
            sent.update(body)
            return "ok"

    assert Completions.create(
        _FakeResource(), model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7, max_tokens=1000,
    ) == "ok"
    assert "temperature" not in sent
    assert "max_tokens" not in sent
    assert "max_completion_tokens" not in sent, "a 4-series cap is not carried over"
    assert sent["model"] == "gpt-5-mini"


def test_the_sdk_patch_caps_how_many_calls_are_in_flight():
    """Vendored clients have no concurrency gate of their own and several fan
    out hard (simplemem's packaged config runs 16 parallel workers). Measured
    with 16 in flight, calls that take seconds alone took 211 s — and then the
    request budget fires and they all retry."""
    import threading
    import time

    mc.install_openai_param_normalisation()
    from openai.resources.chat.completions import Completions

    live = [0]
    peak = [0]
    lock = threading.Lock()

    class _SlowResource:
        def _post(self, _path, *, body, **_kw):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1
            return "ok"

    def call():
        Completions.create(_SlowResource(), model="gpt-4.1-mini",
                           messages=[{"role": "user", "content": "hi"}])

    threads = [threading.Thread(target=call) for _ in range(mc.VENDORED_MAX_CONCURRENT * 4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert peak[0] <= mc.VENDORED_MAX_CONCURRENT, (
        f"{peak[0]} calls were in flight at once, gate is "
        f"{mc.VENDORED_MAX_CONCURRENT}")
    assert peak[0] > 1, "the gate must not serialise every call"


def test_sdk_patch_is_idempotent():
    from openai.resources.chat.completions import AsyncCompletions, Completions

    mc.install_openai_param_normalisation()          # ensure installed first
    before = (Completions.create, AsyncCompletions.create)
    mc.install_openai_param_normalisation()
    assert (Completions.create, AsyncCompletions.create) == before
    for cls in (Completions, AsyncCompletions):
        # A second install must not wrap the wrapper — that would normalise
        # twice and could re-rename an already-renamed param.
        assert not hasattr(cls.create._real_create, "_real_create")


# ---------------- embedder factory ----------------
#
# sentence-transformers is a heavy dependency that only the per-baseline uv
# projects carry, so stand in a fake module. The factory only ever needs the
# constructor.

class _FakeST:
    instances = 0

    def __init__(self, model_name_or_path=None, device=None, **kwargs):
        _FakeST.instances += 1
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.kwargs = kwargs

    def encode(self, sentences, **kwargs):
        return sentences


# The fake module below is installed into sys.modules. Under pytest the whole
# suite shares one process, so leaving it there makes OTHER files import a
# `sentence_transformers` whose `SentenceTransformer` is a plain function —
# tests/test_run_record.py then does `SentenceTransformer.__new__(...)` and dies
# with `TypeError: function.__new__(X)`. Restore whatever was (or was not) there.
try:
    import pytest

    @pytest.fixture(autouse=True, scope="module")
    def _restore_sentence_transformers():
        had = "sentence_transformers" in sys.modules
        prev = sys.modules.get("sentence_transformers")
        yield
        if had:
            sys.modules["sentence_transformers"] = prev
        else:
            sys.modules.pop("sentence_transformers", None)
        mc._model_cache.clear()
        mc._factory_installed = False
except ImportError:      # zero-dependency runner path — no pytest available
    pass


def _install_fake_sentence_transformers():
    """Reset model_config's global patch state and install a fake ST module."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeST
    sys.modules["sentence_transformers"] = module
    _FakeST.instances = 0
    mc._model_cache.clear()
    mc._factory_installed = False
    return module


def test_factory_memoizes_local_models_across_users():
    # A fresh MemoClass is built PER USER; without memoization the weights
    # reload for every conversation in the split.
    module = _install_fake_sentence_transformers()
    mc.install_embedder_factory()
    a = module.SentenceTransformer("all-MiniLM-L6-v2")
    b = module.SentenceTransformer("all-MiniLM-L6-v2")
    assert a is b
    assert _FakeST.instances == 1


def test_factory_serializes_encode_on_shared_local_models():
    # One local model is shared by every user, and users' hooks run on worker
    # threads — so the factory puts a per-model lock around `encode`, in place
    # (the object keeps its type for vendored isinstance checks).
    _install_fake_sentence_transformers()
    model = mc.get_embedder("all-MiniLM-L6-v2")
    assert isinstance(model, _FakeST)
    assert getattr(model.encode, "_serialized", False) is True
    assert model.encode(["a"]) == ["a"]
    assert not getattr(mc.get_embedder("text-embedding-3-small").encode, "_serialized", False), \
        "API embedders are network-bound and must not be serialized"


def test_concurrent_first_calls_load_a_local_model_once():
    # Users' hooks run on worker threads, so the first requests for a model can
    # arrive together; each must get the SAME object from a single load, not
    # one copy per thread (large local embedders would exhaust the GPU).
    import threading
    import time

    _install_fake_sentence_transformers()
    real_init = _FakeST.__init__

    def slow_init(self, *args, **kwargs):
        time.sleep(0.2)
        real_init(self, *args, **kwargs)

    _FakeST.__init__ = slow_init
    try:
        got = []
        threads = [threading.Thread(target=lambda: got.append(mc.get_embedder("all-MiniLM-L6-v2")))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert _FakeST.instances == 1, _FakeST.instances
        assert len(got) == 4 and all(g is got[0] for g in got)
    finally:
        _FakeST.__init__ = real_init


def test_factory_keys_distinct_models_separately():
    module = _install_fake_sentence_transformers()
    mc.install_embedder_factory()
    a = module.SentenceTransformer("all-MiniLM-L6-v2")
    b = module.SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    assert a is not b
    assert _FakeST.instances == 2


def test_factory_is_idempotent_and_keeps_the_real_class_reachable():
    module = _install_fake_sentence_transformers()
    mc.install_embedder_factory()
    patched = module.SentenceTransformer
    mc.install_embedder_factory()
    assert module.SentenceTransformer is patched
    assert patched._real_sentence_transformer is _FakeST


def test_factory_returns_an_api_embedder_for_an_api_model_name():
    # simplemem + amem reach the API arm this way: the configured name IS the
    # name the vendored code requests, so the factory dispatches on it.
    module = _install_fake_sentence_transformers()
    mc.install_embedder_factory()
    got = module.SentenceTransformer("text-embedding-3-small")
    assert isinstance(got, mc.APIEmbedder)
    assert got.model_name == "text-embedding-3-small"
    assert _FakeST.instances == 0          # no local weights were loaded
    assert got.get_sentence_embedding_dimension() == 1536
    assert got.get_config_dict() == {"model_name": "text-embedding-3-small"}


def test_get_embedder_does_not_recurse_through_the_installed_factory():
    """`get_embedder` must resolve the GENUINE class even after the constructor
    has been replaced by the factory that calls it — otherwise the two bounce
    off each other forever."""
    _install_fake_sentence_transformers()
    mc.install_embedder_factory()
    got = mc.get_embedder("all-MiniLM-L6-v2")
    assert isinstance(got, _FakeST)
    assert _FakeST.instances == 1


def test_get_embedder_works_without_the_factory_installed():
    """memoryos calls get_embedder() directly and never installs the patch."""
    _install_fake_sentence_transformers()
    assert not mc._factory_installed
    assert isinstance(mc.get_embedder("all-MiniLM-L6-v2"), _FakeST)
    assert isinstance(mc.get_embedder("text-embedding-3-small"), mc.APIEmbedder)


def test_factory_forwards_the_callers_constructor_kwargs():
    """simplemem's Qwen3 path passes trust_remote_code / model_kwargs /
    tokenizer_kwargs and lightmem passes model_kwargs. Dropping those would
    silently change how the model loads — a regression against the per-baseline
    caches this factory replaces, which forwarded *args/**kwargs verbatim."""
    module = _install_fake_sentence_transformers()
    mc.install_embedder_factory()
    got = module.SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"attn_implementation": "flash_attention_2"},
        tokenizer_kwargs={"padding_side": "left"},
        trust_remote_code=True,
    )
    assert got.kwargs == {
        "model_kwargs": {"attn_implementation": "flash_attention_2"},
        "tokenizer_kwargs": {"padding_side": "left"},
        "trust_remote_code": True,
    }
    assert got.model_name_or_path == "Qwen/Qwen3-Embedding-0.6B"


def test_factory_accepts_the_name_as_a_keyword():
    module = _install_fake_sentence_transformers()
    mc.install_embedder_factory()
    got = module.SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
    assert got.model_name_or_path == "all-MiniLM-L6-v2"
    assert module.SentenceTransformer("all-MiniLM-L6-v2") is got   # same cache slot


def test_async_patch_stays_a_coroutine_function():
    # The SDK's own AsyncCompletions.create is `async def`; callers and
    # inspect.iscoroutinefunction may rely on that.
    import inspect

    mc.install_openai_param_normalisation()
    from openai.resources.chat.completions import AsyncCompletions

    assert inspect.iscoroutinefunction(AsyncCompletions.create)


def test_factory_forwards_the_callers_device():
    # lightmem routes its resolved device here via `model_kwargs: {"device": ...}`.
    module = _install_fake_sentence_transformers()
    mc.install_embedder_factory()
    got = module.SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    assert got.device == "cpu"
    # device is part of the cache key — a different device is a different model
    assert module.SentenceTransformer("all-MiniLM-L6-v2", device="cuda") is not got


def test_api_embedder_encode_shapes(monkeypatched_embedding=None):
    """encode() must mirror sentence-transformers' shape convention: a single
    string in gives a 1-D vector, a list gives a 2-D (N, D) array."""
    embedder = mc.APIEmbedder.__new__(mc.APIEmbedder)   # skip the common.llm wiring
    embedder.model_name = "text-embedding-3-small"
    embedder._dim = 3
    embedder._embedding = lambda texts: [[3.0, 0.0, 4.0] for _ in texts]

    one = embedder.encode("hello")
    assert one.shape == (3,)

    many = embedder.encode(["a", "b"], convert_to_numpy=True)
    assert many.shape == (2, 3)

    # Unknown kwargs from vendored callers must not raise.
    assert embedder.encode(["a"], show_progress_bar=False, prompt_name="query").shape == (1, 3)

    normed = embedder.encode(["a"], normalize_embeddings=True)
    assert abs(float((normed[0] ** 2).sum()) - 1.0) < 1e-6

    assert embedder.encode([]).shape[0] == 0


# ---------------- the config surface ----------------
#
# Each memo.py declares its method config as MODULE-LEVEL data —
# CONFIG_DEFAULTS (the faithful arm) and UNIFIED_MODEL_KEYS (where `arm: unified`
# writes unified_models) — and eval_harness.py resolves it. Read out of memo.py
# by AST so no baseline's heavy deps are needed.

EXAMPLE_UNIFIED = {"llm": "gpt-5-mini", "embedding": "text-embedding-3-small"}


def _module_data(memo_py: Path):
    """(CONFIG_DEFAULTS, UNIFIED_MODEL_KEYS) literals from memo.py's MODULE level."""
    found = {}
    for node in ast.parse(memo_py.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) in ("CONFIG_DEFAULTS", "UNIFIED_MODEL_KEYS"):
                    found[t.id] = ast.literal_eval(node.value)
    assert set(found) == {"CONFIG_DEFAULTS", "UNIFIED_MODEL_KEYS"}, f"{memo_py}: {sorted(found)}"
    return found["CONFIG_DEFAULTS"], found["UNIFIED_MODEL_KEYS"]


def _baseline_dirs():
    from baselines.harness.eval_harness import MEMOS
    return sorted(HARNESS_DIR / name for name in MEMOS)


def test_registry_names_every_baseline_dir_and_nothing_else():
    from baselines.harness.eval_harness import MEMOS
    on_disk = {d.name for d in HARNESS_DIR.iterdir() if (d / "memo.py").exists()}
    assert on_disk == set(MEMOS), (on_disk, set(MEMOS))
    for name, spec in MEMOS.items():
        assert spec.startswith(f"baselines.harness.{name}.memo:"), spec


def test_no_baseline_ships_a_runner_or_its_own_config():
    # A harness dir provides only its MemoClass.
    # (config.paper.yaml is the record of a published reproduction — allowed.)
    for d in _baseline_dirs():
        assert not (d / "run.py").exists(), d.name
        for stray in ("config.example.yaml", "config.unified.yaml"):
            assert not (d / stray).exists(), f"{d.name}/{stray}"


def test_memo_classes_carry_no_config_machinery():
    """The class only reads self.config: no defaults or resolution on it."""
    for d in _baseline_dirs():
        for node in ast.walk(ast.parse((d / "memo.py").read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                names = {t.id for stmt in node.body if isinstance(stmt, ast.Assign)
                         for t in stmt.targets if isinstance(t, ast.Name)}
                names |= {stmt.name for stmt in node.body if isinstance(stmt, ast.FunctionDef)}
                leaked = names & {"CONFIG_DEFAULTS", "UNIFIED_OVERRIDES", "UNIFIED_MODEL_KEYS", "resolve_config"}
                assert not leaked, f"{d.name}.{node.name}: {sorted(leaked)}"


def test_no_memo_py_touches_the_environment():
    """Configuration is explicit: integration code never reads or writes a
    setting through os.environ / os.getenv (credentials are left to the SDKs)."""
    import re
    pattern = re.compile(r"os\.(environ|getenv|putenv)")
    for path in [*(d / "memo.py" for d in _baseline_dirs()), HARNESS_DIR / "model_config.py"]:
        hits = [f"{path.parent.name}/{path.name}:{i}" for i, line in
                enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                if pattern.search(line) and not line.lstrip().startswith("#")]
        assert not hits, hits


def test_unified_model_keys_are_declared_defaults_with_known_roles():
    from baselines.harness.eval_harness import MODEL_ROLES
    for d in _baseline_dirs():
        defaults, model_keys = _module_data(d / "memo.py")
        assert set(model_keys) <= MODEL_ROLES, f"{d.name}: {sorted(model_keys)}"
        if not defaults and not model_keys:
            continue        # no_memory: calls no model, so there is none to unify
        assert {"llm", "embedding"} <= set(model_keys), f"{d.name}: must map both models"
        keys = {k for ks in model_keys.values() for k in ks}
        assert keys <= set(defaults), f"{d.name}: undeclared {sorted(keys - set(defaults))}"


def test_unified_arm_is_one_llm_and_one_embedder_and_nothing_else():
    """Switching arms must change the MODELS and nothing else, or a score
    difference stops being attributable to the model swap."""
    from baselines.harness.eval_harness import resolve_memo_config
    for d in _baseline_dirs():
        defaults, model_keys = _module_data(d / "memo.py")
        if not defaults and not model_keys:
            continue        # no_memory: no models, so no arm difference to check
        faithful = resolve_memo_config(defaults, model_keys, arm="faithful")
        unified = resolve_memo_config(defaults, model_keys, arm="unified", unified_models=EXAMPLE_UNIFIED)
        for key in model_keys["llm"]:
            assert unified[key] == "gpt-5-mini", f"{d.name}: {key}"
        for key in model_keys["embedding"]:
            assert unified[key] == "text-embedding-3-small", f"{d.name}: {key}"
        controlled = {k for ks in model_keys.values() for k in ks}
        drift = {k for k in defaults if k not in controlled and faithful[k] != unified[k]}
        assert not drift, f"{d.name}: non-model keys differ between arms: {drift}"


def test_unified_lightmem_moves_its_dimension_with_the_embedder():
    # lightmem is the one baseline carrying an explicit dims knob: it sizes the
    # Qdrant collection AND is sent as the API `dimensions` parameter, so a
    # stale 384 against a 1536-dim embedder is a hard failure.
    from baselines.harness.eval_harness import resolve_memo_config
    defaults, model_keys = _module_data(HARNESS_DIR / "lightmem" / "memo.py")
    unified = resolve_memo_config(defaults, model_keys, arm="unified", unified_models=EXAMPLE_UNIFIED)
    assert unified["embedding_model"] == "text-embedding-3-small"
    assert unified["embedding_dims"] == 1536


def test_api_embedding_dims_knows_the_openai_models_and_refuses_the_rest():
    assert mc.api_embedding_dims("text-embedding-3-small") == 1536
    assert mc.api_embedding_dims("text-embedding-3-large") == 3072
    try:
        mc.api_embedding_dims("text-embedding-9")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown model")


def test_no_default_hardcodes_cuda():
    # The regression this replaces: lightmem (x2) and zep (x1) defaulted to
    # "cuda" and crashed outright on a CPU-only box.
    for d in _baseline_dirs():
        defaults, _ = _module_data(d / "memo.py")
        for key in ("device", "embedding_device", "llmlingua_device"):
            if key in defaults:
                assert defaults[key] is None, f"{d.name}: {key}={defaults[key]!r}"


def test_every_shipped_config_a_readme_points_at_exists():
    """A README naming a SHIPPED config that isn't there is a broken instruction."""
    import re

    missing = []
    for readme in [HARNESS_DIR / "README.md", *(d / "README.md" for d in _baseline_dirs())]:
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8", errors="ignore")
        for name in set(re.findall(r"--config (\S+\.yaml)", text)):
            if not Path(name).name.startswith("config."):
                continue
            if not (PROJECT_ROOT / name).exists() and not (readme.parent / Path(name).name).exists():
                missing.append(f"{readme.parent.name}/README.md -> {name}")
    assert not missing, f"README points at configs that do not exist: {missing}"


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
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
