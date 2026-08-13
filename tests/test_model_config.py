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
from common.config import load_config_file, validate_exact_config  # noqa: E402

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
    given = {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 1000}
    assert mc.normalise_chat_params(given) == given


def test_normalise_drops_sampling_params_for_reasoning_models():
    out = mc.normalise_chat_params({
        "model": "gpt-5-mini", "temperature": 0.1, "top_p": 0.9,
        "presence_penalty": 0.5, "frequency_penalty": 0.5,
    })
    for dropped in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        assert dropped not in out, dropped
    assert out["model"] == "gpt-5-mini"


def test_normalise_renames_max_tokens_for_reasoning_models():
    out = mc.normalise_chat_params({"model": "gpt-5-mini", "max_tokens": 1000})
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 1000


def test_normalise_does_not_clobber_an_explicit_max_completion_tokens():
    out = mc.normalise_chat_params(
        {"model": "o3", "max_tokens": 10, "max_completion_tokens": 99})
    assert out["max_completion_tokens"] == 99
    assert "max_tokens" not in out


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
    assert sent["max_completion_tokens"] == 1000
    assert sent["model"] == "gpt-5-mini"


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

def _required_keys(run_py: Path):
    """REQUIRED_KEYS out of a run.py without importing it (memo.py pulls heavy deps)."""
    for node in ast.walk(ast.parse(run_py.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "REQUIRED_KEYS" for t in node.targets
        ):
            return frozenset(ast.literal_eval(node.value.args[0]))
    raise AssertionError(f"no REQUIRED_KEYS in {run_py}")


def _baseline_dirs():
    return sorted(d for d in HARNESS_DIR.iterdir()
                  if d.is_dir() and (d / "run.py").exists())


def _config_files(d: Path):
    """Every config a human is told to pass to this baseline's run.py."""
    return sorted(list(d.glob("config.*.yaml")) + list(d.glob("smoke_*.yaml")))


def test_every_baseline_config_matches_its_required_keys():
    # Guards the whole surface at once: a new model key added to run.py but
    # forgotten in any config (or vice versa) fails here.
    #
    # The smoke_*.yaml files are included deliberately. They were written against
    # the old `strict_config: false` + DEFAULT_CONFIG layering, and when exact
    # config replaced it they silently became unloadable — every one of them
    # aborted before running anything, while three READMEs still told you to run
    # them. Nothing caught it because nothing validated them.
    for d in _baseline_dirs():
        keys = _required_keys(d / "run.py")
        for cfg in _config_files(d):
            validate_exact_config(load_config_file(cfg) or {}, keys,
                                  context=f"{cfg.parent.name}/{cfg.name}")


def test_no_config_carries_the_removed_strict_config_knob():
    # `strict_config` was removed when completeness became unconditional; a
    # config still carrying it is one that predates the change.
    for d in _baseline_dirs():
        for cfg in _config_files(d):
            assert "strict_config" not in (load_config_file(cfg) or {}), cfg.name


def test_every_config_a_readme_points_at_exists():
    """A README naming a config that isn't there is a broken instruction —
    simplemem's README documented `smoke_locomo.yaml` before the file existed."""
    import re

    missing = []
    for d in _baseline_dirs():
        readme = d / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8", errors="ignore")
        for name in set(re.findall(r"--config (\S+\.yaml)", text)):
            base = Path(name).name
            # Only SHIPPED configs are checked. A README may also walk you
            # through creating your own (hipporag2's `my_hr.yaml`), and those
            # are supposed to be absent.
            if not (base.startswith("config.") or base.startswith("smoke_")):
                continue
            if not (d / base).exists():
                missing.append(f"{d.name}/README.md -> {name}")
    assert not missing, f"README points at configs that do not exist: {missing}"


def test_every_baseline_ships_a_unified_preset():
    for d in _baseline_dirs():
        assert (d / "config.unified.yaml").exists(), f"{d.name} has no unified arm"


def test_unified_arm_is_one_llm_and_one_embedder_everywhere():
    """The point of the unified arm: everything except the memory design is
    held fixed, so a difference in score is a difference in method."""
    llm_keys = {"amem_llm_model", "hipporag2_llm_model", "lightmem_llm_model",
                "mem0_llm_model", "memoryos_llm_model", "simplemem_llm_model",
                "graph_llm_model", "llm_model", "judge_model"}
    embedder_keys = {"amem_embedding_model", "memoryos_embedding_model",
                     "embedding_model", "embedder_model", "embedding"}
    for d in _baseline_dirs():
        cfg = load_config_file(d / "config.unified.yaml") or {}
        for key in llm_keys & set(cfg):
            assert cfg[key] == "gpt-5-mini", f"{d.name}: {key}={cfg[key]!r}"
        for key in embedder_keys & set(cfg):
            assert cfg[key] == "text-embedding-3-small", f"{d.name}: {key}={cfg[key]!r}"


def test_unified_lightmem_moves_its_dimension_with_the_embedder():
    # lightmem is the one baseline carrying an explicit dims knob: it sizes the
    # Qdrant collection AND is sent as the API `dimensions` parameter, so a
    # stale 384 against a 1536-dim embedder is a hard failure.
    cfg = load_config_file(HARNESS_DIR / "lightmem" / "config.unified.yaml") or {}
    assert cfg["embedding_model"] == "text-embedding-3-small"
    assert cfg["embedding_dims"] == 1536


def test_no_config_hardcodes_cuda():
    # The regression this replaces: lightmem (x2) and zep (x1) defaulted to
    # "cuda" and crashed outright on a CPU-only box.
    for d in _baseline_dirs():
        for cfg_path in sorted(d.glob("config.*.yaml")):
            cfg = load_config_file(cfg_path) or {}
            for key in ("device", "embedding_device", "llmlingua_device"):
                if key in cfg:
                    assert cfg[key] is None, f"{cfg_path.name}: {key}={cfg[key]!r}"


def test_src_is_untouched_by_the_configurable_model_layer():
    """Byte-identity guard. Every lever in model_config.py acts at a boundary
    OUTSIDE the vendored source, so no `src/` tree may import it.

    Matches the IMPORT specifically, not the bare name — pydantic v2 models use
    a `model_config` attribute all over the vendored trees.
    """
    needle = "baselines.harness.model_config"
    offenders = [str(p.relative_to(PROJECT_ROOT))
                 for d in _baseline_dirs() for p in (d / "src").rglob("*.py")
                 if needle in p.read_text(encoding="utf-8", errors="ignore")]
    assert not offenders, f"vendored source imports model_config: {offenders}"


# -------------------- runner --------------------

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
