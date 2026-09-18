"""Package a harness baseline as a forge harness, so a search can seed from it.

A harness baseline and a forge harness are the same contract reached two
different ways. `eval_harness` imports `baselines.harness.<name>.memo` from the
repo, in that baseline's own venv, and resolves its config from `arm` /
`unified_models`. forge instead runs ONE directory inside a container that sees
neither `baselines/` nor any of those venvs: `harness.py`, whatever it can
import from beside itself, and `requirements.txt` installed on top of the eval
base image.

This tool bridges the two by making the baseline self-contained::

    <out>/harness.py                     the class forge loads (config baked in)
    <out>/requirements.txt               pinned from the baseline's uv.lock
    <out>/meta.json                      description, parent_ids
    <out>/baselines/harness/model_config.py, concurrency.py
    <out>/baselines/harness/<name>/memo.py, src/…

The copied tree keeps the REAL package paths (`baselines.harness.<name>.memo`),
so memo.py's own imports work untouched — no rewriting, no import shims, and
the vendored `src/` stays byte-identical. `sys.path` in the container already
includes the harness dir, and `baselines/` is not mounted there, so nothing
can shadow anything.

The resolved method config is written INTO harness.py rather than resolved at
run time: the container has no `arm`/`unified_models` machinery, and a seed
that carries its own numbers should carry the exact settings that produced
them.

Run it in the baseline's own venv — reading `CONFIG_DEFAULTS` means importing
memo.py, which imports that baseline's dependencies::

    uv run --project baselines/harness/mem0 python -m tools.package_baseline \\
        mem0 --arm unified --out seeds_src/mem0_unified

Then seed a search from the result::

    ... python -m forge.orchestrator --config my.yaml --seed seeds_src/mem0_unified

Why the unified LLM defaults to a 4-series model
-----------------------------------------------
These methods were written against 4-series models: they batch large prompts
(mem0 hands its extractor a 33.6 k-char system prompt plus a conversation
session), several fan out hard (simplemem runs 16 parallel workers), and they
carry their own client timeouts. Pointed at gpt-5-mini the same call measured
15-23 s at low effort against 2-3 s on a 4-series model, emitted ~5x the
completion tokens — which lands straight in the efficiency metric — and
repeatedly hit windows where it did not return at all, with 150 s, 180 s and
900 s budgets failing alike while a small request to the same endpoint
answered in 2 s. LightMem and SimpleMem never completed a single build that
way.

Between the 4-series models, measured 20 rounds alternating (the endpoint
drifts by the hour, so the two models take turns) with SDK retries off, on
mem0's real extraction call: gpt-4.1-mini p50 2.4 s / p90 4.6 s, gpt-4o-mini
p50 7.5 s / p90 11.6 s, both 19/19 once warm. The one difference in failures
was the first call of a session, which gpt-4o-mini lost twice out of two
observations — and a cold failure is not free here: LightMem swallows a failed
call as `usage: None` and then dies somewhere unrelated.

Pass `--unified-llm gpt-5-mini/low` (or any "model/effort" string) to use a
reasoning model anyway; `model_config.normalise_chat_params` splits the suffix
into `reasoning_effort` before the SDK sees it.

What this does NOT do is make every baseline fit the container: lightmem pins
`transformers<5` and hipporag2 pins `torch==2.5.1` against a base image built
with torch 2.6, so their requirements.txt installs a second multi-GB stack on
top. mem0 (no local models, ~280 MB of deps) is the one that fits comfortably;
the rest are why the base image needs revisiting.
"""
from __future__ import annotations

import argparse
import re
import datetime as _dt
import json
import pprint
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forge.paths import ENTRY_FILE, IMPL_DIR

HARNESS_DIR = PROJECT_ROOT / "baselines" / "harness"

#: Modules from `baselines/harness/` a memo.py may import: the embedder/param
#: shims it installs, the concurrency helpers its hooks use, and the shared
#: app-log rendering. Copied rather than imported — `baselines/` is not
#: mounted in the container.
_SHARED_MODULES = ("model_config.py", "concurrency.py", "passages.py")

#: What the baseline's own adapter is called once it is inside `src/`. Not
#: `memo.py`: that name belongs to the interface at the root, and two files
#: with one name in one tree is exactly the confusion this layout removes.
_ADAPTER_MODULE = "adapter"

#: `baselines.harness.X` and `baselines.harness.<name>.src.X` both become a
#: plain `X` once everything is flat inside `src/`. These are absolute package
#: imports in the repo — the only thing the old nested layout was buying.
_IMPORT_REWRITES = (
    (re.compile(r"\bbaselines\.harness\.[A-Za-z_][A-Za-z0-9_]*\.src\."), ""),
    (re.compile(r"\bbaselines\.harness\."), ""),
)

#: The adapters find their vendored package at `<their dir>/src`. Flattened,
#: the package sits beside the adapter instead.
_SRC_REWRITE = ('Path(__file__).resolve().parent / "src"',
                'Path(__file__).resolve().parent')


def _flatten_imports(source: str) -> str:
    """Rewrite one module for the flat `src/` layout."""
    for pattern, replacement in _IMPORT_REWRITES:
        source = pattern.sub(replacement, source)
    return source.replace(*_SRC_REWRITE)


def _write_flattened(source: Path, dest: Path) -> None:
    dest.write_text(_flatten_imports(source.read_text(encoding="utf-8")),
                    encoding="utf-8")



def _export_requirements(project_dir: Path) -> str:
    """Pinned requirements for the container, from the baseline's uv.lock.

    Exported by `uv` rather than read out of the lock by hand. The lock holds
    every platform's resolution, so a naive read pins Windows-only packages
    (portalocker pulls `pywin32`) as if they were needed, and the image build
    dies on them. `uv export` writes each pin with its environment marker —
    `pywin32==312 ; sys_platform == 'win32'` — which pip then skips on Linux.
    """
    cmd = [
        "uv", "export", "--frozen", "--no-hashes", "--no-emit-project",
        "--no-dev", "--project", str(project_dir),
    ]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise SystemExit(
            "packaging needs `uv` on PATH: the baseline's pinned dependencies "
            "come from `uv export` (reading uv.lock by hand pins Windows-only "
            "wheels that break the image build). Run this on a machine with uv "
            "— the same one that has the baseline's venv."
        )
    if done.returncode != 0:
        raise SystemExit(f"`uv export` failed for {project_dir}:\n{done.stderr.strip()}")
    return done.stdout


def _resolved_config(name: str, arm: str, unified_models: Dict[str, str] | None) -> Dict[str, Any]:
    """The method config `eval_harness` would hand this baseline."""
    from baselines.harness.eval_harness import load_memo, resolve_memo_config

    load_memo(name)                       # import errors surface here, clearly
    module = sys.modules[f"baselines.harness.{name}.memo"]
    return resolve_memo_config(
        module.CONFIG_DEFAULTS, module.UNIFIED_MODEL_KEYS,
        arm=arm, unified_models=unified_models,
    )


def _memo_py(name: str, class_name: str, arm: str, config: Dict[str, Any]) -> str:
    # A Python literal, not JSON: `null`/`true` would not parse, and the entry
    # file is imported, not loaded as data.
    pretty = pprint.pformat(config, indent=4, width=84, sort_dicts=True)
    impl_dir, adapter = IMPL_DIR, _ADAPTER_MODULE
    return f'''"""{name} as a forge harness (arm: {arm}).

Generated by tools/package_baseline.py on {_dt.date.today().isoformat()} — do
not edit by hand; re-package instead. This is the harness interface; the
method itself — adapter, shared helpers, vendored package — is flat under
`src/` beside it, the same shape an evolved candidate has.

CONFIG is what `eval_harness` resolved for arm={arm!r}: baked in because the
container has no arm/unified_models machinery, and because a seed should carry
the exact settings its numbers came from.
"""
import sys
import tempfile
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "{impl_dir}"
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))

import {adapter} as _memo   # noqa: E402
from {adapter} import {class_name}   # noqa: E402

CONFIG = {pretty}


def _redirect_writable_paths() -> None:
    """Send the baseline's on-disk store somewhere writable.

    Five of the baselines keep their per-user store (Qdrant, LanceDB, FAISS,
    JSON) in an `outputs/` directory NEXT TO THEIR OWN SOURCE — fine in the
    repo, impossible here: forge mounts the harness read-only. The container's
    /tmp is private to it (`singularity --containall`) and discarded with it,
    which is exactly the lifetime a per-run store wants.

    `OUTPUTS_DIR` is read at call time by every baseline that has one, so
    rebinding it is enough — except where the path was already baked into a
    module-level setting at import (simplemem's LANCEDB_PATH), which is why
    the string rewrite below exists too.
    """
    old = getattr(_memo, "OUTPUTS_DIR", None)
    if old is None:
        return
    new = Path(tempfile.mkdtemp(prefix="memevol-{name}-"))
    _memo.OUTPUTS_DIR = new
    for value in vars(_memo).values():
        if not isinstance(value, dict):
            continue
        for key, entry in list(value.items()):
            if isinstance(entry, str) and entry.startswith(str(old)):
                value[key] = entry.replace(str(old), str(new), 1)


_redirect_writable_paths()


class PackagedHarness({class_name}):
    """The baseline, with its resolved config supplied by default.

    forge constructs harnesses with no arguments; `eval_harness` passes a
    config. Defaulting rather than overriding keeps both callers working.
    """

    def __init__(self, config=None):
        super().__init__(config if config is not None else dict(CONFIG))
'''


def package(name: str, arm: str, unified_models: Dict[str, str] | None,
            out: Path) -> Path:
    src = HARNESS_DIR / name
    if not (src / "memo.py").exists():
        raise SystemExit(f"no such baseline: {name} (looked in {src})")

    config = _resolved_config(name, arm, unified_models)
    module = sys.modules[f"baselines.harness.{name}.memo"]
    from common.memo_select import select_memo_class
    class_name = select_memo_class(
        [obj for obj in vars(module).values() if isinstance(obj, type)
         and getattr(obj, "__module__", "") == module.__name__],
        str(src / "memo.py"),
    ).__name__

    if out.exists():
        shutil.rmtree(out)
    impl = out / IMPL_DIR
    impl.mkdir(parents=True)

    # Everything the method needs goes into ONE fixed subdirectory, flat: the
    # adapter, the three shared helpers, and the vendored package. A packaged
    # baseline then has the same shape as an evolved candidate — `memo.py`
    # beside `src/` — so forge, the proposer and a reader treat them alike.
    for module_name in _SHARED_MODULES:
        source = HARNESS_DIR / module_name
        if source.exists():
            _write_flattened(source, impl / module_name)
    _write_flattened(src / "memo.py", impl / f"{_ADAPTER_MODULE}.py")
    if (src / IMPL_DIR).exists():
        for child in sorted((src / IMPL_DIR).iterdir()):
            if child.name == "__pycache__":
                continue
            (shutil.copytree if child.is_dir() else shutil.copy2)(child, impl / child.name)

    (out / ENTRY_FILE).write_text(_memo_py(name, class_name, arm, config),
                                  encoding="utf-8")

    if (src / "uv.lock").exists():
        (out / "requirements.txt").write_text(
            f"# Exported from baselines/harness/{name}/uv.lock — the exact set "
            f"this baseline was tested with. Pins carry their environment "
            f"markers; pip applies them.\n"
            + _export_requirements(src),
            encoding="utf-8")

    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "parent_ids": [],
            "description": f"{name} harness baseline (arm: {arm}), packaged for forge",
            "packaged_from": f"baselines/harness/{name}",
            "arm": arm,
            "unified_models": unified_models,
            "packaged_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2, ensure_ascii=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline", help="a name under baselines/harness/ (e.g. mem0)")
    p.add_argument("--arm", default="faithful", choices=["faithful", "unified"])
    p.add_argument("--unified-llm", default="gpt-4.1-mini",
                   help="arm=unified only: the internal LLM every baseline "
                        "uses. A 4-series model ON PURPOSE — see the module "
                        "docstring. Accepts the repo's \"model/effort\" form.")
    p.add_argument("--unified-embedding", default="text-embedding-3-small",
                   help="arm=unified only: the embedder (must be an API model)")
    p.add_argument("--out", type=Path, required=True,
                   help="directory to write the packaged harness into (replaced)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    unified = ({"llm": args.unified_llm, "embedding": args.unified_embedding}
               if args.arm == "unified" else None)
    out = package(args.baseline, args.arm, unified, args.out.resolve())
    print(f"packaged {args.baseline} (arm={args.arm}) → {out}")
    print(f"  seed a search with:  --seed {out}")


if __name__ == "__main__":
    main()
