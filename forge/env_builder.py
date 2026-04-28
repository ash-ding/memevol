"""Per-harness image builder with content-addressed cache.

Given a harness directory that may contain `requirements.txt`:

  - No requirements.txt (or empty) → return the base image.
  - Otherwise:
      sha = sha256(sorted, stripped requirements.txt lines)[:16]
      cached = forge_images/<sha>.sif
      if cached exists: return it (cache hit).
      else: build a delta image (Bootstrap: localimage, From: eval-base.sif,
            %post pip install -r ...) via `singularity build` + proot.

Content addressing means two harnesses with identical deps share one image.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from forge.paths import EVAL_BASE_SIF, FORGE_IMAGES_DIR, PROOT_BIN_DIR

log = logging.getLogger("forge.env_builder")

_BUILD_TIMEOUT_S = 15 * 60


class EnvBuildError(RuntimeError):
    """Raised on build timeout / non-zero exit / missing artifact."""


def _normalize_requirements(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    lines.sort()
    return "\n".join(lines) + ("\n" if lines else "")


def _hash_requirements(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _delta_def(reqs_filename: str) -> str:
    return f"""Bootstrap: localimage
From: {EVAL_BASE_SIF}

%files
    {reqs_filename} /opt/extra_requirements.txt

%post
    set -e
    pip install --no-cache-dir -r /opt/extra_requirements.txt
    rm /opt/extra_requirements.txt

%labels
    ForgeDelta 1
"""


async def ensure_image(
    harness_dir: Path,
    build_log_path: Optional[Path] = None,
) -> Path:
    """Resolve the image path for evaluating `harness_dir`.

    Raises EnvBuildError on build failure. Build log (stdout+stderr) is
    written to <harness_dir>/build.log unless overridden.
    """
    if not EVAL_BASE_SIF.exists():
        raise EnvBuildError(f"Base image missing at {EVAL_BASE_SIF}")

    req_file = harness_dir / "requirements.txt"
    if not req_file.exists():
        return EVAL_BASE_SIF

    normalized = _normalize_requirements(req_file.read_text(encoding="utf-8"))
    if not normalized.strip():
        return EVAL_BASE_SIF

    sha = _hash_requirements(normalized)
    FORGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    cached = FORGE_IMAGES_DIR / f"{sha}.sif"
    if cached.exists():
        log.info(f"env_builder: cache hit for {harness_dir.name} → {cached.name}")
        return cached

    log.info(f"env_builder: building delta for {harness_dir.name} → {cached.name}")
    if build_log_path is None:
        build_log_path = harness_dir / "build.log"
    build_log_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        reqs_staged = Path(tmp) / "requirements.txt"
        reqs_staged.write_text(normalized, encoding="utf-8")
        def_path = Path(tmp) / "delta.def"
        def_path.write_text(_delta_def(reqs_staged.name), encoding="utf-8")

        env = {
            **os.environ,
            "PATH": f"{PROOT_BIN_DIR}:{os.environ.get('PATH', '')}",
        }

        cmd = [
            "singularity", "build", "--force",
            str(cached),
            str(def_path),
        ]

        with build_log_path.open("wb") as logf:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=tmp,
                env=env,
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=_BUILD_TIMEOUT_S)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                raise EnvBuildError(
                    f"Build timeout after {_BUILD_TIMEOUT_S}s; see {build_log_path}"
                )

    if rc != 0 or not cached.exists():
        # Remove partial artifact so a retry doesn't falsely hit cache
        try:
            if cached.exists():
                cached.unlink()
        except OSError:
            pass
        raise EnvBuildError(f"Build failed (rc={rc}); see {build_log_path}")

    log.info(
        f"env_builder: built {cached.name} ({cached.stat().st_size / 1e6:.0f} MB)"
    )
    return cached
