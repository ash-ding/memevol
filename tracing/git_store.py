"""Per-user git repository — the agent-navigable substrate.

One git repo per user at ``<trace_root>/<run_id>/traces/git/<user_key>/``
(``workspace/`` is gitignored; the root is overridable via
``MEMEVOL_TRACE_ROOT`` for tests/demo). Each memory item is ONE text file at
a STABLE path ``memory/<adapter_kind>/<sanitized-hashed-id>.md`` — no
sequence number in the filename — so a single item's full multi-commit
history is followable with ``git log`` / ``git blame``. Create / update /
delete of an item map to add / modify / delete of that one file, making
``git diff`` a genuine per-item change set.

Only text is ever written: YAML frontmatter (name-only ``embedding_model``,
never a vector) plus a plain-text body already vetted by the adapters'
redaction rules. Commits carry deterministic, code-computed descriptors with
machine-parseable git trailers. Nothing here calls an LLM.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("main")

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Fixed identity so commits succeed without any global git config, and so the
# repo is reproducible across hosts / containers.
_GIT_ENV_ARGS = [
    "-c", "user.name=memevol-tracer",
    "-c", "user.email=tracer@memevol.local",
    "-c", "commit.gpgsign=false",
    "-c", "core.autocrlf=false",
]


def safe_item_filename(item_id: str) -> str:
    """Stable, collision-safe, filesystem-safe filename for an item id.

    Sanitised prefix (readable in ``git log``) plus a short content hash
    (so distinct ids that sanitise to the same prefix never collide). The
    mapping is deterministic: the SAME item id always yields the SAME name.
    """
    raw = str(item_id)
    base = _FILENAME_SANITIZE_RE.sub("_", raw).strip("_")[:64]
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    stem = f"{base}-{digest}" if base else digest
    return f"{stem}.md"


class GitStore:
    """Filesystem + git mechanics for one user's trace repo. Never crashes an
    eval — every git error is caught, logged, and swallowed."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = Path(repo_dir)

    # -- repo lifecycle ----------------------------------------------------

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *_GIT_ENV_ARGS, *args],
            cwd=str(self.repo_dir),
            check=check,
            capture_output=True,
            text=True,
        )

    def ensure_repo(self) -> None:
        """Lazy ``git init`` on first use."""
        if (self.repo_dir / ".git").exists():
            return
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q")

    def head_exists(self) -> bool:
        if not (self.repo_dir / ".git").exists():
            return False
        return self._git("rev-parse", "--verify", "-q", "HEAD",
                         check=False).returncode == 0

    # -- item files --------------------------------------------------------

    def item_rel_path(self, adapter_kind: str, item_id: str) -> str:
        return f"memory/{adapter_kind}/{safe_item_filename(item_id)}"

    def write_item(self, rel_path: str, content: str) -> None:
        path = self.repo_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def delete_item(self, rel_path: str) -> None:
        path = self.repo_dir / rel_path
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("[tracing] could not delete %s: %r", rel_path, exc)

    # -- commit ------------------------------------------------------------

    def commit(self, message: str) -> Optional[str]:
        """Stage everything and commit. Returns the commit SHA, or None if the
        commit could not be made (never raises). No ``--allow-empty`` — an
        unchanged tree is a caller bug (commit-on-change guards against it),
        but we tolerate it gracefully."""
        try:
            self.ensure_repo()
            self._git("add", "-A")
            proc = self._git("commit", "-q", "-m", message, check=False)
            if proc.returncode != 0:
                # "nothing to commit" is the only expected non-zero here.
                if "nothing to commit" not in (proc.stdout + proc.stderr):
                    log.warning("[tracing] git commit failed: %s",
                                (proc.stdout + proc.stderr).strip())
                return None
            sha = self._git("rev-parse", "HEAD", check=False).stdout.strip()
            return sha or None
        except Exception as exc:  # never crash the eval on a trace failure
            log.warning("[tracing] git commit error: %r", exc)
            return None

    # -- read helpers (used by the demo / tests / a future analysis agent) --

    def log_lines(self, *extra: str) -> List[str]:
        if not self.head_exists():  # no repo / no commits yet
            return []
        proc = self._git("log", "--format=%H %s", *extra, check=False)
        if proc.returncode != 0:
            return []
        return [ln for ln in proc.stdout.splitlines() if ln.strip()]

    def file_history(self, rel_path: str) -> List[str]:
        if not self.head_exists():
            return []
        proc = self._git("log", "--format=%H", "--", rel_path, check=False)
        if proc.returncode != 0:
            return []
        return [ln for ln in proc.stdout.splitlines() if ln.strip()]
