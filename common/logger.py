"""Shared logger used by alma-style code.

Log directory resolution (lazy — happens inside `get_logger`, not at import):
  1. `EVALS_LOG_DIR` env var if set (forge sets this to the per-eval out_dir
     via the Singularity --env binding; alma's run_main.py sets it to
     `baselines/evolve/alma/logs/`). Inside containers, `/out` is bound R/W.
  2. `<project_root>/baselines/evolve/alma/logs/` IF that directory is
     reachable and writable (host-side use; on stripped-bind containers it
     isn't).
  3. `tempfile.gettempdir() / "memevol_logs"` — always works on any container.

Importing `common.logger` has no filesystem side effects.
"""
import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
try:
    from rich.console import Console
    from rich.logging import RichHandler
    USE_RICH = True
except ImportError:
    USE_RICH = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HOST_DEFAULT_LOG_DIR = _PROJECT_ROOT / "baselines" / "evolve" / "alma" / "logs"


def _resolve_log_dir() -> Path:
    """Pick a writable log dir lazily. See module docstring for the chain."""
    env_dir = os.environ.get("EVALS_LOG_DIR")
    if env_dir:
        return Path(env_dir)
    # Host-side: alma path is conventional and parent exists.
    if _HOST_DEFAULT_LOG_DIR.parent.is_dir() and os.access(_HOST_DEFAULT_LOG_DIR.parent, os.W_OK):
        return _HOST_DEFAULT_LOG_DIR
    # Container-side fallback: /tmp is always writable.
    return Path(tempfile.gettempdir()) / "memevol_logs"


# Allow per-run log file via environment variable (e.g. "train_all_at_once.log")
_DEFAULT_LOG_FILE = os.environ.get("MEMEVOL_LOG_FILE", ".log")

DEFAULT_LEVEL_STYLES = {
    "DEBUG": {"color": "cyan"},
    "INFO": {"color": "green"},
    "WARNING": {"icon": "⚠️", "color": "yellow"},
    "ERROR": {"icon": "💥", "color": "red"},
    "CRITICAL": {"icon": "🔥", "color": "bold magenta"},
}

_initialized_loggers = {}
console = Console(force_terminal=True, soft_wrap=True) if USE_RICH else None


def get_logger(name="", level=logging.INFO, log_file=None, level_styles=None):
    if log_file is None:
        log_file = _DEFAULT_LOG_FILE
    if name in _initialized_loggers:
        return _initialized_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    styles = level_styles or DEFAULT_LEVEL_STYLES

    if not logger.handlers:
        if USE_RICH:
            rich_handler = RichHandler(
                console=console,
                show_time=True,
                show_level=False,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )

            class EmojiFormatter(logging.Formatter):
                def format(self, record):
                    style = styles.get(record.levelname, {"icon": "", "color": "white"})
                    icon = style.get("icon", "")
                    color = style.get("color", "white")
                    with console._lock:
                        return f"[{color}]{icon} {record.getMessage()}[/{color}]"

            rich_handler.setFormatter(EmojiFormatter())
            logger.addHandler(rich_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
            ))
            logger.addHandler(console_handler)

        log_dir = _resolve_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_dir / log_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

        _initialized_loggers[name] = logger
    return logger
