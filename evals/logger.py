import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
try:
    from rich.console import Console
    from rich.logging import RichHandler
    USE_RICH = True
except ImportError:
    USE_RICH = False

LOG_DIR = Path(os.environ.get("EVALS_LOG_DIR", "./logs"))
LOG_DIR.mkdir(exist_ok=True)

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

        file_handler = RotatingFileHandler(LOG_DIR / log_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

        _initialized_loggers[name] = logger
    return logger

