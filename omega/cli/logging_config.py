from __future__ import annotations
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal
from omega.environment.conf_loader import omega_settings

LOG_FILENAME = "omega.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def get_log_dir() -> Path:
    log_directory = Path(omega_settings.memory_dir).expanduser() / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    return log_directory

def get_log_file() -> Path:
    return get_log_dir() / LOG_FILENAME

def _remove_managed_handlers(root_logger: logging.Logger) -> None:
    for handler in list(root_logger.handlers):
        if getattr(handler, "_omega_managed", False):
            root_logger.removeHandler(handler)
            handler.close()

def configure_logging(mode: Literal["tui", "cli"] = "cli", level: int = logging.INFO) -> Path | None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    _remove_managed_handlers(root_logger)

    formatter = logging.Formatter(LOG_FORMAT)
    if mode == "tui":
        log_file = get_log_file()
        handler: logging.Handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    elif mode == "cli":
        log_file = None
        handler = logging.StreamHandler(sys.stderr)
    else:
        raise ValueError(f"Unknown logging mode: {mode}")

    handler.setFormatter(formatter)
    handler._omega_managed = True
    root_logger.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return log_file

def setup_tui_logging(level: int = logging.INFO) -> Path:
    log_file = configure_logging(mode="tui", level=level)
    assert log_file is not None
    logging.getLogger(__name__).info("TUI logging configured at %s", log_file)
    return log_file