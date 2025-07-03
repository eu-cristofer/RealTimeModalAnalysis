"""
Centralised logger for CocoaEngineUI
------------------------------------
✓ Console output always on (root logger).
✓ Optional rotating-file handler (call `setup_logging(log_to_file=True)`).
✓ Includes module name in log output.
"""

from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Format strings ----------------------------------------------------
_FMT = "%(asctime)s | %(levelname)-5s | %(module)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# --- Logging Setup -----------------------------------------------------
def setup_logging(level: int = logging.INFO,
                  log_to_file: bool = False,
                  log_file: str | Path = "rtma_app.log",
                  max_bytes: int = 1_000_000,
                  backup_count: int = 3) -> bool:
    """
    Configure root logger once. Subsequent calls are ignored.

    Parameters
    ----------
    level : logging level (default INFO)
    log_to_file : when True, adds a rotating-file handler
    log_file : path to *.log* file
    max_bytes : file size before rotation
    backup_count : how many old logs to keep

    Returns
    -------
    bool : True if setup was performed, False if already configured
    """
    root = logging.getLogger()
    if root.handlers:
        return False

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(console)

    # Optional file handler
    if log_to_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes,
                                           backupCount=backup_count,
                                           encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        root.addHandler(file_handler)

    root.setLevel(level)
    return True

# --- Logger Access -----------------------------------------------------
def get_logger(name: str | None = None) -> logging.Logger:
    """Helper so every module does `log = get_logger(__name__)`."""
    return logging.getLogger(name)
