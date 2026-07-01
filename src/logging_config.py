"""Centralized structured logging for TDA-Longevity-Resilience.

Provides a standard `get_logger(__name__)` pattern used by all src/ modules.
Formats: [timestamp] [LEVEL] [module] message — outputs to stderr to avoid
interfering with pipeline stdout.
"""

import logging
import sys

_FORMAT = "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _setup_root_logger():
    """Configure the root logger once."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any default handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ["matplotlib", "PIL", "urllib3", "numexpr", "ripser"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Usage:
        from src.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("message")
    """
    _setup_root_logger()
    return logging.getLogger(name)
