import logging
import os
import sys
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str | None = None,
    log_file: str | Path | None = "chatmem.log",
    console: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Configure structured logging for the chatmem application.

    Args:
        level: Logging level (e.g. logging.INFO, 'DEBUG'). Defaults to env var LOG_LEVEL or INFO.
        log_file: Path to log file. If None or empty, file logging is disabled.
        console: Whether to log to standard error. Defaults to True.
        propagate: Whether package logger propagates to root logger. Defaults to False.

    Returns:
        The configured 'chatmem' parent logger.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("chatmem")
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = propagate
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger under the chatmem namespace.

    Example:
        logger = get_logger(__name__)
    """
    if name is None:
        return logging.getLogger("chatmem")

    # If already fully qualified with chatmem, return directly
    if name.startswith("chatmem"):
        return logging.getLogger(name)

    return logging.getLogger(f"chatmem.{name}")

