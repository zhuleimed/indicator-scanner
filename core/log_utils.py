"""
Simple logging utilities for the backtest project.

Provides colored, timestamped logging output via the standard logging module,
plus a console_out helper for messages that should always display.

Usage:
    from core.log_utils import get_logger, console_out

    logger = get_logger(__name__)
    logger.info("Processing data...")
    console_out("Progress: checked 1000 rows")
"""

import logging
import sys

# ---------------------------------------------------------------------------
# Coloured formatter
# ---------------------------------------------------------------------------

_RESET = "\033[0m"

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",      # cyan
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[41m",   # red background
}


class _ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colour and strips it when output is not a TTY."""

    def __init__(self):
        super().__init__("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        asctime = self.formatTime(record, "%H:%M:%S")
        color = _LEVEL_COLORS.get(record.levelno, _RESET)
        prefix = f"{color}[{asctime}]{_RESET}"
        return f"{prefix} {record.getMessage()}"


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a cached logger with coloured console output.

    Parameters
    ----------
    name : str
        Logger name (convention: use ``__name__``).
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers when get_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColoredFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    _loggers[name] = logger
    return logger


# ---------------------------------------------------------------------------
# Always-show output helper
# ---------------------------------------------------------------------------

def console_out(*args, **kwargs) -> None:
    """Print a status / progress message that bypasses logging levels.

    This is intended for messages that should *always* reach the terminal
    (e.g. progress counters, section headings).  It writes directly to
    stdout via ``print`` so it works with progress-bar libraries that
    also write to stdout.
    """
    print(*args, **kwargs)
