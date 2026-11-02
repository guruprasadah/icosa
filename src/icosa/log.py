"""Colorized logging setup shared by the transmitter and receiver.

Colors are emitted only when the target stream is a TTY, so piped or
redirected output stays plain. The implementation is dependency-free.
"""

from __future__ import annotations

import logging
import sys

# ANSI SGR sequences.
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_LEVEL_COLORS = {
    logging.DEBUG: "\033[2m",  # dim
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}

_LOG_FMT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FMT = "%H:%M:%S"


class ColorFormatter(logging.Formatter):
    """Formatter that wraps the level name and logger name in ANSI colors."""

    def __init__(self, *, use_color: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)

        color = _LEVEL_COLORS.get(record.levelno, "")
        # Colorize a copy so we don't mutate the shared record.
        original_levelname = record.levelname
        original_name = record.name
        try:
            record.levelname = f"{color}{record.levelname}{_RESET}"
            record.name = f"{_BOLD}{original_name}{_RESET}"
            return super().format(record)
        finally:
            record.levelname = original_levelname
            record.name = original_name


def _logger_for(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    *,
    logger_name: str | None = None,
    stream=None,
) -> logging.Logger:
    """Install a colorized stream handler and return the configured logger.

    A dedicated handler is attached to the requested logger (or the root
    logger) exactly once per process to avoid duplicate lines on repeated
    calls (e.g. across multiple ``asyncio.run`` invocations in one process).

    :param level: Minimum level to emit.
    :param logger_name: Logger to configure; defaults to the root logger.
    :param stream: Output stream; defaults to ``sys.stderr``.
    """
    stream = stream if stream is not None else sys.stderr
    use_color = bool(getattr(stream, "isatty", lambda: False)())

    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        ColorFormatter(use_color=use_color, fmt=_LOG_FMT, datefmt=_DATE_FMT)
    )
    handler.setLevel(level)

    logger = _logger_for(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(level)

    for existing in logger.handlers:
        if getattr(existing, "_icosa_color_handler", False):
            existing.setLevel(level)
            existing.setFormatter(handler.formatter)
            return logger

    handler._icosa_color_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger
