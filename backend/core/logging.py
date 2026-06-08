"""Logging configuration."""

import sys

from loguru import logger


def configure_logging() -> None:
    """Configure structured console logging for the application."""

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
