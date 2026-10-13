"""Loader lifecycle diagnostics through standard logging without configuring application handlers."""

from __future__ import annotations

import logging
from typing import TypeAlias

LOGGER_NAME = "dataset_rt.loader"
LogValue: TypeAlias = str | int | bool | None


def loader_event(message: str, *args: LogValue) -> None:
    """Emit INFO through configured handlers, otherwise print a bounded event to stderr.

    Application handlers and filters take precedence. The temporary fallback is
    never registered, so later logging configuration cannot produce duplicate output.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.hasHandlers():
        logger.info(message, *args)
        return
    if logger.level != logging.NOTSET and not logger.isEnabledFor(logging.INFO):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    try:
        record = logger.makeRecord(LOGGER_NAME, logging.INFO, "", 0, message, args, None)
        handler.handle(record)
    finally:
        handler.close()
