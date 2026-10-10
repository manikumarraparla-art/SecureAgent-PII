"""Operational logging setup for the `app` logger hierarchy.

Application log messages contain request IDs, counts, types and error codes
only; prompt and response contents are never logged.
"""

from __future__ import annotations

import logging

_HANDLER_NAME = "secureagent-console"


def configure_logging(level: str = "INFO") -> None:
    """Attach a single console handler to the `app` logger (idempotent)."""
    app_logger = logging.getLogger("app")
    app_logger.setLevel(level)
    if any(h.get_name() == _HANDLER_NAME for h in app_logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    app_logger.addHandler(handler)
