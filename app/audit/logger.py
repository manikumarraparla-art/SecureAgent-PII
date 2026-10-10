"""Audit logger and sinks.

`JsonlAuditSink` appends one JSON object per line to a local file created
with owner-only permissions. Audit write failures are reported through the
standard logger (without event content) and do not abort the user request;
see architecture.md for the rationale.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from app.audit.events import AuditEvent

logger = logging.getLogger(__name__)


class AuditSink(Protocol):
    def write(self, event: AuditEvent) -> None:
        ...


class JsonlAuditSink:
    """Thread-safe append-only JSON Lines file sink (mode 0600)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":")) + "\n"
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line)


class InMemoryAuditSink:
    """Collects events in memory (tests and diagnostics)."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        with self._lock:
            self.events.append(event)


class AuditLogger:
    """Fans audit events out to one or more sinks."""

    def __init__(self, sinks: Iterable[AuditSink]) -> None:
        self._sinks = list(sinks)

    def record(self, event: AuditEvent) -> None:
        for sink in self._sinks:
            try:
                sink.write(event)
            except Exception as exc:  # never let auditing crash a request
                logger.error(
                    "Audit sink %s failed (%s) for event %s",
                    type(sink).__name__, type(exc).__name__, event.event.value,
                )
