"""Queue-based log handler for GUI log polling."""

from __future__ import annotations

import logging
import queue
from typing import Tuple


class QueueHandler(logging.Handler):
    """Push formatted log lines into a queue without blocking producer threads."""

    def __init__(self, log_queue: "queue.Queue[Tuple[str, str]]", maxsize: int = 5000):
        super().__init__()
        self._queue = log_queue
        self._maxsize = maxsize

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = (record.levelname.upper(), self.format(record))
            try:
                self._queue.put_nowait(payload)
                return
            except queue.Full:
                pass

            # Drop oldest entry and retry once.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                return
        except Exception:
            self.handleError(record)
