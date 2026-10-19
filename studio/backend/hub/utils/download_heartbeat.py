# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional


def new_path() -> str:
    return os.path.join(tempfile.gettempdir(), f"unsloth-dl-heartbeat-{uuid.uuid4().hex}")


def _read(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read(path: Optional[str]) -> Optional[int]:
    return _read(path).get("activity_bytes")


def read_transfer(path: Optional[str]) -> Optional[dict]:
    payload = _read(path)
    network_bytes = payload.get("network_bytes")
    if network_bytes is None:
        return None
    return {
        "attempt_id": Path(path).name,
        "network_bytes": network_bytes,
        "elapsed_seconds": payload["elapsed_seconds"],
        "age_seconds": max(0.0, time.time() - payload["updated_at"]),
    }


def age(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        return max(0.0, time.time() - Path(path).stat().st_mtime)
    except OSError:
        return None


def remove(path: Optional[str]) -> None:
    if not path:
        return
    for candidate in (path, f"{path}.tmp"):
        try:
            Path(candidate).unlink(missing_ok = True)
        except OSError:
            pass


class HeartbeatWriter:
    def __init__(
        self,
        path: str,
        interval: float = 1.0,
    ) -> None:
        self._path = path
        self._interval = interval
        self._total = 0
        self._started_at = time.monotonic()
        self._network_bytes: Optional[int] = None
        self._transfer_elapsed = None
        self._transfer_updated_at = None
        self._flushed_at = float("-inf")
        self._lock = threading.Lock()

    def add(
        self,
        n: int,
        *,
        transfer: bool = False,
    ) -> None:
        with self._lock:
            self._total += n
            now = time.monotonic()
            if transfer:
                self._network_bytes = (self._network_bytes or 0) + n
                self._transfer_elapsed = now - self._started_at
                self._transfer_updated_at = time.time()
            if now - self._flushed_at < self._interval:
                return
            self._flushed_at = now
            tmp = f"{self._path}.tmp"
            try:
                with open(tmp, "w", encoding = "utf-8") as handle:
                    json.dump(
                        {
                            "activity_bytes": self._total,
                            "network_bytes": self._network_bytes,
                            "elapsed_seconds": self._transfer_elapsed,
                            "updated_at": self._transfer_updated_at,
                        },
                        handle,
                    )
                os.replace(tmp, self._path)
            except OSError:
                pass
