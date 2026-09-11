"""Adaptateur : l'horloge système. Les tests utilisent une horloge fixe (tests/fakes.py)."""

from __future__ import annotations

from datetime import datetime, timezone


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)
