"""2026 regular-season window for live Statcast refresh."""

from __future__ import annotations

from datetime import date

SEASON_START = date(2026, 3, 1)
SEASON_END = date(2026, 10, 5)


def in_season(today: date | None = None) -> bool:
    d = today or date.today()
    return SEASON_START <= d <= SEASON_END
