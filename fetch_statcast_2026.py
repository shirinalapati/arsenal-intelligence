"""Fetch 2026 Statcast pitch data (monthly chunks) for live season updates."""

from __future__ import annotations

import argparse
import os
import time
from datetime import date

import pandas as pd
import pybaseball
from pybaseball import statcast

from season_config import SEASON_END, in_season

pybaseball.cache.enable()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

COLS_NEEDED = [
    "game_date", "player_name", "pitcher", "batter",
    "pitch_type", "pitch_name",
    "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "release_extension",
    "release_pos_x", "release_pos_z",
    "description", "type", "events",
    "estimated_woba_using_speedangle",
    "launch_speed", "launch_angle",
    "stand", "p_throws", "zone",
    "balls", "strikes", "outs_when_up", "inning",
    "inning_topbot", "home_team", "away_team", "game_pk",
]

MONTHS_2026 = [
    ("2026-03-25", "2026-04-30"),
    ("2026-05-01", "2026-05-31"),
    ("2026-06-01", "2026-06-30"),
    ("2026-07-01", "2026-07-31"),
    ("2026-08-01", "2026-08-31"),
    ("2026-09-01", "2026-09-30"),
    ("2026-10-01", "2026-10-05"),
]


def _trim_end(end: str) -> str:
    cap = min(date.today(), SEASON_END).isoformat()
    return end if end <= cap else cap


def _cache_stale(cache_path: str, start: str, force: bool) -> bool:
    """Re-fetch the current month's chunk so new games are picked up."""
    if force or not os.path.exists(cache_path):
        return True
    month_prefix = start[:7]
    if month_prefix == date.today().strftime("%Y-%m"):
        return True
    return False


def fetch_2026(force: bool = False) -> pd.DataFrame:
    if not in_season() and not force:
        out_path = os.path.join(DATA_DIR, "statcast_2026.parquet")
        if os.path.exists(out_path):
            print("Outside 2026 season window — using existing statcast_2026.parquet")
            return pd.read_parquet(out_path)
        raise SystemExit("Outside 2026 season and no cached statcast_2026.parquet.")

    frames: list[pd.DataFrame] = []
    for start, end in MONTHS_2026:
        end = _trim_end(end)
        if start > end:
            continue
        cache_path = os.path.join(DATA_DIR, f"statcast_2026_{start[:7]}.parquet")
        if not _cache_stale(cache_path, start, force):
            print(f"  {start[:7]}: cache")
            frames.append(pd.read_parquet(cache_path))
            continue
        for attempt in range(3):
            try:
                print(f"  {start} → {end} (attempt {attempt + 1})")
                df = statcast(start_dt=start, end_dt=end)
                if df is None or df.empty:
                    print("    empty chunk")
                    break
                keep = [c for c in COLS_NEEDED if c in df.columns]
                df = df[keep].copy()
                df["season"] = 2026
                df.to_parquet(cache_path, index=False)
                frames.append(df)
                print(f"    saved {len(df):,} rows")
                break
            except Exception as exc:
                print(f"    error: {exc}")
                time.sleep(8)

    if not frames:
        out_path = os.path.join(DATA_DIR, "statcast_2026.parquet")
        if os.path.exists(out_path):
            return pd.read_parquet(out_path)
        raise SystemExit("No 2026 Statcast data fetched.")

    out = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "statcast_2026.parquet")
    out.to_parquet(out_path, index=False)
    print(f"\n2026 total: {len(out):,} pitches → {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch all monthly chunks")
    args = parser.parse_args()
    fetch_2026(force=args.force)
