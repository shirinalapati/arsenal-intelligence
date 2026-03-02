"""
Fetch 3 seasons of Statcast pitch-level data (2023-2025) and cache locally.
Run this once before the main notebook.
"""

import os
import pandas as pd
from pybaseball import statcast
import pybaseball

pybaseball.cache.enable()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEASONS = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

COLS_NEEDED = [
    "game_date", "season", "player_name", "pitcher", "batter",
    "pitch_type", "pitch_name",
    "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z",                     # horizontal / induced vertical break (ft)
    "plate_x", "plate_z",                  # location at plate
    "release_extension",
    "release_pos_x", "release_pos_z",
    "description",                          # swing/whiff/called-strike etc.
    "type",                                 # B/S/X
    "events",                               # strikeout, home_run, etc.
    "estimated_woba_using_speedangle",      # xwOBA
    "launch_speed", "launch_angle",
    "stand",                               # batter handedness
    "p_throws",                            # pitcher handedness
    "zone",
    "balls", "strikes", "outs_when_up",
    "inning",
]


def fetch_season(year, start, end):
    out_path = os.path.join(DATA_DIR, f"statcast_{year}.parquet")
    if os.path.exists(out_path):
        print(f"  {year}: loading from cache ({out_path})")
        return pd.read_parquet(out_path)

    print(f"  {year}: pulling from Baseball Savant ({start} → {end}) ...")
    df = statcast(start_dt=start, end_dt=end)
    df["season"] = year

    # keep only columns we need (drop missing cols gracefully)
    keep = [c for c in COLS_NEEDED if c in df.columns]
    df = df[keep].copy()

    df.to_parquet(out_path, index=False)
    print(f"  {year}: saved {len(df):,} rows → {out_path}")
    return df


if __name__ == "__main__":
    frames = []
    for year, (s, e) in SEASONS.items():
        frames.append(fetch_season(year, s, e))

    combined = pd.concat(frames, ignore_index=True)
    out = os.path.join(DATA_DIR, "statcast_2023_2025.parquet")
    combined.to_parquet(out, index=False)
    print(f"\nCombined dataset: {len(combined):,} pitches → {out}")
