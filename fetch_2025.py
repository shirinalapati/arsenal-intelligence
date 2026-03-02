"""Retry 2025 Statcast fetch with monthly chunks for reliability."""
import os, time
import pandas as pd
from pybaseball import statcast
import pybaseball

pybaseball.cache.enable()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

MONTHS_2025 = [
    ("2025-03-27", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-28"),
]

COLS_NEEDED = [
    "game_date", "player_name", "pitcher", "batter",
    "pitch_type", "pitch_name",
    "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "release_extension",
    "release_pos_x", "release_pos_z",
    "description",
    "type",
    "events",
    "estimated_woba_using_speedangle",
    "launch_speed", "launch_angle",
    "stand",
    "p_throws",
    "zone",
    "balls", "strikes", "outs_when_up",
    "inning",
]

frames = []
for start, end in MONTHS_2025:
    cache_path = os.path.join(DATA_DIR, f"statcast_2025_{start[:7]}.parquet")
    if os.path.exists(cache_path):
        print(f"  {start[:7]}: loading from cache")
        frames.append(pd.read_parquet(cache_path))
        continue
    for attempt in range(3):
        try:
            print(f"  {start} → {end} (attempt {attempt+1})")
            df = statcast(start_dt=start, end_dt=end)
            keep = [c for c in COLS_NEEDED if c in df.columns]
            df = df[keep].copy()
            df["season"] = 2025
            df.to_parquet(cache_path, index=False)
            frames.append(df)
            print(f"    saved {len(df):,} rows")
            break
        except Exception as e:
            print(f"    error: {e}")
            time.sleep(5)

if frames:
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(os.path.join(DATA_DIR, "statcast_2025.parquet"), index=False)
    print(f"\n2025 total: {len(out):,} pitches")

    # Rebuild combined
    all_frames = []
    for yr in [2023, 2024, 2025]:
        p = os.path.join(DATA_DIR, f"statcast_{yr}.parquet")
        if os.path.exists(p):
            df_yr = pd.read_parquet(p)
            if "season" not in df_yr.columns:
                df_yr["season"] = yr
            all_frames.append(df_yr)
            print(f"{yr}: {len(df_yr):,} pitches")
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_parquet(os.path.join(DATA_DIR, "statcast_2023_2025.parquet"), index=False)
    print(f"\nCombined: {len(combined):,} pitches")
