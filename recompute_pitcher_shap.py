"""
Recompute only the per-pitcher SHAP breakdown (section 10) using already-trained GBC models.
Much faster than running the full precompute.py — no model retraining needed.
"""
import os, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

STUFF_FEATURES = [
    "perceived_velo", "pfx_z_in", "break_arm",
    "release_spin_rate", "extension", "plate_x", "plate_z"
]
FEATURE_LABELS = {
    "perceived_velo":    "Perceived Velocity",
    "pfx_z_in":          "Induced Vertical Break",
    "break_arm":         "Arm-Side Horiz Break",
    "release_spin_rate": "Spin Rate",
    "extension":         "Release Extension",
    "plate_x":           "Horizontal Location",
    "plate_z":           "Vertical Location",
}
PITCH_TYPE_MAP = {
    "FF": "Four-Seam FB", "FA": "Four-Seam FB",
    "SI": "Sinker",       "FT": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",       "ST": "Sweeper",
    "CU": "Curveball",    "KC": "Knuckle-Curve",
    "CH": "Changeup",     "FS": "Splitter",
    "SV": "Slurve",
}

MIN_PITCHER_PITCHES = 100

# ── Load already-scored pitch data ───────────────────────────────────────────
print("Loading pitch_type_scores to get all_df...")

# Rebuild all_df from scratch using already-saved parquet — faster than re-reading raw
# Load the scored pitch-type data to identify pitchers/roles, but we need raw pitch
# data with SHAP features, so load from the combined parquet.
combined_path = os.path.join(DATA_DIR, "statcast_2023_2025.parquet")
if os.path.exists(combined_path):
    raw = pd.read_parquet(combined_path)
else:
    frames = []
    for yr in [2023, 2024, 2025]:
        p = os.path.join(DATA_DIR, f"statcast_{yr}.parquet")
        df_yr = pd.read_parquet(p)
        if "season" not in df_yr.columns:
            df_yr["season"] = yr
        frames.append(df_yr)
    raw = pd.concat(frames, ignore_index=True)
print(f"  {len(raw):,} pitches loaded")

df = raw.copy()
df["pfx_x_in"]     = pd.to_numeric(df["pfx_x"], errors="coerce") * 12
df["pfx_z_in"]     = pd.to_numeric(df["pfx_z"], errors="coerce") * 12
df["break_arm"]    = np.where(df["p_throws"] == "L", -df["pfx_x_in"], df["pfx_x_in"])
df["extension"]    = pd.to_numeric(df["release_extension"], errors="coerce").fillna(6.0)
df["perceived_velo"] = pd.to_numeric(df["release_speed"], errors="coerce") + (df["extension"] - 6.0) * 0.5
df["pitch_group"]  = df["pitch_type"].map(PITCH_TYPE_MAP)
df = df[df["pitch_group"].notna()].copy()
for c in STUFF_FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df_clean = df.dropna(subset=["release_speed","pfx_x_in","pfx_z_in",
                               "release_spin_rate","plate_x","plate_z"]).copy()
df_clean = df_clean[
    df_clean["release_speed"].between(50, 110) &
    df_clean["release_spin_rate"].between(500, 4000) &
    df_clean["pfx_z_in"].between(-30, 30) &
    df_clean["pfx_x_in"].between(-30, 30)
]

# Reattach role from arsenal_scores
arsenal = pd.read_parquet(os.path.join(DATA_DIR, "arsenal_scores.parquet"))
role_map = (
    arsenal[["pitcher", "season", "role"]]
    .drop_duplicates()
)
df_clean = df_clean.merge(role_map, on=["pitcher", "season"], how="left")
df_clean["role"] = df_clean["role"].fillna("RP")

# Need stuff_plus — load from pitch_type_scores and merge back
pt_scores = pd.read_parquet(os.path.join(DATA_DIR, "pitch_type_scores.parquet"))
stuff_map = pt_scores[["pitcher","season","role","pitch_group","stuff_plus"]].copy()
# Get per-pitch stuff_plus via a groupby merge (mean per pitcher-season-role-pitchgroup already stored)
df_clean = df_clean.merge(
    stuff_map.rename(columns={"stuff_plus": "stuff_plus_mean"}),
    on=["pitcher","season","role","pitch_group"], how="left"
)
df_clean["stuff_plus"] = df_clean["stuff_plus_mean"]

all_seasons = sorted(df_clean["season"].unique())
print(f"  Seasons to compute SHAP for: {all_seasons}")

# ── Enumerate available trained models — all seasons ─────────────────────────
print("\nLoading trained GBC models...")
all_pitch_groups = sorted(df_clean["pitch_group"].unique())
pitcher_shap_frames = []

for season in all_seasons:
    print(f"\n  Season {season}:")
    for role in ["SP", "RP"]:
        for pg in all_pitch_groups:
            pkl_name = f"shap_gbc_{role.lower()}_{pg.replace(' ','_').replace('-','').lower()}.pkl"
            pkl_path = os.path.join(DATA_DIR, pkl_name)
            if not os.path.exists(pkl_path):
                continue

            with open(pkl_path, "rb") as f:
                saved = pickle.load(f)
            gbc       = saved["gbc"]
            scaler    = saved["scaler"]
            explainer = shap.TreeExplainer(gbc)

            pt_all = df_clean[
                (df_clean["pitch_group"] == pg) &
                (df_clean["season"] == season) &
                (df_clean["role"] == role)
            ].dropna(subset=STUFF_FEATURES)

            qual = pt_all.groupby("pitcher").size()
            qual = qual[qual >= MIN_PITCHER_PITCHES].index
            pt_qual = pt_all[pt_all["pitcher"].isin(qual)].copy()
            if pt_qual.empty:
                continue

            sv_all = explainer.shap_values(scaler.transform(pt_qual[STUFF_FEATURES].values))
            for i, feat in enumerate(STUFF_FEATURES):
                pt_qual[f"shap_{feat}"] = sv_all[:, i]

            shap_cols = [f"shap_{f}" for f in STUFF_FEATURES]
            ps = (
                pt_qual.groupby(["pitcher", "player_name"])
                .agg({**{c: "mean" for c in shap_cols},
                      "stuff_plus": "mean", "release_speed": "size"})
                .rename(columns={"release_speed": "n_pitches"})
                .reset_index()
            )
            ps.rename(columns={f"shap_{f}": FEATURE_LABELS[f] for f in STUFF_FEATURES}, inplace=True)
            ps["season"]      = season
            ps["role"]        = role
            ps["pitch_group"] = pg
            pitcher_shap_frames.append(ps)
            print(f"    {role} {pg:20s}  {len(ps)} pitchers")

pitcher_shap = pd.concat(pitcher_shap_frames, ignore_index=True)
pitcher_shap.to_parquet(os.path.join(DATA_DIR, "pitcher_shap_allpitch.parquet"), index=False)

fb = pitcher_shap[pitcher_shap["pitch_group"] == "Four-Seam FB"]
fb.to_parquet(os.path.join(DATA_DIR, "pitcher_shap_4seam.parquet"), index=False)

print(f"\nDone. Saved pitcher_shap_allpitch.parquet ({len(pitcher_shap)} rows)")
print(f"      Saved pitcher_shap_4seam.parquet    ({len(fb)} rows)")
