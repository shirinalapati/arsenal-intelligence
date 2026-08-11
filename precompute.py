"""
Precompute all model outputs and save to data/ for fast Streamlit loading.
Fits separate models for Starters (SP) and Relievers (RP) within each pitch type.
Run once after fetching Statcast data.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import shap
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
FROZEN_SEASONS = {2023, 2024, 2025}
CURRENT_SEASON = 2026
ANALYSIS_SEASONS = {2023, 2024, 2025, CURRENT_SEASON}
POOLED_SEASON_SCOPE = "2023-2026"

# ── 1. Load raw data ─────────────────────────────────────────────────────────
print("Loading data...")
frames = []
for yr in [2023, 2024, 2025, CURRENT_SEASON]:
    p = os.path.join(DATA_DIR, f"statcast_{yr}.parquet")
    if os.path.exists(p):
        df_yr = pd.read_parquet(p)
        if "season" not in df_yr.columns:
            df_yr["season"] = yr
        frames.append(df_yr)
        print(f"  {yr}: {len(df_yr):,} pitches")
    elif yr == CURRENT_SEASON:
        print(f"  {yr}: no file (run fetch_statcast_2026.py)")

# CI / fresh clone: fall back to combined 2023–2025 bundle if present
if not any(
    os.path.exists(os.path.join(DATA_DIR, f"statcast_{y}.parquet")) for y in (2023, 2024, 2025)
):
    combined = os.path.join(DATA_DIR, "statcast_2023_2025.parquet")
    if os.path.exists(combined):
        print(f"  Loading training bundle → {combined}")
        bundle = pd.read_parquet(combined)
        if "season" not in bundle.columns:
            bundle["season"] = pd.to_numeric(bundle.get("game_date", "").astype(str).str[:4], errors="coerce")
        for yr in (2023, 2024, 2025):
            chunk = bundle[bundle["season"] == yr]
            if len(chunk):
                frames.append(chunk)
                print(f"  {yr}: {len(chunk):,} pitches (from bundle)")

if not frames:
    raise SystemExit("No Statcast parquet files found in data/")

raw = pd.concat(frames, ignore_index=True)
max_season = int(raw["season"].max())
print(f"  {len(raw):,} pitches loaded (seasons {sorted(raw['season'].unique())})")

# ── 2. Feature engineering ───────────────────────────────────────────────────
df = raw.copy()
df["pfx_x_in"] = pd.to_numeric(df["pfx_x"], errors="coerce") * 12
df["pfx_z_in"] = pd.to_numeric(df["pfx_z"], errors="coerce") * 12
df["break_arm"] = np.where(df["p_throws"] == "L", -df["pfx_x_in"], df["pfx_x_in"])
df["extension"] = pd.to_numeric(df["release_extension"], errors="coerce").fillna(6.0)
df["perceived_velo"] = (
    pd.to_numeric(df["release_speed"], errors="coerce") + (df["extension"] - 6.0) * 0.5
)

whiff_descs = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
swing_descs = whiff_descs | {"hit_into_play", "foul", "foul_bunt", "missed_bunt"}

df["is_swing"]   = df["description"].isin(swing_descs).fillna(False).astype(int)
df["is_whiff"]   = df["description"].isin(whiff_descs).fillna(False).astype(int)
df["is_csw"]     = df["description"].isin(whiff_descs | {"called_strike"}).fillna(False).astype(int)
zone_num         = pd.to_numeric(df["zone"], errors="coerce")
df["in_zone"]    = zone_num.between(1, 9).fillna(False)
df["is_chase"]   = ((df["is_swing"] == 1) & (~df["in_zone"])).astype(int)
launch           = pd.to_numeric(df["launch_speed"], errors="coerce")
df["is_hard_hit"]= (launch >= 95).fillna(False).astype(int)
df["xwoba"]      = pd.to_numeric(df["estimated_woba_using_speedangle"], errors="coerce")

PITCH_TYPE_MAP = {
    "FF": "Four-Seam FB", "FA": "Four-Seam FB",
    "SI": "Sinker",       "FT": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",       "ST": "Sweeper",
    "CU": "Curveball",    "KC": "Knuckle-Curve",
    "CH": "Changeup",     "FS": "Splitter",
    "SV": "Slurve",
}
df["pitch_group"] = df["pitch_type"].map(PITCH_TYPE_MAP)
df = df[df["pitch_group"].notna()].copy()

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
print(f"  {len(df_clean):,} clean pitches")

# ── 3. Role classification (SP vs RP) ────────────────────────────────────────
print("Classifying pitcher roles...")

# Manual role overrides for pitchers where the heuristic misclassifies them
# due to split-season usage (e.g. started early, returned from IL as a reliever).
# Format: {(pitcher_id, season): "SP" or "RP"}
ROLE_OVERRIDES = {
    (676477, 2023): "RP",   # Garrett Whitlock — 9 starts then returned from IL as RP
}

# Average pitches per game appearance per season
# Starters routinely throw 80-100 pitches; relievers 15-30
per_game = (
    df_clean.groupby(["pitcher", "game_date", "season"])
    .size()
    .reset_index(name="pitches_in_game")
)
avg_per_game = (
    per_game.groupby(["pitcher", "season"])["pitches_in_game"]
    .mean()
    .reset_index(name="avg_pitches_per_game")
)
# A pitcher is a starter in a given season if their avg pitches/game >= 50
avg_per_game["role"] = avg_per_game["avg_pitches_per_game"].apply(
    lambda x: "SP" if x >= 50 else "RP"
)
# Apply manual overrides
for (pid, ssn), override_role in ROLE_OVERRIDES.items():
    mask = (avg_per_game["pitcher"] == pid) & (avg_per_game["season"] == ssn)
    avg_per_game.loc[mask, "role"] = override_role
    if mask.any():
        print(f"  Override: pitcher {pid} season {ssn} → {override_role}")

# In-season 2026: supplement with MLB games started / games pitched when sample is thin
if max_season >= CURRENT_SEASON and (avg_per_game["season"] == CURRENT_SEASON).any():
    try:
        from team_utils import fetch_pitcher_roles_2026

        pids_2026 = avg_per_game.loc[avg_per_game["season"] == CURRENT_SEASON, "pitcher"].astype(int).tolist()
        role_map = fetch_pitcher_roles_2026(pids_2026, CURRENT_SEASON)
        for pid, role in role_map.items():
            mask = (avg_per_game["pitcher"] == pid) & (avg_per_game["season"] == CURRENT_SEASON)
            avg_per_game.loc[mask, "role"] = role
        print(f"  2026 MLB API role overrides: {len(role_map)} pitchers")
    except Exception as exc:
        print(f"  2026 role API skipped: {exc}")

df_clean = df_clean.merge(avg_per_game[["pitcher","season","role","avg_pitches_per_game"]],
                           on=["pitcher","season"], how="left")
df_clean["role"] = df_clean["role"].fillna("RP")

role_counts = df_clean.groupby("role").size()
print(f"  SP pitches: {role_counts.get('SP', 0):,} | RP pitches: {role_counts.get('RP', 0):,}")

# ── 4. Fit per-role × per-pitch-type logistic regression models ──────────────
print("\nFitting models (SP and RP separately)...")
swing_df = df_clean[df_clean["is_swing"] == 1].dropna(subset=STUFF_FEATURES).copy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Lower minimum for role-specific models
MIN_SWINGS = 2000

models    = {}   # (role, pitch_group) -> fitted pipeline
cv_aucs   = {}
coef_rows = []

for role in ["SP", "RP"]:
    role_swing = swing_df[swing_df["role"] == role]
    pitch_groups = (
        role_swing["pitch_group"].value_counts()
        [lambda s: s >= MIN_SWINGS].index.tolist()
    )
    print(f"\n  {role} pitch types: {pitch_groups}")
    for pg in pitch_groups:
        subset = role_swing[role_swing["pitch_group"] == pg]
        X, y   = subset[STUFF_FEATURES].values, subset["is_whiff"].values
        pipe   = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ])
        aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(X, y)
        models[(role, pg)] = pipe
        cv_aucs[(role, pg)] = aucs.mean()
        coefs = pipe.named_steps["lr"].coef_[0]
        for feat, coef in zip(STUFF_FEATURES, coefs):
            coef_rows.append({
                "role": role, "pitch_group": pg,
                "feature": feat, "feature_label": FEATURE_LABELS[feat],
                "coefficient": coef, "baseline_whiff": y.mean(),
                "auc": aucs.mean(),
            })
        print(f"    {role} {pg:20s} n={len(subset):>7,} | whiff%={y.mean():.1%} | AUC={aucs.mean():.3f}")

coef_df = pd.DataFrame(coef_rows)
coef_df.to_parquet(os.path.join(DATA_DIR, "model_coefs.parquet"), index=False)
print("\n  Saved model_coefs.parquet")

# ── 5. Score all pitches → Stuff+ (role-aware z-scoring) ────────────────────
print("Scoring pitches...")
all_df = df_clean.dropna(subset=STUFF_FEATURES).copy()
all_df["stuff_prob"] = np.nan

for (role, pg), pipe in models.items():
    mask = (all_df["role"] == role) & (all_df["pitch_group"] == pg)
    if mask.sum() == 0:
        continue
    all_df.loc[mask, "stuff_prob"] = pipe.predict_proba(
        all_df.loc[mask, STUFF_FEATURES].values
    )[:, 1]

# Z-score within role × pitch_group so 100 = avg for that role+pitch type
def to_stuffplus(s):
    mu, sd = s.mean(), s.std()
    return 100 + 10 * (s - mu) / (sd if sd > 0 else 1)

all_df["stuff_plus"] = all_df.groupby(["role","pitch_group"])["stuff_prob"].transform(to_stuffplus)

# Drop pitches with no model (rare pitch types in a role)
all_df = all_df[all_df["stuff_prob"].notna()].copy()
print(f"  {len(all_df):,} pitches scored")

# ── 6. Pitcher-season aggregates ─────────────────────────────────────────────
print("Aggregating pitcher stats...")
pitch_base = (
    all_df.groupby(["pitcher","player_name","season","role","pitch_group"])
    .agg(
        n_pitches  = ("stuff_plus","size"),
        stuff_plus = ("stuff_plus","mean"),
        csw_rate   = ("is_csw","mean"),
        avg_velo   = ("release_speed","mean"),
        avg_spin   = ("release_spin_rate","mean"),
        avg_ivb    = ("pfx_z_in","mean"),
        avg_hb     = ("pfx_x_in","mean"),
        avg_xwoba  = ("xwoba","mean"),
        avg_ext    = ("extension","mean"),
    )
    .reset_index()
)
whiff_by_pitch = (
    all_df[all_df["is_swing"] == 1]
    .groupby(["pitcher","season","role","pitch_group"])["is_whiff"]
    .mean().reset_index().rename(columns={"is_whiff":"whiff_rate"})
)
pitch_base = pitch_base.merge(
    whiff_by_pitch, on=["pitcher","season","role","pitch_group"], how="left"
)

# ── Merge frozen 2023–2025 rows; keep live 2026 ─────────────────────────────
def _merge_frozen(computed: pd.DataFrame, stem: str) -> pd.DataFrame:
    frozen_path = os.path.join(DATA_DIR, f"frozen_{stem}_2023_2025.parquet")
    if not os.path.exists(frozen_path):
        return computed
    frozen = pd.read_parquet(frozen_path)
    live = computed[computed["season"] == CURRENT_SEASON].copy()
    if live.empty:
        return frozen
    for col in computed.columns:
        if col not in frozen.columns:
            frozen[col] = None if computed[col].dtype == object else np.nan
    for col in frozen.columns:
        if col not in live.columns:
            live[col] = None if frozen[col].dtype == object else np.nan
    cols = [c for c in computed.columns if c in frozen.columns and c in live.columns]
    return pd.concat([frozen[cols], live[cols]], ignore_index=True)

# Arsenal-level (usage-weighted avg Stuff+) — use full computed pitch_base
pitch_base["weighted"] = pitch_base["n_pitches"] * pitch_base["stuff_plus"]
arsenal = (
    pitch_base.groupby(["pitcher","player_name","season","role"])
    .agg(
        weighted_sum  = ("weighted","sum"),
        total_pitches = ("n_pitches","sum"),
        avg_velo      = ("avg_velo","mean"),
        avg_spin      = ("avg_spin","mean"),
        avg_ivb       = ("avg_ivb","mean"),
        avg_hb        = ("avg_hb","mean"),
        avg_xwoba     = ("avg_xwoba","mean"),
    )
    .reset_index()
)
arsenal["arsenal_stuff"] = arsenal["weighted_sum"] / arsenal["total_pitches"]
arsenal = arsenal.drop(columns=["weighted_sum"])

season_csw = (
    all_df.groupby(["pitcher","season","role"])
    .agg(csw_rate=("is_csw","mean"))
    .reset_index()
)
season_whiff = (
    all_df[all_df["is_swing"] == 1]
    .groupby(["pitcher","season","role"])["is_whiff"]
    .mean().reset_index().rename(columns={"is_whiff":"whiff_rate"})
)
arsenal = (
    arsenal
    .merge(season_csw,   on=["pitcher","season","role"], how="left")
    .merge(season_whiff, on=["pitcher","season","role"], how="left")
)

# Low-sample flag for live 2026 season
arsenal["low_sample"] = False
s26 = arsenal[arsenal["season"] == CURRENT_SEASON]
for role, floor in [("SP", 150), ("RP", 80)]:
    sub = s26[s26["role"] == role]
    if sub.empty:
        continue
    th = max(floor, float(sub["total_pitches"].median()) * 0.35)
    mask = (arsenal["season"] == CURRENT_SEASON) & (arsenal["role"] == role) & (arsenal["total_pitches"] < th)
    arsenal.loc[mask, "low_sample"] = True

arsenal_live = arsenal.copy()
arsenal = _merge_frozen(arsenal, "arsenal_scores")
if "low_sample" not in arsenal.columns:
    arsenal["low_sample"] = False
try:
    from team_utils import attach_teams
    if "team" not in arsenal.columns or arsenal["team"].isna().all():
        arsenal = attach_teams(arsenal)
    else:
        missing = arsenal["team"].isna() | (arsenal["team"] == "")
        if missing.any():
            filled = attach_teams(arsenal.loc[missing].drop(columns=["team"], errors="ignore"))
            arsenal.loc[missing, "team"] = filled["team"].values
except Exception as exc:
    print(f"  Team lookup skipped for arsenal_scores: {exc}")
    if "team" not in arsenal.columns:
        arsenal["team"] = ""

arsenal.to_parquet(os.path.join(DATA_DIR,"arsenal_scores.parquet"), index=False)
print("  Saved arsenal_scores.parquet")

pt_out = _merge_frozen(pitch_base.drop(columns=["weighted"], errors="ignore"), "pitch_type_scores")
try:
    from team_utils import attach_teams
    pt_out = attach_teams(pt_out)
except Exception as exc:
    print(f"  Team lookup skipped for pitch_type_scores: {exc}")
    if "team" not in pt_out.columns:
        pt_out["team"] = ""
pt_out.to_parquet(os.path.join(DATA_DIR, "pitch_type_scores.parquet"), index=False)
print("  Saved pitch_type_scores.parquet")

# ── 7. Decile validation (per role, pooled 2023–2026) ───────────────────────
pool_df = all_df[all_df["season"].isin(ANALYSIS_SEASONS)]
decile_frames = []
for role in ["SP", "RP", "ALL"]:
    sub = pool_df if role == "ALL" else pool_df[pool_df["role"] == role]
    sub = sub.copy()
    sub["stuff_decile"] = (
        pd.qcut(sub["stuff_plus"], q=10, labels=False, duplicates="drop")
        .astype(float).fillna(0).astype(int) + 1
    )
    base = sub.groupby("stuff_decile").agg(
        n=("stuff_plus","size"), csw_rate=("is_csw","mean"),
        avg_xwoba=("xwoba","mean"), avg_stuff_plus=("stuff_plus","mean")
    )
    wh = (
        sub[sub["is_swing"]==1]
        .groupby("stuff_decile")["is_whiff"].mean().rename("whiff_rate")
    )
    d = base.join(wh).reset_index()
    d["role"] = role
    d["season_scope"] = POOLED_SEASON_SCOPE
    decile_frames.append(d)

decile_df = pd.concat(decile_frames, ignore_index=True)
decile_df.to_parquet(os.path.join(DATA_DIR,"decile_outcomes.parquet"), index=False)
print("  Saved decile_outcomes.parquet")

# ── 8. Pitch-type summary (per role, pooled 2023–2026) ──────────────────────
pool_clean = df_clean[df_clean["season"].isin(ANALYSIS_SEASONS)]
pt_frames = []
swings = pool_clean[pool_clean["is_swing"] == 1]
for role in ["SP", "RP", "ALL"]:
    sub       = pool_clean if role == "ALL" else pool_clean[pool_clean["role"] == role]
    sub_sw    = swings   if role == "ALL" else swings[swings["role"] == role]
    pt_base   = sub.groupby("pitch_group").agg(
        n_pitches=("is_swing","size"), csw_rate=("is_csw","mean"),
        avg_velo=("release_speed","mean"), avg_spin=("release_spin_rate","mean"),
        avg_ivb=("pfx_z_in","mean"), avg_hb=("pfx_x_in","mean"), avg_xwoba=("xwoba","mean")
    )
    pt_whiff  = sub_sw.groupby("pitch_group")["is_whiff"].mean().rename("whiff_rate")
    auc_series = pd.Series(
        {pg: cv_aucs.get((role, pg), cv_aucs.get(("SP", pg), cv_aucs.get(("RP", pg), np.nan)))
         for pg in pt_base.index},
        name="auc"
    )
    pt = pt_base.join(pt_whiff).join(auc_series).reset_index()
    pt["role"] = role
    pt["season_scope"] = POOLED_SEASON_SCOPE
    pt_frames.append(pt)

pt_summary = pd.concat(pt_frames, ignore_index=True)
pt_summary.to_parquet(os.path.join(DATA_DIR,"pitch_type_summary.parquet"), index=False)
print("  Saved pitch_type_summary.parquet")

# ── 9. SHAP models — all pitch types, per role ───────────────────────────────
print("\nTraining SHAP models for all pitch types (SP and RP)...")
MIN_SHAP_SWINGS = 1000   # minimum swings needed to train a GBC for a pitch type
shap_outputs = {}        # (role, pitch_group) -> (shap_df, explainer, scaler, gbc)
shap_global_frames = []

all_pitch_groups = sorted(swing_df["pitch_group"].unique())

for role in ["SP", "RP"]:
    for pg in all_pitch_groups:
        pt_swings = swing_df[
            (swing_df["pitch_group"] == pg) & (swing_df["role"] == role)
        ].dropna(subset=STUFF_FEATURES)
        if len(pt_swings) < MIN_SHAP_SWINGS:
            print(f"  {role} {pg}: only {len(pt_swings)} swings — skipping SHAP")
            continue

        pt_sample = pt_swings.sample(n=min(60_000, len(pt_swings)), random_state=42)
        X_pt = pt_sample[STUFF_FEATURES].values
        y_pt = pt_sample["is_whiff"].values

        scaler_pt = StandardScaler().fit(X_pt)
        X_sc = scaler_pt.transform(X_pt)

        gbc = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, min_samples_leaf=40
        )
        gbc.fit(X_sc, y_pt)

        pkl_name = f"shap_gbc_{role.lower()}_{pg.replace(' ','_').replace('-','').lower()}.pkl"
        with open(os.path.join(DATA_DIR, pkl_name), "wb") as f:
            pickle.dump({"gbc": gbc, "scaler": scaler_pt}, f)

        explainer = shap.TreeExplainer(gbc)
        shap_sample = pt_swings.sample(n=min(6_000, len(pt_swings)), random_state=1)
        X_shap_raw = shap_sample[STUFF_FEATURES].values
        sv = explainer.shap_values(scaler_pt.transform(X_shap_raw))

        shap_df = pd.DataFrame(X_shap_raw, columns=[f"val_{c}" for c in STUFF_FEATURES])
        for i, feat in enumerate(STUFF_FEATURES):
            shap_df[f"shap_{feat}"] = sv[:, i]
        shap_df["role"]        = role
        shap_df["pitch_group"] = pg

        shap_outputs[(role, pg)] = (shap_df, explainer, scaler_pt, gbc)
        shap_global_frames.append(shap_df)
        print(f"  {role} {pg:20s} n={len(pt_swings):>7,} | trained OK")

shap_global = pd.concat(shap_global_frames, ignore_index=True)
shap_global.to_parquet(os.path.join(DATA_DIR, "shap_values_allpitch.parquet"), index=False)

# Keep backward-compat file for Four-Seam FB (used by How It Works beeswarm)
fb_shap = shap_global[shap_global["pitch_group"] == "Four-Seam FB"]
fb_shap.to_parquet(os.path.join(DATA_DIR, "shap_values_4seam.parquet"), index=False)
print(f"  Saved shap_values_allpitch.parquet ({len(shap_global):,} rows)")
print(f"  Saved shap_values_4seam.parquet    ({len(fb_shap):,} rows, backward compat)")

# ── 10. Per-pitcher SHAP breakdown — all pitch types, all seasons ────────────
print("Computing per-pitcher SHAP breakdown (all pitch types, all seasons)...")
all_seasons = sorted(all_df["season"].unique())
MIN_PITCHER_PITCHES = 100
pitcher_shap_frames = []

for season in all_seasons:
    for (role, pg), (_, explainer, scaler_pt, gbc) in shap_outputs.items():
        pt_all = all_df[
            (all_df["pitch_group"] == pg) &
            (all_df["season"] == season) &
            (all_df["role"] == role)
        ].dropna(subset=STUFF_FEATURES)
        qual = pt_all.groupby("pitcher").size()
        qual = qual[qual >= MIN_PITCHER_PITCHES].index
        pt_qual = pt_all[pt_all["pitcher"].isin(qual)].copy()
        if pt_qual.empty:
            continue

        sv_all = explainer.shap_values(scaler_pt.transform(pt_qual[STUFF_FEATURES].values))
        for i, feat in enumerate(STUFF_FEATURES):
            pt_qual[f"shap_{feat}"] = sv_all[:, i]

        shap_cols = [f"shap_{f}" for f in STUFF_FEATURES]
        ps = (
            pt_qual.groupby(["pitcher", "player_name"])
            .agg({**{c: "mean" for c in shap_cols},
                  "stuff_plus": "mean", "is_whiff": "mean", "release_speed": "size"})
            .rename(columns={"release_speed": "n_pitches", "is_whiff": "whiff_rate"})
            .reset_index()
        )
        ps.rename(columns={f"shap_{f}": FEATURE_LABELS[f] for f in STUFF_FEATURES}, inplace=True)
        ps["season"]      = season
        ps["role"]        = role
        ps["pitch_group"] = pg
        pitcher_shap_frames.append(ps)

pitcher_shap = pd.concat(pitcher_shap_frames, ignore_index=True)
pitcher_shap = _merge_frozen(pitcher_shap, "pitcher_shap_allpitch")
pitcher_shap.to_parquet(os.path.join(DATA_DIR, "pitcher_shap_allpitch.parquet"), index=False)

# Backward-compat Four-Seam file
fb_pitcher_shap = pitcher_shap[pitcher_shap["pitch_group"] == "Four-Seam FB"]
fb_pitcher_shap.to_parquet(os.path.join(DATA_DIR, "pitcher_shap_4seam.parquet"), index=False)
print(f"  Saved pitcher_shap_allpitch.parquet ({len(pitcher_shap)} rows, {all_seasons})")
print(f"  Saved pitcher_shap_4seam.parquet    ({len(fb_pitcher_shap)} rows, backward compat)")

from datetime import datetime, timezone
(Path := __import__("pathlib").Path)(DATA_DIR).joinpath("last_updated.txt").write_text(
    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
)

# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("All outputs saved.")
print(f"  Seasons: 2023–{max_season}  |  Roles: SP, RP")
sp_count = arsenal[(arsenal["season"]==max_season) & (arsenal["role"]=="SP")]["total_pitches"].count()
rp_count = arsenal[(arsenal["season"]==max_season) & (arsenal["role"]=="RP")]["total_pitches"].count()
print(f"  {max_season} qualified pitchers: {sp_count} SP | {rp_count} RP")
