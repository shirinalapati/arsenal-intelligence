"""
MLB Pitch Stuff+ Dashboard
An interpretable pitch quality scoring app built on 2023–2026 Statcast data.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from team_filter import MLB_TEAMS, pitcher_matches_team

try:
    import statsmodels.api  # noqa: F401 — required by plotly trendline="ols"

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def px_scatter_trendline(df: pd.DataFrame, **kwargs):
    """Scatter plot with OLS trendline when statsmodels is available."""
    tl = "ols" if _HAS_STATSMODELS and len(df) >= 3 else None
    try:
        return px.scatter(df, trendline=tl, **kwargs)
    except Exception:
        return px.scatter(df, trendline=None, **kwargs)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arsenal Intelligence",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colors ─────────────────────────────────────────────────────────────
ACCENT_COLOR   = "#4CBB17"
DARK_BG        = "#0E1117"
CARD_BG        = "#1A1D27"
TEXT_MUTED     = "#8B9DC3"
WHITE          = "#FFFFFF"

st.markdown(f"""
<style>
    /* global */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {DARK_BG};
        color: {WHITE};
    }}
    [data-testid="stSidebar"] {{
        background-color: {CARD_BG};
    }}
    /* metric cards */
    [data-testid="stMetric"] {{
        background: {CARD_BG};
        border-radius: 10px;
        padding: 12px 18px;
        border: 1px solid #2A2D3A;
    }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; font-size: 0.8rem; }}
    [data-testid="stMetricValue"] {{ color: {WHITE} !important; font-size: 1.8rem; font-weight: 700; }}
    /* section headers */
    .section-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {ACCENT_COLOR};
        padding-bottom: 4px;
        margin: 20px 0 12px 0;
    }}
    /* badges */
    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        background: {ACCENT_COLOR};
        color: white;
        margin-right: 4px;
    }}
    /* tables */
    thead tr th {{ background-color: {CARD_BG} !important; color: {ACCENT_COLOR} !important; }}
    /* selectbox / slider */
    .stSelectbox label, .stSlider label {{ color: {TEXT_MUTED}; }}
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data(ttl=120)
def load_data():
    arsenal     = pd.read_parquet(os.path.join(DATA_DIR, "arsenal_scores.parquet"))
    pt_scores   = pd.read_parquet(os.path.join(DATA_DIR, "pitch_type_scores.parquet"))
    coefs       = pd.read_parquet(os.path.join(DATA_DIR, "model_coefs.parquet"))
    decile      = pd.read_parquet(os.path.join(DATA_DIR, "decile_outcomes.parquet"))
    pt_summary  = pd.read_parquet(os.path.join(DATA_DIR, "pitch_type_summary.parquet"))
    # Load all-pitch SHAP if available, fall back to 4-seam only
    allpitch_path = os.path.join(DATA_DIR, "shap_values_allpitch.parquet")
    pitcher_allpitch_path = os.path.join(DATA_DIR, "pitcher_shap_allpitch.parquet")
    if os.path.exists(allpitch_path):
        shap_global = pd.read_parquet(allpitch_path)
    else:
        shap_global = pd.read_parquet(os.path.join(DATA_DIR, "shap_values_4seam.parquet"))
        shap_global["pitch_group"] = "Four-Seam FB"
    if os.path.exists(pitcher_allpitch_path):
        shap_pitch = pd.read_parquet(pitcher_allpitch_path)
    else:
        shap_pitch = pd.read_parquet(os.path.join(DATA_DIR, "pitcher_shap_4seam.parquet"))
        shap_pitch["pitch_group"] = "Four-Seam FB"
    return arsenal, pt_scores, coefs, decile, pt_summary, shap_global, shap_pitch

arsenal, pt_scores, coefs, decile, pt_summary, shap_global, shap_pitch = load_data()

SEASONS       = sorted(arsenal["season"].unique(), reverse=True)
CURRENT_SEASON = max(SEASONS) if len(SEASONS) else 2026
POOLED_SEASONS_LABEL = "2023–2026"
POOLED_SEASONS_KEY = "2023-2026"

def _filter_pooled(df: pd.DataFrame, role_code: str) -> pd.DataFrame:
    """Rows for pooled 2023–2026 aggregates (decile validation, pitch-type table)."""
    role = role_code if role_code != "ALL" else "ALL"
    out = df[df["role"] == role]
    if "season_scope" in out.columns:
        out = out[out["season_scope"] == POOLED_SEASONS_KEY]
    return out

_pooled_pt = _filter_pooled(pt_summary, "ALL")
TOTAL_PITCHES = int(_pooled_pt["n_pitches"].sum()) if not _pooled_pt.empty else 0
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LAST_UPDATED_PATH = os.path.join(DATA_DIR, "last_updated.txt")
last_updated = ""
if os.path.exists(LAST_UPDATED_PATH):
    last_updated = open(LAST_UPDATED_PATH).read().strip()

RANK_PAGE_TITLES = {
    "arsenal_stuff": "Full Arsenal Stuff+ Rankings",
    "whiff_rate": "Full Whiff Rate Rankings",
    "csw_rate": "Full CSW Rate Rankings",
    "avg_xwoba": "Full xwOBA Against Rankings",
}
PITCH_GROUPS  = sorted(pt_scores["pitch_group"].unique())
FEATURE_LABELS = [
    "Perceived Velocity", "Induced Vertical Break", "Arm-Side Horiz Break",
    "Spin Rate", "Release Extension", "Horizontal Location", "Vertical Location"
]

def fmt_name(name):
    """Convert 'Last, First' → 'First Last' for display."""
    if isinstance(name, str) and ", " in name:
        last, first = name.split(", ", 1)
        return f"{first} {last}"
    return name


def stuffplus_color(val):
    if val >= 115: return "#00C851"
    if val >= 110: return "#7BCF5C"
    if val >= 105: return "#B8D96A"
    if val >= 100: return "#FFD700"
    if val >= 95:  return "#FFA040"
    return "#FF4444"

GLOSSARY = {
    "Score Metrics": [
        ("Stuff+", "The core score of this dashboard. A normalized measure of pitch quality based purely on physical characteristics — velocity, movement, spin, and location. Scaled so that 100 = league average for that pitch type and role, and each 10 points = one standard deviation. A Stuff+ of 115 means the pitch is 1.5 standard deviations above average for its type."),
        ("Arsenal Stuff+", "A pitcher's overall stuff score, calculated as the usage-weighted average of their individual pitch Stuff+ scores across all pitch types they throw. This is the headline number on the Leaderboard and Pitcher Explorer pages."),
        ("Whiff Rate", "The percentage of swings that result in a swing-and-miss (swinging strike). Calculated as swinging strikes ÷ total swings. Does not include called strikes or foul balls. A high whiff rate indicates a pitch is difficult to make contact with when a batter decides to swing."),
        ("CSW Rate", "Called Strike + Whiff rate. The percentage of all pitches (not just swings) that result in either a called strike or a swinging strike. CSW% = (called strikes + swinging strikes) ÷ total pitches. A broader measure of how often a pitch produces a strike of any kind."),
        ("xwOBA", "Expected Weighted On-Base Average. A Statcast metric that estimates the quality of contact based on exit velocity and launch angle, regardless of what actually happened (park effects, fielder positioning, etc.). Lower xwOBA against means weaker contact quality allowed. Scale: 0.000 (perfect) to ~0.500+ (very hittable). League average is typically around 0.315–0.320."),
    ],
    "Physical Pitch Characteristics": [
        ("Perceived Velocity", "A pitcher's effective velocity as perceived by the batter. Calculated as release speed + an extension bonus: each foot of release extension beyond 6 ft adds approximately 0.5 mph. A pitcher who releases the ball at 6.5 ft extension effectively 'speeds up' their 95 mph pitch to a perceived 95.25 mph."),
        ("Release Speed", "Raw pitch velocity in mph at the point of release from the pitcher's hand, as measured by Statcast radar/cameras."),
        ("Induced Vertical Break (iVB)", "The vertical movement on a pitch caused by spin, measured in inches, after removing the effect of gravity. A positive iVB means the pitch rises relative to a ball thrown with no spin (e.g., four-seam fastballs typically have +12 to +18 inches of ride). A negative iVB means the pitch drops more than gravity alone (e.g., curveballs have −5 to −15 inches)."),
        ("Horizontal Break (pfx_x)", "The horizontal movement of a pitch in inches, relative to a no-spin pitch. Positive values move toward the arm side for RHP (inside to righties), negative toward the glove side. In this model, 'Arm-Side Horiz Break' is adjusted for pitcher handedness so positive always means arm-side movement."),
        ("Arm-Side Horiz Break", "Horizontal break adjusted so that positive = movement toward the pitcher's arm side, regardless of whether they throw right or left. This makes comparisons across handedness meaningful."),
        ("Spin Rate", "How fast the ball spins at release, measured in rotations per minute (RPM). Higher spin generally produces more movement (more ride on fastballs, more break on curveballs), though the relationship depends on spin axis and spin efficiency. Typical ranges: fastballs 2,100–2,600 RPM; curveballs 2,400–3,000 RPM; changeups 1,500–1,900 RPM."),
        ("Release Extension", "How far toward home plate the pitcher releases the ball, measured in feet from the pitching rubber. Higher extension effectively shortens the batter's reaction time. League average is around 6.0–6.2 ft; elite extension pitchers reach 7+ ft."),
        ("Plate Location (plate_x, plate_z)", "Where the pitch crosses the front of home plate. plate_x is horizontal position in feet (negative = glove side for RHP), plate_z is height in feet above the ground. Used to capture the contribution of pitch location — not just pitch movement — to whiff probability."),
    ],
    "Model & Explainability": [
        ("Logistic Regression", "A statistical model that predicts the probability of a binary outcome (whiff vs. no whiff). It learns a linear combination of input features — essentially a weighted recipe — making it directly interpretable: the coefficient for each feature tells you how much that feature increases or decreases the predicted probability of a whiff."),
        ("L2 Regularization (Ridge)", "A penalty applied during model training that shrinks large coefficients toward zero. Prevents overfitting to noise in the data, especially useful when features are correlated (e.g., velocity and perceived velocity)."),
        ("AUC (Area Under the ROC Curve)", "A model performance metric measuring how well the model distinguishes between whiffs and non-whiffs across all possible thresholds. 0.5 = no better than random, 1.0 = perfect. In this project, values range from 0.54 (sinker, which isn't a whiff pitch) to 0.82 (knuckle-curve). The higher the AUC, the more the physical characteristics of that pitch type drive whiff outcomes."),
        ("SHAP Values", "SHapley Additive exPlanations — a game-theory-based method for explaining individual predictions from a machine learning model. Each SHAP value represents how much a specific feature pushed a particular pitch's predicted whiff probability higher or lower. Positive SHAP = the feature increased the predicted whiff probability for that pitch; negative SHAP = it decreased it."),
        ("Gradient Boosted Model (GBM)", "A more powerful but less transparent ML model used alongside the logistic regression. The GBM can capture nonlinear interactions (e.g., the velocity effect might be different at 98 mph than at 90 mph). SHAP values from the GBM are used to audit and deepen the interpretability of the simpler LR model."),
        ("Z-Score Normalization", "A statistical transformation that converts raw predicted probabilities into a scale relative to the average and spread for that group. Formula: Stuff+ = 100 + 10 × (raw_prob − group_mean) / group_std. Ensures that 100 always means average and 10 points always means one standard deviation, regardless of pitch type or role."),
        ("5-Fold Cross-Validation", "A method for evaluating model performance without overfitting to the training data. The dataset is split into 5 equal parts; the model is trained on 4 parts and tested on the held-out 5th, repeated 5 times. The AUC scores shown are averages across all 5 folds, giving a reliable estimate of real-world performance."),
    ],
    "Pitch Types": [
        ("Four-Seam Fastball (FF)", "The standard fastball. Thrown with backspin, producing ride (high iVB). Typically the hardest pitch in a pitcher's arsenal. Generates whiffs primarily through high location and elite velocity."),
        ("Sinker / Two-Seamer (SI)", "A fastball variant with more arm-side run and less vertical ride than a four-seamer. Designed to generate ground balls rather than swings-and-misses — reflected in the model's low AUC (0.54) for this pitch type."),
        ("Cutter (FC)", "A fastball with late glove-side cut, typically 87–92 mph. Harder than a slider but with less break. Generates weak contact and soft hits rather than swing-and-miss."),
        ("Slider (SL)", "An offspeed breaking ball with glove-side horizontal break and moderate velocity (82–88 mph). One of the highest whiff-rate pitch types. Model AUC of 0.735."),
        ("Sweeper (ST)", "A slider variant with extreme horizontal sweep (10+ inches) and lower velocity. Became prominent in the early 2020s. Generates whiffs through horizontal deception."),
        ("Curveball (CU)", "A breaking ball with large downward vertical break and lower velocity (74–82 mph). High spin rate drives the sharp drop. Model AUC of 0.779 — physical characteristics strongly predict whiffs."),
        ("Knuckle-Curve (KC)", "A curveball variant with a knuckleball-style grip, producing a tighter, later-breaking curve. Often has higher spin and sharper break than a standard curveball. Highest AUC in this dataset (0.819)."),
        ("Changeup (CH)", "An offspeed pitch designed to match fastball arm action while arriving 8–12 mph slower. The deception (not the movement) generates whiffs. Arm-side fade is typical."),
        ("Splitter (FS)", "A pitch gripped between the index and middle fingers, producing significant downward drop with lower spin. Common in East Asian pitching traditions. High whiff rate (34%+) due to late diving action."),
    ],
    "Roles": [
        ("SP — Starter", "A pitcher whose primary role is to begin the game and pitch multiple innings, typically 5–7. Starters pace themselves, vary pitch usage across a lineup over multiple looks, and are classified here as averaging ≥ 50 pitches per game appearance."),
        ("RP — Reliever", "A pitcher who enters after the starter, typically for 1–2 innings. Relievers throw at maximum effort for shorter outings, which inflates velocity and can inflate whiff rates. Classified here as averaging < 50 pitches per game appearance. Separate Stuff+ models are fit for SP and RP so scores are always compared within the correct peer group."),
    ],
}

def show_glossary():
    """Render the collapsible glossary expander — called on every page."""
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖  Show / Hide Glossary", expanded=False):
        for section, entries in GLOSSARY.items():
            st.markdown(f"""
            <div style='color:{ACCENT_COLOR}; font-weight:700; font-size:1rem;
                        margin: 20px 0 8px 0; border-bottom:1px solid #2A2D3A; padding-bottom:6px;'>
                {section}
            </div>
            """, unsafe_allow_html=True)
            for i, (term, definition) in enumerate(entries):
                bg = CARD_BG if i % 2 == 0 else DARK_BG
                st.markdown(f"""
                <div style='display:flex; padding:12px 16px; background:{bg};
                            border-radius:8px; margin-bottom:4px; align-items:flex-start;'>
                    <div style='color:{WHITE}; font-weight:700; font-size:0.88rem;
                                min-width:220px; padding-right:20px; padding-top:1px;
                                flex-shrink:0;'>{term}</div>
                    <div style='color:{TEXT_MUTED}; font-size:0.87rem;
                                line-height:1.65;'>{definition}</div>
                </div>
                """, unsafe_allow_html=True)

# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 8px 0 20px 0;'>
        <span style='font-size:2.4rem;'>⚾</span>
        <div style='font-size:1.3rem; font-weight:800; color:{ACCENT_COLOR}; margin-top:4px;'>
            Arsenal Intelligence
        </div>
        <div style='font-size:0.75rem; color:{TEXT_MUTED};'>
            MLB Pitch Quality Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["About", "Overview", "Pitcher Explorer", "Leaderboard", "How the Model Works + Important Findings"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    # Pages where filters are irrelevant (static content only)
    _no_filters = page in ("About", "How the Model Works + Important Findings")

    # Season filter — hidden on About and How the Model Works tabs
    if not _no_filters:
        _override = st.session_state.pop("_season_override_idx", None)
        if _override is not None:
            season_filter = st.selectbox("Season", SEASONS, index=_override, key="season_sel")
        else:
            season_filter = st.selectbox("Season", SEASONS, index=0, key="season_sel")
    else:
        st.session_state.pop("_season_override_idx", None)
        season_filter = SEASONS[0]  # default (unused on these tabs)

    # Role filter — hidden on About tab; still shown on How the Model Works (affects heatmap/SHAP)
    if page != "About":
        role_filter = st.radio(
            "Role",
            ["All", "SP — Starters", "RP — Relievers"],
            index=0,
        )
        ROLE_CODE = {"All": "ALL", "SP — Starters": "SP", "RP — Relievers": "RP"}[role_filter]
    else:
        ROLE_CODE = "ALL"  # default (unused on About tab)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown(f"<div style='color:{ACCENT_COLOR}; font-size:2.8rem; font-weight:800; letter-spacing:0.04em; margin-bottom:2px;'>Arsenal Intelligence</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:{WHITE}; margin-bottom:0;'>Overview</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:1rem; margin-top:4px;'>An interpretable pitch quality model built on 2023–{CURRENT_SEASON} Statcast data</p>", unsafe_allow_html=True)
    if last_updated:
        st.caption(f"Data last updated: **{last_updated}** (Statcast refresh)")
    st.divider()

    # ── Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    season_arsenal = arsenal[arsenal["season"] == season_filter]
    if ROLE_CODE != "ALL":
        season_arsenal = season_arsenal[season_arsenal["role"] == ROLE_CODE]
    role_label = {"ALL": "All pitchers", "SP": "Starters", "RP": "Relievers"}[ROLE_CODE]
    with col1:
        n_p = int(season_arsenal["total_pitches"].sum()) if not season_arsenal.empty else 0
        st.metric("Pitches Analyzed", f"{n_p:,}", f"{season_filter} · {role_label}")
    with col2:
        st.metric("Models", "SP + RP", "separate per pitch type")
    with col3:
        if not season_arsenal.empty:
            best = season_arsenal.loc[season_arsenal["arsenal_stuff"].idxmax(), "player_name"]
            best_val = season_arsenal["arsenal_stuff"].max()
            st.metric("Top Arsenal Stuff+", f"{best_val:.1f}", best)
        else:
            st.metric("Top Arsenal Stuff+", "—")
    with col4:
        min_p = 200 if ROLE_CODE == "RP" else 500
        qual = season_arsenal[season_arsenal["total_pitches"] >= min_p]
        st.metric("Qualified Pitchers", len(qual), f"≥ {min_p} pitches")

    st.markdown(f"<div class='section-header'>Stuff+ vs Outcomes — Decile Validation ({POOLED_SEASONS_LABEL})</div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.88rem;'>"
        f"Pooled across {POOLED_SEASONS_LABEL} (including live {CURRENT_SEASON} in-season data). "
        f"Pitches binned into 10 equal groups by Stuff+ score. Higher stuff predicts more whiffs and suppressed contact quality."
        f"</p>",
        unsafe_allow_html=True,
    )

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Whiff Rate (on swings)", "CSW Rate", "xwOBA Against"],
        horizontal_spacing=0.08
    )
    palette = px.colors.diverging.RdYlGn
    step = len(palette) // 10
    decile_filtered = _filter_pooled(decile, ROLE_CODE)

    for d_row in decile_filtered.itertuples():
        color = palette[min((d_row.stuff_decile - 1) * step, len(palette)-1)]
        d_label = str(int(d_row.stuff_decile))
        fig.add_trace(go.Bar(x=[d_label], y=[d_row.whiff_rate], marker_color=color, showlegend=False, name=d_label), 1, 1)
        fig.add_trace(go.Bar(x=[d_label], y=[d_row.csw_rate],   marker_color=color, showlegend=False, name=d_label), 1, 2)
        fig.add_trace(go.Bar(x=[d_label], y=[d_row.avg_xwoba],  marker_color=color, showlegend=False, name=d_label), 1, 3)

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=WHITE), height=340,
        margin=dict(t=40, b=30, l=10, r=10),
    )
    for i, fmt in enumerate([".0%", ".0%", ".3f"], start=1):
        fig.update_yaxes(tickformat=fmt, gridcolor="#2A2D3A", row=1, col=i)
        fig.update_xaxes(title_text="Stuff+ Decile →", row=1, col=i)
    st.plotly_chart(fig, use_container_width=True)

    desc_col1, desc_col2, desc_col3 = st.columns(3)
    with desc_col1:
        st.caption(
            "**Whiff Rate (on swings):** Of all pitches in each Stuff+ decile that a batter swung at, "
            "what percentage resulted in a complete miss. Decile 1 = lowest Stuff+ pitches, "
            "Decile 10 = highest. A clear upward trend here means higher Stuff+ directly predicts "
            "more swing-and-miss."
        )
    with desc_col2:
        st.caption(
            "**CSW Rate (Called Strike + Whiff):** The percentage of all pitches — swings and takes — "
            "that produced either a swinging strike or a called strike. This is broader than whiff rate "
            "since it includes pitches the batter didn't swing at. The non-monotonic pattern (dip at "
            "decile 10) reflects that elite pitches are often thrown out of the zone on purpose, "
            "generating whiffs but fewer called strikes."
        )
    with desc_col3:
        st.caption(
            "**xwOBA Against:** Expected weighted on-base average on contact, based on exit velocity "
            "and launch angle. Lower = weaker contact allowed. The downward trend confirms that higher "
            "Stuff+ pitches suppress quality of contact, not just swing-and-miss — validating the score "
            "against a second independent outcome."
        )

    col_left, col_right = st.columns(2)

    # ── Pitch type summary table
    with col_left:
        st.markdown(f"<div class='section-header'>Pitch Type Outcomes ({POOLED_SEASONS_LABEL})</div>", unsafe_allow_html=True)
        st.caption(
            f"League-wide averages pooled across {POOLED_SEASONS_LABEL}, including live {CURRENT_SEASON} data. "
            "Use the season selector for year-by-year distributions in the chart on the right."
        )
        disp = _filter_pooled(pt_summary, ROLE_CODE)[
            ["pitch_group","n_pitches","avg_velo","avg_spin","avg_ivb","whiff_rate","csw_rate","avg_xwoba","auc"]
        ].copy()
        disp = disp.sort_values("whiff_rate", ascending=False)
        disp.columns = ["Pitch Type","Pitches","Velo","Spin","iVB","Whiff%","CSW%","xwOBA","Model AUC"]
        disp["Whiff%"] = disp["Whiff%"].map("{:.1%}".format)
        disp["CSW%"]   = disp["CSW%"].map("{:.1%}".format)
        disp["xwOBA"]  = disp["xwOBA"].map("{:.3f}".format)
        disp["Model AUC"] = disp["Model AUC"].map("{:.3f}".format)
        disp["Velo"]   = disp["Velo"].map("{:.1f}".format)
        disp["Spin"]   = disp["Spin"].map("{:.0f}".format)
        disp["iVB"]    = disp["iVB"].map("{:.1f}\"".format)
        disp["Pitches"] = disp["Pitches"].map("{:,.0f}".format)
        st.dataframe(disp, hide_index=True, use_container_width=True)

    # ── Whiff rate by pitch type bar
    with col_right:
        role_label = {"ALL": "All Pitchers", "SP": "Starters (SP)", "RP": "Relievers (RP)"}[ROLE_CODE]
        st.markdown("<div class='section-header'>League-Wide Stuff+ Distribution by Pitch Type (Year by Year)</div>", unsafe_allow_html=True)
        st.caption(
            f"Every individual pitch thrown in **{season_filter}** by {role_label} — pooled across all pitchers. "
            f"Use the **Season** selector in the sidebar (defaults to {CURRENT_SEASON}, the live season). "
            "Each violin shows how spread out Stuff+ scores are for that pitch type. "
            "All types are centered at 100 by design; width shows variance, tails show outliers. "
            "To see a specific pitcher's distribution, use the Pitcher Explorer page."
        )
        pts = pt_scores[pt_scores["season"] == season_filter]
        if ROLE_CODE != "ALL":
            pts = pts[pts["role"] == ROLE_CODE]
        pts = pts.copy()
        violin_data = []
        for pg in PITCH_GROUPS:
            sub = pts[pts["pitch_group"] == pg]["stuff_plus"].dropna()
            if len(sub) > 10:
                violin_data.append(go.Violin(
                    x=[pg]*len(sub), y=sub,
                    name=pg, box_visible=True,
                    meanline_visible=True,
                    opacity=0.7, showlegend=False
                ))
        fig2 = go.Figure(violin_data)
        fig2.add_hline(y=100, line_dash="dash", line_color=ACCENT_COLOR,
                       annotation_text="League Avg (100)", annotation_font_color=ACCENT_COLOR)
        fig2.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(color=WHITE), height=350,
            margin=dict(t=10, b=60, l=10, r=10),
            yaxis=dict(title="Stuff+", gridcolor="#2A2D3A"),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig2, use_container_width=True)

    show_glossary()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: PITCHER EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Pitcher Explorer":
    st.markdown(f"<div style='color:{ACCENT_COLOR}; font-size:2.8rem; font-weight:800; letter-spacing:0.04em; margin-bottom:2px;'>Arsenal Intelligence</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:{WHITE};'>Pitcher Explorer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_MUTED};'>Search any pitcher to see their full arsenal breakdown, SHAP explainability, and season trends.</p>", unsafe_allow_html=True)
    st.caption(
        "**Thresholds:** "
        "A pitcher appears as qualifying for a season if they threw ≥ 500 pitches (SP) or ≥ 200 pitches (RP). "
        "Their full arsenal is shown for any pitch type with ≥ 2,000 league-wide swing events across 2023–2026 (required to train a reliable model). "
        "The SHAP breakdown requires the pitcher to have thrown ≥ 100 of that specific pitch type in the selected season."
    )
    st.divider()

    # Pitcher search — persists selection across season/role changes
    # Thresholds: 500 pitches for SP, 200 for RP (same as Leaderboard)
    SP_MIN, RP_MIN = 500, 200

    season_arsenal = arsenal[arsenal["season"] == season_filter].copy()
    if ROLE_CODE != "ALL":
        season_arsenal = season_arsenal[season_arsenal["role"] == ROLE_CODE]

    # Apply role-appropriate threshold per pitcher
    season_arsenal["min_pitches"] = season_arsenal["role"].map(
        {"SP": SP_MIN, "RP": RP_MIN}
    ).fillna(RP_MIN)
    season_qual = season_arsenal[
        season_arsenal["total_pitches"] >= season_arsenal["min_pitches"]
    ].sort_values("arsenal_stuff", ascending=False)

    qual_pitchers = season_qual["player_name"].unique().tolist()

    # Also include pitchers who qualified in ANY season (but not the selected one),
    # so they remain reachable via the dropdown rather than disappearing entirely.
    # These appear at the bottom of the list, separated by a label.
    all_seasons_qual = set()
    for role, min_p in [("SP", SP_MIN), ("RP", RP_MIN)]:
        ever_qual = (
            arsenal[
                (arsenal["role"] == role) &
                (arsenal["total_pitches"] >= min_p) &
                (arsenal["season"] != season_filter)
            ]["player_name"].unique()
        )
        all_seasons_qual.update(ever_qual)
    if ROLE_CODE != "ALL":
        # Only keep pitchers of the right role
        role_names = set(arsenal[arsenal["role"] == ROLE_CODE]["player_name"].unique())
        all_seasons_qual &= role_names

    prior_only = sorted(all_seasons_qual - set(qual_pitchers))
    full_pitcher_list = qual_pitchers + prior_only

    # Build display names (First Last) while keeping internal "Last, First" for data lookups
    display_list = [fmt_name(n) for n in full_pitcher_list]

    # Restore previously selected pitcher; fall back to first in list
    prev = st.session_state.get("pitcher_explorer_name", None)
    default_idx = full_pitcher_list.index(prev) if prev in full_pitcher_list else 0

    selected_display = st.selectbox("Select Pitcher", display_list, index=default_idx,
                                    key="pitcher_explorer_sel")
    # Map display name back to internal "Last, First" format used in data
    pitcher_name = full_pitcher_list[display_list.index(selected_display)]
    st.session_state["pitcher_explorer_name"] = pitcher_name

    p_data = arsenal[arsenal["player_name"] == pitcher_name].sort_values("season")

    # Auto-switch sidebar to pitcher's best qualifying season — but ONLY when the
    # pitcher itself changes (not when the user manually navigates between seasons).
    pitcher_just_changed = st.session_state.get("_prev_pitcher_name") != pitcher_name
    st.session_state["_prev_pitcher_name"] = pitcher_name

    if pitcher_name in prior_only:
        # Determine the role this pitcher is known for under the active filter.
        # If a role filter is active, use that role; otherwise use their most
        # recent season's role.
        if ROLE_CODE != "ALL":
            pitcher_role_for_warn = ROLE_CODE
        else:
            pitcher_role_for_warn = p_data.sort_values("season", ascending=False).iloc[0].get("role", "RP") \
                if not p_data.empty else "RP"
        threshold_used = SP_MIN if pitcher_role_for_warn == "SP" else RP_MIN

        # Only count seasons where the pitcher meets the threshold AND their role
        # matches the active filter (so a role-switch season isn't shown as "available").
        qualifying_rows = []
        for _, r in p_data.iterrows():
            row_role = r.get("role", "RP")
            min_p = SP_MIN if row_role == "SP" else RP_MIN
            if r["total_pitches"] >= min_p:
                if ROLE_CODE == "ALL" or row_role == ROLE_CODE:
                    qualifying_rows.append(int(r["season"]))

        # On first selection, auto-switch to the most recent qualifying season
        if qualifying_rows and pitcher_just_changed:
            best_season = max(qualifying_rows)
            if best_season in SEASONS and best_season != season_filter:
                st.session_state["_season_override_idx"] = SEASONS.index(best_season)
                st.rerun()

        # User manually chose a non-qualifying season — show message and stop
        avail_str = ", ".join(str(s) for s in sorted(qualifying_rows, reverse=True)) \
            if qualifying_rows else "none"
        st.warning(
            f"**{fmt_name(pitcher_name)}** does not have qualifying data for {season_filter} "
            f"(minimum {threshold_used} pitches for {pitcher_role_for_warn}). "
            f"Use the **Season** selector on the left to view their profile. "
            f"Available: **{avail_str}**."
        )
        st.stop()

    effective_season = season_filter

    p_latest = p_data[p_data["season"] == effective_season]
    if p_latest.empty:
        st.warning(f"{fmt_name(pitcher_name)} has no data for {effective_season}.")
        st.stop()

    row = p_latest.iloc[0]
    pitcher_role = row.get("role", "RP")
    role_color = ACCENT_COLOR if pitcher_role == "SP" else "#4A90D9"

    # ── Hero stats
    c1, c2, c3, c4, c5 = st.columns(5)
    stuff_val = row["arsenal_stuff"]
    color = stuffplus_color(stuff_val)
    with c1:
        st.markdown(f"""
        <div style='background:{CARD_BG}; border-radius:10px; padding:16px; border:2px solid {color}; text-align:center;'>
            <div style='font-size:0.8rem; color:{TEXT_MUTED};'>Arsenal Stuff+</div>
            <div style='font-size:2.6rem; font-weight:800; color:{color};'>{stuff_val:.1f}</div>
            <span style='background:{role_color}; color:white; font-size:0.7rem; font-weight:700;
                         padding:2px 10px; border-radius:20px;'>{pitcher_role}</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.metric("Pitches Thrown", f"{int(row['total_pitches']):,}")
    with c3:
        st.metric("Whiff Rate", f"{row['whiff_rate']:.1%}")
    with c4:
        st.metric("CSW Rate", f"{row['csw_rate']:.1%}")
    with c5:
        st.metric("xwOBA Against", f"{row['avg_xwoba']:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1])

    # ── Arsenal pitch-type breakdown
    with col_left:
        st.markdown(f"<div class='section-header'>Pitch Arsenal Breakdown — {effective_season}</div>", unsafe_allow_html=True)
        p_pitches = pt_scores[
            (pt_scores["player_name"] == pitcher_name) &
            (pt_scores["season"] == effective_season) &
            (pt_scores["role"] == pitcher_role)
        ].sort_values("n_pitches", ascending=False)

        if not p_pitches.empty:
            fig3 = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, (_, prow) in enumerate(p_pitches.iterrows()):
                pct = prow["n_pitches"] / p_pitches["n_pitches"].sum()
                fig3.add_trace(go.Bar(
                    x=[prow["pitch_group"]],
                    y=[prow["stuff_plus"]],
                    name=prow["pitch_group"],
                    marker_color=colors[i % len(colors)],
                    text=f"{pct:.0%}<br>n={prow['n_pitches']:.0f}",
                    textposition="outside",
                    width=0.5,
                ))
            fig3.add_hline(y=100, line_dash="dash", line_color=ACCENT_COLOR,
                           annotation_text="Avg", annotation_font_color=ACCENT_COLOR)
            fig3.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font=dict(color=WHITE), height=320,
                showlegend=False, barmode="group",
                yaxis=dict(title="Stuff+", range=[80, max(135, p_pitches["stuff_plus"].max() + 5)],
                           gridcolor="#2A2D3A"),
                margin=dict(t=10, b=40, l=10, r=10),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Pitch stats table
            disp = p_pitches[["pitch_group","n_pitches","stuff_plus","avg_velo",
                               "avg_spin","avg_ivb","whiff_rate","csw_rate","avg_xwoba"]].copy()
            disp.columns = ["Pitch","Pitches","Stuff+","Velo","Spin","iVB","Whiff%","CSW%","xwOBA"]
            disp["Stuff+"]  = disp["Stuff+"].map("{:.1f}".format)
            disp["Velo"]    = disp["Velo"].map("{:.1f}".format)
            disp["Spin"]    = disp["Spin"].map("{:.0f}".format)
            disp["iVB"]     = disp["iVB"].map("{:.1f}\"".format)
            disp["Whiff%"]  = disp["Whiff%"].map("{:.1%}".format)
            disp["CSW%"]    = disp["CSW%"].map("{:.1%}".format)
            disp["xwOBA"]   = disp["xwOBA"].map("{:.3f}".format)
            disp["Pitches"] = disp["Pitches"].map("{:,.0f}".format)
            st.dataframe(disp, hide_index=True, use_container_width=True)

    # ── SHAP explanation — pitcher's primary pitch type
    with col_right:
        st.markdown("<div class='section-header'>What Drives Their Stuff+ Score? (SHAP Breakdown)</div>", unsafe_allow_html=True)

        # Filter SHAP by pitcher, role, AND the effective season
        p_shap_all = shap_pitch[
            (shap_pitch["player_name"] == pitcher_name) &
            (shap_pitch["role"] == pitcher_role) &
            (shap_pitch["season"] == effective_season)
        ] if "season" in shap_pitch.columns else shap_pitch[
            (shap_pitch["player_name"] == pitcher_name) &
            (shap_pitch["role"] == pitcher_role)
        ]

        if not p_shap_all.empty:
            # Order available pitch types by n_pitches descending (primary pitch first)
            available_pitches = (
                p_shap_all.sort_values("n_pitches", ascending=False)["pitch_group"].tolist()
            )
            selected_shap_pitch = st.selectbox(
                "Pitch type",
                available_pitches,
                key="shap_pitch_sel",
                help="Select which pitch type to show the SHAP feature breakdown for."
            )
            shap_row = p_shap_all[p_shap_all["pitch_group"] == selected_shap_pitch].iloc[0]
            shap_feats = [c for c in shap_pitch.columns if c in FEATURE_LABELS]

            if shap_feats:
                vals = [shap_row[f] for f in shap_feats]
                bar_colors = [ACCENT_COLOR if v > 0 else "#4A90D9" for v in vals]
                fig4 = go.Figure(go.Bar(
                    x=vals, y=shap_feats, orientation="h",
                    marker_color=bar_colors, marker_line_width=0,
                ))
                fig4.add_vline(x=0, line_color="white", line_width=1)
                fig4.update_layout(
                    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                    font=dict(color=WHITE), height=300,
                    margin=dict(t=10, b=30, l=10, r=10),
                    xaxis=dict(title="SHAP value (→ more whiffs)", gridcolor="#2A2D3A",
                               zeroline=False),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig4, use_container_width=True)
                n_shap_pitches = int(shap_row["n_pitches"]) if "n_pitches" in shap_row.index else None
                caption_text = (
                    f"{selected_shap_pitch} SHAP breakdown for {fmt_name(pitcher_name)} ({effective_season}). "
                    "Each bar shows how that specific pitcher's pitch characteristic pushes their "
                    "whiff probability up (green) or down (blue) relative to the league average "
                    "for that pitch type. Values are pitcher-specific, not league-wide weights."
                )
                if n_shap_pitches:
                    caption_text += f" Based on {n_shap_pitches:,} pitches in {effective_season}."
                st.caption(caption_text)

                # Show note when pitch types in the arsenal are missing from SHAP
                arsenal_pitches   = set(p_pitches["pitch_group"].tolist()) if not p_pitches.empty else set()
                shap_pitch_groups = set(p_shap_all["pitch_group"].tolist())
                missing_from_shap = arsenal_pitches - shap_pitch_groups
                if missing_from_shap:
                    missing_str = ", ".join(sorted(missing_from_shap))
                    st.info(
                        f"**Note:** {missing_str} "
                        f"{'is' if len(missing_from_shap) == 1 else 'are'} not available in this "
                        f"dropdown because fewer than 100 pitches of that type were thrown in "
                        f"{effective_season}. "
                        f"{'Its' if len(missing_from_shap) == 1 else 'Their'} Stuff+ score still "
                        f"appears in the arsenal breakdown on the left."
                    )
        else:
            st.info(
                "⚠ **SHAP breakdown not available.** "
                f"No pitch type thrown by {fmt_name(pitcher_name)} in {effective_season} meets the "
                "minimum of 100 pitches required for a reliable SHAP estimate."
            )

    # ── Season trend
    if len(p_data) > 1:
        st.markdown("<div class='section-header'>Season Trend</div>", unsafe_allow_html=True)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=p_data["season"], y=p_data["arsenal_stuff"],
            mode="lines+markers+text",
            text=p_data["arsenal_stuff"].map("{:.1f}".format),
            textposition="top center",
            line=dict(color=ACCENT_COLOR, width=3),
            marker=dict(size=10, color=ACCENT_COLOR),
            name="Arsenal Stuff+",
        ))
        fig5.add_hline(y=100, line_dash="dash", line_color=TEXT_MUTED, line_width=1,
                       annotation_text="League Avg")
        fig5.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(color=WHITE), height=240,
            margin=dict(t=10, b=30, l=10, r=10),
            xaxis=dict(tickvals=list(p_data["season"]), title="Season", gridcolor="#2A2D3A"),
            yaxis=dict(title="Arsenal Stuff+", gridcolor="#2A2D3A"),
        )
        st.plotly_chart(fig5, use_container_width=True)

    show_glossary()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Leaderboard":
    st.markdown(f"<div style='color:{ACCENT_COLOR}; font-size:2.8rem; font-weight:800; letter-spacing:0.04em; margin-bottom:2px;'>Arsenal Intelligence</div>", unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        min_pitches = st.slider("Min pitches", 100, 1000, 300, step=50)
    with col_f2:
        rank_labels = {
            "arsenal_stuff": "Arsenal Stuff+",
            "stuff_plus":    "Stuff+ (by pitch type)",
            "whiff_rate":    "Whiff Rate",
            "csw_rate":      "CSW Rate",
            "avg_xwoba":     "xwOBA Against",
        }
        rank_by = st.selectbox("Rank by", list(rank_labels.keys()),
                               format_func=lambda x: rank_labels[x])

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        search_q = st.text_input("Search pitcher", placeholder="Type a name…")
    with col_s2:
        team_q = st.selectbox("Team", ["All"] + MLB_TEAMS)

    rank_title = RANK_PAGE_TITLES.get(rank_by, "Full Rankings")
    st.markdown(f"<h1 style='color:{WHITE};'>{rank_title}</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED};'>Season {season_filter}"
        + (f" · updated {last_updated}" if last_updated else "")
        + "</p>",
        unsafe_allow_html=True,
    )
    if season_filter == CURRENT_SEASON:
        st.info(
            "**2026 live season:** Rankings include all pitchers with Statcast data. "
            "Rows highlighted in **yellow** are below the low-sample pitch threshold — "
            "interpret Stuff+ and outcome rates with caution until more pitches accumulate."
        )
    st.divider()

    # Pitch type picker only appears when "Stuff+ (by pitch type)" is selected
    if rank_by == "stuff_plus":
        pitch_type_sel = st.selectbox(
            "Select pitch type",
            ["— choose a pitch type —"] + PITCH_GROUPS,
            help="Rankings show Stuff+ for this pitch type only. Scores are z-scored within each pitch type, so cross-type comparisons are not meaningful."
        )
        pitch_filter = [pitch_type_sel] if pitch_type_sel != "— choose a pitch type —" else []
        if not pitch_filter:
            st.caption("← Select a pitch type above to see pitch-specific Stuff+ rankings.")
    else:
        pitch_filter = []

    # ── Pitch-type view: shows single-pitch Stuff+
    if pitch_filter:
        lb_df = pt_scores[
            (pt_scores["season"] == season_filter) &
            (pt_scores["pitch_group"].isin(pitch_filter)) &
            (pt_scores["n_pitches"] >= 100)
        ]
        if ROLE_CODE != "ALL":
            lb_df = lb_df[lb_df["role"] == ROLE_CODE]
        lb_df = lb_df.copy().sort_values("stuff_plus", ascending=False).head(50)
        score_col  = "stuff_plus"
        score_label = "Pitch Stuff+"
        show_cols = ["player_name", "role", "team", "pitch_group", "n_pitches", "stuff_plus",
                     "avg_velo", "avg_spin", "avg_ivb", "whiff_rate", "csw_rate", "avg_xwoba"]

    # ── Arsenal view: shows full-arsenal weighted average
    else:
        # "stuff_plus (by pitch type)" selected but no pitch types chosen yet — default to arsenal_stuff
        sort_col = "arsenal_stuff" if rank_by == "stuff_plus" else rank_by
        lb_df = arsenal[
            (arsenal["season"] == season_filter) &
            (arsenal["total_pitches"] >= min_pitches)
        ]
        if ROLE_CODE != "ALL":
            lb_df = lb_df[lb_df["role"] == ROLE_CODE]
        asc = sort_col == "avg_xwoba"
        lb_df = lb_df.copy().sort_values(sort_col, ascending=asc).reset_index(drop=True)
        lb_df.index = lb_df.index + 1
        score_col   = sort_col
        score_label = rank_labels.get(sort_col, sort_col)
        show_cols = ["player_name", "role", "team", "total_pitches", "arsenal_stuff",
                     "avg_velo", "avg_spin", "whiff_rate", "csw_rate", "avg_xwoba"]
        if rank_by != "arsenal_stuff":
            rank_title = RANK_PAGE_TITLES.get(sort_col, rank_title)

    # Search / team filters (all seasons including 2026)
    if search_q.strip():
        lb_df = lb_df[lb_df["player_name"].str.contains(search_q.strip(), case=False, na=False)]
    if team_q != "All" and "team" in lb_df.columns:
        lb_df = lb_df[lb_df["team"].fillna("").apply(lambda t: pitcher_matches_team(str(t), team_q))]

    if pitch_filter:
        rank_title = f"Full {pitch_filter[0]} Stuff+ Rankings"
        st.markdown(f"<h1 style='color:{WHITE};'>{rank_title}</h1>", unsafe_allow_html=True)

    # ── Top 10 bar chart
    bar_fmt = {
        "arsenal_stuff": "{:.1f}", "stuff_plus": "{:.1f}",
        "whiff_rate": "{:.1%}", "csw_rate": "{:.1%}", "avg_xwoba": "{:.3f}",
    }
    top10 = lb_df.head(10)
    bar_vals = top10[score_col][::-1]
    bar_text = [bar_fmt.get(score_col, "{:.2f}").format(v) for v in bar_vals]
    bar_colors = [stuffplus_color(v) if score_col in ("arsenal_stuff", "stuff_plus")
                  else ACCENT_COLOR for v in bar_vals]
    fig6 = go.Figure(go.Bar(
        y=top10["player_name"][::-1],
        x=bar_vals,
        orientation="h",
        marker_color=bar_colors,
        text=bar_text,
        textposition="outside",
    ))
    if score_col in ("arsenal_stuff", "stuff_plus"):
        fig6.add_vline(x=100, line_dash="dash", line_color=TEXT_MUTED, line_width=1,
                       annotation_text="League Avg", annotation_font_color=TEXT_MUTED)
    fig6.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=WHITE), height=340,
        margin=dict(t=10, b=30, l=0, r=80),
        xaxis=dict(title=score_label, gridcolor="#2A2D3A",
                   tickformat=".1%" if score_col in ("whiff_rate", "csw_rate") else
                              ".3f" if score_col == "avg_xwoba" else ""),
        yaxis=dict(autorange=True),
    )
    st.plotly_chart(fig6, use_container_width=True)

    # ── Full sortable table
    table_title = rank_title
    st.markdown(f"<div class='section-header'>{table_title}</div>", unsafe_allow_html=True)
    disp_raw = lb_df[show_cols + (["low_sample"] if "low_sample" in lb_df.columns else [])].copy()
    disp = disp_raw.drop(columns=["low_sample"], errors="ignore").copy()
    fmt_map = {
        "arsenal_stuff": "{:.1f}", "stuff_plus": "{:.1f}",
        "avg_velo": "{:.1f}", "avg_spin": "{:.0f}", "avg_ivb": "{:.1f}",
        "whiff_rate": "{:.1%}", "csw_rate": "{:.1%}", "avg_xwoba": "{:.3f}",
        "n_pitches": "{:,.0f}", "total_pitches": "{:,.0f}",
    }
    for col, fmt in fmt_map.items():
        if col in disp.columns:
            disp[col] = disp[col].map(fmt.format)
    col_labels = {
        "player_name": "Pitcher", "pitch_group": "Pitch Type", "team": "Team",
        "total_pitches": "Pitches", "n_pitches": "Pitches",
        "arsenal_stuff": "Arsenal Stuff+", "stuff_plus": "Stuff+",
        "avg_velo": "Velo (mean)", "avg_spin": "Spin (mean)", "avg_ivb": "iVB (mean)",
        "whiff_rate": "Whiff% (mean)", "csw_rate": "CSW% (mean)", "avg_xwoba": "xwOBA (mean)",
    }
    disp = disp.rename(columns=col_labels)

    if season_filter == CURRENT_SEASON and "low_sample" in disp_raw.columns and disp_raw["low_sample"].any():
        low_flags = disp_raw["low_sample"].tolist()

        def _highlight(row):
            if row.name < len(low_flags) and low_flags[row.name]:
                return ["background-color: #3d3520"] * len(row)
            return [""] * len(row)

        st.dataframe(disp.style.apply(_highlight, axis=1), use_container_width=True, height=500)
    else:
        st.dataframe(disp, use_container_width=True, height=500)

    # ── Scatter: Stuff+ vs outcome
    st.markdown("<div class='section-header'>Stuff+ vs Outcomes (Pitcher Level)</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    scatter_df = arsenal[
        (arsenal["season"] == season_filter) &
        (arsenal["total_pitches"] >= min_pitches)
    ]
    if ROLE_CODE != "ALL":
        scatter_df = scatter_df[scatter_df["role"] == ROLE_CODE]
    scatter_df = scatter_df.dropna(subset=["whiff_rate", "avg_xwoba"])
    if search_q.strip():
        scatter_df = scatter_df[
            scatter_df["player_name"].str.contains(search_q.strip(), case=False, na=False)
        ]
    if team_q != "All" and "team" in scatter_df.columns:
        scatter_df = scatter_df[
            scatter_df["team"].fillna("").apply(lambda t: pitcher_matches_team(str(t), team_q))
        ]

    if scatter_df.empty:
        st.caption("Not enough pitchers match the current filters to show scatter plots.")
    else:
        with c1:
            fig7 = px_scatter_trendline(
                scatter_df, x="arsenal_stuff", y="whiff_rate",
                hover_data={"player_name": True, "total_pitches": True,
                            "arsenal_stuff": ":.1f", "whiff_rate": ":.1%"},
                color="whiff_rate", color_continuous_scale="RdYlGn",
                opacity=0.7,
                labels={"arsenal_stuff": "Arsenal Stuff+", "whiff_rate": "Whiff Rate"},
                title="Stuff+ vs Whiff Rate",
            )
            corr_w = scatter_df["arsenal_stuff"].corr(scatter_df["whiff_rate"])
            fig7.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font=dict(color=WHITE),
                height=380, margin=dict(t=40, b=30), showlegend=False,
                coloraxis_showscale=False,
                title=dict(text=f"Stuff+ vs Whiff Rate  (r = {corr_w:.2f})"),
                yaxis=dict(tickformat=".0%", gridcolor="#2A2D3A"),
                xaxis=dict(gridcolor="#2A2D3A"),
            )
            st.plotly_chart(fig7, use_container_width=True)
            st.caption(
                f"Each dot is one pitcher-season. Arsenal Stuff+ (x-axis) is the usage-weighted average "
                f"pitch quality score across their entire arsenal. Whiff Rate (y-axis) is the percentage "
                f"of swings that resulted in a miss. The blue trendline shows the positive relationship — "
                f"pitchers with higher Arsenal Stuff+ tend to generate more swing-and-misses. "
                f"r = {corr_w:.2f} indicates a {'moderate' if abs(corr_w) < 0.5 else 'strong'} correlation. "
                f"NOTE: Unlike the decile chart above which measures individual pitch Stuff+, this chart "
                f"aggregates to the pitcher-season level — introducing noise from pitch mix, command, "
                f"sequencing, and opponent quality."
            )

        with c2:
            fig8 = px_scatter_trendline(
                scatter_df, x="arsenal_stuff", y="avg_xwoba",
                hover_data={"player_name": True, "total_pitches": True,
                            "arsenal_stuff": ":.1f", "avg_xwoba": ":.3f"},
                color="avg_xwoba", color_continuous_scale="RdYlGn_r",
                opacity=0.7,
                labels={"arsenal_stuff": "Arsenal Stuff+", "avg_xwoba": "xwOBA Against"},
                title="Stuff+ vs xwOBA Against",
            )
            corr_x = scatter_df["arsenal_stuff"].corr(scatter_df["avg_xwoba"])
            fig8.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font=dict(color=WHITE),
                height=380, margin=dict(t=40, b=30), showlegend=False,
                coloraxis_showscale=False,
                title=dict(text=f"Stuff+ vs xwOBA Against  (r = {corr_x:.2f})"),
                yaxis=dict(gridcolor="#2A2D3A"),
                xaxis=dict(gridcolor="#2A2D3A"),
            )
            st.plotly_chart(fig8, use_container_width=True)
            st.caption(
                f"Each dot is one pitcher-season. Arsenal Stuff+ (x-axis) is plotted against xwOBA Against "
                f"(y-axis) — the expected quality of contact allowed, where lower is better for the pitcher. "
                f"The downward-sloping trendline confirms that higher Arsenal Stuff+ is associated with weaker contact "
                f"quality allowed. r = {corr_x:.2f} is a weaker relationship than the whiff rate chart, "
                f"which is expected: contact quality is influenced by many factors beyond raw pitch stuff, "
                f"including pitch location, sequencing, ballpark, and opponent quality."
            )

    show_glossary()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "How the Model Works + Important Findings":
    st.markdown(f"<div style='color:{ACCENT_COLOR}; font-size:2.8rem; font-weight:800; letter-spacing:0.04em; margin-bottom:2px;'>Arsenal Intelligence</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:{WHITE};'>How the Model Works</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_MUTED};'>Model design, feature weights, and SHAP explainability.</p>", unsafe_allow_html=True)
    st.caption("A pitch type is included in the model only if it has at least 2,000 swing events in the 2023–2026 dataset. This ensures each model has enough data for reliable training. Pitch types below this threshold (e.g. rare variants) are excluded.")
    st.divider()

    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.markdown(f"""
        <div style='background:{CARD_BG}; border-radius:12px; padding:20px; border:1px solid #2A2D3A;'>
            <div style='color:{ACCENT_COLOR}; font-weight:700; font-size:1rem; margin-bottom:12px;'>
                Model Design
            </div>
            <p style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.6;'>
                A separate <b style='color:{WHITE};'>L2-penalized logistic regression</b> is fit for each pitch type,
                predicting whether a swung-at pitch results in a swing-and-miss.
            </p>
            <p style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.6;'>
                Inputs are only <b style='color:{WHITE};'>physical characteristics</b> — no batter, count, or
                game-state context — so the score captures pure pitch quality, not situational factors.
            </p>
            <p style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.6;'>
                The predicted probability is then <b style='color:{WHITE};'>z-score normalized</b> within each
                pitch type to produce Stuff+ (100 = league average, 10 points = 1 standard deviation).
            </p>
            <hr style='border-color:#2A2D3A; margin:16px 0;'>
            <div style='color:{ACCENT_COLOR}; font-weight:700; margin-bottom:8px;'>Features Used</div>
            <ul style='color:{TEXT_MUTED}; font-size:0.88rem; line-height:1.8;'>
                <li><b style='color:{WHITE};'>Perceived Velocity</b> — release speed + extension bonus</li>
                <li><b style='color:{WHITE};'>Induced Vertical Break</b> — inches of spin-driven rise/drop</li>
                <li><b style='color:{WHITE};'>Arm-Side Horiz Break</b> — handedness-adjusted horizontal movement</li>
                <li><b style='color:{WHITE};'>Spin Rate</b> — raw RPM at release</li>
                <li><b style='color:{WHITE};'>Release Extension</b> — how close to the plate the pitch is released</li>
                <li><b style='color:{WHITE};'>Plate Location (x, z)</b> — where the ball crosses the zone</li>
            </ul>
            <hr style='border-color:#2A2D3A; margin:16px 0;'>
            <div style='color:{ACCENT_COLOR}; font-weight:700; margin-bottom:8px;'>Model Performance (5-fold CV AUC) ({POOLED_SEASONS_LABEL})</div>
        </div>
        """, unsafe_allow_html=True)

        # AUC table — filter by role
        coefs_role = coefs if ROLE_CODE == "ALL" else coefs[coefs["role"] == ROLE_CODE]
        auc_df = (
            coefs_role[["role","pitch_group","auc","baseline_whiff"]]
            .drop_duplicates(subset=["role","pitch_group"])
            .sort_values("auc", ascending=False)
            .rename(columns={
                "role": "Role", "pitch_group": "Pitch Type",
                "auc": "AUC", "baseline_whiff": "Baseline Whiff%"
            })
        )
        auc_df["AUC"] = auc_df["AUC"].map("{:.3f}".format)
        auc_df["Baseline Whiff%"] = auc_df["Baseline Whiff%"].map("{:.1%}".format)
        st.dataframe(auc_df, hide_index=True, use_container_width=True)

    with col_right:
        st.markdown(f"<div class='section-header'>Feature Weight Heatmap by Pitch Type ({POOLED_SEASONS_LABEL})</div>", unsafe_allow_html=True)
        hm_role_label = {"ALL": "All Pitchers", "SP": "Starters (SP)", "RP": "Relievers (RP)"}[ROLE_CODE]
        st.caption(f"Logistic regression coefficients ({hm_role_label}) — features are z-score scaled so values are directly comparable. Green = drives more whiffs, Red = drives fewer whiffs.")

        if ROLE_CODE == "ALL":
            # Average SP and RP coefficients across both roles
            coefs_hm = (
                coefs.groupby(["pitch_group", "feature_label"])["coefficient"]
                .mean().reset_index()
            )
        else:
            coefs_hm = coefs_role
        pivot = coefs_hm.pivot(index="pitch_group", columns="feature_label", values="coefficient")
        col_order = pivot.abs().mean().sort_values(ascending=False).index
        pivot = pivot[col_order]

        fig9 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(title="Coefficient", tickfont=dict(color=WHITE),
                          title_font=dict(color=WHITE)),
        ))
        fig9.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(color=WHITE), height=340,
            margin=dict(t=10, b=80, l=10, r=10),
            xaxis=dict(tickangle=-25, side="bottom"),
        )
        st.plotly_chart(fig9, use_container_width=True)

        # ── SHAP beeswarm — selectable pitch type
        st.markdown(f"<div class='section-header'>SHAP Analysis ({POOLED_SEASONS_LABEL})</div>", unsafe_allow_html=True)

        label_map = {
            "perceived_velo": "Perceived Velocity",
            "pfx_z_in": "Induced Vertical Break",
            "break_arm": "Arm-Side Horiz Break",
            "release_spin_rate": "Spin Rate",
            "extension": "Release Extension",
            "plate_x": "Horizontal Location",
            "plate_z": "Vertical Location",
        }

        shap_role_label = {"ALL": "All Pitchers", "SP": "Starters", "RP": "Relievers"}[ROLE_CODE]
        available_shap_pitches = sorted(shap_global["pitch_group"].unique()) if "pitch_group" in shap_global.columns else ["Four-Seam FB"]
        default_idx = available_shap_pitches.index("Four-Seam FB") if "Four-Seam FB" in available_shap_pitches else 0
        beeswarm_pitch = st.selectbox(
            "Pitch type",
            available_shap_pitches,
            index=default_idx,
            key="beeswarm_pitch_sel",
        )

        if "pitch_group" in shap_global.columns:
            role_filter_shap = shap_global["role"] == ROLE_CODE if ROLE_CODE != "ALL" else pd.Series([True] * len(shap_global), index=shap_global.index)
            shap_role_df = shap_global[role_filter_shap & (shap_global["pitch_group"] == beeswarm_pitch)]
        else:
            role_filter_shap = shap_global["role"] == ROLE_CODE if ROLE_CODE != "ALL" else pd.Series([True] * len(shap_global), index=shap_global.index)
            shap_role_df = shap_global[role_filter_shap]

        st.caption(
            f"Gradient boosted model SHAP values — {beeswarm_pitch} "
            f"({shap_role_label}). "
            "Each dot is a pitch. Color = feature value (red = high, blue = low). "
            "Spread on x-axis = how much that feature shifts whiff probability."
        )

        sample = shap_role_df.sample(n=min(3000, len(shap_role_df)), random_state=1) if len(shap_role_df) > 0 else shap_role_df

        shap_cols_val  = [c for c in sample.columns if c.startswith("val_")]
        shap_cols_shap = [c for c in sample.columns if c.startswith("shap_")]

        if sample.empty or not shap_cols_shap:
            st.info(f"No SHAP data available for {beeswarm_pitch} ({shap_role_label}).")
        else:
            # Mean |SHAP| ordering
            mean_abs = sample[shap_cols_shap].abs().mean()
            order    = mean_abs.sort_values(ascending=True).index.tolist()
            ordered_feats = [f.replace("shap_", "") for f in order]

            # Beeswarm: numeric y positions so jitter works, custom tick labels for feature names
            tick_vals   = list(range(len(ordered_feats)))
            tick_labels = [label_map.get(f, f) for f in ordered_feats]

            fig10 = go.Figure()
            for y_pos, (feat, shap_col) in enumerate(zip(ordered_feats, order)):
                val_col    = f"val_{feat}"
                label      = label_map.get(feat, feat)
                vals       = sample[val_col].values
                sv         = sample[shap_col].values
                vmin, vmax = vals.min(), vals.max()
                norm       = (vals - vmin) / (vmax - vmin + 1e-9)
                colors     = [f"rgba({int(255*n)}, {int(60*(1-n))}, {int(255*(1-n))}, 0.5)"
                              for n in norm]
                jitter     = np.random.uniform(-0.25, 0.25, len(sv))
                fig10.add_trace(go.Scatter(
                    x=sv,
                    y=y_pos + jitter,
                    mode="markers",
                    marker=dict(color=colors, size=3, opacity=0.6),
                    name=label,
                    showlegend=False,
                    hovertemplate=f"{label}<br>SHAP=%{{x:.3f}}<extra></extra>",
                ))
            fig10.add_vline(x=0, line_color="white", line_width=1)
            fig10.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font=dict(color=WHITE), height=340,
                margin=dict(t=10, b=30, l=10, r=10),
                xaxis=dict(title="SHAP value (impact on whiff probability)", gridcolor="#2A2D3A"),
                yaxis=dict(
                    tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
                    title="", gridcolor="#2A2D3A",
                ),
            )
            st.plotly_chart(fig10, use_container_width=True)

    # ── Key findings callouts
    st.markdown("<div class='section-header'>Key Findings</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    findings = [
        ("Location > Velocity", "Vertical plate location is the top whiff driver for four-seamers — a well-placed 93 outperforms a flat 97."),
        ("Pitch-type specificity", "Feature weights differ drastically by pitch type. A curveball's value comes from drop; a sweeper's from horizontal break."),
        ("Spin × Velocity interaction", "SHAP shows high-spin fastballs at 96+ mph are disproportionately effective — the interaction matters, not just the raw numbers."),
        ("Stuff ≠ Results", "r = 0.40 with whiff rate shows strong signal, but r = −0.21 with xwOBA means sequencing, command, and mix still matter enormously."),
    ]
    for col, (title, body) in zip([c1, c2, c3, c4], findings):
        with col:
            st.markdown(f"""
            <div style='background:{CARD_BG}; border-radius:10px; padding:16px;
                        border-left:3px solid {ACCENT_COLOR}; height:160px;'>
                <div style='color:{ACCENT_COLOR}; font-weight:700; font-size:0.9rem;
                            margin-bottom:8px;'>{title}</div>
                <div style='color:{TEXT_MUTED}; font-size:0.82rem; line-height:1.5;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    show_glossary()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About":
    st.markdown(f"<div style='color:{ACCENT_COLOR}; font-size:2.8rem; font-weight:800; letter-spacing:0.04em; margin-bottom:2px;'>Arsenal Intelligence</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:{WHITE};'>About This Dashboard</h1>", unsafe_allow_html=True)
    st.divider()

    # ── What it is
    col_a, col_b = st.columns([1.6, 1])
    with col_a:
        st.markdown(f"""
        <div style='background:{CARD_BG}; border-radius:12px; padding:24px 28px;
                    border:1px solid #2A2D3A; margin-bottom:20px;'>
            <div style='color:{ACCENT_COLOR}; font-weight:800; font-size:1.15rem;
                        margin-bottom:12px;'>What Is This?</div>
            <p style='color:{TEXT_MUTED}; line-height:1.75; font-size:0.92rem;'>
                This dashboard builds and explains a <b style='color:{WHITE};'>pitch "stuff" quality score</b>
                for every MLB pitcher using Statcast tracking data (2023–2026).
                The score answers one question: <em style='color:{WHITE};'>based purely on a pitch's
                physical characteristics, how likely is it to generate a swing-and-miss?</em>
            </p>
            <p style='color:{TEXT_MUTED}; line-height:1.75; font-size:0.92rem;'>
                Unlike ERA or WHIP, which are heavily influenced by defense, luck, and sequencing,
                Stuff+ isolates the <b style='color:{WHITE};'>raw physical quality</b> of each pitch —
                velocity, movement, spin, and location — and converts it into a single comparable number.
            </p>
            <div style='background:#1e3a5f; border-radius:8px; padding:14px 18px;
                        border:1px solid #2A2D3A; margin:12px 0;'>
                <div style='color:{ACCENT_COLOR}; font-weight:700; margin-bottom:6px;'>Live now — {CURRENT_SEASON} season</div>
                <div style='color:{TEXT_MUTED}; font-size:0.88rem; line-height:1.65;'>
                    This dashboard is <b style='color:{WHITE};'>live for the {CURRENT_SEASON} MLB season right now</b>.
                    In-season Statcast data refreshes automatically through the regular season (ends early October).
                    GitHub Actions pulls new pitches about <b style='color:{WHITE};'>4× per day</b>; this app reloads data
                    every <b style='color:{WHITE};'>2 minutes</b>. Completed seasons (2023–2025) are frozen
                    snapshots — selecting those years shows the same scores as before. For {CURRENT_SEASON}, sample sizes
                    are still building: pitchers highlighted in <b style='color:#fbbf24;'>yellow</b> on the
                    Leaderboard are below the current low-sample pitch threshold (~35% of role median).
                    SP/RP labels use the same avg-pitches-per-game rule as prior seasons, supplemented by
                    MLB games-started data when appearances are still sparse.
                </div>
            </div>
            <p style='color:{TEXT_MUTED}; line-height:1.75; font-size:0.92rem;'>
                Crucially, the model is <b style='color:{WHITE};'>interpretable by design</b>: you can see
                exactly which components drive any pitcher's score, making it useful for scouting,
                player development, and identifying pitchers whose underlying stuff is
                better (or worse) than their surface results suggest.
            </p>
            <div style='background:#0E1117; border-radius:8px; padding:12px 16px;
                        border-left:3px solid {ACCENT_COLOR}; margin-top:4px;'>
                <span style='color:{ACCENT_COLOR}; font-weight:700; font-size:0.85rem;'>📖 Glossary available on every page</span>
                <span style='color:{TEXT_MUTED}; font-size:0.85rem;'> — unfamiliar with a metric? Click <b style='color:{WHITE};'>Show / Hide Glossary</b> at the bottom of any page for plain-English definitions of every score, feature, pitch type, and model term used throughout this dashboard.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # How the score is calculated — use st.html() to bypass markdown parser
        st.html(f"""
        <div style='background:{CARD_BG}; border-radius:12px; padding:24px 28px;
                    border:1px solid #2A2D3A; margin-bottom:20px; font-family:sans-serif;'>
            <div style='color:{ACCENT_COLOR}; font-weight:800; font-size:1.1rem;
                        margin-bottom:16px;'>How the Score Is Calculated</div>

            <div style='display:flex; gap:14px; flex-wrap:wrap; margin-bottom:20px;'>

                <div style='flex:1; min-width:160px; background:{DARK_BG}; border-radius:10px;
                            padding:14px 16px; border-left:3px solid {ACCENT_COLOR};'>
                    <div style='color:{WHITE}; font-weight:700; font-size:0.85rem;
                                margin-bottom:6px;'>Step 1 — Filter to swings</div>
                    <div style='color:{TEXT_MUTED}; font-size:0.82rem; line-height:1.6;'>
                        Only pitches the batter swung at are used. This isolates pure pitch
                        quality — the question is whether the pitch fooled the batter, not
                        whether the batter chose to swing.
                    </div>
                </div>

                <div style='flex:1; min-width:160px; background:{DARK_BG}; border-radius:10px;
                            padding:14px 16px; border-left:3px solid {ACCENT_COLOR};'>
                    <div style='color:{WHITE}; font-weight:700; font-size:0.85rem;
                                margin-bottom:6px;'>Step 2 — Extract physical features</div>
                    <div style='color:{TEXT_MUTED}; font-size:0.82rem; line-height:1.6;'>
                        Seven Statcast measurements per pitch: perceived velocity, induced
                        vertical break, arm-side horizontal break, spin rate, release
                        extension, and plate location (x, z).
                    </div>
                </div>

                <div style='flex:1; min-width:160px; background:{DARK_BG}; border-radius:10px;
                            padding:14px 16px; border-left:3px solid {ACCENT_COLOR};'>
                    <div style='color:{WHITE}; font-weight:700; font-size:0.85rem;
                                margin-bottom:6px;'>Step 3 — Fit logistic regression</div>
                    <div style='color:{TEXT_MUTED}; font-size:0.82rem; line-height:1.6;'>
                        A separate model per pitch type and role (SP/RP) learns a
                        weighted recipe predicting whiff probability. The coefficients
                        are the interpretable "recipe" — readable directly.
                    </div>
                </div>

                <div style='flex:1; min-width:160px; background:{DARK_BG}; border-radius:10px;
                            padding:14px 16px; border-left:3px solid {ACCENT_COLOR};'>
                    <div style='color:{WHITE}; font-weight:700; font-size:0.85rem;
                                margin-bottom:6px;'>Step 4 — Normalize to Stuff+</div>
                    <div style='color:{TEXT_MUTED}; font-size:0.82rem; line-height:1.6;'>
                        Raw probabilities are z-scored within each role &times; pitch type group:<br>
                        <code style='background:#2A2D3A; padding:2px 6px; border-radius:3px;
                        color:{WHITE}; font-size:0.8rem;'>Stuff+ = 100 + 10 &times; (prob &minus; mean) / std</code><br><br>
                        100 = league avg for that type, every 10 pts = 1 standard deviation.
                    </div>
                </div>

            </div>

            <div style='padding-top:16px; border-top:1px solid #2A2D3A;'>
                <div style='color:{WHITE}; font-weight:700; font-size:0.88rem;
                            margin-bottom:12px;'>Stuff+ vs. Arsenal Stuff+</div>
                <div style='display:flex; gap:14px; flex-wrap:wrap;'>

                    <div style='flex:1; min-width:200px; background:{DARK_BG};
                                border-radius:8px; padding:14px 16px;'>
                        <div style='color:{ACCENT_COLOR}; font-weight:700; font-size:0.85rem;
                                    margin-bottom:8px;'>Stuff+ &mdash; pitch level</div>
                        <code style='display:block; background:#2A2D3A; padding:8px 10px;
                                     border-radius:6px; color:{WHITE}; font-size:0.8rem;
                                     margin-bottom:10px; line-height:1.6;'>
                            Stuff+ = 100 + 10 &times; (predicted_whiff_prob &minus; group_mean) / group_std
                        </code>
                        <div style='color:{TEXT_MUTED}; font-size:0.82rem; line-height:1.6;'>
                            Every individual pitch gets scored. The predicted whiff probability
                            comes from the logistic regression, then it's z-scored within that
                            pitch type + role group. So a four-seam fastball is only compared
                            against other four-seamers from the same role (SP or RP).
                        </div>
                        <table style='width:100%; margin-top:12px; border-collapse:collapse;
                                      font-size:0.8rem;'>
                            <tr style='border-bottom:1px solid #2A2D3A;'>
                                <td style='padding:6px 10px; color:{WHITE}; font-weight:700;
                                           white-space:nowrap; font-family:monospace;'>predicted_whiff_prob</td>
                                <td style='padding:6px 10px; color:{TEXT_MUTED}; line-height:1.5;'>
                                    The logistic regression's predicted probability of a whiff
                                    for this specific pitch (0 to 1)
                                </td>
                            </tr>
                            <tr style='border-bottom:1px solid #2A2D3A;'>
                                <td style='padding:6px 10px; color:{WHITE}; font-weight:700;
                                           white-space:nowrap; font-family:monospace;'>group_mean</td>
                                <td style='padding:6px 10px; color:{TEXT_MUTED}; line-height:1.5;'>
                                    Average predicted whiff probability across all pitches
                                    of the same type + role
                                </td>
                            </tr>
                            <tr>
                                <td style='padding:6px 10px; color:{WHITE}; font-weight:700;
                                           white-space:nowrap; font-family:monospace;'>group_std</td>
                                <td style='padding:6px 10px; color:{TEXT_MUTED}; line-height:1.5;'>
                                    Standard deviation of those predicted probabilities —
                                    measures how spread out the scores are within the group
                                </td>
                            </tr>
                        </table>
                    </div>

                    <div style='flex:1; min-width:200px; background:{DARK_BG};
                                border-radius:8px; padding:14px 16px;'>
                        <div style='color:{ACCENT_COLOR}; font-weight:700; font-size:0.85rem;
                                    margin-bottom:8px;'>Arsenal Stuff+ &mdash; pitcher level</div>
                        <code style='display:block; background:#2A2D3A; padding:8px 10px;
                                     border-radius:6px; color:{WHITE}; font-size:0.8rem;
                                     margin-bottom:10px; line-height:1.6;'>
                            Arsenal Stuff+ = &Sigma;(n_pitches_of_type &times; Stuff+_of_type) / total_pitches
                        </code>
                        <div style='color:{TEXT_MUTED}; font-size:0.82rem; line-height:1.6;'>
                            It's a weighted average — each pitch type's Stuff+ score is weighted
                            by how often that pitcher actually throws it. Example: 55% four-seamers
                            (108) + 30% sliders (114) + 15% changeups (97):<br><br>
                            <code style='background:#2A2D3A; padding:2px 6px; border-radius:3px;
                            color:{WHITE}; font-size:0.79rem;'>(0.55&times;108) + (0.30&times;114) + (0.15&times;97) = 108.0</code>
                        </div>
                        <table style='width:100%; margin-top:12px; border-collapse:collapse;
                                      font-size:0.8rem;'>
                            <tr style='border-bottom:1px solid #2A2D3A;'>
                                <td style='padding:6px 10px; color:{WHITE}; font-weight:700;
                                           white-space:nowrap; font-family:monospace;'>n_pitches_of_type</td>
                                <td style='padding:6px 10px; color:{TEXT_MUTED}; line-height:1.5;'>
                                    Number of pitches thrown of that specific pitch type by this pitcher
                                </td>
                            </tr>
                            <tr style='border-bottom:1px solid #2A2D3A;'>
                                <td style='padding:6px 10px; color:{WHITE}; font-weight:700;
                                           white-space:nowrap; font-family:monospace;'>Stuff+_of_type</td>
                                <td style='padding:6px 10px; color:{TEXT_MUTED}; line-height:1.5;'>
                                    Mean pitch-level Stuff+ score for that pitch type for this pitcher
                                </td>
                            </tr>
                            <tr>
                                <td style='padding:6px 10px; color:{WHITE}; font-weight:700;
                                           white-space:nowrap; font-family:monospace;'>total_pitches</td>
                                <td style='padding:6px 10px; color:{TEXT_MUTED}; line-height:1.5;'>
                                    Total pitches thrown across all pitch types by this pitcher
                                </td>
                            </tr>
                        </table>
                    </div>

                </div>

                <div style='margin-top:14px; background:#0E1117; border-radius:8px;
                            padding:14px 18px; border-left:3px solid #4A90D9;'>
                    <div style='color:#4A90D9; font-weight:700; font-size:0.85rem;
                                margin-bottom:8px;'>Is Arsenal Stuff+ biased by pitch count?</div>
                    <div style='color:{TEXT_MUTED}; font-size:0.82rem; line-height:1.7;'>
                        <b style='color:{WHITE};'>No.</b> The formula divides by total pitches, so it's a weighted
                        <em>average</em>, not a sum. It doesn't matter whether a pitcher threw 200 pitches or
                        2,000 — the score is always on the same 100-point scale.<br><br>
                        What <b style='color:{WHITE};'>does</b> matter is pitch mix. A pitcher who throws 3 pitch types
                        but one of them is terrible will have that bad pitch drag their Arsenal Stuff+ down
                        proportionally to how often they throw it. A pitcher who throws only one elite pitch
                        will have a high score regardless of volume.<br><br>
                        The only legitimate bias is that <b style='color:{WHITE};'>pitchers with very few pitches
                        get less stable estimates</b> — a reliever who threw 80 pitches in a season has a noisier
                        score than one who threw 500. That's why the leaderboard applies a minimum pitch threshold
                        (200 for RP, 500 for SP) to filter out small-sample noise. The score itself isn't inflated
                        by volume — it's just less reliable at low volumes.
                    </div>
                </div>

            </div>
        </div>
        """)

        # How to use it
        st.markdown(f"""
        <div style='background:{CARD_BG}; border-radius:12px; padding:24px 28px;
                    border:1px solid #2A2D3A;'>
            <div style='color:{ACCENT_COLOR}; font-weight:800; font-size:1.15rem;
                        margin-bottom:16px;'>How to Use Each Page</div>
            <table style='width:100%; border-collapse:collapse; font-size:0.88rem;'>
                <tr style='border-bottom:1px solid #2A2D3A;'>
                    <td style='padding:10px 12px; color:{WHITE}; font-weight:700; width:25%;'>Overview</td>
                    <td style='padding:10px 12px; color:{TEXT_MUTED};'>
                        High-level validation of the model. Decile charts and the pitch-type outcomes table
                        pool data across {POOLED_SEASONS_LABEL} (including live {CURRENT_SEASON}). The
                        Stuff+ distribution violin chart is year-by-year — use the season selector in the sidebar.
                    </td>
                </tr>
                <tr style='border-bottom:1px solid #2A2D3A;'>
                    <td style='padding:10px 12px; color:{WHITE}; font-weight:700;'>Pitcher Explorer</td>
                    <td style='padding:10px 12px; color:{TEXT_MUTED};'>
                        Search any pitcher. See their full pitch arsenal with individual Stuff+
                        scores, a SHAP breakdown showing <em>why</em> their pitches score the
                        way they do, and a year-over-year trend line.
                    </td>
                </tr>
                <tr style='border-bottom:1px solid #2A2D3A;'>
                    <td style='padding:10px 12px; color:{WHITE}; font-weight:700;'>Leaderboard</td>
                    <td style='padding:10px 12px; color:{TEXT_MUTED};'>
                        Ranked pitcher table by any metric. Filter by role (SP/RP) or specific
                        pitch type. Scatter plots show the relationship between Stuff+ and
                        outcomes at the pitcher level.
                    </td>
                </tr>
                <tr style='border-bottom:1px solid #2A2D3A;'>
                    <td style='padding:10px 12px; color:{WHITE}; font-weight:700;'>How the Model Works + Important Findings</td>
                    <td style='padding:10px 12px; color:{TEXT_MUTED};'>
                        The model internals. Feature weight heatmap across all pitch types,
                        SHAP beeswarm for four-seamers, and AUC scores showing how predictive
                        the model is for each pitch type.
                    </td>
                </tr>
                <tr>
                    <td style='padding:10px 12px; color:{WHITE}; font-weight:700;'>About</td>
                    <td style='padding:10px 12px; color:{TEXT_MUTED};'>
                        You are here. Background on the project, and full methodology.
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style='background:{CARD_BG}; border-radius:12px; padding:24px 28px;
                    border:1px solid #2A2D3A; margin-bottom:20px;'>
            <div style='color:{ACCENT_COLOR}; font-weight:800; font-size:1.1rem;
                        margin-bottom:14px;'>Data & Methodology</div>
            <div style='color:{TEXT_MUTED}; font-size:0.87rem; line-height:1.8;'>
                <b style='color:{WHITE};'>Source</b><br>
                Baseball Savant Statcast<br>
                MLB seasons 2023–{CURRENT_SEASON} ({CURRENT_SEASON} live)<br>
                ~{TOTAL_PITCHES:,} individual pitches<br><br>
                <b style='color:{WHITE};'>Modeling approach</b><br>
                L2-penalized logistic regression<br>
                Fit separately per pitch type<br>
                Separate models for SP and RP<br>
                Outcome: whiff on swing (binary)<br><br>
                <b style='color:{WHITE};'>Scoring</b><br>
                Predicted whiff probability<br>
                → z-scored within role + pitch type<br>
                → Stuff+ (100 = avg, ±10 = 1 SD)<br><br>
                <b style='color:{WHITE};'>Explainability</b><br>
                LR coefficients (linear weights)<br>
                + SHAP values from gradient<br>
                boosted model (nonlinear audit)<br><br>
                <b style='color:{WHITE};'>Role classification</b><br>
                SP: avg ≥ 50 pitches/appearance<br>
                RP: avg &lt; 50 pitches/appearance
            </div>
        </div>
        <div style='background:{CARD_BG}; border-radius:12px; padding:24px 28px;
                    border:1px solid #2A2D3A;'>
            <div style='color:{ACCENT_COLOR}; font-weight:800; font-size:1.1rem;
                        margin-bottom:14px;'>Stuff+ Scale</div>
            <div style='font-size:0.87rem;'>
                {''.join([
                    f"<div style='display:flex; align-items:center; margin-bottom:8px;'>"
                    f"<span style='background:{color}; width:12px; height:12px; border-radius:50%;"
                    f"display:inline-block; margin-right:10px; flex-shrink:0;'></span>"
                    f"<span style='color:{WHITE}; font-weight:700; width:40px;'>{band}</span>"
                    f"<span style='color:{TEXT_MUTED};'>{desc}</span></div>"
                    for color, band, desc in [
                        ("#00C851", "115+",  "Elite — top ~2% of pitches"),
                        ("#7BCF5C", "110+",  "Excellent — top ~10%"),
                        ("#B8D96A", "105+",  "Above average"),
                        ("#FFD700", "100",   "League average"),
                        ("#FFA040", "95–99", "Below average"),
                        ("#FF4444", "<95",   "Well below average"),
                    ]
                ])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    show_glossary()

    st.markdown("<br><br>", unsafe_allow_html=True)
