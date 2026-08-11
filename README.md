# Arsenal Intelligence — MLB Pitch Stuff+ Dashboard

An interpretable pitch quality scoring model built on Statcast pitch-level data (2023–2026).

## What It Does

Builds a **Stuff+ score** for every MLB pitch — a weighted composite of velocity, movement, spin, and location — that predicts swing-and-miss probability. The score is explainable by design: you can see exactly which physical characteristics drive each pitcher's rating.

## Live App

Deploy on [Streamlit Community Cloud](https://share.streamlit.io) from this repo (`app.py` entrypoint).  
Repo: **github.com/shirinalapati/arsenal-intelligence**

**2026 live season:** GitHub Actions refreshes Statcast data **4× daily** through the regular season (ends ~Oct 5). The app reloads parquet outputs every **2 minutes**. Seasons 2023–2025 are frozen benchmarks.

## Pages

| Page | Description |
|---|---|
| **Overview** | Key metrics, decile validation charts, pitch-type distributions |
| **Pitcher Explorer** | Search any pitcher — arsenal breakdown, SHAP explanation, season trend |
| **Leaderboard** | Ranked table + Stuff+ vs. outcomes scatter for any season |
| **How It Works** | Model design, feature weight heatmap, SHAP beeswarm |

## Methodology

- **Data**: Baseball Savant Statcast, 2023–2025, ~2.16M pitches
- **Model**: Pitch-type-specific L2 logistic regression predicting whiff probability on swings
- **Features**: Perceived velocity, induced vertical break, arm-side horizontal break, spin rate, release extension, plate location
- **Score**: Predicted whiff probability z-score normalized within each pitch type (100 = league avg, 10pts = 1 SD)
- **Explainability**: Logistic regression coefficients (linear weights) + SHAP values from gradient boosted model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fetch data (run once)
python fetch_statcast.py
python fetch_2025.py

# Precompute model outputs (run once after fetching)
python precompute.py

# Launch app
streamlit run app.py
```

## Key Findings

- **Location dominates fastball stuff** — vertical plate location has a higher coefficient than velocity for four-seamers
- **Pitch-type specificity matters** — a curveball's value comes from drop; a sweeper's from horizontal sweep; a sinker's from nothing (AUC 0.54 — sinkers are ground-ball pitches, not whiff pitches)
- **SHAP confirms nonlinear spin × velocity interaction** — high-spin fastballs at 96+ mph are disproportionately effective
- **Stuff+ correlates r=0.40 with whiff rate** — strong signal, but sequencing and command still matter
- **Best stuff pitchers are almost all relievers** — Félix Bautista (117.8), Aroldis Chapman (112.3), Josh Hader (110.3)
