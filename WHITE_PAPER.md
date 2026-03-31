# Arsenal Intelligence: Interpretable Pitch-Level Stuff Modeling for Baseball Operations

**Technical white paper**  
**Version 1.0** — March 2026  

---

## Executive summary

This document summarizes the methodology, empirical validation, and baseball-operations implications of **Arsenal Intelligence**, an interpretable modeling system that scores MLB pitch quality using three seasons (2023–2025) of Statcast pitch-level data (~2.16 million pitches). The primary model predicts **swing-and-miss probability** from purely physical pitch characteristics—velocity, movement, spin, extension, and plate location—fit separately by **pitch type** and **pitcher role** (starter vs. reliever). Outputs are expressed as **Stuff+** (pitch-type-level) and **Arsenal Stuff+** (usage-weighted arsenal summaries), with **logistic regression coefficients** and **SHAP values** from auxiliary gradient-boosted models supporting transparent explanation.

The work demonstrates how **interpretable predictive models** can be deployed in a front-office context: not as a replacement for scouting or medical judgment, but as a **common language** between analysts, coaches, and player-development staff—grounded in validated relationships between measurable pitch properties and outcomes, with explicit limits on what the score does and does not capture.

---

## 1. Problem statement and operational relevance

Professional baseball organizations routinely ask:

- Which pitchers have **underlying stuff** that may not yet show up in run prevention?
- For a given pitcher, which **physical levers** (velocity, shape, location) most drive swing-and-miss relative to peers throwing the same pitch type in the same role?
- How do we compare **starters** and **relievers** fairly, given different usage patterns and average velocity bands?

Traditional outcomes (ERA, FIP, whiff rate alone) blend **stuff**, **command**, **sequencing**, **defense**, and **luck**. Arsenal Intelligence isolates a **stuff-centric slice** of the problem: conditional on a swing, how much does the *physics of the pitch* associate with missing the bat? That framing aligns with how player-development and scouting groups often reason about “raw stuff” versus “execution”—and it is deliberately narrow so that the model’s limits remain clear.

---

## 2. Data scope and preprocessing

| Item | Specification |
|------|----------------|
| Source | Baseball Savant (Statcast), via `pybaseball` |
| Seasons | 2023, 2024, 2025 (pooled for modeling and scoring) |
| Unit of observation | Individual pitches |
| Swing subset | Only pitches on which the batter swung (whiff vs. non-whiff is defined on that set) |
| Pitch types | Mapped from Statcast codes to canonical groups (e.g., FF/FA → Four-Seam FB; ST → Sweeper); rare types excluded below minimum swing thresholds |

**Quality filters** applied before modeling include plausible ranges for release speed, spin rate, and movement (see `precompute.py`) to reduce tracking errors and corrupted rows.

**Role classification:** For each pitcher-season, **average pitches per game appearance** is computed. **≥50 pitches/game** classifies as **starter (SP)**; **&lt;50** as **reliever (RP)**. Manual overrides are used for known split-role cases (e.g., injury-return usage changes). All models and z-score baselines are **role-aware**, so a reliever’s four-seamer is compared to other reliever four-seamers, not to starters’.

---

## 3. Modeling approach

### 3.1 Primary model: L2 logistic regression (per pitch type × role)

For each combination of **pitch group** *G* and **role** *R* (SP or RP), a separate **pipeline** is fit:

1. **Standardize** the feature vector (zero mean, unit variance within the training fold).
2. **Logistic regression** with L2 penalty (`C=1.0`) predicts **P(whiff | swing)**.

**Features (seven):**

| Feature | Baseball interpretation |
|--------|-------------------------|
| Perceived velocity | Release speed adjusted for extension (approximation of hitter perception) |
| Induced vertical break | Vertical movement (inches) |
| Arm-side horizontal break | Horizontally oriented break relative to throwing arm |
| Spin rate | Revolutions per minute |
| Release extension | Distance down the mound toward home plate |
| Horizontal plate location | Left/right in the strike zone plane |
| Vertical plate location | Up/down in the strike zone plane |

**Outcome:** Binary **whiff** on swings (swinging strikes / foul tip, as coded in Statcast).

**Minimum training volume:** Pitch type × role groups require **≥2,000 swing events** to train a stable model; groups below that are not assigned a dedicated model (prevents unstable coefficients on rare combinations).

**Model evaluation:** **5-fold stratified cross-validation**; performance reported as **mean ROC-AUC** by pitch type and role.

### 3.2 From probability to Stuff+

For each scored pitch, the model outputs a predicted whiff probability **p**. Within each **(role, pitch group)** group, **p** is z-scored:

**Stuff+ = 100 + 10 × (p − μ_group) / σ_group**

- **100** = league average for that pitch type and role (on the predicted-probability scale).  
- **Each 10 points** ≈ **one standard deviation** within that group.

This scaling is comparable across years within the pooled sample and preserves interpretability for non-technical stakeholders (“110 is roughly one SD above average for that pitch type as a starter/reliever”).

### 3.3 Arsenal Stuff+

**Arsenal Stuff+** is a **usage-weighted average** of pitch-type **Stuff+** values for a pitcher-season (each pitch type’s mean Stuff+ weighted by share of total pitches). It is **not** a sum over pitch count—pitch volume does not inflate the index; **mix** matters (a poor pitch thrown often drags the average proportionally).

### 3.4 Explainability stack

- **Linear coefficients** from logistic regression provide a **global, auditable** “recipe” per pitch type and role (e.g., relative emphasis of vertical location vs. velocity for four-seamers).
- **Gradient-boosted classifiers** and **SHAP** values are used as a **nonlinear audit**—e.g., highlighting **spin × velocity** interactions that a linear model may understate—while the primary score remains anchored in the interpretable LR structure.

---

## 4. Empirical validation: what the numbers show

### 4.1 Discriminative power (AUC) by pitch type and role

ROC-AUC answers: “How well do physical features alone separate whiffs from non-whiffs on swings?” **High AUC** means shape and velocity are **strongly informative** for that pitch type; **low AUC** means whiffs are driven by other factors (or the pitch is not primarily a whiff pitch).

Representative **5-fold CV AUC** results from the fitted models include:

| Pitch type (example) | Role | Mean AUC (illustrative) |
|----------------------|------|-------------------------|
| Knuckle-Curve | RP | **0.839** |
| Knuckle-Curve | SP | **0.811** |
| Curveball | SP | **0.781** |
| Splitter | RP | **0.751** |
| Slider | SP | **0.735** |
| Sweeper | SP | **0.720** |
| Changeup | SP | **0.695** |
| Four-Seam FB | SP | **0.674** |
| Cutter | SP | **0.549** |
| Sinker | SP | **0.538** |

**Interpretation:** Breaking balls and splitters show **high** AUC—swing-and-miss on those pitches is **well explained** by measured movement and location. **Sinkers and cutters** show **modest** AUC—consistent with their **primary value** often being **weak contact and ground balls**, not empty swings. The model is **honest** about where “stuff features” explain outcomes: it is strongest for whiff-oriented pitch types.

### 4.2 Outcome alignment (deciles and correlations)

Aggregated **decile** analyses (by pitch-level Stuff+) show **monotonic** relationships with **whiff rate** and **contact quality** (e.g., xwOBA): higher Stuff+ bins associate with more whiffs and better contact suppression on average. **Pitcher-level** correlations between arsenal-weighted metrics and outcomes are **meaningful but not perfect** (e.g., whiff rate correlations on the order of **r ≈ 0.40** in line with README summaries)—which is **expected**: sequencing, game planning, defense, and command beyond a single pitch’s location all matter for runs.

### 4.3 Substantive findings (cross-pitch themes)

Consistent with the coefficient and SHAP analyses:

1. **For four-seam fastballs, vertical plate location often matters as much as or more than raw velocity** in the linear model—“flat” premium velocity can underperform well-located average velocity.
2. **Pitch-type heterogeneity** is large: curveballs and sweepers emphasize different movement axes than sinkers; treating all pitches with one global model would **obscure** these differences—hence **separate** models per pitch type (and role).
3. **Nonlinear interactions** (e.g., high spin at high velocity) appear in SHAP-based diagnostics even when the primary score is linear—supporting a **dual** use: LR for communication, GBM+SHAP for **sanity checks** and coaching conversations about “when velo and spin work together.”

---

## 5. Front office applications (how this maps to real decisions)

### 5.1 Scouting and acquisition

- **Stuff+ vs. results gaps:** Pitchers with **high Arsenal Stuff+** but mediocre run prevention or whiff rates may warrant deeper **video, medical, and role** review—potential buy-low candidates if command or health improves.
- **Pitch-type granularity:** Comparisons within **pitch type and role** reduce **apples-to-oranges** errors (e.g., comparing a starter’s sinker usage to a reliever’s max-effort four-seamer).

### 5.2 Player development

- **Coefficient and SHAP panels** help phrase **coachable questions**: “For this pitcher’s slider, is the model flagging **shape** or **location** as the gap vs. MLB average?”—not as automated instruction, but as **prioritization** for lab and bullpen work.
- **Role-specific baselines** align with how organizations already separate **development tracks** for starters vs. relievers.

### 5.3 Research and communication

- **Interpretable models** travel well across departments: analytics can show **explicit weights**, not only leaderboard ranks.
- **Thresholds** (minimum swings for training, minimum pitches for leaderboard inclusion, minimum pitches per type for pitcher-level SHAP) are **policy choices** that can be documented and debated—mirroring how teams set **qualifying plate appearance** rules for internal reports.

---

## 6. Limitations and responsible use

1. **Outcome choice:** The primary target is **whiff on swing**, not **runs** or **xwOBA on contact** alone. Sinkers and cutters may be **underrated** if their value is **ground balls**, not misses.
2. **No game state:** Count, inning, batter handedness, and pitch sequencing are **not** in the model—**by design** for purity of “stuff,” but **not** a full picture of pitcher effectiveness.
3. **Sample size and thresholds:** Small-sample pitchers and rare pitch types are **down-weighted or excluded** by rule; edge cases should be flagged in any report.
4. **Not a stand-alone decision system:** The model supports **hypothesis generation**, **prioritization**, and **communication**—not contract or roster decisions without **human** and **medical** context.

---

## 7. Conclusion

Arsenal Intelligence implements a **transparent, validated** pipeline from Statcast physics to **Stuff+** and **Arsenal Stuff+**, with **role-aware** and **pitch-type-specific** modeling, **documented** performance (AUC, deciles), and a **dual** explainability layer (logistic regression + SHAP). For baseball operations, the value proposition is not a single number on a page—it is a **repeatable framework** for asking: *what does the data say about raw pitch quality, where does that align with outcomes, and where must we bring in everything else the model deliberately leaves out?*

---

## References and artifacts

- **Interactive application:** Streamlit dashboard (leaderboards, pitcher explorer, methodology, glossary).  
- **Codebase:** `precompute.py` (training, scoring, aggregates), `app.py` (visualization and UI).  
- **Data:** Baseball Savant Statcast, 2023–2025.

---

*This white paper describes the author’s independent research implementation and is not affiliated with or endorsed by Major League Baseball or any MLB club.*
