# MMM Analysis Report
*Generated 2026-05-01 · 128 training periods · 4 holdout periods*

## The Three Models

### Ridge Regression MMM
*Fast, transparent baseline — always runs*

Fits a regularised linear regression on adstock-transformed spend. Bootstraps 200 samples to estimate coefficient uncertainty. Alpha (regularisation strength) selected via 5-fold cross-validation.

**Pros:** Runs in seconds — no heavy dependencies · Coefficients are fully interpretable · Bootstrap gives confidence intervals per channel · Good baseline to sanity-check other models against

**Cons:** Assumes linear spend-response (no saturation curve) · Prone to overfitting on small samples (R²=1.0 is a warning sign) · Shrinks small coefficients to zero — may under-attribute niche channels · No uncertainty propagation — point estimates only

**Best for:** Quick first-pass analysis when data is limited · Sanity-checking Bayesian results · Teams without Python/stats expertise (simple to explain) · Datasets with 50+ weekly observations and low collinearity

### Bayesian MMM (PyMC-Marketing)
*Full posterior uncertainty — the gold standard*

Uses DelayedSaturatedMMM from PyMC-Marketing. Fits a full Bayesian model with prior distributions on adstock decay, Hill saturation, and channel coefficients. MCMC sampling (NUTS) produces a posterior distribution over all parameters — every ROI estimate comes with a credible interval.

**Pros:** Full uncertainty quantification — know how confident you are · Encodes domain knowledge via priors (e.g. TV decays slower than Digital) · Handles small samples better by regularising through priors · Posterior predictive checks catch model misspecification · Industry standard for rigorous MMM (used by Meta, Google)

**Cons:** Slow — 10–60 minutes depending on data size and chain count · Requires pymc-marketing installation and familiarity with Bayesian stats · Prior choices matter — wrong priors can bias results · Results harder to explain to non-technical stakeholders

**Best for:** Production MMM where budget decisions involve significant spend · Datasets with 80+ weekly observations · When you need credible intervals on ROI for board-level decisions · Teams with data science capability to interpret posteriors

### LightweightMMM / NNLS
*Google's JAX-based MMM with NNLS fallback*

Google's LightweightMMM uses JAX for fast GPU/CPU Bayesian inference with built-in Hill adstock. When JAX is unavailable, falls back to Non-Negative Least Squares (NNLS via scipy) — a constrained regression that enforces positive channel coefficients, preventing the negative-ROI problem common in Ridge.

**Pros:** NNLS fallback enforces positive ROI — no negative coefficients · LightweightMMM is significantly faster than PyMC on GPU · Built-in Hill + adstock transforms — no manual feature engineering · Open source from Google — well tested at scale · Good middle ground between Ridge speed and PyMC rigour

**Cons:** NNLS fallback loses uncertainty quantification · JAX installation can be complex (especially on Mac M-series) · LightweightMMM is less actively maintained than PyMC-Marketing · NNLS can over-attribute to channels that happen to correlate with KPI

**Best for:** Teams who want faster Bayesian inference than PyMC · Google Cloud / GCP environments where JAX runs natively · As a cross-check against PyMC results · When you need positive-constrained estimates without full Bayes

## Model Comparison

| Dimension | Ridge | PyMC | LightweightMMM |
|---|---|---|---|
| Speed | ⚡ Seconds | 🐢 10–60 min | 🔄 Minutes (JAX) |
| Uncertainty | Bootstrap CIs | Full posterior | NNLS: none / JAX: posterior |
| Saturation curve | ❌ Linear only | ✅ Hill (built-in) | ✅ Hill (built-in) |
| Adstock | Manual (prepare.py) | ✅ Built-in | ✅ Built-in |
| Negative ROI risk | ⚠️ Yes (shrinkage) | Low (priors) | ✅ No (NNLS positive) |
| Min data needed | 50+ weekly obs | 80+ weekly obs | 50+ weekly obs |
| Install complexity | ✅ pip sklearn | ⚠️ pymc-marketing | ⚠️ JAX required |
| Explainability | ✅ High | Medium | Medium |
| Industry adoption | Baseline standard | ⭐ Gold standard | Google internal |

## When to Use Which Model

| Scenario | Data | Recommended | Reason |
|---|---|---|---|
| Quick first-pass analysis | Any size | **Ridge** | Results in seconds, easy to interpret, good sanity check |
| Small dataset (<30 periods) | Monthly, <30 obs | **PyMC** | Priors regularise better than Ridge shrinkage on thin data |
| Production budget decisions | Weekly, 80+ obs | **PyMC** | Full credible intervals needed for significant spend decisions |
| Fast iteration / experiment loop | Weekly, 50+ obs | **Ridge → LightweightMMM** | Ridge for speed, LightweightMMM for positive-constrained check |
| GCP / GPU environment | Any size | **LightweightMMM** | JAX runs natively on GCP — fastest Bayesian option |
| Board-level reporting | Weekly, 100+ obs | **PyMC + Ridge as sanity check** | Posterior credible intervals defensible; Ridge validates direction |
| High collinearity between channels | Any | **PyMC** | Priors on each channel help disentangle correlated spend patterns |

## Model Fit (This Run)

| model           |   train_r2 | train_mape   | test_mape   | status                                                                |
|:----------------|-----------:|:-------------|:------------|:----------------------------------------------------------------------|
| ridge           |     0.3538 | 39.06%       | 23.17%      | OK                                                                    |
| pymc            |     0.7954 | 23.0%        | 42.83%      | OK                                                                    |
| lightweight_mmm |     0.5366 | 36.32%       | 15.06%      | LightweightMMM/JAX timed out after 120s — used BayesianRidge fallback |

## ROI by Channel

*Higher = more revenue per unit of spend. Agreement = cross-model consistency.*

| channel         |   lightweight_mmm |      pymc |    ridge | has_negative   |   mean_roi |   std_roi |   cv_pct | agreement      |
|:----------------|------------------:|----------:|---------:|:---------------|-----------:|----------:|---------:|:---------------|
| meta_other      |            7.5496 | 1089.69   | 264.217  | False          |   453.818  |  565.436  |    124.6 | ❌ Low         |
| google_shopping |            9.5612 |  230.053  |  16.7001 | False          |    85.4379 |  125.291  |    146.6 | ❌ Low         |
| google_search   |           23.3215 |   48.621  |  40.0696 | False          |    37.3374 |   12.8691 |     34.5 | ⚠️ Medium       |
| meta_instagram  |            1.6533 |    2.9215 |   1.2977 | False          |     1.9575 |    0.8536 |     43.6 | ⚠️ Medium       |
| meta_facebook   |            1.5975 |    1.5231 |   1.4767 | False          |     1.5324 |    0.0609 |      4   | ✅ High        |
| google_video    |          -22.4064 |   23.4488 |  -4.0132 | True           |    -0.9903 |   23.0766 |   2330.3 | ⚠️ Negative ROI |
| google_pmax     |          -12.9722 |    5.4721 |   0.4417 | True           |    -2.3528 |    9.5344 |    405.2 | ⚠️ Negative ROI |
| google_display  |          -78.512  |   95.0974 | -51.03   | True           |   -11.4815 |   93.3173 |    812.8 | ⚠️ Negative ROI |

## Channel Contribution (%)

| channel         |   lightweight_mmm |   pymc |   ridge |   mean_pct |   spread |
|:----------------|------------------:|-------:|--------:|-----------:|---------:|
| meta_facebook   |             38.74 |  24.65 |   35.81 |      33.07 |    14.09 |
| google_search   |             12.7  |  22.1  |   21.82 |      18.87 |     9.4  |
| meta_instagram  |             12.34 |  14.71 |    9.69 |      12.25 |     5.02 |
| google_shopping |              1.09 |  20.13 |    1.9  |       7.71 |    19.04 |
| meta_other      |              0.13 |  12.7  |    4.59 |       5.81 |    12.57 |
| google_video    |            -15.96 |   9.85 |   -2.86 |      -2.99 |    25.81 |
| google_display  |            -20.79 |  15.81 |  -13.51 |      -6.16 |    36.6  |
| google_pmax     |            -30.55 |   9.93 |    1.04 |      -6.53 |    40.48 |

## Top Channels by Model

- **ridge**: meta_other, google_search, google_shopping
- **pymc**: meta_other, google_shopping, google_display
- **lightweight_mmm**: google_search, google_shopping, meta_other

## High Disagreement Channels

*Interpret with caution — models give significantly different answers.*

| channel         |   mean_roi |   cv_pct | agreement      |
|:----------------|-----------:|---------:|:---------------|
| meta_other      |   453.818  |    124.6 | ❌ Low         |
| google_shopping |    85.4379 |    146.6 | ❌ Low         |
| google_search   |    37.3374 |     34.5 | ⚠️ Medium       |
| meta_instagram  |     1.9575 |     43.6 | ⚠️ Medium       |
| google_video    |    -0.9903 |   2330.3 | ⚠️ Negative ROI |
| google_pmax     |    -2.3528 |    405.2 | ⚠️ Negative ROI |
| google_display  |   -11.4815 |    812.8 | ⚠️ Negative ROI |

## Data Caveat

This analysis uses 132 monthly periods. MMM typically benefits from 2+ years of weekly data. Interpret uncertainty ranges with this in mind.
