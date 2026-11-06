# IPA Forecasting Technical Optimization Note (2026-03-04)

## 1) Why this rework was needed

Baseline run (before this rework) showed a large mismatch between cross-validation and holdout:

- `CV_MAPE` (tree model): about `18.54%`
- Holdout `MAPE` (tree model): about `1.72%`

This gap suggested the old pipeline had evaluation inconsistency and feature-time alignment issues.

## 2) Three optimization rounds executed

### Round 1: Feature Engineering + Time Alignment

- Switched to **quarterly-first feature generation** for modeling.
- Removed the pattern of "weekly engineered features then quarterly averaging", which can blur lag/rolling semantics.
- Added dedicated quarterly feature pipeline:
  - target lag/rolling/change on quarterly index
  - time features on quarterly timestamps
  - exogenous lag/rolling/interaction features on quarterly index

Validation:

- Unit tests passed after refactor.
- Pipeline still reproducible with cached data and same CLI.

### Round 2: Model Selection + Backtesting

- Implemented walk-forward backtest for both tree model and SARIMA.
- Added SARIMA hyperparameter search under the same walk-forward protocol.
- Added fold-level diagnostics export (`walk_forward_backtest.csv`).
- Ensemble weights now prefer **CV error** over test error to avoid leakage.

Validation:

- 55 unit tests passing.
- CV metrics and holdout metrics are now in a consistent scale.

### Round 3: Ensemble Robustness + Forecast Consistency + Output

- Fixed ensemble blending bug where absent model weights could still be mixed in.
- Added robust weighting:
  - error inverse-square weighting
  - automatic filtering of clearly weak models (`eligibility_ratio = 3.0`)
- Future recursive prediction now recomputes exogenous derived features (`lag`, `rolling`, `interaction`) each step.
- Extended outputs:
  - `walk_forward_backtest.csv`
  - manifest now includes backtest row count
  - report includes backtest artifact link

Validation:

- Holdout 2025 run after final round:
  - `GradientBoosting` MAPE: `2.31%`
  - `CV_MAPE`: `4.73%`
  - `Ensemble` auto-fallback to best stable model when alternative is weak

## 3) Architecture impact

- Data path is now clearer:
  1. Weekly merge and cleaning
  2. Quarterly aggregation
  3. Quarterly feature engineering
  4. Walk-forward tuning/backtesting
  5. Holdout evaluation + recursive future generation
- This structure is easier to reason about for temporal causality and auditing.

## 4) Research basis used for design choices

Primary sources used to guide this rework:

1. XGBoost objective and regularized tree boosting
   - Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*.
   - DOI: https://doi.org/10.1145/2939672.2939785

2. Rolling-origin / out-of-sample forecasting evaluation
   - Tashman, L. J. (2000). *Out-of-sample tests of forecasting accuracy: an analysis and review*.
   - DOI: https://doi.org/10.1016/S0169-2070(00)00065-0

3. Time-series cross-validation cautions and blocked/rolling validation
   - Bergmeir, C., & Benitez, J. M. (2012). *On the use of cross-validation for time series predictor evaluation*.
   - DOI: https://doi.org/10.1016/j.ins.2011.12.028

4. Multi-step forecasting strategy trade-offs (recursive/direct hybrid idea)
   - Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). *Recursive and direct multi-step forecasting: the best of both worlds*.
   - Working paper: https://robjhyndman.com/publications/rectify/index.html

5. Forecast combination and robust benchmark behavior at scale
   - Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). *The M4 Competition: Results, findings, conclusion and way forward*.
   - DOI: https://doi.org/10.1016/j.ijforecast.2018.06.001

6. Practical time-series cross-validation guidelines
   - Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*, section 5.10.
   - https://otexts.com/fpp3/tscv.html

## 5) What remains known-risk

- SARIMA may raise convergence warnings on some folds due to short quarterly samples.
- In this environment, `xgboost` package is unavailable, so tree backend falls back to `GradientBoostingRegressor`.
- If external data distribution shifts sharply, exogenous projection still depends on recent-trend assumptions (damped drift).
