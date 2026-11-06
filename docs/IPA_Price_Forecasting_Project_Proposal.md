# IPA Price Prediction Project Optimization Proposal

Document version: v1.1  
Updated: 2026-02-22  
Applicable period: 2025Q1–2026Q4 forecasting and course deliverable

## 1. Project Positioning and Goals

This project serves as an IPA (Isopropyl Alcohol) price prediction application for the Engineering Economics course, emphasizing **interpretability, reproducibility, and deliverability**.

Key objectives:
- Build a reproducible quarterly forecasting pipeline that outputs 2025 and 2026 Q1–Q4 price ranges.
- Incorporate multi-factor influences reflecting cost, exchange rates, demand, and event shocks.
- Provide charts and HTML reports to support course presentations and report archiving.

## 2. Problem Definition

Prediction target:
- Target variable: IPA unit price (TWD/KG)
- Frequency: Quarterly
- Forecast horizon: 4 quarters per run (Q1–Q4)

Decision use cases:
- Procurement budget estimation
- Cost sensitivity analysis
- Baseline / optimistic / pessimistic scenario comparison

## 3. Data Strategy and Sources

### 3.1 Available Data

1. Target series:
   - Source: IPA price series reconstructed from existing charts (2012-01 to 2024-12)
     (not distributed with this repository; see the README for the data format and the synthetic sample)
   - Frequency: Weekly data, subsequently aggregated to quarterly

2. Exogenous variables:
   - Yahoo Finance: `CL=F`, `BZ=F`, `NG=F`, `USDTWD=X`, `DX-Y.NYB`, `^TWII`, `TSM`
   - Event indicators: COVID-19, Russia-Ukraine War, Red Sea Crisis, US-China Trade War

### 3.2 Data Governance

- Type/field integrity checks
- Time index deduplication and sorting
- Forward-fill only, to avoid look-ahead bias
- Outlier quantile clipping (1%/99%)
- Local cache for stability and reproducibility (`Code/Data/market_data_*.csv`)

## 4. Feature Engineering Design

Core features:
- Autoregressive: lag1–lag4
- Rolling statistics: MA/STD/MAX/MIN (4, 8, 12 windows)
- Rate of change: 1-period, 4-period, 13-period
- Time features: Year/Quarter/Month/Week + sine/cosine seasonal encoding
- Exogenous features: energy prices, exchange rates, stock market proxies, event indicators

Principles:
- Prioritize interpretability
- Maintain recursive compatibility so future periods can generate features

## 5. Modeling Strategy

### 5.1 Model Combination

1. XGBoost:
   - Primary model for handling non-linearity and feature interactions
   - Hyperparameter selection via walk-forward validation

2. SARIMA:
   - Time-series baseline model
   - Provides supplementary trend/seasonal structure

### 5.2 Ensemble Method

- Weights derived from inverse MAPE:
  - Weight = `(1 / MAPE_i) / sum(1 / MAPE_j)`
- Auto-fallback to available model output when a single model fails

### 5.3 Future Period Forecasting

- XGBoost uses recursive quarter-by-quarter prediction:
  - Predict next quarter
  - Back-fill target-derived features (lag/rolling/change)
  - Continue advancing to target quarters
- Exogenous variables use damped drift projection:
  - Estimate drift from the most recent 8 quarters with per-step decay
  - Event-type variables maintain their state values
- SARIMA produces multi-step forecasts directly, sliced to the target year

### 5.4 Uncertainty Intervals

- Intervals are dynamically estimated from historical volatility + test residual distribution
- Avoids distortion from fixed-percentage upper/lower bounds

## 6. Evaluation Methodology

- Time-series train/test split without shuffling temporal order
- Walk-forward validation within the training set

Key metrics:
- MAE
- RMSE
- MAPE
- Direction Accuracy

Suggested acceptance thresholds:
- MAPE <= 8%
- Direction Accuracy >= 65%

## 7. Common Issues and Handling

1. External data format/download failures
   - Robust handling for Yahoo Finance response formats (including MultiIndex close extraction)
   - Caching reduces external dependency risk

2. Time-series data leakage
   - No backward-fill allowed throughout the pipeline
   - Only historical data and recursively generated features are used

3. Repeated training overhead
   - Single training run with multi-year output (`--years 2025 2026`)

4. Structural event shocks
   - Event dummy variables are retained
   - Scenario intervals reflect residual uncertainty

## 8. Final Optimization Results

- `ipa_data_collector.py`
  - Enhanced yfinance compatibility
  - Added cache read/write and refresh control
  - Maintained causal forward-fill
- `ipa_price_prediction.py`
  - CLI parameterization (`--years`, `--refresh-cache`, `--no-cache`)
  - Single training, multi-year output
  - Exogenous variable damped drift projection
  - Recursive quarterly prediction + adaptive uncertainty intervals
- `ipa_models.py`
  - Cleaned up noisy output and unnecessary imports
- Documentation and reports
  - Removed emoji/icon characters
  - Updated README and report outputs

## 9. Closure and Maintenance Recommendations

This project is recommended to enter "closure maintenance mode":
- No new model families or changes to core methodology
- Only essential maintenance is permitted:
  - API compatibility fixes
  - Report year updates
  - Cache refresh

## 10. Execution and Reproduction

Installation:

```bash
cd Code
pip install -r requirements.txt
```

Execution (with your own `Data/ipa_prices.json`, or the synthetic sample):

```bash
python ipa_price_prediction.py --years 2025 2026
python ipa_price_prediction.py --years 2025 2026 --price-file Data/ipa_prices.sample.json
```

Refresh data cache:

```bash
python ipa_price_prediction.py --years 2025 2026 --refresh-cache
```

Output (generated, not tracked in git):
- Charts: `Code/figures/`
- Reports: `Code/reports/ipa_forecast_2025.html`
- Reports: `Code/reports/ipa_forecast_2026.html`
