# IPA Price Forecasting

[![CI](https://github.com/Shuruyue/IPA-Price-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Shuruyue/IPA-Price-Forecasting/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Result.** On the bundled synthetic price series, each quarter is forecast one quarter ahead from what is known at the end of the previous quarter. Over the nine test quarters, SARIMA's mean absolute percentage error is 3.6 %, the naive forecast (next quarter = this quarter) 5.0 %, and gradient-boosted trees 6.1 %, so the tree model does not beat the naive forecast. An earlier version of this repository reported 2.3 % for the tree model on the original data; that figure came from features that contained the price being forecast and is withdrawn ([correction](#correction-of-earlier-results)).

Quarterly price forecasting for isopropyl alcohol (IPA, TWD/kg), built as the final project of an Engineering Economics course. A tree-based model and SARIMA are tuned by walk-forward validation on 2012–2024 data, compared with a naive forecast, combined into an ensemble, and used to forecast 2025 and 2026 quarter by quarter, with energy prices, exchange rates, equity proxies and geopolitical event indicators as exogenous drivers.

> **Data notice.** The IPA price series used in the course project was digitized from a price
> chart whose original source is not recorded, so it is **not distributed** here. The repository
> ships a clearly labelled **synthetic** series (`Code/Data/ipa_prices.sample.json`) with the same
> format and period, so the pipeline and tests run; results obtained with it are not forecasts of
> real IPA prices. To reproduce the study, supply your own monthly price file (format below).

## Correction of earlier results

The pipeline up to commit `29318da` let information from the quarter being forecast into the model:

1. **Target in the features.** Rolling means, standard deviations, maxima, minima and percentage changes of the price were computed over windows ending at the quarter being forecast. Together with the one-quarter lag, the percentage change determines the target exactly. The forecast step computed the same features from earlier quarters only, so the model was also applied to different inputs than it was trained on.
2. **Current-quarter drivers.** Oil, gas, exchange-rate and equity features were averages over the quarter being forecast, which are not known when the forecast is made.
3. **Interpolated target.** The quarterly price was the mean of a weekly series interpolated (cubic) from the monthly prices of the whole history, so the last weeks of a quarter depended on the next month's price. Market-data outliers were also clipped at quantiles of the whole period, test quarters included.

Two further problems affected the comparison and the forecasts. SARIMA was scored on forecasts up to nine quarters ahead, while the tree model was scored one quarter ahead with the actual previous prices. SARIMA's 2025–2026 forecasts started from the end of the training split, about nine quarters before the last observed quarter.

The effect on the synthetic sample, over the same nine test quarters (2022Q4–2024Q4):

| | Tree model MAPE | Tree direction accuracy | SARIMA MAPE | Naive MAPE |
|---|---|---|---|---|
| Before (commit `29318da`) | 2.33 % | 100 % | 9.33 % (up to 9 quarters ahead) | 4.57 % (interpolated target) |
| After | 6.10 % | 33 % | 3.62 % (1 quarter ahead) | 4.98 % |

The earlier pipeline gives nearly the same numbers on the synthetic series (2.3 %, 100 %) as it reported for the real data (2.3 %, 100 %). It also showed the same sign of trouble: a test error far below the cross-validation error. The original monthly price file is no longer available, so the real-data results have not been recomputed.

## Method

| Stage | What it does |
|---|---|
| Target | Mean of the three monthly IPA prices of each quarter (incomplete quarters are dropped) |
| Exogenous data | Yahoo Finance: WTI `CL=F`, Brent `BZ=F`, natural gas `NG=F`, `USDTWD=X`, dollar index `DX-Y.NYB`, `^TWII`, `TSM`, as quarterly means of daily closes; event indicators for COVID-19, the Russia–Ukraine war, the Red Sea crisis and the US–China trade war (share of the quarter) |
| Features | Built from earlier quarters only: price lags 1–4; mean, standard deviation, maximum and minimum of the last 2, 4 and 8 quarters; price change over the last 1, 2 and 4 quarters; previous-quarter driver values, their 2- and 4-quarter means and products; previous-quarter event indicators; the calendar quarter. Training and forecasting call the same function (`build_quarterly_features`) |
| Models | Gradient-boosted trees (XGBoost, with scikit-learn `GradientBoosting` as automatic fallback), SARIMA and a naive forecast. The tree model and SARIMA are tuned by walk-forward cross-validation |
| Evaluation | The last 20 % of quarters are held out. Every model forecasts each test quarter one quarter ahead from the actual history, in cross-validation and on the test set. Direction accuracy counts the quarters where the forecast moves away from the last known price in the same direction as the actual price |
| Ensemble | Tree model and SARIMA, weighted by inverse squared CV-MAPE; a model much weaker than the best one is suppressed |
| Forecast | Both models are refitted on all quarters. The tree model forecasts recursively; drivers are projected with a damped drift. Confidence bands come from recent volatility and the test residuals |
| Outputs | Figures, HTML/CSV reports, feature importance, sensitivity analysis, walk-forward backtest table and a run manifest |

Background and design notes: [`docs/IPA_Price_Forecasting_Project_Proposal.md`](docs/IPA_Price_Forecasting_Project_Proposal.md)
and [`docs/Technical_Optimization_2026.md`](docs/Technical_Optimization_2026.md) (written before the correction above).

## Results on the synthetic sample

Quarters 2014Q1–2024Q4 with all features (44); the last nine are the test set, each forecast one quarter ahead. CV columns are the walk-forward cross-validation on the training quarters.

| Model | Test MAE (TWD/kg) | Test MAPE | Direction accuracy | CV MAPE |
|---|---|---|---|---|
| Gradient-boosted trees | 2.51 | 6.1 % | 33 % | 4.6 % |
| SARIMA (1,1,0)(1,1,0)₄ | 1.57 | 3.6 % | 67 % | 3.5 % |
| Naive (last quarter) | 2.09 | 5.0 % | — | 4.0 % |
| Ensemble | 1.83 | 4.3 % | 33 % | — |

The figures are from `reports/model_metrics.csv` written by the Quick start command below. They depend on the market data downloaded at run time. With nine test quarters, a difference of one percentage point between models is within the noise. The synthetic series is a slow cycle, an annual season and AR(1) noise, generated independently of the market data (`make_sample_prices.py`). The drivers therefore carry no information about it, and the tree model can learn only from the price's own history. SARIMA models exactly that structure, so its lead on this series says nothing about real IPA prices.

## Quick start

```bash
git clone https://github.com/Shuruyue/IPA-Price-Forecasting.git
cd IPA-Price-Forecasting/Code
pip install -r requirements.txt

# run on the synthetic sample (market data is downloaded from Yahoo Finance and cached in Data/)
python ipa_price_prediction.py --years 2025 2026 --price-file Data/ipa_prices.sample.json

# run on your own data: put it at Data/ipa_prices.json, or pass --price-file
python ipa_price_prediction.py --years 2025 2026

# force a fresh market-data download (e.g. after a failed or partial download)
python ipa_price_prediction.py --years 2025 2026 --refresh-cache
```

Figures go to `Code/figures/` and reports to `Code/reports/` (both git-ignored; change them with
`--figures-dir` / `--reports-dir`). `python ipa_price_prediction.py --help` lists all options.

### Price file format

```json
{
  "unit": "TWD/KG",
  "prices": [
    {"date": "2012-01", "price": 48.0},
    {"date": "2012-02", "price": 47.0}
  ]
}
```

One entry per month (`YYYY-MM`). `Data/ipa_prices.json` is git-ignored so a private series never
gets committed by accident. `python make_sample_prices.py` regenerates the synthetic sample
deterministically.

## Tests

```bash
cd Code
python -m pytest -q tests      # 79 tests, no network access needed
```

Besides unit tests, the suite guards against look-ahead. It checks the following:

- changing a quarter and everything after it leaves that quarter's features unchanged;
- a quarter's target does not move when the next month's price changes;
- a quarter to be forecast gets the same features as in training;
- SARIMA's one-step predictions ignore later quarters.

An offline end-to-end test runs the whole pipeline on the synthetic prices with synthetic market data. CI runs the tests on Ubuntu and Windows with Python 3.10 and 3.12.

## Limitations

- About 44 usable quarters, nine of them for testing: the error estimates are rough, and the model ranking can change with the data.
- Results here are for a synthetic series; the real-data study cannot be rerun without the original price file.
- Future driver values are a damped extrapolation of their recent trend, and the confidence bands do not include the uncertainty of that projection.
- Yahoo Finance data can be revised or become unavailable, which changes the numbers between runs.

## Repository layout

```text
IPA-Price-Forecasting/
├── Code/
│   ├── ipa_data_collector.py       # Yahoo Finance download, caching, event indicators
│   ├── ipa_feature_engineering.py  # price loading, quarterly target, past-only features
│   ├── ipa_models.py               # tree model, SARIMA, evaluation metrics
│   ├── ipa_price_prediction.py     # pipeline, walk-forward tuning, ensemble, forecasts, reports
│   ├── make_sample_prices.py       # writes the synthetic sample series
│   ├── Data/ipa_prices.sample.json # synthetic demo data (not real prices)
│   ├── requirements.txt
│   └── tests/
└── docs/                           # project proposal and optimization notes
```

## Origin

Developed in 2026 for an Engineering Economics course. The project first lived in a folder of a
course repository; it was moved here as a standalone project without the price data, so the
earlier commit history is not included.

## License

[MIT](LICENSE)
