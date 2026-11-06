# IPA Price Forecasting

[![CI](https://github.com/Shuruyue/IPA-Price-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Shuruyue/IPA-Price-Forecasting/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Quarterly price forecasting for isopropyl alcohol (IPA, TWD/kg), built as the final project of an
Engineering Economics course. A tree-based model and SARIMA are tuned by walk-forward
validation on 2012–2024 data, combined into an ensemble, and used to forecast 2025 and 2026
quarter by quarter, with energy prices, exchange rates, equity proxies and geopolitical event
indicators as exogenous drivers.

> **Data notice.** The IPA price series used in the course project was digitized from a price
> chart whose original source is not recorded, so it is **not distributed** here. The repository
> ships a clearly labelled **synthetic** series (`Code/Data/ipa_prices.sample.json`) with the same
> format and period, so the pipeline and tests run; results obtained with it are not forecasts of
> real IPA prices. To reproduce the study, supply your own monthly price file (format below).

## Method

| Stage | What it does |
|---|---|
| Target series | Monthly IPA prices → weekly interpolation → quarterly averages |
| Exogenous data | Yahoo Finance: WTI `CL=F`, Brent `BZ=F`, natural gas `NG=F`, `USDTWD=X`, dollar index `DX-Y.NYB`, `^TWII`, `TSM`; event indicators for COVID-19, the Russia–Ukraine war, the Red Sea crisis and the US–China trade war |
| Features | Lags, rolling statistics, percentage changes, seasonal encodings and lagged exogenous features, built at quarterly frequency (no averaging of weekly lag features) |
| Models | Gradient-boosted trees (XGBoost, with scikit-learn `GradientBoosting` as automatic fallback) and SARIMA, each tuned by walk-forward cross-validation |
| Ensemble | Inverse-squared CV-MAPE weights; a model much weaker than the best one is suppressed |
| Forecast | Recursive quarter-by-quarter prediction; exogenous drivers projected with a damped drift; confidence bands from recent volatility and the residual distribution |
| Outputs | Figures, HTML/CSV reports, feature importance, sensitivity analysis, walk-forward backtest table and a run manifest |

Background and design notes: [`docs/IPA_Price_Forecasting_Project_Proposal.md`](docs/IPA_Price_Forecasting_Project_Proposal.md)
and [`docs/Technical_Optimization_2026.md`](docs/Technical_Optimization_2026.md).

## Results on the original (private) dataset

Quarterly data 2012–2024, last 20 % of quarters held out as the test set; CV columns are the
walk-forward cross-validation used for tuning.

| Model | Test MAE (TWD/kg) | Test MAPE | Direction accuracy | CV MAPE |
|---|---|---|---|---|
| Gradient-boosted trees | 1.03 | 2.3 % | 100 % | 4.7 % |
| SARIMA (0,1,1)(0,1,1)₄ | 4.22 | 9.1 % | 75 % | 15.1 % |
| Ensemble | 1.03 | 2.3 % | 100 % | — |

SARIMA's CV error was far above the tree model's, so the weak-model rule gave it zero weight and
the ensemble equals the tree model. The held-out test set covers only about nine quarters, so
these figures are indicative rather than a precise accuracy estimate.

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
python -m pytest -q tests      # 85 tests, no network access needed
```

CI runs the tests on Ubuntu and Windows with Python 3.10 and 3.12.

## Repository layout

```text
IPA-Price-Forecasting/
├── Code/
│   ├── ipa_data_collector.py       # Yahoo Finance download, caching, event indicators
│   ├── ipa_feature_engineering.py  # price loading, resampling, lag/rolling/seasonal features
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
