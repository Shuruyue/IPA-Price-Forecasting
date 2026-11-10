"""
Unit tests for IPA Feature Engineering Module

Besides the basic behaviour, these tests guard against look-ahead: the target of a quarter
must depend only on that quarter's monthly prices, and every feature of a quarter only on
earlier quarters.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the Code directory is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ipa_feature_engineering import SAMPLE_IPA_JSON_PATH, TARGET_COL, IPAFeatureEngineer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fe():
    return IPAFeatureEngineer()


def _write_prices(tmp_path, prices, name="prices.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"prices": [{"date": d, "price": p} for d, p in prices]}), encoding='utf-8')
    return str(path)


@pytest.fixture
def quarterly_base():
    """Quarterly table: price, two market drivers and an event flag (random, reproducible)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range('2012-03-31', periods=40, freq='QE')
    return pd.DataFrame(
        {
            TARGET_COL: 40 + np.cumsum(rng.normal(0, 1.5, len(dates))),
            'WTI_Price': 70 + np.cumsum(rng.normal(0, 3, len(dates))),
            'USD_TWD': 30 + np.cumsum(rng.normal(0, 0.3, len(dates))),
            'COVID19': (np.arange(len(dates)) >= 32).astype(float),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

class TestValidateTargetSeries:
    def test_empty_df_raises(self, fe):
        with pytest.raises(ValueError, match="empty"):
            fe._validate_target_series(pd.DataFrame(), 'IPA_Price_TWD')

    def test_missing_column_raises(self, fe):
        df = pd.DataFrame({'other': [1]}, index=pd.to_datetime(['2020-01-01']))
        with pytest.raises(ValueError, match="Missing target column"):
            fe._validate_target_series(df, 'IPA_Price_TWD')

    def test_non_datetime_index_raises(self, fe):
        df = pd.DataFrame({'IPA_Price_TWD': [1]}, index=[0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            fe._validate_target_series(df, 'IPA_Price_TWD')

    def test_all_nan_raises(self, fe):
        df = pd.DataFrame(
            {'IPA_Price_TWD': [np.nan, np.nan]},
            index=pd.to_datetime(['2020-01-01', '2020-02-01']),
        )
        with pytest.raises(ValueError, match="only NaN"):
            fe._validate_target_series(df, 'IPA_Price_TWD')

    def test_unsorted_gets_sorted(self, fe):
        dates = pd.to_datetime(['2020-03-01', '2020-01-01', '2020-02-01'])
        df = pd.DataFrame({'IPA_Price_TWD': [3, 1, 2]}, index=dates)
        result = fe._validate_target_series(df, 'IPA_Price_TWD')
        assert result.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Price loading and the quarterly target
# ---------------------------------------------------------------------------

class TestPriceLoading:
    def test_load_from_json(self, fe, tmp_path):
        path = _write_prices(tmp_path, [("2020-01", 40), ("2020-02", 42), ("2020-03", 44)])
        points = fe._load_price_points_from_json(path)
        assert len(points) == 3
        assert points[0] == ('2020-01', 40)

    def test_load_from_json_empty_raises(self, fe, tmp_path):
        json_file = tmp_path / "empty.json"
        json_file.write_text('{"prices": []}', encoding='utf-8')
        with pytest.raises(ValueError, match="No price entries"):
            fe._load_price_points_from_json(str(json_file))

    def test_sample_file_is_synthetic_and_complete(self, fe):
        with open(SAMPLE_IPA_JSON_PATH, encoding='utf-8') as f:
            assert json.load(f)['synthetic'] is True
        monthly = fe.load_monthly_prices(SAMPLE_IPA_JSON_PATH)
        assert len(monthly) == 156  # 13 years x 12 months
        assert len(fe.quarterly_target(monthly)) == 52

    def test_missing_price_file_raises(self, fe, tmp_path):
        with pytest.raises(FileNotFoundError, match="not included in this repository"):
            fe.load_monthly_prices(str(tmp_path / "missing.json"))

    def test_duplicate_month_raises(self, fe, tmp_path):
        path = _write_prices(tmp_path, [("2020-01", 40), ("2020-01", 41)])
        with pytest.raises(ValueError, match="Duplicate months"):
            fe.load_monthly_prices(path)

    def test_quarterly_target_is_mean_of_three_months(self, fe, tmp_path):
        path = _write_prices(tmp_path, [("2020-01", 40), ("2020-02", 42), ("2020-03", 47), ("2020-04", 50)])
        target = fe.quarterly_target(fe.load_monthly_prices(path))
        assert list(target.index.to_period('Q').astype(str)) == ['2020Q1']  # Q2 incomplete: dropped
        assert target.iloc[0] == pytest.approx(43.0)

    def test_quarterly_target_ignores_the_next_quarter(self, fe, tmp_path):
        # Cubic interpolation to a weekly grid made Q1 depend on the April price.
        months = [(f"2020-{m:02d}", 40.0 + m) for m in range(1, 13)]
        bumped = [(d, p + 10 if d == "2020-04" else p) for d, p in months]
        base = fe.quarterly_target(fe.load_monthly_prices(_write_prices(tmp_path, months, "a.json")))
        changed = fe.quarterly_target(fe.load_monthly_prices(_write_prices(tmp_path, bumped, "b.json")))
        assert changed.iloc[0] == base.iloc[0]                       # Q1 unchanged
        assert changed.iloc[1] == pytest.approx(base.iloc[1] + 10 / 3)  # Q2 moves by 10/3


# ---------------------------------------------------------------------------
# Features: only earlier quarters
# ---------------------------------------------------------------------------

class TestQuarterlyFeatures:
    def test_every_feature_uses_only_earlier_quarters(self, fe, quarterly_base):
        """Changing quarter t and everything after it must leave the features of t unchanged."""
        reference = fe.build_quarterly_features(quarterly_base)
        rng = np.random.default_rng(0)
        for t in range(1, len(quarterly_base)):
            perturbed = quarterly_base.copy()
            perturbed.iloc[t:] = perturbed.iloc[t:] * (1 + rng.uniform(0.2, 0.5, perturbed.iloc[t:].shape))
            features = fe.build_quarterly_features(perturbed)
            pd.testing.assert_series_equal(features.iloc[t], reference.iloc[t], check_names=False)

    def test_no_contemporaneous_columns(self, fe, quarterly_base):
        features = fe.build_quarterly_features(quarterly_base)
        for col in quarterly_base.columns:
            assert col not in features.columns

    def test_definitions(self, fe, quarterly_base):
        f = fe.build_quarterly_features(quarterly_base)
        y = quarterly_base[TARGET_COL]
        t = 20
        assert f[f'{TARGET_COL}_lag1'].iloc[t] == y.iloc[t - 1]
        assert f[f'{TARGET_COL}_ma4'].iloc[t] == pytest.approx(y.iloc[t - 4:t].mean())
        assert f[f'{TARGET_COL}_std4'].iloc[t] == pytest.approx(y.iloc[t - 4:t].std())
        assert f[f'{TARGET_COL}_chg1'].iloc[t] == pytest.approx(y.iloc[t - 1] / y.iloc[t - 2] - 1)
        assert f[f'{TARGET_COL}_chg4'].iloc[t] == pytest.approx(y.iloc[t - 1] / y.iloc[t - 5] - 1)
        wti = quarterly_base['WTI_Price']
        assert f['WTI_Price_lag1'].iloc[t] == wti.iloc[t - 1]
        assert f['WTI_Price_ma2'].iloc[t] == pytest.approx(wti.iloc[t - 2:t].mean())
        assert f['COVID19_lag1'].iloc[t] == quarterly_base['COVID19'].iloc[t - 1]
        assert f['WTI_Price_x_USD_TWD_lag1'].iloc[t] == pytest.approx(
            wti.iloc[t - 1] * quarterly_base['USD_TWD'].iloc[t - 1]
        )

    def test_forecast_row_matches_training_row(self, fe, quarterly_base):
        """A quarter whose price is not yet known gets the same features as in training."""
        training = fe.build_quarterly_features(quarterly_base)
        for t in (12, 25, len(quarterly_base) - 1):
            unknown = quarterly_base.iloc[:t + 1].copy()
            unknown.iloc[t] = np.nan  # price and drivers of the quarter being forecast
            forecast = fe.build_quarterly_features(unknown)
            pd.testing.assert_series_equal(forecast.iloc[t], training.iloc[t], check_names=False)

    def test_prepare_quarterly_features(self, fe, quarterly_base):
        table = fe.prepare_quarterly_features(quarterly_base, TARGET_COL)
        assert TARGET_COL in table.columns
        assert set(fe.feature_columns) | {TARGET_COL} == set(table.columns)
        assert not table.isna().any().any()
        assert table.index[0] == quarterly_base.index[8]  # 8-quarter window needs 8 earlier quarters

    def test_empty_exogenous_column_names_the_cause(self, fe):
        # A failed download leaves an all-NaN column; dropna() used to wipe every row.
        dates = pd.date_range('2014-03-31', periods=44, freq='QE')
        df = pd.DataFrame(
            {
                'IPA_Price_TWD': np.linspace(35, 55, len(dates)),
                'WTI_Price': np.nan,
                'USD_TWD': np.linspace(29, 33, len(dates)),
            },
            index=dates,
        )
        with pytest.raises(ValueError, match=r"WTI_Price.*download most likely failed"):
            fe.prepare_quarterly_features(df, 'IPA_Price_TWD')

    def test_resample_to_quarterly_means(self, fe):
        days = pd.date_range('2020-01-01', '2020-06-30', freq='D')
        df = pd.DataFrame({'WTI_Price': np.where(days.quarter == 1, 50.0, 70.0)}, index=days)
        quarterly = fe.resample_to_quarterly(df)
        assert list(quarterly['WTI_Price']) == [50.0, 70.0]


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class TestScaling:
    def test_scale_and_inverse(self, fe):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = pd.Series([10.0, 20.0, 30.0])
        X_scaled, y_scaled = fe.scale_features(X, y, fit=True)
        assert X_scaled.shape == X.shape
        assert y_scaled is not None
        y_inv = fe.inverse_scale_target(y_scaled)
        np.testing.assert_allclose(y_inv, y.values, atol=1e-6)

    def test_scale_without_target(self, fe):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_scaled, y_out = fe.scale_features(X, None, fit=True)
        assert X_scaled.shape == X.shape
        assert y_out is None
