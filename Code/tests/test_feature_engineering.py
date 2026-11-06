"""
Unit tests for IPA Feature Engineering Module
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the Code directory is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ipa_feature_engineering import SAMPLE_IPA_JSON_PATH, IPAFeatureEngineer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fe():
    return IPAFeatureEngineer()


@pytest.fixture
def sample_weekly_df():
    """Minimal weekly IPA price DataFrame for testing."""
    dates = pd.date_range('2020-01-05', periods=20, freq='W-SUN')
    prices = np.linspace(40, 50, 20) + np.random.default_rng(42).normal(0, 0.5, 20)
    df = pd.DataFrame({'IPA_Price_TWD': prices}, index=dates)
    df.index.name = 'Date'
    return df


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
# JSON loading
# ---------------------------------------------------------------------------

class TestIPAPriceDataLoading:
    def test_load_from_json(self, fe, tmp_path):
        json_data = {
            "prices": [
                {"date": "2020-01", "price": 40},
                {"date": "2020-02", "price": 42},
                {"date": "2020-03", "price": 44},
            ]
        }
        json_file = tmp_path / "test_prices.json"
        json_file.write_text(json.dumps(json_data), encoding='utf-8')

        points = fe._load_price_points_from_json(str(json_file))
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
        points = fe._load_price_points_from_json(SAMPLE_IPA_JSON_PATH)
        # 13 years × 12 months = 156 entries
        assert len(points) == 156

    def test_missing_price_file_raises(self, fe, tmp_path):
        with pytest.raises(FileNotFoundError, match="not included in this repository"):
            fe.create_ipa_price_data(json_path=str(tmp_path / "missing.json"))

    def test_create_ipa_price_data_returns_weekly(self, fe):
        data = fe.create_ipa_price_data(interpolation='linear', json_path=SAMPLE_IPA_JSON_PATH)
        assert isinstance(data.index, pd.DatetimeIndex)
        assert 'IPA_Price_TWD' in data.columns
        assert not data['IPA_Price_TWD'].isna().any()
        # Weekly data should be significantly more than 156 monthly points
        assert len(data) > 156

    def test_create_ipa_price_data_with_custom_json(self, fe, tmp_path):
        json_data = {
            "prices": [
                {"date": "2020-01", "price": 40},
                {"date": "2020-02", "price": 42},
                {"date": "2020-03", "price": 44},
                {"date": "2020-04", "price": 46},
            ]
        }
        json_file = tmp_path / "custom.json"
        json_file.write_text(json.dumps(json_data), encoding='utf-8')

        data = fe.create_ipa_price_data(interpolation='linear', json_path=str(json_file))
        assert len(data) > 0
        assert data['IPA_Price_TWD'].min() >= 39  # close to 40


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

class TestLagFeatures:
    def test_lag_columns_created(self, fe, sample_weekly_df):
        result = fe.create_lag_features(sample_weekly_df, 'IPA_Price_TWD', lags=[1, 2])
        assert 'IPA_Price_TWD_lag1' in result.columns
        assert 'IPA_Price_TWD_lag2' in result.columns

    def test_lag_values_correct(self, fe, sample_weekly_df):
        result = fe.create_lag_features(sample_weekly_df, 'IPA_Price_TWD', lags=[1])
        # lag1 at row 1 should equal original value at row 0
        assert result['IPA_Price_TWD_lag1'].iloc[1] == pytest.approx(
            sample_weekly_df['IPA_Price_TWD'].iloc[0]
        )

    def test_lag_first_value_is_nan(self, fe, sample_weekly_df):
        result = fe.create_lag_features(sample_weekly_df, 'IPA_Price_TWD', lags=[1])
        assert np.isnan(result['IPA_Price_TWD_lag1'].iloc[0])


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

class TestRollingFeatures:
    def test_rolling_columns_created(self, fe, sample_weekly_df):
        result = fe.create_rolling_features(sample_weekly_df, 'IPA_Price_TWD', windows=[4])
        for suffix in ['ma4', 'std4', 'max4', 'min4']:
            assert f'IPA_Price_TWD_{suffix}' in result.columns

    def test_rolling_mean_correct(self, fe, sample_weekly_df):
        result = fe.create_rolling_features(sample_weekly_df, 'IPA_Price_TWD', windows=[4])
        # rolling mean at index 3 should be mean of first 4 values
        expected = sample_weekly_df['IPA_Price_TWD'].iloc[:4].mean()
        assert result['IPA_Price_TWD_ma4'].iloc[3] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Change features
# ---------------------------------------------------------------------------

class TestChangeFeatures:
    def test_change_columns_created(self, fe, sample_weekly_df):
        result = fe.create_change_features(sample_weekly_df, 'IPA_Price_TWD')
        assert 'IPA_Price_TWD_pct_change' in result.columns
        assert 'IPA_Price_TWD_monthly_change' in result.columns
        assert 'IPA_Price_TWD_quarterly_change' in result.columns


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

class TestTimeFeatures:
    def test_time_columns_created(self, fe, sample_weekly_df):
        result = fe.create_time_features(sample_weekly_df)
        for col in ['Year', 'Month', 'Quarter', 'Week', 'Month_sin', 'Month_cos',
                     'Quarter_sin', 'Quarter_cos']:
            assert col in result.columns

    def test_sin_cos_range(self, fe, sample_weekly_df):
        result = fe.create_time_features(sample_weekly_df)
        assert result['Month_sin'].between(-1, 1).all()
        assert result['Quarter_cos'].between(-1, 1).all()


# ---------------------------------------------------------------------------
# Exogenous features
# ---------------------------------------------------------------------------

class TestExogenousFeatures:
    def test_exogenous_lag_and_rolling_created(self, fe):
        dates = pd.date_range('2020-01-05', periods=20, freq='W-SUN')
        df = pd.DataFrame({
            'IPA_Price_TWD': np.linspace(40, 50, 20),
            'WTI_Price': np.linspace(60, 80, 20),
            'USD_TWD': np.linspace(29, 31, 20),
        }, index=dates)
        result = fe.create_exogenous_features(df, windows=[4])
        assert 'WTI_Price_lag1' in result.columns
        assert 'WTI_Price_ma4' in result.columns
        assert 'WTI_Price_x_USD_TWD' in result.columns

    def test_missing_exogenous_cols_skipped(self, fe, sample_weekly_df):
        result = fe.create_exogenous_features(sample_weekly_df, windows=[4])
        # Should not fail even though no exogenous columns exist
        assert len(result) == len(sample_weekly_df)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestPrepareFeaturesFullPipeline:
    def test_prepare_features_runs(self, fe, sample_weekly_df):
        result = fe.prepare_features(sample_weekly_df, 'IPA_Price_TWD')
        # After dropna, should have fewer rows than input
        assert len(result) < len(sample_weekly_df)
        assert 'IPA_Price_TWD' in result.columns

    def test_resample_to_quarterly(self, fe, sample_weekly_df):
        featured = fe.prepare_features(sample_weekly_df, 'IPA_Price_TWD')
        quarterly = fe.resample_to_quarterly(featured)
        assert 'Quarter_Label' in quarterly.columns
        assert len(quarterly) <= len(featured)

    def test_prepare_quarterly_features(self, fe):
        dates = pd.date_range('2014-03-31', periods=44, freq='QE')
        df = pd.DataFrame(
            {
                'IPA_Price_TWD': np.linspace(35, 55, len(dates)),
                'WTI_Price': np.linspace(60, 90, len(dates)),
                'USD_TWD': np.linspace(29, 33, len(dates)),
            },
            index=dates,
        )
        featured = fe.prepare_quarterly_features(df, 'IPA_Price_TWD')
        assert len(featured) > 0
        assert 'IPA_Price_TWD_lag1' in featured.columns
        assert 'WTI_Price_lag1' in featured.columns

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
