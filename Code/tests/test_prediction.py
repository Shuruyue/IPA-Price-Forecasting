"""
Unit tests for IPA Price Prediction main module (IPAPricePredictor utilities)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ipa_price_prediction import (
    ENSEMBLE_ELIGIBILITY_RATIO,
    ENSEMBLE_WEIGHT_POWER,
    IPAPricePredictor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def predictor():
    """Create a predictor without running data loading."""
    return IPAPricePredictor(target_year=2025, random_seed=42)


# ---------------------------------------------------------------------------
# Static / utility methods
# ---------------------------------------------------------------------------

class TestSafeMAPE:
    def test_normal_case(self):
        y_true = np.array([40.0, 42.0, 44.0])
        y_pred = np.array([41.0, 43.0, 43.0])
        result = IPAPricePredictor._safe_mape(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0

    def test_zero_denominator_protection(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        # Should not raise - uses 1.0 when denominator is near zero
        result = IPAPricePredictor._safe_mape(y_true, y_pred)
        assert np.isfinite(result)

    def test_perfect_prediction(self):
        y = np.array([40.0, 42.0])
        result = IPAPricePredictor._safe_mape(y, y)
        assert result == pytest.approx(0.0)


class TestIsTargetDerivedFeature:
    def test_lag_feature(self):
        assert IPAPricePredictor._is_target_derived_feature('IPA_Price_TWD_lag1')

    def test_ma_feature(self):
        assert IPAPricePredictor._is_target_derived_feature('IPA_Price_TWD_ma4')

    def test_non_target_feature(self):
        assert not IPAPricePredictor._is_target_derived_feature('WTI_Price')
        assert not IPAPricePredictor._is_target_derived_feature('Year')


class TestSafePctChange:
    def test_normal(self):
        values = np.array([40.0, 42.0, 44.0])
        result = IPAPricePredictor._safe_pct_change(values, periods=1)
        # (44 - 42) / 42
        assert result == pytest.approx((44.0 - 42.0) / 42.0)

    def test_too_short(self):
        values = np.array([40.0])
        result = IPAPricePredictor._safe_pct_change(values, periods=1)
        assert result == 0.0

    def test_zero_base(self):
        values = np.array([0.0, 1.0])
        result = IPAPricePredictor._safe_pct_change(values, periods=1)
        assert result == 0.0


class TestDirectionAccuracy:
    def test_perfect_direction(self):
        y_true = np.array([40.0, 42.0, 44.0, 46.0])
        y_pred = np.array([39.0, 41.0, 43.0, 45.0])
        result = IPAPricePredictor._direction_accuracy(y_true, y_pred)
        assert result == pytest.approx(100.0)

    def test_single_value(self):
        result = IPAPricePredictor._direction_accuracy(np.array([40.0]), np.array([41.0]))
        assert result == 0.0


# ---------------------------------------------------------------------------
# Clip outliers
# ---------------------------------------------------------------------------

class TestClipOutliers:
    def test_basic_clipping(self, predictor):
        df = pd.DataFrame({
            'a': [1, 2, 3, 100, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        result = predictor._clip_outliers(df, quantile_low=0.1, quantile_high=0.9)
        assert result['a'].max() <= df['a'].quantile(0.9)

    def test_exclude_cols(self, predictor):
        df = pd.DataFrame({
            'IPA_Price_TWD': [1, 2, 3, 100, 5],
            'other': [1, 2, 3, 100, 5],
        })
        result = predictor._clip_outliers(df, exclude_cols=['IPA_Price_TWD'])
        # Excluded column should remain unchanged
        assert result['IPA_Price_TWD'].max() == 100
        # Other column should be clipped
        assert result['other'].max() <= df['other'].quantile(0.99)


# ---------------------------------------------------------------------------
# Ensemble weight derivation
# ---------------------------------------------------------------------------

class TestDeriveEnsembleWeights:
    def test_basic_weights(self, predictor):
        predictor.results = [
            {'ModelKey': 'xgboost', 'MAPE': 2.0},
            {'ModelKey': 'sarima', 'MAPE': 8.0},
        ]
        predictor._derive_ensemble_weights()
        assert 'xgboost' in predictor.ensemble_weights
        # If both survive filter, xgboost should have larger weight.
        if 'sarima' in predictor.ensemble_weights:
            assert predictor.ensemble_weights['xgboost'] > predictor.ensemble_weights['sarima']
        # Weights should sum to 1
        total = sum(predictor.ensemble_weights.values())
        assert total == pytest.approx(1.0)

    def test_no_valid_results(self, predictor):
        predictor.results = [
            {'ModelKey': 'unknown', 'MAPE': 5.0},
        ]
        predictor._derive_ensemble_weights()
        assert predictor.ensemble_weights == {}

    def test_nan_mape_excluded(self, predictor):
        predictor.results = [
            {'ModelKey': 'xgboost', 'MAPE': 2.0},
            {'ModelKey': 'sarima', 'MAPE': float('nan')},
        ]
        predictor._derive_ensemble_weights()
        assert 'sarima' not in predictor.ensemble_weights
        assert predictor.ensemble_weights['xgboost'] == pytest.approx(1.0)

    def test_cv_mape_priority(self, predictor):
        predictor.results = [
            {'ModelKey': 'xgboost', 'MAPE': 1.0, 'CV_MAPE': 9.0},
            {'ModelKey': 'sarima', 'MAPE': 8.0, 'CV_MAPE': 4.0},
        ]
        predictor._derive_ensemble_weights()
        assert predictor.ensemble_weights['sarima'] > predictor.ensemble_weights['xgboost']

    def test_eligibility_filtering(self, predictor):
        """Model with error > ELIGIBILITY_RATIO * best should be excluded."""
        predictor.results = [
            {'ModelKey': 'xgboost', 'MAPE': 1.0},
            {'ModelKey': 'sarima', 'MAPE': 1.0 * ENSEMBLE_ELIGIBILITY_RATIO + 1.0},
        ]
        predictor._derive_ensemble_weights()
        assert 'sarima' not in predictor.ensemble_weights


class TestGetModelBlendWeights:
    def test_missing_model_weight_is_zero(self, predictor):
        predictor.ensemble_weights = {'xgboost': 1.0}
        weights = predictor._get_model_blend_weights(['xgboost', 'sarima'])
        assert weights['xgboost'] == pytest.approx(1.0)
        assert weights['sarima'] == pytest.approx(0.0)

    def test_uniform_when_empty(self, predictor):
        predictor.ensemble_weights = {}
        weights = predictor._get_model_blend_weights(['xgboost', 'sarima'])
        assert weights['xgboost'] == pytest.approx(0.5)
        assert weights['sarima'] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Build future feature row
# ---------------------------------------------------------------------------

class TestBuildFutureFeatureRow:
    def test_lag_features_populated(self, predictor):
        feature_cols = ['IPA_Price_TWD_lag1', 'IPA_Price_TWD_lag2', 'Year', 'Quarter']
        predictor.feature_cols = feature_cols
        predictor.exogenous_feature_cols = []

        template = pd.Series(
            [0.0, 0.0, 2024, 4],
            index=feature_cols,
            dtype=float,
        )
        target_history = [40.0, 42.0, 44.0, 46.0]
        future_period = pd.Period('2025Q1', freq='Q')

        row = predictor._build_future_feature_row(future_period, target_history, template)

        assert row['IPA_Price_TWD_lag1'] == 46.0  # last value
        assert row['IPA_Price_TWD_lag2'] == 44.0  # second last
        assert row['Year'] == 2025
        assert row['Quarter'] == 1

    def test_exogenous_derived_recomputed(self, predictor):
        predictor.feature_cols = [
            'IPA_Price_TWD_lag1',
            'WTI_Price',
            'WTI_Price_lag1',
            'WTI_Price_ma2',
            'USD_TWD',
            'WTI_Price_x_USD_TWD',
            'Year',
            'Quarter',
        ]
        predictor.base_exogenous_feature_cols = ['WTI_Price', 'USD_TWD']

        template = pd.Series([0.0] * len(predictor.feature_cols), index=predictor.feature_cols, dtype=float)
        target_history = [40.0, 42.0, 44.0, 46.0]
        future_period = pd.Period('2025Q1', freq='Q')
        exog_history = pd.DataFrame(
            {
                'WTI_Price': [70.0, 72.0, 75.0],
                'USD_TWD': [31.0, 31.2, 31.5],
            },
            index=pd.date_range('2024-06-30', periods=3, freq='QE'),
        )
        exog_values = pd.Series({'WTI_Price': 75.0, 'USD_TWD': 31.5})

        row = predictor._build_future_feature_row(
            future_period,
            target_history,
            template,
            exogenous_values=exog_values,
            exogenous_history=exog_history,
        )
        assert row['WTI_Price'] == pytest.approx(75.0)
        assert row['WTI_Price_lag1'] == pytest.approx(72.0)
        assert row['WTI_Price_ma2'] == pytest.approx((72.0 + 75.0) / 2)
        assert row['WTI_Price_x_USD_TWD'] == pytest.approx(75.0 * 31.5)

    def test_rolling_stats_computed(self, predictor):
        """Test that rolling std/max/min features are correctly computed."""
        predictor.feature_cols = [
            'IPA_Price_TWD_lag1', 'IPA_Price_TWD_ma2',
            'IPA_Price_TWD_std2', 'IPA_Price_TWD_max2', 'IPA_Price_TWD_min2',
            'Year', 'Quarter',
        ]
        predictor.exogenous_feature_cols = []
        predictor.base_exogenous_feature_cols = []

        template = pd.Series([0.0] * len(predictor.feature_cols), index=predictor.feature_cols, dtype=float)
        target_history = [40.0, 42.0, 44.0, 46.0]
        future_period = pd.Period('2025Q1', freq='Q')

        row = predictor._build_future_feature_row(future_period, target_history, template)
        assert row['IPA_Price_TWD_ma2'] == pytest.approx(np.mean([44.0, 46.0]))
        assert row['IPA_Price_TWD_std2'] == pytest.approx(np.std([44.0, 46.0], ddof=0))
        assert row['IPA_Price_TWD_max2'] == pytest.approx(46.0)
        assert row['IPA_Price_TWD_min2'] == pytest.approx(44.0)


# ---------------------------------------------------------------------------
# JSON serialization utility
# ---------------------------------------------------------------------------

class TestJsonCompatible:
    def test_numpy_int(self):
        assert IPAPricePredictor._json_compatible(np.int64(42)) == 42

    def test_numpy_float(self):
        result = IPAPricePredictor._json_compatible(np.float64(3.14))
        assert isinstance(result, float)

    def test_numpy_array(self):
        result = IPAPricePredictor._json_compatible(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_timestamp(self):
        ts = pd.Timestamp('2025-01-01')
        result = IPAPricePredictor._json_compatible(ts)
        assert isinstance(result, str)

    def test_passthrough_string(self):
        assert IPAPricePredictor._json_compatible("hello") == "hello"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_eligibility_ratio_positive(self):
        assert ENSEMBLE_ELIGIBILITY_RATIO > 1.0

    def test_weight_power_positive(self):
        assert ENSEMBLE_WEIGHT_POWER > 0
