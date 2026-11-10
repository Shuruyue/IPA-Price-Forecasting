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


class TestDirectionAccuracy:
    def test_perfect_direction(self):
        y_true = np.array([40.0, 42.0, 44.0, 46.0])
        y_pred = np.array([39.0, 41.0, 43.0, 45.0])
        result = IPAPricePredictor._direction_accuracy(y_true, y_pred)
        assert result == pytest.approx(100.0)

    def test_single_value(self):
        result = IPAPricePredictor._direction_accuracy(np.array([40.0]), np.array([41.0]))
        assert result == 0.0

    def test_relative_to_previous_actual(self):
        # one-quarter-ahead forecasts are judged against the last known price
        y_prev = np.array([40.0, 42.0, 41.0])
        y_true = np.array([42.0, 41.0, 43.0])   # up, down, up
        y_pred = np.array([41.0, 43.0, 42.0])   # up, up,   up
        result = IPAPricePredictor._direction_accuracy(y_true, y_pred, y_prev)
        assert result == pytest.approx(200 / 3)


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


# ---------------------------------------------------------------------------
# End to end, offline: synthetic sample prices and synthetic market data
# ---------------------------------------------------------------------------

@pytest.fixture
def offline_run(tmp_path, monkeypatch):
    from ipa_data_collector import IPADataCollector
    from ipa_feature_engineering import SAMPLE_IPA_JSON_PATH

    days = pd.date_range('2012-01-01', '2024-12-31', freq='D')
    rng = np.random.default_rng(3)
    market = pd.DataFrame(
        {col: 50 + np.cumsum(rng.normal(0, 0.5, len(days))) for col in IPADataCollector.MARKET_COLUMNS},
        index=days,
    )
    for col in IPADataCollector.EVENT_COLUMNS:
        market[col] = (days.year >= 2020).astype(int)
    monkeypatch.setattr(IPADataCollector, 'collect_all_data', lambda self: market.copy())

    p = IPAPricePredictor(
        target_year=2025, price_file=SAMPLE_IPA_JSON_PATH, cache_dir=str(tmp_path),
        figures_dir=str(tmp_path / 'fig'), reports_dir=str(tmp_path / 'rep'),
    )
    p.load_and_prepare_data()
    p.train_models(test_size=0.2)
    forecast = p.predict_future(quarters_ahead=4)
    return p, forecast


class TestEndToEndOffline:
    def test_features_are_past_only_columns(self, offline_run):
        p, _ = offline_run
        raw = set(p.quarterly_base.columns)
        assert not raw & set(p.feature_cols)  # no quarter-t price or driver is a feature

    def test_all_models_and_naive_baseline_reported(self, offline_run):
        p, _ = offline_run
        keys = {r['ModelKey'] for r in p.results}
        assert {'xgboost', 'sarima', 'naive', 'ensemble'} <= keys
        n_test = len(p.y_test)
        assert all(np.isfinite(r['MAPE']) for r in p.results)
        assert len(p.predictions['sarima_test']) == n_test  # one-step forecasts, one per quarter

    def test_forecast_models_use_all_quarters(self, offline_run):
        p, forecast = offline_run
        # SARIMA used to forecast from the end of the training split, ~9 quarters too early
        assert p.models['sarima'].fitted_model.nobs == len(p.y)
        assert list(forecast['Quarter']) == ['2025Q1', '2025Q2', '2025Q3', '2025Q4']
        assert np.isfinite(forecast[['Base', 'Optimistic', 'Pessimistic']].to_numpy()).all()

    def test_first_forecast_uses_observed_data_only(self, offline_run):
        p, forecast = offline_run
        # The first forecast quarter needs no projected value: its features come from 2024 and
        # earlier, built by the same function as the training rows.
        table = p.quarterly_base.copy()
        q1 = pd.Timestamp('2025-03-31')
        table.loc[q1] = np.nan
        features = p.feature_engineer.build_quarterly_features(table).loc[q1, p.feature_cols]
        assert not features.isna().any()
        expected = float(p.models['xgboost'].predict(features.to_numpy(dtype=float).reshape(1, -1))[0])
        assert forecast['XGBoost'].iloc[0] == pytest.approx(expected)
