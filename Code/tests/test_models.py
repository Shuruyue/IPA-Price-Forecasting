"""
Unit tests for IPA Models Module
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ipa_models import ModelEvaluator, ARIMAModel, XGBoostModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Synthetic regression data for model tests."""
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 4))
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.normal(0, 0.2, n) + 40
    split = 60
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------

class TestModelEvaluator:
    def test_evaluate_returns_dict(self):
        y_true = np.array([40, 42, 44, 46])
        y_pred = np.array([41, 43, 43, 45])
        result = ModelEvaluator.evaluate(y_true, y_pred, "TestModel")
        assert isinstance(result, dict)
        for key in ['Model', 'MAE', 'RMSE', 'MAPE', 'Direction_Accuracy']:
            assert key in result

    def test_perfect_prediction(self):
        y = np.array([40, 42, 44])
        result = ModelEvaluator.evaluate(y, y, "Perfect")
        assert result['MAE'] == pytest.approx(0.0)
        assert result['RMSE'] == pytest.approx(0.0)
        assert result['MAPE'] == pytest.approx(0.0)

    def test_direction_accuracy_perfect(self):
        y_true = np.array([40, 42, 44, 46])  # all up
        y_pred = np.array([39, 41, 43, 45])  # all up
        result = ModelEvaluator.evaluate(y_true, y_pred, "Dir")
        assert result['Direction_Accuracy'] == pytest.approx(100.0)

    def test_direction_accuracy_none(self):
        y_true = np.array([40, 42, 44])  # up, up
        y_pred = np.array([40, 38, 36])  # down, down
        result = ModelEvaluator.evaluate(y_true, y_pred, "Dir")
        assert result['Direction_Accuracy'] == pytest.approx(0.0)

    def test_single_value_direction(self):
        result = ModelEvaluator.evaluate(np.array([40]), np.array([40]), "Single")
        assert result['Direction_Accuracy'] == 0  # not enough data


# ---------------------------------------------------------------------------
# XGBoostModel (or GradientBoosting fallback)
# ---------------------------------------------------------------------------

class TestXGBoostModel:
    def test_fit_predict(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = XGBoostModel(n_estimators=30, max_depth=3, learning_rate=0.1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_display_name_set(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = XGBoostModel(n_estimators=10)
        model.fit(X_train, y_train)
        name = model.get_display_name()
        assert name in ('XGBoost', 'GradientBoosting')

    def test_feature_importance(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = XGBoostModel(n_estimators=30)
        model.fit(X_train, y_train)
        fi = model.get_feature_importance([f'f{i}' for i in range(4)])
        assert len(fi) == 4
        assert 'importance' in fi.columns
        assert fi['importance'].sum() == pytest.approx(1.0, abs=0.01)

    def test_reasonable_mape(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = XGBoostModel(n_estimators=50, max_depth=4)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        result = ModelEvaluator.evaluate(y_test, preds, "XGB")
        # MAPE should be reasonable (< 20%) for this easy synthetic data
        assert result['MAPE'] < 20.0

    def test_predict_before_fit_raises(self):
        model = XGBoostModel()
        with pytest.raises(Exception):
            model.predict(np.array([[1, 2, 3]]))

    def test_backend_name_set(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = XGBoostModel(n_estimators=10)
        model.fit(X_train, y_train)
        assert model.backend_name in ('xgboost', 'gradient_boosting')


# ---------------------------------------------------------------------------
# ARIMAModel
# ---------------------------------------------------------------------------

class TestARIMAModel:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.normal(0.1, 0.5, 40)) + 40
        model = ARIMAModel(order=(1, 1, 0), seasonal_order=(0, 0, 0, 4))
        model.fit(y)
        predictions = model.predict(4)
        assert len(predictions) == 4
        # Predictions should be in a reasonable range
        assert all(p > 0 for p in predictions)

    def test_fitted_values(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.normal(0.1, 0.5, 30)) + 40
        model = ARIMAModel(order=(1, 1, 0), seasonal_order=(0, 0, 0, 4))
        model.fit(y)
        fitted = model.get_fitted_values()
        assert len(fitted) == len(y)

    def test_predict_before_fit_raises(self):
        model = ARIMAModel()
        with pytest.raises(ValueError, match="not trained"):
            model.predict(4)
