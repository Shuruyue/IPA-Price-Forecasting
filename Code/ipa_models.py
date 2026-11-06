"""
IPA Price Prediction Model - Machine Learning Models Module
ML Models for Isopropyl Alcohol Price Prediction

Models included:
- SARIMA time series model
- XGBoost / GradientBoosting tree model (with auto-fallback)
- Ensemble weighting utilities

Optional (not used in main pipeline):
- LSTM neural network (requires TensorFlow)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency availability flags
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    SARIMAX = None  # type: ignore[assignment,misc]
    ARIMA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    XGB_AVAILABLE = False

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.models import Sequential
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False


# ===================================================================
# Model Evaluator
# ===================================================================

class ModelEvaluator:
    """Model evaluation metrics calculator."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        """Calculate evaluation metrics (MAE, RMSE, MAPE, Direction Accuracy)."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_acc = float(np.mean(true_direction == pred_direction) * 100)
        else:
            direction_acc = 0.0

        return {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_acc,
        }

    @staticmethod
    def print_results(results: Dict[str, Any]) -> None:
        """Print evaluation results."""
        print(f"\n{'=' * 50}")
        print(f"Model: {results['Model']}")
        print(f"{'=' * 50}")
        print(f"MAE:  {results['MAE']:.4f}")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"MAPE: {results['MAPE']:.2f}%")
        print(f"Direction Accuracy: {results['Direction_Accuracy']:.2f}%")


# ===================================================================
# ARIMA / SARIMA
# ===================================================================

class ARIMAModel:
    """SARIMA Time Series Model wrapper."""

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 4),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.model: Any = None
        self.fitted_model: Any = None

    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None) -> ARIMAModel:
        """Train SARIMA model."""
        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels not installed")

        print(f"Training SARIMA{self.order}x{self.seasonal_order}...")

        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_model = self.model.fit(disp=False)
        print("[OK] SARIMA training complete")
        return self

    def predict(self, steps: int, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast *steps* periods ahead."""
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
        return self.fitted_model.forecast(steps=steps, exog=exog)

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values."""
        return self.fitted_model.fittedvalues


# ===================================================================
# LSTM (optional, not used in main pipeline)
# ===================================================================

class LSTMModel:
    """LSTM Neural Network Model (requires TensorFlow)."""

    def __init__(
        self,
        lookback: int = 8,
        units: int = 64,
        dropout: float = 0.2,
        epochs: int = 100,
    ) -> None:
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.model: Any = None

    def _prepare_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Prepare LSTM sequence data."""
        X_seq = []
        y_seq = []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback:i])
            if y is not None:
                y_seq.append(y[i])

        X_seq = np.array(X_seq)
        if y is not None:
            return X_seq, np.array(y_seq)
        return X_seq

    def build_model(self, input_shape: tuple):
        """Build LSTM model architecture."""
        if not LSTM_AVAILABLE:
            raise ImportError("TensorFlow not installed")

        model = Sequential([
            Bidirectional(LSTM(self.units, return_sequences=True), input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train LSTM model."""
        print("Training LSTM neural network...")

        X_seq, y_seq = self._prepare_sequences(X, y)
        self.model = self.build_model((X_seq.shape[1], X_seq.shape[2]))

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
        )
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=16,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0,
        )
        print(f"[OK] LSTM training complete (epochs: {len(history.history['loss'])})")
        return self, history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from input features."""
        X_seq = self._prepare_sequences(X)
        return self.model.predict(X_seq, verbose=0).flatten()


# ===================================================================
# XGBoost / GradientBoosting
# ===================================================================

class XGBoostModel:
    """XGBoost Gradient Boosting Model (falls back to sklearn GradientBoostingRegressor)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.model: Any = None
        self.backend_name: Optional[str] = None
        self.display_name: Optional[str] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostModel:
        """Train model, auto-selecting XGBoost or sklearn fallback."""
        if not XGB_AVAILABLE:
            print("XGBoost unavailable, using GradientBoostingRegressor backend...")
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
            )
            self.backend_name = 'gradient_boosting'
            self.display_name = 'GradientBoosting'
        else:
            print("Training XGBoost...")
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
                n_jobs=1,
            )
            self.backend_name = 'xgboost'
            self.display_name = 'XGBoost'

        self.model.fit(X, y)
        print(f"[OK] {self.display_name} training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from input features."""
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance sorted descending."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)

    def get_display_name(self) -> str:
        return self.display_name or 'XGBoost'


# Main test program
if __name__ == "__main__":
    print("Model Module Test")
    print("=" * 50)

    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 + 40

    train_size = int(n_samples * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print("\nTesting XGBoost model...")
    xgb_model = XGBoostModel(n_estimators=50)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    results = ModelEvaluator.evaluate(y_test, xgb_pred, "XGBoost")
    ModelEvaluator.print_results(results)

    print("\n[OK] Model module test complete!")
