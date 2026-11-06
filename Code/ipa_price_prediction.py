"""
IPA Price Prediction Model - Main Script
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from ipa_data_collector import IPADataCollector
from ipa_feature_engineering import IPAFeatureEngineer
from ipa_models import ARIMAModel, ModelEvaluator, XGBoostModel

# ---------------------------------------------------------------------------
# Tuning & ensemble constants
# ---------------------------------------------------------------------------
ENSEMBLE_ELIGIBILITY_RATIO: float = 3.0
ENSEMBLE_WEIGHT_POWER: float = 2.0
EXOGENOUS_DAMPING_FACTOR: float = 0.85
EXOGENOUS_SLOPE_CAP: float = 0.08
EXOGENOUS_LOOKBACK: int = 8
SENSITIVITY_PERTURBATION: float = 0.1
MIN_TRAIN_SIZE: int = 12
MIN_UNCERTAINTY_RATIO: float = 0.03
RESIDUAL_QUANTILE: float = 0.75

# XGBoost hyperparameter search space
XGBOOST_PARAM_GRID: List[Dict[str, Any]] = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
    {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.05},
    {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.08},
    {'n_estimators': 180, 'max_depth': 5, 'learning_rate': 0.08},
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8},
    {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
    {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7},
    {'n_estimators': 120, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8},
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.9},
    {'n_estimators': 250, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8},
]

# SARIMA hyperparameter search space
SARIMA_PARAM_GRID: List[Dict[str, tuple]] = [
    {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 4)},
    {'order': (1, 1, 0), 'seasonal_order': (1, 1, 0, 4)},
    {'order': (0, 1, 1), 'seasonal_order': (0, 1, 1, 4)},
    {'order': (2, 1, 1), 'seasonal_order': (1, 1, 0, 4)},
]

# Key features for sensitivity analysis
SENSITIVITY_KEY_FEATURES: List[str] = [
    'WTI_Price', 'Brent_Price', 'USD_TWD', 'DXY',
    'NatGas_Price', 'IPA_Price_TWD_lag1', 'TSM_Price', 'TWII',
]


class IPAPricePredictor:
    TIME_FEATURES = {
        'Year', 'Quarter', 'Month', 'Week',
        'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos',
    }
    BASE_EXOGENOUS_FEATURES = tuple(
        IPAFeatureEngineer.EXOGENOUS_COLS + IPADataCollector.EVENT_COLUMNS
    )
    EVENT_FEATURE_KEYWORDS = {
        'COVID19', 'Ukraine_War', 'RedSea_Crisis', 'US_China_Trade', 'Geopolitical_Risk',
    }
    TARGET_CHANGE_PERIODS = {
        'pct_change': 1,
        'monthly_change': 2,
        'quarterly_change': 4,
    }

    def __init__(
        self,
        target_year: int = 2025,
        start_date: str = '2012-01-01',
        end_date: str = '2024-12-31',
        cache_dir: str = 'Data',
        figures_dir: str = 'figures',
        reports_dir: str = 'reports',
        use_cache: bool = True,
        refresh_cache: bool = False,
        random_seed: int = 42,
        price_file: Optional[str] = None,
    ) -> None:
        self.target_year = target_year
        self.start_date = start_date
        self.end_date = end_date
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.price_file = self._resolve_path(price_file) if price_file else None
        self.cache_dir = self._resolve_path(cache_dir)
        self.figure_dir = self._resolve_path(figures_dir)
        self.report_dir = self._resolve_path(reports_dir)
        self.use_cache = use_cache
        self.refresh_cache = refresh_cache
        self.random_seed = random_seed

        self.data: Optional[pd.DataFrame] = None
        self.quarterly_data: Optional[pd.DataFrame] = None
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.results: List[Dict[str, Any]] = []
        self.feature_cols: List[str] = []
        self.exogenous_feature_cols: List[str] = []
        self.base_exogenous_feature_cols: List[str] = []
        self.backtest_results: List[Dict[str, Any]] = []
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.future_predictions: Optional[pd.DataFrame] = None
        self.sensitivity_results: Optional[Dict[str, float]] = None
        self.feature_importance_df: Optional[pd.DataFrame] = None
        self.ensemble_weights: Dict[str, float] = {}
        self.artifacts: Dict[str, str] = {}
        self.model_display_names: Dict[str, str] = {
            'xgboost': 'XGBoost',
            'sarima': 'SARIMA',
            'ensemble': 'Ensemble',
        }
        self.generated_at: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        np.random.seed(self.random_seed)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_mape(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denominator = np.where(np.abs(y_true) < 1e-6, 1.0, np.abs(y_true))
        return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)

    @staticmethod
    def _is_target_derived_feature(feature_name: str) -> bool:
        return feature_name.startswith('IPA_Price_TWD_')

    @staticmethod
    def _direction_accuracy(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size <= 1:
            return 0.0
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return float(np.mean(true_direction == pred_direction) * 100)

    @staticmethod
    def _safe_pct_change(values, periods: int) -> float:
        values = np.asarray(values, dtype=float)
        if values.size <= periods:
            return 0.0
        base = values[-(periods + 1)]
        return 0.0 if abs(base) < 1e-6 else float((values[-1] - base) / base)

    @staticmethod
    def _json_compatible(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _clip_outliers(
        self,
        df: pd.DataFrame,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        exclude_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        exclude = set(exclude_cols or [])
        result = df.copy()
        for col in result.select_dtypes(include=[np.number]).columns:
            if col in exclude:
                continue
            lower = result[col].quantile(quantile_low)
            upper = result[col].quantile(quantile_high)
            result[col] = result[col].clip(lower=lower, upper=upper)
        return result

    def load_and_prepare_data(self) -> IPAPricePredictor:
        print("\n" + "=" * 60)
        print(f"IPA Price Prediction Model - Forecasting {self.target_year}")
        print("=" * 60)

        # Reset run-specific states before data preparation.
        self.models = {}
        self.predictions = {}
        self.results = []
        self.feature_cols = []
        self.exogenous_feature_cols = []
        self.base_exogenous_feature_cols = []
        self.backtest_results = []
        self.future_predictions = None
        self.sensitivity_results = None
        self.feature_importance_df = None
        self.ensemble_weights = {}

        feature_engineer = IPAFeatureEngineer()

        print("\n[1/4] Creating IPA price data...")
        ipa_data = feature_engineer.create_ipa_price_data(json_path=self.price_file)

        print("\n[2/4] Collecting external data...")
        collector = IPADataCollector(
            start_date=self.start_date,
            end_date=self.end_date,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            refresh_cache=self.refresh_cache,
        )
        external_data = collector.collect_all_data()

        print("\n[3/4] Merging data...")
        if external_data is None or external_data.empty:
            self.data = ipa_data.copy()
        else:
            self.data = ipa_data.join(external_data, how='left')
        self.data = self.data.sort_index().ffill()
        self.data = self._clip_outliers(self.data, exclude_cols=['IPA_Price_TWD'])

        print("\n[4/4] Feature engineering...")
        quarterly_base = feature_engineer.resample_to_quarterly(self.data).sort_index()
        self.quarterly_data = feature_engineer.prepare_quarterly_features(
            quarterly_base, target_col='IPA_Price_TWD',
        ).sort_index()

        print("\n[OK] Data preparation complete")
        print(f"  Weekly merged data: {len(self.data)} records")
        print(f"  Quarterly data: {len(self.quarterly_data)} records")
        return self

    # ------------------------------------------------------------------
    # Walk-forward backtesting
    # ------------------------------------------------------------------

    def _walk_forward_backtest(
        self,
        y: np.ndarray,
        model_key: str,
        params: Dict[str, Any],
        X: Optional[np.ndarray] = None,
        n_splits: int = 4,
        min_train_size: int = MIN_TRAIN_SIZE,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        y = np.asarray(y, dtype=float)
        if y.size < (min_train_size + 4):
            return {'CV_MAPE': float('inf'), 'CV_Folds': 0}, []

        usable_splits = min(n_splits, max(2, y.size // 8))
        splitter = TimeSeriesSplit(n_splits=usable_splits)
        fold_rows: List[Dict[str, Any]] = []

        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(y), start=1):
            if len(train_idx) < min_train_size or len(val_idx) == 0:
                continue

            y_train, y_val = y[train_idx], y[val_idx]

            try:
                if model_key == 'xgboost':
                    model = XGBoostModel(**params)
                    model.fit(X[train_idx], y_train)
                    pred = np.asarray(model.predict(X[val_idx]), dtype=float)
                elif model_key == 'sarima':
                    model = ARIMAModel(order=params['order'], seasonal_order=params['seasonal_order'])
                    model.fit(y_train)
                    pred = np.asarray(model.predict(len(val_idx)), dtype=float)
                else:
                    raise ValueError(f"Unsupported model_key for backtest: {model_key}")
            except Exception as e:
                logger.warning("Backtest fold %s failed for %s: %s", fold_id, model_key, e)
                continue

            fold_rows.append({
                'ModelKey': model_key,
                'Fold': fold_id,
                'TrainSize': int(len(train_idx)),
                'ValSize': int(len(val_idx)),
                'MAE': float(np.mean(np.abs(y_val - pred))),
                'RMSE': float(np.sqrt(np.mean((y_val - pred) ** 2))),
                'MAPE': self._safe_mape(y_val, pred),
                'Direction_Accuracy': self._direction_accuracy(y_val, pred),
            })

        if not fold_rows:
            return {'CV_MAPE': float('inf'), 'CV_Folds': 0}, []

        fold_df = pd.DataFrame(fold_rows)
        summary = {
            'CV_MAE': float(fold_df['MAE'].mean()),
            'CV_RMSE': float(fold_df['RMSE'].mean()),
            'CV_MAPE': float(fold_df['MAPE'].mean()),
            'CV_Direction_Accuracy': float(fold_df['Direction_Accuracy'].mean()),
            'CV_Folds': int(len(fold_df)),
        }
        return summary, fold_rows

    # ------------------------------------------------------------------
    # Model fitting with CV
    # ------------------------------------------------------------------

    def _fit_xgboost_with_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[XGBoostModel, Dict, Dict, List]:
        print("Walk-forward tuning for XGBoost...")
        best_params, best_summary, best_rows = None, {'CV_MAPE': float('inf')}, []

        for params in XGBOOST_PARAM_GRID:
            cv_summary, cv_rows = self._walk_forward_backtest(
                y_train, model_key='xgboost', params=params, X=X_train,
            )
            cv_mape = cv_summary.get('CV_MAPE', float('inf'))
            print(f"  Params {params} -> CV MAPE: {cv_mape:.2f}%")
            if cv_mape < best_summary.get('CV_MAPE', float('inf')):
                best_params, best_summary, best_rows = params, cv_summary, cv_rows

        if best_params is None:
            best_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
            best_summary = {'CV_MAPE': float('nan'), 'CV_Folds': 0}

        print(f"Best XGBoost params: {best_params}")
        model = XGBoostModel(**best_params)
        model.fit(X_train, y_train)
        return model, best_params, best_summary, best_rows

    def _fit_sarima_with_cv(
        self,
        y_train: np.ndarray,
    ) -> Tuple[ARIMAModel, Dict, Dict, List]:
        print("Walk-forward tuning for SARIMA...")
        best_params, best_summary, best_rows = None, {'CV_MAPE': float('inf')}, []

        for params in SARIMA_PARAM_GRID:
            cv_summary, cv_rows = self._walk_forward_backtest(
                y_train, model_key='sarima', params=params,
            )
            cv_mape = cv_summary.get('CV_MAPE', float('inf'))
            print(f"  Params {params} -> CV MAPE: {cv_mape:.2f}%")
            if cv_mape < best_summary.get('CV_MAPE', float('inf')):
                best_params, best_summary, best_rows = params, cv_summary, cv_rows

        if best_params is None:
            best_params = {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 4)}
            best_summary = {'CV_MAPE': float('nan'), 'CV_Folds': 0}

        print(f"Best SARIMA params: {best_params}")
        model = ARIMAModel(order=best_params['order'], seasonal_order=best_params['seasonal_order'])
        model.fit(y_train)
        return model, best_params, best_summary, best_rows

    # ------------------------------------------------------------------
    # Ensemble weights
    # ------------------------------------------------------------------

    def _derive_ensemble_weights(self) -> None:
        valid_rows = []
        for row in self.results:
            if row.get('ModelKey') not in {'xgboost', 'sarima'}:
                continue
            cv_error = row.get('CV_MAPE')
            test_error = row.get('MAPE')
            selected_error = cv_error if cv_error is not None and np.isfinite(cv_error) else test_error
            if selected_error is None or not np.isfinite(selected_error):
                continue
            valid_rows.append((row['ModelKey'], float(selected_error)))

        if not valid_rows:
            self.ensemble_weights = {}
            return

        error_by_model = dict(valid_rows)
        best_error = min(error_by_model.values())
        eligible_errors = {
            k: v for k, v in error_by_model.items()
            if v <= best_error * ENSEMBLE_ELIGIBILITY_RATIO
        } or error_by_model

        inverse_error = {
            k: (1.0 / max(v, 1e-6)) ** ENSEMBLE_WEIGHT_POWER
            for k, v in eligible_errors.items()
        }
        total = sum(inverse_error.values())
        self.ensemble_weights = {m: w / total for m, w in inverse_error.items()} if total > 0 else {}

        print("Adaptive ensemble weights:")
        for model_key, weight in self.ensemble_weights.items():
            print(f"  {self.model_display_names.get(model_key, model_key)}: {weight:.2%}")

    def _get_model_blend_weights(self, model_keys: List[str]) -> Dict[str, float]:
        if not model_keys:
            return {}
        raw_weights = {k: float(self.ensemble_weights.get(k, 0.0)) for k in model_keys}
        total = sum(raw_weights.values())
        if total <= 0:
            uniform = 1.0 / len(model_keys)
            return {k: uniform for k in model_keys}
        return {k: v / total for k, v in raw_weights.items()}

    # ------------------------------------------------------------------
    # Ensemble evaluation on test set
    # ------------------------------------------------------------------

    def _evaluate_ensemble_on_test(self) -> Optional[Dict[str, Any]]:
        if self.y_test is None or len(self.y_test) == 0:
            return None
        if 'xgboost_test' not in self.predictions and 'sarima_test' not in self.predictions:
            return None

        self.results = [r for r in self.results if r.get('ModelKey') != 'ensemble']
        ensemble_test = self._blend_predictions('xgboost_test', 'sarima_test')

        ensemble_results = ModelEvaluator.evaluate(self.y_test, ensemble_test, self.model_display_names['ensemble'])
        ensemble_results['ModelKey'] = 'ensemble'
        self.results.append(ensemble_results)
        ModelEvaluator.print_results(ensemble_results)
        self.predictions['ensemble_test'] = ensemble_test
        return ensemble_results

    def _blend_predictions(self, xgb_key: str, sarima_key: str) -> np.ndarray:
        """Blend XGBoost and SARIMA predictions using ensemble weights."""
        has_xgb = xgb_key in self.predictions
        has_sarima = sarima_key in self.predictions

        if has_xgb and has_sarima:
            weights = self._get_model_blend_weights(['xgboost', 'sarima'])
            return (
                np.asarray(self.predictions[xgb_key], dtype=float) * weights['xgboost']
                + np.asarray(self.predictions[sarima_key], dtype=float) * weights['sarima']
            )
        elif has_xgb:
            return np.asarray(self.predictions[xgb_key], dtype=float)
        else:
            return np.asarray(self.predictions[sarima_key], dtype=float)

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_models(self, test_size: float = 0.2) -> IPAPricePredictor:
        print("\n" + "=" * 60)
        print("Model Training")
        print("=" * 60)

        if self.quarterly_data is None or self.quarterly_data.empty:
            raise ValueError("Quarterly data is empty. Run load_and_prepare_data() first.")

        df = self.quarterly_data.copy()
        target_col = 'IPA_Price_TWD'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]
        if not feature_cols:
            raise ValueError("No numeric feature columns available for training.")

        X = df[feature_cols].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        split_idx = int(len(X) * (1 - test_size))
        split_idx = min(max(split_idx, 8), len(X) - 4)
        if split_idx <= 0 or split_idx >= len(X):
            raise ValueError("Dataset too small for time-series split.")

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        print(f"\nTraining set: {len(X_train)} records")
        print(f"Test set: {len(X_test)} records")

        # --- XGBoost ---
        print("\n" + "-" * 40)
        print("Training XGBoost model...")
        xgb_model, best_params, xgb_cv_summary, xgb_cv_rows = self._fit_xgboost_with_cv(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_name = xgb_model.get_display_name()
        self.model_display_names['xgboost'] = xgb_name

        xgb_results = ModelEvaluator.evaluate(y_test, xgb_pred, xgb_name)
        xgb_results['ModelKey'] = 'xgboost'
        xgb_results['CV_MAPE'] = xgb_cv_summary.get('CV_MAPE')
        xgb_results['CV_RMSE'] = xgb_cv_summary.get('CV_RMSE')
        xgb_results['CV_Direction_Accuracy'] = xgb_cv_summary.get('CV_Direction_Accuracy')
        xgb_results['CV_Folds'] = xgb_cv_summary.get('CV_Folds')
        xgb_results['Params'] = str(best_params)
        self.results.append(xgb_results)
        for fold_row in xgb_cv_rows:
            self.backtest_results.append({**fold_row, 'Model': xgb_name})
        ModelEvaluator.print_results(xgb_results)
        self.models['xgboost'] = xgb_model
        self.predictions['xgboost_test'] = np.asarray(xgb_pred, dtype=float)
        self.feature_importance_df = xgb_model.get_feature_importance(feature_cols)
        print("\nTop 10 important features:")
        print(self.feature_importance_df.head(10).to_string(index=False))

        # --- SARIMA ---
        print("\n" + "-" * 40)
        print("Training SARIMA model...")
        try:
            arima_model, sarima_params, sarima_cv_summary, sarima_cv_rows = self._fit_sarima_with_cv(y_train)
            arima_pred_array = np.asarray(arima_model.predict(len(y_test)), dtype=float)
            arima_results = ModelEvaluator.evaluate(y_test, arima_pred_array, "SARIMA")
            arima_results['ModelKey'] = 'sarima'
            arima_results['CV_MAPE'] = sarima_cv_summary.get('CV_MAPE')
            arima_results['CV_RMSE'] = sarima_cv_summary.get('CV_RMSE')
            arima_results['CV_Direction_Accuracy'] = sarima_cv_summary.get('CV_Direction_Accuracy')
            arima_results['CV_Folds'] = sarima_cv_summary.get('CV_Folds')
            arima_results['Params'] = str(sarima_params)
            self.results.append(arima_results)
            for fold_row in sarima_cv_rows:
                self.backtest_results.append({**fold_row, 'Model': 'SARIMA'})
            ModelEvaluator.print_results(arima_results)
            self.models['sarima'] = arima_model
            self.predictions['sarima_test'] = arima_pred_array
        except Exception as e:
            logger.warning("SARIMA training failed: %s", e)
            print(f"SARIMA training failed: {e}")

        self.X, self.y, self.y_test, self.feature_cols = X, y, y_test, feature_cols
        self.exogenous_feature_cols = [
            c for c in self.feature_cols
            if (not self._is_target_derived_feature(c)) and (c not in self.TIME_FEATURES)
        ]
        self.base_exogenous_feature_cols = [
            c for c in self.feature_cols if c in self.BASE_EXOGENOUS_FEATURES
        ]
        self._derive_ensemble_weights()

        print("\n" + "-" * 40)
        print("Evaluating Ensemble model...")
        self._evaluate_ensemble_on_test()

        print("\n" + "-" * 40)
        print("Sensitivity analysis...")
        self.sensitivity_results = self._perform_sensitivity_analysis()
        return self

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def _perform_sensitivity_analysis(self) -> Optional[Dict[str, float]]:
        if 'xgboost' not in self.models:
            return None
        model = self.models['xgboost']
        base_pred = model.predict(self.X)
        base_mean = np.mean(base_pred)
        if abs(base_mean) < 1e-6:
            return None

        sensitivity: Dict[str, float] = {}
        for i, feature in enumerate(self.feature_cols):
            feature_name = str(feature).split(',')[0].replace('(', '').replace("'", "").strip()
            if not any(kf in feature_name for kf in SENSITIVITY_KEY_FEATURES):
                continue
            X_up, X_down = self.X.copy(), self.X.copy()
            X_up[:, i] *= (1 + SENSITIVITY_PERTURBATION)
            X_down[:, i] *= (1 - SENSITIVITY_PERTURBATION)
            impact = (np.mean(model.predict(X_up)) - np.mean(model.predict(X_down))) / (2 * base_mean) * 100
            sensitivity[feature_name] = impact

        sorted_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\nKey variable sensitivity (+/-{SENSITIVITY_PERTURBATION * 100:.0f}% perturbation):")
        print("-" * 45)
        for feat, impact in sorted_sens[:5]:
            direction = "positive" if impact > 0 else "negative"
            print(f"  {feat:20s}: {direction:8s} {abs(impact):.2f}%")
        return dict(sorted_sens)

    # ------------------------------------------------------------------
    # Future prediction
    # ------------------------------------------------------------------

    def _compute_forecast_horizon(
        self, quarters_ahead: int,
    ) -> Tuple[int, int, List, List]:
        last_period = self.quarterly_data.index[-1].to_period('Q')
        next_period = last_period + 1
        target_start = pd.Period(f"{self.target_year}Q1", freq='Q')
        step_offset = target_start.ordinal - next_period.ordinal
        if step_offset < 0:
            raise ValueError(
                f"Target year {self.target_year} is earlier than available data end {last_period}."
            )
        total_steps = step_offset + quarters_ahead
        periods = [next_period + i for i in range(total_steps)]
        return total_steps, step_offset, periods, periods[step_offset:step_offset + quarters_ahead]

    def _project_exogenous_features(self, future_periods: List) -> pd.DataFrame:
        """Project exogenous variables into the future with damped trend."""
        if not self.base_exogenous_feature_cols:
            return pd.DataFrame(index=pd.PeriodIndex(future_periods, freq='Q'))

        history_df = self.quarterly_data[self.base_exogenous_feature_cols].copy()
        forecast_index = pd.PeriodIndex(future_periods, freq='Q')
        projection = pd.DataFrame(
            index=forecast_index,
            columns=self.base_exogenous_feature_cols,
            dtype=float,
        )

        for col in self.base_exogenous_feature_cols:
            series = history_df[col].dropna().astype(float)
            if series.empty:
                projection[col] = 0.0
                continue
            last_value = float(series.iloc[-1])

            # Event features: hold constant
            if any(key in col for key in self.EVENT_FEATURE_KEYWORDS):
                projection[col] = last_value
                continue

            # Trend projection with damping
            lookback = min(EXOGENOUS_LOOKBACK, len(series))
            recent = series.iloc[-lookback:].to_numpy(dtype=float)
            slope = float(np.polyfit(np.arange(lookback), recent, 1)[0]) if lookback >= 3 else 0.0
            scale = max(abs(np.mean(recent)), 1.0)
            slope = float(np.clip(slope, -EXOGENOUS_SLOPE_CAP * scale, EXOGENOUS_SLOPE_CAP * scale))
            value = last_value
            for step, period in enumerate(forecast_index, start=1):
                value += slope * (EXOGENOUS_DAMPING_FACTOR ** (step - 1))
                projection.at[period, col] = max(value, 0.0) if last_value >= 0 else value

        return projection.ffill().bfill().fillna(0.0)

    def _update_exogenous_derived_features(
        self,
        row: pd.Series,
        exogenous_history: Optional[pd.DataFrame],
    ) -> pd.Series:
        if exogenous_history is None or exogenous_history.empty:
            return row

        latest = exogenous_history.iloc[-1]
        previous = exogenous_history.iloc[-2] if len(exogenous_history) >= 2 else latest
        rolling_windows = [2, 4, 8]

        for col in self.base_exogenous_feature_cols:
            if col in row.index and col in latest.index:
                row[col] = float(latest[col])

            lag_col = f'{col}_lag1'
            if lag_col in row.index and col in previous.index:
                row[lag_col] = float(previous[col])

            if col not in exogenous_history.columns:
                continue
            series = exogenous_history[col].dropna().astype(float).to_numpy(dtype=float)
            for w in rolling_windows:
                ma_col = f'{col}_ma{w}'
                if ma_col not in row.index:
                    continue
                d = series[-w:] if series.size >= w else series
                row[ma_col] = float(np.mean(d)) if d.size else 0.0

        for col_a, col_b in IPAFeatureEngineer.INTERACTION_PAIRS:
            interaction_col = f'{col_a}_x_{col_b}'
            if interaction_col in row.index and col_a in row.index and col_b in row.index:
                row[interaction_col] = float(row[col_a]) * float(row[col_b])

        return row

    def _build_future_feature_row(
        self,
        future_period,
        target_history: List[float],
        template_row: pd.Series,
        exogenous_values: Optional[pd.Series] = None,
        exogenous_history: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        row = template_row.copy()
        history = np.asarray(target_history, dtype=float)
        ts = future_period.to_timestamp(how='end')

        # Set base exogenous values
        if exogenous_values is not None:
            for col in self.base_exogenous_feature_cols:
                if col in row.index and col in exogenous_values.index:
                    row[col] = float(exogenous_values[col])

        # Set time features
        if 'Year' in row.index:
            row['Year'] = ts.year
        if 'Quarter' in row.index:
            row['Quarter'] = future_period.quarter
        if 'Month' in row.index:
            row['Month'] = ts.month
        if 'Week' in row.index:
            row['Week'] = int(ts.isocalendar().week)
        if 'Month_sin' in row.index:
            row['Month_sin'] = np.sin(2 * np.pi * ts.month / 12)
        if 'Month_cos' in row.index:
            row['Month_cos'] = np.cos(2 * np.pi * ts.month / 12)
        if 'Quarter_sin' in row.index:
            row['Quarter_sin'] = np.sin(2 * np.pi * future_period.quarter / 4)
        if 'Quarter_cos' in row.index:
            row['Quarter_cos'] = np.cos(2 * np.pi * future_period.quarter / 4)

        # Recompute target-derived features from history
        for col in row.index:
            lag_match = re.fullmatch(r'IPA_Price_TWD_lag(\d+)', col)
            if lag_match:
                lag = int(lag_match.group(1))
                if history.size >= lag:
                    row[col] = history[-lag]
                continue

            ma_match = re.fullmatch(r'IPA_Price_TWD_ma(\d+)', col)
            if ma_match:
                w = int(ma_match.group(1))
                d = history[-w:] if history.size >= w else history
                row[col] = float(np.mean(d)) if d.size else 0.0
                continue

            std_match = re.fullmatch(r'IPA_Price_TWD_std(\d+)', col)
            if std_match:
                w = int(std_match.group(1))
                d = history[-w:] if history.size >= w else history
                row[col] = float(np.std(d, ddof=0)) if d.size else 0.0
                continue

            max_match = re.fullmatch(r'IPA_Price_TWD_max(\d+)', col)
            if max_match:
                w = int(max_match.group(1))
                d = history[-w:] if history.size >= w else history
                row[col] = float(np.max(d)) if d.size else 0.0
                continue

            min_match = re.fullmatch(r'IPA_Price_TWD_min(\d+)', col)
            if min_match:
                w = int(min_match.group(1))
                d = history[-w:] if history.size >= w else history
                row[col] = float(np.min(d)) if d.size else 0.0
                continue

            if col == 'IPA_Price_TWD_pct_change':
                row[col] = self._safe_pct_change(history, periods=self.TARGET_CHANGE_PERIODS['pct_change'])
            elif col == 'IPA_Price_TWD_monthly_change':
                row[col] = self._safe_pct_change(history, periods=self.TARGET_CHANGE_PERIODS['monthly_change'])
            elif col == 'IPA_Price_TWD_quarterly_change':
                row[col] = self._safe_pct_change(history, periods=self.TARGET_CHANGE_PERIODS['quarterly_change'])

        row = self._update_exogenous_derived_features(row, exogenous_history)
        return row.reindex(self.feature_cols).fillna(0.0).astype(float)

    def predict_future(self, quarters_ahead: int = 4) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print(f"Predicting {self.target_year} Q1-Q4")
        print("=" * 60)

        total_steps, step_offset, all_periods, target_periods = self._compute_forecast_horizon(quarters_ahead)
        predictions: Dict[str, Any] = {}
        exog_projection = self._project_exogenous_features(all_periods)

        exog_history = pd.DataFrame()
        if self.base_exogenous_feature_cols:
            exog_history = self.quarterly_data[self.base_exogenous_feature_cols].copy()

        # --- XGBoost recursive prediction ---
        if 'xgboost' in self.models:
            xgb_path: List[float] = []
            target_history = self.y.tolist()
            template_row = self.quarterly_data[self.feature_cols].iloc[-1].copy()

            for period in all_periods:
                exogenous_values = exog_projection.loc[period] if not exog_projection.empty else None
                if exogenous_values is not None and not exog_history.empty:
                    period_ts = period.to_timestamp(how='end')
                    exog_history.loc[period_ts] = (
                        exogenous_values.reindex(self.base_exogenous_feature_cols).to_numpy(dtype=float)
                    )
                feature_row = self._build_future_feature_row(
                    period, target_history, template_row,
                    exogenous_values=exogenous_values,
                    exogenous_history=exog_history,
                )
                next_pred = float(
                    self.models['xgboost'].predict(feature_row.to_numpy(dtype=float).reshape(1, -1))[0]
                )
                xgb_path.append(next_pred)
                target_history.append(next_pred)
                template_row = feature_row.copy()
            predictions['xgboost'] = xgb_path[step_offset:step_offset + quarters_ahead]

        # --- SARIMA prediction ---
        if 'sarima' in self.models:
            try:
                sarima_values = np.asarray(self.models['sarima'].predict(total_steps), dtype=float)
                predictions['sarima'] = sarima_values[step_offset:step_offset + quarters_ahead].tolist()
            except Exception as e:
                logger.warning("SARIMA prediction failed: %s", e)
                print(f"SARIMA prediction failed: {e}")

        # --- Ensemble blend ---
        if 'xgboost' in predictions and 'sarima' in predictions:
            weights = self._get_model_blend_weights(['xgboost', 'sarima'])
            predictions['ensemble'] = (
                np.asarray(predictions['xgboost'], dtype=float) * weights['xgboost']
                + np.asarray(predictions['sarima'], dtype=float) * weights['sarima']
            ).tolist()
        elif 'xgboost' in predictions:
            predictions['ensemble'] = predictions['xgboost']
        elif 'sarima' in predictions:
            predictions['ensemble'] = predictions['sarima']
        else:
            fallback = float(np.mean(self.y[-4:])) if len(self.y) >= 4 else float(np.mean(self.y))
            predictions['ensemble'] = [fallback] * quarters_ahead

        # --- Confidence intervals ---
        base_pred = np.asarray(predictions['ensemble'], dtype=float)
        uncertainty_ratio = self._compute_uncertainty_ratio()
        predictions['optimistic'] = (base_pred * (1 + uncertainty_ratio)).tolist()
        predictions['pessimistic'] = (base_pred * (1 - uncertainty_ratio)).tolist()

        # --- Build output DataFrame ---
        pred_df = pd.DataFrame({
            'Quarter': [str(p) for p in target_periods],
            'Base': predictions['ensemble'],
            'Optimistic': predictions['optimistic'],
            'Pessimistic': predictions['pessimistic'],
        })
        if 'xgboost' in predictions:
            pred_df[self.model_display_names.get('xgboost', 'XGBoost')] = predictions['xgboost']
        if 'sarima' in predictions:
            pred_df[self.model_display_names.get('sarima', 'SARIMA')] = predictions['sarima']

        self.future_predictions = pred_df
        print("\n" + "-" * 40)
        print(f"{self.target_year} Price Prediction (TWD/KG)")
        print("-" * 40)
        print(pred_df.to_string(index=False))
        print(f"Adaptive uncertainty band: +/- {uncertainty_ratio * 100:.2f}%")
        return pred_df

    def _compute_uncertainty_ratio(self) -> float:
        """Derive adaptive confidence band width from volatility and residuals."""
        recent_history = self.y[-12:] if len(self.y) >= 12 else self.y
        mean_history = np.mean(np.abs(recent_history))
        market_volatility = float(np.std(recent_history) / mean_history) if mean_history > 1e-6 else MIN_UNCERTAINTY_RATIO

        residual_ratio = MIN_UNCERTAINTY_RATIO
        if self.y_test is not None and len(self.y_test) > 0:
            base_test = self._get_ensemble_test_prediction()
            if base_test is not None:
                residual = np.asarray(self.y_test) - base_test
                denom = np.mean(np.abs(self.y_test))
                denom = denom if denom > 1e-6 else 1.0
                q75_abs_res = float(np.quantile(np.abs(residual), RESIDUAL_QUANTILE))
                residual_ratio = max(residual_ratio, q75_abs_res / denom)

        return max(MIN_UNCERTAINTY_RATIO, market_volatility, residual_ratio)

    def _get_ensemble_test_prediction(self) -> Optional[np.ndarray]:
        """Get blended test-set prediction for uncertainty estimation."""
        has_xgb = 'xgboost_test' in self.predictions
        has_sarima = 'sarima_test' in self.predictions
        if not has_xgb and not has_sarima:
            return None
        return self._blend_predictions('xgboost_test', 'sarima_test') if (has_xgb or has_sarima) else None

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def create_visualizations(self, save_path: Optional[str] = None) -> List:
        print("\n" + "=" * 60)
        print("Generating Visualization Charts")
        print("=" * 60)

        save_path = self.figure_dir if save_path is None else self._resolve_path(save_path)
        os.makedirs(save_path, exist_ok=True)
        figures = []

        # Historical price trend
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(
            self.quarterly_data.index,
            self.quarterly_data['IPA_Price_TWD'],
            'b-', linewidth=2, label='IPA Price',
        )
        ax1.set_title('Isopropyl Alcohol (IPA) Historical Price Trend (2012-2024)', fontsize=14)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Price (TWD/KG)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.savefig(f'{save_path}/historical_prices.png', dpi=120, bbox_inches='tight')
        figures.append(fig1)
        print("  [OK] Historical price trend chart")

        # Prediction chart
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        pred_df = self.future_predictions
        x = range(len(pred_df))
        ax2.fill_between(x, pred_df['Pessimistic'], pred_df['Optimistic'], alpha=0.3, color='blue', label='Prediction Range')
        ax2.plot(x, pred_df['Base'], 'bo-', linewidth=2, markersize=10, label='Base Prediction')
        ax2.plot(x, pred_df['Optimistic'], 'g--', linewidth=1.5, label='Optimistic Scenario')
        ax2.plot(x, pred_df['Pessimistic'], 'r--', linewidth=1.5, label='Pessimistic Scenario')
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(pred_df['Quarter'])
        ax2.set_title(f'{self.target_year} IPA Price Prediction', fontsize=14)
        ax2.set_xlabel('Quarter', fontsize=12)
        ax2.set_ylabel('Price (TWD/KG)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        for i, row in pred_df.iterrows():
            ax2.annotate(
                f"{row['Base']:.1f}", (i, row['Base']),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10,
            )
        fig2.savefig(f'{save_path}/prediction_{self.target_year}.png', dpi=120, bbox_inches='tight')
        figures.append(fig2)
        print(f"  [OK] {self.target_year} prediction chart")

        # Model comparison
        if self.results:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            model_names = [r['Model'] for r in self.results]
            mapes = [r['MAPE'] for r in self.results]
            palette = ['steelblue', 'seagreen', 'coral', 'slategray']
            bars = ax3.bar(model_names, mapes, color=palette[:len(model_names)])
            ax3.set_title('Model MAPE Comparison', fontsize=14)
            ax3.set_ylabel('MAPE (%)', fontsize=12)
            for bar, mape in zip(bars, mapes):
                ax3.annotate(
                    f'{mape:.2f}%',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 5), ha='center',
                )
            fig3.savefig(f'{save_path}/model_comparison.png', dpi=120, bbox_inches='tight')
            figures.append(fig3)
            print("  [OK] Model comparison chart")

        plt.close('all')
        print(f"\n[OK] All charts saved to: {save_path}/")
        return figures

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_path: Optional[str] = None) -> str:
        print("\n" + "=" * 60)
        print("Generating Analysis Report")
        print("=" * 60)

        output_path = self.report_dir if output_path is None else self._resolve_path(output_path)
        os.makedirs(output_path, exist_ok=True)

        rows_html = ''.join([
            f"<tr><td>{row['Quarter']}</td><td><strong>{row['Base']:.2f}</strong></td>"
            f"<td style='color:green'>{row['Optimistic']:.2f}</td>"
            f"<td style='color:red'>{row['Pessimistic']:.2f}</td></tr>"
            for _, row in self.future_predictions.iterrows()
        ])

        metrics_html = ''.join([
            f"<div><strong>{r['Model']}</strong>: MAPE {r['MAPE']:.2f}%"
            + (f" | CV MAPE {float(r['CV_MAPE']):.2f}%" if r.get('CV_MAPE') is not None and np.isfinite(r.get('CV_MAPE')) else "")
            + "</div>"
            for r in self.results
        ])

        weights_html = ''.join([
            f"<li>{self.model_display_names.get(k, k)}: {v:.2%}</li>"
            for k, v in self.ensemble_weights.items()
        ]) or "<li>N/A</li>"

        model_method = (
            f"{self.model_display_names.get('xgboost', 'XGBoost')} + "
            f"{self.model_display_names.get('sarima', 'SARIMA')} adaptive ensemble"
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IPA Price Prediction Report - {self.target_year}</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #f5f7fa;
      --card: #ffffff;
      --accent: #2563eb;
      --accent-light: #dbeafe;
      --text: #1e293b;
      --text-muted: #64748b;
      --green: #16a34a;
      --red: #dc2626;
      --border: #e2e8f0;
      --shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Inter', 'Segoe UI', 'Microsoft JhengHei', sans-serif;
      background: var(--bg); color: var(--text);
      max-width: 960px; margin: 0 auto; padding: 24px; line-height: 1.6;
    }}
    .header {{
      background: linear-gradient(135deg, #1e40af, #3b82f6);
      color: white; padding: 32px; border-radius: 16px;
      margin-bottom: 24px; box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }}
    .header h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 8px; }}
    .header p {{ opacity: 0.85; font-size: 0.9rem; }}
    .card {{
      background: var(--card); border-radius: 12px;
      padding: 24px; margin-bottom: 20px;
      border: 1px solid var(--border); box-shadow: var(--shadow);
    }}
    .card h2 {{
      font-size: 1.1rem; font-weight: 600; margin-bottom: 16px;
      padding-bottom: 8px; border-bottom: 2px solid var(--accent-light);
    }}
    .stats-grid {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
    }}
    .stat-box {{
      background: var(--accent-light); border-radius: 10px; padding: 16px; text-align: center;
    }}
    .stat-box .value {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
    .stat-box .label {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ background: var(--accent); color: white; padding: 10px 12px;
         font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    td {{ padding: 10px 12px; text-align: center; border-bottom: 1px solid var(--border); }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    tr:hover {{ background: var(--accent-light); }}
    .positive {{ color: var(--green); font-weight: 500; }}
    .negative {{ color: var(--red); font-weight: 500; }}
    .metric-bar {{ display: flex; align-items: center; gap: 8px;
      padding: 8px 12px; border-radius: 8px; background: #f8fafc; margin-bottom: 6px; }}
    .metric-bar strong {{ min-width: 140px; }}
    img {{ max-width: 100%; border-radius: 10px; margin: 8px 0; }}
    ul {{ padding-left: 20px; }}
    li {{ margin-bottom: 4px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .disclaimer {{
      font-size: 0.8rem; color: var(--text-muted);
      border-top: 1px solid var(--border); padding-top: 16px; margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>IPA Price Prediction Report</h1>
    <p>Forecast Year: {self.target_year} &nbsp;|&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  </div>

  <div class="card">
    <h2>Summary</h2>
    <div class="stats-grid">
      <div class="stat-box">
        <div class="value">{self.future_predictions['Base'].mean():.2f}</div>
        <div class="label">Base Mean (TWD/KG)</div>
      </div>
      <div class="stat-box" style="background:#dcfce7">
        <div class="value" style="color:var(--green)">{self.future_predictions['Optimistic'].mean():.2f}</div>
        <div class="label">Optimistic Mean</div>
      </div>
      <div class="stat-box" style="background:#fee2e2">
        <div class="value" style="color:var(--red)">{self.future_predictions['Pessimistic'].mean():.2f}</div>
        <div class="label">Pessimistic Mean</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Quarterly Prediction</h2>
    <table>
      <tr><th>Quarter</th><th>Base</th><th>Optimistic</th><th>Pessimistic</th></tr>
      {rows_html}
    </table>
  </div>

  <div class="card">
    <h2>Visualizations</h2>
    <img src="../figures/historical_prices.png" alt="Historical Prices">
    <img src="../figures/prediction_{self.target_year}.png" alt="Prediction {self.target_year}">
    <img src="../figures/model_comparison.png" alt="Model Comparison">
  </div>

  <div class="card">
    <h2>Model Evaluation</h2>
    {metrics_html}
  </div>

  <div class="card">
    <h2>Ensemble Weights</h2>
    <ul>{weights_html}</ul>
  </div>

  <div class="card">
    <h2>Exported Files</h2>
    <ul>
      <li><a href="./ipa_forecast_{self.target_year}.csv">Quarterly forecast CSV</a></li>
      <li><a href="./model_metrics.csv">Model metrics CSV</a></li>
      <li><a href="./walk_forward_backtest.csv">Walk-forward backtest CSV</a></li>
      <li><a href="./feature_importance.csv">Feature importance CSV</a></li>
      <li><a href="./sensitivity_analysis.csv">Sensitivity analysis CSV</a></li>
      <li><a href="./run_manifest_{self.target_year}.json">Run manifest JSON</a></li>
    </ul>
  </div>

  <div class="card">
    <h2>Notes</h2>
    <ul>
      <li>Data range: January 2012 to December 2024</li>
      <li>Method: Walk-forward tuned {model_method}</li>
      <li>Factors: energy prices, exchange rates, geopolitical events, seasonality</li>
      <li>Scenario range: based on market volatility and residual distribution</li>
    </ul>
    <p class="disclaimer">Disclaimer: prediction is for reference only.</p>
  </div>
</body>
</html>
"""

        report_file = f'{output_path}/ipa_forecast_{self.target_year}.html'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[OK] HTML report saved: {report_file}")
        return report_file

    # ------------------------------------------------------------------
    # Artifact export
    # ------------------------------------------------------------------

    def save_artifacts(self, output_path: Optional[str] = None) -> Dict[str, str]:
        output_path = self.report_dir if output_path is None else self._resolve_path(output_path)
        os.makedirs(output_path, exist_ok=True)
        artifact_paths: Dict[str, str] = {}

        if self.future_predictions is not None and not self.future_predictions.empty:
            pred_csv = os.path.join(output_path, f'ipa_forecast_{self.target_year}.csv')
            self.future_predictions.to_csv(pred_csv, index=False)
            artifact_paths['forecast_csv'] = pred_csv

        if self.results:
            metrics_df = pd.DataFrame(self.results)
            if 'ModelKey' in metrics_df.columns:
                metrics_df = metrics_df.drop(columns=['ModelKey'])
            metrics_csv = os.path.join(output_path, 'model_metrics.csv')
            metrics_df.to_csv(metrics_csv, index=False)
            artifact_paths['model_metrics_csv'] = metrics_csv

        if self.backtest_results:
            backtest_df = pd.DataFrame(self.backtest_results)
            backtest_csv = os.path.join(output_path, 'walk_forward_backtest.csv')
            backtest_df.to_csv(backtest_csv, index=False)
            artifact_paths['walk_forward_backtest_csv'] = backtest_csv

        if self.feature_importance_df is not None and not self.feature_importance_df.empty:
            fi_csv = os.path.join(output_path, 'feature_importance.csv')
            self.feature_importance_df.to_csv(fi_csv, index=False)
            artifact_paths['feature_importance_csv'] = fi_csv

        if self.sensitivity_results:
            sens_df = pd.DataFrame(
                [{'feature': k, 'impact_pct': v} for k, v in self.sensitivity_results.items()]
            )
            sens_csv = os.path.join(output_path, 'sensitivity_analysis.csv')
            sens_df.to_csv(sens_csv, index=False)
            artifact_paths['sensitivity_csv'] = sens_csv

        weights_json = os.path.join(output_path, 'ensemble_weights.json')
        weight_payload = {
            self.model_display_names.get(k, k): float(v)
            for k, v in self.ensemble_weights.items()
        }
        with open(weights_json, 'w', encoding='utf-8') as f:
            json.dump(weight_payload, f, ensure_ascii=False, indent=2)
        artifact_paths['ensemble_weights_json'] = weights_json

        manifest_json = os.path.join(output_path, f'run_manifest_{self.target_year}.json')
        manifest_payload = {
            'generated_at': self.generated_at,
            'target_year': int(self.target_year),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'cache_dir': self.cache_dir,
            'use_cache': bool(self.use_cache),
            'refresh_cache': bool(self.refresh_cache),
            'random_seed': int(self.random_seed),
            'price_file': self.price_file,
            'model_display_names': self.model_display_names,
            'ensemble_weights': {k: float(v) for k, v in self.ensemble_weights.items()},
            'backtest_rows': int(len(self.backtest_results)),
        }
        with open(manifest_json, 'w', encoding='utf-8') as f:
            json.dump(manifest_payload, f, ensure_ascii=False, indent=2, default=self._json_compatible)
        artifact_paths['manifest_json'] = manifest_json

        self.artifacts[self.target_year] = artifact_paths
        print(f"[OK] Artifacts exported: {output_path}")
        return artifact_paths

    # ------------------------------------------------------------------
    # Pipeline entry points
    # ------------------------------------------------------------------

    def run_full_pipeline(self, test_size: float = 0.2) -> IPAPricePredictor:
        self.load_and_prepare_data()
        self.train_models(test_size=test_size)
        self.predict_future()
        self.create_visualizations(save_path=self.figure_dir)
        self.generate_report(output_path=self.report_dir)
        self.save_artifacts(output_path=self.report_dir)
        print("\n" + "=" * 60)
        print("[OK] Prediction pipeline complete")
        print("=" * 60)
        return self

    def run_forecasts_for_years(
        self,
        years: List[int],
        test_size: float = 0.2,
    ) -> Dict[int, Dict[str, Any]]:
        years = sorted(set(int(y) for y in years))
        if not years:
            raise ValueError("At least one forecast year is required.")

        self.target_year = years[0]
        self.load_and_prepare_data()
        self.train_models(test_size=test_size)

        outputs: Dict[int, Dict[str, Any]] = {}
        summary_frames: List[pd.DataFrame] = []

        for year in years:
            self.target_year = year
            pred_df = self.predict_future()
            self.create_visualizations(save_path=self.figure_dir)
            report_file = self.generate_report(output_path=self.report_dir)
            artifact_paths = self.save_artifacts(output_path=self.report_dir)
            summary_frames.append(pred_df.assign(Forecast_Year=year))
            outputs[year] = {
                'prediction': pred_df.copy(),
                'report_file': report_file,
                'artifacts': artifact_paths,
            }

        if summary_frames:
            summary_df = pd.concat(summary_frames, axis=0, ignore_index=True)
            summary_file = os.path.join(self.report_dir, 'ipa_forecast_all_years.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"[OK] Multi-year summary CSV saved: {summary_file}")

        print("\n" + "=" * 60)
        print(f"[OK] Multi-year forecasting complete: {', '.join(str(y) for y in years)}")
        print("=" * 60)
        return outputs


# ===================================================================
# CLI entry point
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IPA quarterly price forecasting pipeline.")
    parser.add_argument('--years', nargs='+', type=int, default=[2025, 2026],
                        help='Forecast years, e.g. --years 2025 2026')
    parser.add_argument('--start-date', type=str, default='2012-01-01',
                        help='Historical start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                        help='Historical end date (YYYY-MM-DD)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test ratio for final time split (default: 0.2)')
    parser.add_argument('--cache-dir', type=str, default='Data',
                        help='Cache directory for merged market data')
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help='Directory for output figures')
    parser.add_argument('--reports-dir', type=str, default='reports',
                        help='Directory for output reports and CSV/JSON artifacts')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable local cache')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Force refresh market data from source')
    parser.add_argument('--price-file', type=str, default=None,
                        help='Monthly IPA price JSON (default: Data/ipa_prices.json; '
                             'use Data/ipa_prices.sample.json for synthetic demo data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity level (default: INFO)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    predictor = IPAPricePredictor(
        target_year=min(args.years),
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        figures_dir=args.figures_dir,
        reports_dir=args.reports_dir,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
        random_seed=args.seed,
        price_file=args.price_file,
    )
    predictor.run_forecasts_for_years(args.years, test_size=args.test_size)
