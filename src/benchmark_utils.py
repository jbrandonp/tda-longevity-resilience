"""Benchmark utilities — aging clock wrappers for ComputAgeBench compatibility.

Wraps TDA-derived features into the standard aging clock interface so they
can be compared head-to-head with classical epigenetic clocks (Horvath,
Hannum, GrimAge, PhenoAge) using the ComputAgeBench framework.

Reference: ComputAgeBench (https://github.com/computage/computage-bench)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


class TDAgingClock(BaseEstimator, RegressorMixin):
    """Wraps a TDA-based aging model as a scikit-learn-compatible clock.

    ComputAgeBench expects: predict(X) -> chronological age (years).
    """

    def __init__(self, tda_model=None, scaler=None):
        self.tda_model = tda_model
        self.scaler = scaler

    def fit(self, X, y):
        if self.tda_model is not None:
            self.tda_model.fit(X, y)
        return self

    def predict(self, X):
        if self.tda_model is not None:
            return self.tda_model.predict(X)
        return np.full(X.shape[0], np.nan)


def wrap_as_aging_clock(
    pipeline,
    clock_name: str = "TDA-AgingClock",
) -> TDAgingClock:
    """Wrap a TDA pipeline as a ComputAgeBench-compatible aging clock.

    Args:
        pipeline: fitted sklearn Pipeline.
        clock_name: human-readable name for the clock.

    Returns:
        TDAgingClock wrapping the pipeline.
    """
    clock = TDAgingClock(tda_model=pipeline)
    clock.__class__.__name__ = clock_name
    logger.info(f"Wrapped pipeline as ComputAgeBench clock: {clock_name}")
    return clock


def compare_with_classical(
    tda_clock: TDAgingClock,
    classical_clocks: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Compare TDA clock against classical epigenetic clocks.

    Args:
        tda_clock: fitted TDAgingClock.
        classical_clocks: {name: fitted model with predict(X)->age}.
        X_test: test data matrix.
        y_test: true chronological ages.

    Returns:
        DataFrame with clock_name, MAE, RMSE, Pearson_r.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    rows = []
    for name, clock in [("TDA", tda_clock)] + list(classical_clocks.items()):
        pred = clock.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r, _ = pearsonr(y_test, pred)
        rows.append({
            "clock": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "Pearson_r": round(r, 4),
        })

    df = pd.DataFrame(rows)
    logger.info(f"TDA clock benchmark: MAE={rows[0]['MAE']:.2f} yrs")
    return df


def accelerated_aging_detection(
    clock: TDAgingClock,
    X: np.ndarray,
    chronological_age: np.ndarray,
    threshold_sigma: float = 1.0,
) -> pd.Series:
    """Detect accelerated aging using a fitted aging clock.

    Args:
        clock: fitted TDAgingClock.
        X: omics data matrix.
        chronological_age: true age in years.
        threshold_sigma: how many std above mean counts as 'accelerated'.

    Returns:
        Series with 'normal', 'accelerated', 'decelerated' labels.
    """
    predicted_age = clock.predict(X)
    age_acceleration = predicted_age - chronological_age
    std = np.std(age_acceleration)

    labels = np.full(len(age_acceleration), "normal", dtype=object)
    labels[age_acceleration > threshold_sigma * std] = "accelerated"
    labels[age_acceleration < -threshold_sigma * std] = "decelerated"

    counts = pd.Series(labels).value_counts().to_dict()
    logger.info(f"Accelerated aging detection: {counts}")
    return pd.Series(labels, name="aging_status")
