"""Tests for benchmark_utils.py — aging clock wrappers."""
import numpy as np, pandas as pd, pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.benchmark_utils import (
    TDAgingClock, wrap_as_aging_clock,
    compare_with_classical, accelerated_aging_detection,
)


class TestTDAgingClock:
    def test_fit_predict(self):
        X = np.random.randn(100, 10); y = np.random.randn(100) * 10 + 50
        pipe = Pipeline([("lr", LinearRegression())])
        clock = TDAgingClock(tda_model=pipe)
        clock.fit(X, y)
        assert clock.predict(X[:5]).shape == (5,)
    def test_default_predict(self):
        clock = TDAgingClock()
        assert np.all(np.isnan(clock.predict(np.random.randn(5, 3))))


class TestWrapAsAgingClock:
    def test_returns_tda_aging_clock(self):
        pipe = Pipeline([("lr", LinearRegression())])
        pipe.fit(np.random.randn(10, 5), np.random.randn(10))
        clock = wrap_as_aging_clock(pipe, "TDA-LR")
        assert isinstance(clock, TDAgingClock)


class TestCompareWithClassical:
    def test_returns_dataframe(self):
        X = np.random.randn(100, 5); y = np.random.randn(100) * 5 + 40
        pipe = Pipeline([("lr", LinearRegression())])
        clock = wrap_as_aging_clock(pipe.fit(X, y), "TDA")
        result = compare_with_classical(clock, {}, X, y)
        assert isinstance(result, pd.DataFrame)
        assert "MAE" in result.columns


class TestAcceleratedAgingDetection:
    def test_returns_series(self):
        X = np.random.randn(100, 5); age = np.random.randn(100) * 10 + 50
        pipe = Pipeline([("lr", LinearRegression())])
        clock = wrap_as_aging_clock(pipe.fit(X, age), "TDA")
        result = accelerated_aging_detection(clock, X, age)
        assert isinstance(result, pd.Series)
        assert set(result.unique()).issubset({"normal", "accelerated", "decelerated"})
