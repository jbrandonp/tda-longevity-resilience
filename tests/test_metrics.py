"""Tests for metrics.py — distance metrics for TDA."""

import numpy as np
import pytest
from src.metrics import (
    aitchison_distance,
    get_distance_matrix,
    recommended_metric_for_data,
)


class TestAitchisonDistance:
    def test_returns_correct_shape(self):
        X = np.abs(np.random.randn(30, 10)) + 0.1
        d = aitchison_distance(X)
        expected_pairs = 30 * 29 // 2
        assert len(d) == expected_pairs

    def test_identical_data_zero_distance(self):
        X = np.ones((10, 5)) * 5.0
        d = aitchison_distance(X)
        assert np.allclose(d, 0, atol=1e-10)

    def test_positive_values_required(self):
        X = np.random.randn(20, 5)  # may have negatives
        d = aitchison_distance(X)
        assert not np.any(np.isnan(d)), "Should handle negatives via pseudocount"


class TestGetDistanceMatrix:
    def test_euclidean_square(self):
        X = np.random.randn(20, 5)
        dm = get_distance_matrix(X, metric="euclidean")
        assert dm.shape == (20, 20)
        assert np.allclose(np.diag(dm), 0)

    def test_aitchison_square(self):
        X = np.abs(np.random.randn(20, 5)) + 0.1
        dm = get_distance_matrix(X, metric="aitchison")
        assert dm.shape == (20, 20)

    def test_unknown_metric_raises(self):
        X = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="Unknown metric"):
            get_distance_matrix(X, metric="nonexistent")


class TestRecommendedMetric:
    def test_count_data_returns_spearman(self):
        import pandas as pd
        df = pd.DataFrame(np.random.randint(0, 100, (20, 5)))
        m = recommended_metric_for_data(df)
        assert m == "spearman"

    def test_continuous_returns_euclidean(self):
        import pandas as pd
        df = pd.DataFrame(np.random.randn(20, 5))
        m = recommended_metric_for_data(df)
        assert m == "euclidean"
