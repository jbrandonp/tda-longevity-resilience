"""Tests for features.py — topological feature extractors."""

import numpy as np
import pytest
from src.features import (
    PersistenceImageTransformer,
    PersistenceLandscapeTransformer,
    BettiCurveTransformer,
    extract_all_features,
)
from src.config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


def _make_simple_dgm(n_pairs=5):
    """Generate a simple persistence diagram for testing."""
    births = rng.uniform(0, 0.5, n_pairs)
    deaths = births + rng.uniform(0.1, 1.0, n_pairs)
    return np.column_stack([births, deaths])


class TestPersistenceImageTransformer:
    def test_fit_transform_shape(self):
        pi = PersistenceImageTransformer(spread=0.1, pixels=(20, 20))
        dgms = [_make_simple_dgm(5) for _ in range(10)]
        X = pi.fit_transform(dgms)
        assert X.shape == (10, 400)  # 20*20

    def test_empty_diagram(self):
        pi = PersistenceImageTransformer(spread=0.1, pixels=(10, 10))
        dgms = [np.empty((0, 2))]
        X = pi.fit_transform(dgms)
        assert X.shape == (1, 100)

    def test_list_of_list_diagrams(self):
        pi = PersistenceImageTransformer(spread=0.1, pixels=(10, 10))
        # Simulate [dgm0, dgm1] structure
        dgms_list = [
            [np.empty((0, 2)), _make_simple_dgm(3)],  # H0 empty, H1 has 3 pairs
        ]
        X = pi.fit_transform(dgms_list)
        assert X.shape == (1, 100)

    def test_scikit_learn_compatible(self):
        from sklearn.pipeline import Pipeline
        pi = PersistenceImageTransformer(spread=0.1, pixels=(10, 10))
        dgms = [_make_simple_dgm(5) for _ in range(20)]
        y = np.array([0] * 10 + [1] * 10)
        pipe = Pipeline([("pi", pi)])
        X = pipe.fit_transform(dgms, y)
        assert X.shape == (20, 100)


class TestPersistenceLandscapeTransformer:
    def test_fit_transform_shape(self):
        pl = PersistenceLandscapeTransformer(n_layers=3, n_bins=50)
        dgms = [_make_simple_dgm(5) for _ in range(10)]
        X = pl.fit_transform(dgms)
        assert X.shape == (10, 150)  # 3 * 50

    def test_empty_diagram(self):
        pl = PersistenceLandscapeTransformer(n_layers=3, n_bins=20)
        dgms = [np.empty((0, 2))]
        X = pl.fit_transform(dgms)
        assert X.shape == (1, 60)


class TestBettiCurveTransformer:
    def test_fit_transform_shape(self):
        bc = BettiCurveTransformer(n_bins=50, dim=1)
        dgms = [_make_simple_dgm(5) for _ in range(10)]
        X = bc.fit_transform(dgms)
        assert X.shape == (10, 50)

    def test_empty_diagram(self):
        bc = BettiCurveTransformer(n_bins=30)
        dgms = [np.empty((0, 2))]
        X = bc.fit_transform(dgms)
        assert X.shape == (1, 30)


class TestExtractAllFeatures:
    def test_returns_dict_with_all_types(self):
        dgms = [_make_simple_dgm(5) for _ in range(5)]
        features = extract_all_features(dgms, spread=0.1, pixels=(20, 20))
        assert "persistence_images" in features
        assert "landscapes" in features
        assert "betti_curves" in features
        assert features["persistence_images"].shape == (5, 400)
        assert features["betti_curves"].shape == (5, 100)
