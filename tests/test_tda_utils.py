"""Tests for tda_utils.py — persistent homology computation and distances."""
import numpy as np, pytest
from src.tda_utils import (
    compute_persistence_diagrams, compute_all_layers_dgms,
    wasserstein_distance, bottleneck_distance,
    multi_view_topological_distance, permutation_test,
    diagnose_persistence, persistence_stability,
    sparse_persistence, bootstrap_diagram_stability,
)
from src.data_utils import _finite_dgm
from src.config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


class TestFiniteDiagram:
    def test_filters_infinite_deaths(self):
        result = _finite_dgm(np.array([[0.0, 1.0], [0.5, np.inf], [0.2, 0.8]]))
        assert result.shape == (2, 2); assert not np.any(np.isinf(result))
    def test_empty_diagram(self):
        assert _finite_dgm(np.empty((0, 2))).shape == (0, 2)
    def test_none_returns_empty(self):
        assert _finite_dgm(None).shape == (0, 2)


class TestComputePersistenceDiagrams:
    def test_returns_list(self):
        dgms = compute_persistence_diagrams(rng.standard_normal((30, 10)), max_dim=1, use_cache=False)
        assert isinstance(dgms, list) and len(dgms) == 2
    def test_h0_nonempty(self):
        dgms = compute_persistence_diagrams(rng.standard_normal((20, 5)), max_dim=0, use_cache=False)
        assert len(dgms) >= 1
    def test_small_dataset(self):
        dgms = compute_persistence_diagrams(rng.standard_normal((10, 3)), max_dim=1, use_cache=False)
        assert len(dgms) == 2
    def test_no_cache(self):
        data = rng.standard_normal((15, 5))
        dgms1 = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
        dgms2 = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
        assert len(dgms1) == len(dgms2)


class TestComputeAllLayers:
    def test_multi_layer(self):
        layers = {"a": rng.standard_normal((20, 8)), "b": rng.standard_normal((20, 6))}
        dgms = compute_all_layers_dgms(layers, max_dim=1)
        assert set(dgms.keys()) == {"a", "b"}


class TestWassersteinDistance:
    def test_self_zero(self):
        dgm = [np.array([[0.0, 1.0], [0.2, 0.8]])]
        d = wasserstein_distance(dgm, dgm, dim=0)
        assert d == pytest.approx(0.0, abs=1e-6) or d == float("inf")
    def test_empty(self):
        empty = [np.empty((0, 2)), np.empty((0, 2))]
        assert wasserstein_distance(empty, empty, dim=1) == 0.0

class TestBottleneckDistance:
    def test_self_zero(self):
        dgm = [np.array([[0.0, 1.0], [0.2, 0.8]])]
        d = bottleneck_distance(dgm, dgm, dim=0)
        assert d == pytest.approx(0.0, abs=1e-6) or d == float("inf")
    def test_empty(self):
        empty = [np.empty((0, 2)), np.empty((0, 2))]
        assert bottleneck_distance(empty, empty, dim=1) == 0.0


class TestMultiViewDistance:
    def test_returns_float(self):
        lyr = {"a": [np.array([[0.0, 1.0]])], "b": [np.array([[0.1, 0.5]])]}
        d = multi_view_topological_distance(lyr, lyr, dim=0)
        assert isinstance(d, float)


class TestPermutationTest:
    def test_returns_dict(self):
        lyr = {"a": [np.array([[0.0, 1.0]])], "b": [np.array([[0.1, 0.5]])]}
        r = permutation_test(lyr, lyr, n_perm=10, dim=0)
        assert "p_value" in r and "effect_size" in r and "significant" in r


class TestDiagnosePersistence:
    def test_keys(self):
        dgms = [np.array([[0.0, 1.0], [0.2, np.inf]]), np.array([[0.1, 0.5]])]
        info = diagnose_persistence(dgms)
        assert "H0" in info and "H1" in info
        assert info["H0"]["n_features"] >= 1
        assert "mean_lifetime" in info["H0"]


class TestPersistenceStability:
    def test_returns_float(self):
        val = persistence_stability(np.array([[0.1, 0.5], [0.15, 0.45]]), n_bootstrap=5)
        assert isinstance(val, float) and val >= 0


class TestSparsePersistence:
    def test_returns_dict(self):
        r = sparse_persistence(rng.standard_normal((50, 10)), subsample_ratio=0.4, n_subsamples=3)
        assert "H0" in r and "H1" in r and "subsample_sizes" in r


class TestBootstrapStability:
    def test_returns_dict(self):
        r = bootstrap_diagram_stability(rng.standard_normal((40, 8)), n_bootstrap=5)
        for dim in ["H0", "H1"]:
            assert dim in r and "stability_ratio" in r[dim]
