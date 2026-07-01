"""Shared pytest fixtures for tda-longevity-resilience tests."""

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


@pytest.fixture(scope="session")
def random_seed():
    return RANDOM_SEED


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate a small synthetic multi-omics dataset for integration tests."""
    from src.data_utils import generate_synthetic_multimics
    return generate_synthetic_multimics(n_samples=50, topology_type="circle", noise=0.05, n_features=20)


@pytest.fixture(scope="module")
def simple_dgm():
    """A simple persistence diagram with known structure."""
    h0 = np.array([[0.0, 0.5], [0.0, 0.8], [0.0, np.inf]])
    h1 = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.4]])
    return [h0, h1]


@pytest.fixture(scope="module")
def empty_dgm():
    """Empty persistence diagram."""
    return [np.empty((0, 2)), np.empty((0, 2))]


@pytest.fixture(scope="module")
def sample_labels():
    """Binary labels for classification tests."""
    return pd.Series(["accelerated"] * 25 + ["resilient"] * 25)


@pytest.fixture(scope="module")
def sample_dataframe():
    """A simple DataFrame with numeric columns."""
    return pd.DataFrame(rng.standard_normal((50, 10)), columns=[f"gene_{i}" for i in range(10)])
