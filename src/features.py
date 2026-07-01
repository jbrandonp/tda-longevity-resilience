"""Topological feature extraction: persistence images, landscapes, Betti curves.

Public API (all scikit-learn compatible transformers):
    PersistenceImageTransformer
    PersistenceLandscapeTransformer
    BettiCurveTransformer
    extract_all_features(dgms) -> np.ndarray
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from .config import PI_SPREAD, PI_PIXELS
except ImportError:
    from config import PI_SPREAD, PI_PIXELS

try:
    from .data_utils import _finite_dgm
except ImportError:
    from data_utils import _finite_dgm


class PersistenceImageTransformer(BaseEstimator, TransformerMixin):
    """Transform persistence diagrams into persistence images."""

    def __init__(self, spread: float = PI_SPREAD, pixels: tuple = PI_PIXELS):
        self.spread = spread
        self.pixels = pixels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X is a list of persistence diagrams (one per sample)."""
        try:
            from persim import PersImage
        except ImportError:
            return np.zeros((len(X), self.pixels[0] * self.pixels[1]))

        pim = PersImage(spread=self.spread, pixels=self.pixels, verbose=False)
        vectors = []
        for dgm in X:
            if isinstance(dgm, list):
                # Use H1 diagram by default
                d = _finite_dgm(dgm[1]) if len(dgm) > 1 else _finite_dgm(dgm[0])
            else:
                d = _finite_dgm(dgm)
            if len(d) == 0:
                vectors.append(np.zeros(self.pixels[0] * self.pixels[1]))
            else:
                vectors.append(pim.transform(d).flatten())
        return np.array(vectors)


class PersistenceLandscapeTransformer(BaseEstimator, TransformerMixin):
    """Transform persistence diagrams into persistence landscape vectors."""

    def __init__(self, n_layers: int = 5, n_bins: int = 100):
        self.n_layers = n_layers
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X is a list of persistence diagrams."""
        try:
            from persim import PersLandscape
        except ImportError:
            return np.zeros((len(X), self.n_layers * self.n_bins))

        # Simple discretized landscape
        vectors = []
        for dgm in X:
            if isinstance(dgm, list):
                d = _finite_dgm(dgm[1]) if len(dgm) > 1 else _finite_dgm(dgm[0])
            else:
                d = _finite_dgm(dgm)

            if len(d) == 0:
                vectors.append(np.zeros(self.n_layers * self.n_bins))
                continue

            # Compute landscape by discretizing
            births = d[:, 0]
            deaths = d[:, 1]
            lifetimes = deaths - births

            t_min = float(np.min(births) if len(births) > 0 else 0)
            t_max = float(np.max(deaths) if len(deaths) > 0 else 1)
            bins = np.linspace(t_min, t_max, self.n_bins)
            landscape = np.zeros((self.n_layers, self.n_bins))

            for i, b in enumerate(bins):
                # Count how many intervals span point b, weighted by layer
                for layer in range(min(self.n_layers, len(d))):
                    covering = (d[:, 0] <= b) & (d[:, 1] >= b)
                    if np.any(covering):
                        # Sort lifetimes descending, pick the layer-th
                        sorted_lifetimes = np.sort(lifetimes[covering])[::-1]
                        if layer < len(sorted_lifetimes):
                            landscape[layer, i] = sorted_lifetimes[layer]

            vectors.append(landscape.flatten())

        return np.array(vectors)


class BettiCurveTransformer(BaseEstimator, TransformerMixin):
    """Transform persistence diagrams into Betti curves."""

    def __init__(self, n_bins: int = 100, dim: int = 1):
        self.n_bins = n_bins
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X is a list of persistence diagrams."""
        vectors = []
        for dgm in X:
            if isinstance(dgm, list):
                d = _finite_dgm(dgm[self.dim]) if len(dgm) > self.dim else _finite_dgm(dgm[0])
            else:
                d = _finite_dgm(dgm)

            if len(d) == 0:
                vectors.append(np.zeros(self.n_bins))
                continue

            births = d[:, 0]
            deaths = d[:, 1]

            t_min = float(np.min(births))
            t_max = float(np.max(deaths)) if len(deaths) > 0 else t_min + 1
            bins = np.linspace(t_min, t_max, self.n_bins)

            curve = np.array([np.sum((births <= b) & (deaths >= b)) for b in bins])
            vectors.append(curve)

        return np.array(vectors)


def extract_all_features(dgms_list: list, spread: float = None, pixels: tuple = None) -> dict:
    """Extract PI, PL, and Betti curve features from a list of diagrams.

    Args:
        dgms_list: list of persistence diagrams (one per sample).
        spread: PI spread parameter.
        pixels: PI pixel dimensions.

    Returns:
        dict with 'persistence_images', 'landscapes', 'betti_curves' as np.ndarray.
    """
    s = spread or PI_SPREAD
    p = pixels or PI_PIXELS

    pi_transformer = PersistenceImageTransformer(spread=s, pixels=p)
    pl_transformer = PersistenceLandscapeTransformer()
    bc_transformer = BettiCurveTransformer()

    return {
        "persistence_images": pi_transformer.transform(dgms_list),
        "landscapes": pl_transformer.transform(dgms_list),
        "betti_curves": bc_transformer.transform(dgms_list),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence Landscape Statistical Inference (Bubenik 2015)
# ═══════════════════════════════════════════════════════════════════════════════

def landscape_statistics(
    dgms_list: list,
    n_layers: int = 5,
    n_bins: int = 100,
    dim: int = 1,
) -> dict:
    """Compute mean landscape, variance, and confidence bands for a group of diagrams.

    Persistence landscapes form a Banach space, enabling classical statistical
    inference: pointwise mean, variance, and Central Limit Theorem-based
    confidence bands.

    Args:
        dgms_list: list of persistence diagrams (one per individual).
        n_layers: number of landscape layers.
        n_bins: discretization resolution.
        dim: homology dimension.

    Returns:
        dict with:
            'mean_landscape': (n_layers, n_bins) — pointwise mean
            'std_landscape': (n_layers, n_bins) — pointwise standard deviation
            'ci_lower': (n_layers, n_bins) — 95% CI lower bound (CLT)
            'ci_upper': (n_layers, n_bins) — 95% CI upper bound (CLT)
            'n_samples': int
            't_values': (n_bins,) — filtration parameter grid
    """
    pl = PersistenceLandscapeTransformer(n_layers=n_layers, n_bins=n_bins)
    landscapes = pl.transform(dgms_list)  # (n_samples, n_layers * n_bins)
    n_samples = landscapes.shape[0]

    # Reshape to (n_samples, n_layers, n_bins)
    landscapes_3d = landscapes.reshape(n_samples, n_layers, n_bins)

    mean_landscape = np.mean(landscapes_3d, axis=0)
    std_landscape = np.std(landscapes_3d, axis=0, ddof=1)

    # 95% CLT confidence bands: mean ± 1.96 * std / sqrt(n)
    se = std_landscape / np.sqrt(n_samples)
    ci_lower = mean_landscape - 1.96 * se
    ci_upper = mean_landscape + 1.96 * se

    # t_values: determine filtration grid from data
    all_births = []
    all_deaths = []
    for dgm in dgms_list:
        d = dgm[dim] if isinstance(dgm, list) and len(dgm) > dim else dgm
        finite = _finite_dgm(d)
        if len(finite) > 0:
            all_births.append(np.min(finite[:, 0]))
            all_deaths.append(np.max(finite[:, 1]))

    t_min = float(np.min(all_births)) if all_births else 0.0
    t_max = float(np.max(all_deaths)) if all_deaths else 1.0
    t_values = np.linspace(t_min, t_max, n_bins)

    return {
        "mean_landscape": mean_landscape,
        "std_landscape": std_landscape,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_samples": n_samples,
        "t_values": t_values,
    }


def landscape_distance(
    stats_a: dict,
    stats_b: dict,
    layer: int = 0,
) -> float:
    """L² distance between mean landscapes of two groups.

    Uses the Banach space structure: ||λ_A - λ_B||₂ integrated over the
    filtration parameter domain.

    Args:
        stats_a, stats_b: outputs of landscape_statistics() for each group.
        layer: which landscape layer to compare (0 = first layer).

    Returns:
        L² distance between mean landscapes.
    """
    mean_a = stats_a["mean_landscape"][layer]
    mean_b = stats_b["mean_landscape"][layer]
    t = stats_a["t_values"]

    # Trapezoidal integration of (λ_A - λ_B)²
    diff_sq = (mean_a - mean_b) ** 2
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    l2_distance = np.sqrt(np.trapezoid(diff_sq, t)) if len(t) > 1 else np.sqrt(np.mean(diff_sq))

    return float(l2_distance)
