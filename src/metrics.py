"""Proper distance metrics for TDA — compositional, correlation, and robust variants.

Public API:
    aitchison_distance(X) -> np.ndarray (condensed)
    robust_correlation_distance(X) -> np.ndarray
    get_distance_matrix(X, metric) -> np.ndarray (square form)
    recommended_metric_for_data(df) -> str
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def aitchison_distance(X: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    """Aitchison distance for compositional data (e.g., microbiome, metabolomics).

    Implements the centred log-ratio (clr) transform followed by Euclidean distance.
    Properly handles the simplex constraint of compositional data.

    Args:
        X: (n_samples, n_features) — strictly positive compositional data.
        pseudocount: added to zeros before log transform.

    Returns:
        Condensed distance vector (n_pairs,).
    """
    X_pos = np.maximum(X, 0) + pseudocount
    # Centred log-ratio transform
    gm = np.exp(np.mean(np.log(X_pos), axis=1, keepdims=True))
    clr = np.log(X_pos / gm)
    return pdist(clr, metric="euclidean")


def robust_correlation_distance(X: np.ndarray, method: str = "spearman") -> np.ndarray:
    """Robust correlation distance using Spearman's ρ.

    Converts Spearman ρ to a distance: d = 1 - |ρ| (abs because negative
    correlations are also informative structure).

    More robust to outliers and non-linearities than Pearson correlation.

    Args:
        X: (n_samples, n_features).
        method: 'spearman' only for now.

    Returns:
        Condensed distance vector.
    """
    n = X.shape[1]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if method == "spearman":
                rho, _ = spearmanr(X[:, i], X[:, j])
                corr[i, j] = 1.0 - abs(rho)
                corr[j, i] = corr[i, j]

    # Return condensed form
    return squareform(corr, checks=False)


def get_distance_matrix(X: np.ndarray, metric: str = "euclidean", **kwargs) -> np.ndarray:
    """Compute a square distance matrix with the specified metric.

    Supported metrics:
        'euclidean' — standard Euclidean
        'correlation' — 1 - Pearson r (note: not a true metric)
        'spearman' — 1 - |Spearman ρ| (robust to outliers)
        'aitchison' — for compositional data (microbiome, metabolomics)
        'cosine' — cosine distance
        'cityblock' — Manhattan (L1)
    """
    if metric == "aitchison":
        dist_vec = aitchison_distance(X, **kwargs)
    elif metric == "spearman":
        dist_vec = robust_correlation_distance(X, **kwargs)
    elif metric in ("euclidean", "correlation", "cosine", "cityblock"):
        dist_vec = pdist(X, metric=metric)
    else:
        raise ValueError(
            f"Unknown metric: {metric}. "
            f"Supported: euclidean, correlation, spearman, aitchison, cosine, cityblock"
        )

    return squareform(dist_vec)


def recommended_metric_for_data(df) -> str:
    """Heuristic to recommend a distance metric based on data characteristics.

    - Compositional (all columns sum to ~constant): 'aitchison'
    - Count data (non-negative integers): 'spearman' (robust)
    - Log-transformed continuous: 'euclidean'
    - Default: 'spearman'
    """
    import pandas as pd
    data = df.select_dtypes(include=[np.number]).values

    # Check if compositional (rows sum to ~constant)
    row_sums = data.sum(axis=1)
    cv = np.std(row_sums) / (np.mean(row_sums) + 1e-10)
    if cv < 0.1 and np.all(data >= 0):
        return "aitchison"

    # Check if count data
    if np.all(data == data.astype(int)) and np.all(data >= 0):
        return "spearman"

    return "euclidean"
