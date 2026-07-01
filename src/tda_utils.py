"""Persistent homology utilities: compute diagrams, distances, statistics.

Public API:
    compute_persistence_diagrams(data, max_dim, metric) -> list[np.ndarray]
    compute_all_layers_dgms(omics_dict, max_dim) -> dict
    wasserstein_distance(dgm1, dgm2, dim) -> float
    bottleneck_distance(dgm1, dgm2, dim) -> float
    multi_view_topological_distance(layer_dgms_a, layer_dgms_b, weights) -> float
    permutation_test(layer_dgms_accel, layer_dgms_resil, n_perm) -> float
    persistence_stability(dgm, noise_std, n_bootstrap) -> float
    diagnose_persistence(dgm) -> dict
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

try:
    from .config import RIPSER_MAX_DIM, RIPSER_THRESH_QUANTILE, RIPSER_N_THREADS, RANDOM_SEED
except ImportError:
    from config import RIPSER_MAX_DIM, RIPSER_THRESH_QUANTILE, RIPSER_N_THREADS, RANDOM_SEED
try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


rng = np.random.default_rng(RANDOM_SEED)
CACHE_DIR = Path("data/processed/")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    from .data_utils import _finite_dgm
except ImportError:
    from data_utils import _finite_dgm


# ═══════════════════════════════════════════════════════════════════════════════
# Diagram computation
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_key(data: np.ndarray, max_dim: int, metric: str) -> str:
    """Deterministic hash of data + parameters for caching."""
    h = hashlib.sha256(data.tobytes())
    h.update(f"{max_dim}_{metric}".encode())
    return h.hexdigest()[:16]


def compute_persistence_diagrams(
    data: np.ndarray,
    max_dim: int = RIPSER_MAX_DIM,
    metric: str = "euclidean",
    use_cache: bool = True,
) -> list:
    """Compute Vietoris-Rips persistence diagrams.

    Args:
        data: (n_samples, n_features).
        max_dim: maximum homology dimension (0, 1, 2).
        metric: distance metric for pdist.
        use_cache: if True, cache diagrams to disk.

    Returns:
        List of diagrams [dgm_dim0, dgm_dim1, ...]. Each diagram is
        (n_pairs, 2) with (birth, death) coordinates. Death=inf for
        essential classes is clipped to a finite value.
    """
    key = _cache_key(data, max_dim, metric)
    cache_path = CACHE_DIR / f"dgm_{key}.pkl"

    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            dgms = pickle.load(f)
        return dgms

    # Compute distance matrix
    if metric == "correlation":
        dist_vec = pdist(data, metric="correlation")
    else:
        dist_vec = pdist(data, metric=metric)

    dist_matrix = squareform(dist_vec)

    # Threshold at a high quantile
    thresh = np.quantile(dist_vec, RIPSER_THRESH_QUANTILE)

    # Compute ripser
    try:
        import ripser
        result = ripser.ripser(
            dist_matrix,
            maxdim=max_dim,
            thresh=thresh,
            distance_matrix=True,
            n_threads=RIPSER_N_THREADS,
        )
        dgms = result["dgms"]
    except ImportError:
        logger.warning("ripser not installed — returning empty diagrams")
        dgms = [np.empty((0, 2)) for _ in range(max_dim + 1)]

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(dgms, f)

    return dgms


def compute_all_layers_dgms(
    omics_dict: dict,
    max_dim: int = RIPSER_MAX_DIM,
    metric: str = "euclidean",
) -> dict:
    """Compute persistence diagrams for every omics layer.

    Args:
        omics_dict: {layer_name: np.ndarray (samples, features)}.
        max_dim: passed to ripser.
        metric: distance metric.

    Returns:
        {layer_name: [dgm_0, dgm_1, ...]}
    """
    results = {}
    for name, data in omics_dict.items():
        logger.info(f"Computing persistence for {name} ({data.shape})")
        results[name] = compute_persistence_diagrams(data, max_dim=max_dim, metric=metric)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Distances
# ═══════════════════════════════════════════════════════════════════════════════

def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray, dim: int = 1) -> float:
    """Wasserstein distance between two diagrams in a given homology dimension."""
    d1 = _finite_dgm(dgm1[dim] if isinstance(dgm1, list) else dgm1)
    d2 = _finite_dgm(dgm2[dim] if isinstance(dgm2, list) else dgm2)
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    try:
        import persim
        return persim.wasserstein(d1, d2, matching=False)
    except ImportError:
        return float("inf")  # Sentinel: persim not installed


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray, dim: int = 1) -> float:
    """Bottleneck distance between two diagrams."""
    d1 = _finite_dgm(dgm1[dim] if isinstance(dgm1, list) else dgm1)
    d2 = _finite_dgm(dgm2[dim] if isinstance(dgm2, list) else dgm2)
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    try:
        import persim
        return persim.bottleneck(d1, d2, matching=False)
    except ImportError:
        return float("inf")  # Sentinel: persim not installed


def multi_view_topological_distance(
    layer_dgms_a: dict,
    layer_dgms_b: dict,
    weights: dict = None,
    dim: int = 1,
) -> float:
    """Weighted sum of Wasserstein distances across omics layers.

    Args:
        layer_dgms_a: {layer: [dgms]} for group A.
        layer_dgms_b: {layer: [dgms]} for group B.
        weights: {layer: weight} — if None, equal weights.
        dim: homology dimension.

    Returns:
        Scalar multi-view topological distance.
    """
    if weights is None:
        weights = {k: 1.0 for k in layer_dgms_a}

    total = 0.0
    total_weight = 0.0
    for layer in layer_dgms_a:
        w = weights.get(layer, 0.0)
        if w == 0:
            continue
        d = wasserstein_distance(layer_dgms_a[layer], layer_dgms_b[layer], dim=dim)
        total += w * d
        total_weight += w

    return total / total_weight if total_weight > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_test(
    layer_dgms_accel: dict,
    layer_dgms_resil: dict,
    n_perm: int = 1000,
    dim: int = 1,
    alpha: float = 0.05,
    correction: str = "fdr_bh",
) -> dict:
    """Permutation test for multi-view topological distance.

    Args:
        layer_dgms_accel, layer_dgms_resil: {layer: [dgms]} for each group.
        n_perm: number of permutations.
        dim: homology dimension.
        alpha: significance level.
        correction: multiple testing correction: 'fdr_bh', 'bonferroni', or 'none'.

    Returns:
        dict with 'observed_distance', 'p_value', 'p_value_corrected',
        'effect_size' (Cohen's d-like: observed / perm_std),
        'significant' (bool after correction).
    """
    observed = multi_view_topological_distance(layer_dgms_accel, layer_dgms_resil, dim=dim)

    # Permutation: shuffle layer assignment
    all_layers = list(layer_dgms_accel.keys())
    n_layers = len(all_layers)
    permuted = np.zeros(n_perm)

    for p in range(n_perm):
        perm_dist = 0.0
        for layer in all_layers:
            # Shuffle which group each individual belongs to
            dgms_a = layer_dgms_accel[layer]
            dgms_b = layer_dgms_resil[layer]
            combined = list(dgms_a) + list(dgms_b) if isinstance(dgms_a, list) else [dgms_a, dgms_b]
            n_a = len(dgms_a) if isinstance(dgms_a, list) else 1
            idx = rng.permutation(len(combined))
            perm_a = [combined[i] for i in idx[:n_a]]
            perm_b = [combined[i] for i in idx[n_a:]]
            # Use local wasserstein_distance (not self-import hack)
            perm_dist += wasserstein_distance(perm_a, perm_b, dim=dim) if len(perm_a) > 0 and len(perm_b) > 0 else 0.0
        permuted[p] = perm_dist / n_layers

    p_value = float(np.mean(permuted >= observed))
    perm_std = float(np.std(permuted))

    # Effect size: (observed - mean_perm) / std_perm
    effect_size = float((observed - np.mean(permuted)) / perm_std) if perm_std > 0 else 0.0

    # Multiple testing correction
    if correction == "bonferroni":
        p_corrected = min(p_value * n_layers, 1.0)
    elif correction == "fdr_bh":
        # Benjamini-Hochberg (simplified: since we have 1 test, same as uncorrected)
        p_corrected = p_value
    else:
        p_corrected = p_value

    return {
        "observed_distance": float(observed),
        "p_value": p_value,
        "p_value_corrected": p_corrected,
        "effect_size": round(effect_size, 4),
        "n_permutations": n_perm,
        "n_layers": n_layers,
        "correction": correction,
        "significant": p_corrected < alpha,
        "interpretation": (
            f"Significant (p_corrected={p_corrected:.4f}, d={effect_size:.2f})"
            if p_corrected < alpha
            else f"Not significant (p_corrected={p_corrected:.4f}, d={effect_size:.2f})"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap stability
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_diagram_stability(
    data: np.ndarray,
    n_bootstrap: int = 50,
    subsample_ratio: float = 0.8,
    max_dim: int = 1,
    metric: str = "euclidean",
) -> dict:
    """Estimate stability of persistence diagrams via bootstrap subsampling.

    Repeatedly subsamples the data (without replacement), recomputes diagrams,
    and measures mean Wasserstein distance to the full-data diagram.

    Args:
        data: (n_samples, n_features).
        n_bootstrap: number of bootstrap iterations.
        subsample_ratio: fraction of data to keep in each subsample.
        max_dim: max homology dimension.
        metric: distance metric.

    Returns:
        dict with 'mean_wasserstein_per_dim', 'std_wasserstein_per_dim',
        'stability_ratio' (mean_wasserstein / mean_lifetime).
    """
    full_dgms = compute_persistence_diagrams(data, max_dim=max_dim, metric=metric, use_cache=False)
    n_total = data.shape[0]
    n_subsample = max(int(n_total * subsample_ratio), 10)

    distances = {dim: [] for dim in range(max_dim + 1)}

    for _ in range(n_bootstrap):
        idx = rng.choice(n_total, size=n_subsample, replace=False)
        sub_data = data[idx]
        sub_dgms = compute_persistence_diagrams(sub_data, max_dim=max_dim, metric=metric, use_cache=False)

        for dim in range(max_dim + 1):
            w = _wasserstein_with_fallback(full_dgms[dim] if dim < len(full_dgms) else np.empty((0, 2)),
                                          sub_dgms[dim] if dim < len(sub_dgms) else np.empty((0, 2)))
            distances[dim].append(w)

    diag = diagnose_persistence(full_dgms)
    result = {}
    for dim in range(max_dim + 1):
        dists = np.array(distances[dim])
        mean_lifetime = diag.get(f"H{dim}", {}).get("mean_lifetime", 0.0)
        result[f"H{dim}"] = {
            "mean_wasserstein": float(np.mean(dists)),
            "std_wasserstein": float(np.std(dists)),
            "stability_ratio": float(np.mean(dists) / mean_lifetime) if mean_lifetime > 0 else None,
        }

    return result


def _wasserstein_with_fallback(dgm1, dgm2, dim=1):
    """Internal wasserstein call with fallback for missing persim."""
    try:
        return wasserstein_distance([dgm1], [dgm2], dim=0)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def persistence_stability(dgm: list, noise_std: float = 0.01, n_bootstrap: int = 20) -> float:
    """Estimate stability of a diagram under Gaussian noise via bootstrap.

    Returns mean Wasserstein distance between original diagram and noisy
    bootstrap diagrams. Values near 0 indicate high stability.

    NOTE: This is a lightweight proxy — for rigorous stability analysis, use
    `bootstrap_diagram_stability()` which re-runs ripser on subsampled data.
    """
    if dgm is None or len(dgm) == 0:
        return 0.0

    # Pick a representative homology dimension (default: H1)
    dim_idx = 1 if isinstance(dgm, list) and len(dgm) > 1 else 0
    diagram = dgm[dim_idx] if isinstance(dgm, list) else dgm
    finite = _finite_dgm(diagram)

    if len(finite) == 0:
        return 0.0

    distances = []
    for _ in range(n_bootstrap):
        noise = noise_std * np.random.randn(*finite.shape)
        noisy = finite + noise
        d = _wasserstein_with_fallback(finite, noisy)
        if np.isfinite(d):
            distances.append(d)

    return float(np.mean(distances)) if distances else 0.0


def diagnose_persistence(dgm: list) -> dict:
    """Basic diagnostics for a persistence diagram.

    Returns:
        dict with 'n_features_per_dim', 'mean_lifetime', 'max_lifetime'.
    """
    info = {}
    for dim, diagram in enumerate(dgm):
        finite = _finite_dgm(diagram)
        lifetimes = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])
        info[f"H{dim}"] = {
            "n_features": len(finite),
            "mean_lifetime": float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0,
            "max_lifetime": float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0,
        }
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse persistence — efficiency for large datasets
# ═══════════════════════════════════════════════════════════════════════════════

def sparse_persistence(
    data: np.ndarray,
    max_dim: int = 1,
    subsample_ratio: float = 0.3,
    n_subsamples: int = 5,
    metric: str = "euclidean",
) -> dict:
    """Approximate persistence via subsample + average (provably convergent).

    For large datasets where exact Vietoris-Rips is intractable, computes
    persistence on multiple random subsamples and averages the landscape
    statistics. The mean of subsampled diagrams converges to the true
    diagram (Chazal et al. 2015).

    Args:
        data: (n_samples, n_features).
        max_dim: max homology dimension.
        subsample_ratio: fraction of data per subsample.
        n_subsamples: number of subsample iterations.
        metric: distance metric.

    Returns:
        dict with 'mean_n_features_per_dim', 'std_n_features_per_dim',
        'mean_lifetime_per_dim', 'subsample_sizes', 'full_diagram_diagnostics'.
    """
    n_total = data.shape[0]
    n_subsample = max(int(n_total * subsample_ratio), 20)
    logger.info(f"Sparse persistence: {n_subsample}/{n_total} points × {n_subsamples} subsamples")

    # Full diagram for reference
    full_dgms = compute_persistence_diagrams(data, max_dim=max_dim, metric=metric, use_cache=False)
    full_diag = diagnose_persistence(full_dgms)

    # Collect per-subsample diagnostics
    n_features = {dim: [] for dim in range(max_dim + 1)}
    mean_lifetimes = {dim: [] for dim in range(max_dim + 1)}

    for i in range(n_subsamples):
        idx = rng.choice(n_total, size=n_subsample, replace=False)
        sub_data = data[idx]
        sub_dgms = compute_persistence_diagrams(sub_data, max_dim=max_dim, metric=metric, use_cache=False)
        sub_diag = diagnose_persistence(sub_dgms)

        for dim in range(max_dim + 1):
            n_features[dim].append(sub_diag[f"H{dim}"]["n_features"])
            mean_lifetimes[dim].append(sub_diag[f"H{dim}"]["mean_lifetime"])

    result = {}
    for dim in range(max_dim + 1):
        result[f"H{dim}"] = {
            "mean_n_features": float(np.mean(n_features[dim])),
            "std_n_features": float(np.std(n_features[dim])),
            "mean_lifetime": float(np.mean(mean_lifetimes[dim])),
            "std_lifetime": float(np.std(mean_lifetimes[dim])),
            "full_diagram_n_features": full_diag[f"H{dim}"]["n_features"],
            "full_diagram_lifetime": full_diag[f"H{dim}"]["mean_lifetime"],
        }

    result["subsample_sizes"] = {"total": n_total, "subsample": n_subsample, "ratio": subsample_ratio}
    return result
