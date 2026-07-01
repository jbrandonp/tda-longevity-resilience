"""Data loading, preprocessing, normalization, and synthetic data generation.

Public API:
    load_dataset(name: str) -> pd.DataFrame
    preprocess_omics(df, method='standard') -> np.ndarray
    integrate_multiomics(omics_dict) -> np.ndarray
    assign_groups_from_tian_score(metadata, col='tian_score') -> pd.Series
    generate_synthetic_multimics(n_samples, topology_type) -> np.ndarray
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers (used across multiple src/ modules)
# ═══════════════════════════════════════════════════════════════════════════════

def _finite_dgm(dgm):
    """Return only finite (birth, death) pairs. Shared utility — single source of truth."""
    if dgm is None or len(dgm) == 0:
        return np.empty((0, 2))
    finite = np.isfinite(dgm[:, 1])
    return dgm[finite]

try:
    from .config import RANDOM_SEED
except ImportError:
    from config import RANDOM_SEED
try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


rng = np.random.default_rng(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(name: str) -> pd.DataFrame:
    """Load a named dataset (stub — override with real data paths)."""
    path = f"data/raw/{name}.csv"
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {name}: {df.shape}")
        return df
    except FileNotFoundError:
        logger.info(f"Dataset '{name}' not found at {path} — returning empty DataFrame.")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_omics(
    df: pd.DataFrame,
    method: str = "standard",
    n_components: int = None,
) -> np.ndarray:
    """Normalise and optionally reduce dimensionality.

    Args:
        df: (n_samples, n_features) — omics data.
        method: 'standard' (z-score), 'robust' (median/IQR), or 'log' (log1p + standard).
        n_components: if provided, apply PCA to this many components (or float for variance ratio).

    Returns:
        np.ndarray of shape (n_samples, n_features_reduced).
    """
    data = df.select_dtypes(include=[np.number]).values

    if method == "standard":
        data = StandardScaler().fit_transform(data)
    elif method == "robust":
        data = RobustScaler().fit_transform(data)
    elif method == "log":
        data = np.log1p(data.clip(min=0))
        data = StandardScaler().fit_transform(data)
    else:
        raise ValueError(f"Unknown method: {method}")

    if n_components is not None:
        if isinstance(n_components, float):
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            data = pca.fit_transform(data)
        else:
            pca = PCA(n_components=min(n_components, data.shape[1]), random_state=RANDOM_SEED)
            data = pca.fit_transform(data)
        logger.info(f"PCA reduced from {df.shape[1]} to {data.shape[1]} components")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-omics integration
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_multiomics(
    omics_dict: dict,
    method: str = "concat",
    combat_batch: np.ndarray = None,
    n_factors: int = 10,
) -> np.ndarray:
    """Integrate multiple omics layers into a single matrix.

    Methods:
        'concat': Standardize each layer separately, then concatenate.
            Robust default — works with any number of layers.
        'concat_raw': Concatenate without scaling (use if data is pre-normalized).
        'mofa': Multi-Omics Factor Analysis via mofapy2 (requires `pip install mofapy2`).
            Extracts latent factors capturing shared variance across layers.

    Args:
        omics_dict: {layer_name: np.ndarray (n_samples, n_features)}.
            All arrays MUST share the same n_samples and be in the same order.
        method: integration method ('concat', 'concat_raw', 'mofa').
        combat_batch: optional batch labels for ComBat harmonization (requires pycombat).
            If provided and method='concat', applies ComBat per layer before concatenation.
        n_factors: number of latent factors for MOFA.

    Returns:
        integrated: np.ndarray of shape (n_samples, total_features).
    """
    # Validate input alignment
    arrays = list(omics_dict.values())
    array_names = list(omics_dict.keys())
    n_samples = arrays[0].shape[0]
    for name, arr in zip(array_names, arrays):
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Layer '{name}' has {arr.shape[0]} samples, "
                f"expected {n_samples}"
            )

    if method == "concat_raw":
        integrated = np.hstack(arrays)
        logger.info(f"Integrated {len(arrays)} omics layers → {integrated.shape} (raw)")
        return integrated

    elif method == "concat":
        scaled_data = []

        for name, arr in zip(array_names, arrays):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(arr)

            # Optional: ComBat batch harmonization
            if combat_batch is not None and len(np.unique(combat_batch)) > 1:
                try:
                    from pycombat import ComBat
                    scaled = ComBat(scaled.T, combat_batch).T
                    logger.info(f"ComBat harmonization applied to {name}")
                except ImportError:
                    logger.info(f"pycombat not installed — skipping batch correction for {name}")

            scaled_data.append(scaled)

        integrated = np.hstack(scaled_data)
        total_features = integrated.shape[1]
        logger.info(f"Integrated {len(arrays)} omics layers → "
                    f"{n_samples} samples × {total_features} features "
                    f"({' + '.join(f'{a.shape[1]}' for a in arrays)} per layer)")
        return integrated

    elif method == "mofa":
        try:
            import mofapy2.run
        except ImportError:
            raise ImportError(
                "mofapy2 is required for MOFA integration. "
                "Install with: pip install mofapy2"
            )

        # Prepare: transpose to (features, samples) for MOFA
        data_matrices = [arr.T for arr in arrays]

        # Initialize and train MOFA
        model = mofapy2.run.MOFA(n_factors=n_factors, verbose=False)
        model.fit(data_matrices)

        # Extract latent factors (samples × n_factors)
        factors = model.get_factors()["group1"]
        logger.info(f"MOFA integrated {len(arrays)} layers → "
                    f"{n_samples} samples × {n_factors} factors")
        return factors

    else:
        raise ValueError(f"Unknown integration method: {method}. "
                         f"Choose from: concat, concat_raw, mofa")


# ═══════════════════════════════════════════════════════════════════════════════
# Tian score group assignment
# ═══════════════════════════════════════════════════════════════════════════════

def assign_groups_from_tian_score(
    metadata: pd.DataFrame,
    col: str = "tian_score",
    accel_thresh: float = 1.0,
    resil_thresh: float = -1.0,
) -> pd.Series:
    """Assign 'accelerated' / 'resilient' / 'neutral' groups from a Tian score.

    Accelerated:  tian_score > +accel_thresh * std
    Resilient:    tian_score < resil_thresh * std
    Neutral:      everything else

    Reference: Tian 2026 — generational accelerated aging framework.
    """
    scores = metadata[col].values
    std = np.std(scores)
    mean = np.mean(scores)

    labels = np.full(len(scores), "neutral", dtype=object)
    labels[scores > mean + accel_thresh * std] = "accelerated"
    labels[scores < mean + resil_thresh * std] = "resilient"

    counts = pd.Series(labels).value_counts()
    logger.info(f"Group assignment: {counts.to_dict()}")
    return pd.Series(labels, index=metadata.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Santos-Pujol supercentenarian comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_with_santos_pujol(
    dgm_resilient_pool: list,
    dgm_santos_pujol: list,
    dim: int = 1,
) -> dict:
    """Compare a supercentenarian persistence diagram against a resilient pool.

    The Santos-Pujol individual provides an extreme example of resilience.
    This function measures topological similarity via Wasserstein distance.

    Args:
        dgm_resilient_pool: persistence diagrams for the resilient group.
        dgm_santos_pujol: persistence diagram for the supercentenarian.
        dim: homology dimension.

    Returns:
        dict with 'wasserstein_to_pool_mean', 'wasserstein_to_pool_std',
        'z_score' (how many std away from pool mean), 'percentile_rank'.
    """
    from .tda_utils import wasserstein_distance

    distances = []
    for dgm_r in dgm_resilient_pool:
        d = wasserstein_distance(dgm_r, dgm_santos_pujol, dim=dim)
        distances.append(d)

    distances = np.array(distances)
    mu = float(np.mean(distances))
    sigma = float(np.std(distances))
    z = (0.0 - mu) / sigma if sigma > 0 else 0.0  # distance to self = 0

    # Percentile: what fraction of resilient pool is closer to itself than Santos-Pujol?
    pairwise_distances = []
    for i, dgm_a in enumerate(dgm_resilient_pool):
        for j, dgm_b in enumerate(dgm_resilient_pool):
            if i < j:
                pairwise_distances.append(wasserstein_distance(dgm_a, dgm_b, dim=dim))

    pairwise_distances = np.array(pairwise_distances)
    percentile = float(np.mean(distances <= np.percentile(pairwise_distances, 90)))

    return {
        "wasserstein_to_pool_mean": mu,
        "wasserstein_to_pool_std": sigma,
        "z_score": z,
        "percentile_rank": round(percentile, 4),
        "n_pool_samples": len(dgm_resilient_pool),
        "interpretation": (
            "Santos-Pujol is topologically similar to the resilient pool"
            if percentile > 0.5 else
            "Santos-Pujol exhibits a distinct topological signature"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_multimics(
    n_samples: int = 200,
    topology_type: str = "circle",
    noise: float = 0.1,
    n_features: int = 50,
) -> dict:
    """Generate synthetic multi-omics data with known topological structure.

    Args:
        n_samples: number of samples (individuals).
        topology_type: one of 'circle', 'torus', 'figure8', 'sphere', 'noise'.
        noise: additive Gaussian noise std.
        n_features: ambient dimension.

    Returns:
        dict with 'transcriptomics', 'metabolomics', 'epigenomics' arrays,
        'labels' (ground-truth groups), and 'metadata' DataFrame.
    """
    # Base topology
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    if topology_type == "circle":
        base = np.column_stack([np.cos(t), np.sin(t)])
    elif topology_type == "torus":
        u, v = np.meshgrid(t[:50], t[:50])
        base = np.column_stack([u.ravel(), v.ravel()])[:n_samples]
    elif topology_type == "figure8":
        base = np.column_stack([np.sin(t), np.sin(2 * t)])
    elif topology_type == "sphere":
        phi = np.arccos(1 - 2 * rng.random(n_samples))
        theta = 2 * np.pi * rng.random(n_samples)
        base = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
    elif topology_type == "noise":
        base = rng.standard_normal((n_samples, 3))
    else:
        raise ValueError(f"Unknown topology_type: {topology_type}")

    # Project to high-dimensional space via random projection + noise
    def _lift(arr: np.ndarray, n_feat: int) -> np.ndarray:
        proj = rng.standard_normal((arr.shape[1], n_feat)) * 0.3
        return arr @ proj + noise * rng.standard_normal((arr.shape[0], n_feat))

    transcriptomics = _lift(base, n_features)
    metabolomics = _lift(base, n_features)
    epigenomics = _lift(base, n_features)

    # Assign synthetic Tian scores
    tian_score = np.sum(base, axis=1) + noise * rng.standard_normal(n_samples)
    labels = assign_groups_from_tian_score(
        pd.DataFrame({"tian_score": tian_score}),
        col="tian_score",
        accel_thresh=0.8,
        resil_thresh=-0.8,
    )

    metadata = pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(n_samples)],
        "tian_score": tian_score,
        "group": labels.values,
    })

    logger.info(f"Generated synthetic '{topology_type}' data: {n_samples} samples × {n_features} features")
    return {
        "transcriptomics": transcriptomics,
        "metabolomics": metabolomics,
        "epigenomics": epigenomics,
        "labels": labels,
        "metadata": metadata,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Feature selection — reduce dimensionality before TDA
# ═══════════════════════════════════════════════════════════════════════════════

def select_features(df, method="variance", k=100, **kwargs):
    """Select top-k features to reduce dimensionality before TDA.

    Methods: 'variance' (top-k by variance), 'lasso' (L1-regularized),
             'mutual_info' (mutual information with target), 'random'.
    """
    data = df.select_dtypes(include=[np.number])
    if k >= data.shape[1]:
        return df

    if method == "variance":
        selected = data.var().sort_values(ascending=False).index[:k]
    elif method == "lasso":
        from sklearn.linear_model import LassoCV
        target = kwargs.get("target")
        if target is None:
            raise ValueError("lasso requires target array")
        lasso = LassoCV(cv=3, max_iter=2000, random_state=RANDOM_SEED)
        lasso.fit(data.values, target)
        selected = data.columns[np.argsort(np.abs(lasso.coef_))[::-1][:k]]
    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_regression
        target = kwargs.get("target")
        if target is None:
            raise ValueError("mutual_info requires target array")
        mi = mutual_info_regression(data.values, target, random_state=RANDOM_SEED)
        selected = data.columns[np.argsort(mi)[::-1][:k]]
    elif method == "random":
        selected = data.columns[np.random.choice(data.shape[1], size=k, replace=False)]
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Feature selection ({method}): {data.shape[1]} → {k}")
    return df[selected]
