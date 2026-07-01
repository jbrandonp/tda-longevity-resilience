"""
Longitudinal persistence + Dask support for large-scale TDA.

Handles time-series multi-omics data with:
  - Sliding window persistence (track topological features over time)
  - Takens embedding → persistence (attractor reconstruction)
  - Dask-backed parallel computation for large cohorts
"""
import numpy as np

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


def sliding_window_persistence(data, window_size=10, step=5, max_dim=1):
    """
    Compute persistence diagrams for each sliding window over time.

    Parameters:
      data: (n_timepoints, n_features) — longitudinal omics
      window_size: samples per window
      step: stride between windows
      max_dim: max homology dimension

    Returns:
      list of dict: [{time_mid: float, dgm: list, diag: dict}, ...]
    """
    n = len(data)
    if n < window_size:
        logger.warning(f"Data length {n} < window {window_size}")
        return []

    results = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = data[start:end]
        time_mid = (start + end) / 2

        try:
            from .tda_utils import compute_persistence_diagrams, diagnose_persistence
        except ImportError:
            from tda_utils import compute_persistence_diagrams, diagnose_persistence

        dgm = compute_persistence_diagrams(window, max_dim=max_dim, use_cache=False)
        diag = diagnose_persistence(dgm)

        results.append({
            "time_mid": float(time_mid),
            "time_start": int(start),
            "time_end": int(end),
            "diagram": dgm,
            "diagnostics": diag,
        })

    return results


def takens_embedding_persistence(time_series, embedding_dim=3, tau=1, max_dim=1):
    """
    Takens embedding → persistence diagram.

    Reconstructs the attractor topology from a single time series,
    then computes persistent homology on the embedded point cloud.

    Parameters:
      time_series: 1D array (n_timepoints,)
      embedding_dim: dimension of the Takens embedding
      tau: time delay
      max_dim: max homology dimension

    Returns:
      dict with 'embedding', 'diagram', 'diagnostics'
    """
    n = len(time_series)
    m = embedding_dim
    N = n - (m - 1) * tau

    if N < 10:
        logger.warning(f"Not enough points for Takens embedding: {n} → {N}")
        return {"embedding": np.empty((0, m)), "diagram": [], "diagnostics": {}}

    embedded = np.zeros((N, m))
    for i in range(m):
        embedded[:, i] = time_series[i * tau : i * tau + N]

    try:
        from .tda_utils import compute_persistence_diagrams, diagnose_persistence
    except ImportError:
        from tda_utils import compute_persistence_diagrams, diagnose_persistence

    dgm = compute_persistence_diagrams(embedded, max_dim=max_dim, use_cache=False)
    diag = diagnose_persistence(dgm)

    return {
        "embedding": embedded,
        "embedding_dim": m,
        "tau": tau,
        "diagram": dgm,
        "diagnostics": diag,
    }


def dask_parallel_persistence(datasets, max_dim=1, n_workers=4):
    """
    Compute persistence on multiple datasets in parallel using Dask.

    Parameters:
      datasets: dict {name: array (n×d)} — multiple omics layers or subjects
      max_dim: max homology dimension
      n_workers: Dask worker count

    Returns:
      dict {name: {"diagram": list, "diagnostics": dict}}
    """
    try:
        import dask
        from dask.distributed import Client
    except ImportError:
        logger.warning("Dask not installed — falling back to sequential")
        from .tda_utils import compute_persistence_diagrams, diagnose_persistence
        results = {}
        for name, data in datasets.items():
            dgm = compute_persistence_diagrams(data, max_dim=max_dim, use_cache=False)
            results[name] = {"diagram": dgm, "diagnostics": diagnose_persistence(dgm)}
        return results

    from .tda_utils import compute_persistence_diagrams, diagnose_persistence

    @dask.delayed
    def _compute_one(name, data):
        dgm = compute_persistence_diagrams(data, max_dim=max_dim, use_cache=False)
        return name, {"diagram": dgm, "diagnostics": diagnose_persistence(dgm)}

    with Client(n_workers=n_workers, processes=False) as client:
        tasks = [_compute_one(name, data) for name, data in datasets.items()]
        results = dict(dask.compute(*tasks))

    return results


def time_series_topological_trajectory(data, window_size=20, step=10):
    """
    Track topological features over time in longitudinal data.

    Returns a summary of how H0, H1 counts and lifetimes evolve across time.
    """
    windows = sliding_window_persistence(data, window_size=window_size, step=step, max_dim=1)

    if not windows:
        return pd.DataFrame()

    rows = []
    for w in windows:
        diag = w["diagnostics"]
        rows.append({
            "time_mid": w["time_mid"],
            "H0_count": diag["H0"]["n_features"],
            "H0_lifetime": diag["H0"]["mean_lifetime"],
            "H1_count": diag.get("H1", {}).get("n_features", 0),
            "H1_lifetime": diag.get("H1", {}).get("mean_lifetime", 0),
            "H0_max": diag["H0"]["max_lifetime"],
            "H1_max": diag.get("H1", {}).get("max_lifetime", 0),
        })

    return pd.DataFrame(rows)
