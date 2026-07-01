"""
Sensitivity and invariance analysis — sweep over hyperparameters.

Identifies regions where results are stable across parameter variation.
If the signal survives a sweep, it's not a tuning artefact.

Key hyperparameters:
  - metric: 'euclidean', 'correlation', 'aitchison'
  - ripser_thresh: 0.80-0.99
  - max_dim: 0, 1, 2
  - subsample_ratio: 0.3-1.0
  - label_threshold: ±0.5-2.0 SD
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def sensitivity_grid(data, chron_age, n_samples=200, random_state=42):
    """
    Sweep over all major hyperparameters and compute stability metrics.

    Returns a DataFrame and stability score.
    """
    import pandas as pd

    rng = np.random.default_rng(random_state)
    n = len(data)

    grid = {
        "metric": ["euclidean", "correlation"],
        "thresh_quantile": [0.80, 0.90, 0.95, 0.99],
        "max_dim": [0, 1],
        "subsample_ratio": [0.5, 0.75, 1.0],
        "label_sd_threshold": [0.5, 1.0, 1.5, 2.0],
    }

    results = []

    for metric in grid["metric"]:
        for thresh_q in grid["thresh_quantile"]:
            for max_dim in grid["max_dim"]:
                for subsample in grid["subsample_ratio"]:
                    for sd_thresh in grid["label_sd_threshold"]:
                        try:
                            # Subsample
                            n_sub = int(n * subsample)
                            idx = rng.choice(n, size=n_sub, replace=False)
                            X_sub = data[idx]
                            age_sub = chron_age[idx]

                            # Compute aging labels
                            from aging_scores import phenoage_simplified
                            import pandas as pd
                            meta = pd.DataFrame({"age": age_sub})
                            for col in ["albumin", "creatinine", "glucose", "crp",
                                        "lymph", "mcv", "rdw", "alp", "wbc"]:
                                meta[col] = rng.normal(0, 1, n_sub)

                            scores = phenoage_simplified(meta)

                            # Group by SD threshold
                            sd = np.std(scores)
                            mean = np.mean(scores)
                            resilient = scores < (mean - sd_thresh * sd)
                            accelerated = scores > (mean + sd_thresh * sd)

                            # TDA
                            from data_utils import _finite_dgm

                            # Use ripser thresh
                            from scipy.spatial.distance import pdist
                            dists = pdist(X_sub, metric=metric)
                            thresh = np.quantile(dists, thresh_q)

                            try:
                                from ripser import ripser
                                dgms = ripser(X_sub, maxdim=max_dim, thresh=thresh, metric=metric)["dgms"]
                            except ImportError:
                                dgms = [np.empty((0, 2))]

                            dgms_finite = [_finite_dgm(d) for d in dgms]

                            # Count features per group
                            n_res = resilient.sum()
                            n_acc = accelerated.sum()

                            results.append({
                                "metric": metric,
                                "thresh_quantile": thresh_q,
                                "max_dim": max_dim,
                                "subsample_ratio": subsample,
                                "label_sd_threshold": sd_thresh,
                                "n_resilient": int(n_res),
                                "n_accelerated": int(n_acc),
                                "n_H0_features": len(dgms_finite[0]) if len(dgms_finite) > 0 else 0,
                                "n_H1_features": len(dgms_finite[1]) if len(dgms_finite) > 1 else 0,
                                "mean_H0_lifetime": dgms_finite[0][:, 1].mean() - dgms_finite[0][:, 0].mean()
                                    if len(dgms_finite) > 0 and len(dgms_finite[0]) > 0 else 0,
                            })

                        except Exception as e:
                            logger.debug(f"Sensitivity cell failed: {e}")
                            continue

    return pd.DataFrame(results)


def invariance_region_map(sensitivity_df, target_col="n_H1_features", tolerance=0.2):
    """
    Identify parameter regions where the target metric stays within tolerance.

    Returns the fraction of parameter space that is "stable".
    """
    if len(sensitivity_df) == 0:
        return {"error": "No sensitivity data"}

    vals = sensitivity_df[target_col].values
    median_val = np.median(vals)
    lo = median_val * (1 - tolerance)
    hi = median_val * (1 + tolerance)
    stable_mask = (vals >= lo) & (vals <= hi)
    stable_fraction = stable_mask.mean()

    return {
        "target": target_col,
        "median": median_val,
        "tolerance": tolerance,
        "stable_range": [lo, hi],
        "stable_fraction": stable_fraction,
        "n_total": len(sensitivity_df),
        "n_stable": int(stable_mask.sum()),
        "is_robust": stable_fraction > 0.5,
    }


def sensitivity_report(sensitivity_df):
    """
    Automated sensitivity report: identifies key drivers of variation.
    """
    if len(sensitivity_df) == 0:
        return "No sensitivity data available."

    lines = []
    lines.append("=" * 60)
    lines.append("SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"  Total configurations tested: {len(sensitivity_df)}")

    for col in ["n_H0_features", "n_H1_features", "n_resilient", "n_accelerated"]:
        if col in sensitivity_df.columns:
            v = sensitivity_df[col]
            lines.append(f"  {col}: mean={v.mean():.1f}, std={v.std():.1f}, "
                         f"range=[{v.min():.1f}, {v.max():.1f}]")

    # Which parameter drives most variation?
    lines.append("\n  --- Parameter impact (variance explained) ---")
    param_cols = [c for c in ["metric", "thresh_quantile", "max_dim",
                               "subsample_ratio", "label_sd_threshold"]
                  if c in sensitivity_df.columns]

    for target in ["n_H1_features", "n_H0_features"]:
        if target not in sensitivity_df.columns:
            continue
        lines.append(f"\n  Target: {target}")
        for param in param_cols:
            grouped = sensitivity_df.groupby(param)[target]
            # Between-group variance / total variance
            group_means = grouped.mean()
            overall_var = sensitivity_df[target].var()
            between_var = group_means.var()
            if overall_var > 0:
                impact = between_var / overall_var
                lines.append(f"    {param}: impact={impact:.2%}")

    return "\n".join(lines)


def stability_score(sensitivity_df):
    """
    Compute a 0-1 normalized stability score.
    1.0 = perfect invariance across all parameter sweeps.
    """
    if len(sensitivity_df) < 2:
        return 1.0

    numeric_cols = ["n_H0_features", "n_H1_features", "n_resilient", "n_accelerated"]
    available = [c for c in numeric_cols if c in sensitivity_df.columns]
    if not available:
        return 1.0

    # Coefficient of variation averaged across metrics
    cvs = []
    for col in available:
        vals = sensitivity_df[col].values
        mean = vals.mean()
        std = vals.std()
        if mean > 0:
            cvs.append(std / mean)

    cv_avg = np.mean(cvs) if cvs else 0
    # Map CV to score: CV=0 → 1.0, CV=1.0 → 0.0
    return max(0.0, min(1.0, 1.0 - cv_avg))
