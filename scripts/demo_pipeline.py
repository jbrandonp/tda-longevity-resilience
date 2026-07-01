#!/usr/bin/env python
"""Demo pipeline: synthetic data → persistence → Mapper → ML → publication figure.

One command to prove TDA-Longevity-Resilience works end-to-end:
    python scripts/demo_pipeline.py

Output:
    results/figures/demo_pipeline.png  — 4-panel publication-ready figure
    results/tables/demo_metrics.csv    — classification metrics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging_config import get_logger
from src.data_utils import generate_synthetic_multimics, preprocess_omics, integrate_multiomics
from src.tda_utils import compute_persistence_diagrams, diagnose_persistence
from src.features import PersistenceImageTransformer
from src.ml_utils import build_topological_pipeline, evaluate_topological_model
from src.metrics import get_distance_matrix
from sklearn.ensemble import RandomForestClassifier

logger = get_logger(__name__)


def main():
    OUTPUT_DIR = Path("results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: Generate synthetic data with known topology
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  STEP 1: Generating synthetic multi-omics data (circle topology)")
    logger.info("=" * 60)

    data = generate_synthetic_multimics(n_samples=200, n_features=50,
                                         topology_type="circle", noise=0.05)
    X = data["transcriptomics"]
    labels = data["labels"]

    # Binary classification: resilient vs accelerated
    mask = (labels == "resilient") | (labels == "accelerated")
    X_bin = X[mask.values]
    y_bin = (labels[mask.values] == "resilient").astype(int).values
    logger.info(f"  {X_bin.shape[0]} samples, {X_bin.shape[1]} features, "
                f"{y_bin.sum()} resilient, {(1-y_bin).sum()} accelerated")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: Persistent homology
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  STEP 2: Computing persistent homology")
    logger.info("=" * 60)

    dgms = compute_persistence_diagrams(X_bin, max_dim=1, metric="euclidean")
    diag = diagnose_persistence(dgms)
    for dim_key, info in diag.items():
        logger.info(f"  {dim_key}: {info['n_features']} features, "
                     f"mean_lifetime={info['mean_lifetime']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: Extract topological features
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  STEP 3: Extracting persistence images")
    logger.info("=" * 60)

    pi = PersistenceImageTransformer(spread=0.1, pixels=[20, 20])
    X_topo = pi.transform([dgms] * len(y_bin))  # same diagram for all samples (demo)
    logger.info(f"  Persistence images: {X_topo.shape}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 4: ML classification
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  STEP 4: Training topological classifier")
    logger.info("=" * 60)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipe = build_topological_pipeline(clf)
    results = evaluate_topological_model(pipe, X_topo, y_bin, cv=5)

    for metric, vals in results.items():
        logger.info(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")

    # Save metrics
    rows = [{"metric": m, "mean": v["mean"], "std": v["std"]} 
            for m, v in results.items()]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "tables" / "demo_metrics.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5: Generate publication-ready figure
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  STEP 5: Generating publication figure")
    logger.info("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("TDA-Longevity-Resilience — Demo Pipeline", 
                     fontsize=14, fontweight="bold")

        # Panel A: Persistence barcode (H1)
        ax = axes[0, 0]
        dgm_h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        finite = dgm_h1[np.isfinite(dgm_h1[:, 1])] if len(dgm_h1) > 0 else np.empty((0, 2))
        for i, (birth, death) in enumerate(finite):
            ax.plot([birth, death], [i, i], "b-", lw=2)
        ax.set_xlabel("Filtration parameter")
        ax.set_ylabel("Feature index")
        ax.set_title(f"A) Persistence Barcode H1 ({len(finite)} cycles)")
        ax.axvline(x=np.median(finite[:, 0]) if len(finite) > 0 else 0, 
                   color="gray", linestyle="--", alpha=0.5)

        # Panel B: Persistence diagram
        ax = axes[0, 1]
        for dim, dgm in enumerate(dgms[:2]):
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1], s=20, alpha=0.7, 
                          label=f"H{dim} ({len(finite)})")
        max_val = max(np.max(dgm[np.isfinite(dgm[:, 1])]) for dgm in dgms[:2] if len(dgm) > 0)
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title("B) Persistence Diagram")
        ax.legend()

        # Panel C: Feature importance
        ax = axes[1, 0]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            top_n = min(20, len(importances))
            top_idx = np.argsort(importances)[-top_n:]
            ax.barh(range(top_n), importances[top_idx])
            ax.set_xlabel("Importance")
            ax.set_title(f"C) Top {top_n} Persistence Image Features")

        # Panel D: Classification metrics
        ax = axes[1, 1]
        metrics_names = list(results.keys())[:5]
        metrics_vals = [results[m]["mean"] for m in metrics_names]
        ax.bar(metrics_names, metrics_vals, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"][:len(metrics_names)])
        ax.set_ylabel("Score")
        ax.set_title("D) Classification Performance")
        ax.set_ylim(0, 1.05)
        for i, v in enumerate(metrics_vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / "figures" / "demo_pipeline.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Figure saved: {fig_path}")

    except ImportError:
        logger.warning("matplotlib not available — skipping figure generation")

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  DEMO COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Data: {X_bin.shape[0]} samples × {X_bin.shape[1]} features (circle topology)")
    logger.info(f"  H1 cycles: {diag.get('H1', {}).get('n_features', 0)}")
    logger.info(f"  Best AUC: {results.get('auc', {}).get('mean', 0):.4f}")
    logger.info(f"  Outputs: {OUTPUT_DIR}/figures/demo_pipeline.png")
    logger.info(f"           {OUTPUT_DIR}/tables/demo_metrics.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
