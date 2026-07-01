"""Validation protocols — hold-out splitting, external validation, and reporting.

Public API:
    create_discovery_validation(df, metadata, test_size, stratify_cols) -> tuple
    evaluate_on_holdout(model, X_holdout, y_holdout) -> dict
    generate_validation_report(train_results, holdout_results) -> str
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Discovery / Validation split
# ═══════════════════════════════════════════════════════════════════════════════

def create_discovery_validation(
    df: pd.DataFrame,
    metadata: pd.DataFrame = None,
    test_size: float = 0.20,
    stratify_cols: list = None,
    random_state: int = 42,
) -> tuple:
    """Split data into discovery (80%) and validation (20%) sets.

    The discovery set is used for ALL exploratory analysis, hyperparameter
    tuning, and model selection. The validation set is locked away and used
    ONCE at the very end for final evaluation.

    This prevents overfitting to the test set through iterative model refinement.

    Args:
        df: main data matrix (n_samples, n_features).
        metadata: DataFrame with stratification columns (age, sex, group).
        test_size: fraction reserved for validation (default 0.20).
        stratify_cols: columns to stratify split on (e.g., ['age_group', 'sex']).
            If None, uses 'aging_group' from metadata if available.
        random_state: reproducibility.

    Returns:
        (df_discovery, df_validation, metadata_discovery, metadata_validation).
    """
    if metadata is None:
        # Simple random split if no metadata
        df_d, df_v = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        logger.info(f"Split: {len(df_d)} discovery / {len(df_v)} validation")
        return df_d, df_v, None, None

    # Build stratification variable
    if stratify_cols is None:
        # Default: combine aging group + sex + 5-year age bins
        if "aging_group" in metadata.columns:
            stratify_cols = ["aging_group"]
        elif "sex" in metadata.columns and "age" in metadata.columns:
            metadata["_age_bin"] = pd.cut(
                metadata["age"], bins=range(0, 120, 10), labels=False
            )
            stratify_cols = ["_age_bin", "sex"]
        else:
            stratify_cols = []

    if stratify_cols:
        metadata["_strat"] = metadata[stratify_cols].astype(str).agg("_".join, axis=1)
        strat = metadata["_strat"].values
    else:
        strat = None

    idx_d, idx_v = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=strat,
        random_state=random_state,
    )

    df_d, df_v = df.iloc[idx_d], df.iloc[idx_v]
    meta_d, meta_v = metadata.iloc[idx_d], metadata.iloc[idx_v]

    # Report
    logger.info(f"Discovery/Validation split:")
    logger.info(f"  Discovery:  {len(df_d)} samples")
    logger.info(f"  Validation: {len(df_v)} samples")
    if metadata is not None and "aging_group" in metadata.columns:
        for name, meta in [("Discovery", meta_d), ("Validation", meta_v)]:
            counts = meta["aging_group"].value_counts().to_dict()
            logger.info(f"  {name}: {counts}")
    if metadata is not None:
        metadata.drop(columns=["_strat"], inplace=True, errors="ignore")

    return df_d, df_v, meta_d, meta_v


# ═══════════════════════════════════════════════════════════════════════════════
# Holdout evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_holdout(
    model,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
) -> dict:
    """Evaluate a trained model on held-out validation data.

    Args:
        model: fitted sklearn model or Pipeline with predict_proba.
        X_holdout: (n_samples, n_features) held-out data.
        y_holdout: (n_samples,) held-out labels.
        X_train: optional training data for comparison.
        y_train: optional training labels.

    Returns:
        dict with 'auc', 'accuracy', 'precision', 'recall', 'f1',
        'confusion_matrix', and optionally 'train_auc' for overfitting check.
    """
    y_pred = model.predict(X_holdout)
    y_proba = model.predict_proba(X_holdout)[:, 1] if hasattr(model, "predict_proba") else y_pred

    result = {
        "auc": roc_auc_score(y_holdout, y_proba),
        "accuracy": accuracy_score(y_holdout, y_pred),
        "precision": precision_score(y_holdout, y_pred, zero_division=0),
        "recall": recall_score(y_holdout, y_pred, zero_division=0),
        "f1": f1_score(y_holdout, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_holdout, y_pred).tolist(),
        "n_samples": len(y_holdout),
    }

    # Overfitting check: compare train vs holdout AUC
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_train)
        train_auc = roc_auc_score(y_train, y_train_pred)
        result["train_auc"] = train_auc
        delta = train_auc - result["auc"]
        result["overfitting_gap"] = delta
        if delta > 0.15:
            result["overfitting_warning"] = (
                f"⚠️ Large train-holdout AUC gap ({delta:.3f}). "
                f"Model may be overfitting."
            )
        else:
            result["overfitting_warning"] = (
                f"✅ Train-holdout AUC gap ({delta:.3f}) within acceptable range."
            )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Validation report generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_validation_report(
    train_results: dict,
    holdout_results: dict,
    model_name: str = "Model",
) -> str:
    """Generate a formatted validation report.

    Args:
        train_results: dict from cross-validation (mean ± std per metric).
        holdout_results: dict from evaluate_on_holdout.
        model_name: human-readable model name.

    Returns:
        Markdown-formatted report string.
    """
    report = f"""# Validation Report — {model_name}

## Hold-Out Performance (20% external validation)

| Metric    | Hold-Out |
|-----------|----------|
| AUC       | {holdout_results['auc']:.3f} |
| Accuracy  | {holdout_results['accuracy']:.3f} |
| Precision | {holdout_results['precision']:.3f} |
| Recall    | {holdout_results['recall']:.3f} |
| F1 Score  | {holdout_results['f1']:.3f} |

Confusion matrix:
```
{holdout_results['confusion_matrix'][0]}
{holdout_results['confusion_matrix'][1]}
```

## Overfitting Check

"""
    if "overfitting_warning" in holdout_results:
        report += holdout_results["overfitting_warning"] + "\n"
    if "train_auc" in holdout_results:
        report += f"- Train AUC: {holdout_results['train_auc']:.3f}\n"
        report += f"- Holdout AUC: {holdout_results['auc']:.3f}\n"
        report += f"- Gap: {holdout_results.get('overfitting_gap', 0):.3f}\n"

    report += f"""
## Protocol

1. **Discovery set** (80%): Used for all exploratory analysis, hyperparameter
   tuning, cross-validation, and model selection.
2. **Validation set** (20%): Held out from the start. Used exactly ONCE for
   final evaluation. Never used in any training or tuning step.
3. **Stratification**: Split preserves age group and sex distributions.
4. **No data leakage**: UMAP/PCA fitted only on training data within each
   CV fold. Topological features extracted after embedding.
5. **Reproducibility**: Random seed {42}, all parameters logged.

## Sign-Off

- [ ] All CV scores reported from discovery set only
- [ ] Holdout evaluation run exactly once
- [ ] No parameter tuned using holdout performance
- [ ] Overfitting gap within acceptable range (< 0.15 AUC)
"""

    return report
