"""Machine learning pipelines for topological feature classification.

Public API:
    build_topological_pipeline(classifier) -> sklearn.pipeline.Pipeline
    evaluate_topological_model(pipeline, X, y) -> dict
    compare_with_baseline(X_topo, X_classic, y) -> pd.DataFrame
    shap_analysis(pipeline, X, feature_names) -> np.ndarray
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    from .config import CV_FOLDS, ML_RANDOM_STATE, PCA_VARIANCE_RATIO
except ImportError:
    from config import CV_FOLDS, ML_RANDOM_STATE, PCA_VARIANCE_RATIO

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Topological Feature Extractor — UMAP inside CV (NO data leakage)
# ═══════════════════════════════════════════════════════════════════════════════

class TopologicalFeatureExtractor:
    """Extract topological features from data, fitting UMAP only on training data.

    This solves the data leakage problem: UMAP is fit on train split only,
    then applied to test split before computing persistence diagrams.
    Persistence images, landscapes, and Betti curves are computed from
    the embedded data.

    Usage:
        tfe = TopologicalFeatureExtractor(maxdim=1, spread=0.1, pixels=[50, 50])
        tfe.fit(X_train)
        X_train_topo = tfe.transform(X_train)
        X_test_topo = tfe.transform(X_test)
    """

    def __init__(
        self,
        maxdim: int = 1,
        spread: float = 0.1,
        pixels: list = None,
        umap_n_components: int = 2,
        umap_random_state: int = 42,
        ripser_thresh: float = None,
        verbose: bool = False,
    ):
        self.maxdim = maxdim
        self.spread = spread
        self.pixels = pixels or [50, 50]
        self.umap_n_components = umap_n_components
        self.umap_random_state = umap_random_state
        self.ripser_thresh = ripser_thresh
        self.verbose = verbose

        self._umap = None
        self._pim = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y=None):
        """Fit UMAP on training data only."""
        from umap import UMAP
        self._umap = UMAP(
            n_components=self.umap_n_components,
            random_state=self.umap_random_state,
        )
        self._umap.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data: UMAP embed → persistence diagrams → feature vector.

        MUST call fit() first.
        """
        if not self._is_fitted:
            raise RuntimeError("TopologicalFeatureExtractor.fit() must be called first")

        # 1. Embed via fitted UMAP
        X_embedded = self._umap.transform(X)

        # 2. Compute persistence diagram
        import ripser
        dgms = ripser.ripser(
            X_embedded,
            maxdim=self.maxdim,
            thresh=self.ripser_thresh,
        )["dgms"]

        # 3. Vectorize via Persistence Image
        from persim import PersImage
        if self._pim is None:
            self._pim = PersImage(
                spread=self.spread,
                pixels=self.pixels,
                verbose=self.verbose,
            )

        features = []
        for dim in range(self.maxdim + 1):
            if len(dgms[dim]) > 0:
                feats = self._pim.transform(dgms[dim])
            else:
                feats = np.zeros(self.pixels[0] * self.pixels[1])
            features.append(feats.flatten() if hasattr(feats, "flatten") else feats)

        return np.concatenate(features)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit on X, then transform X."""
        self.fit(X, y)
        return self.transform(X)


def prepare_topological_features_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    maxdim: int = 1,
    spread: float = 0.1,
    pixels: list = None,
    classifier=None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Cross-validated topological feature extraction with NO data leakage.

    WARNING — previous versions of this pipeline applied UMAP to the full dataset
    before splitting, causing data leakage and artificially inflated AUC.
    This version fits UMAP ONLY on training splits for each CV fold.

    Args:
        X: (n_samples, n_features) input data matrix.
        y: (n_samples,) binary labels.
        n_splits: number of CV folds.
        maxdim: max homology dimension for persistence.
        spread: PersistenceImage Gaussian spread.
        pixels: PersistenceImage resolution [h, w].
        classifier: sklearn classifier (defaults to RandomForest).
        random_state: reproducibility seed.
        verbose: print per-fold diagnostics.

    Returns:
        dict with 'aucs', 'mean_auc', 'std_auc', 'fold_details'.
    """
    pixels = pixels or [50, 50]
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state
        )

    aucs = []
    fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # CRITICAL: fit UMAP on train only
        tfe = TopologicalFeatureExtractor(
            maxdim=maxdim,
            spread=spread,
            pixels=pixels,
            verbose=False,
        )
        tfe.fit(X_train)

        X_train_topo = tfe.transform(X_train)
        X_test_topo = tfe.transform(X_test)

        # Scale + classify
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])
        pipe.fit(X_train_topo, y_train)
        y_pred = pipe.predict_proba(X_test_topo)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        aucs.append(auc)

        if verbose:
            logger.info(
                f"Fold {fold_idx + 1}/{n_splits}: "
                f"AUC={auc:.4f} "
                f"(train={len(train_idx)}, test={len(test_idx)}, "
                f"topo_dim={X_train_topo.shape[1]})"
            )

        fold_details.append({
            "fold": fold_idx + 1,
            "auc": auc,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "topo_features": X_train_topo.shape[1],
        })

    return {
        "aucs": aucs,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "fold_details": fold_details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pipelines
# ═══════════════════════════════════════════════════════════════════════════════

def build_topological_pipeline(
    classifier=None,
    use_pca: bool = False,
) -> Pipeline:
    """Build a scikit-learn pipeline for topological features.

    Args:
        classifier: sklearn classifier. Defaults to RandomForest.
        use_pca: if True, add PCA step.

    Returns:
        sklearn Pipeline.
    """
    steps = [("scaler", StandardScaler())]

    if use_pca:
        steps.append(("pca", PCA(n_components=PCA_VARIANCE_RATIO, random_state=ML_RANDOM_STATE)))

    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=ML_RANDOM_STATE,
        )

    steps.append(("classifier", classifier))
    return Pipeline(steps)


def evaluate_topological_model(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = CV_FOLDS,
) -> dict:
    """Cross-validated evaluation of a topological ML pipeline.

    Returns:
        dict with mean ± std for AUC, accuracy, precision, recall, F1.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=ML_RANDOM_STATE)
    scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    results = {}
    for metric in scoring:
        scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring=metric)
        results[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist(),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison with classic baseline
# ═══════════════════════════════════════════════════════════════════════════════

def compare_with_baseline(
    X_topo: np.ndarray,
    X_classic: np.ndarray,
    y: np.ndarray,
    classifiers: dict = None,
) -> pd.DataFrame:
    """Compare topological features vs classic features across classifiers.

    Args:
        X_topo: topological feature matrix.
        X_classic: classical feature matrix (e.g., raw genes, PCA).
        y: binary labels.
        classifiers: {name: classifier}. Defaults to RF, SVM, GBM.

    Returns:
        DataFrame with model, feature_type, AUC_mean, AUC_std.
    """
    if classifiers is None:
        classifiers = {
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=ML_RANDOM_STATE),
            "SVM": SVC(kernel="rbf", probability=True, random_state=ML_RANDOM_STATE),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=ML_RANDOM_STATE),
        }

    rows = []
    for clf_name, clf in classifiers.items():
        for feat_name, X_feat in [("topological", X_topo), ("classic", X_classic)]:
            pipe = build_topological_pipeline(clf)
            eval_result = evaluate_topological_model(pipe, X_feat, y)
            rows.append({
                "model": clf_name,
                "feature_type": feat_name,
                "AUC_mean": eval_result["roc_auc"]["mean"],
                "AUC_std": eval_result["roc_auc"]["std"],
                "accuracy_mean": eval_result["accuracy"]["mean"],
                "f1_mean": eval_result["f1"]["mean"],
            })

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Interpretability
# ═══════════════════════════════════════════════════════════════════════════════

def shap_analysis(
    pipeline: Pipeline,
    X: np.ndarray,
    feature_names: list = None,
) -> np.ndarray:
    """Compute SHAP values for a fitted pipeline.

    Returns:
        SHAP values array (n_samples, n_features).
    """
    try:
        import shap
        model = pipeline.named_steps.get("classifier", pipeline[-1])
        X_scaled = pipeline.named_steps.get("scaler", StandardScaler()).fit_transform(X)

        if hasattr(model, "predict_proba"):
            explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") else shap.KernelExplainer(
                model.predict_proba, X_scaled[:100]  # subsample for speed
            )
        else:
            explainer = shap.KernelExplainer(model.predict, X_scaled[:100])

        shap_values = explainer.shap_values(X_scaled)
        return shap_values
    except ImportError:
        logger.warning("shap not installed")
        return np.zeros(X.shape)
