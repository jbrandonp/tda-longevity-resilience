"""Tests for ml_utils.py — ML pipelines and topological feature extraction."""
import numpy as np, pytest
from sklearn.ensemble import RandomForestClassifier
from src.ml_utils import (
    build_topological_pipeline, evaluate_topological_model,
    compare_with_baseline, shap_analysis,
    TopologicalFeatureExtractor, prepare_topological_features_cv,
)


class TestBuildPipeline:
    def test_default(self):
        pipe = build_topological_pipeline()
        assert "scaler" in pipe.named_steps
        assert "classifier" in pipe.named_steps
    def test_with_svm(self):
        from sklearn.svm import SVC
        pipe = build_topological_pipeline(SVC(probability=True))
        assert isinstance(pipe.named_steps["classifier"], SVC)
    def test_with_pca(self):
        pipe = build_topological_pipeline(use_pca=True)
        assert "pca" in pipe.named_steps


class TestEvaluateModel:
    def test_returns_dict(self):
        X = np.random.randn(50, 10); y = np.array([0, 1] * 25)
        pipe = build_topological_pipeline()
        r = evaluate_topological_model(pipe, X, y, cv=3)
        assert "roc_auc" in r and "accuracy" in r
        assert "mean" in r["roc_auc"]
        assert 0 <= r["roc_auc"]["mean"] <= 1


class TestCompareBaseline:
    def test_returns_dataframe(self):
        import pandas as pd
        X_t = np.random.randn(40, 10); X_c = np.random.randn(40, 8)
        y = np.array([0, 1] * 20)
        df = compare_with_baseline(X_t, X_c, y)
        assert isinstance(df, pd.DataFrame)
        assert "AUC_mean" in df.columns


class TestSHAP:
    def test_returns_array(self):
        X = np.random.randn(30, 8); y = np.array([0, 1] * 15)
        pipe = build_topological_pipeline()
        pipe.fit(X, y)
        vals = shap_analysis(pipe, X)
        assert isinstance(vals, np.ndarray)


class TestTopologicalFeatureExtractor:
    def test_class_instantiable(self):
        tfe = TopologicalFeatureExtractor(maxdim=1, spread=0.1, pixels=[20, 20])
        assert hasattr(tfe, "fit") and hasattr(tfe, "transform")
    def test_fit_requires_umap(self):
        tfe = TopologicalFeatureExtractor(maxdim=0)
        X = np.random.randn(20, 10)
        try:
            tfe.fit(X)
            assert tfe._is_fitted
        except ImportError:
            pytest.skip("umap not installed")


class TestPrepareTopologicalFeaturesCV:
    def test_returns_dict(self):
        X = np.random.randn(30, 10); y = np.array([0, 1] * 15)
        try:
            r = prepare_topological_features_cv(X, y, n_splits=3, maxdim=0, verbose=False)
            assert "mean_auc" in r and "fold_details" in r
        except ImportError:
            pytest.skip("umap or ripser not installed")
