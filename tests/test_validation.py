"""Tests for validation.py — hold-out splitting and evaluation."""
import numpy as np, pandas as pd, pytest
from sklearn.linear_model import LogisticRegression
from src.validation import (
    create_discovery_validation, evaluate_on_holdout, generate_validation_report,
)


class TestCreateDiscoveryValidation:
    def test_returns_four(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))
        meta = pd.DataFrame({
            "age": np.random.randint(30, 90, 100),
            "aging_group": np.random.choice(["accelerated", "resilient", "normal"], 100),
        })
        df_d, df_v, meta_d, meta_v = create_discovery_validation(df, meta, test_size=0.2)
        assert len(df_d) + len(df_v) == 100
        assert len(df_v) >= 15
        assert meta_d is not None and meta_v is not None

    def test_no_metadata(self):
        df = pd.DataFrame(np.random.randn(50, 5))
        df_d, df_v, m_d, m_v = create_discovery_validation(df, test_size=0.2)
        assert len(df_d) + len(df_v) == 50


class TestEvaluateOnHoldout:
    def test_returns_dict(self):
        X = np.random.randn(100, 10); y = np.random.randint(0, 2, 100)
        model = LogisticRegression().fit(X, y)
        r = evaluate_on_holdout(model, X[:20], y[:20])
        assert "auc" in r and "accuracy" in r and "f1" in r and "confusion_matrix" in r


class TestGenerateValidationReport:
    def test_returns_markdown_string(self):
        train = {"roc_auc": {"mean": 0.85, "std": 0.05}}
        holdout = {"auc": 0.82, "accuracy": 0.78, "precision": 0.75, "recall": 0.72,
                   "f1": 0.73, "confusion_matrix": [[12, 3], [2, 8]], "n_samples": 25,
                   "overfitting_warning": "OK"}
        report = generate_validation_report(train, holdout, "Test")
        assert isinstance(report, str)
        assert "0.82" in report
