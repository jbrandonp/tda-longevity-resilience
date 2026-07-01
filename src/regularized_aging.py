"""
Regularized biological age estimation — replace heuristic formulas with optimization.

Formulates PhenoAge as a regularized regression problem:
  min_θ ||chron_age - f(X; θ)||² + λ R(θ)

where R(θ) is L1 (Lasso), L2 (Ridge), or ElasticNet penalty.
This gives an explicit objective function and automatically selects
which biomarkers matter most for biological age.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RegularizedLatentAge:
    """
    Estimate biological age via regularized regression on biomarkers.

    The model: biological_age = w₁·albumin + w₂·creatinine + ... + b

    Regularization:
      - 'lasso' (L1): sparse biomarker selection
      - 'ridge' (L2): shrinkage for collinearity
      - 'elasticnet' (L1+L2): best of both

    Parameters:
      alpha: regularization strength
      l1_ratio: mix ratio (0=ridge, 1=lasso) for elasticnet
      penalty: 'lasso', 'ridge', or 'elasticnet'
      scale: whether to StandardScale biomarkers first
    """

    def __init__(self, alpha=0.1, l1_ratio=0.5, penalty='elasticnet', scale=True, random_state=42):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.penalty = penalty
        self.scale = scale
        self.random_state = random_state
        self._fitted = False
        self.weights_ = None
        self.intercept_ = None
        self.selected_features_ = None

    def fit(self, X, chron_age):
        """
        Fit regularized biological age model.

        Args:
          X: (n_samples, n_biomarkers) — biomarker values
          chron_age: (n,) — chronological ages

        Returns: self
        """
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(chron_age, dtype=np.float64).ravel()

        if self.scale:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        if self.penalty == 'lasso':
            model = Lasso(alpha=self.alpha, max_iter=5000, random_state=self.random_state)
        elif self.penalty == 'ridge':
            model = Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.penalty == 'elasticnet':
            model = ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio,
                max_iter=5000, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown penalty: {self.penalty}")

        model.fit(X, y)
        self.model_ = model
        self.weights_ = model.coef_
        self.intercept_ = model.intercept_

        # Track selected features (non-zero weights)
        self.selected_features_ = np.where(np.abs(self.weights_) > 1e-6)[0]
        self.n_features_ = X.shape[1]
        self.n_selected_ = len(self.selected_features_)

        # Compute training R²
        self.train_score_ = model.score(X, y)

        self._fitted = True
        logger.info(
            f"RegularizedLatentAge ({self.penalty}): "
            f"selected {self.n_selected_}/{self.n_features_} features, "
            f"R²={self.train_score_:.4f}, alpha={self.alpha}"
        )
        return self

    def predict(self, X):
        """Predict biological age from biomarkers."""
        if not self._fitted:
            raise RuntimeError("Not fitted")
        X = np.asarray(X, dtype=np.float64)
        if self.scale:
            X = self.scaler_.transform(X)
        return self.model_.predict(X)

    def fit_predict(self, X, chron_age):
        return self.fit(X, chron_age).predict(X)

    def acceleration(self, X, chron_age):
        """Age acceleration = biological_age - chronological_age."""
        bio_age = self.predict(X)
        return bio_age - np.asarray(chron_age).ravel()

    def score(self, X, chron_age):
        return self.model_.score(
            self.scaler_.transform(X) if self.scale else X,
            chron_age
        )


class AdaptiveRegularizedAge:
    """
    Data-driven λ selection via BIC.

    Fits multiple α values, picks the one minimizing:
      BIC = n × log(MSE) + k × log(n)
    where k = number of non-zero weights.
    """

    def __init__(self, alphas=None, penalty='elasticnet', l1_ratio=0.5, scale=True, random_state=42):
        self.alphas = alphas if alphas is not None else np.logspace(-3, 1, 20)
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.scale = scale
        self.random_state = random_state

    def fit(self, X, chron_age):
        """Fit with BIC-selected α."""
        best_bic = np.inf
        best_model = None
        n = len(chron_age)

        for alpha in self.alphas:
            model = RegularizedLatentAge(
                alpha=alpha, l1_ratio=self.l1_ratio,
                penalty=self.penalty, scale=self.scale,
                random_state=self.random_state
            )
            model.fit(X, chron_age)
            mse = np.mean((model.predict(X) - chron_age) ** 2)
            k = model.n_selected_ + 1  # params = selected features + intercept
            bic = n * np.log(mse + 1e-10) + k * np.log(n)

            if bic < best_bic:
                best_bic = bic
                best_model = model

        self.best_model_ = best_model
        self.selected_alpha_ = best_model.alpha
        self.best_bic_ = best_bic
        self._fitted = True

        logger.info(
            f"AdaptiveRegularizedAge: best α={self.selected_alpha_:.4f}, "
            f"selected {best_model.n_selected_}/{best_model.n_features_} features"
        )
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def acceleration(self, X, chron_age):
        return self.best_model_.acceleration(X, chron_age)


def regularized_group_assignment(scores, method='gmm', n_groups=3, priors=None):
    """
    Regularized group assignment with probabilistic outputs.

    Defaults to Gaussian Mixture Model for soft clustering of age acceleration.
    Falls back to quantile-based for compatibility.
    """
    try:
        from .bayesian_labels import probabilistic_group_assignment, bayesian_latent_class
    except ImportError:
        from bayesian_labels import probabilistic_group_assignment, bayesian_latent_class

    if method == 'gmm':
        return probabilistic_group_assignment(scores, n_components=n_groups)
    elif method == 'bayesian':
        return bayesian_latent_class(scores, priors=priors, n_classes=n_groups)
    else:
        # Quantile fallback
        scores = np.asarray(scores).ravel()
        q_low, q_high = np.percentile(scores, [33, 67])
        groups = np.full(len(scores), "normal", dtype=object)
        groups[scores < q_low] = "resilient"
        groups[scores > q_high] = "accelerated"
        return {"assignments": groups, "hard_labels": np.where(groups == "resilient", 0,
                np.where(groups == "accelerated", 2, 1))}
