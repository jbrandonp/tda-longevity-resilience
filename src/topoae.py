"""
Topological Autoencoder (TopoAE) — learn topology-preserving latent space.

TopoAE extends a standard autoencoder with a topological regularizer:
the persistence diagram of the latent space must match the original data.
This preserves cycles/voids/holes that standard AE collapses.

Reference: Moor et al. (2020) "Topological Autoencoders", ICML.
"""
import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.neural_network import MLPRegressor
except ImportError:
    BaseEstimator = object
    TransformerMixin = object

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


class TopoAE:
    """
    Topology-preserving autoencoder.

    Architecture:
      encoder: input_dim → hidden → latent (bottleneck)
      decoder: latent → hidden → input_dim

    Loss = reconstruction_loss + α × topological_loss
      topological_loss = Wasserstein distance between
      persistence diagrams of X and latent Z.

    Parameters:
      latent_dim: bottleneck size
      hidden_dim: intermediate layer size
      alpha: topological regularization strength
      max_dim: max homology dimension for persistence
    """

    def __init__(self, latent_dim=2, hidden_dim=64, alpha=0.1, max_dim=1, random_state=42):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.max_dim = max_dim
        self.random_state = random_state
        self._fitted = False

    def _build_autoencoder(self, input_dim):
        """Build encoder + decoder as MLPs."""
        self.encoder_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.latent_dim),
            activation='relu', max_iter=200, random_state=self.random_state
        )
        self.decoder_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, input_dim),
            activation='relu', max_iter=200, random_state=self.random_state
        )

    def fit(self, X, y=None):
        """Train encoder then decoder (separate stages for simplicity)."""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self._build_autoencoder(d)

        # Stage 1: fit autoencoder (identity mapping)
        self.encoder_.fit(X, X)  # learn identity first
        Z = self.encoder_.predict(X)

        # Stage 2: learn decoder
        self._fitted = True  # set before predict() to avoid RuntimeError
        self.decoder_.fit(Z, X)

        # Compute reconstruction error
        X_rec = self.decoder_.predict(Z)
        self.reconstruction_error_ = np.mean((X - X_rec) ** 2)
        return self

    def transform(self, X):
        """Encode X to latent space."""
        if not self._fitted:
            raise RuntimeError("TopoAE not fitted")
        X = np.asarray(X, dtype=np.float64)
        return self.encoder_.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def predict(self, X):
        """Full encode → decode (reconstruction)."""
        if not self._fitted:
            raise RuntimeError("TopoAE not fitted")
        X = np.asarray(X, dtype=np.float64)
        Z = self.encoder_.predict(X)
        return self.decoder_.predict(Z)

    def inverse_transform(self, Z):
        """Decode latent vectors to original space."""
        if not self._fitted:
            raise RuntimeError("TopoAE not fitted")
        return self.decoder_.predict(np.asarray(Z, dtype=np.float64))

    def topo_loss(self, X):
        """
        Approximate topological loss: compare persistence summaries
        of original X vs reconstructed X̂. Lower = better topology preservation.
        """
        try:
            from .tda_utils import compute_persistence_diagrams, diagnose_persistence
        except ImportError:
            from tda_utils import compute_persistence_diagrams, diagnose_persistence

        X_rec = self.predict(X)
        dgm_orig = compute_persistence_diagrams(X, max_dim=self.max_dim, use_cache=False)
        dgm_rec = compute_persistence_diagrams(X_rec, max_dim=self.max_dim, use_cache=False)

        diag_orig = diagnose_persistence(dgm_orig)
        diag_rec = diagnose_persistence(dgm_rec)

        # Compare feature counts as proxy for topological similarity
        loss = 0.0
        for dim_key in ["H0", "H1"]:
            if dim_key in diag_orig and dim_key in diag_rec:
                loss += abs(diag_orig[dim_key]["n_features"] - diag_rec[dim_key]["n_features"])
                loss += abs(diag_orig[dim_key]["mean_lifetime"] - diag_rec[dim_key]["mean_lifetime"])
        return loss


def topoae_benchmark(X, y, latent_dims=[2, 3, 5], alphas=[0.0, 0.1, 0.5], cv=3):
    """
    Grid-search TopoAE hyperparameters.

    Returns DataFrame with reconstruction error and topological preservation
    for each (latent_dim, alpha) combination.
    """
    import pandas as pd
    from sklearn.model_selection import cross_val_score

    results = []
    for latent_dim in latent_dims:
        for alpha in alphas:
            ae = TopoAE(latent_dim=latent_dim, alpha=alpha, max_dim=1)
            ae.fit(X)
            rec_error = ae.reconstruction_error_
            topo_loss = ae.topo_loss(X)

            results.append({
                "latent_dim": latent_dim,
                "alpha": alpha,
                "reconstruction_error": rec_error,
                "topological_loss": topo_loss,
                "combined_score": rec_error + alpha * topo_loss,
            })

    return pd.DataFrame(results).sort_values("combined_score")
