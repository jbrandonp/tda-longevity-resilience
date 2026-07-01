"""
Geometric embeddings of topological objects — learn a smooth manifold over diagrams.

Uses kernel PCA and diffusion maps to embed persistence diagrams into a
lower-dimensional Riemannian manifold where classification/clustering is easier.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GeometricFeatureSpace:
    """
    Learn a geometry-aware embedding of topological features.

    Pipeline:
      1. Compute kernel Gram matrix over diagrams (PSSK/PWGK/PFK)
      2. Kernel PCA → lower-dimensional representation
      3. Optional: diffusion maps refinement
    """

    def __init__(self, n_components=5, kernel='pssk', sigma=1.0, use_diffusion=False):
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma
        self.use_diffusion = use_diffusion
        self._fitted = False
        self.embedding_ = None
        self.explained_variance_ = None

    def fit(self, diagrams_or_gram):
        """
        Fit embedding from list of diagrams or precomputed Gram matrix.

        Args:
          diagrams_or_gram: list of dgms OR (n×n) precomputed Gram matrix
        """
        from sklearn.decomposition import KernelPCA

        if isinstance(diagrams_or_gram, np.ndarray) and diagrams_or_gram.ndim == 2:
            K = diagrams_or_gram
        else:
            K = self._compute_gram(diagrams_or_gram)

        n = K.shape[0]

        # Kernel PCA
        kpca = KernelPCA(
            n_components=min(self.n_components, n),
            kernel='precomputed',
            random_state=42
        )
        self.embedding_ = kpca.fit_transform(K)

        # Explained variance from eigenvalues
        ev = kpca.eigenvalues_[:min(self.n_components, n)]
        self.explained_variance_ = ev / (ev.sum() + 1e-10)
        self.kpca_ = kpca
        self.gram_matrix_ = K

        if self.use_diffusion:
            self.embedding_ = self._diffusion_refinement(self.embedding_, n_iter=5)

        self._fitted = True
        logger.info(
            f"GeometricFeatureSpace: {self.n_components}D embedding, "
            f"top-3 variance: {self.explained_variance_[:3].sum():.1%}"
        )
        return self

    def fit_transform(self, diagrams_or_gram):
        return self.fit(diagrams_or_gram).embedding_

    def transform(self, diagrams):
        """Project new diagrams into existing embedding (approximate)."""
        if not self._fitted:
            raise RuntimeError("Not fitted")
        K_new = self._compute_gram(diagrams)
        return self.kpca_.transform(K_new)

    def _compute_gram(self, diagrams):
        """Compute kernel Gram matrix."""
        try:
            from .topological_kernels import pssk_gram, pwgk_gram, pfk_gram
        except ImportError:
            from topological_kernels import pssk_gram, pwgk_gram, pfk_gram

        if self.kernel == 'pssk':
            return pssk_gram(diagrams, sigma=self.sigma)
        elif self.kernel == 'pwgk':
            return pwgk_gram(diagrams, sigma=self.sigma)
        elif self.kernel == 'pfk':
            return pfk_gram(diagrams, sigma=self.sigma, bandwidth=self.sigma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _diffusion_refinement(self, X, n_iter=5):
        """Diffusion map smoothing on existing embedding."""
        from scipy.spatial.distance import pdist, squareform

        # Build affinity matrix via Gaussian kernel
        dists = squareform(pdist(X))
        sigma_d = np.median(dists[dists > 0]) if (dists > 0).any() else 1.0
        A = np.exp(-dists**2 / (2 * sigma_d**2))

        # Row-normalize → Markov transition matrix
        D = A.sum(axis=1)
        P = A / (D[:, np.newaxis] + 1e-10)

        # Diffuse
        for _ in range(n_iter):
            X = P @ X

        return X


def diffusion_maps(X, n_components=2, sigma=None, n_iter=1, random_state=42):
    """
    Diffusion maps embedding (Coifman & Lafon 2006).

    Captures the intrinsic geometry of the data at multiple scales.
    """
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist, squareform

    # Affinity matrix
    dists = squareform(pdist(X))
    if sigma is None:
        sigma = np.median(dists[dists > 0]) if (dists > 0).any() else 1.0
    A = np.exp(-dists**2 / (2 * sigma**2))

    # Row-normalize
    D = A.sum(axis=1)
    P = A / (D[:, np.newaxis] + 1e-10)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(P)

    # Take largest eigenvalues (excluding the trivial λ=1)
    idx = np.argsort(np.abs(eigenvalues))[::-1][1:n_components + 1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Diffusion map embedding: ψ_t(x) = (λ₁ᵗφ₁(x), λ₂ᵗφ₂(x), ...)
    embedding = eigenvectors * (eigenvalues ** n_iter)

    return {
        "embedding": embedding,
        "eigenvalues": eigenvalues,
        "sigma": sigma,
        "diffusion_time": n_iter,
    }


def topology_manifold_embedding(diagrams, n_components=3, method='kpca', kernel='pssk'):
    """
    High-level function: diagrams → smooth manifold embedding.

    Returns a low-dimensional representation where samples with
    similar topological structure are close together.
    """
    geo = GeometricFeatureSpace(
        n_components=n_components,
        kernel=kernel,
        sigma=1.0,
        use_diffusion=(method == 'diffusion')
    )
    embedding = geo.fit_transform(diagrams)

    return {
        "embedding": embedding,
        "method": method,
        "kernel": kernel,
        "n_components": n_components,
        "explained_variance": geo.explained_variance_.tolist() if geo.explained_variance_ is not None else None,
    }
