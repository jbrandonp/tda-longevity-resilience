"""
Topological kernel embeddings — turn persistence diagrams into RKHS features.

Instead of using a single fixed distance (Wasserstein/bottleneck), embed
diagrams into a reproducing kernel Hilbert space (RKHS) where standard
ML tools (SVM, ridge regression, kernel PCA) can operate directly.

References:
  - Reininghaus et al. (2015) PSSK: Persistence Scale-Space Kernel
  - Kusano et al. (2018) PWGK: Persistence Weighted Gaussian Kernel
  - Le & Yamada (2018) PFK: Persistence Fisher Kernel
"""
import numpy as np

logger = __import__('logging').getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Persistence Scale-Space Kernel (Reininghaus 2015)
# ═══════════════════════════════════════════════════════════════════

def pssk(dgm1, dgm2, sigma=1.0):
    """
    Persistence Scale-Space Kernel between two diagrams.

    K(D1, D2) = (1 / 8πσ) Σ_{p∈D1} Σ_{q∈D2}
      exp(-||p-q||² / 8σ²) - exp(-||p-\bar{q}||² / 8σ²)

    where \bar{q} is the reflection of q across the diagonal.
    """
    d1 = np.asarray(dgm1, dtype=np.float64)
    d2 = np.asarray(dgm2, dtype=np.float64)
    if len(d1) == 0 or len(d2) == 0:
        return 1e-10

    s2 = 8.0 * sigma * sigma
    K = 0.0
    for p in d1:
        for q in d2:
            q_bar = np.array([q[1], q[0]])
            K += np.exp(-np.sum((p - q)**2) / s2) - np.exp(-np.sum((p - q_bar)**2) / s2)

    return K / (8.0 * np.pi * sigma)


def pssk_gram(diagrams, sigma=1.0):
    """Compute Gram matrix for a list of persistence diagrams."""
    n = len(diagrams)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k = pssk(diagrams[i], diagrams[j], sigma)
            K[i, j] = K[j, i] = k
    return K


# ═══════════════════════════════════════════════════════════════════
# Persistence Weighted Gaussian Kernel (Kusano 2018)
# ═══════════════════════════════════════════════════════════════════

def pwgk(dgm1, dgm2, sigma=1.0, tau=1.0):
    """
    Persistence Weighted Gaussian Kernel.

    K(D1, D2) = Σ_{p∈D1} Σ_{q∈D2} w(p) w(q) exp(-||p-q||² / 2σ²)
    where w((b,d)) = arctan(τ × persistence(b,d))
    """
    d1 = np.asarray(dgm1, dtype=np.float64)
    d2 = np.asarray(dgm2, dtype=np.float64)
    if len(d1) == 0 or len(d2) == 0:
        return 1e-10

    def arcweight(dgm):
        lifetimes = np.maximum(dgm[:, 1] - dgm[:, 0], 1e-10)
        return np.arctan(tau * lifetimes)

    w1 = arcweight(d1)
    w2 = arcweight(d2)
    s2 = 2.0 * sigma * sigma

    K = 0.0
    for i in range(len(d1)):
        for j in range(len(d2)):
            K += w1[i] * w2[j] * np.exp(-np.sum((d1[i] - d2[j])**2) / s2)

    return K


def pwgk_gram(diagrams, sigma=1.0, tau=1.0):
    """Gram matrix via PWGK."""
    n = len(diagrams)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k = pwgk(diagrams[i], diagrams[j], sigma, tau)
            K[i, j] = K[j, i] = k
    return K


# ═══════════════════════════════════════════════════════════════════
# Persistence Fisher Kernel (Le & Yamada 2018)
# ═══════════════════════════════════════════════════════════════════

def pfk(dgm1, dgm2, sigma=1.0, bandwidth=1.0):
    """
    Persistence Fisher Kernel via vectorized persistence images.

    Simplified: computes persistence images for both diagrams and
    uses the Gaussian kernel on the flat vectors.
    """
    d1 = np.asarray(dgm1, dtype=np.float64)
    d2 = np.asarray(dgm2, dtype=np.float64)
    if len(d1) == 0 or len(d2) == 0:
        return 1e-10

    # Vectorize: (birth, persistence) → weighted point
    def vectorize(dgm):
        b = dgm[:, 0]
        p = dgm[:, 1] - dgm[:, 0]
        p_norm = p / (p.max() + 1e-10)
        return np.column_stack([b, p_norm])

    v1 = vectorize(d1)
    v2 = vectorize(d2)

    # Fisher kernel: use Gaussian on flattened mean+std
    def fisher_stat(vec):
        return np.concatenate([vec.mean(axis=0), vec.std(axis=0)])

    s1 = fisher_stat(v1)
    s2 = fisher_stat(v2)

    return np.exp(-np.sum((s1 - s2)**2) / (2.0 * bandwidth * bandwidth))


def pfk_gram(diagrams, sigma=1.0, bandwidth=1.0):
    """Gram matrix via PFK."""
    n = len(diagrams)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k = pfk(diagrams[i], diagrams[j], sigma, bandwidth)
            K[i, j] = K[j, i] = k
    return K


# ═══════════════════════════════════════════════════════════════════
# Model Selection via Cross-Validation
# ═══════════════════════════════════════════════════════════════════

def cv_select_best_kernel(diagrams, labels, kernels=None, cv=5, random_state=42):
    """
    Select the best topological kernel via cross-validated SVM accuracy.

    Args:
      diagrams: list of dgms (one per sample)
      labels: (n,) classification labels
      kernels: dict {name: kernel_fn(diagrams) → Gram}, None uses defaults
      cv: K-fold CV
      random_state: seed

    Returns:
      dict with best_kernel, all_scores, gram_matrices
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import SVC

    if kernels is None:
        kernels = {
            "PSSK": lambda d: pssk_gram(d, sigma=1.0),
            "PWGK": lambda d: pwgk_gram(d, sigma=1.0, tau=1.0),
            "PFK": lambda d: pfk_gram(d, sigma=1.0, bandwidth=1.0),
        }

    scores = {}
    grams = {}
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, kernel_fn in kernels.items():
        try:
            K = kernel_fn(diagrams)
            np.fill_diagonal(K, K.max())  # stabilize
            grams[name] = K
            svm = SVC(kernel='precomputed', C=1.0, random_state=random_state)
            acc = cross_val_score(svm, K, labels, cv=cv_splitter).mean()
            scores[name] = acc
            logger.info(f"  {name}: CV accuracy = {acc:.4f}")
        except Exception as e:
            logger.warning(f"  {name}: failed — {e}")
            scores[name] = np.nan

    best = max(scores, key=lambda k: scores.get(k, 0) or 0)
    return {
        "best_kernel": best,
        "scores": scores,
        "gram_matrices": grams,
        "best_score": scores[best],
    }
