"""
Probabilistic group assignment — replace hard aging labels with soft membership.

Instead of fixed thresholds (±1 SD from mean), use Gaussian mixture models
to estimate group membership probabilities. Each sample gets P(accelerated),
P(normal), P(resilient) — these probabilities propagate through the pipeline.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def probabilistic_group_assignment(scores, n_components=3, random_state=42):
    """
    Fit a Gaussian mixture model to aging scores and return soft labels.

    Args:
      scores: array (n_samples,) — age acceleration or PhenoAge residuals
      n_components: number of aging groups (default 3: accelerated/normal/resilient)
      random_state: seed for reproducibility

    Returns:
      dict with:
        probabilities: (n, k) — soft membership probabilities
        means: k sorted component means
        assignments: (n,) — hard MAP assignments (for backward compatibility)
        confidence: (n,) — probability of the MAP class
    """
    from sklearn.mixture import GaussianMixture

    X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        n_init=10
    )
    gmm.fit(X)
    probs = gmm.predict_proba(X)

    # Sort components by mean (lowest = resilient, highest = accelerated)
    order = np.argsort(gmm.means_.ravel())
    probs_ordered = probs[:, order]
    means_ordered = gmm.means_.ravel()[order]
    hard = np.argmax(probs_ordered, axis=1)
    confidence = probs_ordered.max(axis=1)

    label_map = {0: "resilient", 1: "normal", 2: "accelerated"}
    return {
        "probabilities": probs_ordered,
        "means": means_ordered,
        "assignments": np.array([label_map.get(h, f"group_{h}") for h in hard]),
        "hard_labels": hard,
        "confidence": confidence,
        "model": gmm,
        "order": order,
    }


def bayesian_latent_class(scores, priors=None, n_classes=3, n_iter=100, random_state=42):
    """
    Bayesian latent-class model with informative priors.

    Uses a simple EM-like procedure with Dirichlet priors on class
    proportions and Normal-Inverse-Gamma priors on component means/variances.

    Args:
      scores: (n,) age acceleration scores
      priors: dict with 'means' (k,), 'vars' (k,) — prior component parameters
      n_classes: number of latent classes
      n_iter: maximum EM iterations
      random_state: seed

    Returns:
      dict with probabilities, parameters, convergence info
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(scores, dtype=np.float64).ravel()
    n = len(X)

    # Initialize with k-means
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=random_state)
    labels = km.fit_predict(X.reshape(-1, 1))

    # EM-like Bayesian iteration
    pi = np.ones(n_classes) / n_classes  # mixing proportions
    mu = np.array([X[labels == k].mean() if (labels == k).any() else rng.normal(0, 1)
                    for k in range(n_classes)])
    sigma2 = np.array([X[labels == k].var() if (labels == k).sum() > 1 else 1.0
                        for k in range(n_classes)])

    # Prior hyperparameters (weakly informative)
    alpha_prior = np.ones(n_classes)  # Dirichlet(1,...,1)
    mu_prior = np.array(priors["means"]) if priors and "means" in priors else np.zeros(n_classes)
    tau_prior = 0.01  # precision on mean prior

    log_lik = -np.inf
    for iteration in range(n_iter):
        # E-step: compute responsibilities
        resp = np.zeros((n, n_classes))
        for k in range(n_classes):
            resp[:, k] = pi[k] * np.exp(-0.5 * (X - mu[k])**2 / sigma2[k]) / np.sqrt(2 * np.pi * sigma2[k])
        resp /= resp.sum(axis=1, keepdims=True) + 1e-10

        # M-step with Bayesian regularization
        Nk = resp.sum(axis=0) + 1e-10
        pi = (Nk + alpha_prior - 1) / (n + n_classes * (alpha_prior[0] - 1))
        pi = np.maximum(pi, 1e-5)
        pi /= pi.sum()

        for k in range(n_classes):
            mu[k] = (tau_prior * mu_prior[k] + (resp[:, k] * X).sum()) / (tau_prior + Nk[k])
            sigma2[k] = ((resp[:, k] * (X - mu[k])**2).sum() + 1.0) / (Nk[k] + 2)

        # Convergence check
        ll_new = np.sum(np.log(resp.sum(axis=1)))
        if abs(ll_new - log_lik) < 1e-6:
            break
        log_lik = ll_new

    # Sort by mean
    order = np.argsort(mu)
    probs = resp[:, order]
    hard = np.argmax(probs, axis=1)

    label_map = {0: "resilient", 1: "normal", 2: "accelerated"}

    return {
        "probabilities": probs,
        "means": mu[order],
        "vars": sigma2[order],
        "pi": pi[order],
        "assignments": np.array([label_map.get(h, f"group_{h}") for h in hard]),
        "hard_labels": hard,
        "confidence": probs.max(axis=1),
        "converged": abs(ll_new - log_lik) < 1e-6,
        "iterations": iteration + 1,
    }
