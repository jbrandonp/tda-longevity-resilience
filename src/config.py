"""Global configuration and fixed random seeds for reproducibility."""

import numpy as np

# ── Random seeds (fixed for reproducibility) ──────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── TDA hyperparameters ──────────────────────────────────────────────────────
RIPSER_MAX_DIM = 2
RIPSER_THRESH_QUANTILE = 0.95  # Vietoris-Rips distance threshold quantile
RIPSER_N_THREADS = -1          # use all cores

# ── Mapper hyperparameters ───────────────────────────────────────────────────
MAPPER_N_CUBES = 15
MAPPER_PERC_OVERLAP = 0.5
MAPPER_CLUSTER_EPS = 0.5
MAPPER_CLUSTER_MIN_SAMPLES = 5
MAPPER_UMAP_N_COMPONENTS = 2

# ── Persistence Image parameters ─────────────────────────────────────────────
PI_SPREAD = 0.1
PI_PIXELS = (50, 50)

# ── ML parameters ────────────────────────────────────────────────────────────
CV_FOLDS = 5
TEST_SIZE = 0.2
ML_RANDOM_STATE = 42

# ── Dimensionality reduction ─────────────────────────────────────────────────
UMAP_N_COMPONENTS_FULL = 50
PCA_VARIANCE_RATIO = 0.95
