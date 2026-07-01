# API Reference

*Auto-generated documentation with pdoc3 / Sphinx. Run `pdoc --html src/` to regenerate.*

## Core Modules

- **`src.config`** — Global configuration: random seeds, TDA hyperparameters, ML parameters
- **`src.data_utils`** — Data loading, preprocessing, multi-omics integration, synthetic data generation, Tian 2026 & Santos-Pujol group assignment
- **`src.tda_utils`** — Persistent homology computation (Ripser), Wasserstein/Bottleneck distances, permutation testing, diagram diagnostics
- **`src.mapper_utils`** — Mapper graph construction (KeplerMapper), auto-parameterization, enrichment analysis, comparison
- **`src.features`** — Topological feature extractors: Persistence Images, Landscapes, Betti Curves (scikit-learn compatible)
- **`src.ml_utils`** — ML pipelines for topological classification, GridSearchCV, baseline comparison, SHAP interpretability
- **`src.visualization`** — Barcodes, persistence diagrams, Mapper graphs, ROC curves, confusion matrices

## Quick Reference

### data_utils

```python
from src.data_utils import (
    load_dataset,           # load_dataset(name: str) -> pd.DataFrame
    preprocess_omics,       # preprocess_omics(df, method='standard') -> np.ndarray
    integrate_multiomics,   # integrate_multiomics(omics_dict) -> np.ndarray
    assign_groups_from_tian_score,  # assign_groups_from_tian_score(metadata, col) -> pd.Series
    compare_with_santos_pujol,      # compare_with_santos_pujol(dgms_pool, dgm_sp, dim) -> dict
    generate_synthetic_multimics,   # generate_synthetic_multimics(n_samples, topology_type) -> dict
)
```

### tda_utils

```python
from src.tda_utils import (
    compute_persistence_diagrams,  # (data, max_dim, metric) -> list[np.ndarray]
    compute_all_layers_dgms,       # (omics_dict, max_dim) -> dict
    wasserstein_distance,          # (dgm1, dgm2, dim) -> float
    bottleneck_distance,           # (dgm1, dgm2, dim) -> float
    multi_view_topological_distance,  # (dgms_a, dgms_b, weights, dim) -> float
    permutation_test,              # (dgms_a, dgms_b, n_perm, dim) -> dict
    diagnose_persistence,          # (dgm) -> dict
)
```

### features

```python
from src.features import (
    PersistenceImageTransformer,      # sklearn-compatible transformer
    PersistenceLandscapeTransformer,  # sklearn-compatible transformer
    BettiCurveTransformer,            # sklearn-compatible transformer
    extract_all_features,             # (dgms_list) -> dict
)
```

### ml_utils

```python
from src.ml_utils import (
    build_topological_pipeline,    # (classifier, use_pca) -> Pipeline
    evaluate_topological_model,    # (pipeline, X, y, cv) -> dict
    compare_with_baseline,         # (X_topo, X_classic, y) -> pd.DataFrame
    shap_analysis,                 # (pipeline, X, feature_names) -> np.ndarray
)
```

### visualization

```python
from src.visualization import (
    plot_barcode,              # (dgm, dim, title) -> ax
    plot_persistence_diagram,  # (dgm, title) -> ax
    plot_barcode_comparison,   # (dgms_a, dgms_b, labels, dim) -> fig
    plot_mapper_graph,         # (graph, labels, title) -> html
    plot_roc_curves,           # (results_dict, title) -> fig
    plot_confusion_matrix,     # (y_true, y_pred, labels, title) -> fig
)
```

*For full documentation, regenerate with: `pdoc --html --output-dir docs/api src/ --force`*
