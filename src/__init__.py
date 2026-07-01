"""TDA-Longevity-Resilience: Topological Data Analysis for Longevity Research.

Apply persistent homology and the Mapper algorithm to multi-omics longevity
data to extract structural signatures of extreme resilience and compare them
to generational accelerated aging.

Quick start:
    >>> from tda_longevity import generate_synthetic_multimics, compute_persistence_diagrams
    >>> data = generate_synthetic_multimics(100, 'circle')
    >>> dgms = compute_persistence_diagrams(data['transcriptomics'])

CLI:
    python -m tda_longevity run              # full pipeline
    python -m tda_longevity demo             # quick demo with synthetic data
"""

__version__ = "1.0.0"
__author__ = "Brandon Palhano Machado"

# ── Core imports (always available) ─────────────────────────────────────────
from .config import RANDOM_SEED, RIPSER_MAX_DIM, PI_SPREAD, PI_PIXELS

# data_utils
from .data_utils import (
    load_dataset, preprocess_omics, integrate_multiomics, select_features,
    assign_groups_from_tian_score, generate_synthetic_multimics,
    compare_with_santos_pujol, _finite_dgm,
)

# tda_utils
from .tda_utils import (
    compute_persistence_diagrams, compute_all_layers_dgms,
    wasserstein_distance, bottleneck_distance,
    multi_view_topological_distance, permutation_test,
    bootstrap_diagram_stability, sparse_persistence,
    diagnose_persistence, persistence_stability,
)

# features
from .features import (
    PersistenceImageTransformer, PersistenceLandscapeTransformer,
    BettiCurveTransformer, extract_all_features,
    landscape_statistics, landscape_distance,
)

# ml_utils
from .ml_utils import (
    build_topological_pipeline, evaluate_topological_model,
    compare_with_baseline, shap_analysis,
    TopologicalFeatureExtractor, prepare_topological_features_cv,
)

# visualization
from .visualization import (
    plot_barcode, plot_barcode_comparison, plot_barcode_interactive,
    plot_persistence_diagram, plot_mapper_graph,
    plot_roc_curves, plot_confusion_matrix,
)

# metrics
from .metrics import (
    aitchison_distance, robust_correlation_distance,
    get_distance_matrix, recommended_metric_for_data,
)

# ── Optional imports (may fail if dependencies missing) ──────────────────────
try:
    from .mapper_utils import (
        build_mapper_graph, auto_mapper,
        compare_mapper_graphs, enrich_mapper_nodes, export_mapper_html,
    )
except ImportError:
    pass

try:
    from .bio_enrichment import (
        enrich_mapper_genes, annotate_persistent_cycle, compare_with_genage,
        extract_cycle_genes, run_enrichment, enrich_mapper_node,
        cross_reference_genage, plot_enrichment_heatmap,
        pathway_persistence_profile, run_enrichr,
    )
except ImportError:
    pass

try:
    from .aging_scores import (
        phenoage_simplified, assign_acceleration_group,
        dunedin_pace_proxy, compute_all_scores,
    )
except ImportError:
    pass

try:
    from .benchmark_utils import (
        TDAgingClock, wrap_as_aging_clock,
        compare_with_classical, accelerated_aging_detection,
    )
except ImportError:
    pass

try:
    from .topo_format import save_topo, load_topo, to_anndata_uns
except ImportError:
    pass

try:
    from .validation import (
        create_discovery_validation, evaluate_on_holdout,
        generate_validation_report,
    )
except ImportError:
    pass
