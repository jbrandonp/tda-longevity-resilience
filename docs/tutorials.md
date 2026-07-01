# Tutorials

## Tutorial 1: Your First Persistent Homology

*Corresponds to Notebook 00*

```python
from src.data_utils import generate_synthetic_multimics, preprocess_omics
from src.tda_utils import compute_persistence_diagrams, diagnose_persistence
from src.visualization import plot_barcode, plot_persistence_diagram
import pandas as pd

# 1. Generate data with known topology
ds = generate_synthetic_multimics(n_samples=100, topology_type='circle', noise=0.05)
data = preprocess_omics(pd.DataFrame(ds['transcriptomics']), method='standard')

# 2. Compute persistence
dgms = compute_persistence_diagrams(data, max_dim=2)

# 3. Diagnose
info = diagnose_persistence(dgms)
for dim, stats in info.items():
    print(f"{dim}: {stats['n_features']} features, lifetime={stats['mean_lifetime']:.3f}")

# 4. Visualize
plot_barcode(dgms, dim=1, title='H1 Barcode — Circle Topology')
```

## Tutorial 2: Comparing Two Groups

*Corresponds to Notebook 01*

```python
from src.tda_utils import wasserstein_distance

# Compute diagrams for each group
dgm_accel = compute_persistence_diagrams(data_accel, max_dim=2)
dgm_resil = compute_persistence_diagrams(data_resil, max_dim=2)

# Compute distance
w_dist = wasserstein_distance(dgm_accel, dgm_resil, dim=1)
print(f"Wasserstein H1 distance: {w_dist:.4f}")
```

## Tutorial 3: Building a Mapper Graph

*Corresponds to Notebook 02*

```python
from src.data_utils import integrate_multiomics
from src.mapper_utils import auto_mapper, enrich_mapper_nodes

# Integrate omics layers
omics = {
    'transcriptomics': data_t,
    'metabolomics': data_m,
}
integrated = integrate_multiomics(omics)

# Build Mapper graph
graph = auto_mapper(integrated, labels=labels)

# Analyze node enrichment
enrichment = enrich_mapper_nodes(graph, labels)
print(enrichment[enrichment['dominant_group'] == 'resilient'])
```

## Tutorial 4: ML Classification with Topological Features

*Corresponds to Notebook 04*

```python
from src.features import extract_all_features
from src.ml_utils import build_topological_pipeline, evaluate_topological_model
from sklearn.ensemble import RandomForestClassifier

# Extract features
features = extract_all_features(dgms_list)

# Build pipeline
pipe = build_topological_pipeline(RandomForestClassifier(n_estimators=200))

# Evaluate
results = evaluate_topological_model(pipe, features['persistence_images'], y)
print(f"AUC: {results['roc_auc']['mean']:.3f} ± {results['roc_auc']['std']:.3f}")
```

## Tutorial 5: Santos-Pujol Comparison

```python
from src.data_utils import compare_with_santos_pujol

result = compare_with_santos_pujol(
    dgm_resilient_pool=[dgm_r1, dgm_r2, ...],
    dgm_santos_pujol=dgm_sp,
    dim=1
)
print(f"Percentile rank in resilient pool: {result['percentile_rank']}")
print(f"Interpretation: {result['interpretation']}")
```
