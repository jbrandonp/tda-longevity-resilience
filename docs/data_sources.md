# Data Sources

## Strategy

Three-tier data strategy, from ideal to pragmatic:

### Tier 1 — Public Multi-Omics Longevity Data (Ideal)

| Dataset | Omics layers | Sample size | Accessibility | Reference |
|---------|-------------|-------------|--------------|-----------|
| UK Biobank | Genomics, metabolomics, proteomics | ~500K | Application required | Bycroft et al. 2018 |
| InCHIANTI | Transcriptomics, metabolomics | ~1,500 | Public (dbGaP) | Ferrucci et al. 2000 |
| ELSA | Epigenomics, proteomics | ~12,000 | Application required | Steptoe et al. 2013 |
| Longevity Genes Project (Einstein) | Genomics | ~500 centenarians | Public | Barzilai et al. 2003 |
| Biagi et al. 2016 | Microbiome (centenarians) | ~70 | Public (ENA) | Biagi et al. 2016 |
| Horvath DNA Methylation | Epigenomics (multi-tissue) | ~8,000 | Public (GEO) | Horvath 2013 |

### Tier 2 — Real Data + Simulated Labels (Pragmatic)

Use publicly available datasets (e.g., TCGA) augmented with **synthetic Tian aging scores**. A subset of individuals is labeled "resilient" based on preserved topological structure under perturbation.

### Tier 3 — Fully Synthetic (Pedagogical)

Generated via `data_utils.generate_synthetic_multimics()` with known topological structure (circle, torus, figure-8, sphere, noise). Included as proof of concept. No ethical concerns.

## Synthetic Data Generation

```python
from data_utils import generate_synthetic_multimics

ds = generate_synthetic_multimics(
    n_samples=200,
    topology_type='circle',
    noise=0.05,
    n_features=50
)
# Returns: transcriptomics, metabolomics, epigenomics arrays
#          labels (Tian-assigned), metadata DataFrame
```

## Data Download Scripts

See `notebooks/` for download scripts. Data files are not tracked in the repository.
Use `pooch`, `kagglehub`, or `zenodo_get` to download public datasets automatically.

## Ethical Considerations

- All data used is publicly available and anonymized
- No personally identifiable information (PII) is included
- Licenses of source datasets must be verified before use
- Results should not be interpreted as clinical predictions
