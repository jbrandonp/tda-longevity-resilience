# Competitive Landscape — TDA-Longevity-Resilience

## Direct Competitors: TDA-Based Genomics

### InterTADs (NARGAB 2022)
- **What:** R framework integrating multi-omics via topologically associated domains (TADs)
- **Topology used:** Genomic/chromatin only (Hi-C TADs), NOT algebraic topology
- **Strengths:** End-to-end pipeline, KEGG/GO enrichment, validated on 135 CLL patients
- **Weaknesses:** Cancer-focused, R-only, no persistent homology, no ML classification, no Mapper visualization
- **Our edge:** True persistent homology (cycles/vacancies TADs can't see) + Mapper graphs + Python ML pipeline

### Single-Cell Topological Alignment (biorxiv 2024)
- **What:** Topological methods for aligning single-cell multi-omics
- **Weaknesses:** Integration-only, no group comparison (resilient vs accelerated), no biological enrichment layer
- **Our edge:** Comparative group analysis + pathway annotation via bio_enrichment.py

### UMAP+Density Clustering Approaches
- **What:** Substitutes UMAP+clustering for genuine TDA
- **Weaknesses:** No mathematical guarantees (stability theorems, multi-scale persistence)
- **Our edge:** ripser/giotto-tda with stability theorems + proper persistence landscapes

## Indirect Competitors: Aging Clocks

### Epigenetic Clocks (Horvath, GrimAge, DunedinPACE)
- **Strengths:** Massive validation cohorts, clinical adoption, geroscience trials
- **Weaknesses:** Scalar output only — discards topology of biomarker interactions; cannot detect qualitatively distinct modes of resilience vs decline
- **Our edge:** Interpretable topological features + mechanistic pathway annotation + qualitative "modes" via Mapper

### Deep Aging Clocks
- **Weaknesses:** Black-box deep learning, poor interpretability
- **Our edge:** SHAP on persistence images + biological enrichment = mechanistic insight

## Structural Advantages

| Capability | InterTADs | Deep Clocks | This Project |
|---|---|---|---|
| Persistent homology | No | No | ✅ |
| Mapper visualization | No | No | ✅ |
| CI/CD + tests | Partial | Varies | ✅ |
| Docker + Binder | No | Varies | ✅ |
| Full documentation | Partial | Varies | ✅ 15+ docs |
| Open Python codebase | R only | Varies | ✅ |
| Resilience vs acceleration | No | No | ✅ Core focus |

## Winning Strategy

1. **Don't compete on raw accuracy** (deep learning + huge cohorts wins there)
2. **Compete on interpretability + mechanistic insight** — SHAP + KEGG/GO enrichment
3. **Benchmark explicitly** against PCA+SVM and epigenetic clock baselines
4. **Ship the reproducibility story** — Docker, CI/CD, tests, docs
5. **Frame results biologically** — each persistent cycle annotated via bio_enrichment.py

## References
- InterTADs: NARGAB 4(1), lqab121 (2022)
- TDA in aging position paper: *Mechanisms of Ageing and Development* (2020)
- Deep aging clocks: *Aging* 11(2), 2019
