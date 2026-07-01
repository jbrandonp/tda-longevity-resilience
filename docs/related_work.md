# Related Work — Survey of TDA in Aging Research

## Current State

To our knowledge, **no study has applied persistent homology and the Mapper algorithm to multi-omics longevity data** to extract signatures of extreme resilience. This gap motivates the present work.

## TDA in Biology (Adjacent Fields)

| Application | Method | Reference | Key Finding |
|-----------|--------|-----------|------------|
| Breast cancer subtyping | Mapper | Nicolau et al. 2011 | Novel subtype with 100% survival missed by clustering |
| Single-cell RNA-seq | Mapper | Rizvi et al. 2017 | Continuous differentiation trajectories |
| Microbiome community | Mapper | Dey et al. 2017 | Structure of gut microbiome in health/disease |
| Protein folding | Persistence | Cang & Wei 2017 | Topological features predict binding affinity |
| Drug discovery | Persistence images | Cang et al. 2018 | PI-based features outperform classical descriptors |
| Brain connectivity | Persistence | Lee et al. 2017 | Persistent homology of fMRI networks |

## Aging / Longevity Research (Classical Methods)

| Study | Method | Key Finding |
|-------|--------|------------|
| Horvath 2013 | Elastic net regression | Epigenetic clock predicts chronological age across tissues |
| Levine et al. 2018 (PhenoAge) | Cox regression | Clinical biomarker-based aging clock |
| Lu et al. 2019 (GrimAge) | Elastic net | Mortality risk predictor from DNA methylation |
| Biagi et al. 2016 | 16S rRNA sequencing | Centenarian gut microbiome differs from elderly controls |
| Sebastiani et al. 2019 | GWAS | Genetic signatures of extreme longevity |

## Gap Identified

1. **No TDA** has been applied to multi-omics longevity data
2. **No topological comparison** between accelerated and resilient aging
3. **No Mapper-based exploration** of centenarian multi-omics space
4. **No topological feature extraction** for longevity ML

## This Project's Contribution

1. First application of persistent homology to multi-omics longevity data
2. Multi-view topological distance framework (Tian 2026 integration)
3. Santos-Pujol supercentenarian topological comparison
4. Scikit-learn compatible topological feature extractors
5. Benchmark of topological vs classical features for aging classification
