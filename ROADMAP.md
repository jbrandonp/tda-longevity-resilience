# Roadmap

## v1.1 — Enhanced Biological Integration
- [ ] Integrate real public multi-omics data (UK Biobank, InCHIANTI, ELSA)
- [ ] GenAge & LongevityMap gene set enrichment for Mapper nodes
- [ ] KEGG/Reactome pathway annotation of persistent H1 cycles
- [ ] Comparison with epigenetic clocks (Horvath, GrimAge, PhenoAge)

## v1.2 — Advanced ML
- [ ] Individual-level persistence diagrams (not just group-level)
- [ ] Topological Autoencoder (TopoAE) for latent space
- [ ] Graph Neural Network on Mapper graphs with attention
- [ ] Streamlit/Voilà dashboard for interactive data exploration

## v1.3 — Reproducibility & Scale
- [ ] Nextflow/Snakemake pipeline for automated end-to-end runs
- [ ] MLflow/W&B experiment tracking
- [ ] PyPI and conda-forge packaging
- [ ] Docker image on Docker Hub (for non-Conda users)

## v2.0 — Publication & Clinical
- [ ] Executable article (Quarto/Jupyter Book)
- [ ] Preprint on bioRxiv/medRxiv
- [ ] Conference submissions (ISMB, RECOMB, ARDD)
- [ ] Clinical validation on longitudinal aging cohorts
- [ ] Integration with electronic health records (EHR)

## Backlog
| ID | Idea | Priority |
|----|------|----------|
| B1 | Topological autoencoder (TopoAE) | High |
| B2 | GNN with topological attention | Medium |
| B3 | Longitudinal time-series persistence (Dask for large volumes) | Low |
| B4 | Epigenetic clock comparison (Horvath, GrimAge) | High |
| B5 | Nextflow automated pipeline | Low |
| B6 | PyPI + conda-forge packaging | Medium |
| B7 | GenAge/LongevityMap database integration | High |
