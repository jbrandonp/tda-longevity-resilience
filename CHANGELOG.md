# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-07-01

### Added
- Initial release
- `data_utils.py`: data loading, preprocessing, multi-omics integration, synthetic data generation, Tian 2026 group assignment, Santos-Pujol comparison
- `tda_utils.py`: persistent homology computation (Ripser), Wasserstein/Bottleneck distances, permutation testing, diagram diagnostics
- `mapper_utils.py`: Mapper graph construction (KeplerMapper), auto-parameterization, enrichment analysis, HTML export
- `features.py`: Persistence Images, Persistence Landscapes, Betti Curves (scikit-learn compatible transformers)
- `ml_utils.py`: ML pipelines for topological classification, GridSearchCV, SHAP interpretability, baseline comparison
- `visualization.py`: barcodes, persistence diagrams, Mapper graphs, ROC curves, confusion matrices
- 5 Jupyter notebooks: synthetic validation, persistent homology, Mapper analysis, group comparison, ML classification
- Full documentation suite (mathematical background, biological interpretation, glossary, references, etc.)
- CI/CD pipelines (test, lint, docs)
- Pre-commit hooks (black, isort, flake8, nbstripout)
- GitHub issue/PR templates

[1.0.0]: https://github.com/jbrandonp/tda-longevity-resilience/releases/tag/v1.0.0
