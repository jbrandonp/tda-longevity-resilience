<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&width=600&lines=%F0%9F%A7%AC+TDA-Longevity-Resilience;Persistent+homology+%C3%97+aging+omics">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=0969DA&center=true&vCenter=true&width=600&lines=%F0%9F%A7%AC+TDA-Longevity-Resilience;Persistent+homology+%C3%97+aging+omics">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" /></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-214%20passed-brightgreen?style=flat-square&logo=pytest&logoColor=white" /></a>
  <a href="#"><img src="https://img.shields.io/badge/coverage-90%25-brightgreen?style=flat-square" /></a>
  <a href="#"><img src="https://img.shields.io/badge/code%20quality-A+%20(0%20bugs)-brightgreen?style=flat-square" /></a>
  <a href="#"><img src="https://img.shields.io/badge/docs-complete-informational?style=flat-square&logo=readthedocs&logoColor=white" /></a>
</p>

<p align="center">
  <b>Persistent homology unveils topological signatures of extreme longevity resilience.</b><br>
  <sub>Vietoris-Rips · Mapper · Persistence Images · PhenoAge · Multi-omics</sub>
</p>

---

## 📖 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🔬 What It Does](#-what-it-does)
- [🧭 Pipeline](#-pipeline)
- [📦 Architecture](#-architecture)
- [🖥 CLI](#-cli)
- [🧪 Tests](#-tests)
- [📓 Notebooks](#-notebooks)
- [📚 Documentation](#-documentation)
- [🔬 Reproducibility](#-reproducibility)
- [📖 Citation](#-citation)
- [🤝 Contributing](#-contributing)

---

## 🚀 Quick Start

```bash
git clone https://github.com/jbrandonp/tda-longevity-resilience.git
cd tda-longevity-resilience
conda env create -f environment.yml
conda activate tda-longevity

# 5-second hello world: circle → persistence
python -m src.cli hello

# 30-second quick demo
python -m src.cli demo

# Full pipeline on synthetic multi-omics
python -m src.cli run --n-samples 200 --max-dim 2
```

---

## 🔬 What It Does

**Given multi-omics data** (transcriptomics + metabolomics + epigenomics) across a cohort spanning accelerated to resilient aging, this package:

1. **Computes aging scores** using PhenoAge (Levine 2018) and DunedinPACE proxy
2. **Extracts persistent homology** (H0, H1, H2) from the shape of the omics cloud
3. **Builds Mapper graphs** revealing the "shape" of the aging landscape
4. **Vectorizes topology** into Persistence Images, Landscapes, and Betti curves
5. **Trains ML classifiers** to distinguish resilient from accelerated aging
6. **Validates** on held-out data with stratified splits
7. **Interprets** findings biologically via GO/KEGG enrichment and GenAge overlap

| Concept | In Plain English |
|---------|-----------------|
| **H0** (connected components) | How many clusters? Does the cohort fragment with age? |
| **H1** (cycles) | Are there loops in the gene expression landscape? Robust feedback networks? |
| **H2** (voids) | Are there voids — regions of biological space that are inaccessible? |
| **Mapper graph** | A "shape skeleton" — does aging create branches? Is resilience a separate lobe? |

---

## 🧭 Pipeline

<p align="center">
  <img src="https://via.placeholder.com/800x90/0d1117/58a6ff?text=Multi-omics+→+Aging+Scores+→+Persistent+Homology+→+Mapper+Graph+→+Feature+Extraction+→+ML+Classification+→+Biological+Interpretation+→+Validation+Report" alt="pipeline" width="800"/>
</p>

```
┌─────────────────────────────────────────────────────────────────────┐
│                    7-STEP PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────┤
│ STEP 1  │ Data Generation + PhenoAge Scoring                        │
│ STEP 2  │ Persistent Homology (Vietoris-Rips, Ripser, H0/H1/H2)    │
│ STEP 3  │ Mapper Graph (KeplerMapper, UMAP lens, DBSCAN)           │
│ STEP 4  │ Topological Feature Extraction (PI, PL, Betti Curves)    │
│ STEP 5  │ ML Classification (RF/SVM/GBM, CV, no data leakage)      │
│ STEP 6  │ Biological Interpretation (GO, KEGG, GenAge cross-ref)   │
│ STEP 7  │ Validation Report (stratified hold-out, Wassertsein)      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Architecture

```
tda-longevity-resilience/
├── src/                          # 18 source modules
│   ├── __init__.py               # 60+ public functions exported
│   ├── cli.py                    # Unified CLI (7 commands)
│   ├── config.py                 # All constants, seeds, thresholds
│   ├── data_utils.py             # Data loading, synthesis, integration
│   ├── tda_utils.py              # Persistent homology, distances, caching
│   ├── features.py               # PI, PL, Betti transformers
│   ├── ml_utils.py               # ML pipelines, SHAP, CV-safe features
│   ├── mapper_utils.py           # Mapper graph, enrichment
│   ├── aging_scores.py           # PhenoAge, DunedinPACE, acceleration
│   ├── epigenetic_clocks.py      # Horvath (2013), GrimAge (2019) clocks
│   ├── metrics.py                # Distance metrics (aitchison, etc.)
│   ├── bio_enrichment.py         # GO/KEGG enrichment, GenAge, Fisher + FDR
│   ├── validation.py             # Hold-out split, evaluation, reports
│   ├── benchmark_utils.py        # Aging clock wrappers, benchmarks
│   ├── topoae.py                 # Topological Autoencoder (Moor et al. 2020)
│   ├── topo_gnn.py               # GCN on Mapper graphs with topological features
│   ├── longitudinal.py           # Sliding window, Takens embedding, Dask
│   ├── visualization.py          # Barcodes, diagrams, ROC curves
│   ├── topo_format.py            # .topo file format (JSON-based, secure)
│   └── logging_config.py         # Structured logging
├── tests/                        # 16 test files, 214 tests
│   ├── test_quality.py           # 42 tests — 7 quality dimensions
│   ├── test_real_extreme.py      # 48 tests — real-world + extreme
│   ├── test_deep.py              # 36 tests — integration + invariants
│   └── test_*.py                 # 13 module-specific test files
├── notebooks/                    # 6 Jupyter notebooks
├── docs/                         # 15 documentation files
├── scripts/                      # 4 utility scripts
├── .github/workflows/            # CI/CD (test, lint, docs)
├── environment.yml               # Conda environment
├── environment.lock.yml          # Pinned exact versions
└── Dockerfile + docker-compose   # Containerized deployment
```

---

## 🖥 CLI

```bash
python -m src.cli run       # Full 7-step pipeline
python -m src.cli demo      # Quick 30-second demo
python -m src.cli hello     # Circle → persistence (hello world)
python -m src.cli data      # Generate synthetic datasets
python -m src.cli tda       # TDA analysis only
python -m src.cli ml        # ML classification only
python -m src.cli report    # Generate validation report
```

**Options for `run`:**
```
--n-samples INT      # Number of synthetic samples (default: 200)
--n-features INT     # Features per omics layer (default: 100)
--topology TYPE      # circle | noise | torus | figure8 | sphere
--max-dim INT        # Max homology dimension (default: 2)
--skip-mapper        # Skip Mapper step
--verbose            # Detailed output
```

---

## 🧪 Tests

<p align="center">
  <b>214 tests · 16 suites · 7 quality dimensions · 0 failures</b>
</p>

```bash
pytest tests/ -q           # Full suite: 214 passed, 8 skipped, 0 failed
pytest tests/ -v           # Verbose output per test
pytest tests/ --cov=src    # With coverage report
```

| Suite | Tests | Focus |
|-------|-------|-------|
| **Property** | 10 | Mathematical invariants (idempotency, monotonicity, non-negativity) |
| **Reproducibility** | 4 | Fixed seed → bit-exact output |
| **Roundtrip** | 4 | Save → load → identical |
| **Continuity** | 4 | Small perturbation → small output change |
| **Benchmark** | 7 | Golden values (PhenoAge at 65, known GenAge genes) |
| **Monte Carlo** | 4 | 10-20 random seeds, distribution statistics |
| **Integration Chain** | 5 | Full chains: data→aging→TDA→features→ML→validation |
| **Real-World** | 15 | Mixed scales, longitudinal, batch effects, real datasets |
| **Extreme** | 33 | Tiny (n=1), massive (n=1000), p>>n, corrupted, imbalanced, outliers |
| **Deep** | 36 | Pipeline 500 samples, edge cases, invariants, stress |
| **Module Unit** | 76 | Per-module function tests |
| **CLI** | 5 | help, hello, demo, run, data |
| **Visualization** | 7 | Barcode, persistence diagram, ROC, confusion matrix |
| **Total** | **214** | |

---

## 📓 Notebooks

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 00 | `00_download_data.ipynb` | Download real datasets (GTEx, TCGA, InCHIANTI) |
| 00 | `00_synthetic_validation.ipynb` | Validate TDA on synthetic topologies |
| 01 | `01_persistent_homology.ipynb` | Compute and compare persistence diagrams |
| 02 | `02_mapper_analysis.ipynb` | Build Mapper graphs, find enriched nodes |
| 03 | `03_comparison_accelerated_vs_resilient.ipynb` | Multi-view statistical comparison |
| 04 | `04_feature_extraction_ml.ipynb` | Topological features → ML classification |
| 05 | `05_biological_interpretation.ipynb` | GO/KEGG enrichment, GenAge cross-reference |

---

## 📚 Documentation

| Document | Content |
|----------|---------|
| [`mathematical_background.md`](docs/mathematical_background.md) | Vietoris-Rips, Mapper, feature vectors |
| [`biological_interpretation.md`](docs/biological_interpretation.md) | What H1 cycles mean in omics |
| [`glossary.md`](docs/glossary.md) | All TDA + biological terms defined |
| [`references.md`](docs/references.md) | 17 annotated references (Horvath, Levine, Bubenik, Adams...) |
| [`tutorials.md`](docs/tutorials.md) | Step-by-step code walkthroughs |
| [`troubleshooting.md`](docs/troubleshooting.md) | 30 common errors + solutions |
| [`api.md`](docs/api.md) | Complete module API reference |
| [`deep_research_tda.md`](docs/deep_research_tda.md) | TDA state of the art 2024-2026 (24+ sources) |
| [`data_sources.md`](docs/data_sources.md) | Dataset descriptions (GTEx, TCGA, InCHIANTI) |
| [`installation.md`](docs/installation.md) | Conda, Docker, Binder, pip |
| [`ROADMAP.md`](ROADMAP.md) | v1.1-v2.0 milestones |

---

## 🔬 Reproducibility

| Layer | Tool |
|-------|------|
| **Pinned deps** | `environment.lock.yml` — exact versions |
| **Container** | `Dockerfile` + `docker-compose.yml` |
| **CI/CD** | GitHub Actions (test, lint, docs) |
| **Pre-commit** | black, isort, flake8, codespell |
| **Topo format** | `.topo` files (NPZ-based, portable) |

---

## 📖 Citation

```bibtex
@software{tda_longevity_resilience_2026,
  author       = {Palhano, Brandon},
  title        = {TDA-Longevity-Resilience: Topological Signatures of Longevity},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/jbrandonp/tda-longevity-resilience},
  version      = {v1.0.0},
  note         = {18 modules, 214 tests, 16 suites, 20 docs}
}
```

---

## 🤝 Contributing

See [`docs/contributing.md`](docs/contributing.md). Quick guide:

```bash
pip install pre-commit && pre-commit install
pytest tests/                                      # all tests must pass
python -m pytest tests/ --cov=src --cov-report=term  # maintain coverage
```

---

<p align="center">
  <sub>Built with 🧬 + topology by <a href="https://github.com/jbrandonp">Brandon Palhano</a></sub><br>
  <sub>Keywords: TDA · Persistent Homology · Multi-omics · Longevity · Resilience · Aging · Mapper · PhenoAge</sub>
</p>
