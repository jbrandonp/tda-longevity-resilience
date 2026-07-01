# 🧬 TDA-Longevity-Resilience

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-87%20passed-brightgreen.svg)]()
[![DOI](https://zenodo.org/badge/DOI/placeholder.svg)](https://zenodo.org/)

**Persistent homology unveils topological signatures of extreme longevity resilience.**

Apply **Topological Data Analysis** — persistent homology + Mapper algorithm — to multi‑omics aging data to extract structural signatures of resilience and compare them to accelerated aging.

---

## 🚀 Quick Start

```bash
git clone https://github.com/jbrandonp/tda-longevity-resilience.git
cd tda-longevity-resilience
conda env create -f environment.yml
conda activate tda-longevity

# 30-second hello world
python -m src.cli hello

# Full pipeline (synthetic data)
python -m src.cli run --n-samples 200 --max-dim 2

# Quick demo
python -m src.cli demo
```

---

## 🧭 Pipeline

```
Multi-omics Data (transcriptomics, metabolomics, epigenomics)
 │
 ├─ STEP 1  ── Data generation + PhenoAge scoring
 ├─ STEP 2  ── Persistent homology (Vietoris-Rips, Ripser)
 ├─ STEP 3  ── Mapper graph (KeplerMapper, UMAP lens)
 ├─ STEP 4  ── Topological feature extraction (PI, PL, Betti)
 ├─ STEP 5  ── ML classification (RF, SVM, GBM — CV, no leakage)
 ├─ STEP 6  ── Biological interpretation (GO, KEGG, GenAge)
 └─ STEP 7  ── Validation report
```

---

## 📦 Package

```python
from src.data_utils import generate_synthetic_multimics, integrate_multiomics
from src.tda_utils import compute_persistence_diagrams, diagnose_persistence
from src.aging_scores import phenoage_simplified, assign_acceleration_group
from src.ml_utils import TopologicalFeatureExtractor, prepare_topological_features_cv
from src.validation import create_discovery_validation, evaluate_on_holdout
```

All 50+ public functions exported — `from src import *` gives you everything.

---

## 🖥 CLI

```
python -m src.cli run       # Full pipeline (data → TDA → ML → report)
python -m src.cli demo      # Quick 30-second demo
python -m src.cli hello     # Circle → persistence hello world
python -m src.cli data      # Generate synthetic dataset
python -m src.cli tda       # TDA analysis only
python -m src.cli ml        # ML classification only
```

---

## 📓 Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 00 | `00_synthetic_validation` | Validate TDA on synthetic topologies |
| 01 | `01_persistent_homology` | Compute & compare persistence diagrams |
| 02 | `02_mapper_analysis` | Build Mapper graph, identify enriched nodes |
| 03 | `03_comparison_accelerated_vs_resilient` | Multi-view statistical comparison |
| 04 | `04_feature_extraction_ml` | Topological features → ML classification |
| 05 | `05_biological_interpretation` | GO/KEGG enrichment, GenAge cross-ref |

---

## 🧪 Tests

```bash
pytest tests/ -q    # 87 tests, 13 test files
```

| Suite | Tests | Status |
|-------|-------|--------|
| Unit (pytest) | 87 | ✅ |
| Syntax (all .py) | 30+ | ✅ |
| Imports (13 modules) | 13 | ✅ |
| CLI (help, hello, demo, run) | 4 | ✅ |
| Notebooks (JSON validity) | 6 | ✅ |

---

## 🧬 Methods

- **Persistent Homology** — Vietoris-Rips filtration via `ripser`, H0/H1/H2 diagrams
- **Mapper Algorithm** — UMAP lens, DBSCAN clustering, KeplerMapper visualization
- **Aging Scores** — PhenoAge (Levine 2018), DunedinPACE proxy, age acceleration groups
- **Features** — Persistence Images (Adams 2017), Landscapes (Bubenik 2015), Betti curves
- **ML** — RF, SVM, Gradient Boosting with CV (no data leakage), SHAP interpretability
- **Validation** — Stratified hold-out, Wasserstein distances, permutation tests
- **Bio Enrichment** — Fisher's exact test, Gene Ontology, KEGG, GenAge cross-reference

---

## 📚 Docs

| Document | Content |
|----------|---------|
| `mathematical_background.md` | Persistent homology, Vietoris-Rips, Mapper, feature vectors |
| `biological_interpretation.md` | What H1 cycles mean in omics, resilience vs aging |
| `glossary.md` | All TDA + biological terms defined |
| `references.md` | Annotated bibliography (17 references) |
| `tutorials.md` | Step-by-step code tutorials |
| `api.md` | Module API reference |
| `data_sources.md` | Dataset descriptions and access |
| `installation.md` | Conda, Docker, Binder setup |
| `ROADMAP.md` | Future features and milestones |

---

## 🔬 Reproducibility

- **Pinned environment:** `environment.lock.yml` — exact versions for every dependency
- **Docker:** `docker-compose up` for zero-install Jupyter Lab
- **CI/CD:** GitHub Actions for tests, lint, docs
- **Pre-commit:** black, isort, flake8

---

## 📖 Citation

```bibtex
@software{tda_longevity_resilience_2026,
  author = {Palhano Machado, Brandon},
  title = {TDA-Longevity-Resilience: Topological Signatures of Extreme Longevity},
  year = {2026},
  url = {https://github.com/jbrandonp/tda-longevity-resilience},
}
```

---

**Keywords:** Topological Data Analysis, Persistent Homology, Multi-omics, Longevity, Resilience, Aging, Mapper Algorithm, Persistence Images, PhenoAge
