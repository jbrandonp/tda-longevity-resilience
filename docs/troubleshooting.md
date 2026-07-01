# Troubleshooting Guide

## Installation

| Symptom | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'ripser'` | `pip install ripser` or `conda install -c conda-forge ripser` |
| `ImportError: cannot import name 'PersImage'` | `pip install persim` |
| `ModuleNotFoundError: No module named 'kmapper'` | `pip install kmapper` (Mapper requires optional dep) |
| `ImportError: libpython3.x.so` | Recreate conda env: `conda env create -f environment.lock.yml` |
| `pip install` conflicts with conda | Use `environment.lock.yml` — do NOT mix pip + conda |
| `CUDA not found` | Ripser is CPU-only; ignore CUDA warnings |

## Runtime

| Symptom | Solution |
|---------|----------|
| `MemoryError` on large dataset | Use `--n-samples 200`; for >10K samples use `sparse_persistence()` |
| `KeyError: 'aging_group'` | Run `assign_acceleration_group(meta)` before ML |
| `ValueError: n_samples=0` | `--n-samples` must be ≥ 10 |
| `FileNotFoundError: data/raw/xxx.csv` | Use synthetic: `python -m src.cli run` (no --real flag) |
| `np.quantile` on empty array | Dataset too small (<2 samples); use ≥10 |
| ripser returns empty diagrams | ripser not installed — `pip install ripser` |
| `TopoAE not fitted` | Call `.fit(X)` before `.transform(X)` |
| `ConvergenceWarning` from MLP | Normal for TopoAE with max_iter=200; ignore or increase |

## CLI

| Symptom | Solution |
|---------|----------|
| `python -m src.cli: No module named src` | Run from project root: `cd tda-longevity-resilience` |
| `--n-samples` not recognized | Use `python -m src.cli run --n-samples 100` |
| `--help` shows old commands | Pull latest: `git pull` |

## TDA / Biological

| Symptom | Solution |
|---------|----------|
| All H1 cycles have 0 lifetime | Data is noise-only; use circle topology for signal |
| PhenoAge returns NaN | Missing biomarker columns — run `meta` with all 9 biomarkers |
| Enrichment returns empty | Gene names must be HGNC symbols (e.g., "MTOR" not "mTOR") |

## GitHub / CI

| Symptom | Solution |
|---------|----------|
| CI failing on `pytest` | Check Python version: 3.10+ required |
| `gh` not authenticated | `gh auth login` |
| Push rejected (non-fast-forward) | `git pull --rebase && git push` |
