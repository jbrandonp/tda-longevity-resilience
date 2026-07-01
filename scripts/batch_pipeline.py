#!/usr/bin/env python
"""Batch pipeline for cohort-scale TDA — compute per individual, cache, aggregate.

Pattern:
    1. Split cohort into batches
    2. Per sample: preprocess → persistence diagram → cache (.pkl)
    3. Aggregate: load cached diagrams → cohort-level statistics

Usage:
    python scripts/batch_pipeline.py --input data.csv --metadata meta.csv --batch-size 100
"""

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import preprocess_omics, assign_groups_from_tian_score
from src.tda_utils import compute_persistence_diagrams, diagnose_persistence
from src.config import RANDOM_SEED

CACHE_DIR = Path("data/processed/batch_cache")
OUTPUT_DIR = Path("results/batch")


def _cache_path(sample_id: str, layer: str, params_hash: str) -> Path:
    """Deterministic cache path for a persistence diagram."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"dgm_{sample_id}_{layer}_{params_hash}.pkl"


def _params_hash(max_dim: int, metric: str, threshold: float) -> str:
    h = hashlib.sha256()
    h.update(f"{max_dim}_{metric}_{threshold}".encode())
    return h.hexdigest()[:12]


def process_batch(
    data: np.ndarray,
    sample_ids: list,
    batch_idx: int,
    max_dim: int,
    metric: str,
    threshold: float,
    layer: str = "transcriptomics",
) -> list:
    """Compute persistence diagrams for a batch of samples, caching each result.

    Returns:
        list of (sample_id, diagram) tuples.
    """
    phash = _params_hash(max_dim, metric, threshold)
    results = []

    for i, sid in enumerate(sample_ids):
        cache_path = _cache_path(str(sid), layer, phash)

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                dgm = pickle.load(f)
        else:
            # Per-individual: use a small neighborhood for stability
            # In practice, this would be the individual's own omics vector
            # For batch processing, we treat each row as an individual
            sample_data = data[i:i + 1]  # single sample
            if sample_data.shape[0] < 3:
                # Bootstrap: resample the point with noise for minimum viable point cloud
                sample_data = np.tile(sample_data, (10, 1)) + np.random.randn(10, sample_data.shape[1]) * 0.01

            dgm = compute_persistence_diagrams(sample_data, max_dim=max_dim, metric=metric, use_cache=False)

            with open(cache_path, "wb") as f:
                pickle.dump(dgm, f)

        results.append((sid, dgm))

    n_cached = sum(1 for sid, _ in results if _cache_path(str(sid), layer, phash).exists())
    print(f"  Batch {batch_idx}: {len(results)} samples ({n_cached} cached)")
    return results


def aggregate_cohort(diagrams: list) -> dict:
    """Aggregate per-individual diagrams into cohort-level statistics."""
    all_diag = []
    for sid, dgm in diagrams:
        diag = diagnose_persistence(dgm)
        diag["sample_id"] = sid
        all_diag.append(diag)

    df = pd.DataFrame(all_diag)

    summary = {}
    for dim_key in ["H0", "H1", "H2"]:
        if dim_key in df.columns or any(dim_key in str(c) for c in df.columns):
            nfeat_col = [c for c in df.columns if "n_features" in str(c)]
            lifetime_col = [c for c in df.columns if "mean_lifetime" in str(c)]
            if nfeat_col:
                summary[f"{dim_key}_n_features_mean"] = float(df[nfeat_col[0]].mean())
                summary[f"{dim_key}_n_features_std"] = float(df[nfeat_col[0]].std())
            if lifetime_col:
                summary[f"{dim_key}_lifetime_mean"] = float(df[lifetime_col[0]].mean())

    summary["n_samples"] = len(diagrams)
    summary["n_cached"] = len(list(CACHE_DIR.glob("*.pkl")))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch TDA pipeline for large cohorts")
    parser.add_argument("--input", required=True, help="CSV with omics data")
    parser.add_argument("--metadata", help="CSV with sample metadata")
    parser.add_argument("--batch-size", type=int, default=100, help="Samples per batch")
    parser.add_argument("--max-dim", type=int, default=1)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output", default="results/batch/cohort_summary.json")
    args = parser.parse_args()

    print(f"[batch] Loading: {args.input}")
    df = pd.read_csv(args.input)
    data = preprocess_omics(df, method="standard")
    n_samples = data.shape[0]

    # Split into batches
    n_batches = max(1, n_samples // args.batch_size)
    sample_ids = list(range(n_samples))
    batches = np.array_split(sample_ids, n_batches)

    print(f"[batch] {n_samples} samples → {len(batches)} batches (batch_size={args.batch_size})")

    all_diagrams = []
    for batch_idx, batch_ids in enumerate(batches):
        batch_data = data[batch_ids]
        results = process_batch(
            batch_data,
            [f"S{i:06d}" for i in batch_ids],
            batch_idx,
            args.max_dim,
            args.metric,
            args.threshold,
        )
        all_diagrams.extend(results)

    # Aggregate
    print(f"\n[batch] Aggregating {len(all_diagrams)} diagrams...")
    summary = aggregate_cohort(all_diagrams)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import json

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[batch] Complete — {summary['n_samples']} samples processed")
    print(f"[batch] Cache: {summary['n_cached']} diagrams in {CACHE_DIR}")
    print(f"[batch] Summary: {output_path}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
