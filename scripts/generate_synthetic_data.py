#!/usr/bin/env python
"""Standalone synthetic multi-omics data generator for TDA-Longevity-Resilience.

Generates datasets with known topological structures (circle, torus, figure-8,
sphere, noise) for validating TDA pipelines before applying to real data.

Usage:
    python scripts/generate_synthetic_data.py --topology circle --n-samples 200 --noise 0.05 --out data/raw/

Options:
    --topology    One of: circle, torus, figure8, sphere, noise (default: circle)
    --n-samples   Number of synthetic individuals (default: 200)
    --n-features  Ambient dimension per omics layer (default: 50)
    --noise       Gaussian noise standard deviation (default: 0.05)
    --seed        Random seed for reproducibility (default: 42)
    --out         Output directory (default: data/raw/)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def _assign_groups(tian_score, accel_thresh=0.8, resil_thresh=-0.8):
    std = np.std(tian_score)
    mean = np.mean(tian_score)
    labels = np.full(len(tian_score), "neutral", dtype=object)
    labels[tian_score > mean + accel_thresh * std] = "accelerated"
    labels[tian_score < mean + resil_thresh * std] = "resilient"
    return labels


def _lift(arr, n_feat, noise):
    proj = rng.standard_normal((arr.shape[1], n_feat)) * 0.3
    return arr @ proj + noise * rng.standard_normal((arr.shape[0], n_feat))


def generate(topology="circle", n_samples=200, n_features=50, noise=0.05, seed=42):
    """Generate synthetic multi-omics dataset with known topology."""
    global rng
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    if topology == "circle":
        base = np.column_stack([np.cos(t), np.sin(t)])
    elif topology == "torus":
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 50), np.linspace(0, 2 * np.pi, 50))
        base = np.column_stack([u.ravel(), v.ravel()])[:n_samples]
    elif topology == "figure8":
        base = np.column_stack([np.sin(t), np.sin(2 * t)])
    elif topology == "sphere":
        phi = np.arccos(1 - 2 * rng.random(n_samples))
        theta = 2 * np.pi * rng.random(n_samples)
        base = np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    elif topology == "noise":
        base = rng.standard_normal((n_samples, 3))
    else:
        raise ValueError(f"Unknown topology: {topology}. Choose: circle, torus, figure8, sphere, noise")

    layers = {
        "transcriptomics": _lift(base, n_features, noise),
        "metabolomics": _lift(base, n_features, noise),
        "epigenomics": _lift(base, n_features, noise),
    }

    tian_score = np.sum(base, axis=1) + noise * rng.standard_normal(n_samples)
    labels = _assign_groups(tian_score)

    metadata = pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(n_samples)],
        "tian_score": tian_score,
        "group": labels,
    })

    return layers, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-omics data for TDA validation")
    parser.add_argument("--topology", default="circle", choices=["circle", "torus", "figure8", "sphere", "noise"])
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.topology} data: n={args.n_samples}, feat={args.n_features}, noise={args.noise}")
    layers, metadata = generate(
        topology=args.topology,
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        seed=args.seed,
    )

    for name, arr in layers.items():
        path = out_dir / f"synthetic_{args.topology}_{name}.csv"
        pd.DataFrame(arr).to_csv(path, index=False)
        print(f"  Saved {path} ({arr.shape})")

    meta_path = out_dir / f"synthetic_{args.topology}_metadata.csv"
    metadata.to_csv(meta_path, index=False)
    print(f"  Saved {meta_path}")

    counts = metadata["group"].value_counts().to_dict()
    print(f"\nGroup distribution: {counts}")
    print("Done.")


if __name__ == "__main__":
    main()
