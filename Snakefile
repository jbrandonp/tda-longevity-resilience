"""
Snakemake workflow for TDA-Longevity-Resilience.

One-command full pipeline execution:
    snakemake -j4

Individual targets:
    snakemake synthetic      # Generate synthetic data
    snakemake persistence    # Compute persistence diagrams
    snakemake mapper         # Build Mapper graph
    snakemake classify       # Train topological classifier
    snakemake all            # Run everything

Configuration: edit config/snakemake_config.yaml
"""

import os

# ── Configuration ───────────────────────────────────────────────────────────
configfile: "config/snakemake_config.yaml"

# ── Rules ───────────────────────────────────────────────────────────────────

rule all:
    input:
        "results/tables/classification_report.csv",
        "results/figures/mapper_graph.html",
        "results/figures/barcode_H1.png",


rule synthetic:
    """Generate synthetic multi-omics data with known topology."""
    output:
        transcriptomics="data/raw/synthetic_{topology}_transcriptomics.csv",
        metabolomics="data/raw/synthetic_{topology}_metabolomics.csv",
        epigenomics="data/raw/synthetic_{topology}_epigenomics.csv",
        metadata="data/raw/synthetic_{topology}_metadata.csv",
    params:
        topology=config.get("topology", "circle"),
        n_samples=config.get("n_samples", 200),
        n_features=config.get("n_features", 50),
        noise=config.get("noise", 0.05),
        seed=config.get("seed", 42),
    shell:
        """
        python -m src.cli synthetic \
            --topology {params.topology} \
            --n-samples {params.n_samples} \
            --n-features {params.n_features} \
            --noise {params.noise} \
            --output-dir data/raw/
        """


rule persistence:
    """Compute Vietoris-Rips persistence diagrams."""
    input:
        transcriptomics="data/raw/synthetic_{topology}_transcriptomics.csv",
    output:
        barcode="results/figures/barcode_H1.png",
    params:
        topology=config.get("topology", "circle"),
        max_dim=config.get("max_dim", 2),
    shell:
        """
        python -m src.cli persistence \
            --input {input.transcriptomics} \
            --max-dim {params.max_dim} \
            --output {output.barcode}
        """


rule mapper:
    """Build Mapper graph from integrated multi-omics."""
    input:
        transcriptomics="data/raw/synthetic_{topology}_transcriptomics.csv",
        metabolomics="data/raw/synthetic_{topology}_metabolomics.csv",
        epigenomics="data/raw/synthetic_{topology}_epigenomics.csv",
        metadata="data/raw/synthetic_{topology}_metadata.csv",
    output:
        graph="results/figures/mapper_graph.html",
    params:
        topology=config.get("topology", "circle"),
    shell:
        """
        python -m src.cli mapper \
            --input {input.transcriptomics} \
            --labels {input.metadata} \
            --output {output.graph}
        """


rule classify:
    """Train topological ML classifier and produce report."""
    input:
        transcriptomics="data/raw/synthetic_{topology}_transcriptomics.csv",
        metadata="data/raw/synthetic_{topology}_metadata.csv",
    output:
        report="results/tables/classification_report.csv",
    params:
        topology=config.get("topology", "circle"),
    run:
        import pandas as pd
        import numpy as np
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

        from src.data_utils import preprocess_omics, assign_groups_from_tian_score
        from src.tda_utils import compute_persistence_diagrams
        from src.features import extract_all_features
        from src.ml_utils import build_topological_pipeline, evaluate_topological_model
        from sklearn.ensemble import RandomForestClassifier

        # Load and preprocess
        df = pd.read_csv(input.transcriptomics)
        data = preprocess_omics(df)
        meta = pd.read_csv(input.metadata)
        labels = assign_groups_from_tian_score(meta) if "tian_score" in meta.columns else meta["group"]

        # Filter binary
        mask = (labels == "accelerated") | (labels == "resilient")
        X = data[mask.values]
        y = (labels[mask.values] == "resilient").astype(int).values

        # Compute persistence
        dgms = compute_persistence_diagrams(X, max_dim=2)

        # Extract features
        feats = extract_all_features([dgms] * len(y))
        X_topo = feats["persistence_images"]

        # Classify
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        pipe = build_topological_pipeline(clf)
        results = evaluate_topological_model(pipe, X_topo, y)

        # Save report
        rows = []
        for metric, vals in results.items():
            rows.append({"metric": metric, "mean": vals["mean"], "std": vals["std"]})
        pd.DataFrame(rows).to_csv(output.report, index=False)
        print(f"Classification report saved to {output.report}")


rule clean:
    """Remove all generated outputs."""
    shell:
        """
        rm -rf results/figures/*.png results/figures/*.html results/tables/*.csv
        rm -rf data/raw/synthetic_*.csv
        echo "Cleaned."
        """
