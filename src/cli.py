#!/usr/bin/env python3
"""TDA-Longevity-Resilience — unified command-line interface.

Usage:
    python -m tda_longevity run              # Full pipeline (synthetic data)
    python -m tda_longevity run --real       # With real dataset download
    python -m tda_longevity demo             # Quick 30-second demo
    python -m tda_longevity data             # Download/generate datasets
    python -m tda_longevity tda              # TDA analysis only
    python -m tda_longevity ml               # ML classification only
    python -m tda_longevity report           # Generate validation report

The `run` command executes the complete pipeline:
    1. Generate/load multi-omics data
    2. Compute aging scores (PhenoAge)
    3. Persistent homology per omics layer
    4. Mapper graph construction
    5. Topological feature extraction
    6. ML classification (no data leakage)
    7. Biological interpretation
    8. Validation report
"""
import sys, os, argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on path when run as script
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from config import RANDOM_SEED, CV_FOLDS
from logging_config import get_logger
logger = get_logger("cli")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ═══════════════════════════════════════════════════════════════════════════════

def step_data(args):
    """Step 1: Generate or load multi-omics data."""
    logger.info("=" * 60)
    logger.info("STEP 1/7: Data Generation")
    logger.info("=" * 60)

    from data_utils import generate_synthetic_multimics, preprocess_omics
    from aging_scores import assign_acceleration_group

    t0 = time.time()
    ds = generate_synthetic_multimics(
        n_samples=args.n_samples,
        topology_type=args.topology,
        noise=args.noise,
        n_features=args.n_features,
    )

    # ── Bio-marker simulation for PhenoAge (Levine et al. 2018, Aging 10(4):573-591)
    # Reference ranges from PhenoAge publication:
    #   albumin 3.5-5.0 g/dL (mean 4.2, sd 0.35) — normal distribution
    #   creatinine 0.5-1.2 mg/dL (mean 0.9, sd 0.25) — normal distribution  
    #   glucose 70-110 mg/dL (mean 95, sd 15) — normal distribution
    #   CRP 0-10 mg/L (median 1.5, lognormal) — lognormal distribution
    #   lymphocytes 20-40% (mean 30, sd 7) — normal distribution
    #   MCV 80-100 fL (mean 90, sd 5) — normal distribution
    #   RDW 11.5-14.5% (mean 13, sd 5) — normal distribution
    #   ALP 44-147 IU/L (mean 80, sd 30) — normal distribution
    #   WBC 4-11 ×10³/µL (mean 7, sd 2) — lognormal distribution
    rng = np.random.default_rng(RANDOM_SEED)
    n = args.n_samples
    meta = ds["metadata"].copy()
    meta["age"] = np.clip(rng.normal(65, 12, n), 30, 95).astype(int)
    meta["sex"] = rng.binomial(1, 0.5, n)
    meta["albumin"] = np.clip(rng.normal(4.2, 0.35, n), 3.0, 5.5)
    meta["creatinine"] = np.clip(rng.normal(0.9, 0.25, n), 0.4, 2.0)
    meta["glucose"] = np.clip(rng.normal(95, 15, n), 60, 200)
    meta["crp"] = np.clip(rng.lognormal(np.log(1.5), 0.6, n), 0.01, 15)    # ref: median 1.5 mg/L
    meta["lymph"] = np.clip(rng.normal(30, 7, n), 10, 55)
    meta["mcv"] = np.clip(rng.normal(90, 5, n), 70, 110)
    meta["rdw"] = np.clip(rng.normal(13, 5, n), 11, 20)
    meta["alp"] = np.clip(rng.normal(80, 30, n), 30, 200)
    meta["wbc"] = np.clip(rng.lognormal(np.log(7), 0.3, n), 2, 20)

    meta = assign_acceleration_group(meta)
    ds["metadata"] = meta

    layers = {
        name: preprocess_omics(pd.DataFrame(arr), method="standard")
        for name, arr in ds.items()
        if isinstance(arr, np.ndarray) and arr.ndim == 2
    }

    elapsed = time.time() - t0
    logger.info(f"Generated {n} samples × {len(layers)} layers in {elapsed:.1f}s")
    logger.info(f"Aging groups: {meta['aging_group'].value_counts().to_dict()}")

    return ds, layers, meta


def step_tda(layers, meta, args):
    """Step 2-3: Persistent homology + Mapper."""
    logger.info("=" * 60)
    logger.info("STEP 2/7: Persistent Homology (all layers)")
    logger.info("=" * 60)
    t0 = time.time()

    from tda_utils import compute_all_layers_dgms, diagnose_persistence
    dgms = compute_all_layers_dgms(layers, max_dim=args.max_dim)

    for name, dgm in dgms.items():
        diag = diagnose_persistence(dgm)
        for dim, info in diag.items():
            logger.info(f"  {name} {dim}: {info['n_features']} features, "
                        f"lifetime={info['mean_lifetime']:.4f}")

    logger.info(f"Persistence computed in {time.time()-t0:.1f}s")

    # Mapper
    if args.skip_mapper:
        return dgms, None

    logger.info("=" * 60)
    logger.info("STEP 3/7: Mapper Graph")
    logger.info("=" * 60)
    t0 = time.time()

    from data_utils import integrate_multiomics
    from mapper_utils import auto_mapper, enrich_mapper_nodes

    integrated = integrate_multiomics(layers, method="concat")
    graph = auto_mapper(integrated, labels=meta["aging_group"], verbose=False)
    enrichment = enrich_mapper_nodes(graph, meta["aging_group"])

    n_nodes = len(graph.get("nodes", {}))
    n_resil = len(enrichment[enrichment["dominant_group"] == "resilient"]) if len(enrichment) > 0 else 0
    logger.info(f"Mapper: {n_nodes} nodes, {n_resil} resilient-enriched "
                f"({time.time()-t0:.1f}s)")

    return dgms, graph


def step_ml(layers, meta, dgms, args):
    """Step 4-5: Feature extraction + ML classification."""
    logger.info("=" * 60)
    logger.info("STEP 4/7: Topological Feature Extraction")
    logger.info("=" * 60)
    t0 = time.time()

    from data_utils import integrate_multiomics
    from features import extract_all_features
    from ml_utils import prepare_topological_features_cv, compare_with_baseline

    # Build per-sample diagram list (each layer → one diagram per individual)
    integrated = integrate_multiomics(layers, method="concat")
    y = (meta["aging_group"] != "normal").astype(int).values  # binary: resilient vs rest

    logger.info("STEP 5/7: ML Classification (CV, no leakage)")
    result = prepare_topological_features_cv(
        integrated, y,
        n_splits=args.cv_folds,
        maxdim=args.max_dim,
        verbose=True,
    )
    logger.info(f"Topological AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f} "
                f"({time.time()-t0:.1f}s)")

    return result


def step_biology(dgms, layers, meta, args):
    """Step 6: Biological interpretation."""
    logger.info("=" * 60)
    logger.info("STEP 6/7: Biological Interpretation")
    logger.info("=" * 60)

    from bio_enrichment import extract_cycle_genes, run_enrichment, cross_reference_genage

    # Get transcriptomics layer
    trans_key = next((k for k in layers if "trans" in k.lower()), list(layers.keys())[0])
    data = layers[trans_key]

    cycles = extract_cycle_genes(dgms.get(trans_key, []), data, dim=1)
    if cycles:
        all_genes = list(set().union(*cycles))[:50]
        enrichment = run_enrichment(all_genes)
        genage_hits = cross_reference_genage(all_genes)
        logger.info(f"  H1 cycles: {len(cycles)}")
        logger.info(f"  Enriched pathways: {len(enrichment)}")
        logger.info(f"  GenAge overlap: {len(genage_hits)} genes")
    else:
        logger.info("  No persistent H1 cycles detected")

    return {"n_cycles": len(cycles), "n_enriched": 0}


def step_validation(layers, meta, ml_result, args):
    """Step 7: Hold-out validation."""
    logger.info("=" * 60)
    logger.info("STEP 7/7: Validation Report")
    logger.info("=" * 60)

    from data_utils import integrate_multiomics
    from validation import create_discovery_validation, generate_validation_report

    integrated = integrate_multiomics(layers, method="concat")
    df = pd.DataFrame(integrated)

    df_d, df_v, meta_d, meta_v = create_discovery_validation(
        df, meta, test_size=0.2
    )

    report = generate_validation_report(
        {"roc_auc": {"mean": ml_result["mean_auc"], "std": ml_result["std_auc"]}},
        {"auc": ml_result["mean_auc"], "accuracy": 0.0, "precision": 0.0,
         "recall": 0.0, "f1": 0.0, "confusion_matrix": [[0,0],[0,0]],
         "n_samples": len(df_v), "overfitting_warning": "CV-only — holdout pending"},
        "TDA-Longevity Pipeline",
    )
    print(report)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_run(args):
    """Full pipeline: data → TDA → ML → biology → report."""
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  TDA-Longevity-Resilience — Full Pipeline               ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    t_start = time.time()

    ds, layers, meta = step_data(args)
    dgms, graph = step_tda(layers, meta, args)
    ml_result = step_ml(layers, meta, dgms, args)
    bio_result = step_biology(dgms, layers, meta, args)
    step_validation(layers, meta, ml_result, args)

    total = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE — {total:.1f}s total")
    logger.info(f"  Topological CV AUC: {ml_result['mean_auc']:.4f} ± {ml_result['std_auc']:.4f}")
    logger.info(f"  H1 cycles detected: {bio_result['n_cycles']}")
    logger.info(f"{'='*60}")


def cmd_demo(args):
    """Quick demo: small synthetic dataset, fast run."""
    logger.info("Quick Demo — 30-second TDA pipeline")
    from data_utils import generate_synthetic_multimics, preprocess_omics, integrate_multiomics
    from tda_utils import compute_all_layers_dgms, diagnose_persistence
    from features import extract_all_features

    t0 = time.time()
    ds = generate_synthetic_multimics(n_samples=100, topology_type="circle", noise=0.05, n_features=30)
    layers = {k: preprocess_omics(pd.DataFrame(v), method="standard")
              for k, v in ds.items() if isinstance(v, np.ndarray) and v.ndim == 2}

    dgms = compute_all_layers_dgms(layers, max_dim=1)
    for name, dgm in dgms.items():
        diag = diagnose_persistence(dgm)
        logger.info(f"  {name} H1: {diag['H1']['n_features']} features, "
                    f"lifetime={diag['H1']['mean_lifetime']:.4f}")

    logger.info(f"Demo complete in {time.time()-t0:.1f}s ✅")


def cmd_hello(args):
    """30-second hello world: circle → persistence → explanation."""
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    logger.info("=" * 55)
    logger.info("  TDA-Longevity-Resilience — Hello World")
    logger.info("=" * 55)

    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    logger.info(f"Generated circle: {circle.shape[0]} points in 2D (has H1 hole)")

    dist_vec = pdist(circle, metric="euclidean")
    dist_matrix = squareform(dist_vec)
    logger.info(f"Distance matrix: {dist_matrix.shape}, range [{dist_vec.min():.3f}, {dist_vec.max():.3f}]")

    try:
        import ripser
        dgms = ripser.ripser(dist_matrix, maxdim=1, distance_matrix=True)["dgms"]
        h0 = dgms[0][np.isfinite(dgms[0][:, 1])]
        h1 = dgms[1][np.isfinite(dgms[1][:, 1])]
        logger.info(f"H0 (components): {len(h0)} features")
        logger.info(f"H1 (cycles): {len(h1)} features")
        for birth, death in h1:
            logger.info(f"  H1 cycle: birth={birth:.3f}, death={death:.3f}, lifetime={death-birth:.3f}")
        if len(h1) >= 1:
            logger.info("FOUND: 1 persistent cycle — the circle's hole!")
        else:
            logger.warning("No H1 cycle detected")
    except ImportError:
        logger.warning("ripser not installed. Install: pip install ripser")


def cmd_data(args):
    """Download or generate datasets."""
    logger.info("Data management — use scripts/download_data.py for real datasets")
    from data_utils import generate_synthetic_multimics
    ds = generate_synthetic_multimics(
        n_samples=args.n_samples,
        topology_type=args.topology,
        noise=args.noise,
        n_features=args.n_features,
    )
    logger.info(f"Generated {args.n_samples} samples")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TDA-Longevity-Resilience CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tda_longevity run                     # Full pipeline (synthetic)
  python -m tda_longevity run --n-samples 500     # Larger dataset
  python -m tda_longevity demo                    # Quick demo
  python -m tda_longevity tda --max-dim 2         # TDA only, H2 included
""",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # Common args for data generation
    def add_data_args(p):
        p.add_argument("--n-samples", type=int, default=200)
        p.add_argument("--topology", default="circle",
                       choices=["circle", "torus", "figure8", "sphere", "noise"])
        p.add_argument("--noise", type=float, default=0.08)
        p.add_argument("--n-features", type=int, default=100)

    def add_tda_args(p):
        p.add_argument("--max-dim", type=int, default=1)
        p.add_argument("--cv-folds", type=int, default=5)
        p.add_argument("--skip-mapper", action="store_true")

    # run
    p_run = sub.add_parser("run", help="Full pipeline")
    add_data_args(p_run)
    add_tda_args(p_run)

    # demo
    sub.add_parser("demo", help="Quick 30-second demo")

    # hello
    sub.add_parser("hello", help="Circle→persistence hello world")

    # data
    p_data = sub.add_parser("data", help="Generate synthetic dataset")
    add_data_args(p_data)

    # tda
    p_tda = sub.add_parser("tda", help="TDA analysis only")
    add_data_args(p_tda)
    add_tda_args(p_tda)

    # ml
    p_ml = sub.add_parser("ml", help="ML classification only")
    add_data_args(p_ml)
    add_tda_args(p_ml)

    # report
    sub.add_parser("report", help="Generate validation report")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "run": cmd_run,
        "demo": cmd_demo,
        "hello": cmd_hello,
        "data": cmd_data,
        "tda": lambda a: cmd_run(a),  # simplified
        "ml": lambda a: cmd_run(a),
        "report": lambda a: logger.info("Report generation — see step_validation()"),
    }

    fn = commands.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
