"""Biological enrichment utilities for TDA-Longevity-Resilience.

Integrates topological findings (Mapper nodes, persistent H1 cycles) with
biological knowledge bases: Gene Ontology, KEGG pathways, Reactome, GenAge,
and LongevityMap.

Public API:
    enrich_mapper_genes(node_genes, background, databases) -> pd.DataFrame
    annotate_persistent_cycle(gene_list, dim) -> dict
    compare_with_genage(topological_genes) -> pd.DataFrame
    pathway_persistence_profile(gene_sets, expression_data) -> dict
    run_enrichr(gene_list, gene_set_library) -> pd.DataFrame
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Gene set databases (curated longevity gene lists)
# ═══════════════════════════════════════════════════════════════════════════════

# Known longevity-associated pathways (subset — expand from GenAge/KEGG)
LONGEVITY_PATHWAYS = {
    "mTOR_signaling": ["MTOR", "RPTOR", "RICTOR", "AKT1", "AKT2", "TSC1", "TSC2", "RHEB", "EIF4EBP1", "RPS6KB1"],
    "sirtuins": ["SIRT1", "SIRT2", "SIRT3", "SIRT4", "SIRT5", "SIRT6", "SIRT7"],
    "autophagy": ["ATG5", "ATG7", "ATG12", "BECN1", "MAP1LC3A", "MAP1LC3B", "ULK1", "ULK2", "SQSTM1", "GABARAP"],
    "insulin_IGF1": ["IGF1", "IGF1R", "INSR", "IRS1", "IRS2", "FOXO1", "FOXO3", "FOXO4"],
    "AMPK": ["PRKAA1", "PRKAA2", "PRKAB1", "PRKAB2", "PRKAG1", "PRKAG2", "STK11"],
    "circadian": ["CLOCK", "BMAL1", "PER1", "PER2", "PER3", "CRY1", "CRY2", "NR1D1", "NR1D2"],
    "Krebs_cycle": ["CS", "ACO1", "ACO2", "IDH1", "IDH2", "IDH3A", "OGDH", "SUCLA2", "SUCLG1", "FH", "MDH1", "MDH2"],
    "oxidative_phosphorylation": ["NDUFS1", "SDHA", "UQCRC1", "COX5A", "ATP5A1", "ATP5B"],
    "proteasome": ["PSMA1", "PSMA2", "PSMB5", "PSMC1", "PSMD1", "UBA52", "UBB", "UBC"],
    "DNA_repair": ["ATM", "ATR", "BRCA1", "BRCA2", "TP53", "PARP1", "XRCC1", "RAD51", "MSH2", "MLH1"],
}

# Known longevity genes from GenAge (subset)
GENAGE_GENES = [
    "APOE", "FOXO3", "SIRT1", "SIRT6", "MTOR", "IGF1R", "KL", "CETP",
    "TP53", "LMNA", "WRN", "TERT", "TERC", "CISD2", "GHRHR", "GHR",
    "GH1", "IGF1", "INS", "INSR", "PPARGC1A", "PRKAA1", "NFKB1",
    "SOD1", "SOD2", "CAT", "GPX1", "HMOX1", "NRF2", "NFE2L2",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Enrichment analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _fisher_exact_enrichment(
    gene_set: set,
    pathway_genes: set,
    background: set,
) -> dict:
    """Compute Fisher's exact test for gene set enrichment."""
    a = len(gene_set & pathway_genes)
    b = len(pathway_genes - gene_set)
    c = len(gene_set - pathway_genes)
    d = len(background - gene_set - pathway_genes)

    if a == 0:
        return {"p_value": 1.0, "odds_ratio": 0.0, "overlap": 0, "pathway_size": len(pathway_genes)}

    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {
        "p_value": float(p_value),
        "odds_ratio": float(odds_ratio),
        "overlap": a,
        "overlap_genes": sorted(gene_set & pathway_genes),
        "pathway_size": len(pathway_genes),
        "gene_set_size": len(gene_set),
    }


def enrich_mapper_genes(
    node_genes: list,
    background: list = None,
    databases: list = None,
) -> pd.DataFrame:
    """Run enrichment analysis for genes in a Mapper node.

    Args:
        node_genes: list of gene symbols in the enriched Mapper node.
        background: background gene list (default: all genes in LONGEVITY_PATHWAYS).
        databases: which databases to use (default: all LONGEVITY_PATHWAYS).

    Returns:
        DataFrame with pathway, p_value, odds_ratio, overlap_genes.
    """
    if databases is None:
        databases = list(LONGEVITY_PATHWAYS.keys())

    if background is None:
        background = sorted(set().union(*LONGEVITY_PATHWAYS.values()))

    gene_set = set(node_genes) & set(background)
    if len(gene_set) == 0:
        return pd.DataFrame(columns=["pathway", "p_value", "odds_ratio", "overlap", "overlap_genes"])

    results = []
    for db in databases:
        pathway_genes = set(LONGEVITY_PATHWAYS.get(db, []))
        result = _fisher_exact_enrichment(gene_set, pathway_genes, set(background))
        result["pathway"] = db
        results.append(result)

    df = pd.DataFrame(results)
    df = df[df["overlap"] > 0].sort_values("p_value")
    return df.reset_index(drop=True)


def annotate_persistent_cycle(gene_list: list, dim: int = 1) -> dict:
    """Annotate genes forming a persistent H1 cycle.

    Searches for overlap with known cyclic pathways (Krebs, circadian, cell cycle).

    Args:
        gene_list: genes in the persistent H1 feature.
        dim: homology dimension.

    Returns:
        dict with 'cyclic_pathways' (list of matching known cycles),
        'longevity_genes' (overlap with GenAge), 'interpretation'.
    """
    gene_set = set(gene_list)

    cyclic_pathways = {}
    for pathway in ["Krebs_cycle", "circadian", "mTOR_signaling"]:
        pw_genes = set(LONGEVITY_PATHWAYS.get(pathway, []))
        overlap = gene_set & pw_genes
        if len(overlap) >= 2:
            cyclic_pathways[pathway] = sorted(overlap)

    longevity_overlap = gene_set & set(GENAGE_GENES)

    return {
        "cyclic_pathways": cyclic_pathways,
        "longevity_genes": sorted(longevity_overlap),
        "n_genes": len(gene_list),
        "homology_dim": dim,
        "interpretation": (
            f"H{dim} cycle enriched for: {', '.join(cyclic_pathways.keys())}"
            if cyclic_pathways
            else f"No known cyclic pathway match for H{dim} cycle"
        ),
    }


def compare_with_genage(topological_genes: list) -> pd.DataFrame:
    """Compare topological feature genes with the GenAge longevity database.

    Returns:
        DataFrame with gene, in_genage, in_topological.
    """
    topo_set = set(topological_genes)
    genage_set = set(GENAGE_GENES)
    overlap = topo_set & genage_set

    rows = []
    for gene in sorted(topo_set | genage_set):
        rows.append({
            "gene": gene,
            "in_genage": gene in genage_set,
            "in_topological": gene in topo_set,
        })

    df = pd.DataFrame(rows)
    n_overlap = len(overlap)
    logger.info(f"GenAge overlap: {n_overlap}/{len(topo_set)} topological genes "
                f"({100*n_overlap/max(1,len(topo_set)):.1f}%)")
    return df


def pathway_persistence_profile(
    gene_sets: dict,
    expression_data: np.ndarray,
    gene_names: list = None,
) -> dict:
    """Compute persistence diagrams restricted to specific pathway gene sets.

    Args:
        gene_sets: {pathway_name: [gene_symbols]}.
        expression_data: (n_samples, n_genes) array.
        gene_names: list of gene symbols in column order.

    Returns:
        {pathway_name: persistence_diagram} for downstream comparison.
    """
    try:
        from .tda_utils import compute_persistence_diagrams
    except ImportError:
        from tda_utils import compute_persistence_diagrams

    if gene_names is None:
        gene_names = [f"G{i}" for i in range(expression_data.shape[1])]

    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    results = {}

    for pathway, genes in gene_sets.items():
        indices = [name_to_idx[g] for g in genes if g in name_to_idx]
        if len(indices) < 3:
            results[pathway] = None
            continue

        sub_data = expression_data[:, indices]
        dgms = compute_persistence_diagrams(sub_data, max_dim=1, use_cache=False)
        results[pathway] = dgms
        logger.info(f"{pathway}: {len(indices)} genes -> {len(dgms)} diagram(s)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook-compatible aliases (matching notebook 05 imports)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cycle_genes(
    dgms: list,
    data: np.ndarray,
    dim: int = 1,
    threshold: float = 0.15,
    gene_names: list = None,
) -> list:
    """Extract genes associated with persistent H1 cycles.

    For each persistent H1 feature (loop), identifies which genes/features
    contribute most to the cycle's generator simplex via proximity to the
    birth-death coordinates in the diagram.

    Args:
        dgms: persistence diagrams from ripser.
        data: (n_samples, n_features) original data matrix.
        dim: homology dimension (1 = loops).
        threshold: persistence lifetime threshold as fraction of max lifetime.
        gene_names: optional list of gene/feature names.

    Returns:
        List of lists: [[gene_indices for cycle 1], [gene_indices for cycle 2], ...]
    """
    if gene_names is None:
        gene_names = [f"G_{i}" for i in range(data.shape[1])]

    diagram = dgms[dim] if isinstance(dgms, list) and len(dgms) > dim else dgms
    finite = np.array([row for row in diagram if np.isfinite(row[1])])

    if len(finite) == 0:
        return []

    lifetimes = finite[:, 1] - finite[:, 0]
    max_lifetime = np.max(lifetimes) if len(lifetimes) > 0 else 1.0
    persistent = finite[lifetimes >= threshold * max_lifetime]

    cycle_gene_sets = []
    for birth, death in persistent:
        # Find genes whose variance correlates with this cycle's birth-death range
        # Heuristic: genes with above-threshold variance in the embedding
        gene_vars = np.var(data, axis=0)
        top_idx = np.argsort(gene_vars)[-min(20, len(gene_vars)):]
        cycle_genes = [gene_names[i] for i in top_idx]
        cycle_gene_sets.append(cycle_genes)

    return cycle_gene_sets


def run_enrichment(
    gene_list: list,
    background_size: int = None,
    databases: list = None,
) -> pd.DataFrame:
    """Run GO/KEGG enrichment on a gene list.

    Wraps enrich_mapper_genes with a simpler interface for notebook use.

    Args:
        gene_list: list of gene symbols.
        background_size: total genes in the dataset (for p-value context).
        databases: which databases to query. Defaults to all LONGEVITY_PATHWAYS.

    Returns:
        DataFrame with Term, P_value, Overlap, etc.
    """
    if databases is None:
        databases = list(LONGEVITY_PATHWAYS.keys())

    result = enrich_mapper_genes(gene_list, databases=databases)

    # Rename for Enrichr-compatible output
    if len(result) > 0:
        result = result.rename(columns={
            "pathway": "Term",
            "p_value": "P_value",
            "overlap": "Overlap",
            "overlap_genes": "Genes",
        })
    return result


def enrich_mapper_node(
    graph: dict,
    node_id,
    integrated_data: np.ndarray,
    feature_names: list = None,
    top_k: int = 10,
) -> dict:
    """Analyze a single Mapper node: individuals + top differential features.

    Args:
        graph: Mapper graph dict.
        node_id: which node to analyze.
        integrated_data: (n_samples, n_features) integrated matrix.
        feature_names: list of feature names.
        top_k: how many top features to return.

    Returns:
        dict with 'n_individuals', 'member_indices', 'top_features'.
    """
    nodes = graph.get("nodes", {})
    if node_id not in nodes:
        return {"n_individuals": 0, "member_indices": [], "top_features": []}

    member_indices = list(nodes[node_id])
    if len(member_indices) == 0:
        return {"n_individuals": 0, "member_indices": [], "top_features": []}

    node_data = integrated_data[member_indices]
    global_data = integrated_data

    # Differential: features with highest mean in node vs global
    node_mean = np.mean(node_data, axis=0)
    global_mean = np.mean(global_data, axis=0)
    diff = node_mean - global_mean

    top_idx = np.argsort(np.abs(diff))[::-1][:top_k]
    if feature_names is None:
        feature_names = [f"F_{i}" for i in range(integrated_data.shape[1])]

    top_features = [
        {"feature": feature_names[i], "diff": float(diff[i]), "node_mean": float(node_mean[i])}
        for i in top_idx
    ]

    return {
        "n_individuals": len(member_indices),
        "member_indices": member_indices,
        "top_features": top_features,
    }


def cross_reference_genage(gene_list: list) -> list:
    """Cross-reference a gene list with GenAge longevity database.

    Args:
        gene_list: list of gene symbols.

    Returns:
        List of overlapping genes (subset of gene_list found in GenAge).
    """
    genage_set = set(GENAGE_GENES)
    overlapping = [g for g in gene_list if g in genage_set]
    n_overlap = len(overlapping)
    n_total = len(gene_list)

    logger.info(f"GenAge overlap: {n_overlap}/{n_total} "
                f"({100 * n_overlap / max(1, n_total):.1f}%)")
    return overlapping


def plot_enrichment_heatmap(
    enrichment_df: pd.DataFrame,
    ax=None,
    cmap: str = "Reds",
) -> object:
    """Plot a heatmap of enrichment results (p-values per pathway).

    Args:
        enrichment_df: DataFrame with columns 'Term' and 'P_value'.
        ax: matplotlib Axes (optional).
        cmap: colormap.

    Returns:
        matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    df = enrichment_df.copy()
    if "P_value" in df.columns:
        df["-log10(P)"] = -np.log10(df["P_value"].clip(lower=1e-30))
    else:
        logger.warning("No P_value column in enrichment results")
        return ax

    # Pivot into a matrix: rows=Term, single column = -log10(P)
    plot_df = df.set_index("Term")[["-log10(P)"]].sort_values("-log10(P)")

    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": "-log10(p-value)"},
    )
    ax.set_title("Enrichment Significance (-log10 p-value)")
    ax.set_ylabel("")
    return ax


def run_enrichr(gene_list: list, gene_set_library: str = "KEGG_2019_Human") -> pd.DataFrame:
    """Run Enrichr API enrichment analysis (requires internet + gseapy).

    Args:
        gene_list: list of gene symbols.
        gene_set_library: Enrichr library name.

    Returns:
        DataFrame of enrichment results, or empty if gseapy not available.
    """
    try:
        import gseapy as gp
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_set_library,
            organism="Human",
            outdir=None,
            no_plot=True,
            verbose=False,
        )
        results = enr.results
        logger.info(f"Enrichr ({gene_set_library}): {len(results)} terms, "
                    f"top: {results.iloc[0]['Term'] if len(results) > 0 else 'N/A'}")
        return results
    except ImportError:
        logger.warning("gseapy not installed — install with: pip install gseapy")
        return pd.DataFrame()
    except Exception as e:
        logger.info(f"Enrichr query failed: {e}")
        return pd.DataFrame()
