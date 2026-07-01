"""Mapper algorithm construction, analysis, and comparison utilities.

Public API:
    build_mapper_graph(data, filter_data, cover, clusterer) -> dict
    auto_mapper(data, labels) -> dict
    compare_mapper_graphs(graph_a, graph_b) -> dict
    enrich_mapper_nodes(graph, labels) -> pd.DataFrame
    export_mapper_html(graph, path)
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN

try:
    from .config import (
        MAPPER_N_CUBES,
        MAPPER_PERC_OVERLAP,
        MAPPER_CLUSTER_EPS,
        MAPPER_CLUSTER_MIN_SAMPLES,
        MAPPER_UMAP_N_COMPONENTS,
        RANDOM_SEED,
    )
except ImportError:
    from config import (
        MAPPER_N_CUBES,
        MAPPER_PERC_OVERLAP,
        MAPPER_CLUSTER_EPS,
        MAPPER_CLUSTER_MIN_SAMPLES,
        MAPPER_UMAP_N_COMPONENTS,
        RANDOM_SEED,
    )

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


def build_mapper_graph(
    data: np.ndarray,
    filter_data: np.ndarray = None,
    cover: object = None,
    clusterer: object = None,
    verbose: bool = True,
) -> dict:
    """Build a Mapper graph from data.

    Args:
        data: (n_samples, n_features) — the point cloud.
        filter_data: (n_samples, n_filter_dims) — lens/filter values. If None, use UMAP 2D.
        cover: kmapper.Cover instance. If None, use defaults.
        clusterer: sklearn clusterer. If None, use DBSCAN.

    Returns:
        Mapper graph dict with 'nodes', 'links', 'meta_data'.
    """
    try:
        import kmapper as km
    except ImportError:
        logger.warning("kmapper not installed — returning empty graph")
        return {"nodes": {}, "links": [], "meta_data": {}}

    if filter_data is None:
        try:
            import umap
            reducer = umap.UMAP(
                n_components=MAPPER_UMAP_N_COMPONENTS,
                random_state=RANDOM_SEED,
            )
            filter_data = reducer.fit_transform(data)
        except ImportError:
            filter_data = data[:, :2]

    if cover is None:
        cover = km.Cover(n_cubes=MAPPER_N_CUBES, perc_overlap=MAPPER_PERC_OVERLAP)

    if clusterer is None:
        clusterer = DBSCAN(eps=MAPPER_CLUSTER_EPS, min_samples=MAPPER_CLUSTER_MIN_SAMPLES)

    mapper = km.KeplerMapper(verbose=1 if verbose else 0)
    graph = mapper.map(filter_data, data, cover=cover, clusterer=clusterer)

    return graph


def auto_mapper(
    data: np.ndarray,
    labels: pd.Series = None,
    verbose: bool = True,
) -> dict:
    """One-shot Mapper with sensible defaults and UMAP lens."""
    try:
        import umap
        reducer = umap.UMAP(
            n_components=MAPPER_UMAP_N_COMPONENTS,
            random_state=RANDOM_SEED,
        )
        lens = reducer.fit_transform(data)
    except ImportError:
        lens = data[:, :2]

    graph = build_mapper_graph(data, filter_data=lens, verbose=verbose)

    if labels is not None:
        graph["labels"] = labels.values if hasattr(labels, "values") else labels

    return graph


def compare_mapper_graphs(graph_a: dict, graph_b: dict) -> dict:
    """Compare two Mapper graphs by structural statistics.

    Returns:
        dict with node_count, edge_count, connected_components, mean_degree.
    """
    def _stats(g):
        nodes = g.get("nodes", {})
        links = g.get("links", [])
        n_nodes = len(nodes)
        n_edges = len(links)
        if n_nodes > 0:
            G = nx.Graph()
            for nid in nodes:
                G.add_node(nid)
            for link in links:
                G.add_edge(link["source"], link["target"])
            cc = nx.number_connected_components(G)
            mean_deg = np.mean([d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0
        else:
            cc = 0
            mean_deg = 0.0
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "connected_components": cc,
            "mean_degree": round(mean_deg, 2),
        }

    return {"graph_a": _stats(graph_a), "graph_b": _stats(graph_b)}


def enrich_mapper_nodes(
    graph: dict,
    labels: pd.Series,
    group_col: str = "group",
) -> pd.DataFrame:
    """Compute enrichment of each Mapper node for a categorical label.

    Returns:
        DataFrame with node_id, size, composition, dominant_group.
    """
    nodes = graph.get("nodes", {})
    rows = []
    unique_labels = labels.unique()

    for node_id, member_indices in nodes.items():
        member_labels = labels.iloc[list(member_indices)] if hasattr(labels, "iloc") else [labels[i] for i in member_indices]
        counts = pd.Series(member_labels).value_counts().to_dict()
        dominant = max(counts, key=counts.get) if counts else "unknown"
        rows.append({
            "node_id": node_id,
            "size": len(member_indices),
            "dominant_group": dominant,
            **{f"count_{k}": v for k, v in counts.items()},
        })

    return pd.DataFrame(rows)


def export_mapper_html(graph: dict, path: str = "results/figures/mapper.html"):
    """Export Mapper graph to an interactive HTML file using KeplerMapper."""
    try:
        import kmapper as km
        mapper = km.KeplerMapper(verbose=1)
        # Reuse existing graph data
        html = km.visualize(graph, path_html=path, title="Mapper Graph — TDA Longevity")
        logger.info(f"Mapper graph exported to {path}")
        return html
    except ImportError:
        logger.warning("kmapper not available for HTML export")
        return None
