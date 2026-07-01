"""Visualization utilities: barcodes, persistence diagrams, Mapper graphs, ROC curves.

Public API:
    plot_barcode(dgm, dim, title, ax) -> matplotlib.axes.Axes
    plot_persistence_diagram(dgm, title, ax)
    plot_barcode_comparison(dgms_a, dgms_b, labels, dim)
    plot_mapper_graph(graph, labels, title)
    plot_roc_curves(results_dict, title)
    plot_confusion_matrix(y_true, y_pred, labels, title)
"""

import numpy as np


# ── Lazy imports ──────────────────────────────────────────────────────────────
_plt = None
_sns = None
_mpatches = None


def _ensure_mpl():
    """Lazy-load matplotlib (and optionally seaborn). Raises ImportError if missing."""
    global _plt, _sns, _mpatches
    if _plt is None:
        import matplotlib.pyplot as _p
        import matplotlib.patches as _m
        _plt = _p
        _mpatches = _m
    if _sns is None:
        try:
            import seaborn as _s
            _sns = _s
        except ImportError:
            pass


try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)

try:
    from .data_utils import _finite_dgm
except ImportError:
    from data_utils import _finite_dgm


def _mpl():
    _ensure_mpl()
    return _plt, _mpatches


# ═══════════════════════════════════════════════════════════════════════════════
# Barcodes & Persistence Diagrams
# ═══════════════════════════════════════════════════════════════════════════════

def plot_barcode(dgm, dim=1, title="Persistence Barcode", ax=None, max_lifetime=None):
    """Plot a persistence barcode for a given homology dimension."""
    plt, mpatches = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    diagram = dgm[dim] if isinstance(dgm, list) else dgm
    finite = _finite_dgm(diagram)

    if len(finite) == 0:
        ax.text(0.5, 0.5, f"No finite H{dim} features", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return ax

    births = finite[:, 0]
    lifetimes = finite[:, 1] - finite[:, 0]
    deaths = finite[:, 1]
    order = np.argsort(lifetimes)[::-1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(order)))

    for i, idx in enumerate(order):
        ax.plot([births[idx], deaths[idx]], [i, i], color=colors[i], linewidth=2, alpha=0.8)
        ax.scatter(births[idx], i, color=colors[i], s=30, zorder=5)
        ax.scatter(deaths[idx], i, color=colors[i], s=30, marker="D", zorder=5)

    ax.set_xlabel("Filtration parameter")
    ax.set_ylabel("Feature index (sorted by lifetime)")
    ax.set_title(title)
    if max_lifetime:
        ax.set_ylim(-1, max_lifetime)

    birth_patch = mpatches.Patch(color="gray", label="Birth (○)")
    death_patch = mpatches.Patch(color="gray", label="Death (◆)")
    ax.legend(handles=[birth_patch, death_patch], loc="upper right")
    return ax


def plot_persistence_diagram(dgm, title="Persistence Diagram", ax=None, dim=1):
    """Plot birth-death pairs on a persistence diagram."""
    plt, _ = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    diagram = dgm[dim] if isinstance(dgm, list) else dgm
    finite = _finite_dgm(diagram)

    if len(finite) == 0:
        ax.text(0.5, 0.5, f"No finite H{dim} features", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return ax

    births = finite[:, 0]
    deaths = finite[:, 1]
    lifetimes = deaths - births
    max_val = max(np.max(deaths), np.max(births)) * 1.05

    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="y = x (diagonal)")
    sc = ax.scatter(births, deaths, c=lifetimes, cmap="plasma", s=40,
                    edgecolors="black", linewidth=0.5, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Lifetime")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    return ax


def plot_barcode_comparison(dgms_a, dgms_b, labels=("Accelerated", "Resilient"), dim=1, figsize=(12, 5)):
    """Side-by-side barcode comparison for two groups."""
    plt, _ = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plot_barcode(dgms_a, dim=dim, title=f"H{dim} Barcode — {labels[0]}", ax=ax1)
    plot_barcode(dgms_b, dim=dim, title=f"H{dim} Barcode — {labels[1]}", ax=ax2)
    fig.suptitle(f"Persistent Homology Comparison (H{dim})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive Plotly Barcodes
# ═══════════════════════════════════════════════════════════════════════════════

def plot_barcode_interactive(dgm, dim=1, title="Interactive Persistence Barcode"):
    """Plotly-based interactive persistence barcode (zoomable, hover details).

    Requires: pip install plotly
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed — use plot_barcode() instead")
        return None

    diagram = dgm[dim] if isinstance(dgm, list) else dgm
    finite = _finite_dgm(diagram)

    if len(finite) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No finite H{dim} features", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title)
        return fig

    births = finite[:, 0]
    lifetimes = finite[:, 1] - finite[:, 0]
    order = np.argsort(lifetimes)[::-1]

    fig = go.Figure()
    for i, idx in enumerate(order):
        fig.add_trace(go.Scatter(
            x=[births[idx], finite[idx, 1]],
            y=[i, i],
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=6, symbol=["circle", "diamond"]),
            name=f"Feature {i}",
            hovertemplate=f"Birth: {births[idx]:.3f}<br>Death: {finite[idx,1]:.3f}<br>Lifetime: {lifetimes[idx]:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Filtration parameter",
        yaxis_title="Feature index (sorted by lifetime)",
        showlegend=False,
        height=400,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Mapper
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mapper_graph(graph, labels=None, title="Mapper Graph", save_path=None):
    """Visualise a Mapper graph coloured by group labels."""
    try:
        import kmapper as km
        mapper = km.KeplerMapper(verbose=0)
        html_path = save_path or "results/figures/mapper_graph.html"
        mapper.visualize(graph, path_html=html_path, title=title)
        logger.info(f"Mapper graph saved to {html_path}")
        return html_path
    except ImportError:
        logger.warning("kmapper not available")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ML Evaluation Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(results_dict, title="ROC Curves — Topological vs Classic", figsize=(8, 6)):
    """Plot comparative ROC curves.

    Args:
        results_dict: {model_name: {"fpr": [...], "tpr": [...], "auc": float}}.
    """
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

    for (name, data), color in zip(results_dict.items(), colors):
        ax.plot(data.get("fpr", [0, 1]), data.get("tpr", [0, 1]),
                label=f"{name} (AUC={data.get('auc', 0):.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", figsize=(6, 5)):
    """Plot a confusion matrix heatmap."""
    plt, _ = _mpl()
    _ensure_mpl()
    from sklearn.metrics import confusion_matrix as cm_fn

    cm = cm_fn(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)

    if _sns is not None:
        _sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                     xticklabels=labels or ["Accelerated", "Resilient"],
                     yticklabels=labels or ["Accelerated", "Resilient"])
    else:
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Styling (applied only if matplotlib is available)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    _ensure_mpl()
    if _sns is not None:
        _sns.set_style("whitegrid")
        _sns.set_context("notebook", font_scale=1.2)
    _plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })
except ImportError:
    pass
