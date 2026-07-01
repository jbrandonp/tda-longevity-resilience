""".topo file format — portable serialization of persistence diagrams.

Enables saving/loading persistence diagrams alongside metadata, compatible
with AnnData .uns['topology'] for integration with scanpy workflows.

Format: JSON with fields:
    - version: str
    - diagrams: list of [[birth, death], ...] per homology dimension
    - metadata: dict with source_data, parameters, timestamps
"""

import json
from pathlib import Path

import numpy as np


def _numpy_to_list(arr):
    """Convert numpy arrays to JSON-safe lists."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, list):
        return [_numpy_to_list(a) for a in arr]
    return arr


def save_topo(
    diagrams: list,
    path: str,
    metadata: dict = None,
    version: str = "1.0.0",
) -> str:
    """Save persistence diagrams to a .topo JSON file.

    Args:
        diagrams: list of (n_pairs, 2) arrays, one per homology dimension.
        path: output file path (.topo extension).
        metadata: dict with source_data, parameters, etc.
        version: format version string.

    Returns:
        Absolute path to the saved file.
    """
    data = {
        "version": version,
        "diagrams": [_numpy_to_list(d) for d in diagrams],
        "metadata": metadata or {},
    }
    out = Path(path).with_suffix(".topo")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    return str(out.resolve())


def load_topo(path: str) -> dict:
    """Load persistence diagrams from a .topo JSON file.

    Returns:
        dict with 'version', 'diagrams' (list of np.ndarray), 'metadata'.
    """
    with open(path) as f:
        data = json.load(f)

    data["diagrams"] = [
        np.array(d, dtype=np.float64) for d in data["diagrams"]
    ]
    return data


def to_anndata_uns(diagrams: list) -> dict:
    """Convert persistence diagrams to AnnData .uns['topology'] compatible dict.

    Usage:
        adata.uns['topology'] = to_anndata_uns(dgms)
    """
    return {
        "version": "1.0.0",
        "n_dimensions": len(diagrams),
        "diagrams": {
            f"H{i}": {
                "births": _numpy_to_list(dg[:, 0]),
                "deaths": _numpy_to_list(dg[:, 1]),
                "n_pairs": len(dg),
            }
            for i, dg in enumerate(diagrams)
        },
    }
