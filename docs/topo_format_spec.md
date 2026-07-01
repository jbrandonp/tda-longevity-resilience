# .topo Format Specification v1.0

A standardized, open file format for storing topological data analysis results
— persistence diagrams, Mapper graphs, and associated metadata — in a single,
interoperable container.

**Extension:** `.topo`
**MIME type:** `application/x-topo+json` (proposed)
**Encoding:** UTF-8, JSON (optionally gzipped: `.topo.gz`)

## Motivation

Biomedical TDA produces heterogeneous artifacts: persistence diagrams (NumPy
arrays), Mapper graphs (networkx/igraph), and metadata (JSON). There is no
standard way to exchange these between tools, share with collaborators, or
archive for reproducibility. `.topo` solves this.

## Schema

```json
{
  "topo_version": "1.0",
  "created": "2026-07-01T17:00:00Z",
  "project": "tda-longevity-resilience",
  "data": {
    "n_samples": 200,
    "n_features": 50,
    "omics_layers": ["transcriptomics", "metabolomics"],
    "metric": "euclidean",
    "description": "Synthetic circle-topology multi-omics data"
  },
  "persistence": {
    "method": "vietoris-rips",
    "max_dim": 2,
    "threshold": 0.95,
    "diagrams": {
      "H0": [[0.0, 0.5], [0.0, 0.8]],
      "H1": [[0.1, 0.5], [0.2, 0.6]],
      "H2": []
    },
    "diagnostics": {
      "H0": {"n_features": 2, "mean_lifetime": 0.65},
      "H1": {"n_features": 2, "mean_lifetime": 0.40}
    }
  },
  "mapper": {
    "filter": "umap",
    "filter_params": {"n_components": 2, "n_neighbors": 15},
    "cover": {"n_cubes": 15, "perc_overlap": 0.5},
    "clusterer": "dbscan",
    "clusterer_params": {"eps": 0.5, "min_samples": 5},
    "nodes": {
      "node_0": {"size": 12, "members": [0, 1, 2]},
      "node_1": {"size": 8, "members": [3, 4]}
    },
    "edges": [
      {"source": "node_0", "target": "node_1", "weight": 1}
    ]
  },
  "features": {
    "persistence_image": {"spread": 0.1, "pixels": [50, 50], "dim": 1},
    "landscape": {"n_layers": 5, "n_bins": 100, "dim": 1}
  },
  "groups": {
    "accelerated": {"n": 60, "mean_tian_score": 1.5},
    "neutral": {"n": 80, "mean_tian_score": 0.0},
    "resilient": {"n": 60, "mean_tian_score": -1.2}
  },
  "provenance": {
    "code_version": "1.0.0",
    "git_commit": "abc123def",
    "environment": "tda-longevity (conda)",
    "random_seed": 42
  }
}
```

## Requirements

| Field | Required | Type |
|-------|----------|------|
| `topo_version` | Yes | string |
| `created` | Yes | ISO 8601 datetime |
| `persistence.diagrams` | Yes | {H_dim: [[birth, death], ...]} |
| `provenance` | Yes | object |
| `mapper` | No | object (null if no Mapper analysis) |
| `features` | No | object |
| `groups` | No | object |

## Reference Implementation

See `src/topo_format.py` for serialization/deserialization utilities.

## Interoperability

- **→ AnnData (`.h5ad`):** persistence diagrams stored in `.uns['topo']`
- **→ JSON-LD:** add `@context` for linked data
- **→ NetworkX:** `nodes` + `edges` → `nx.Graph()`
- **→ pkl/npz:** legacy format, `.topo` is the standard migration target
