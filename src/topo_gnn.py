"""
Topological Graph Neural Network — GNN with persistence features on Mapper graphs.

Builds a graph from Mapper output, computes persistent homology features
per node (based on the point cloud of samples assigned to that node),
and uses them as input to a graph neural network for node/edge/graph classification.

Architecture:
  Mapper graph → per-node point clouds → persistence diagrams → PI vectors
  → GNN (GCN/GAT/GraphSAGE) → node classifications (resilient/accelerated)
"""
import numpy as np
import pandas as pd

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


class TopologicalGNN:
    """
    Graph Neural Network with topological node features.

    Uses a Mapper graph as input structure, then:
    1. Extracts persistence diagrams from each node's point cloud
    2. Vectorizes them (persistence images)
    3. Feeds them through a GNN (basic GCN) for node classification

    Parameters:
      hidden_dim: GNN hidden layer size
      n_layers: number of GNN layers
      dropout: dropout probability
      max_dim: max persistence homology dimension
    """

    def __init__(self, hidden_dim=64, n_layers=2, dropout=0.3, max_dim=1):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_dim = max_dim
        self._fitted = False

    def _extract_node_features(self, graph, data):
        """
        Extract topological features for each Mapper node.

        For each node, get its member sample indices, extract the point cloud,
        compute persistence, and vectorize.
        """
        try:
            from .features import PersistenceImageTransformer
        except ImportError:
            from features import PersistenceImageTransformer

        pi = PersistenceImageTransformer(spread=0.1, pixels=(20, 20))

        node_features = {}
        for node_id, members in graph.get("nodes", {}).items():
            if isinstance(members, dict):
                members = members.get("members", members.get("indices", []))
            if len(members) < 3:
                node_features[node_id] = np.zeros(400)  # 20×20
                continue

            # Extract point cloud for this node
            point_cloud = data[np.array(members, dtype=int)]
            from data_utils import _finite_dgm

            # Compute persistence
            from ripser import ripser
            dgm = ripser(point_cloud, maxdim=self.max_dim)["dgms"]
            dgm_finite = [_finite_dgm(d) for d in dgm]

            # Vectorize
            vec = pi.fit_transform([dgm_finite[1] if len(dgm_finite) > 1 else dgm_finite[0]])
            node_features[node_id] = vec.flatten()

        return node_features

    def _build_adjacency(self, graph, node_ids):
        """Build adjacency matrix from Mapper links."""
        n = len(node_ids)
        adj = np.zeros((n, n))
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        for link in graph.get("links", []):
            src = link.get("source")
            tgt = link.get("target")
            if src in id_to_idx and tgt in id_to_idx:
                i, j = id_to_idx[src], id_to_idx[tgt]
                adj[i, j] = adj[j, i] = 1.0
        return adj

    def fit(self, graph, data, labels, epochs=100, lr=0.01):
        """
        Fit a simple GCN on the Mapper graph.

        Uses manual feed-forward via numpy (no PyTorch dependency)
        for the basic GCN forward pass.

        Parameters:
          graph: Mapper graph dict (from mapper_utils)
          data: full data matrix (n_samples × n_features)
          labels: node classifications (dict node_id → class)
        """
        try:
            from .mapper_utils import enrich_mapper_nodes
        except ImportError:
            from mapper_utils import enrich_mapper_nodes

        # Extract features
        logger.info("Extracting topological node features...")
        node_features = self._extract_node_features(graph, data)

        # Build feature matrix + adjacency
        node_ids = sorted(node_features.keys())
        n = len(node_ids)
        d = node_features[node_ids[0]].shape[0]

        X = np.zeros((n, d))
        for i, nid in enumerate(node_ids):
            X[i] = node_features[nid]

        A = self._build_adjacency(graph, node_ids)
        D_inv = np.diag(1.0 / np.maximum(A.sum(axis=1), 1))

        # Target labels
        y = np.zeros(n)
        for i, nid in enumerate(node_ids):
            y[i] = labels.get(nid, 0)

        # Simple GCN forward (2-layer)
        rng = np.random.default_rng(42)
        self.W1_ = rng.standard_normal((d, self.hidden_dim)) * 0.01
        self.W2_ = rng.standard_normal((self.hidden_dim, 2)) * 0.01

        # Training loop
        for epoch in range(epochs):
            # Layer 1: (Â·X·W1) → ReLU
            H1 = (D_inv @ A + np.eye(n)) @ X @ self.W1_
            H1 = np.maximum(H1, 0)  # ReLU

            # Layer 2: (Â·H1·W2) → softmax
            logits = (D_inv @ A + np.eye(n)) @ H1 @ self.W2_
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

            # Cross-entropy loss + SGD
            loss = -np.mean(np.log(probs[np.arange(n), y.astype(int)] + 1e-8))
            grad = probs.copy()
            grad[np.arange(n), y.astype(int)] -= 1
            grad /= n

            # Backprop (simplified)
            dW2 = H1.T @ ((D_inv @ A + np.eye(n)).T @ grad)
            dW1 = X.T @ ((D_inv @ A + np.eye(n)).T @ (grad @ self.W2_.T * (H1 > 0)))

            self.W2_ -= lr * dW2
            self.W1_ -= lr * dW1

        self._fitted = True
        self.train_loss_ = loss
        self.node_ids_ = node_ids
        logger.info(f"TopologicalGNN trained — final loss: {loss:.4f}")

        return self

    def predict(self, graph, data):
        """Predict node classes for a Mapper graph."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        features = self._extract_node_features(graph, data)
        nids = sorted(features.keys())
        n = len(nids)
        d = features[nids[0]].shape[0]

        X = np.zeros((n, d))
        for i, nid in enumerate(nids):
            X[i] = features[nid]

        A = self._build_adjacency(graph, nids)
        D_inv = np.diag(1.0 / np.maximum(A.sum(axis=1), 1))

        H1 = np.maximum((D_inv @ A + np.eye(n)) @ X @ self.W1_, 0)
        logits = (D_inv @ A + np.eye(n)) @ H1 @ self.W2_
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        return {nid: int(p) for nid, p in zip(nids, probs.argmax(axis=1))}
