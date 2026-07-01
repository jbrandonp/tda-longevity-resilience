"""Tests for mapper_utils.py — Mapper graph construction and analysis."""

import numpy as np
import pandas as pd
import pytest
from src.mapper_utils import (
    build_mapper_graph,
    auto_mapper,
    compare_mapper_graphs,
    enrich_mapper_nodes,
)
from src.config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


class TestBuildMapperGraph:
    def test_returns_dict_structure(self):
        data = rng.standard_normal((50, 10))
        graph = build_mapper_graph(data, verbose=False)
        assert isinstance(graph, dict)
        assert "nodes" in graph
        assert "links" in graph

    def test_small_dataset(self):
        data = rng.standard_normal((20, 5))
        graph = build_mapper_graph(data, verbose=False)
        assert isinstance(graph["nodes"], dict)


class TestAutoMapper:
    def test_returns_graph_with_labels(self):
        data = rng.standard_normal((50, 10))
        labels = pd.Series(["accel"] * 25 + ["resil"] * 25)
        graph = auto_mapper(data, labels=labels, verbose=False)
        assert "nodes" in graph

    def test_no_labels(self):
        data = rng.standard_normal((30, 8))
        graph = auto_mapper(data, verbose=False)
        assert isinstance(graph, dict)


class TestCompareMapperGraphs:
    def test_returns_comparison_dict(self):
        data_a = rng.standard_normal((40, 8))
        data_b = rng.standard_normal((40, 8))
        graph_a = auto_mapper(data_a, verbose=False)
        graph_b = auto_mapper(data_b, verbose=False)
        comp = compare_mapper_graphs(graph_a, graph_b)
        assert "graph_a" in comp
        assert "graph_b" in comp
        assert "n_nodes" in comp["graph_a"]
        assert "mean_degree" in comp["graph_a"]


class TestEnrichMapperNodes:
    def test_returns_dataframe(self):
        data = rng.standard_normal((50, 10))
        labels = pd.Series(np.where(rng.random(50) > 0.5, "resilient", "accelerated"))
        graph = auto_mapper(data, labels=labels, verbose=False)
        df = enrich_mapper_nodes(graph, labels, group_col="group")
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "node_id" in df.columns
            assert "size" in df.columns
            assert "dominant_group" in df.columns
