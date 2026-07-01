"""Tests for bio_enrichment.py — biological annotation of topological findings."""

import pytest
from src.bio_enrichment import (
    enrich_mapper_genes,
    annotate_persistent_cycle,
    compare_with_genage,
)


class TestEnrichMapperGenes:
    def test_returns_dataframe(self):
        import pandas as pd
        result = enrich_mapper_genes(["MTOR", "SIRT1", "CLOCK", "GARBAGE123"])
        assert isinstance(result, pd.DataFrame)

    def test_mtor_hits_pathway(self):
        result = enrich_mapper_genes(["MTOR", "SIRT1"])
        assert len(result) >= 1

    def test_empty_gene_list(self):
        result = enrich_mapper_genes([])
        assert len(result) == 0

    def test_multiple_databases(self):
        result = enrich_mapper_genes(
            ["MTOR", "SIRT1", "IGF1"],
            databases=["mTOR_signaling", "sirtuins", "insulin_signaling"],
        )
        assert len(result) >= 1


class TestAnnotatePersistentCycle:
    def test_returns_dict(self):
        result = annotate_persistent_cycle(["MTOR", "SIRT1", "CLOCK", "GARBAGE"])
        assert isinstance(result, dict)

    def test_empty_cycle(self):
        result = annotate_persistent_cycle([])
        assert isinstance(result, dict)


class TestCompareWithGenAge:
    def test_returns_dataframe(self):
        import pandas as pd
        result = compare_with_genage(["SIRT1", "MTOR", "FOXO3", "GARBAGE"])
        assert isinstance(result, pd.DataFrame)

    def test_has_columns(self):
        result = compare_with_genage(["GARBAGE123", "NONSENSE456"])
        assert len(result) > 0  # each gene gets one row
