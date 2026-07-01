"""Tests for topo_format.py — .topo file serialization."""
import numpy as np, pytest, tempfile, os
from src.topo_format import save_topo, load_topo, to_anndata_uns


class TestSaveLoadTopo:
    def test_roundtrip(self):
        dgm0 = np.array([[0.0, 0.5], [0.0, 0.8]]); dgm1 = np.array([[0.1, 0.5], [0.2, 0.6]])
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "test.topo")
            save_topo([dgm0, dgm1], p, metadata={"n": 30})
            data = load_topo(p)
            assert len(data["diagrams"]) == 2
            assert data["diagrams"][0].shape == (2, 2)
            assert data["metadata"]["n"] == 30

    def test_version_field(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "v.topo")
            save_topo([np.array([[0.0, 0.5]])], p, version="2.0.0")
            assert load_topo(p)["version"] == "2.0.0"


class TestToAnnData:
    def test_returns_nested_dict(self):
        dgm = np.array([[0.0, 0.5], [0.1, 0.6]])
        result = to_anndata_uns([dgm])
        assert "diagrams" in result
        assert "H0" in result["diagrams"]
        assert "births" in result["diagrams"]["H0"]
