"""Tests for visualization.py — plotting functions."""
import numpy as np, pytest

# Lazy import — matplotlib may not be installed
@pytest.fixture
def dgm():
    return [np.array([[0.0, 0.5], [0.0, 0.8], [0.0, np.inf]]),
            np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.4]])]


class TestPlotBarcode:
    def test_returns_axes(self, dgm):
        try:
            from src.visualization import plot_barcode
            ax = plot_barcode(dgm, dim=1, title="test")
            assert ax is not None
        except ImportError:
            pytest.skip("matplotlib not installed")
    def test_empty_diagram(self):
        try:
            from src.visualization import plot_barcode
            ax = plot_barcode([np.empty((0, 2))], dim=0, title="empty")
            assert ax is not None
        except ImportError:
            pytest.skip("matplotlib not installed")


class TestPlotPersistenceDiagram:
    def test_returns_axes(self, dgm):
        try:
            from src.visualization import plot_persistence_diagram
            ax = plot_persistence_diagram(dgm, dim=1, title="pd")
            assert ax is not None
        except ImportError:
            pytest.skip("matplotlib not installed")


class TestPlotBarcodeComparison:
    def test_returns_figure(self, dgm):
        try:
            from src.visualization import plot_barcode_comparison
            fig = plot_barcode_comparison(dgm, dgm, dim=1, figsize=(8, 4))
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not installed")


class TestPlotROCCurves:
    def test_returns_figure(self):
        try:
            from src.visualization import plot_roc_curves
            data = {"Model": {"fpr": [0, 0.2, 1], "tpr": [0, 0.8, 1], "auc": 0.9}}
            fig = plot_roc_curves(data)
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not installed")


class TestPlotConfusionMatrix:
    def test_returns_figure(self):
        try:
            from src.visualization import plot_confusion_matrix
            fig = plot_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not installed")


class TestPlotBarcodeInteractive:
    def test_returns_figure_or_none(self, dgm):
        from src.visualization import plot_barcode_interactive
        fig = plot_barcode_interactive(dgm, dim=1, title="test")
        # Returns None if plotly missing, Figure if available
        assert fig is None or hasattr(fig, "show")


class TestPlotMapperGraph:
    def test_returns_or_none(self):
        from src.visualization import plot_mapper_graph
        graph = {"nodes": {"n0": [0, 1]}, "links": []}
        result = plot_mapper_graph(graph, title="test")
        assert result is None or isinstance(result, str)
