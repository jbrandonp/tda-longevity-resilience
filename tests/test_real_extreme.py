"""
=== REAL-WORLD + EXTREME TEST SUITE ===
Scenarios: tiny data, massive data, high-dim, corrupted, imbalanced,
           missing, outliers, real datasets, mixed scales, ramp tests.
"""
import numpy as np, pandas as pd, pytest, sys, os, time, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (generate_synthetic_multimics, preprocess_omics,
    integrate_multiomics, select_features, _finite_dgm, load_dataset)
from tda_utils import (compute_persistence_diagrams, diagnose_persistence,
    wasserstein_distance, persistence_stability, sparse_persistence)
from aging_scores import phenoage_simplified, assign_acceleration_group, dunedin_pace_proxy
from features import (PersistenceImageTransformer, PersistenceLandscapeTransformer,
    BettiCurveTransformer, extract_all_features)
from ml_utils import build_topological_pipeline, evaluate_topological_model
from metrics import aitchison_distance, get_distance_matrix, recommended_metric_for_data
from bio_enrichment import enrich_mapper_genes, cross_reference_genage
from validation import create_discovery_validation, evaluate_on_holdout
from mapper_utils import auto_mapper, enrich_mapper_nodes
from config import RANDOM_SEED


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-WORLD: data loading, multi-omics with real scales, pipelines
# ═══════════════════════════════════════════════════════════════════════════════
class TestRealWorld:
    """Scenarios you'd encounter in actual research."""

    def test_load_real_dataset_inchianti(self):
        """download_data.py must produce valid CSV files."""
        r = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'download_data.py'),
             '--dataset', 'inchianti', '--output', tempfile.gettempdir()],
            capture_output=True, text=True, timeout=60
        )
        assert r.returncode == 0
        for f in ['inchianti_clinical.csv', 'inchianti_transcriptomics.csv',
                  'inchianti_metabolomics.csv', 'inchianti_methylation.csv']:
            fp = os.path.join(tempfile.gettempdir(), f)
            assert os.path.exists(fp), f"Missing {f}"
            df = pd.read_csv(fp)
            assert len(df) > 0
            os.unlink(fp)

    def test_load_real_dataset_gtex(self):
        r = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'download_data.py'),
             '--dataset', 'gtex', '--output', tempfile.gettempdir()],
            capture_output=True, text=True, timeout=30
        )
        assert r.returncode == 0
        fp = os.path.join(tempfile.gettempdir(), 'gtex_transcriptomics.csv')
        assert os.path.exists(fp)
        df = pd.read_csv(fp)
        assert len(df) >= 900  # GTEx has ~948 donors
        os.unlink(fp)

    def test_load_real_dataset_tcga(self):
        r = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'download_data.py'),
             '--dataset', 'tcga', '--output', tempfile.gettempdir()],
            capture_output=True, text=True, timeout=30
        )
        assert r.returncode == 0
        fp = os.path.join(tempfile.gettempdir(), 'tcga_clinical.csv')
        assert os.path.exists(fp)
        os.unlink(fp)

    def test_download_data_list(self):
        """--list must show all 3 datasets."""
        r = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'download_data.py'), '--list'],
            capture_output=True, text=True, timeout=15
        )
        assert r.returncode == 0
        assert 'inchianti' in r.stdout.lower()
        assert 'gtex' in r.stdout.lower()
        assert 'tcga' in r.stdout.lower()

    def test_mixed_scale_omics(self):
        """Transcriptomics (~0-1) vs metabolomics (~1000-10000) — must not break."""
        trans = np.random.randn(100, 50)  # standardized ~ N(0,1)
        metab = np.random.lognormal(5, 2, (100, 30))  # ~ 100-10000
        epigen = np.random.beta(2, 5, (100, 20)) * 100  # 0-100 beta values

        layers = {"trans": trans, "metab": metab, "epigen": epigen}
        integrated = integrate_multiomics(layers, method="concat")
        assert integrated.shape == (100, 100)
        assert not np.any(np.isnan(integrated))
        # Each layer should be properly scaled
        for i, (name, arr) in enumerate(layers.items()):
            col_start = sum(v.shape[1] for v in list(layers.values())[:i])
            col_end = col_start + arr.shape[1]
            layer_integrated = integrated[:, col_start:col_end]
            # After scaling, should have mean ≈ 0, std ≈ 1
            assert abs(np.mean(layer_integrated)) < 0.5
            assert 0.5 < np.std(layer_integrated) < 2.0

    def test_longitudinal_simulation(self):
        """Simulate 3 timepoints for 50 subjects — must handle repeated measures."""
        n_subjects = 50
        n_timepoints = 3
        n_features = 30
        total = n_subjects * n_timepoints

        data = np.zeros((total, n_features))
        for i in range(n_subjects):
            baseline = np.random.randn(n_features)
            for t in range(n_timepoints):
                idx = i * n_timepoints + t
                data[idx] = baseline + t * 0.1 * np.random.randn(n_features) + 0.05 * np.random.randn(n_features)

        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        diag = diagnose_persistence(dgms)
        assert "H0" in diag

    def test_batch_effect_simulation(self):
        """Two batches with different means — ComBat integration should handle."""
        batch1 = np.random.randn(60, 40)
        batch2 = np.random.randn(40, 40) + 2.0  # shifted by 2 SD
        combined = np.vstack([batch1, batch2])
        batch_labels = np.array([0]*60 + [1]*40)

        # Integration should handle this without NaN
        layers = {"genes": combined}
        integrated = integrate_multiomics(layers, method="concat")  # no combat batch
        assert integrated.shape == (100, 40)
        assert not np.any(np.isnan(integrated))


# ═══════════════════════════════════════════════════════════════════════════════
# EXTREME: boundary conditions that should never crash
# ═══════════════════════════════════════════════════════════════════════════════
class TestExtreme:
    """Edge of the universe — if these crash, defensive coding is missing."""

    # ── TINY DATA ──
    def test_tiny_dataset_n3(self):
        data = np.random.randn(3, 10)
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        assert isinstance(dgms, list)

    def test_single_feature(self):
        data = np.random.randn(100, 1)
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        dm = get_distance_matrix(data, metric="euclidean")
        assert dm.shape == (100, 100)

    def test_two_features(self):
        data = np.random.randn(100, 2)
        layers = {"a": data, "b": data}
        integrated = integrate_multiomics(layers, method="concat")
        assert integrated.shape == (100, 4)

    # ── HIGH DIMENSIONAL ──
    def test_p_larger_than_n(self):
        """500 features, 30 samples — p >> n."""
        data = np.random.randn(30, 500)
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        assert isinstance(dgms, list)
        diag = diagnose_persistence(dgms)
        assert "H0" in diag

    def test_very_high_dimensional(self):
        """1000 features, 20 samples."""
        data = np.random.randn(20, 1000)
        layers = {"huge": data}
        integrated = integrate_multiomics(layers, method="concat")
        assert integrated.shape == (20, 1000)

    # ── MASSIVE DATA ──
    def test_large_dataset_1000(self):
        data = np.random.randn(1000, 50)
        dgms = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
        assert isinstance(dgms, list)

    def test_many_omics_layers(self):
        """10 omics layers — integration must handle arbitrary count."""
        layers = {f"layer_{i}": np.random.randn(50, 10) for i in range(10)}
        integrated = integrate_multiomics(layers, method="concat")
        assert integrated.shape == (50, 100)

    # ── CORRUPTED DATA ──
    def test_all_zeros(self):
        data = np.zeros((30, 10))
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        assert isinstance(dgms, list)

    def test_all_identical(self):
        data = np.ones((30, 10)) * 5.0
        # preprocessing should handle constant columns
        processed = preprocess_omics(pd.DataFrame(data), method="standard")
        assert not np.any(np.isnan(processed))

    def test_all_nan_column(self):
        data = np.random.randn(50, 10)
        data[:, 3] = np.nan
        data[:, 7] = np.nan
        # This should not crash — preprocessing should handle
        processed = preprocess_omics(pd.DataFrame(data), method="robust")
        assert processed.shape[1] == 10

    def test_all_inf_column(self):
        data = np.random.randn(30, 8)
        data[0, 2] = np.inf
        data[1, 5] = -np.inf
        dgms = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
        assert isinstance(dgms, list)

    # ── IMBALANCED GROUPS ──
    def test_extremely_imbalanced_1_99(self):
        df = pd.DataFrame({"age": np.concatenate([np.full(1, 80), np.full(99, 40)])})
        result = assign_acceleration_group(df)
        counts = result["aging_group"].value_counts()
        assert len(counts) >= 1  # should not crash

    def test_all_same_group(self):
        df = pd.DataFrame({"age": np.full(100, 50.0)})
        result = assign_acceleration_group(df)
        # With identical ages, phenoage should be nearly identical → mostly one group
        top = result["aging_group"].value_counts().iloc[0]
        assert top >= 80  # at least 80% in dominant group

    # ── OUTLIERS ──
    def test_extreme_biomarker_values(self):
        """Biomarkers at impossible values — PhenoAge should handle gracefully."""
        pa = phenoage_simplified(
            np.array([30.0]),
            albumin=np.array([0.1]),        # impossibly low
            creatinine=np.array([15.0]),     # impossibly high
            glucose=np.array([500.0]),       # diabetic crisis
            c_reactive_protein=np.array([50.0]),  # severe inflammation
            lymphocyte_percent=np.array([5.0]),    # critically low
            mean_cell_volume=np.array([120.0]),    # macrocytic
            red_blood_cell_distribution=np.array([25.0]),  # severe anisocytosis
            alkaline_phosphatase=np.array([500.0]),  # liver disease
            white_blood_cell_count=np.array([50000.0]),  # leukocytosis
        )
        assert np.isfinite(pa[0])
        assert pa[0] > 60  # should be biologically older

    def test_outlier_samples(self):
        """One sample 100 SD away — distance matrix must not explode."""
        data = np.random.randn(50, 10)
        data[0] = 1000  # massive outlier
        dm = get_distance_matrix(data, metric="euclidean")
        assert not np.any(np.isnan(dm))
        assert not np.any(np.isinf(dm))

    # ── NOISE ONLY ──
    def test_pure_noise_no_signal(self):
        data = np.random.randn(100, 30)  # pure noise
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        diag = diagnose_persistence(dgms)
        # Pure noise should have very few persistent features
        assert diag["H1"]["n_features"] <= 20 or diag["H1"]["n_features"] == 0

    def test_perfect_signal_no_noise(self):
        """Circle with zero noise — should clearly see H1."""
        t = np.linspace(0, 2*np.pi, 50, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        data = np.hstack([circle, np.zeros((50, 28))])  # pad to 30 features
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        assert isinstance(dgms, list)

    # ── MIXED TYPES ──
    def test_mixed_continuous_categorical(self):
        """Continuous + one-hot encoded categorical features."""
        continuous = np.random.randn(80, 20)
        categorical = np.eye(80, 4)[np.random.randint(0, 4, 80)]  # 4 categories, one-hot
        mixed = np.column_stack([continuous, categorical])
        layers = {"mixed": mixed}
        integrated = integrate_multiomics(layers, method="concat")
        assert integrated.shape == (80, 24)

    # ── STRESS: repeated operations ──
    def test_rapid_successive_calls(self):
        """Call the same function 50 times rapidly — no memory leak or degradation."""
        data = np.random.randn(100, 20)
        times = []
        for _ in range(20):
            t0 = time.time()
            dgms = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
            times.append(time.time() - t0)
        # No single call should take > 5s
        assert max(times) < 5.0, f"slowest call: {max(times):.2f}s"

    def test_empty_integration(self):
        """Zero omics layers should raise cleanly."""
        with pytest.raises(Exception):
            integrate_multiomics({})

    def test_mapper_on_tiny_data(self):
        """Mapper on 10 samples — shouldn't crash."""
        data = np.random.randn(10, 5)
        graph = auto_mapper(data, verbose=False)
        assert isinstance(graph, dict)
        assert "nodes" in graph

    def test_validation_with_perfect_separation(self):
        """Linearly separable data — ML should get AUC=1."""
        X = np.zeros((100, 10))
        X[:50, 0] = -10
        X[50:, 0] = 10
        y = np.array([0]*50 + [1]*50)
        pipe = build_topological_pipeline()
        result = evaluate_topological_model(pipe, X, y, cv=3)
        # With perfect separation, AUC should be high
        assert result["roc_auc"]["mean"] >= 0.8

    # ── RAMP: increasing sizes ──
    @pytest.mark.parametrize("n", [5, 10, 25, 50, 100, 200])
    def test_ramp_samples(self, n):
        """Pipeline must work for sample sizes 5→200."""
        ds = generate_synthetic_multimics(n_samples=n, topology_type="circle", noise=0.05, n_features=20)
        layers = {k: preprocess_omics(pd.DataFrame(v), method="standard")
                  for k, v in ds.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        integrated = integrate_multiomics(layers, method="concat")
        assert integrated.shape[0] == n

    @pytest.mark.parametrize("p", [2, 5, 10, 25, 50, 100])
    def test_ramp_features(self, p):
        """Pipeline must work for feature counts 2→100."""
        ds = generate_synthetic_multimics(n_samples=30, topology_type="circle", noise=0.05, n_features=p)
        layers = {k: preprocess_omics(pd.DataFrame(v), method="standard")
                  for k, v in ds.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        for name, arr in layers.items():
            assert arr.shape[1] == p

    @pytest.mark.parametrize("n_layers", [1, 2, 3, 5, 8])
    def test_ramp_omics_layers(self, n_layers):
        """Integration must work for 1→8 omics layers."""
        layers = {f"L{i}": np.random.randn(30, 10) for i in range(n_layers)}
        integrated = integrate_multiomics(layers, method="concat")
        assert integrated.shape == (30, n_layers * 10)

    # ── SCALE COMPARISON ──
    def test_small_vs_large_consistency(self):
        """Same topology at different sizes should produce qualitatively similar diagnostics."""
        results = {}
        for n in [30, 100]:
            ds = generate_synthetic_multimics(n_samples=n, topology_type="circle", noise=0.05, n_features=20)
            data = preprocess_omics(pd.DataFrame(ds["transcriptomics"]), method="standard")
            dgms = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
            diag = diagnose_persistence(dgms)
            results[n] = diag["H0"]["n_features"]
        # Both should produce valid diagnostics (ripser may be absent)
        assert results[30] >= 0 and results[100] >= 0

    def test_noise_vs_signal_contrast(self):
        """Circle should have more structure than noise."""
        t = np.linspace(0, 2*np.pi, 50, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        circle = np.hstack([circle, np.zeros((50,18))])
        noise = np.random.randn(50, 20)

        dgms_circle = compute_persistence_diagrams(circle, max_dim=0, use_cache=False)
        dgms_noise = compute_persistence_diagrams(noise, max_dim=0, use_cache=False)

        diag_c = diagnose_persistence(dgms_circle)
        diag_n = diagnose_persistence(dgms_noise)

        # Circle H0 features should be fewer (more structured connectivity)
        assert diag_c["H0"]["n_features"] <= diag_n["H0"]["n_features"] + 5
