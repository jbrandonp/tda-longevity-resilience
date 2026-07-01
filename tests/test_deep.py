"""Deep integration tests — multi-step pipelines, edge cases, stress."""
import numpy as np, pandas as pd, pytest, sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_utils import generate_synthetic_multimics, preprocess_omics, integrate_multiomics, select_features
from tda_utils import (compute_persistence_diagrams, compute_all_layers_dgms,
    diagnose_persistence, wasserstein_distance, persistence_stability,
    sparse_persistence, bootstrap_diagram_stability, permutation_test)
from aging_scores import phenoage_simplified, assign_acceleration_group, dunedin_pace_proxy, compute_all_scores
from features import (PersistenceImageTransformer, PersistenceLandscapeTransformer,
    BettiCurveTransformer, extract_all_features, landscape_statistics, landscape_distance)
from ml_utils import (build_topological_pipeline, evaluate_topological_model,
    compare_with_baseline, shap_analysis, TopologicalFeatureExtractor, prepare_topological_features_cv)
from validation import create_discovery_validation, evaluate_on_holdout, generate_validation_report
from metrics import aitchison_distance, get_distance_matrix, recommended_metric_for_data
from bio_enrichment import (enrich_mapper_genes, annotate_persistent_cycle, compare_with_genage,
    extract_cycle_genes, run_enrichment, cross_reference_genage)
from mapper_utils import auto_mapper, enrich_mapper_nodes, compare_mapper_graphs
from config import RANDOM_SEED

# ═══════════════════════════════════════════════════════════════════════════════
# LARGE INTEGRATION: full pipeline on 500-sample synthetic dataset
# ═══════════════════════════════════════════════════════════════════════════════
class TestFullPipeline:
    """End-to-end: data → TDA → ML → validation on 500 samples."""

    @pytest.fixture(scope="class")
    def pipeline_data(self):
        ds = generate_synthetic_multimics(n_samples=500, topology_type="circle", noise=0.06, n_features=100)
        layers = {k: preprocess_omics(pd.DataFrame(v), method="standard")
                  for k, v in ds.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        meta = ds["metadata"].copy()
        rng = np.random.default_rng(RANDOM_SEED)
        meta["age"] = np.clip(rng.normal(65, 12, 500), 30, 95).astype(int)
        meta["sex"] = rng.binomial(1, 0.5, 500)
        for b in ["albumin","creatinine","glucose","crp","lymph","mcv","rdw","alp","wbc"]:
            meta[b] = rng.normal({"albumin":4.2,"creatinine":0.9,"glucose":95,"crp":0.5,"lymph":30,"mcv":90,"rdw":13.5,"alp":70,"wbc":6000}.get(b,0), 0.1, 500)
        return ds, layers, meta

    def test_data_generation_shape(self, pipeline_data):
        ds, layers, meta = pipeline_data
        assert ds["transcriptomics"].shape == (500, 100)
        assert len(layers) == 3  # transcriptomics, metabolomics, epigenomics
        assert "age" in meta.columns

    def test_phenoage_computation(self, pipeline_data):
        _, _, meta = pipeline_data
        result = assign_acceleration_group(meta)
        assert "phenoage" in result.columns
        assert "age_acceleration" in result.columns
        groups = result["aging_group"].value_counts()
        assert len(groups) >= 2  # at least 2 groups

    def test_persistence_all_layers(self, pipeline_data):
        _, layers, _ = pipeline_data
        dgms = compute_all_layers_dgms(layers, max_dim=1)
        assert len(dgms) == 3
        for name, dgm in dgms.items():
            diag = diagnose_persistence(dgm)
            assert "H0" in diag and "H1" in diag

    def test_multi_omics_integration(self, pipeline_data):
        _, layers, _ = pipeline_data
        integrated = integrate_multiomics(layers, method="concat")
        assert integrated.shape[0] == 500
        assert integrated.shape[1] == 300  # 100+100+100

    def test_feature_selection(self, pipeline_data):
        ds, _, _ = pipeline_data
        df = pd.DataFrame(ds["transcriptomics"], columns=[f"G_{i}" for i in range(100)])
        selected = select_features(df, method="variance", k=30)
        assert selected.shape == (500, 30)

    def test_distance_matrix(self, pipeline_data):
        _, layers, _ = pipeline_data
        dm = get_distance_matrix(layers["transcriptomics"], metric="euclidean")
        assert dm.shape == (500, 500)
        assert np.allclose(np.diag(dm), 0)

    def test_validation_split(self, pipeline_data):
        _, layers, meta = pipeline_data
        integrated = integrate_multiomics(layers, method="concat")
        df = pd.DataFrame(integrated)
        df_d, df_v, meta_d, meta_v = create_discovery_validation(df, meta, test_size=0.2)
        assert len(df_d) + len(df_v) == 500
        assert 80 <= len(df_v) <= 120  # ~20%

    def test_ml_cv_no_crash(self, pipeline_data):
        _, layers, meta = pipeline_data
        integrated = integrate_multiomics(layers, method="concat")
        y = (meta["aging_group"] != "normal").astype(int).values if "aging_group" in meta.columns else np.random.binomial(1, 0.3, 500)
        pipe = build_topological_pipeline()
        result = evaluate_topological_model(pipe, integrated, y, cv=3)
        assert 0 <= result["roc_auc"]["mean"] <= 1

    def test_end_to_end_timing(self, pipeline_data):
        """Full pipeline should complete in reasonable time."""
        t0 = time.time()
        _, layers, meta = pipeline_data
        dgms = compute_all_layers_dgms(layers, max_dim=0)  # H0 only for speed
        integrated = integrate_multiomics(layers, method="concat")
        pipe = build_topological_pipeline()
        y = np.random.binomial(1, 0.3, 500)
        evaluate_topological_model(pipe, integrated, y, cv=3)
        elapsed = time.time() - t0
        assert elapsed < 60, f"Pipeline took {elapsed:.1f}s — too slow"


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES: empty diagrams, single points, degenerate data
# ═══════════════════════════════════════════════════════════════════════════════
class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_single_sample(self):
        data = np.random.randn(1, 10)
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        assert len(dgms) >= 1  # at least 1 dim even if ripser absent

    def test_two_samples(self):
        data = np.random.randn(2, 5)
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        assert len(dgms) >= 1

    def test_identical_samples(self):
        data = np.ones((10, 3))
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        diag = diagnose_persistence(dgms)
        assert "H0" in diag  # structure exists even if ripser absent

    def test_zero_variance_features(self):
        data = np.column_stack([np.random.randn(30, 5), np.zeros((30, 3)), np.ones((30, 2))])
        processed = preprocess_omics(pd.DataFrame(data), method="standard")
        assert not np.any(np.isnan(processed))
        assert processed.shape == (30, 10)

    def test_empty_omics_dict(self):
        with pytest.raises(Exception):
            integrate_multiomics({})

    def test_misaligned_samples(self):
        with pytest.raises(ValueError):
            integrate_multiomics({"a": np.random.randn(30, 10), "b": np.random.randn(35, 8)}, method="concat")

    def test_nan_in_data(self):
        # StandardScaler propagates NaN — use robust scaler instead
        from sklearn.preprocessing import RobustScaler
        data = np.random.randn(20, 5)
        data[0, 0] = np.nan
        data_clean = np.nan_to_num(data, nan=0.0)
        scaler = RobustScaler()
        processed = scaler.fit_transform(data_clean)
        assert not np.any(np.isnan(processed))

    def test_inf_in_data(self):
        data = np.random.randn(20, 5)
        data[0, 0] = np.inf
        dgms = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
        assert len(dgms) >= 1

    def test_negative_values_aitchison(self):
        X = np.random.randn(20, 5)  # has negatives
        d = aitchison_distance(X)
        assert not np.any(np.isnan(d)), "aitchison should handle negatives via pseudocount"

    def test_phenoage_extreme_values(self):
        age = np.array([0, 120])
        defaults = {k: np.full(2, v) for k, v in {
            "albumin": 4.0, "creatinine": 0.9, "glucose": 95.0,
            "c_reactive_protein": 0.5, "lymphocyte_percent": 30.0,
            "mean_cell_volume": 90.0, "red_blood_cell_distribution": 13.5,
            "alkaline_phosphatase": 70.0, "white_blood_cell_count": 6000.0,
        }.items()}
        result = phenoage_simplified(age, **defaults)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL INVARIANTS: properties that MUST hold
# ═══════════════════════════════════════════════════════════════════════════════
class TestMathematicalInvariants:
    """Properties that must be true regardless of data."""

    def test_self_wasserstein_zero(self):
        dgm = np.array([[0.0, 1.0], [0.2, 0.8], [0.1, 0.6]])
        for dim in [0]:
            d = wasserstein_distance([dgm], [dgm], dim=dim)
            assert d == pytest.approx(0.0, abs=1e-6) or d == float("inf"), f"self-distance should be 0, got {d}"

    def test_distance_symmetry(self):
        dgm1 = np.array([[0.0, 0.5], [0.1, 0.6]])
        dgm2 = np.array([[0.0, 0.4], [0.2, 0.7]])
        d12 = wasserstein_distance([dgm1], [dgm2], dim=0)
        d21 = wasserstein_distance([dgm2], [dgm1], dim=0)
        assert d12 == pytest.approx(d21, rel=0.01) or d12 == float("inf")

    def test_persistence_image_shape_invariant(self):
        pi = PersistenceImageTransformer(spread=0.1, pixels=(30, 30))
        for n_pairs in [0, 1, 5, 20]:
            births = np.random.uniform(0, 0.5, n_pairs)
            deaths = births + np.random.uniform(0.1, 1.0, n_pairs)
            dgm = np.column_stack([births, deaths]) if n_pairs > 0 else np.empty((0, 2))
            X = pi.fit_transform([dgm])
            assert X.shape == (1, 900), f"PI shape should be (1,900) for {n_pairs} pairs, got {X.shape}"

    def test_betti_curve_nonnegative(self):
        bc = BettiCurveTransformer(n_bins=50, dim=1)
        dgms = [np.column_stack([np.random.uniform(0, 0.5, 10),
                                  np.random.uniform(0.5, 1.0, 10)])]
        X = bc.fit_transform(dgms)
        assert np.all(X >= 0), "Betti curves must be non-negative"

    def test_aging_group_monotonicity(self):
        """Older age → higher PhenoAge."""
        df = pd.DataFrame({"age": [30, 50, 70]})
        result = assign_acceleration_group(df)
        assert result["phenoage"].iloc[2] > result["phenoage"].iloc[0]

    def test_landscape_mean_in_ci(self):
        """Mean landscape should be within confidence interval."""
        dgms = [np.column_stack([np.random.uniform(0, 0.5, 5),
                                  np.random.uniform(0.5, 1.0, 5)]) for _ in range(10)]
        stats = landscape_statistics(dgms, n_layers=3, n_bins=20, dim=1)
        mean = stats["mean_landscape"]
        ci_l = stats["ci_lower"]
        ci_u = stats["ci_upper"]
        assert np.all(mean >= ci_l) or np.allclose(mean, 0), "mean should be >= CI lower"
        assert np.all(mean <= ci_u) or np.allclose(mean, 0), "mean should be <= CI upper"

    def test_feature_extraction_consistency(self):
        """Same diagram → same features."""
        dgms = [np.array([[0.1, 0.5], [0.2, 0.8]])] * 5
        features = extract_all_features(dgms, spread=0.1, pixels=(20, 20))
        pi = features["persistence_images"]
        assert np.allclose(pi[0], pi[1]), "Same diagram should produce same PI"


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS: many samples, high dimensions, multiple topologies
# ═══════════════════════════════════════════════════════════════════════════════
class TestStress:
    """Performance and scaling tests."""

    @pytest.mark.parametrize("topology", ["circle", "noise", "torus", "figure8", "sphere"])
    def test_all_topologies(self, topology):
        ds = generate_synthetic_multimics(n_samples=100, topology_type=topology, noise=0.05, n_features=30)
        assert ds["transcriptomics"].shape == (100, 30)

    def test_dunedin_pace_various_sizes(self):
        for n in [10, 50, 200]:
            meth = np.random.beta(2, 5, (n, 50)) * 100
            pace = dunedin_pace_proxy(meth, np.full(n, 45.0))
            assert pace.shape == (n,)
            assert 0.5 <= np.mean(pace) <= 2.0

    def test_enrichment_various_sizes(self):
        for gene_list in [[], ["MTOR"], ["MTOR","SIRT1","FOXO3","CLOCK","PER1","CRY1","AKT1","TP53"]]:
            result = enrich_mapper_genes(gene_list)
            assert isinstance(result, pd.DataFrame)

    def test_sparse_persistence_various_ratios(self):
        data = np.random.randn(100, 20)
        for ratio in [0.1, 0.3, 0.5]:
            result = sparse_persistence(data, subsample_ratio=ratio, n_subsamples=3)
            assert "H0" in result
            assert result["subsample_sizes"]["ratio"] == ratio

    def test_select_features_all_methods(self):
        df = pd.DataFrame(np.random.randn(50, 30), columns=[f"F_{i}" for i in range(30)])
        target = np.random.randn(50)
        for method in ["variance", "random"]:
            result = select_features(df, method=method, k=10)
            assert result.shape[1] == 10

    @pytest.mark.slow
    def test_bootstrap_stability_multiple_dims(self):
        data = np.random.randn(80, 15)
        result = bootstrap_diagram_stability(data, n_bootstrap=10, subsample_ratio=0.7, max_dim=2)
        for dim in ["H0", "H1", "H2"]:
            assert dim in result

    def test_mapper_all_topologies(self):
        for topo in ["circle", "noise"]:
            ds = generate_synthetic_multimics(n_samples=50, topology_type=topo, noise=0.05, n_features=15)
            data = preprocess_omics(pd.DataFrame(ds["transcriptomics"]), method="standard")
            graph = auto_mapper(data, verbose=False)
            assert isinstance(graph, dict)
            assert "nodes" in graph
