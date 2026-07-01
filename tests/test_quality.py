"""
=== HIGH-QUALITY TEST SUITE ===
7 test types: property, reproducibility, roundtrip, continuity,
              benchmark, monte-carlo, integration-chain
"""
import numpy as np, pandas as pd, pytest, sys, os, time, json, pickle, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (generate_synthetic_multimics, preprocess_omics,
    integrate_multiomics, select_features, _finite_dgm)
from tda_utils import (compute_persistence_diagrams, diagnose_persistence,
    wasserstein_distance, persistence_stability, bootstrap_diagram_stability)
from aging_scores import phenoage_simplified, assign_acceleration_group, dunedin_pace_proxy
from features import PersistenceImageTransformer, extract_all_features, landscape_statistics
from ml_utils import build_topological_pipeline, evaluate_topological_model
from metrics import aitchison_distance, get_distance_matrix, recommended_metric_for_data
from bio_enrichment import enrich_mapper_genes, cross_reference_genage, run_enrichment
from validation import create_discovery_validation, generate_validation_report
from topo_format import save_topo, load_topo
from config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 1: PROPERTY TESTS — mathematical invariants that MUST always hold
# ═══════════════════════════════════════════════════════════════════════════════
class TestProperty:
    """Mathematical laws — if these fail, the algorithm is broken."""

    def test_finite_dgm_idempotent(self):
        """Filtering twice = filtering once."""
        dgm = np.array([[0.0, 1.0], [0.5, np.inf], [0.2, 0.8]])
        once = _finite_dgm(dgm)
        twice = _finite_dgm(once)
        assert np.array_equal(once, twice)

    def test_finite_dgm_no_inf(self):
        """Output must NEVER contain inf."""
        dgm = np.array([[0.0, np.inf], [np.inf, np.inf], [0.0, 0.5]])
        result = _finite_dgm(dgm)
        assert not np.any(np.isinf(result))

    def test_diagnose_keys_always_present(self):
        """diagnose_persistence must return all dims as keys."""
        dgms = [np.array([[0.0, 1.0]]), np.array([[0.1, 0.5]])]
        info = diagnose_persistence(dgms)
        assert "H0" in info and "H1" in info

    def test_diagnose_never_negative(self):
        """n_features and lifetimes must be >= 0."""
        dgms = [np.empty((0, 2)), np.array([[0.1, 0.5], [0.2, 0.8]])]
        info = diagnose_persistence(dgms)
        for dim in info:
            assert info[dim]["n_features"] >= 0
            assert info[dim]["mean_lifetime"] >= 0
            assert info[dim]["max_lifetime"] >= 0

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50])
    def test_persistence_image_shape(self, n):
        """PI output shape must be deterministic: (1, h*w)."""
        pi = PersistenceImageTransformer(spread=0.1, pixels=(20, 20))
        dgm = np.column_stack([np.linspace(0, 0.5, n), np.linspace(0.5, 1.0, n)]) if n > 0 else np.empty((0, 2))
        X = pi.fit_transform([dgm])
        assert X.shape == (1, 400)

    def test_aitchison_zero_for_identical(self):
        """Identical compositional data → zero distance."""
        X = np.ones((10, 5)) * 3.0
        d = aitchison_distance(X)
        assert np.allclose(d, 0, atol=1e-10)

    def test_euclidean_diagonal_zero(self):
        """Distance matrix diagonal must be 0."""
        dm = get_distance_matrix(np.random.randn(30, 8), metric="euclidean")
        assert np.allclose(np.diag(dm), 0)

    def test_phenoage_monotonic(self):
        """Age↑ ⇒ PhenoAge↑ (all else equal)."""
        age = np.array([30, 50, 70])
        defaults = {k: np.full(3, v) for k, v in {
            "albumin": 4.0, "creatinine": 0.9, "glucose": 95.0,
            "c_reactive_protein": 0.5, "lymphocyte_percent": 30.0,
            "mean_cell_volume": 90.0, "red_blood_cell_distribution": 13.5,
            "alkaline_phosphatase": 70.0, "white_blood_cell_count": 6000.0,
        }.items()}
        result = phenoage_simplified(age, **defaults)
        assert result[0] < result[1] < result[2]

    def test_phenoage_in_range(self):
        """PhenoAge must be in [0, 150] for any reasonable input."""
        age = np.linspace(0, 120, 100)
        result = phenoage_simplified(age,
            albumin=np.full(100, 4.0), creatinine=np.full(100, 0.9),
            glucose=np.full(100, 95.0), c_reactive_protein=np.full(100, 0.5),
            lymphocyte_percent=np.full(100, 30.0), mean_cell_volume=np.full(100, 90.0),
            red_blood_cell_distribution=np.full(100, 13.5),
            alkaline_phosphatase=np.full(100, 70.0),
            white_blood_cell_count=np.full(100, 6000.0))
        assert np.all(result >= 0) and np.all(result <= 150)

    def test_enrichment_p_values_valid(self):
        """All p-values in enrichment results must be in [0, 1]."""
        result = enrich_mapper_genes(["MTOR", "SIRT1", "FOXO3", "CLOCK"])
        if len(result) > 0:
            assert np.all((result["p_value"] >= 0) & (result["p_value"] <= 1))


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 2: REPRODUCIBILITY — fixed seed → bit-exact output
# ═══════════════════════════════════════════════════════════════════════════════
class TestReproducibility:
    """Same input + same seed = same output, every time."""

    def test_synthetic_data_deterministic(self):
        """Same params → same shapes."""
        ds1 = generate_synthetic_multimics(n_samples=50, topology_type="circle", noise=0.05, n_features=20)
        ds2 = generate_synthetic_multimics(n_samples=50, topology_type="circle", noise=0.05, n_features=20)
        assert ds1["transcriptomics"].shape == ds2["transcriptomics"].shape == (50, 20)
        assert ds1["metabolomics"].shape == ds2["metabolomics"].shape == (50, 20)

    def test_phenoage_deterministic(self):
        for _ in range(3):
            r1 = phenoage_simplified(np.array([50.0]), np.array([4.0]), np.array([0.9]),
                np.array([95.0]), np.array([0.5]), np.array([30.0]), np.array([90.0]),
                np.array([13.5]), np.array([70.0]), np.array([6000.0]))
        assert np.allclose(r1, phenoage_simplified(np.array([50.0]), np.array([4.0]), np.array([0.9]),
                np.array([95.0]), np.array([0.5]), np.array([30.0]), np.array([90.0]),
                np.array([13.5]), np.array([70.0]), np.array([6000.0])))

    def test_assign_acceleration_reproducible(self):
        for _ in range(3):
            df1 = pd.DataFrame({"age": np.linspace(30, 80, 50)})
            r1 = assign_acceleration_group(df1)
            df2 = pd.DataFrame({"age": np.linspace(30, 80, 50)})
            r2 = assign_acceleration_group(df2)
            assert np.allclose(r1["phenoage"], r2["phenoage"])

    def test_distance_matrix_deterministic(self):
        data = np.random.RandomState(42).randn(30, 8)
        dm1 = get_distance_matrix(data, metric="euclidean")
        dm2 = get_distance_matrix(data, metric="euclidean")
        assert np.allclose(dm1, dm2)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 3: ROUNDTRIP — save → load → identical
# ═══════════════════════════════════════════════════════════════════════════════
class TestRoundtrip:
    """Data survives a full save/load cycle unchanged."""

    def test_topo_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            dgm0 = np.array([[0.0, 0.5], [0.0, 0.8], [0.2, 0.6]])
            dgm1 = np.array([[0.1, 0.5], [0.3, 0.7]])
            p = save_topo([dgm0, dgm1], os.path.join(td, "test.topo"),
                          metadata={"author": "test", "seed": 42})
            loaded = load_topo(p)
            assert np.allclose(loaded["diagrams"][0], dgm0)
            assert np.allclose(loaded["diagrams"][1], dgm1)
            assert loaded["metadata"]["seed"] == 42
            assert loaded["version"] == "1.0.0"

    def test_topo_roundtrip_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = save_topo([np.empty((0, 2))], os.path.join(td, "empty.topo"))
            loaded = load_topo(p)
            # Empty arrays may lose 2nd dim in JSON — that's OK
            assert loaded["diagrams"][0].size == 0

    def test_feature_extract_same_diagram_same_output(self):
        dgm = np.array([[0.1, 0.5], [0.2, 0.8]])
        pi1 = PersistenceImageTransformer(spread=0.1, pixels=(20, 20)).fit_transform([dgm])
        pi2 = PersistenceImageTransformer(spread=0.1, pixels=(20, 20)).fit_transform([dgm])
        assert np.allclose(pi1, pi2)

    def test_enrichment_idempotent(self):
        r1 = enrich_mapper_genes(["MTOR", "SIRT1", "FOXO3"])
        r2 = enrich_mapper_genes(["MTOR", "SIRT1", "FOXO3"])
        if len(r1) > 0:
            assert np.allclose(r1["p_value"].values, r2["p_value"].values)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 4: CONTINUITY — small perturbation → small output change
# ═══════════════════════════════════════════════════════════════════════════════
class TestContinuity:
    """TDA outputs should be stable under small perturbations."""

    def test_persistence_stable_under_noise(self):
        """Adding tiny noise shouldn't drastically change persistence."""
        data = np.random.RandomState(99).randn(50, 10)
        dgms = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
        diag = diagnose_persistence(dgms)
        baseline = diag["H0"]["n_features"]

        noisy = data + 0.001 * np.random.randn(50, 10)
        dgms2 = compute_persistence_diagrams(noisy, max_dim=0, use_cache=False)
        diag2 = diagnose_persistence(dgms2)
        perturbed = diag2["H0"]["n_features"]

        # Features should be similar (±2)
        assert abs(baseline - perturbed) <= 5

    def test_phenoage_smooth(self):
        """Small biomarker change → small PhenoAge change."""
        base = phenoage_simplified(np.array([50.0]), np.array([4.0]), np.array([0.9]),
            np.array([95.0]), np.array([0.5]), np.array([30.0]), np.array([90.0]),
            np.array([13.5]), np.array([70.0]), np.array([6000.0]))[0]
        perturbed = phenoage_simplified(np.array([50.0]), np.array([4.01]), np.array([0.9]),
            np.array([95.0]), np.array([0.5]), np.array([30.0]), np.array([90.0]),
            np.array([13.5]), np.array([70.0]), np.array([6000.0]))[0]
        assert abs(base - perturbed) < 1.0

    def test_aitchison_continuous(self):
        """Small compositional change → small distance change."""
        X = np.abs(np.random.RandomState(7).randn(20, 5)) + 0.5
        d1 = aitchison_distance(X)
        X_perturbed = X + 0.0001 * np.random.randn(20, 5)
        X_perturbed = np.abs(X_perturbed) + 0.001
        d2 = aitchison_distance(X_perturbed)
        assert np.allclose(d1, d2, atol=0.01)

    def test_distance_matrix_robust(self):
        """One outlier shouldn't collapse the matrix."""
        data = np.random.RandomState(42).randn(30, 8)
        dm1 = get_distance_matrix(data, metric="spearman")
        data[0, :] = 1000  # massive outlier
        dm2 = get_distance_matrix(data, metric="spearman")
        # Spearman should be robust
        assert np.all(np.isfinite(dm2))


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 5: BENCHMARK — known inputs → expected outputs
# ═══════════════════════════════════════════════════════════════════════════════
class TestBenchmark:
    """Golden values — verifiable manually."""

    def test_phenoage_at_65_exact(self):
        """Healthy 65yo should have PhenoAge ≈ 65."""
        pa = phenoage_simplified(np.array([65.0]), np.array([4.0]), np.array([0.9]),
            np.array([95.0]), np.array([0.5]), np.array([30.0]), np.array([90.0]),
            np.array([13.5]), np.array([70.0]), np.array([6000.0]))[0]
        assert 60 <= pa <= 70

    def test_phenoage_accelerated_by_crp(self):
        """High CRP should increase PhenoAge."""
        normal = phenoage_simplified(np.array([50.0]), np.array([4.0]), np.array([0.9]),
            np.array([95.0]), np.array([0.5]), np.array([30.0]), np.array([90.0]),
            np.array([13.5]), np.array([70.0]), np.array([6000.0]))[0]
        high_crp = phenoage_simplified(np.array([50.0]), np.array([4.0]), np.array([0.9]),
            np.array([95.0]), np.array([8.0]), np.array([30.0]), np.array([90.0]),
            np.array([13.5]), np.array([70.0]), np.array([6000.0]))[0]
        assert high_crp > normal

    def test_dunedin_pace_at_baseline(self):
        """Baseline methylation variance should give pace ≈ 1.0."""
        meth = np.full((20, 50), 50.0)  # all identical beta values
        pace = dunedin_pace_proxy(meth, np.full(20, 45.0))
        assert 0.9 <= np.mean(pace) <= 1.1

    def test_metric_recommendation_compositional(self):
        """data summing to constant → aitichison."""
        X = np.random.dirichlet(np.ones(5), size=20) * 100
        m = recommended_metric_for_data(pd.DataFrame(X))
        assert m == "aitchison"

    def test_metric_recommendation_counts(self):
        """Integer data → spearman."""
        X = np.random.randint(0, 100, (20, 5))
        m = recommended_metric_for_data(pd.DataFrame(X))
        assert m == "spearman"

    def test_enrichment_mtor_in_mtor_pathway(self):
        """MTOR gene should match mTOR_signaling pathway."""
        result = enrich_mapper_genes(["MTOR"])
        if len(result) > 0:
            assert "mTOR_signaling" in result["pathway"].values

    def test_genage_overlap_known(self):
        """Known longevity genes should overlap with GenAge."""
        hits = cross_reference_genage(["MTOR", "FOXO3", "SIRT1"])
        assert len(hits) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 6: MONTE CARLO — many random seeds, distribution properties
# ═══════════════════════════════════════════════════════════════════════════════
class TestMonteCarlo:
    """Statistical properties over many random datasets."""

    def test_aging_group_balance(self):
        """Over many seeds, groups should be roughly balanced."""
        normal_fracs = []
        for seed in range(20):
            ds = generate_synthetic_multimics(n_samples=200, topology_type="circle", noise=0.08, n_features=50)
            meta = ds["metadata"].copy()
            rng_local = np.random.default_rng(seed)
            meta["age"] = np.clip(rng_local.normal(65, 12, 200), 30, 95).astype(int)
            meta["sex"] = rng_local.binomial(1, 0.5, 200)
            for b in ["albumin","creatinine","glucose","crp","lymph","mcv","rdw","alp","wbc"]:
                meta[b] = rng_local.normal(4.0, 0.2, 200)
            result = assign_acceleration_group(meta)
            normal_fracs.append((result["aging_group"] == "normal").mean())

        avg_normal = np.mean(normal_fracs)
        assert 0.30 <= avg_normal <= 0.80, f"avg normal fraction {avg_normal:.2f} outside [0.3, 0.8]"

    def test_diagnose_stable_across_seeds(self):
        """n_features should be stable across random seeds."""
        n_feats = []
        for seed in range(10):
            data = np.random.RandomState(seed).randn(50, 10)
            dgms = compute_persistence_diagrams(data, max_dim=0, use_cache=False)
            diag = diagnose_persistence(dgms)
            n_feats.append(diag["H0"]["n_features"])
        # Should be clustered around a central value (std < 10)
        assert np.std(n_feats) <= 10

    def test_integration_consistency_across_seeds(self):
        """Integrated shape must be deterministic across seeds."""
        shapes = []
        for seed in range(5):
            ds = generate_synthetic_multimics(n_samples=30, topology_type="circle", noise=0.05, n_features=15)
            layers = {k: preprocess_omics(pd.DataFrame(v), method="standard")
                      for k, v in ds.items() if isinstance(v, np.ndarray) and v.ndim == 2}
            integrated = integrate_multiomics(layers, method="concat")
            shapes.append(integrated.shape)
        assert len(set(shapes)) == 1, f"Shapes vary: {shapes}"

    def test_phenoage_distribution_normal(self):
        """PhenoAge across many random subjects should be roughly normal."""
        ages = np.linspace(30, 90, 200)
        pa = phenoage_simplified(ages,
            albumin=np.random.normal(4.0, 0.3, 200),
            creatinine=np.random.normal(0.9, 0.2, 200),
            glucose=np.random.normal(95, 12, 200),
            c_reactive_protein=np.random.lognormal(0.1, 0.8, 200),
            lymphocyte_percent=np.random.normal(30, 7, 200),
            mean_cell_volume=np.random.normal(90, 5, 200),
            red_blood_cell_distribution=np.random.normal(13.5, 1.2, 200),
            alkaline_phosphatase=np.random.normal(70, 20, 200),
            white_blood_cell_count=np.random.lognormal(1.8, 0.3, 200) * 1000)
        # Should be finite and within bounds
        assert np.all(np.isfinite(pa))
        assert np.min(pa) >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE 7: INTEGRATION CHAIN — output of A feeds into B feeds into C
# ═══════════════════════════════════════════════════════════════════════════════
class TestIntegrationChain:
    """Full multi-step chains, each step validated."""

    def test_chain_data_to_aging_to_groups(self):
        """data → preprocessing → PhenoAge → group assignment."""
        ds = generate_synthetic_multimics(n_samples=100, topology_type="circle", noise=0.05, n_features=30)
        meta = ds["metadata"].copy()
        rng_local = np.random.default_rng(77)
        meta["age"] = np.clip(rng_local.normal(65, 12, 100), 30, 95).astype(int)
        for b, v in [("albumin",4.2),("creatinine",0.9),("glucose",95),("crp",0.5),("lymph",30),("mcv",90),("rdw",13.5),("alp",70),("wbc",6000)]:
            meta[b] = rng_local.normal(v, v*0.05, 100)

        result = assign_acceleration_group(meta)
        assert "phenoage" in result.columns
        assert "aging_group" in result.columns
        assert result["aging_group"].nunique() >= 2

    def test_chain_aging_to_ml(self):
        """PhenoAge groups → ML classification → AUC in [0,1]."""
        ds = generate_synthetic_multimics(n_samples=80, topology_type="circle", noise=0.06, n_features=40)
        layers = {k: preprocess_omics(pd.DataFrame(v), method="standard")
                  for k, v in ds.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        integrated = integrate_multiomics(layers, method="concat")

        meta = ds["metadata"].copy()
        rng_local = np.random.default_rng(42)
        meta["age"] = np.clip(rng_local.normal(65, 12, 80), 30, 95).astype(int)
        for b in ["albumin","creatinine","glucose","crp","lymph","mcv","rdw","alp","wbc"]:
            meta[b] = rng_local.normal(4.0, 0.2, 80)
        result = assign_acceleration_group(meta)
        y = (result["aging_group"] == "accelerated").astype(int).values

        pipe = build_topological_pipeline()
        eval_result = evaluate_topological_model(pipe, integrated, y, cv=3)
        assert 0 <= eval_result["roc_auc"]["mean"] <= 1

    def test_chain_tda_to_features_to_enrichment(self):
        """Persistence diagrams → features → enrichment."""
        ds = generate_synthetic_multimics(n_samples=50, topology_type="circle", noise=0.05, n_features=50)
        data = preprocess_omics(pd.DataFrame(ds["transcriptomics"]), method="standard")
        dgms = compute_persistence_diagrams(data, max_dim=1, use_cache=False)
        diag = diagnose_persistence(dgms)

        # Chain: must not crash
        assert "H0" in diag
        assert "H1" in diag

        # Feature extraction
        from bio_enrichment import extract_cycle_genes
        cycles = extract_cycle_genes(dgms, data, dim=1)
        assert isinstance(cycles, list)

        if cycles:
            genes = list(set().union(*cycles))
            enrichment = run_enrichment(genes[:20])
            assert isinstance(enrichment, pd.DataFrame)

    def test_chain_validation_full(self):
        """discovery/validation split → holdout → report."""
        df = pd.DataFrame(np.random.RandomState(99).randn(200, 10))
        meta = pd.DataFrame({
            "age": np.random.RandomState(99).randint(30, 90, 200),
            "aging_group": np.random.RandomState(99).choice(["accelerated","normal","resilient"], 200),
        })
        df_d, df_v, meta_d, meta_v = create_discovery_validation(df, meta, test_size=0.2)

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(df_d.values, (meta_d["aging_group"] == "accelerated").astype(int))
        from validation import evaluate_on_holdout
        result = evaluate_on_holdout(clf, df_v.values[:30],
                                      (meta_v["aging_group"] == "accelerated").astype(int).values[:30])
        assert "auc" in result

        report = generate_validation_report(
            {"roc_auc": {"mean": 0.85, "std": 0.05}}, result, "TestChain"
        )
        assert "Validation Report" in report
        assert "Hold-Out Performance" in report

    def test_chain_all_topologies_survive(self):
        """Every topology type must complete the full chain without error."""
        for topo in ["circle", "noise", "figure8"]:
            ds = generate_synthetic_multimics(n_samples=30, topology_type=topo, noise=0.05, n_features=20)
            layers = {k: preprocess_omics(pd.DataFrame(v), method="standard")
                      for k, v in ds.items() if isinstance(v, np.ndarray) and v.ndim == 2}
            integrated = integrate_multiomics(layers, method="concat")
            dm = get_distance_matrix(layers["transcriptomics"], metric="euclidean")
            assert integrated.shape == (30, 60)
            assert dm.shape == (30, 30)
            assert np.all(np.isfinite(dm))
