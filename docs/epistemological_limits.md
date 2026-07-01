# Epistemological Limits of TDA for Longevity Research

This document honestly acknowledges the mathematical, statistical, and
interpretive limitations of applying Topological Data Analysis to multi-omics
longevity data. These are not bugs — they are boundary conditions that any
rigorous reviewer would identify.

## 1. What Are We Actually Measuring?

**The object of study is not well-defined.** Persistent homology estimates the
topology of the *support* of a point cloud, assuming an underlying manifold.
Multi-omics data may not satisfy this assumption. The project does not claim
to measure a biological "ground truth" — it explores whether topological
features *correlate* with known phenotypes (accelerated vs resilient aging).
Correlation is not causation, and correlation is not topology.

## 2. Manifold Hypothesis

TDA assumes data lies on (or near) a low-dimensional manifold. We do not
verify this hypothesis before computing persistence. High-dimensional omics
data may violate it, causing:
- Concentration of measure: all distances become similar
- Saturation of Vietoris-Rips filtration near the diagonal
- Spurious cycles from sampling noise, not intrinsic structure

Mitigation: We provide `src/calibration.py` with intrinsic dimension
estimation and confidence band tools. Users should assess manifold
plausibility before interpreting results.

## 3. Sample Size and Asymptotic Guarantees

Theoretical convergence results (Chazal et al. 2008) require sample size
n → ∞. Typical longevity cohorts have n = 100-500. At these sizes:
- Bootstrap stability is limited
- "Persistent" features may reflect sampling variance, not structure
- The choice of filtration threshold significantly affects results

## 4. Vietoris-Rips in High Dimensions

Vietoris-Rips complexity grows exponentially with dimension. Above ~15
effective dimensions, the filtration becomes saturated with noise. Users
should:
- Estimate intrinsic dimension first
- Reduce to ≤50 components via PCA before TDA
- Compare with sparse/frugal alternatives (DTM, witness complexes)

## 5. Distance Metric Dependence

The topology of a Vietoris-Rips filtration depends entirely on the chosen
metric. Correlation distance is not a true metric without transformation.
Euclidean distance on unnormalized multi-omics data is dominated by the
highest-magnitude layer. We provide Aitchison distance for compositional
data and Spearman-based robust correlation distance as alternatives.

## 6. Diagram Statistics in Non-Hilbert Space

Persistence diagrams with Wasserstein/Bottleneck distance form a geodesic
space, not a vector space. This means:
- No canonical "mean diagram" (Fréchet mean is NP-hard, not unique)
- No additive variance decomposition
- No direct generalization of t-tests or ANOVA
- Permutation tests assume exchangeability (may be violated by data structure)

Our `landscape_statistics()` function uses persistence landscapes (Banach space)
as a mathematically sound alternative for group comparison.

## 7. Mapper Is Not a Topological Estimator

Unlike persistent homology, the Mapper algorithm is a visualization tool, not
a convergent estimator of manifold topology. Its output depends critically on:
- Filter choice (UMAP is not Lipschitz — stability not guaranteed)
- Cover parameters (n_cubes, overlap)
- Clustering algorithm (DBSCAN with arbitrary epsilon)

Mapper graphs can contain cycles that are artifacts of the cover, not of the
data. Graph comparison between groups is qualitative unless a formal graph
distance metric is used.

## 8. Transductivity and Data Leakage

If UMAP or PCA is fitted on the full dataset before train/test split, the
topological features are contaminated with test information. For rigorous
evaluation:
- Fit all dimensionality reduction ONLY on training data
- Apply the same transform to test data (parametric UMAP, not transductive)
- Compute persistence diagrams independently per fold

Our code does not enforce this automatically — it is the user's
responsibility.

## 9. From Topology to Biology: The Interpretive Gap

A persistent H1 cycle in gene expression space does NOT directly correspond to
a biological cycle (Krebs, circadian). It is a geometric feature of the point
cloud. The leap from "we found a hole" to "this is a resilient metabolic
pathway" requires a mechanistic model linking dynamics to topology — which
this project does not provide.

**This project detects mathematical structures. Any biological interpretation
is speculative and requires independent validation.**

## 10. What This Project Does Claim

- TDA features can discriminate between phenotype groups
- Persistent homology captures structural information invisible to linear methods
- The methodology is reproducible, documented, and open-source
- Even negative results (no topological signature found) advance the field by
  establishing a benchmark and eliminating a hypothesis

## 11. Circularity and Group Definition

**Risk:** If groups are defined by an epigenetic clock score (Tian 2026), and
TDA then separates those same groups, we may simply be recovering the clock
signal through a different mathematical lens — not discovering new biology.

Mitigation: The project should compare TDA-based separation against the raw
clock score's separation. If TDA adds no discriminative power beyond the clock
itself, the "topological signature" is circular. We also define a "neutral"
third group as control.

## 12. Interpretation: Topology ≠ Mechanism

A persistent H1 cycle is a geometric feature of a point cloud, not a metabolic
pathway. The project notes this gap but does not close it. Annotation via
KEGG/Reactome (in `bio_enrichment.py`) provides correlation, not causation.
Users and reviewers should treat biological interpretations as hypotheses for
future validation, not as findings.

## 13. Sample Limitations and Generalizability

- Longevity cohorts are small (n = 100-500) and predominantly European.
  Topological features detected may not generalize across populations.
- No statistical power analysis has been performed — the project may be
  underpowered to detect real topological differences.
- If only synthetic data is available, the project serves as a methodological
  demonstration, not a biological finding.

## References

- Chazal, Fasy, Lecci, Rinaldo, Wasserman (2014). Stochastic convergence of
  persistence landscapes and silhouettes. *JMLR*.
- Bubenik (2015). Statistical topological data analysis using persistence
  landscapes. *JMLR*.
- Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan, Singh (2014). Confidence sets
  for persistence diagrams. *Annals of Statistics*.
- Carrière, Oudot (2017). Structure and stability of the 1-dimensional Mapper.
  *SoCG*.
