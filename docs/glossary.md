# Glossary

## Core TDA Terms

| Term | Definition |
|------|-----------|
| **Barcode** | Visual representation of persistence: horizontal bars span [birth, death] for each topological feature. |
| **Betti curve** | $\beta_k(t)$: number of active $k$-dimensional features at filtration time $t$. |
| **Bottleneck distance** | $W_\infty(D_1, D_2)$: maximum displacement of matched points between two persistence diagrams. |
| **Diagram (persistence)** | Multiset of (birth, death) pairs, one for each homology dimension. |
| **Filtration** | Nested sequence of simplicial complexes $K_0 \subseteq K_1 \subseteq \dots$ built by varying a parameter. |
| **Homology** | Algebraic invariant counting $k$-dimensional holes in a space: $H_0$ = components, $H_1$ = loops, $H_2$ = voids. |
| **Mapper** | Algorithm constructing a simplicial complex from a filter function, cover, and clustering of preimages. |
| **Persistence image** | Vector representation of a diagram: sum of Gaussians centered at (birth, death), integrated over pixels. |
| **Persistence landscape** | Piecewise-linear function derived from a diagram, enabling statistical analysis (Bubenik 2015). |
| **Persistence (lifetime)** | $d_i - b_i$: how long a feature persists in the filtration. Longer = more significant. |
| **Simplicial complex** | Generalization of a graph: vertices, edges, triangles, tetrahedra, etc. |
| **Vietoris-Rips complex** | $VR_\epsilon$: $k$-simplex for every $k+1$ points within distance $\epsilon$ of each other. |
| **Wasserstein distance** | $W_p$: $L^p$ optimal transport distance between two persistence diagrams. |

## Biological Terms

| Term | Definition |
|------|-----------|
| **Accelerated aging** | Biological age > chronological age. Measured by epigenetic clocks (Horvath, GrimAge, PhenoAge) or Tian score. |
| **Centenarian** | Individual aged ≥100 years, often exhibiting extreme longevity resilience. |
| **Epigenetic clock** | DNA methylation-based predictor of chronological age. |
| **Longevity resilience** | Ability to maintain biological function despite advanced age or stress. |
| **Multi-omics** | Integration of multiple molecular data types: genomics, transcriptomics, proteomics, metabolomics, epigenomics, microbiome. |
| **Santos-Pujol individual** | Supercentenarian providing an extreme example of longevity resilience. |
| **Tian 2026 framework** | Generational accelerated aging model defining groups via biological acceleration scores. |

## Computing Terms

| Term | Definition |
|------|-----------|
| **Binder** | Free service for running Jupyter notebooks in the cloud from a GitHub repository. |
| **CI/CD** | Continuous Integration / Continuous Deployment: automated testing and deployment pipelines. |
| **MLflow** | Open-source platform for experiment tracking and model management. |
| **SHAP** | SHapley Additive exPlanations: game-theoretic approach to model interpretability. |
