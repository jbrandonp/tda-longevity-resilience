# References — Annotated Bibliography

## Foundational TDA

1. **Edelsbrunner, Letscher, Zomorodian (2002).** Topological persistence and simplification. *Discrete & Computational Geometry*.
   - Introduced persistent homology and the persistence diagram as a stable invariant.

2. **Carlsson (2009).** Topology and data. *Bulletin of the AMS*.
   - Seminal survey of TDA applications. Introduced the Mapper algorithm conceptually.

3. **Zomorodian & Carlsson (2005).** Computing persistent homology. *Discrete & Computational Geometry*.
   - Algorithm for computing persistence over arbitrary fields.

## Mapper Algorithm

4. **Singh, Mémoli, Carlsson (2007).** Topological methods for the analysis of high dimensional data sets and 3D object recognition. *Eurographics Symposium on Point-Based Graphics*.
   - Original Mapper paper. Introduced the filter-cover-cluster pipeline.

## Topological Feature Vectors

5. **Adams et al. (2017).** Persistence images: A stable vector representation of persistent homology. *JMLR*.
   - Persistence Images: differentiable, stable vectorization of diagrams. Foundation for ML with TDA.

6. **Bubenik (2015).** Statistical topological data analysis using persistence landscapes. *JMLR*.
   - Persistence landscapes: functional summary statistics enabling classical statistical inference.

## TDA in Biology

7. **Nicolau, Levine, Carlsson (2011).** Topology based data analysis identifies a subgroup of breast cancers. *PNAS*.
   - Landmark: Mapper identified a novel breast cancer subtype missed by classical methods.

8. **Rizvi et al. (2017).** Single-cell topological RNA-seq analysis reveals insights into cellular differentiation. *Nature Biotechnology*.
   - Mapper applied to single-cell transcriptomics, revealing continuous differentiation trajectories.

9. **Lum et al. (2013).** Extracting insights from the shape of complex data using topology. *Scientific Reports*.
   - Broad survey of TDA applications in biology and medicine.

10. **Cang & Wei (2017).** TopologyNet: Topology based deep convolutional neural networks for biomolecular property predictions. *PLOS Computational Biology*.
    - Persistent homology features as input to deep learning for protein-ligand binding prediction.

## Longevity & Aging

11. **Horvath (2013).** DNA methylation age of human tissues and cell types. *Genome Biology*.
    - The original epigenetic clock. Multi-tissue predictor of chronological age.

12. **Levine et al. (2018).** An epigenetic biomarker of aging for lifespan and healthspan. *Aging*, 10(4), 573-591.
    - PhenoAge: a biological age estimator from 9 blood biomarkers + chronological age. Used as our primary aging score.

13. **Belsky et al. (2022).** DunedinPACE, a DNA methylation biomarker of the pace of aging. *eLife*, 11, e73420.
    - Pace of aging measurement; our `dunedin_pace_proxy` is inspired by this 19-CpG clock.

## Software & Tools

14. **Ripser.** Bauer (2019). Ripser: efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology*.

15. **KeplerMapper.** van Veen et al. (2019). Kepler Mapper: A flexible Python implementation of the Mapper algorithm. *JOSS*.

16. **Persim.** Saul & Tralie (2019). Persim: Python package for persistence diagram analysis. [https://persim.scikit-tda.org](https://persim.scikit-tda.org)

17. **Giotto-TDA.** Tauzin et al. (2021). giotto-tda: A topological data analysis toolkit for machine learning. *JMLR*.
