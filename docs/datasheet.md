# Datasheet — Synthetic Multi-Omics Longevity Data

## Motivation

**For what purpose was the dataset created?**
To provide a validated, reproducible testbed for Topological Data Analysis methods in longevity research. The synthetic data has known topological structure (circle, torus, etc.), enabling verification of persistence diagram computation and Mapper graph construction before applying to real omics data.

**Who created the dataset?**
Brandon Palhano Machado, via the `data_utils.generate_synthetic_multimics()` function.

## Composition

**What do the instances represent?**
Each instance represents a synthetic individual with multi-omics measurements:
- **Transcriptomics**: 20-50 gene expression features
- **Metabolomics**: 20-50 metabolite abundance features
- **Epigenomics**: 20-50 methylation features

**How many instances?**
Configurable: typically 200-500.

**What data does each instance consist of?**
Raw continuous features (numpy arrays) + metadata (tian_score, group label).

**Is there a label?**
Yes: `accelerated`, `resilient`, or `neutral` based on Tian 2026 score thresholding.

## Collection Process

**How was the data generated?**
1. A base topology (circle, torus, figure-8, sphere, noise) is generated in 2-3 dimensions
2. A random linear projection lifts the data to high-dimensional ambient space (20-50 dims)
3. Gaussian noise (σ=0.05-0.1) is added
4. Tian scores are computed from the topology and thresholded into groups

## Uses

**What tasks could this dataset be used for?**
- Validating persistent homology computation
- Testing Mapper graph stability under noise
- Benchmarking topological vs classical ML features
- Educational demonstrations of TDA

**Are there any restrictions?**
No. Fully synthetic, no privacy concerns.

## Distribution

**How is the dataset distributed?**
Generated on-demand via `generate_synthetic_multimics()`. No static files tracked in the repository.

## Maintenance

**Will the dataset be updated?**
The generation function may be extended with additional topologies (Klein bottle, projective space) in future releases.

## Known Limitations

- No real biological signal — purely mathematical structures
- Limited to 2-3D base topologies (H2 features only for sphere)
- No temporal/longitudinal dimension
- Groups are assigned by threshold, not by actual biological processes
