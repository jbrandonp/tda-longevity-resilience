# Mathematical Background — Topological Data Analysis

## 1. Persistent Homology

Persistent homology captures multi-scale topological features of a dataset. Given a point cloud $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$, we build a nested sequence (filtration) of simplicial complexes:

$$K_{t_0} \subseteq K_{t_1} \subseteq \dots \subseteq K_{t_m}$$

### Vietoris-Rips Filtration

For a distance threshold $\epsilon$, the Vietoris-Rips complex $VR_\epsilon(X)$ contains a $k$-simplex for every set of $k+1$ points with pairwise distance $\leq \epsilon$. As $\epsilon$ increases, simplices are added, and topological features (connected components, loops, voids) appear and disappear.

### Persistence Diagrams

A persistence diagram $D = \{(b_i, d_i)\}_{i=1}^k$ records the birth $b_i$ and death $d_i$ of each topological feature. Features with large $d_i - b_i$ (long lifetime) are considered "persistent" — likely to represent real structure rather than noise.

### Homology Groups

- $H_0$: Connected components (0-dimensional holes)
- $H_1$: Loops / cycles (1-dimensional holes)
- $H_2$: Voids / cavities (2-dimensional holes)

## 2. Distances Between Diagrams

### Wasserstein Distance
$$W_p(D_1, D_2) = \left( \inf_{\gamma} \sum_{x \in D_1} \|x - \gamma(x)\|_\infty^p \right)^{1/p}$$
where $\gamma$ is a partial matching between diagrams.

### Bottleneck Distance
$$W_\infty(D_1, D_2) = \inf_\gamma \sup_x \|x - \gamma(x)\|_\infty$$

## 3. The Mapper Algorithm

Mapper (Singh, Mémoli, Carlsson 2007) constructs a simplicial complex representing the topological structure of data:

1. **Filter (lens):** $f: X \to \mathbb{R}^k$ (e.g., UMAP, PCA, density)
2. **Cover:** Partition the image $f(X)$ into overlapping intervals
3. **Pullback:** For each interval, cluster the preimage $f^{-1}(U_i)$
4. **Nerve:** Build a graph where nodes = clusters, edges = overlapping clusters

## 4. Topological Feature Vectors

### Persistence Images (Adams et al. 2017)
Map each persistence pair $(b, d)$ to a Gaussian kernel centered at $(b, d)$ and integrate over a pixel grid.

### Persistence Landscapes (Bubenik 2015)
For each $(b, d)$, define a piecewise-linear function $\Lambda(t) = \max(0, \min(t-b, d-t))$. The $k$-th landscape is the $k$-th largest value at each $t$.

### Betti Curves
$\beta_k(t) = |\{ (b_i, d_i) \in D_k : b_i \leq t < d_i \}|$ — count of active features at filtration time $t$.

## References

- Edelsbrunner, Letscher, Zomorodian (2002). Topological persistence and simplification.
- Carlsson (2009). Topology and data. *Bull. AMS*.
- Bubenik (2015). Statistical topological data analysis using persistence landscapes.
- Adams et al. (2017). Persistence images: A stable vector representation of persistent homology.
- Singh, Mémoli, Carlsson (2007). Topological methods for the analysis of high dimensional data sets and 3D object recognition.
