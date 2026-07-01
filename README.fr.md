# TDA-Longevity-Resilience 🔬🌀

[![License: MIT](https://img.shields.io/badge/Licence-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**L'homologie persistante révèle les signatures topologiques de la résilience extrême à la longévité.**

Ce projet applique l'**Analyse Topologique de Données (TDA)** — homologie persistante et algorithme Mapper — à des données multi-omiques de longévité pour extraire des **signatures structurelles de résilience extrême** et les comparer au **vieillissement accéléré générationnel** (cadre Tian 2026).

## 🚀 Démarrage Rapide

```bash
git clone https://github.com/jbrandonp/tda-longevity-resilience.git
cd tda-longevity-resilience
conda env create -f environment.yml
conda activate tda-longevity
jupyter lab
```

## 📓 Notebooks

| # | Notebook | Objectif |
|---|----------|---------|
| 00 | `00_synthetic_validation.ipynb` | Validation TDA sur données synthétiques |
| 01 | `01_persistent_homology.ipynb` | Diagrammes de persistance par groupe |
| 02 | `02_mapper_analysis.ipynb` | Graphe Mapper et enrichissement |
| 03 | `03_comparison_accelerated_vs_resilient.ipynb` | Comparaison statistique multi-vues |
| 04 | `04_feature_extraction_ml.ipynb` | Features topologiques → classification ML |

## 🔬 Méthodes

- **Homologie Persistante** (filtration de Vietoris-Rips via `ripser`)
- **Algorithme Mapper** (via `kmapper` avec lentille UMAP)
- **Distances** : Wasserstein, Bottleneck
- **Features** : Persistence Images, Landscapes, Courbes de Betti
- **ML** : Random Forest, SVM, Gradient Boosting + SHAP

## 📚 Documentation

- [Fondements Mathématiques](docs/mathematical_background.md)
- [Interprétation Biologique](docs/biological_interpretation.md)
- [Sources de Données](docs/data_sources.md)
- [Glossaire](docs/glossary.md)
- [Références](docs/references.md)
- [Guide d'Installation](docs/installation.md)
- [Contribuer](docs/contributing.md)

## 📖 Citation

```bibtex
@software{tda_longevity_resilience_2026,
  author = {Palhano Machado, Brandon},
  title = {TDA-Longevity-Resilience: Signatures Topologiques de la Longévité Extrême},
  year = {2026},
  url = {https://github.com/jbrandonp/tda-longevity-resilience},
}
```

## 📝 Licence

MIT — voir [LICENSE](LICENSE).

---

**Mots-clés :** Analyse Topologique de Données, Homologie Persistante, Multi-omique, Longévité, Résilience, Vieillissement Accéléré
