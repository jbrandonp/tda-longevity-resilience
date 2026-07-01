.PHONY: help install test lint format clean docs notebook-check security-check all

help: ## Afficher cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installer le package en mode dev
	pip install -e ".[dev,bio,tda]"

install-conda: ## Créer l'environnement Conda
	conda env create -f environment.yml && echo "Activate: conda activate tda-longevity"

test: ## Lancer les tests unitaires
	pytest tests/ -v --tb=short

test-cov: ## Lancer les tests avec couverture
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-nb: ## Vérifier que les notebooks s'exécutent
	pytest --nbval-lax notebooks/00_synthetic_validation.ipynb || echo "⚠️  nbval requires kernel"

lint: ## Linting complet
	flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503
	black --check src/ tests/ --line-length=120
	isort --check-only --profile black src/ tests/

format: ## Formater le code
	black src/ tests/ --line-length=120
	isort src/ tests/ --profile black

security: ## Scan de sécurité
	bandit -c .bandit -r src/

docs: ## Générer la documentation API
	pdoc --html --output-dir docs/api src/ --force 2>/dev/null || echo "Install pdoc3: pip install pdoc3"

clean: ## Nettoyer les fichiers temporaires
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type f -name '*.pyo' -delete 2>/dev/null || true
	rm -rf htmlcov/ docs/api/ 2>/dev/null || true

notebooks: ## Lancer tous les notebooks en ordre
	jupyter nbconvert --to notebook --execute --inplace notebooks/00_synthetic_validation.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_persistent_homology.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_mapper_analysis.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_comparison_accelerated_vs_resilient.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/04_feature_extraction_ml.ipynb

synth: ## Générer des données synthétiques
	python scripts/generate_synthetic_data.py --topology circle --n-samples 200

dashboard: ## Lancer le dashboard Streamlit
	streamlit run notebooks/05_interactive_dashboard.py

all: clean format lint test-cov ## Pipeline complet : clean + format + lint + test
