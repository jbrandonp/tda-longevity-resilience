# Installation Guide

## Prerequisites

- Python 3.10+
- Conda (recommended) or pip + venv
- Git

## Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/jbrandonp/tda-longevity-resilience.git
cd tda-longevity-resilience

# Create the conda environment
conda env create -f environment.yml

# Activate
conda activate tda-longevity

# Verify installation
python -c "import ripser; import persim; import kmapper; print('✅ All packages OK')"

# Launch Jupyter
jupyter lab
```

## Option 2: Pip + venv

```bash
git clone https://github.com/jbrandonp/tda-longevity-resilience.git
cd tda-longevity-resilience

python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -e ".[dev]"

# Install TDA packages
pip install ripser persim kmapper giotto-tda teaspoon

jupyter lab
```

## Option 3: Binder (No Install)

Click the Binder badge in the [README](../README.md) to launch the notebooks in the cloud.

## Option 4: Docker (Optional)

```bash
docker-compose up -d
# Jupyter Lab at http://localhost:8888
```

## Verification

Run the test suite to verify everything works:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_tda_utils.py .....
tests/test_mapper_utils.py .....
tests/test_features.py ........
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ripser` fails to install | `pip install ripser` requires a C++ compiler. On Windows, install Visual Studio Build Tools. |
| `kmapper` visualization fails | Install `plotly` and `ipywidgets`: `pip install plotly ipywidgets` |
| `persim` import error | `pip install persim scikit-learn` |
| Conda environment creation fails | Try `conda env create -f environment.yml --force` |
| Memory error with large persistence | Reduce `RIPSER_MAX_DIM` or increase `RIPSER_THRESH_QUANTILE` in `config.py` |
