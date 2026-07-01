FROM python:3.10-slim

LABEL org.opencontainers.image.title="TDA-Longevity-Resilience"
LABEL org.opencontainers.image.description="Topological Data Analysis for Multi-Omics Longevity Research"
LABEL org.opencontainers.image.url="https://github.com/jbrandonp/tda-longevity-resilience"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .
RUN pip install --no-cache-dir conda-merge 2>/dev/null || true

# Install all TDA packages via pip (lighter than full conda)
RUN pip install --no-cache-dir \
    numpy scipy pandas matplotlib seaborn plotly \
    scikit-learn umap-learn jupyterlab \
    ripser persim kmapper giotto-tda networkx \
    pyyaml pytest pytest-cov teaspoon

COPY . .

RUN pip install -e ".[dev]"

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
