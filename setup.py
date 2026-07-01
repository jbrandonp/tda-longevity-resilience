"""TDA-Longevity-Resilience: Topological Data Analysis for Longevity Research.

Apply persistent homology and the Mapper algorithm to multi-omics longevity
data to extract structural signatures of extreme resilience and compare them
to generational accelerated aging (Tian 2026 framework).
"""

from setuptools import setup, find_packages

setup(
    name="tda-longevity-resilience",
    version="1.0.0",
    description="TDA for multi-omics longevity resilience signatures",
    author="Brandon Palhano Machado",
    author_email="brandonpalhano@gmail.com",
    url="https://github.com/jbrandonp/tda-longevity-resilience",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scikit-learn>=1.3",
        "umap-learn>=0.5",
        "ripser>=0.6.4",
        "persim>=0.3.1",
        "kmapper>=2.0",
        "networkx>=3.1",
        "plotly>=5.15",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "flake8>=6.1",
            "black>=23.7",
            "isort>=5.12",
            "pre-commit>=3.3",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
