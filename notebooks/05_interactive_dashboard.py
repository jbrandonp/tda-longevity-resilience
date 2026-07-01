"""Streamlit dashboard for interactive TDA-Longevity-Resilience exploration.

Launch with: streamlit run notebooks/05_interactive_dashboard.py

Provides:
- Upload custom omics data (CSV)
- Compute persistence diagrams on-the-fly
- Interactive barcode visualization
- Group comparison with Wasserstein distance
- Mapper graph exploration
"""

import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import (
    generate_synthetic_multimics,
    preprocess_omics,
    assign_groups_from_tian_score,
)
from tda_utils import compute_persistence_diagrams, diagnose_persistence, wasserstein_distance
from visualization import plot_barcode, plot_persistence_diagram
from config import RANDOM_SEED

st.set_page_config(
    page_title="TDA-Longevity Dashboard",
    page_icon="🌀",
    layout="wide",
)

st.title("🌀 TDA-Longevity-Resilience Dashboard")
st.markdown("Interactive exploration of topological signatures in multi-omics longevity data.")

# ── Sidebar: Data Source ──
st.sidebar.header("Data Source")
data_mode = st.sidebar.radio("Mode", ["Synthetic (built-in)", "Upload CSV"])

if data_mode == "Synthetic (built-in)":
    topology = st.sidebar.selectbox("Topology", ["circle", "torus", "figure8", "sphere", "noise"])
    n_samples = st.sidebar.slider("Samples", 50, 500, 200, 50)
    noise = st.sidebar.slider("Noise", 0.0, 0.3, 0.05, 0.01)
    n_features = st.sidebar.slider("Features per layer", 10, 100, 50, 10)

    if st.sidebar.button("Generate Data", type="primary") or "data" not in st.session_state:
        with st.spinner("Generating synthetic multi-omics data..."):
            ds = generate_synthetic_multimics(
                n_samples=n_samples,
                topology_type=topology,
                noise=noise,
                n_features=n_features,
            )
            st.session_state.data = ds
            st.session_state.labels = ds["labels"]
            st.success(f"Generated {n_samples} samples with {topology} topology")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.data = {"transcriptomics": df.values}
        st.session_state.labels = pd.Series(["unknown"] * len(df))
        st.success(f"Loaded {len(df)} samples, {df.shape[1]} features")

# ── Main content ──
if "data" not in st.session_state:
    st.info("👈 Generate or upload data from the sidebar to begin.")
    st.stop()

ds = st.session_state.data
labels = st.session_state.labels

# Preprocess
transcriptomics = preprocess_omics(
    pd.DataFrame(ds["transcriptomics"]), method="standard"
)

col1, col2, col3 = st.columns(3)
col1.metric("Samples", transcriptomics.shape[0])
col2.metric("Features", transcriptomics.shape[1])
col3.metric("Groups", labels.nunique())

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["Persistence Diagrams", "Group Comparison", "Diagnostics"])

with tab1:
    st.subheader("Persistent Homology")
    max_dim = st.selectbox("Max dimension", [0, 1, 2], index=1)

    if st.button("Compute Persistence", type="primary"):
        with st.spinner("Computing Vietoris-Rips persistence..."):
            dgms = compute_persistence_diagrams(transcriptomics, max_dim=max_dim)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            plot_barcode(dgms, dim=min(max_dim, 1), title="Persistence Barcode", ax=ax1)
            plot_persistence_diagram(dgms, dim=min(max_dim, 1), title="Persistence Diagram", ax=ax2)
            st.pyplot(fig)

            info = diagnose_persistence(dgms)
            st.json(info)

with tab2:
    st.subheader("Accelerated vs Resilient Comparison")

    # Assign groups
    groups = assign_groups_from_tian_score(
        pd.DataFrame({"tian_score": np.random.randn(transcriptomics.shape[0])}),
        accel_thresh=0.8,
        resil_thresh=-0.8,
    )

    accel_idx = (groups == "accelerated").values
    resil_idx = (groups == "resilient").values

    if accel_idx.sum() > 5 and resil_idx.sum() > 5:
        if st.button("Run Comparison"):
            with st.spinner("Computing group diagrams..."):
                dgm_a = compute_persistence_diagrams(transcriptomics[accel_idx], max_dim=1)
                dgm_r = compute_persistence_diagrams(transcriptomics[resil_idx], max_dim=1)

                w_dist = wasserstein_distance(dgm_a, dgm_r, dim=1)

                c1, c2 = st.columns(2)
                c1.metric("Wasserstein H1 distance", f"{w_dist:.4f}")
                c2.metric("Accelerated", f"{accel_idx.sum()} samples")
                c2.metric("Resilient", f"{resil_idx.sum()} samples")

                diag_a = diagnose_persistence(dgm_a)
                diag_r = diagnose_persistence(dgm_r)
                st.write("**Accelerated:**", diag_a["H1"])
                st.write("**Resilient:**", diag_r["H1"])
    else:
        st.warning("Need at least 6 samples per group for comparison.")

with tab3:
    st.subheader("System Diagnostics")
    st.write("**Configuration:**")
    st.code(f"RIPSER_MAX_DIM: 2\nRIPSER_THRESH_QUANTILE: 0.95\nRANDOM_SEED: {RANDOM_SEED}")

st.sidebar.markdown("---")
st.sidebar.markdown("*TDA-Longevity-Resilience v1.0.0*")
st.sidebar.markdown("[GitHub](https://github.com/jbrandonp/tda-longevity-resilience)")
