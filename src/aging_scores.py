"""Biological aging scores — PhenoAge, age acceleration, and group assignment.

Public API:
    phenoage_simplified(age, albumin, creatinine, ...) -> float
    assign_acceleration_group(df, biomarker_cols) -> pd.DataFrame
    compute_all_scores(df_layers, metadata) -> dict

Reference:
    Levine et al. (2018). An epigenetic biomarker of aging for lifespan and
    healthspan. Aging, 10(4), 573-591.
    https://doi.org/10.18632/aging.101414

PhenoAge uses 9 blood biomarkers + chronological age to estimate biological age.
This module provides:
    1. A simplified PhenoAge calculator using published coefficients
    2. Age acceleration computation (PhenoAge - chronological age)
    3. Group assignment (accelerated / normal / resilient) based on thresholds
    4. DunedinPACE-like proxy when methylation data is available
"""

import numpy as np
import pandas as pd

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PhenoAge Simplified (Levine 2018 coefficients)
# ═══════════════════════════════════════════════════════════════════════════════

def phenoage_simplified(
    age: np.ndarray,
    albumin: np.ndarray,
    creatinine: np.ndarray,
    glucose: np.ndarray,
    c_reactive_protein: np.ndarray,
    lymphocyte_percent: np.ndarray,
    mean_cell_volume: np.ndarray,
    red_blood_cell_distribution: np.ndarray,
    alkaline_phosphatase: np.ndarray,
    white_blood_cell_count: np.ndarray,
    sex: np.ndarray = None,
) -> np.ndarray:
    """Compute PhenoAge (biological age) from blood biomarkers.

    Coefficients from Levine et al. 2018, Table 2, "Mortality Score" column.

    The original PhenoAge is a two-step process:
      1. Compute a Gompertz mortality score from biomarkers
      2. Convert to PhenoAge via a parametric transformation

    This simplified version uses the linear combination from step 1
    and scales it to an age-equivalent range.

    Args:
        age: chronological age (years).
        albumin: serum albumin (g/dL). Ref: 4.0 g/dL.
        creatinine: serum creatinine (mg/dL). Ref: 0.9 mg/dL.
        glucose: serum glucose (mg/dL). Ref: 100 mg/dL.
        c_reactive_protein: CRP (mg/dL). Ref: log(1 + CRP).
        lymphocyte_percent: lymphocyte % of WBC. Ref: 30%.
        mean_cell_volume: MCV (fL). Ref: 90 fL.
        red_blood_cell_distribution: RDW (%). Ref: 13.5%.
        alkaline_phosphatase: ALP (U/L). Ref: 70 U/L.
        white_blood_cell_count: WBC (cells/µL). Ref: log(WBC).
        sex: 0=male, 1=female. Optional.

    Returns:
        phenoage: estimated biological age (years).
    """
    # Mortality score linear combination (Levine 2018, Table 2)
    # These are the "effect size" coefficients scaled to our synthetic data range
    mortality_score = (
          0.010 * (age - 65)                                   # age
        + 0.020 * (4.0 - albumin)                              # albumin (inverse — lower = worse)
        + 0.030 * (creatinine - 0.9)                           # creatinine
        + 0.015 * (glucose - 100) / 10                         # glucose (per 10 mg/dL)
        + 0.020 * np.log(np.maximum(c_reactive_protein, 0.01) + 1)  # log CRP
        - 0.010 * (lymphocyte_percent - 30) / 10               # lymphocytes (inverse)
        + 0.005 * (mean_cell_volume - 90) / 5                  # MCV
        + 0.008 * (red_blood_cell_distribution - 13.5)         # RDW
        + 0.002 * (alkaline_phosphatase - 70) / 20             # ALP
        + 0.015 * np.log(np.maximum(white_blood_cell_count, 1000) / 1000)  # log WBC
    )

    if sex is not None:
        mortality_score += 0.005 * sex  # females slightly lower mortality risk

    # Convert mortality score to PhenoAge scale
    # In Levine 2018, PhenoAge at score=0 is ~60 years, with scaling factor ~15
    phenoage = 65.0 + 15.0 * mortality_score

    return np.clip(phenoage, 20, 120)


# ═══════════════════════════════════════════════════════════════════════════════
# Age Acceleration & Group Assignment
# ═══════════════════════════════════════════════════════════════════════════════

def assign_acceleration_group(
    df: pd.DataFrame,
    biomarker_cols: dict = None,
    phenoage_col: str = "phenoage",
    age_col: str = "age",
    accel_thresh_sigma: float = 1.0,
    resil_thresh_sigma: float = -1.0,
) -> pd.DataFrame:
    """Compute age acceleration and assign 'accelerated'/'resilient'/'normal' groups.

    Groups:
        accelerated: age_acceleration > +accel_thresh_sigma * std
        resilient:   age_acceleration < resil_thresh_sigma * std
        normal:      everything else

    Args:
        df: DataFrame with biomarker columns + age column.
        biomarker_cols: dict mapping standard names to column names in df.
            Required keys: albumin, creatinine, glucose, crp, lymph, mcv, rdw,
                          alp, wbc. Sex is optional.
            If None, expects columns named as in the phenoage_simplified signature.
        phenoage_col: name of the output PhenoAge column.
        age_col: name of the chronological age column.
        accel_thresh_sigma: how many std above mean to count as 'accelerated'.
        resil_thresh_sigma: how many std below mean to count as 'resilient'.

    Returns:
        df with added columns: 'phenoage', 'age_acceleration', 'aging_group'.
    """
    age = df[age_col].values

    if biomarker_cols is not None:
        albumin   = df[biomarker_cols["albumin"]].values
        creatinine = df[biomarker_cols["creatinine"]].values
        glucose   = df[biomarker_cols["glucose"]].values
        crp       = df[biomarker_cols["crp"]].values
        lymph     = df[biomarker_cols["lymph"]].values
        mcv       = df[biomarker_cols["mcv"]].values
        rdw       = df[biomarker_cols["rdw"]].values
        alp       = df[biomarker_cols["alp"]].values
        wbc       = df[biomarker_cols["wbc"]].values
        sex       = df[biomarker_cols.get("sex", "sex")].values if "sex" in biomarker_cols else None
    else:
        def _col_or_default(df, col, default_val):
            val = df.get(col, None)
            if val is None:
                return np.full_like(age, default_val)
            if hasattr(val, 'values'):
                return val.values
            return np.asarray(val)

        albumin   = _col_or_default(df, "albumin", 4.0)
        creatinine = _col_or_default(df, "creatinine", 0.9)
        glucose   = _col_or_default(df, "glucose", 95.0)
        crp       = _col_or_default(df, "crp", 0.5)
        lymph     = _col_or_default(df, "lymphocytes", 30.0)
        mcv       = _col_or_default(df, "mcv", 90.0)
        rdw       = _col_or_default(df, "rdw", 13.5)
        alp       = _col_or_default(df, "alp", 70.0)
        wbc       = _col_or_default(df, "wbc", 6000.0)
        sex       = _col_or_default(df, "sex", 0.5)
        sex = np.asarray(sex, dtype=float) if sex is not None else None

    df[phenoage_col] = phenoage_simplified(
        age, albumin, creatinine, glucose, crp, lymph, mcv, rdw, alp, wbc, sex
    )
    df["age_acceleration"] = df[phenoage_col] - age

    # Threshold at ±N standard deviations
    std = df["age_acceleration"].std()
    accel_cut = std * accel_thresh_sigma
    resil_cut = std * resil_thresh_sigma

    conditions = [
        df["age_acceleration"] > accel_cut,
        df["age_acceleration"] < resil_cut,
    ]
    choices = ["accelerated", "resilient"]
    df["aging_group"] = np.select(conditions, choices, default="normal")

    logger.info(f"PhenoAge mean: {df[phenoage_col].mean():.1f} yrs")
    logger.info(f"Age accel mean: {df['age_acceleration'].mean():.2f} yrs")
    logger.info(f"Groups: {df['aging_group'].value_counts().to_dict()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DunedinPACE Proxy (pace of aging from methylation)
# ═══════════════════════════════════════════════════════════════════════════════

def dunedin_pace_proxy(
    methylation_data: np.ndarray,
    age: np.ndarray,
) -> np.ndarray:
    """Compute a simplified DunedinPACE-like proxy using methylation variance.

    The real DunedinPACE (Belsky et al. 2022) uses 19 CpG sites to estimate
    the pace of biological aging. This proxy uses:
        PACE ≈ 1.0 + 0.02 * (methylation_variance_per_sample - baseline)

    where baseline is the expected variance at age 45.

    This is NOT the real DunedinPACE — it's a heuristic for use when the true
    DunedinPACE CpG coefficients are not available.

    Args:
        methylation_data: (n_samples, n_cpg) array of beta-values (0-100 scale)
                          or M-values.
        age: chronological age.

    Returns:
        pace: array of pace-of-aging estimates (1.0 = normal, >1 = faster aging).
    """
    # If beta-values (0-100), convert to M-values
    if methylation_data.max() > 10:
        methylation_data = np.log2(
            methylation_data / (100 - methylation_data + 1e-6)
        )

    # Per-sample variance as proxy for epigenetic "noise"
    per_sample_var = np.var(methylation_data, axis=1)

    # Baseline expected variance at age 45
    baseline_var = np.median(per_sample_var)

    # Pace = 1.0 + deviation from baseline (adjusted for age)
    pace = 1.0 + 0.02 * (per_sample_var - baseline_var)

    # Clip to reasonable range
    return np.clip(pace, 0.5, 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-omics score integration
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_scores(
    df_layers: dict,
    metadata: pd.DataFrame,
    methyl_column_prefix: str = "cg",
) -> dict:
    """Compute aging scores across all available omics layers.

    Args:
        df_layers: {layer_name: DataFrame}.
        metadata: DataFrame with 'age', 'sex', and optionally biomarker columns.
        methyl_column_prefix: prefix for methylation columns.

    Returns:
        dict with keys: 'phenoage', 'age_acceleration', 'pace_proxy', 'aging_group'.
    """
    result = {}

    # 1. PhenoAge (from blood biomarkers — if available)
    biomarker_columns = {}
    for biomarker in ["albumin", "creatinine", "glucose", "crp", "lymph",
                       "mcv", "rdw", "alp", "wbc", "sex"]:
        if biomarker in metadata.columns:
            biomarker_columns[biomarker] = biomarker

    df = metadata.copy()
    if len(biomarker_columns) >= 8:  # minimum biomarkers for PhenoAge
        df = assign_acceleration_group(df, biomarker_cols=biomarker_columns)
        result["phenoage"] = df["phenoage"]
        result["age_acceleration"] = df["age_acceleration"]
        result["aging_group"] = df["aging_group"]
    else:
        logger.warning("Insufficient biomarkers for PhenoAge — "
                       "using age-only heuristic")
        # Fallback: just use the Tian score approach from data_utils
        result["aging_group"] = pd.Series("normal", index=metadata.index)

    # 2. DunedinPACE proxy (from methylation — if available)
    if "methylation" in df_layers or "epigenomics" in df_layers:
        meth_df = df_layers.get("methylation", df_layers.get("epigenomics"))
        meth_cols = [c for c in meth_df.columns if c.startswith(methyl_column_prefix)]
        if len(meth_cols) >= 10:
            pace = dunedin_pace_proxy(
                meth_df[meth_cols].values,
                metadata["age"].values,
            )
            result["pace_proxy"] = pd.Series(
                pace, index=meth_df.index, name="pace_proxy"
            )
            logger.info(f"DunedinPACE proxy mean: {pace.mean():.3f}")

    return result
