"""Tests for aging_scores.py — PhenoAge calculator and group assignment."""

import numpy as np
import pandas as pd
import pytest
from src.aging_scores import (
    phenoage_simplified,
    assign_acceleration_group,
    dunedin_pace_proxy,
)


class TestPhenoAgeSimplified:
    def test_returns_float_array(self):
        n = 10
        age = np.full(n, 50.0)
        result = phenoage_simplified(
            age,
            albumin=np.full(n, 4.0),
            creatinine=np.full(n, 0.9),
            glucose=np.full(n, 95.0),
            c_reactive_protein=np.full(n, 0.5),
            lymphocyte_percent=np.full(n, 30.0),
            mean_cell_volume=np.full(n, 90.0),
            red_blood_cell_distribution=np.full(n, 13.5),
            alkaline_phosphatase=np.full(n, 70.0),
            white_blood_cell_count=np.full(n, 6000.0),
        )
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))

    def test_older_age_produces_higher_phenoage(self):
        age = np.array([30.0, 60.0])
        defaults = {k: np.full(2, v) for k, v in {
            "albumin": 4.0, "creatinine": 0.9, "glucose": 95.0,
            "c_reactive_protein": 0.5, "lymphocyte_percent": 30.0,
            "mean_cell_volume": 90.0, "red_blood_cell_distribution": 13.5,
            "alkaline_phosphatase": 70.0, "white_blood_cell_count": 6000.0,
        }.items()}
        result = phenoage_simplified(age, **defaults)
        assert result[1] > result[0]

    def test_all_default_healthy_values(self):
        n = 5
        age = np.linspace(30, 70, n)
        result = phenoage_simplified(
            age,
            albumin=np.full(n, 4.0),
            creatinine=np.full(n, 0.9),
            glucose=np.full(n, 95.0),
            c_reactive_protein=np.full(n, 0.5),
            lymphocyte_percent=np.full(n, 30.0),
            mean_cell_volume=np.full(n, 90.0),
            red_blood_cell_distribution=np.full(n, 13.5),
            alkaline_phosphatase=np.full(n, 70.0),
            white_blood_cell_count=np.full(n, 6000.0),
        )
        assert np.all(result > 0)
        assert np.all(result < 120)


class TestAssignAccelerationGroup:
    def test_returns_dataframe_with_aging_group(self):
        df = pd.DataFrame({"age": [30, 50, 70]})
        result = assign_acceleration_group(df)
        assert "aging_group" in result.columns
        assert "phenoage" in result.columns
        assert "age_acceleration" in result.columns

    def test_groups_are_valid(self):
        df = pd.DataFrame({"age": np.linspace(20, 80, 100)})
        result = assign_acceleration_group(df)
        valid = {"accelerated", "resilient", "normal"}
        assert set(result["aging_group"].unique()).issubset(valid)


class TestDunedinPACE:
    def test_returns_array_in_range(self):
        meth = np.random.uniform(1, 99, (20, 50))  # beta-values 0-100
        age = np.full(20, 45.0)
        pace = dunedin_pace_proxy(meth, age)
        assert pace.shape == (20,)
        assert np.all(np.isfinite(pace))
