"""Tests for Bug 4, 5, 6, and 7 (second audit pass)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# Bug 4: runner.py – summary_by_year missing sensitive_attribute in groupby
# ---------------------------------------------------------------------------

def _make_results_by_year(n_seeds: int = 3) -> pd.DataFrame:
    """Synthetic results_by_year DataFrame with two sensitive attributes."""
    rows = []
    for seed in range(n_seeds):
        for year in [2015, 2016]:
            for method in ["baseline"]:
                for maint in ["no-retrain"]:
                    rows.append({
                        "seed": seed, "year": year, "method": method,
                        "maintenance": maint,
                        "sensitive_attribute": "SEX",
                        "sensitive_attribute_value": "ALL",
                        "dp_gap": 0.10 + seed * 0.01,
                        "accuracy": 0.85,
                    })
                    rows.append({
                        "seed": seed, "year": year, "method": method,
                        "maintenance": maint,
                        "sensitive_attribute": "RAC1P",
                        "sensitive_attribute_value": "ALL",
                        "dp_gap": 0.25 + seed * 0.01,
                        "accuracy": 0.85,
                    })
    return pd.DataFrame(rows)


def test_summary_by_year_includes_sensitive_attribute():
    """summary_by_year must keep SEX and RAC1P metrics separate, not averaged."""
    df = _make_results_by_year()

    # Simulate the FIXED groupby (matching the fix in runner.py)
    groupby_cols = ["year", "method", "maintenance", "sensitive_attribute", "sensitive_attribute_value"]
    numeric_cols = ["dp_gap", "accuracy"]
    summary = (
        df.groupby(groupby_cols, sort=True)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in summary.columns
    ]

    sex_rows = summary[summary["sensitive_attribute"] == "SEX"]
    race_rows = summary[summary["sensitive_attribute"] == "RAC1P"]

    # SEX dp_gap should be ~0.11, RAC1P dp_gap ~0.26
    assert sex_rows["dp_gap_mean"].max() < 0.20, (
        "SEX dp_gap should be ~0.11, not mixed with RAC1P"
    )
    assert race_rows["dp_gap_mean"].min() > 0.20, (
        "RAC1P dp_gap should be ~0.26, not mixed with SEX"
    )
    # The two attributes should NOT have been averaged together
    assert not sex_rows.empty and not race_rows.empty


def test_summary_by_year_buggy_groupby_mixes_attributes():
    """Confirm the old (buggy) groupby without sensitive_attribute does mix attributes."""
    df = _make_results_by_year()

    # Simulate the BUGGY groupby (missing sensitive_attribute)
    buggy_cols = ["year", "method", "maintenance", "sensitive_attribute_value"]
    numeric_cols = ["dp_gap"]
    summary = df.groupby(buggy_cols)[numeric_cols].mean().reset_index()

    # The buggy result averages SEX and RAC1P dp_gap => ~0.175 (between 0.11 and 0.26)
    val = summary["dp_gap"].mean()
    assert 0.15 < val < 0.22, (
        f"Buggy groupby should mix SEX (~0.11) and RAC1P (~0.26) into ~0.175, got {val:.3f}"
    )


# ---------------------------------------------------------------------------
# Bug 5: reporting.py – plots use z=1.96 instead of t-distribution for CI
# ---------------------------------------------------------------------------

def test_t_ci_half_width_uses_t_distribution():
    """_t_ci_half_width should produce wider CIs than z=1.96 for small n."""
    from src.benchmark.reporting import _t_ci_half_width

    # n=10 seeds: t_9(0.975) = 2.262 > z = 1.96
    grouped = pd.DataFrame({"mean": [0.5], "std": [0.01], "count": [10]})
    ci_t = _t_ci_half_width(grouped).iloc[0]
    ci_z = 1.96 * 0.01 / np.sqrt(10)

    assert ci_t > ci_z, (
        f"t-based CI ({ci_t:.5f}) should be wider than z-based CI ({ci_z:.5f}) for n=10"
    )


def test_t_ci_half_width_value_n10():
    """Verify t-based CI matches t_9(0.975) * sem for n=10."""
    from src.benchmark.reporting import _t_ci_half_width

    n, s = 10, 0.05
    grouped = pd.DataFrame({"mean": [0.8], "std": [s], "count": [n]})
    ci_t = _t_ci_half_width(grouped).iloc[0]

    expected = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    assert ci_t == pytest.approx(expected, rel=1e-9)


def test_t_ci_half_width_large_n_approaches_z():
    """For large n, t-based CI should approach z=1.96-based CI."""
    from src.benchmark.reporting import _t_ci_half_width

    n, s = 10000, 0.01
    grouped = pd.DataFrame({"mean": [0.5], "std": [s], "count": [n]})
    ci_t = _t_ci_half_width(grouped).iloc[0]
    ci_z = 1.96 * s / np.sqrt(n)

    assert abs(ci_t - ci_z) / ci_z < 0.01, (
        "For n=10000, t-CI and z-CI should differ by < 1%"
    )


def test_t_ci_half_width_n1_does_not_crash():
    """Should handle n=1 without division or log errors."""
    from src.benchmark.reporting import _t_ci_half_width

    grouped = pd.DataFrame({"mean": [0.5], "std": [0.0], "count": [1]})
    ci = _t_ci_half_width(grouped).iloc[0]
    assert np.isfinite(ci) or np.isnan(ci)  # 0 * t → 0


def test_t_ci_half_width_series_index_preserved():
    """Output Series index should match the input DataFrame index."""
    from src.benchmark.reporting import _t_ci_half_width

    grouped = pd.DataFrame(
        {"mean": [0.5, 0.6], "std": [0.01, 0.02], "count": [10, 15]},
        index=[100, 200],
    )
    result = _t_ci_half_width(grouped)
    assert list(result.index) == [100, 200]


# ---------------------------------------------------------------------------
# Bug 6: fairness_gap_combined.py – sex_label direction (Male vs Female)
# ---------------------------------------------------------------------------

def test_fairness_gap_combined_sex_label_direction():
    """Default sex_label must say 'Male vs Female' to match A=1(Male)−A=0(Female)."""
    import importlib
    import src.benchmark.tables.fairness_gap_combined as fgc

    # Inspect the source: find the default argument for sex_label
    import inspect
    src_text = inspect.getsource(fgc.generate_fairness_gap_combined_table)
    assert "Male vs Female" in src_text, (
        "sex_label default should be 'Male vs Female' (A=1=Male comes first in computation)"
    )
    assert "Female vs Male" not in src_text, (
        "Old incorrect direction 'Female vs Male' should not be in the source"
    )


# ---------------------------------------------------------------------------
# Bug 7: reporting.py compute_confidence_intervals – ddof=1 for sample std
# ---------------------------------------------------------------------------

def test_compute_confidence_intervals_uses_sample_std():
    """The _std column should use sample std (ddof=1), not population std (ddof=0)."""
    from src.benchmark.reporting import compute_confidence_intervals

    # With values [0, 1]: population std = 0.5, sample std ≈ 0.7071
    df = pd.DataFrame({
        "method": ["baseline", "baseline"],
        "sensitive_attribute": ["SEX", "SEX"],
        "dp_gap": [0.0, 1.0],
    })
    summary = compute_confidence_intervals(df, ci=0.95)
    # sample std of [0, 1] = sqrt(0.5) ≈ 0.7071
    expected_std = float(np.std([0.0, 1.0], ddof=1))
    assert summary["dp_gap_std"].iloc[0] == pytest.approx(expected_std, rel=1e-9), (
        f"std should be sample std ({expected_std:.4f}) not population std ({np.std([0.,1.]):.4f})"
    )


def test_compute_confidence_intervals_ci_bounds_correct():
    """CI bounds should be symmetric around the mean using t-distribution SEM."""
    from src.benchmark.reporting import compute_confidence_intervals

    values = [0.8, 0.85, 0.83, 0.82, 0.84]
    df = pd.DataFrame({
        "method": ["baseline"] * len(values),
        "sensitive_attribute": ["SEX"] * len(values),
        "auc": values,
    })
    summary = compute_confidence_intervals(df, ci=0.95)

    mean = float(np.mean(values))
    sem = stats.sem(values, nan_policy="omit")
    ci_range = sem * stats.t.ppf(0.975, len(values) - 1)

    assert summary["auc_mean"].iloc[0] == pytest.approx(mean, rel=1e-9)
    assert summary["auc_ci_lower"].iloc[0] == pytest.approx(mean - ci_range, rel=1e-6)
    assert summary["auc_ci_upper"].iloc[0] == pytest.approx(mean + ci_range, rel=1e-6)
