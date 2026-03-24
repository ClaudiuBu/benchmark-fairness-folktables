"""Tests for the cohort_table RACE_LABELS fix and initial_performance task filter fix."""

import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Bug 1: cohort_table RACE_LABELS – ACS RAC1P code → label mapping
# ---------------------------------------------------------------------------

def test_race_labels_import():
    """RACE_LABELS should be importable from cohort_table."""
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert isinstance(RACE_LABELS, dict)


def test_race_labels_code1_white():
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[1] == "White"


def test_race_labels_code2_black():
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[2] == "Black"


def test_race_labels_code3_american_indian():
    """ACS code 3 (American Indian alone) → 'American Indian/Alaska Native'."""
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[3] == "American Indian/Alaska Native"


def test_race_labels_code4_alaska_native():
    """ACS code 4 (Alaska Native alone) must NOT map to 'Asian'."""
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[4] != "Asian", (
        "ACS code 4 is Alaska Native alone, not Asian (code 6)"
    )
    assert RACE_LABELS[4] == "American Indian/Alaska Native"


def test_race_labels_code5_tribes():
    """ACS code 5 (Am Indian/Alaska Native tribes) must NOT map to 'Native Hawaiian/Pacific Islander'."""
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[5] != "Native Hawaiian/Pacific Islander", (
        "ACS code 5 is Am Indian/Alaska Native tribes specified, not Native Hawaiian"
    )
    assert RACE_LABELS[5] == "American Indian/Alaska Native"


def test_race_labels_code6_asian():
    """ACS code 6 (Asian alone) must NOT map to 'Other'."""
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[6] != "Other", (
        "ACS code 6 is Asian alone, not Other (code 8)"
    )
    assert RACE_LABELS[6] == "Asian"


def test_race_labels_code7_native_hawaiian():
    """ACS code 7 (Native Hawaiian/OPI) must NOT map to 'Two or more races'."""
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[7] != "Two or more races", (
        "ACS code 7 is Native Hawaiian and OPI alone, not Two or More Races (code 9)"
    )
    assert RACE_LABELS[7] == "Native Hawaiian/Pacific Islander"


def test_race_labels_code8_other():
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[8] == "Other"


def test_race_labels_code9_two_or_more():
    from src.benchmark.tables.cohort_table import RACE_LABELS
    assert RACE_LABELS[9] == "Two or more races"


def test_race_labels_all_display_order_covered():
    """The display_order categories in _summarize must all appear in RACE_LABELS values."""
    from src.benchmark.tables.cohort_table import RACE_LABELS
    display_order = [
        "White",
        "Black",
        "American Indian/Alaska Native",
        "Asian",
        "Native Hawaiian/Pacific Islander",
        "Other",
        "Two or more races",
    ]
    label_values = set(RACE_LABELS.values())
    for category in display_order:
        assert category in label_values, f"Display category '{category}' not covered by RACE_LABELS"


# ---------------------------------------------------------------------------
# Bug 2: initial_performance.py – task filtering
# ---------------------------------------------------------------------------

def _create_two_task_csv(path: Path):
    path.write_text(
        "method,task,sensitive_attribute,auc_mean,auc_ci_lower,auc_ci_upper\n"
        "baseline,income,SEX,0.85,0.84,0.86\n"
        "baseline,employment,SEX,0.70,0.69,0.71\n"
        "reweighing,income,SEX,0.84,0.83,0.85\n"
        "reweighing,employment,SEX,0.71,0.70,0.72\n"
    )


def test_initial_performance_income_values(tmp_path):
    """Income table should show income-specific metric values."""
    from src.benchmark.tables.initial_performance import generate_initial_performance_tables

    csv_path = tmp_path / "summary.csv"
    _create_two_task_csv(csv_path)

    income_path, _ = generate_initial_performance_tables(str(csv_path), tmp_path)
    content = income_path.read_text()

    assert "0.850" in content, "Income baseline AUC (0.850) not found in income table"
    assert "0.700" not in content, "Employment AUC (0.700) should not appear in income table"


def test_initial_performance_employment_values(tmp_path):
    """Employment table should show employment-specific metric values."""
    from src.benchmark.tables.initial_performance import generate_initial_performance_tables

    csv_path = tmp_path / "summary.csv"
    _create_two_task_csv(csv_path)

    _, employment_path = generate_initial_performance_tables(str(csv_path), tmp_path)
    content = employment_path.read_text()

    assert "0.700" in content, "Employment baseline AUC (0.700) not found in employment table"
    assert "0.850" not in content, "Income AUC (0.850) should not appear in employment table"


def test_initial_performance_no_task_column(tmp_path):
    """When no task column is present, both income and employment tables should contain identical data from all rows."""
    from src.benchmark.tables.initial_performance import generate_initial_performance_tables

    csv_path = tmp_path / "summary_notask.csv"
    csv_path.write_text(
        "method,sensitive_attribute,auc_mean,auc_ci_lower,auc_ci_upper\n"
        "baseline,SEX,0.85,0.84,0.86\n"
        "reweighing,SEX,0.84,0.83,0.85\n"
    )

    income_path, employment_path = generate_initial_performance_tables(str(csv_path), tmp_path)
    # Both tables get the same data when no task column exists — that's the expected fallback
    assert "0.850" in income_path.read_text()
    assert "0.850" in employment_path.read_text()
