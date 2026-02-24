"""Table generation utilities for benchmark reports."""

from src.benchmark.tables.cohort_table import generate_cohort_table
from src.benchmark.tables.initial_performance import (
    generate_initial_performance_tables,
    generate_from_command_line,
    _format_ci,
)

__all__ = [
    "generate_cohort_table",
    "generate_initial_performance_tables",
    "generate_from_command_line",
    "_format_ci",
]
